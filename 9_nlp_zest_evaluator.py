# ==========================================
# COMPATIBILITY PATCHES
# ==========================================
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

# ==========================================
# STANDARD IMPORTS
# ==========================================
import os
import json
import torch
import numpy as np
import gc
import random
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
import httpx

# ZEST Pipeline Imports
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from zest_code.ip_adapter import IPAdapterXL
from zest_code.ip_adapter.utils import register_cross_attention_hook
from transformers import pipeline
from openai import OpenAI
import argparse

GT_PATH = "../ground_truth_eval.json" 
OUTPUT_DIR = Path("../zest_nlp_results")
MATERIAL_BANK_DIR = Path("../colored_material_bank")

# ==========================================
# 🧠 NLP PARSING & SPATIAL ANALYSIS
# ==========================================

def get_spatial_info(gt_data, img_id):
    """Calculates the physical size and position (left/center/right) of every building in the image."""
    instances = gt_data.get(img_id, {}).get("instances", {})
    info = {}
    
    for bid, inst in instances.items():
        if "wall" in inst and not inst["wall"].get("skipped", False):
            # Resolve the mask path relative to the script execution folder
            mask_path = "../" + str(inst["wall"]["mask"]).replace('\\', '/')
            if Path(mask_path).exists():
                mask_np = np.array(Image.open(mask_path).convert("L")) > 127
                ys, xs = np.where(mask_np)
                if len(xs) > 0:
                    area = int(np.sum(mask_np))
                    cx = int(np.mean(xs))
                    h, w = mask_np.shape
                    pos_x = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                    
                    info[bid] = {
                        "area_pixels": area,
                        "horizontal_position": pos_x
                    }
    return info

from dotenv import load_dotenv

def parse_nlp_prompt(user_prompt, instances_info):
    """Uses an LLM to map the user's plain English prompt to exact building IDs and materials."""
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing NVIDIA_API_KEY env var for NLP parsing.")

    proxy_url = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or \
                os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    if proxy_url and not proxy_url.startswith('http'):
        proxy_url = f"http://{proxy_url}"
        
    custom_http_client = httpx.Client(proxies=proxy_url, verify=False) if proxy_url else httpx.Client(verify=False)
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, http_client=custom_http_client)

    sys_prompt = f"""You are an intelligent JSON parser for architectural image editing.
The user provides a natural language prompt to edit a specific building structure.

Available buildings in the target image and their spatial properties (use these to figure out what the user means by "left", "biggest", etc.):
{json.dumps(instances_info, indent=2)}

Rules:
- Output ONLY valid JSON. No markdown fences.
- "target_instance" must be the exact key from the available buildings (e.g., "building_00").
- "target_structure" must be "wall", "roof", "window", or "door".
- "new_color" should be a simple color name (e.g., "red", "white", "gray").
- "new_material" should be a simple material name (e.g., "brick", "concrete", "wood").

User Prompt: "{user_prompt}"
"""
    print(f"🧠 Asking AI to interpret prompt: '{user_prompt}'")
    resp = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.1,
        max_tokens=200,
        messages=[{"role": "user", "content": sys_prompt}]
    )
    
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```json"): raw = raw[7:]
    if raw.startswith("```"): raw = raw[3:]
    if raw.endswith("```"): raw = raw[:-3]
    
    parsed = json.loads(raw.strip())
    print(f"✅ AI Parsed Intent: {json.dumps(parsed, indent=2)}")
    return parsed

# ==========================================
# 🎨 MATERIAL BANK RETRIEVAL
# ==========================================
def get_material_from_bank(color, material):
    """Fetches a high-quality reference texture from the generated material bank."""
    if not MATERIAL_BANK_DIR.exists():
        print(f"❌ Material bank missing at {MATERIAL_BANK_DIR}")
        return None

    # Determine folder name based on whether color was provided
    if color and str(color).lower() not in ["none", "unknown", ""]:
        target_folder_name = f"{color.lower()}_{material.lower()}"
    else:
        target_folder_name = f"original_{material.lower()}"
        
    target_folder = MATERIAL_BANK_DIR / target_folder_name
    
    # Fallback to original if the color combo doesn't exist
    if not target_folder.exists():
        print(f"⚠️ Warning: Folder '{target_folder_name}' not found. Falling back to original_{material.lower()}...")
        target_folder = MATERIAL_BANK_DIR / f"original_{material.lower()}"
        
    if not target_folder.exists():
        print(f"❌ Could not find any material textures for '{material}' in the bank!")
        return None

    # Gather images
    images = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.jpeg")) + list(target_folder.glob("*.png"))
    if not images:
        print(f"❌ Folder '{target_folder.name}' is empty!")
        return None

    selected_image = random.choice(images)
    print(f"✅ Loaded high-quality reference: {target_folder.name}/{selected_image.name}")
    return Image.open(selected_image).convert("RGB")

# ==========================================
# 🖌️ ZEST PIPELINE FUNCTIONS
# ==========================================

def run_ab_comparison(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    target_image.save(OUTPUT_DIR / f"{out_prefix}_original.jpg")

    ip_image = material_img.convert('RGB').resize((224, 224))
    target_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).convert("RGB").resize((1024, 1024))
    
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(target_image, invert_target_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"  -> Running ZEST Edit...")
    guided_result = ip_model.generate(
        pil_image=ip_image, 
        num_samples=1, 
        num_inference_steps=30, 
        seed=42, 
        image=init_img,                 
        control_image=depth_image,      
        mask_image=target_mask,         
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        scale=1.0, 
        controlnet_conditioning_scale=0.85, 
        strength=0.9, 
    )[0]
    guided_result.save(OUTPUT_DIR / f"{out_prefix}_nlp_edited.jpg")
    
    del guided_result
    gc.collect()
    torch.cuda.empty_cache()

def run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_img, prompt, out_prefix):
    """Naive baseline: Uses the NLP prompt/reference but applies it globally without a segmentation mask."""
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    
    # NAIVE MASK: Completely white mask (no segmentation, allows editing everywhere)
    naive_mask = Image.new("RGB", (1024, 1024), (255, 255, 255))
    
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    gray_target_image = target_image.convert('L').convert('RGB')
    init_img = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"  -> Running Naive ZEST Edit (No Segmentation)...")
    naive_result = ip_model.generate(
        pil_image=ip_image, 
        num_samples=1, 
        num_inference_steps=30, 
        seed=42, 
        image=init_img,                 
        control_image=depth_image,      
        mask_image=naive_mask,         
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        scale=1.0, 
        controlnet_conditioning_scale=0.85, 
        strength=0.9, 
    )[0]
    naive_result.save(OUTPUT_DIR / f"{out_prefix}_naive_edited.jpg")
    
    del naive_result
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # =========================================================
    # 🎯 DEFINE YOUR TARGET IMAGE AND NATURAL LANGUAGE PROMPT HERE
    # =========================================================
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run ZEST with a natural language prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="The natural language edit prompt.")
    args = parser.parse_args()
    TARGET_IMAGE_ID = "1003317145127470"  # Replace with the image ID you want to edit
    USER_PROMPT = args.prompt
    # =========================================================

    original_dir = os.getcwd()
    try:
        os.chdir("zest_code")
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        device = "cuda"

        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f).get("data", {})

        if TARGET_IMAGE_ID not in gt_data:
            print(f"❌ Image {TARGET_IMAGE_ID} not found in dataset!")
            return

        # 1. Analyze spatial layout of buildings in the chosen image
        spatial_info = get_spatial_info(gt_data, TARGET_IMAGE_ID)
        if not spatial_info:
            print(f"❌ No valid walls found for image {TARGET_IMAGE_ID}")
            return
            
        # 2. Ask LLM to parse the user's intent based on spatial layout
        parsed_intent = parse_nlp_prompt(USER_PROMPT, spatial_info)
        
        target_bid = parsed_intent.get("target_instance")
        target_struct = parsed_intent.get("target_structure", "wall")
        target_color = parsed_intent.get("new_color")
        target_material = parsed_intent.get("new_material")

        if target_bid not in spatial_info:
            print(f"❌ LLM selected an invalid instance: {target_bid}")
            return

        # 3. Retrieve Mask and RGB paths
        inst_data = gt_data[TARGET_IMAGE_ID]["instances"][target_bid][target_struct]
        target_mask = "../" + inst_data["mask"].replace('\\', '/')
        rgb_path = "../" + gt_data[TARGET_IMAGE_ID]["rgb"].replace('\\', '/')

        print(f"\n🔍 Retrieving reference image of '{target_color} {target_material}' from Material Bank...")
        
        # Pull perfectly clean image from our generated folders
        material_pil_img = get_material_from_bank(target_color, target_material)
        if material_pil_img is None:
            return # Script stops if material is totally missing

        print("\n⏳ Loading ZEST Models...")
        
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
        
        controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "models/image_encoder"
        ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"

        controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            base_model_path, controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
        ).to(device)
        
        pipe.unet = register_cross_attention_hook(pipe.unet)
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
        print("🚀 All Pipelines Loaded.")

        # 4. Run the Edits
        zest_prompt = f"a highly detailed building facade made of {target_color} {target_material}, featuring architectural outlines"
        out_prefix = f"{TARGET_IMAGE_ID}_{target_bid}"
        
        run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_pil_img, zest_prompt, out_prefix)
        run_ab_comparison(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, zest_prompt, out_prefix)
                
        print(f"\n✅ Done! Check the {OUTPUT_DIR.name} folder for:")
        print(f"   - {out_prefix}_naive_edited.jpg (No Mask Baseline)")
        print(f"   - {out_prefix}_nlp_edited.jpg (With Segmented Mask)")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()