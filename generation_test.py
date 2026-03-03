#!/usr/bin/env python3
import argparse
import os
import json
import torch
import numpy as np
import gc
import random
import re
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from pathlib import Path
import httpx

# ==========================================
# COMPATIBILITY PATCHES
# ==========================================
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

# Pipeline Imports
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv

GT_PATH = "ground_truth_eval.json" 
OUTPUT_DIR = Path("nlp_results")
DEBUG_DIR = OUTPUT_DIR / "debug_vis"
MATERIAL_BANK_DIR = Path("colored_material_bank")

# ==========================================
# 🏛️ ARCHITECTURAL PLAUSIBILITY RULES
# ==========================================
PLAUSIBLE_COMBINATIONS = {
    "brick": ["red", "brown", "white", "gray", "yellow", "black", "original"],
    "concrete": ["gray", "white", "beige", "black", "original"],
    "wood": ["brown", "beige", "white", "gray", "black", "red", "original"],
    "plaster": ["white", "beige", "yellow", "gray", "red", "blue", "green", "orange", "original"], 
    "metal": ["gray", "black", "white", "red", "blue", "green", "original"],
    "shingles": ["black", "gray", "brown", "red", "green", "original"],
    "asphalt": ["black", "gray", "original"],
    "stone": ["gray", "brown", "beige", "black", "white", "original"],
    "tile": ["white", "gray", "black", "blue", "green", "red", "brown", "yellow", "original"]
}

# ==========================================
# 🧠 NLP PARSING & SPATIAL ANALYSIS
# ==========================================

def get_spatial_info(gt_data, img_id):
    instances = gt_data.get(img_id, {}).get("instances", {})
    info = {}
    for bid, inst in instances.items():
        if "wall" in inst and not inst["wall"].get("skipped", False):
            mask_path = str(inst["wall"]["mask"]).replace('\\', '/')
            if Path(mask_path).exists():
                mask_np = np.array(Image.open(mask_path).convert("L")) > 127
                ys, xs = np.where(mask_np)
                if len(xs) > 0:
                    area = int(np.sum(mask_np))
                    cx = int(np.mean(xs))
                    h, w = mask_np.shape
                    pos_x = "left" if cx < w/3 else "right" if cx > 2*w/3 else "center"
                    info[bid] = {"area_pixels": area, "horizontal_position": pos_x}
    return info

def parse_nlp_prompt(user_prompt, instances_info):
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key: raise SystemExit("❌ Missing NVIDIA_API_KEY env var.")

    proxy_url = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    if proxy_url and not proxy_url.startswith('http'): proxy_url = f"http://{proxy_url}"
        
    custom_http_client = httpx.Client(proxies=proxy_url, verify=False) if proxy_url else httpx.Client(verify=False)
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, http_client=custom_http_client)

    sys_prompt = f"""You are an intelligent JSON parser for architectural image editing.
Available buildings in the target image and spatial properties:
{json.dumps(instances_info, indent=2)}
Rules:
- Output ONLY valid JSON.
- "target_instance" must be the exact key from the available buildings.
- "target_structure" must be "wall", "roof", "window", or "door".
- "new_color" should be a simple color name (e.g., "red", "white", "gray").
- "new_material" should be a simple material name.
User Prompt: "{user_prompt}"
"""
    print(f"🧠 Asking AI to interpret prompt: '{user_prompt}'")
    resp = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct", temperature=0.1, max_tokens=200,
        messages=[{"role": "user", "content": sys_prompt}]
    )
    
    raw = resp.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        print(f"❌ LLM Output Error. Raw response: {raw}")
        raise ValueError("LLM did not return a valid JSON object.")
        
    clean_json_string = match.group(0)
    parsed = json.loads(clean_json_string)
    print(f"✅ AI Parsed Intent: {json.dumps(parsed, indent=2)}")
    return parsed

# ==========================================
# 🎨 MATERIAL BANK & LIGHTING
# ==========================================
def get_material_from_bank(color, material):
    if not MATERIAL_BANK_DIR.exists(): return None, color
    
    safe_color = str(color).lower() if color and str(color).lower() not in ["none", "unknown", ""] else "original"
    safe_material = str(material).lower()
    
    if safe_material in PLAUSIBLE_COMBINATIONS:
        if safe_color not in PLAUSIBLE_COMBINATIONS[safe_material]:
            print(f"   ⚠️ GUARDRAIL TRIGGERED: '{safe_color} {safe_material}' is not architecturally realistic.")
            safe_color = "original"
    
    target_folder_name = f"{safe_color}_{safe_material}"
    target_folder = MATERIAL_BANK_DIR / target_folder_name
    
    if not target_folder.exists():
        target_folder = MATERIAL_BANK_DIR / f"original_{safe_material}"
    if not target_folder.exists(): return None, safe_color

    images = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.jpeg")) + list(target_folder.glob("*.png"))
    if not images: return None, safe_color
    return Image.open(random.choice(images)).convert("RGB"), safe_color

def apply_luminance_transfer(original_img, generated_img, blend_ratio=0.65):
    orig_np = np.array(original_img).astype(np.float32) / 255.0
    gen_np = np.array(generated_img).astype(np.float32) / 255.0
    
    def get_lum(img_array):
        return 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
        
    orig_lum = get_lum(orig_np)
    gen_lum = get_lum(gen_np)
    
    orig_lum_pil = Image.fromarray((np.clip(orig_lum, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    gen_lum_pil = Image.fromarray((np.clip(gen_lum, 0, 1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5))
    
    orig_lum_blur = np.array(orig_lum_pil).astype(np.float32) / 255.0
    gen_lum_blur = np.array(gen_lum_pil).astype(np.float32) / 255.0
    gen_lum_blur = np.clip(gen_lum_blur, 0.001, 1.0)
    
    ratio = np.expand_dims(orig_lum_blur / gen_lum_blur, axis=-1)
    matched_np = np.clip(gen_np * ratio, 0, 1)
    final_np = (gen_np * (1.0 - blend_ratio)) + (matched_np * blend_ratio)
    
    return Image.fromarray((final_np * 255).astype(np.uint8))

def save_debug_image(img, name, out_prefix):
    """Helper to save intermediate visualization steps."""
    DEBUG_DIR.mkdir(exist_ok=True, parents=True)
    img.save(DEBUG_DIR / f"{out_prefix}_{name}.jpg")

# ==========================================
# 🖌️ PIPELINES
# ==========================================

def run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    naive_mask = Image.new("RGB", (1024, 1024), (255, 255, 255))
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    gray_target_image = target_image.convert('L').convert('RGB')
    init_img = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"  -> Running Naive Edit (No Segmentation)...")
    naive_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=naive_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    naive_result.save(OUTPUT_DIR / f"{out_prefix}_naive_edited.jpg")
    del naive_result; gc.collect(); torch.cuda.empty_cache()

def run_proposed_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    target_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).convert("RGB").resize((1024, 1024))
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(target_image, invert_target_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"  -> Running Proposed Edit (Segmented)...")
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=target_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    guided_result.save(OUTPUT_DIR / f"{out_prefix}_proposed_edited.jpg")
    del guided_result; gc.collect(); torch.cuda.empty_cache()

def run_improved_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    # 1. Load Original & Material
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    save_debug_image(target_image, "01_original_image", out_prefix)
    
    ip_image = material_img.convert('RGB').resize((224, 224))
    save_debug_image(ip_image, "02_ip_adapter_image", out_prefix)
    
    # 2. Prepare Masks
    raw_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).resize((1024, 1024))
    save_debug_image(raw_mask, "03_binary_mask", out_prefix)
    
    feathered_mask = raw_mask.filter(ImageFilter.GaussianBlur(radius=5)) 
    save_debug_image(feathered_mask, "04_feathered_mask", out_prefix)
    
    # 3. Depth Estimation
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    save_debug_image(depth_image, "05_depth_map", out_prefix)
    
    # 4. Grayscale Initialization Composite
    invert_feathered_mask = ImageChops.invert(feathered_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    grayscale_img = Image.composite(gray_target_image, Image.new("RGB", (1024, 1024), "black"), feathered_mask)
    img_black_mask = Image.composite(target_image, Image.new("RGB", (1024, 1024), "black"), invert_feathered_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    save_debug_image(init_img, "06_grayscale_initialization", out_prefix)
    
    negative_prompt = "lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"  -> Running Improved Edit (Lighting Match + Compositing)...")
    
    # 5. Raw Generation (Guided Result)
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=feathered_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, 
        strength=0.85, 
    )[0]
    save_debug_image(guided_result, "07_raw_guided_result", out_prefix)
    
    # 6. Luminance Transfer
    lighting_matched = apply_luminance_transfer(target_image, guided_result, blend_ratio=0.65)
    save_debug_image(lighting_matched, "08_lighting_matched", out_prefix)
    
    # 7. Final Pixel-Perfect Composite
    final_result = Image.composite(lighting_matched, target_image, feathered_mask)
    save_debug_image(final_result, "09_final_output", out_prefix)
    
    final_result.save(OUTPUT_DIR / f"{out_prefix}_improved_edited.jpg")
    
    del guided_result, lighting_matched, final_result; gc.collect(); torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Run generation pipeline with a natural language prompt.")
    parser.add_argument("--prompt", type=str, default=None, help="The natural language edit prompt.")
    parser.add_argument("--json", type=str, default=None, help="JSON file storing instances labels")
    args = parser.parse_args()  
    
    if args.prompt:
        USER_PROMPT = args.prompt
    else:
        USER_PROMPT = input("Enter your edit prompt (e.g., 'Change the wall to white concrete'): ")

    if args.json:
        GT_PATH = args.json

    original_dir = os.getcwd()
    try:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        DEBUG_DIR.mkdir(exist_ok=True, parents=True)
        device = "cuda"

        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f).get("data", {})

        TARGET_IMAGE_ID = next(iter(gt_data)) #"1003317145127470"
        if TARGET_IMAGE_ID not in gt_data: return

        spatial_info = get_spatial_info(gt_data, TARGET_IMAGE_ID)
        parsed_intent = parse_nlp_prompt(USER_PROMPT, spatial_info)
        
        target_bid = parsed_intent.get("target_instance")
        target_struct = parsed_intent.get("target_structure", "wall")
        target_color = parsed_intent.get("new_color")
        target_material = parsed_intent.get("new_material")

        inst_data = gt_data[TARGET_IMAGE_ID]["instances"][target_bid][target_struct]
        target_mask = inst_data["mask"].replace('\\', '/')
        rgb_path = gt_data[TARGET_IMAGE_ID]["rgb"].replace('\\', '/')

        print(f"\n🔍 Retrieving reference image of '{target_color} {target_material}' from Material Bank...")
        material_pil_img, safe_color = get_material_from_bank(target_color, target_material)
        if material_pil_img is None: return

        print("\n⏳ Loading Models...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
        ).to(device)
        pipe.unet = register_cross_attention_hook(pipe.unet)
        ip_model = IPAdapterXL(pipe, "models/image_encoder", "sdxl_models/ip-adapter_sdxl_vit-h.bin", device)
        
        display_color = "" if safe_color == "original" else safe_color
        gen_prompt = f"a highly detailed building facade made of {display_color} {target_material}, featuring architectural outlines".replace("  ", " ").strip()
        out_prefix = f"{TARGET_IMAGE_ID}_{target_bid}"
        
        run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_pil_img, gen_prompt, out_prefix)
        run_proposed_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, gen_prompt, out_prefix)
        run_improved_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, gen_prompt, out_prefix)
                
        print(f"\n✅ Done! Check the {OUTPUT_DIR.name} folder.")
        print(f"   Note: Intermediate debug images saved to {DEBUG_DIR.name}/")
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()