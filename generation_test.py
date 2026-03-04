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

    valid_ids = [str(k).strip() for k in instances_info.keys()]

    proxy_url = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY') or os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    if proxy_url and not proxy_url.startswith('http'): proxy_url = f"http://{proxy_url}"
    
    custom_http_client = httpx.Client(proxies=proxy_url, verify=False) if proxy_url else httpx.Client(verify=False)
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, http_client=custom_http_client)

    sys_prompt = f"""You are a machine that outputs ONLY valid JSON. NO markdown format, NO explanations.
CONTEXT (Available IDs): {json.dumps(valid_ids)}
PROPERTIES: {json.dumps(instances_info, indent=2)}

TASK: Map User Prompt to a "target_instance" ID from the CONTEXT based on properties like "left", "right", or "area_pixels".

EXAMPLE Output format exactly like this:
{{
  "target_instance": "building_05",
  "target_structure": "wall",
  "new_color": "red",
  "new_material": "brick"
}}

User Prompt: "{user_prompt}"
"""
    
    print(f"🧠 Asking AI to interpret prompt...")
    parsed = {}
    try:
        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct", 
            temperature=0.0,
            max_tokens=100,
            messages=[{"role": "user", "content": sys_prompt}]
        )
        raw = resp.choices[0].message.content.strip()
        # Clean markdown code blocks if the LLM ignores instructions
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
    except Exception as e:
        print(f"⚠️ LLM parsing error: {e}")
    
    llm_id = str(parsed.get("target_instance", "")).strip()

    # --- SMARTER FALLBACK LOGIC ---
    if llm_id not in valid_ids:
        print(f"⚠️ LLM Parsing Issue. Raw ID: '{llm_id}'.")
        print("   -> Triggering smart fallback based on user prompt keywords...")
        
        if instances_info:
            prompt_lower = user_prompt.lower().replace(",", "")
            
            # 1. Filter by position
            candidates = instances_info
            if "right" in prompt_lower:
                candidates = {k: v for k, v in instances_info.items() if v.get("horizontal_position") == "right"}
            elif "left" in prompt_lower:
                candidates = {k: v for k, v in instances_info.items() if v.get("horizontal_position") == "left"}
            elif "center" in prompt_lower:
                candidates = {k: v for k, v in instances_info.items() if v.get("horizontal_position") == "center"}
            
            # If position filter empties the list, revert to all instances
            if not candidates: 
                candidates = instances_info

            # 2. Select by size
            if "smallest" in prompt_lower:
                fallback_id = min(candidates, key=lambda k: candidates[k]['area_pixels'])
            else: # Default to biggest
                fallback_id = max(candidates, key=lambda k: candidates[k]['area_pixels'])
            
            parsed["target_instance"] = fallback_id
            
            # 3. Extract material and color
            words = prompt_lower.split()
            materials = ["brick", "concrete", "wood", "stone", "plaster", "metal", "glass", "painted"]
            colors = ["red", "white", "brown", "gray", "black", "blue", "green", "yellow"]
            
            parsed["new_material"] = next((w for w in words if w in materials), "original")
            parsed["new_color"] = next((w for w in words if w in colors), "original")
            parsed["target_structure"] = "wall" 
    else:
        parsed["target_instance"] = llm_id

    print(f"✅ Final Parsed Intent: {json.dumps(parsed, indent=2)}")
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

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    DEBUG_DIR.mkdir(exist_ok=True, parents=True)
    device = "cuda"

    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f).get("data", {})

    print("\n⏳ Loading Models...")
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
    ).to(device)
    pipe.unet = register_cross_attention_hook(pipe.unet)
    ip_model = IPAdapterXL(pipe, "models/image_encoder", "sdxl_models/ip-adapter_sdxl_vit-h.bin", device)

    for img_id in gt_data.keys():
        try:
            spatial_info = get_spatial_info(gt_data, img_id)
            if not spatial_info:
                continue  # Skip if no instances

            print(f"🖼️ Attempting Image ID: {img_id}...")
            parsed_intent = parse_nlp_prompt(USER_PROMPT, spatial_info)
            if not parsed_intent:
                continue
                
            target_bid = parsed_intent.get("target_instance")
            target_struct = parsed_intent.get("target_structure", "wall")
            
            # Retrieve Mask Path
            inst_block = gt_data[img_id]["instances"].get(target_bid, {})
            mask_path = inst_block.get(target_struct, {}).get("mask") or inst_block.get("mask")
            if not mask_path:
                print(f"⚠️ Missing mask for {img_id}. Skipping.")
                continue
                
            rgb_path = gt_data[img_id].get("rgb")
            
            # Prepare Material
            t_mat = parsed_intent.get("new_material", "brick")
            t_col = parsed_intent.get("new_color", "brown")
            mat_img, safe_col = get_material_from_bank(t_col, t_mat)
            if not mat_img:
                print(f"⚠️ Material {t_col} {t_mat} not found. Skipping.")
                continue

            print(f"✅ Processing {img_id} -> {target_bid}...")
            
            # Run Generation
            gen_prompt = f"a high quality building facade made of {safe_col} {t_mat}"
            prefix = f"{img_id}_{target_bid}"
            
            run_naive_pipeline(ip_model, depth_estimator, rgb_path, mat_img, gen_prompt, prefix)
            run_proposed_pipeline(ip_model, depth_estimator, rgb_path, mask_path, mat_img, gen_prompt, prefix)
            run_improved_pipeline(ip_model, depth_estimator, rgb_path, mask_path, mat_img, gen_prompt, prefix)
            
            print(f"\n✅ Success! Generation Complete. Results in {OUTPUT_DIR}/")
            
            # STOP IMMEDIATELY AFTER THE FIRST SUCCESSFUL IMAGE
            break 
            
        except Exception as e:
            print(f"⚠️ Error on {img_id}: {e}. Trying next image...")
            continue

if __name__ == "__main__":
    main()