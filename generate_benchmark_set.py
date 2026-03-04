#!/usr/bin/env python3
import os
import json
import random
import torch
import numpy as np
import gc
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from pathlib import Path
import argparse

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

GT_JSON_PATH = "ground_truth_eval.json" 
OUTPUT_DIR = Path("benchmark_results")
BENCHMARK_JSON_PATH = Path("benchmark_dataset.json")
MATERIAL_BANK_DIR = Path("colored_material_bank")

# Number of images to generate for the evaluation benchmark
NUM_SAMPLES = 300  

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
# LUMINANCE TRANSFER
# ==========================================
def apply_luminance_transfer(original_img, generated_img, blend_ratio=0.65):
    """Transfers the macro-lighting (shadows/ambient occlusion) from the original photo to the generated texture."""
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

# ==========================================
# PIPELINE FUNCTIONS
# ==========================================

def run_proposed_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    target_image.save(OUTPUT_DIR / f"{out_prefix}_original.jpg")

    ip_image = material_img.convert('RGB').resize((224, 224))
    target_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).convert("RGB").resize((1024, 1024))
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(target_image, invert_target_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"    -> Generating Proposed Edit...")
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=target_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    out_path = OUTPUT_DIR / f"{out_prefix}_proposed.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    guided_result.save(out_path)
    del guided_result; gc.collect(); torch.cuda.empty_cache()
    return str(out_path)

def run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    
    naive_mask = Image.new("RGB", (1024, 1024), (255, 255, 255))
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    gray_target_image = target_image.convert('L').convert('RGB')
    init_img = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"    -> Generating Naive Edit...")
    naive_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=naive_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    out_path = OUTPUT_DIR / f"{out_prefix}_naive.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    naive_result.save(out_path)
    del naive_result; gc.collect(); torch.cuda.empty_cache()
    return str(out_path)

def run_improved_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    
    raw_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).resize((1024, 1024))
    feathered_mask = raw_mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    invert_feathered_mask = ImageChops.invert(feathered_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    grayscale_img = Image.composite(gray_target_image, Image.new("RGB", (1024, 1024), "black"), feathered_mask)
    img_black_mask = Image.composite(target_image, Image.new("RGB", (1024, 1024), "black"), invert_feathered_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    
    negative_prompt = "lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"    -> Generating Improved Edit (Lighting Match + Compositing)...")
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=feathered_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.85, 
    )[0]
    
    lighting_matched = apply_luminance_transfer(target_image, guided_result, blend_ratio=0.65)
    final_result = Image.composite(lighting_matched, target_image, feathered_mask)
    
    out_path = OUTPUT_DIR / f"{out_prefix}_improved.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_result.save(out_path)
    del guided_result, lighting_matched, final_result; gc.collect(); torch.cuda.empty_cache()
    return str(out_path)

def get_material_from_bank(color, material):
    target_folder = MATERIAL_BANK_DIR / f"{color}_{material}"
    if not target_folder.exists(): return None
    images = list(target_folder.glob("*.jpg")) + list(target_folder.glob("*.jpeg")) + list(target_folder.glob("*.png"))
    if not images: return None
    return Image.open(random.choice(images)).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Run generation pipeline with a natural language prompt.")
    parser.add_argument("--benchmark-json", type=str, default=None, help="JSON file storing benchmark labels")
    parser.add_argument("--output-dir", type=str, default=None, help="Output folder directory")
    parser.add_argument("--gt-json", type=str, default=None, help="JSON file storing instances ground truth labels")
    args = parser.parse_args()  

    if args.benchmark_json:
        BENCHMARK_JSON_PATH = Path(args.benchmark_json)

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    if args.gt_json:
        GT_JSON_PATH = args.gt_json

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    original_dir = os.getcwd()
    try:
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        device = "cuda"

        print("📖 Loading Ground Truth data...")
        with open(GT_JSON_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f).get("data", {})

        # Load ONLY plausible materials from the Material Bank
        available_combos = []
        if MATERIAL_BANK_DIR.exists():
            for d in MATERIAL_BANK_DIR.iterdir():
                if d.is_dir() and '_' in d.name:
                    c, m = d.name.split('_', 1)
                    # ✅ The Guardrail: Only append if it's architecturally realistic!
                    if m in PLAUSIBLE_COMBINATIONS and c in PLAUSIBLE_COMBINATIONS[m]:
                        available_combos.append((c, m))
                    
        if not available_combos:
            print(f"❌ Material bank not found or lacks plausible materials at {MATERIAL_BANK_DIR}")
            return
        print(f"🎨 Found {len(available_combos)} realistic, plausible material styles in the bank.")

        if BENCHMARK_JSON_PATH.exists():
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
        else:
            benchmark_results = {}

        print("\n⏳ Loading Models (ControlNet & IPAdapter)...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
        ).to(device)
        pipe.unet = register_cross_attention_hook(pipe.unet)
        ip_model = IPAdapterXL(pipe, "models/image_encoder", "sdxl_models/ip-adapter_sdxl_vit-h.bin", device)
        print("🚀 Models Loaded.")

        all_imgs = list(gt_data.keys())
        random.seed(42)  
        random.shuffle(all_imgs)

        generated_count = len(benchmark_results)

        for img_id in all_imgs:
            if generated_count >= NUM_SAMPLES: break
            if img_id in benchmark_results: continue

            instances = gt_data[img_id].get("instances", {})
            valid_targets = []
            
            for bid, inst in instances.items():
                if bid.startswith("building_"):
                    for struct in ["wall", "roof"]:
                        if struct in inst and not inst[struct].get("skipped"):
                            valid_targets.append((bid, struct, inst[struct]))
                elif bid.startswith("road_") or bid.startswith("sidewalk_"):
                    if not inst.get("skipped"):
                        struct = inst.get("type", bid.split("_")[0])
                        valid_targets.append((bid, struct, inst))
            
            if not valid_targets: continue

            target_bid, target_struct, inst_data = random.choice(valid_targets)
            orig_color = inst_data.get("material_descriptor", {}).get("color", "unknown")
            orig_material = inst_data.get("material_descriptor", {}).get("class", "unknown")

            # Pick a random material combo from the FILTERED plausible list
            valid_combos = [c for c in available_combos if (c[0] != orig_color or c[1] != orig_material)]
            if not valid_combos: continue

            new_color, new_material = random.choice(valid_combos)
            
            target_mask = str(inst_data.get("mask", "")).strip().replace('\\', '/')
            rgb_path = str(gt_data[img_id].get("rgb", "")).strip().replace('\\', '/')
            
            # 🛡️ THE FIX: Check if string is empty AND ensure it's an actual file
            if not target_mask or not rgb_path or not Path(target_mask).is_file() or not Path(rgb_path).is_file(): 
                continue

            material_pil_img = get_material_from_bank(new_color, new_material)
            if material_pil_img is None: continue

            display_color = "" if new_color == "original" else new_color
            user_prompt = f"Change the {target_struct} of {target_bid} to {display_color} {new_material}".replace("  ", " ").strip()
            
            if target_struct in ["road", "sidewalk"]:
                gen_prompt = f"a highly detailed {target_struct} surface made of {display_color} {new_material}, featuring ground textures, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            elif target_struct == "roof":
                gen_prompt = f"a highly detailed building roof made of {display_color} {new_material}, featuring architectural geometry, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            else:
                gen_prompt = f"a highly detailed building facade made of {display_color} {new_material}, featuring architectural outlines, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            
            gen_prompt = gen_prompt.replace("  ", " ").strip()
            out_prefix = f"{img_id}_{target_bid}_{target_struct}"

            print(f"\n[{generated_count+1}/{NUM_SAMPLES}] Editing {img_id} -> {user_prompt}")
            
            naive_path = run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_pil_img, gen_prompt, out_prefix)
            proposed_path = run_proposed_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, gen_prompt, out_prefix)
            improved_path = run_improved_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, gen_prompt, out_prefix)

            benchmark_results[img_id] = {
                "target_instance": target_bid,
                "target_structure": target_struct,
                "original_color": orig_color,
                "original_material": orig_material,
                "new_color": new_color,
                "new_material": new_material,
                "simulated_user_prompt": user_prompt,
                "gen_prompt": gen_prompt,
                "mask_path": target_mask,
                "original_image": f"{OUTPUT_DIR}/{out_prefix}_original.jpg",
                "proposed_image": f"{OUTPUT_DIR}/{out_prefix}_proposed.jpg",
                "improved_image": f"{OUTPUT_DIR}/{out_prefix}_improved.jpg",
                "naive_image": f"{OUTPUT_DIR}/{out_prefix}_naive.jpg"
            }

            with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=4)
            
            generated_count += 1

        print(f"\n🎉 Benchmark Generation Complete! Saved {generated_count} pairs to {BENCHMARK_JSON_PATH}")

    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()