#!/usr/bin/env python3
import os
import json
import random
import torch
import numpy as np
import gc
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
from pathlib import Path

# ==========================================
# COMPATIBILITY PATCHES
# ==========================================
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import transformers.utils
if not hasattr(transformers.utils, 'FLAX_WEIGHTS_NAME'):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

# ZEST Pipeline Imports
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from zest_code.ip_adapter import IPAdapterXL
from zest_code.ip_adapter.utils import register_cross_attention_hook
from transformers import pipeline

GT_PATH = "../ground_truth_eval.json" 
OUTPUT_DIR = Path("../zest_benchmark_results")
BENCHMARK_JSON_PATH = Path("../benchmark_dataset.json")

# Number of images to generate for the evaluation benchmark
NUM_SAMPLES = 50  

# ==========================================
# ZEST PIPELINE FUNCTIONS
# ==========================================
def crop_image_to_mask(rgb_path):
    img_pil = Image.open(rgb_path).convert("RGB")
    img = np.array(img_pil)
    mask = np.any(img > 10, axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0: return img_pil 
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return Image.fromarray(img[y0:y1+1, x0:x1+1])

def run_proposed_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    """Runs the ZEST pipeline using the structural mask."""
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

    print(f"    -> Generating Proposed Edit...")
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=target_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    out_path = OUTPUT_DIR / f"{out_prefix}_proposed.jpg"
    guided_result.save(out_path)
    
    del guided_result
    gc.collect()
    torch.cuda.empty_cache()
    return str(out_path)

def run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_img, prompt, out_prefix):
    """Runs the ZEST pipeline globally (white mask) for baseline comparison."""
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
    naive_result.save(out_path)
    
    del naive_result
    gc.collect()
    torch.cuda.empty_cache()
    return str(out_path)

def run_improved_pipeline(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    """Improved pipeline: Adds mask feathering and pixel-perfect post-compositing to bypass VAE degradation."""
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    ip_image = material_img.convert('RGB').resize((224, 224))
    
    # 1. Mask Feathering (Gaussian Blur to soften edges)
    raw_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).resize((1024, 1024))
    feathered_mask = raw_mask.filter(ImageFilter.GaussianBlur(radius=5))
    
    depth_image = depth_estimator(target_image)["depth"].convert("RGB").resize((1024, 1024))
    
    # Prepare init image using the feathered mask
    invert_feathered_mask = ImageChops.invert(feathered_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    
    grayscale_img = Image.composite(gray_target_image, Image.new("RGB", (1024, 1024), "black"), feathered_mask)
    img_black_mask = Image.composite(target_image, Image.new("RGB", (1024, 1024), "black"), invert_feathered_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    print(f"    -> Generating Improved Edit (Feathering + Compositing)...")
    guided_result = ip_model.generate(
        pil_image=ip_image, num_samples=1, num_inference_steps=30, seed=42, 
        image=init_img, control_image=depth_image, mask_image=feathered_mask,         
        prompt=prompt, negative_prompt=negative_prompt, scale=1.0, 
        controlnet_conditioning_scale=0.85, strength=0.9, 
    )[0]
    
    # 2. Pixel-Perfect Post-Compositing
    # Paste the generated SDXL pixels strictly back onto the pure original image using the feathered mask!
    final_result = Image.composite(guided_result, target_image, feathered_mask)
    
    out_path = OUTPUT_DIR / f"{out_prefix}_improved.jpg"
    final_result.save(out_path)
    
    del guided_result, final_result
    gc.collect()
    torch.cuda.empty_cache()
    return str(out_path)

def build_reference_pool(gt_data):
    """Creates a catalog of valid materials we can use to drive the edits (buildings + roads)."""
    pool = []
    for img_id, data in gt_data.items():
        for bid, inst in data.get("instances", {}).items():
            # 1. Building Instances (Nested dictionary)
            if bid.startswith("building_"):
                for struct in ["wall", "roof", "door", "window"]:
                    if struct in inst and not inst[struct].get("skipped"):
                        desc = inst[struct].get("material_descriptor", {})
                        c, m = desc.get("color", "unknown"), desc.get("class", "unknown")
                        if c != "unknown" and m != "unknown":
                            pool.append({
                                "img_id": img_id, "bid": bid, "struct": struct,
                                "color": c, "material": m, 
                                "masked_rgb": "../" + str(inst[struct].get("masked_rgb")).replace('\\', '/')
                            })
            # 2. Road & Sidewalk Instances (Flat dictionary)
            elif bid.startswith("road_") or bid.startswith("sidewalk_"):
                if not inst.get("skipped"):
                    struct = inst.get("type", bid.split("_")[0])
                    desc = inst.get("material_descriptor", {})
                    c, m = desc.get("color", "unknown"), desc.get("class", "unknown")
                    if c != "unknown" and m != "unknown":
                        pool.append({
                            "img_id": img_id, "bid": bid, "struct": struct,
                            "color": c, "material": m, 
                            "masked_rgb": "../" + str(inst.get("masked_rgb")).replace('\\', '/')
                        })
    return pool

def main():
    original_dir = os.getcwd()
    try:
        os.chdir("zest_code")
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        device = "cuda"

        print("📖 Loading Ground Truth data...")
        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f).get("data", {})

        reference_pool = build_reference_pool(gt_data)
        if not reference_pool:
            print("❌ No valid reference images found in Ground Truth.")
            return

        # Load Existing Benchmark Progress to allow resuming
        if BENCHMARK_JSON_PATH.exists():
            with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
                benchmark_results = json.load(f)
        else:
            benchmark_results = {}

        print("\n⏳ Loading ZEST Models (ControlNet & IPAdapter)...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
        ).to(device)
        pipe.unet = register_cross_attention_hook(pipe.unet)
        ip_model = IPAdapterXL(pipe, "models/image_encoder", "sdxl_models/ip-adapter_sdxl_vit-h.bin", device)
        print("🚀 Models Loaded.")

        # Get list of possible target images
        all_imgs = list(gt_data.keys())
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(all_imgs)

        generated_count = len(benchmark_results)

        for img_id in all_imgs:
            if generated_count >= NUM_SAMPLES:
                break
            if img_id in benchmark_results:
                continue

            instances = gt_data[img_id].get("instances", {})
            valid_targets = []
            
            for bid, inst in instances.items():
                if bid.startswith("building_"):
                    # We can edit walls and roofs of buildings
                    for struct in ["wall", "roof"]:
                        if struct in inst and not inst[struct].get("skipped"):
                            valid_targets.append((bid, struct, inst[struct]))
                elif bid.startswith("road_") or bid.startswith("sidewalk_"):
                    # We can edit roads and sidewalks directly
                    if not inst.get("skipped"):
                        struct = inst.get("type", bid.split("_")[0])
                        valid_targets.append((bid, struct, inst))
            
            if not valid_targets: continue

            # Pick a random structure to edit in this image
            target_bid, target_struct, inst_data = random.choice(valid_targets)
            
            orig_color = inst_data.get("material_descriptor", {}).get("color")
            orig_material = inst_data.get("material_descriptor", {}).get("class")

            # Pick a random reference material from a DIFFERENT image
            valid_refs = [r for r in reference_pool if r["img_id"] != img_id and r["struct"] == target_struct and (r["color"] != orig_color or r["material"] != orig_material)]
            if not valid_refs: continue

            ref = random.choice(valid_refs)
            target_mask = "../" + str(inst_data.get("mask", "")).replace('\\', '/')
            rgb_path = "../" + str(gt_data[img_id].get("rgb", "")).replace('\\', '/')
            
            if not Path(target_mask).exists() or not Path(rgb_path).exists(): continue

            user_prompt = f"Change the {target_struct} of {target_bid} to {ref['color']} {ref['material']}"
            
            # Context-Aware ZEST Prompts
            if target_struct in ["road", "sidewalk"]:
                zest_prompt = f"a highly detailed {target_struct} surface made of {ref['color']} {ref['material']}, featuring ground textures, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            elif target_struct == "roof":
                zest_prompt = f"a highly detailed building roof made of {ref['color']} {ref['material']}, featuring architectural geometry, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            else:
                zest_prompt = f"a highly detailed building facade made of {ref['color']} {ref['material']}, featuring architectural outlines, photorealistic street photography, matching natural sunlight, ambient occlusion, 8k resolution"
            out_prefix = f"{img_id}_{target_bid}_{target_struct}"

            print(f"\n[{generated_count+1}/{NUM_SAMPLES}] Editing {img_id} -> {user_prompt}")
            
            # Run Pipelines
            material_pil_img = crop_image_to_mask(ref["masked_rgb"])
            naive_path = run_naive_pipeline(ip_model, depth_estimator, rgb_path, material_pil_img, zest_prompt, out_prefix)
            proposed_path = run_proposed_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, zest_prompt, out_prefix)
            improved_path = run_improved_pipeline(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, zest_prompt, out_prefix)

            # Save Results
            benchmark_results[img_id] = {
                "target_instance": target_bid,
                "target_structure": target_struct,
                "original_color": orig_color,
                "original_material": orig_material,
                "new_color": ref["color"],
                "new_material": ref["material"],
                "simulated_user_prompt": user_prompt,
                "zest_prompt": zest_prompt,
                "mask_path": target_mask,
                "original_image": f"{OUTPUT_DIR.name}/{out_prefix}_original.jpg",
                "proposed_image": f"{OUTPUT_DIR.name}/{out_prefix}_proposed.jpg",
                "improved_image": f"{OUTPUT_DIR.name}/{out_prefix}_improved.jpg",
                "naive_image": f"{OUTPUT_DIR.name}/{out_prefix}_naive.jpg"
            }

            with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(benchmark_results, f, indent=4)
            
            generated_count += 1

        print(f"\n🎉 Benchmark Generation Complete! Saved {generated_count} pairs to {BENCHMARK_JSON_PATH}")

    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()