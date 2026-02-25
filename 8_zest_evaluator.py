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
import json
import torch
import numpy as np
import gc
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
import os

# ZEST Pipeline Imports
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from zest_code.ip_adapter import IPAdapterXL
from zest_code.ip_adapter.utils import register_cross_attention_hook
from transformers import pipeline

GT_PATH = "../ground_truth_eval.json" 
OUTPUT_DIR = Path("../zest_comparison_results")

def print_available_inventory(gt_data):
    """Scans the dataset and prints all available materials and their colors."""
    inventory = {}
    for img_id, data in gt_data.items():
        for bid, inst in data.get("instances", {}).items():
            for struct in ["wall", "roof", "road", "sidewalk"]:
                if struct in inst and not inst[struct].get("skipped"):
                    mat_desc = inst[struct].get("material_descriptor", {})
                    mat_class = mat_desc.get("class", "unknown")
                    mat_color = mat_desc.get("color", "unknown")
                    
                    if mat_class != "unknown":
                        if mat_class not in inventory:
                            inventory[mat_class] = set()
                        if mat_color != "unknown":
                            inventory[mat_class].add(mat_color)
                            
    print("\n📋 AVAILABLE MATERIAL & COLOR INVENTORY:")
    for mat, colors in sorted(inventory.items()):
        colors_str = ", ".join(sorted(colors)) if colors else "No specific colors"
        print(f"  🔸 {mat.upper()}: {colors_str}")
    print("-" * 50 + "\n")

def get_biggest_reference(gt_data, target_material, target_color, struct_type, exclude_img_id, exclude_bid):
    """Scans dataset to find the instance with the largest mask matching material and color."""
    best_candidate = None
    max_area = -1
    candidates_found = 0
    
    for img_id, data in gt_data.items():
        for bid, inst in data.get("instances", {}).items():
            if img_id == exclude_img_id and bid == exclude_bid:
                continue
                
            if struct_type in inst and not inst[struct_type].get("skipped"):
                mat_desc = inst[struct_type].get("material_descriptor", {})
                mat_class = mat_desc.get("class", "unknown")
                mat_color = mat_desc.get("color", "unknown")
                
                # Check BOTH material and color
                if mat_class == target_material and mat_color == target_color:
                    mask_path = "../" + str(inst[struct_type].get("mask")).replace('\\', '/')
                    rgb_path = "../" + str(data.get("rgb")).replace('\\', '/')
                    masked_rgb_path = "../" + str(inst[struct_type].get("masked_rgb")).replace('\\', '/')
                    
                    if Path(mask_path).exists() and Path(rgb_path).exists():
                        candidates_found += 1
                        
                        mask_pil = Image.open(mask_path).convert("L")
                        mask_np = np.array(mask_pil) > 127
                        area = mask_np.sum()
                        
                        if area > max_area:
                            max_area = area
                            best_candidate = {
                                "img_id": img_id,
                                "bid": bid,
                                "masked_rgb": masked_rgb_path
                            }
                            
    print(f"    -> Evaluated {candidates_found} candidates. Biggest mask area: {max_area} pixels")
    return best_candidate

def crop_image_to_mask(rgb_path):
    """Crops the image directly to its non-black bounding box."""
    img_pil = Image.open(rgb_path).convert("RGB")
    img = np.array(img_pil)
    
    mask = np.any(img > 10, axis=-1)
    coords = np.argwhere(mask)
    
    if coords.size == 0: return img_pil 
        
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return Image.fromarray(img[y0:y1+1, x0:x1+1])

def run_ab_comparison(ip_model, depth_estimator, rgb_path, mask_path, material_img, prompt, out_prefix):
    target_image = Image.open(rgb_path).convert('RGB').resize((1024, 1024))
    target_image.save(OUTPUT_DIR / f"{out_prefix}_original.jpg")

    ip_image = material_img.convert('RGB').resize((224, 224))
    target_mask = Image.open(mask_path).convert("L").point(lambda x: 0 if x < 127 else 255).convert("RGB").resize((1024, 1024))
    
    print("  -> Calculating 3D Depth Map to preserve architecture...")
    depth_image = depth_estimator(target_image)["depth"]
    depth_image = depth_image.convert("RGB").resize((1024, 1024))
    
    # Save the Depth Map so you can verify what the AI sees structurally
    depth_image.save(OUTPUT_DIR / f"{out_prefix}_DEPTH_MAP.jpg")
    print("  -> Saved Depth Map for visual verification.")
    
    invert_target_mask = ImageChops.invert(target_mask)
    gray_target_image = target_image.convert('L').convert('RGB')
    gray_target_image = ImageEnhance.Brightness(gray_target_image).enhance(1.0)
    grayscale_img = ImageChops.darker(gray_target_image, target_mask)
    img_black_mask = ImageChops.darker(target_image, invert_target_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)
    
    negative_prompt = "blurry, lowres, flat, solid block, missing windows, smooth, loss of architectural details"

    # ==========================================
    # RUN 1: NAIVE ZEST
    # ==========================================
    print(f"  -> Running Naive ZEST...")
    naive_mask = Image.new("RGB", (1024, 1024), (255, 255, 255))
    
    naive_result = ip_model.generate(
        pil_image=ip_image, 
        num_samples=1, 
        num_inference_steps=30, 
        seed=42, 
        image=target_image,             
        control_image=depth_image,      
        mask_image=naive_mask,          
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        scale=1.0, 
        controlnet_conditioning_scale=0.85, 
        strength=0.9, 
    )[0]
    naive_result.save(OUTPUT_DIR / f"{out_prefix}_naive.jpg")
    
    del naive_result
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # RUN 2: SAM3-GUIDED ZEST
    # ==========================================
    print(f"  -> Running SAM3-Guided ZEST...")
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
    guided_result.save(OUTPUT_DIR / f"{out_prefix}_guided.jpg")
    
    del guided_result
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # Save original directory
    original_dir = os.getcwd()

    try:
        # Change directory
        os.chdir("zest_code")
        OUTPUT_DIR.mkdir(exist_ok=True)
        device = "cuda"

        with open(GT_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f).get("data", {})

        # 1. Print available options before loading models
        print_available_inventory(gt_data)

        print("⏳ Loading Models...")
        
        # We load the Depth Estimator here!
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas", device=0)
        
        # Reverted to Depth ControlNet
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

        # ---------------------------------------------------------
        # SET YOUR TARGETS HERE BASED ON THE INVENTORY PRINTED ABOVE
        # ---------------------------------------------------------
        target_material = "brick" 
        target_color = "red"
        # ---------------------------------------------------------

        target_mask, rgb_path, test_img_id, bid_name, struct_type = None, None, None, "", ""
        
        # 2. Select TARGET
        for img_id, data in gt_data.items():
            for bid, inst in data.get("instances", {}).items():
                if "wall" in inst and not inst["wall"].get("skipped"):
                    target_mask = "../" + inst["wall"]["mask"].replace('\\', '/')
                    rgb_path = "../" + data["rgb"].replace('\\', '/')
                    test_img_id = img_id
                    bid_name = bid
                    struct_type = "wall"
                    break
            if target_mask: break

        if target_mask:
            print(f"\n🧪 Editing Target: {test_img_id} ({bid_name} {struct_type})")
            
            print(f"🔍 Searching dataset for the BIGGEST '{target_color} {target_material}' match...")
            ref_data = get_biggest_reference(gt_data, target_material, target_color, struct_type, test_img_id, bid_name)
            
            if not ref_data:
                print(f"❌ Could not find any other '{struct_type}' made of '{target_color} {target_material}'!")
                return
                
            print(f"✅ Found Optimal Reference: {ref_data['img_id']} ({ref_data['bid']})")
            
            ref_rgb_path = ref_data["masked_rgb"]
            material_pil_img = crop_image_to_mask(ref_rgb_path)
            material_pil_img.save(OUTPUT_DIR / f"{test_img_id}_{bid_name}_OPTIMAL_REFERENCE.jpg")

            # EXPLICITLY TELL THE AI THE MATERIAL AND COLOR
            prompt = f"a highly detailed building facade made of {target_color} {target_material}, featuring glass windows, doors, and architectural outlines"
            
            run_ab_comparison(ip_model, depth_estimator, rgb_path, target_mask, material_pil_img, prompt, f"{test_img_id}_{bid_name}")
            
            print(f"\n✅ Done! Check the {OUTPUT_DIR.name} folder.")

    finally:
        # Restore original directory
        os.chdir(original_dir)

    print("Back to:", os.getcwd())

if __name__ == "__main__":
    main()