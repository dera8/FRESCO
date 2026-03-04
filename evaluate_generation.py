#!/usr/bin/env python3
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings
import argparse

# Suppress annoying warnings from external libraries
warnings.filterwarnings("ignore")

# Metrics
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms.functional as TF

# ==========================================
# CONFIGURATION
# ==========================================
BENCHMARK_JSON_PATH = Path("benchmark_dataset.json")
OUTPUT_JSON_PATH = "generation_evaluation_report.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image_tensor(path, size=(512, 512)):
    """Loads an image and converts it to a BCHW uint8 Tensor for FID."""
    img = Image.open(path).convert("RGB").resize(size)
    tensor = TF.pil_to_tensor(img).unsqueeze(0) # [1, 3, H, W] in uint8
    return tensor

def load_image_gray_np(path, size=(1024, 1024)):
    """Loads a grayscale numpy array for SSIM calculation."""
    img = Image.open(path).convert("L").resize(size)
    return np.array(img)

def resolve_path(path_str):
    """Fixes paths that were saved relative to the script's execution folder."""
    if path_str.startswith("../"):
        return Path(path_str[3:])
    return Path(path_str)

def get_masked_crop(img_pil, mask_np):
    """Applies the mask (blacking out background) and crops to the bounding box for Local CLIP."""
    img_array = np.array(img_pil)
    # Black out the background (~mask_np means where mask is False)
    img_array[~mask_np] = 0
    
    # Crop to the bounding box of the mask
    coords = np.argwhere(mask_np)
    if coords.size == 0:
        return img_pil
    
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    cropped = img_array[y0:y1+1, x0:x1+1]
    return Image.fromarray(cropped)

def main():
    parser = argparse.ArgumentParser(description="Run generation pipeline with a natural language prompt.")
    parser.add_argument("--benchmark-json", type=str, default=None, help="JSON file storing benchmark labels")
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON_PATH, help="Path to save the output JSON results")
    args = parser.parse_args()  

    if args.benchmark_json:
        BENCHMARK_JSON_PATH = Path(args.benchmark_json)

    if not BENCHMARK_JSON_PATH.exists():
        raise FileNotFoundError(f"❌ {BENCHMARK_JSON_PATH} not found. Run generate_benchmark_set.py first!")

    print("📖 Loading Benchmark Data...")
    with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    if not benchmark_data:
        print("❌ Benchmark dataset is empty.")
        return

    print("\n⏳ Loading CLIP Model for Text-Image Alignment...")
    clip_model_id = "openai/clip-vit-base-patch16"
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(DEVICE)
    clip_model.eval()

    print("⏳ Initializing FID Metrics...")
    # FID objects
    fid_original = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid_naive = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid_proposed = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid_improved = FrechetInceptionDistance(feature=2048).to(DEVICE)

    # Score trackers
    results = {
        "original": {"bg_ssim": [], "global_clip": [], "local_clip": []},
        "naive": {"bg_ssim": [], "global_clip": [], "local_clip": []},
        "proposed": {"bg_ssim": [], "global_clip": [], "local_clip": []},
        "improved": {"bg_ssim": [], "global_clip": [], "local_clip": []}
    }

    print("\n🚀 Starting Evaluation...")
    
    for img_id, entry in tqdm(benchmark_data.items(), desc="Evaluating Images"):
        # Resolve Paths
        orig_path = resolve_path(entry["original_image"])
        naive_path = resolve_path(entry["naive_image"])
        prop_path = resolve_path(entry["proposed_image"])
        improved_path = resolve_path(entry["improved_image"])
        mask_path = resolve_path(entry["mask_path"])
        prompt = entry["gen_prompt"]

        if not all([orig_path.exists(), naive_path.exists(), prop_path.exists(), improved_path.exists(), mask_path.exists()]):
            print(f"\n⚠️ Missing files for {img_id}, skipping...")
            continue

        # ---------------------------------------------------------
        # 1. FID Score Updates
        # ---------------------------------------------------------
        orig_tensor = load_image_tensor(orig_path).to(DEVICE)
        naive_tensor = load_image_tensor(naive_path).to(DEVICE)
        prop_tensor = load_image_tensor(prop_path).to(DEVICE)
        improved_tensor = load_image_tensor(improved_path).to(DEVICE)

        # Update real images (Original is the ground truth)
        fid_original.update(orig_tensor, real=True)
        fid_naive.update(orig_tensor, real=True)
        fid_proposed.update(orig_tensor, real=True)
        fid_improved.update(orig_tensor, real=True)

        # Update fake images
        fid_original.update(orig_tensor, real=False) # Orig vs Orig will be ~0
        fid_naive.update(naive_tensor, real=False)
        fid_proposed.update(prop_tensor, real=False)
        fid_improved.update(improved_tensor, real=False)

        # ---------------------------------------------------------
        # 2. Background SSIM (Geometry & Structure Preservation)
        # ---------------------------------------------------------
        orig_gray = load_image_gray_np(orig_path)
        naive_gray = load_image_gray_np(naive_path)
        prop_gray = load_image_gray_np(prop_path)
        improved_gray = load_image_gray_np(improved_path)
        
        # Load mask and invert it to get the "background" (the unedited regions)
        mask_img = Image.open(mask_path).convert("L").resize((1024, 1024))
        mask_np = np.array(mask_img) > 127
        bg_mask = ~mask_np  # True where the image should NOT have been edited

        # Calculate SSIM Maps
        _, orig_ssim_map = ssim(orig_gray, orig_gray, full=True, data_range=255) # Exactly 1.0 everywhere
        _, naive_ssim_map = ssim(orig_gray, naive_gray, full=True, data_range=255)
        _, prop_ssim_map = ssim(orig_gray, prop_gray, full=True, data_range=255)
        _, improved_ssim_map = ssim(orig_gray, improved_gray, full=True, data_range=255)

        # Extract SSIM only for the background
        if np.any(bg_mask):
            results["original"]["bg_ssim"].append(orig_ssim_map[bg_mask].mean())
            results["naive"]["bg_ssim"].append(naive_ssim_map[bg_mask].mean())
            results["proposed"]["bg_ssim"].append(prop_ssim_map[bg_mask].mean())
            results["improved"]["bg_ssim"].append(improved_ssim_map[bg_mask].mean())

        # ---------------------------------------------------------
        # 3. CLIP Scores (Global AND Local Alignment)
        # ---------------------------------------------------------
        orig_img = Image.open(orig_path).convert("RGB")
        naive_img = Image.open(naive_path).convert("RGB")
        prop_img = Image.open(prop_path).convert("RGB")
        improved_img = Image.open(improved_path).convert("RGB")

        # Create localized versions for Masked CLIP
        orig_local = get_masked_crop(orig_img, mask_np)
        naive_local = get_masked_crop(naive_img, mask_np)
        prop_local = get_masked_crop(prop_img, mask_np)
        improved_local = get_masked_crop(improved_img, mask_np)

        with torch.no_grad():
            # --- GLOBAL CLIP ---
            inputs_o = clip_processor(text=[prompt], images=orig_img, return_tensors="pt", padding=True).to(DEVICE)
            results["original"]["global_clip"].append(clip_model(**inputs_o).logits_per_image.item())
            
            inputs_n = clip_processor(text=[prompt], images=naive_img, return_tensors="pt", padding=True).to(DEVICE)
            results["naive"]["global_clip"].append(clip_model(**inputs_n).logits_per_image.item())
            
            inputs_p = clip_processor(text=[prompt], images=prop_img, return_tensors="pt", padding=True).to(DEVICE)
            results["proposed"]["global_clip"].append(clip_model(**inputs_p).logits_per_image.item())

            inputs_i = clip_processor(text=[prompt], images=improved_img, return_tensors="pt", padding=True).to(DEVICE)
            results["improved"]["global_clip"].append(clip_model(**inputs_i).logits_per_image.item())

            # --- LOCAL (MASKED) CLIP ---
            inputs_o_loc = clip_processor(text=[prompt], images=orig_local, return_tensors="pt", padding=True).to(DEVICE)
            results["original"]["local_clip"].append(clip_model(**inputs_o_loc).logits_per_image.item())
            
            inputs_n_loc = clip_processor(text=[prompt], images=naive_local, return_tensors="pt", padding=True).to(DEVICE)
            results["naive"]["local_clip"].append(clip_model(**inputs_n_loc).logits_per_image.item())
            
            inputs_p_loc = clip_processor(text=[prompt], images=prop_local, return_tensors="pt", padding=True).to(DEVICE)
            results["proposed"]["local_clip"].append(clip_model(**inputs_p_loc).logits_per_image.item())

            inputs_i_loc = clip_processor(text=[prompt], images=improved_local, return_tensors="pt", padding=True).to(DEVICE)
            results["improved"]["local_clip"].append(clip_model(**inputs_i_loc).logits_per_image.item())

    # ==========================================
    # AGGREGATE AND PRINT RESULTS
    # ==========================================
    print("\n⏳ Computing final FID scores (this takes a few seconds)...")
    try:
        final_fid_orig = 0.0 # Comparing Real images to Real images mathematically zeroes out
        final_fid_naive = float(fid_naive.compute().item())
        final_fid_prop = float(fid_proposed.compute().item())
        final_fid_improved = float(fid_improved.compute().item())
    except ValueError as e:
        print(f"\n⚠️ FID Warning: {e}")
        final_fid_orig = final_fid_naive = final_fid_prop = final_fid_improved = float('nan')

    mean_orig_ssim = np.mean(results["original"]["bg_ssim"]) if results["original"]["bg_ssim"] else 0
    mean_naive_ssim = np.mean(results["naive"]["bg_ssim"]) if results["naive"]["bg_ssim"] else 0
    mean_prop_ssim = np.mean(results["proposed"]["bg_ssim"]) if results["proposed"]["bg_ssim"] else 0
    mean_improved_ssim = np.mean(results["improved"]["bg_ssim"]) if results["improved"]["bg_ssim"] else 0

    mean_orig_gclip = np.mean(results["original"]["global_clip"]) if results["original"]["global_clip"] else 0
    mean_naive_gclip = np.mean(results["naive"]["global_clip"]) if results["naive"]["global_clip"] else 0
    mean_prop_gclip = np.mean(results["proposed"]["global_clip"]) if results["proposed"]["global_clip"] else 0
    mean_improved_gclip = np.mean(results["improved"]["global_clip"]) if results["improved"]["global_clip"] else 0

    mean_orig_lclip = np.mean(results["original"]["local_clip"]) if results["original"]["local_clip"] else 0
    mean_naive_lclip = np.mean(results["naive"]["local_clip"]) if results["naive"]["local_clip"] else 0
    mean_prop_lclip = np.mean(results["proposed"]["local_clip"]) if results["proposed"]["local_clip"] else 0
    mean_improved_lclip = np.mean(results["improved"]["local_clip"]) if results["improved"]["local_clip"] else 0

    print("\n" + "="*105)
    print(" 📊 EVALUATION RESULTS: ORIGINAL vs NAIVE vs PROPOSED vs IMPROVED")
    print("="*105)
    print(f"{'Metric':<20} | {'Orig (Baseline)':<18} | {'Naive (Global)':<16} | {'Proposed (Masked)':<18} | {'Improved (Comp.)':<18}")
    print("-" * 105)
    print(f"{'Bg SSIM (↑)':<20} | {mean_orig_ssim:^18.4f} | {mean_naive_ssim:^16.4f} | {mean_prop_ssim:^18.4f} | {mean_improved_ssim:^18.4f}")
    print(f"{'Global CLIP (↑)':<20} | {mean_orig_gclip:^18.2f} | {mean_naive_gclip:^16.2f} | {mean_prop_gclip:^18.2f} | {mean_improved_gclip:^18.2f}")
    print(f"{'Masked CLIP (↑)':<20} | {mean_orig_lclip:^18.2f} | {mean_naive_lclip:^16.2f} | {mean_prop_lclip:^18.2f} | {mean_improved_lclip:^18.2f}")
    print(f"{'FID Realism (↓)':<20} | {final_fid_orig:^18.2f} | {final_fid_naive:^16.2f} | {final_fid_prop:^18.2f} | {final_fid_improved:^18.2f}")
    print("="*105)

    # Save quantitative results to JSON
    report = {
        "metrics_explanation": {
            "Background SSIM": "Higher is better. Measures how well the unedited background was preserved. (Max 1.0)",
            "Global CLIP": "Higher is better. Measures full-image alignment to text. Orig baseline should be lowest (before edit).",
            "Masked CLIP": "Higher is better. Evaluates ONLY the targeted edit region. High scores prove localized precision.",
            "FID": "Lower is better. Measures overall realism compared to the original photos."
        },
        "original_images_baseline": {
            "background_ssim": float(mean_orig_ssim),
            "global_clip_score": float(mean_orig_gclip),
            "masked_clip_score": float(mean_orig_lclip),
            "fid_score": final_fid_orig
        },
        "naive_pipeline": {
            "background_ssim": float(mean_naive_ssim),
            "global_clip_score": float(mean_naive_gclip),
            "masked_clip_score": float(mean_naive_lclip),
            "fid_score": final_fid_naive
        },
        "proposed_pipeline": {
            "background_ssim": float(mean_prop_ssim),
            "global_clip_score": float(mean_prop_gclip),
            "masked_clip_score": float(mean_prop_lclip),
            "fid_score": final_fid_prop
        },
        "improved_pipeline": {
            "background_ssim": float(mean_improved_ssim),
            "global_clip_score": float(mean_improved_gclip),
            "masked_clip_score": float(mean_improved_lclip),
            "fid_score": final_fid_improved
        }
    }

    out_json = Path(OUTPUT_JSON_PATH)
    out_json.write_text(json.dumps(report, indent=4))
    print(f"\n✅ Detailed evaluation saved to {out_json.name}")

if __name__ == "__main__":
    main()