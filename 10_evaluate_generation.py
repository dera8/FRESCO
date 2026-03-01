#!/usr/bin/env python3
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import warnings

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
    """Fixes paths that were saved relative to the zest_code subfolder."""
    if path_str.startswith("../"):
        return Path(path_str[3:])
    return Path(path_str)

def main():
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
    fid_naive = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid_proposed = FrechetInceptionDistance(feature=2048).to(DEVICE)
    fid_improved = FrechetInceptionDistance(feature=2048).to(DEVICE)

    # Score trackers
    results = {
        "naive": {"bg_ssim": [], "clip_score": []},
        "proposed": {"bg_ssim": [], "clip_score": []},
        "improved": {"bg_ssim": [], "clip_score": []}
    }

    print("\n🚀 Starting Evaluation...")
    
    for img_id, entry in tqdm(benchmark_data.items(), desc="Evaluating Images"):
        # Resolve Paths
        orig_path = resolve_path(entry["original_image"])
        naive_path = resolve_path(entry["naive_image"])
        prop_path = resolve_path(entry["proposed_image"])
        improved_path = resolve_path(entry["improved_image"])
        mask_path = resolve_path(entry["mask_path"])
        prompt = entry["zest_prompt"]

        if not all([orig_path.exists(), naive_path.exists(), prop_path.exists(), improved_path.exists(), mask_path.exists()]):
            print(f"\n⚠️ Missing files for {img_id}, skipping...")
            continue

        # ---------------------------------------------------------
        # 1. FID Score
        # ---------------------------------------------------------
        orig_tensor = load_image_tensor(orig_path).to(DEVICE)
        naive_tensor = load_image_tensor(naive_path).to(DEVICE)
        prop_tensor = load_image_tensor(prop_path).to(DEVICE)
        improved_tensor = load_image_tensor(improved_path).to(DEVICE)

        # Update real images
        fid_naive.update(orig_tensor, real=True)
        fid_proposed.update(orig_tensor, real=True)
        fid_improved.update(orig_tensor, real=True)

        # Update fake images
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
        _, naive_ssim_map = ssim(orig_gray, naive_gray, full=True, data_range=255)
        _, prop_ssim_map = ssim(orig_gray, prop_gray, full=True, data_range=255)
        _, improved_ssim_map = ssim(orig_gray, improved_gray, full=True, data_range=255)

        # Extract SSIM only for the background
        if np.any(bg_mask):
            results["naive"]["bg_ssim"].append(naive_ssim_map[bg_mask].mean())
            results["proposed"]["bg_ssim"].append(prop_ssim_map[bg_mask].mean())
            results["improved"]["bg_ssim"].append(improved_ssim_map[bg_mask].mean())

        # ---------------------------------------------------------
        # 3. CLIP Score (Text-to-Image Alignment)
        # ---------------------------------------------------------
        naive_img = Image.open(naive_path).convert("RGB")
        prop_img = Image.open(prop_path).convert("RGB")
        improved_img = Image.open(improved_path).convert("RGB")

        with torch.no_grad():
            # Naive CLIP
            inputs_n = clip_processor(text=[prompt], images=naive_img, return_tensors="pt", padding=True).to(DEVICE)
            results["naive"]["clip_score"].append(clip_model(**inputs_n).logits_per_image.item())
            
            # Proposed CLIP
            inputs_p = clip_processor(text=[prompt], images=prop_img, return_tensors="pt", padding=True).to(DEVICE)
            results["proposed"]["clip_score"].append(clip_model(**inputs_p).logits_per_image.item())

            # Improved CLIP
            inputs_i = clip_processor(text=[prompt], images=improved_img, return_tensors="pt", padding=True).to(DEVICE)
            results["improved"]["clip_score"].append(clip_model(**inputs_i).logits_per_image.item())

    # ==========================================
    # AGGREGATE AND PRINT RESULTS
    # ==========================================
    print("\n⏳ Computing final FID scores (this takes a few seconds)...")
    try:
        final_fid_naive = float(fid_naive.compute().item())
        final_fid_prop = float(fid_proposed.compute().item())
        final_fid_improved = float(fid_improved.compute().item())
    except ValueError as e:
        print(f"\n⚠️ FID Warning: {e} (Need more generated images for a reliable statistical FID calculation).")
        final_fid_naive = final_fid_prop = final_fid_improved = float('nan')

    mean_naive_ssim = np.mean(results["naive"]["bg_ssim"]) if results["naive"]["bg_ssim"] else 0
    mean_prop_ssim = np.mean(results["proposed"]["bg_ssim"]) if results["proposed"]["bg_ssim"] else 0
    mean_improved_ssim = np.mean(results["improved"]["bg_ssim"]) if results["improved"]["bg_ssim"] else 0

    mean_naive_clip = np.mean(results["naive"]["clip_score"]) if results["naive"]["clip_score"] else 0
    mean_prop_clip = np.mean(results["proposed"]["clip_score"]) if results["proposed"]["clip_score"] else 0
    mean_improved_clip = np.mean(results["improved"]["clip_score"]) if results["improved"]["clip_score"] else 0

    print("\n" + "="*80)
    print(" 📊 EVALUATION RESULTS: NAIVE vs PROPOSED vs IMPROVED")
    print("="*80)
    print(f"{'Metric':<20} | {'Naive (Global)':<16} | {'Proposed (Masked)':<18} | {'Improved (Composited)':<22}")
    print("-" * 80)
    print(f"{'Bg SSIM (↑)':<20} | {mean_naive_ssim:^16.4f} | {mean_prop_ssim:^18.4f} | {mean_improved_ssim:^22.4f}")
    print(f"{'CLIP Alignment (↑)':<20} | {mean_naive_clip:^16.2f} | {mean_prop_clip:^18.2f} | {mean_improved_clip:^22.2f}")
    print(f"{'FID Realism (↓)':<20} | {final_fid_naive:^16.2f} | {final_fid_prop:^18.2f} | {final_fid_improved:^22.2f}")
    print("="*80)

    # Save quantitative results to JSON
    report = {
        "metrics_explanation": {
            "Background SSIM": "Higher is better. Measures how well the unedited background was preserved. (Max 1.0)",
            "CLIP Alignment": "Higher is better. Measures how well the final image matches the requested material text prompt.",
            "FID": "Lower is better. Measures overall realism compared to the original photos."
        },
        "naive_pipeline": {
            "background_ssim": float(mean_naive_ssim),
            "clip_score": float(mean_naive_clip),
            "fid_score": final_fid_naive
        },
        "proposed_pipeline": {
            "background_ssim": float(mean_prop_ssim),
            "clip_score": float(mean_prop_clip),
            "fid_score": final_fid_prop
        },
        "improved_pipeline": {
            "background_ssim": float(mean_improved_ssim),
            "clip_score": float(mean_improved_clip),
            "fid_score": final_fid_improved
        }
    }

    out_json = Path("generation_evaluation_report.json")
    out_json.write_text(json.dumps(report, indent=4))
    print(f"\n✅ Detailed evaluation saved to {out_json.name}")

if __name__ == "__main__":
    main()