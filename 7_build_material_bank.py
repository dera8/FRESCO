import json
import cv2
import numpy as np
from pathlib import Path

GT_PATH = "ground_truth_eval.json"
OUT_DIR = Path("material_bank")

def get_center_square(mask, size=256):
    """Finds the largest inscribed-ish square inside a binary mask."""
    # Find distance transform to get the point deepest inside the mask
    dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist)
    
    # maxLoc is the center of the largest inscribed circle
    cx, cy = maxLoc
    half = size // 2
    
    # Bounding box for the crop
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    return x1, y1, x2, y2, maxVal

def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f).get("data", {})

    # Track the "best" patch found for each material class
    # Format: {"brick": {"score": 0, "patch": numpy_array}, ...}
    best_patches = {}

    print("🧱 Building Material Reference Bank...")
    for img_id, img_data in gt_data.items():
        rgb_path = str(img_data.get("rgb", "")).replace('\\', '/')
        if not Path(rgb_path).exists(): continue
        
        img_np = cv2.imread(rgb_path)
        
        # Look through all instances
        for bid, inst in img_data.get("instances", {}).items():
            for struct in ["wall", "roof", "road", "sidewalk"]:
                if struct not in inst or inst[struct].get("skipped"): continue
                
                mat_class = inst[struct].get("material_descriptor", {}).get("class", "unknown")
                if mat_class == "unknown": continue
                
                mask_path = str(inst[struct].get("mask", "")).replace('\\', '/')
                if not Path(mask_path).exists(): continue
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                mask_bool = mask > 127
                
                # Get the safest 256x256 crop inside the mask
                x1, y1, x2, y2, depth = get_center_square(mask_bool, size=256)
                
                # If the crop goes outside the image bounds, skip
                h, w = img_np.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h: continue
                
                # Check if this crop is 100% inside the mask
                crop_mask = mask_bool[y1:y2, x1:x2]
                if crop_mask.mean() < 0.95: continue # Needs to be at least 95% pure material
                
                # Score is based on how "deep" inside the mask the center is + AI confidence
                conf = inst[struct]["material_descriptor"].get("confidence", 1.0)
                score = depth * conf
                
                if mat_class not in best_patches or score > best_patches[mat_class]["score"]:
                    crop_bgr = img_np[y1:y2, x1:x2]
                    best_patches[mat_class] = {"score": score, "patch": crop_bgr}

    for mat, data in best_patches.items():
        out_file = OUT_DIR / f"{mat}.jpg"
        cv2.imwrite(str(out_file), data["patch"])
        print(f"  ✅ Saved {mat}.jpg (Score: {data['score']:.2f})")

if __name__ == "__main__":
    main()