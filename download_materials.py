import os
import requests
import time
import io
from pathlib import Path
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
HEADERS = {"User-Agent": "AutoPBR-AcademicResearch/1.0"}
BASE_DIR = Path("material_bank")

TARGET_CATEGORIES = {
    "brick": "brick", "concrete": "concrete", "wood": "wood",
    "plaster": "plaster", "metal": "metal", "roofing": "shingles", 
    "asphalt": "asphalt", "stone": "stone", "paving": "stone", "tiles": "tile"
}

MAX_PER_CATEGORY = 15 
RESOLUTION = "1k"

def find_color_map_url(data, target_res="1k"):
    found_urls = []
    color_keywords = ["diffuse", "color", "albedo", "basecolor", "_col_", "_diff_"]
    
    def search_dict(d):
        if isinstance(d, dict):
            if "url" in d and isinstance(d["url"], str):
                url_lower = d["url"].lower()
                if url_lower.endswith((".jpg", ".png", ".jpeg")):
                    if any(kw in url_lower for kw in color_keywords) and "normal" not in url_lower and "rough" not in url_lower:
                        found_urls.append(d["url"])
            for v in d.values(): search_dict(v)
        elif isinstance(d, list):
            for item in d: search_dict(item)

    search_dict(data)
    if not found_urls: return None
        
    jpg_urls = [u for u in found_urls if u.lower().endswith((".jpg", ".jpeg"))]
    valid_pool = jpg_urls if jpg_urls else found_urls
    
    for url in valid_pool:
        if f"{target_res}" in url.lower(): return url
    for url in valid_pool:
        if "2k" in url.lower(): return url
            
    return valid_pool[0]

def main():
    BASE_DIR.mkdir(exist_ok=True)
    print("🌐 Fetching texture list from Poly Haven API...")
    try:
        all_assets = requests.get("https://api.polyhaven.com/assets?t=textures", headers=HEADERS).json()
    except Exception as e:
        print(f"❌ Failed to reach API: {e}")
        return

    categorized_assets = {val: [] for val in set(TARGET_CATEGORIES.values())}
    for asset_id, data in all_assets.items():
        for cat in data.get("categories", []):
            if cat in TARGET_CATEGORIES:
                categorized_assets[TARGET_CATEGORIES[cat]].append(asset_id)
                break 

    print(f"✅ Found {len(all_assets)} textures. Starting RGB download...\n")
    total_downloaded = 0

    for folder_name, asset_ids in categorized_assets.items():
        folder_path = BASE_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        
        assets_to_download = asset_ids[:MAX_PER_CATEGORY]
        if not assets_to_download: continue
            
        print(f"📂 Category: {folder_name.upper()}")
        for asset_id in assets_to_download:
            save_path = folder_path / f"{asset_id}.jpg"
            if save_path.exists(): continue 

            try:
                files_data = requests.get(f"https://api.polyhaven.com/files/{asset_id}", headers=HEADERS).json()
                img_url = find_color_map_url(files_data, RESOLUTION)
                if not img_url: continue

                # Download and save as pure RGB
                img_data = requests.get(img_url, headers=HEADERS).content
                img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_pil.save(save_path, "JPEG", quality=95)
                
                print(f"   ⬇️ Downloaded RGB: {asset_id}.jpg")
                total_downloaded += 1
                time.sleep(0.5) 
                
            except Exception as e:
                print(f"   ⚠️ Failed on {asset_id}: {e}")

    print(f"\n🎉 Done! Saved {total_downloaded} natural textures.")

if __name__ == "__main__":
    main()