import json
import shutil
import copy
from pathlib import Path

# ==========================================
# CONFIGURATION PATHS
# ==========================================
# Folders
DIR_BASE = Path("data_output/masked_rgb")
DIR_ROADS = Path("data_output/masked_rgb_roads") 

# JSON Files
FILE_BASE_JSON = Path("materials_v2_filtered.json") 
FILE_ROADS_JSON = Path("materials_roads.json")      
FILE_OUT_JSON = Path("materials_full_filtered.json") 

def merge_folders():
    print("📂 Starting folder merge...")
    if not DIR_ROADS.exists():
        print(f"⚠️  Warning: Directory '{DIR_ROADS}' not found. Skipping folder merge.")
        return

    copied_folders = 0
    for img_dir_road in DIR_ROADS.iterdir():
        if not img_dir_road.is_dir(): continue
            
        img_id = img_dir_road.name
        img_dir_base = DIR_BASE / img_id
        img_dir_base.mkdir(parents=True, exist_ok=True)
        
        for item in img_dir_road.iterdir():
            target_path = img_dir_base / item.name
            if item.is_dir():
                if not target_path.exists():
                    shutil.copytree(item, target_path)
                    copied_folders += 1
            elif item.is_file():
                if not target_path.exists():
                    shutil.copy2(item, target_path)
                    
    print(f"✅ Folder merge complete! Copied {copied_folders} new instance folders.")

def clean_paths(node, old_bid=None):
    """Recursively fix Windows slashes, old folder names, and 2-digit building IDs."""
    if isinstance(node, dict):
        return {k: clean_paths(v, old_bid) for k, v in node.items()}
    elif isinstance(node, list):
        return [clean_paths(x, old_bid) for x in node]
    elif isinstance(node, str):
        # 1. Fix slashes
        s = node.replace('\\', '/')
        # 2. Upgrade root instance folder
        s = s.replace("sam3_instances/", "sam3_instances_v2/")
        
        # 3. Upgrade 2-digit to 3-digit building IDs in the path string itself
        if old_bid and old_bid.startswith("building_"):
            try:
                num = int(old_bid.split('_')[1])
                s = s.replace(old_bid, f"building_{num:03d}")
            except ValueError:
                pass
        return s
    else:
        return node

def merge_jsons():
    print("\n📄 Starting JSON merge and path upgrades...")
    
    if not FILE_BASE_JSON.exists() or not FILE_ROADS_JSON.exists():
        print("❌ Error: One or both JSON files are missing. Cannot merge.")
        return

    with open(FILE_BASE_JSON, 'r', encoding='utf-8') as f:
        base_json = json.load(f)
        
    with open(FILE_ROADS_JSON, 'r', encoding='utf-8') as f:
        roads_json = json.load(f)

    out_json = {"meta": base_json.get("meta", {}), "data": {}}
    out_json["meta"]["version"] = "v3.0_merged_buildings_and_roads"
    out_json["meta"]["note"] = "Merged and path-corrected"

    all_img_ids = set(base_json.get("data", {}).keys()).union(set(roads_json.get("data", {}).keys()))

    merged_count = 0
    for img_id in all_img_ids:
        b_data = base_json.get("data", {}).get(img_id, {})
        r_data = roads_json.get("data", {}).get(img_id, {})
        
        merged_data = copy.deepcopy(b_data)
        
        # --- 1. CLEAN AND UPGRADE BASE (BUILDING) DATA ---
        if "global" in merged_data:
            merged_data["global"] = clean_paths(merged_data["global"])
            
        if "instances" in merged_data:
            new_instances = {}
            for old_bid, inst_data in merged_data["instances"].items():
                # Upgrade Dictionary Keys (building_00 -> building_000)
                if old_bid.startswith("building_"):
                    try:
                        num = int(old_bid.split('_')[1])
                        new_bid = f"building_{num:03d}"
                    except ValueError:
                        new_bid = old_bid
                else:
                    new_bid = old_bid
                
                # Clean all paths inside this instance
                new_instances[new_bid] = clean_paths(inst_data, old_bid)
            merged_data["instances"] = new_instances

        # --- 2. MERGE ROADS DATA ---
        if "global" not in merged_data:
            merged_data["global"] = {}
        for key, val in r_data.get("global", {}).items():
            merged_data["global"][key] = clean_paths(val)
                
        if "instances" not in merged_data:
            merged_data["instances"] = {}
        for inst_id, inst_data in r_data.get("instances", {}).items():
            merged_data["instances"][inst_id] = clean_paths(inst_data)
            merged_count += 1
            
        out_json["data"][img_id] = merged_data

    with open(FILE_OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
        
    print(f"✅ JSON merge complete! Added {merged_count} new road/sidewalk instances.")
    print(f"💾 Saved cleanly as: {FILE_OUT_JSON}")

if __name__ == "__main__":
    merge_folders()
    merge_jsons()