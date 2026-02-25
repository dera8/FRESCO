import os
import shutil
import json
from pathlib import Path

# =========================================================
# CONFIGURATION - UPDATE THESE PATHS IF NEEDED
# =========================================================

# The actual physical folders on your hard drive
OLD_MASKED_DIR = Path("data_output/masked_rgb_old")
NEW_MASKED_DIR = Path("data_output/masked_rgb")

OLD_SAM3_DIR = Path("data_output/sam3_instances_v2_old")
NEW_SAM3_DIR = Path("data_output/sam3_instances")

# The JSON files
OLD_JSON_PATH = Path("old_json_files/materials_full_filtered.json")
NEW_JSON_PATH = Path("materials_full_filtered.json")

# What prefixes to look for in instance folders and JSON keys
TARGET_PREFIXES = ("road_", "sidewalk_")

# String replacements for the JSON paths
# (e.g., changing "data_output/sam3_instances_v2/..." to "data_output/sam3_instances/...")
PATH_REPLACEMENTS = {
    "masked_rgb_v2": "masked_rgb",
    "sam3_instances_v2": "sam3_instances"
}

# =========================================================

def copy_target_folders(old_base: Path, new_base: Path):
    """Copies subfolders starting with target prefixes from old_base to new_base."""
    if not old_base.exists():
        print(f"⚠️ Directory not found: {old_base} (Skipping physical copy)")
        return
        
    print(f"📂 Scanning {old_base} for road/sidewalk instances...")
    copied_count = 0
    
    for img_folder in old_base.iterdir():
        if not img_folder.is_dir():
            continue
            
        # Look inside the image folder (e.g., masked_rgb_old/1003317145127470/)
        for inst_folder in img_folder.iterdir():
            if inst_folder.is_dir() and inst_folder.name.startswith(TARGET_PREFIXES):
                
                # Construct the destination path
                dest_folder = new_base / img_folder.name / inst_folder.name
                
                # Copy the folder and its contents
                shutil.copytree(inst_folder, dest_folder, dirs_exist_ok=True)
                copied_count += 1
                
    print(f"   ✅ Copied {copied_count} instance folders to {new_base}")


def merge_json_instances():
    """Extracts target instances from old JSON, updates their paths, and injects them into the new JSON."""
    if not OLD_JSON_PATH.exists() or not NEW_JSON_PATH.exists():
        print("⚠️ Missing one of the JSON files. Cannot merge data.")
        return

    print(f"\n📄 Merging JSON data from {OLD_JSON_PATH} -> {NEW_JSON_PATH}")
    
    with open(OLD_JSON_PATH, "r", encoding="utf-8") as f:
        old_data = json.load(f)
        
    with open(NEW_JSON_PATH, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    merged_count = 0

    old_images = old_data.get("data", {})
    new_images = new_data.get("data", {})

    for img_id, old_img_content in old_images.items():
        if img_id in new_images:
            old_instances = old_img_content.get("instances", {})
            
            # Ensure new JSON has an instances dictionary
            if "instances" not in new_images[img_id]:
                new_images[img_id]["instances"] = {}
                
            new_instances = new_images[img_id]["instances"]

            # Loop through old instances for this image
            for inst_id, inst_data in old_instances.items():
                if inst_id.startswith(TARGET_PREFIXES):
                    
                    # Hack to easily replace paths: convert to string, replace, convert back to dict
                    inst_str = json.dumps(inst_data)
                    for old_name, new_name in PATH_REPLACEMENTS.items():
                        inst_str = inst_str.replace(old_name, new_name)
                    updated_inst_data = json.loads(inst_str)

                    # Inject the updated instance into the new JSON dictionary
                    new_instances[inst_id] = updated_inst_data
                    merged_count += 1
                    
            new_images[img_id]["instances"] = new_instances

    # Save the updated new JSON
    with open(NEW_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        
    print(f"   ✅ Merged {merged_count} instance records into {NEW_JSON_PATH} with updated paths.")


def main():
    print("🚀 Starting Road/Sidewalk Migration...\n")
    
    # 1. Copy masked_rgb folders
    copy_target_folders(OLD_MASKED_DIR, NEW_MASKED_DIR)
    
    # 2. Copy sam3_instances folders
    copy_target_folders(OLD_SAM3_DIR, NEW_SAM3_DIR)
    
    # 3. Merge JSON entries and update paths
    merge_json_instances()
    
    print("\n🎉 Migration Complete!")

if __name__ == "__main__":
    main()