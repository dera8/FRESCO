import json
from pathlib import Path

# Paths
INPUT_JSON = "materials_full_filtered_old.json"
OUTPUT_JSON = "materials_full_filtered.json"  # Saves to a new file to be safe!

def main():
    print("⏳ Loading JSON...")
    if not Path(INPUT_JSON).exists():
        print(f"❌ Error: Could not find {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    converted_count = 0

    # Iterate through every image in the data
    for img_id, img_data in json_data.get("data", {}).items():
        instances = img_data.get("instances", {})
        if not instances:
            continue

        new_instances = {}
        
        # 1. Find the highest existing road index to prevent overwriting
        max_road_idx = -1
        for key in instances.keys():
            if key.startswith("road_"):
                try:
                    idx = int(key.split("_")[1])
                    if idx > max_road_idx:
                        max_road_idx = idx
                except ValueError:
                    pass
        
        # 2. Process all instances
        for key, inst_data in instances.items():
            if key.startswith("sidewalk_"):
                # Increment our road counter for the new name
                max_road_idx += 1
                new_key = f"road_{max_road_idx:03d}"
                
                # Update the internal 'type' field if it exists
                if inst_data.get("type") == "sidewalk":
                    inst_data["type"] = "road"
                
                # Notice we do NOT touch inst_data["mask"] or inst_data["masked_rgb"]
                
                # Save under the new road_XXX key
                new_instances[new_key] = inst_data
                converted_count += 1
            else:
                # Keep everything else (buildings, existing roads) exactly the same
                new_instances[key] = inst_data
                
        # Overwrite the old instances dictionary with the newly mapped one
        img_data["instances"] = new_instances

    # Add a note to the meta section
    if "meta" in json_data:
        json_data["meta"]["note"] = json_data["meta"].get("note", "") + " | Sidewalks converted to Roads"

    # Save the result
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! Converted {converted_count} 'sidewalk' instances into 'road' instances.")
    print(f"💾 Saved cleanly as: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()