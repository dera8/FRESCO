from pathlib import Path

# Adjust this if your folder has a different name
MASKED_RGB_DIR = Path("data_output/masked_rgb")

def main():
    print(f"📂 Scanning {MASKED_RGB_DIR} for old folder names...")
    
    if not MASKED_RGB_DIR.exists():
        print(f"❌ Error: Could not find directory '{MASKED_RGB_DIR}'")
        return

    renamed_count = 0

    # Iterate through every image ID folder
    for img_dir in MASKED_RGB_DIR.iterdir():
        if not img_dir.is_dir():
            continue
            
        # Iterate through the instance folders inside (building_00, road_000, etc.)
        for inst_dir in img_dir.iterdir():
            if not inst_dir.is_dir():
                continue
                
            name = inst_dir.name
            
            # Check if it's a building folder
            if name.startswith("building_"):
                parts = name.split("_")
                if len(parts) == 2:
                    num_str = parts[1]
                    
                    # If the number has fewer than 3 digits (e.g., '00', '05')
                    if len(num_str) < 3 and num_str.isdigit():
                        # Create the new 3-digit name
                        new_name = f"building_{int(num_str):03d}"
                        new_path = img_dir / new_name
                        
                        # Rename the folder on disk
                        if not new_path.exists():
                            inst_dir.rename(new_path)
                            renamed_count += 1
                            print(f"  🔄 Renamed: {img_dir.name}/{name}  ->  {new_name}")
                        else:
                            print(f"  ⚠️ Warning: Skipped {name} -> {new_name} (Target already exists)")

    print(f"\n✅ Done! Successfully upgraded {renamed_count} folders to the 3-digit format.")

if __name__ == "__main__":
    main()