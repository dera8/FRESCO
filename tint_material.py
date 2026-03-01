import os
import shutil
from pathlib import Path
from PIL import Image, ImageOps

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = Path("material_bank")
OUTPUT_DIR = Path("colored_material_bank")

COLOR_MAP = {
    "black": "#222222", "gray": "#808080", "white": "#E0E0E0", 
    "red": "#8B0000", "brown": "#654321", "yellow": "#DAA520", 
    "blue": "#4682B4", "green": "#556B2F", "beige": "#F5F5DC", "orange": "#CD853F"  
}

def main():
    if not INPUT_DIR.exists():
        print(f"❌ Input directory '{INPUT_DIR}' not found!")
        return
        
    OUTPUT_DIR.mkdir(exist_ok=True)
    materials = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    print(f"🎨 Generating color variations + preserving originals...\n")
    
    generated_count = 0
    
    for mat_folder in materials:
        material_name = mat_folder.name
        images = list(mat_folder.glob("*.jpg")) + list(mat_folder.glob("*.png"))
        if not images: continue
            
        print(f"🔄 Processing {material_name.upper()}...")
        
        # 1. Create the "Original" folder and copy the natural RGB textures
        original_folder = OUTPUT_DIR / f"original_{material_name}"
        original_folder.mkdir(exist_ok=True)
        
        for img_path in images:
            # Copy original
            shutil.copy(img_path, original_folder / img_path.name)
            
            # 2. Convert to Grayscale in-memory for tinting
            try:
                img_rgb = Image.open(img_path).convert("RGB")
                img_gray = img_rgb.convert("L")
                
                # Generate all color tints
                for color_name, hex_code in COLOR_MAP.items():
                    combo_folder = OUTPUT_DIR / f"{color_name}_{material_name}"
                    combo_folder.mkdir(exist_ok=True)
                    
                    save_path = combo_folder / img_path.name
                    if save_path.exists(): continue
                    
                    tinted_img = ImageOps.colorize(img_gray, black="black", mid=hex_code, white="white")
                    tinted_img.convert("RGB").save(save_path, "JPEG", quality=90)
                    generated_count += 1
            except Exception as e:
                print(f"   ⚠️ Failed on {img_path.name}: {e}")

    print(f"\n🎉 Done! Generated {generated_count} colorized variations and preserved originals in '{OUTPUT_DIR.name}/'.")

if __name__ == "__main__":
    main()