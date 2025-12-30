import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

# IMPORTANTE: Devi installare transformers dalla versione più recente
# pip install git+https://github.com/huggingface/transformers.git

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("❌ Sam3Processor not found!")
    print("\n💡 SAM3 è disponibile solo nel branch main di transformers.")
    print("   Installa con:")
    print("   pip install git+https://github.com/huggingface/transformers.git")
    exit(1)

IMAGES_DIR = Path("photos")
OUT_DIR = Path("sam3_semantic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt semplici come nell'app web
PROMPTS = {
    "road": "road",
    "roof": "roof",
    "wall": "wall",
}

SCORE_THR = 0.5  # detection threshold (come default app web)
MASK_THR = 0.5   # mask threshold (come default app web)

def combine_masks(masks: torch.Tensor, scores: torch.Tensor) -> np.ndarray:
    """
    Combina tutte le mask in una singola mask binaria.
    Esattamente come fa l'app web.
    """
    if len(masks) == 0:
        return None
    
    # masks è [N, H, W], già filtrate da post_process_instance_segmentation
    # Unisci tutte con OR logico
    combined = masks.cpu().numpy().any(axis=0).astype(np.uint8)
    return (combined * 255).astype(np.uint8)

# ----------------------- Load SAM3 -----------------------
print("🔄 Loading SAM3 model...")
print("⚠️  This requires transformers from GitHub main branch")
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

try:
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print("\n💡 Make sure you have:")
    print("   1. Requested access at https://huggingface.co/facebook/sam3")
    print("   2. Run: huggingface-cli login")
    print("   3. Installed: pip install git+https://github.com/huggingface/transformers.git")
    exit(1)

manifest = []
total_images = len(list(IMAGES_DIR.glob("*.jpg")))

for img_idx, img_path in enumerate(sorted(IMAGES_DIR.glob("*.jpg")), 1):
    print(f"\n{'='*60}")
    print(f"📸 [{img_idx}/{total_images}] {img_path.name}")
    print(f"{'='*60}")
    
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    print(f"📐 Size: {W}x{H}")
    
    entry = {"image": str(img_path), "classes": {}}
    
    for cls_name, text_prompt in PROMPTS.items():
        print(f"\n  🔍 {cls_name.upper()}: '{text_prompt}'")
        
        try:
            # ESATTAMENTE come nell'app web - riga per riga
            inputs = processor(
                images=img, 
                text=text_prompt.strip(), 
                return_tensors="pt"
            ).to(device)
            
            # Converti dtype se necessario
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(model.dtype)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-processing - ESATTAMENTE come nell'app web
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=SCORE_THR,
                mask_threshold=MASK_THR,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            # Estrai risultati
            masks = results['masks']    # [N, H, W]
            scores = results['scores']  # [N]
            
            n_masks = len(masks)
            print(f"  ✅ Found {n_masks} instances")
            
            if n_masks > 0:
                scores_list = scores.cpu().numpy().tolist()
                scores_text = ", ".join([f"{s:.2f}" for s in scores_list[:5]])
                print(f"  📊 Scores: {scores_text}{'...' if n_masks > 5 else ''}")
                
                # Combina tutte le mask
                combined_mask = combine_masks(masks, scores)
                
                if combined_mask is not None:
                    coverage = (combined_mask > 0).sum() / (H * W) * 100
                    print(f"  📊 Coverage: {coverage:.1f}%")
                else:
                    combined_mask = np.zeros((H, W), dtype=np.uint8)
            else:
                print(f"  ⚠️  No objects found (try adjusting thresholds)")
                scores_list = []
                combined_mask = np.zeros((H, W), dtype=np.uint8)
            
            # Salva
            out_png = OUT_DIR / f"{img_path.stem}_{cls_name}.png"
            Image.fromarray(combined_mask).save(out_png)
            
            entry["classes"][cls_name] = {
                "prompt": text_prompt,
                "scores": scores_list,
                "num_instances": n_masks,
                "output_file": str(out_png.name)
            }
            
            print(f"  💾 Saved: {out_png.name}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            entry["classes"][cls_name] = {
                "prompt": text_prompt,
                "error": str(e)
            }
    
    manifest.append(entry)

# Salva manifest
manifest_path = OUT_DIR / "manifest.json"
with open(manifest_path, "w", encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"🎉 DONE!")
print(f"{'='*60}")
print(f"📁 Output: {OUT_DIR}")
print(f"📄 Manifest: {manifest_path}")
print(f"📊 Processed {total_images} images, {len(PROMPTS)} classes each")