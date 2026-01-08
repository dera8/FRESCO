import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
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


# --------------------------- CONFIG ---------------------------

# Usa il manifest prodotto dallo step1 (nuovo)
PHOTOS_MANIFEST = Path("data_input/photos_manifest.json")

# Output directory più esplicita (structure-first)
OUT_DIR = Path("data_input/sam3_structural_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompt base (puoi estenderli a multi-prompt se vuoi, vedi sotto)
PROMPTS = {
    "road": ["road", "street", "asphalt road"],
    "wall": ["building facade", "wall", "building front"],
    "roof": ["roof", "building roof"],
}

# Thresholds: rendiamo più conservative le maschere
SCORE_THR = 0.60   # prima era 0.5
MASK_THR  = 0.50

# Conservative post-processing
ERODE_K = 5        # MinFilter kernel (5 = erosione leggera)
MIN_AREA_PX = 500  # rimuove componenti piccole (sporcizia)

# Roof visibility heuristic (da usare dopo nelle metriche)
ROOF_MIN_COVERAGE_PCT = 1.0  # se <1% => probabilmente roof non visibile in street-view


# --------------------------- UTILS ---------------------------

def _to_uint8_mask(binary: np.ndarray) -> np.ndarray:
    return (binary.astype(np.uint8) * 255)

def _erode_mask_uint8(mask255: np.ndarray, k: int) -> np.ndarray:
    """Erosione leggera usando PIL (senza OpenCV)."""
    if k <= 1:
        return mask255
    im = Image.fromarray(mask255)
    # MinFilter = erosione su immagini binarie
    im = im.filter(ImageFilter.MinFilter(size=k))
    return np.array(im, dtype=np.uint8)

def _remove_small_components(mask255: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    Rimozione componenti piccole senza dipendenze esterne.
    Approccio semplice: bounding-box flood-fill su pixel ON.
    È O(N) e va bene per immagini moderate.
    """
    if min_area_px <= 0:
        return mask255

    H, W = mask255.shape
    mask = (mask255 > 0).astype(np.uint8)
    visited = np.zeros_like(mask, dtype=np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)

    # flood fill 4-neighborhood
    for y in range(H):
        for x in range(W):
            if mask[y, x] == 1 and visited[y, x] == 0:
                stack = [(y, x)]
                visited[y, x] = 1
                comp = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                        if 0 <= ny < H and 0 <= nx < W:
                            if mask[ny, nx] == 1 and visited[ny, nx] == 0:
                                visited[ny, nx] = 1
                                stack.append((ny, nx))
                                comp.append((ny, nx))
                if len(comp) >= min_area_px:
                    for (py, px) in comp:
                        out[py, px] = 1

    return _to_uint8_mask(out)

def _coverage_pct(mask255: np.ndarray) -> float:
    H, W = mask255.shape
    return float((mask255 > 0).sum() / (H * W) * 100.0)

def _infer_one_prompt(processor, model, device, img: Image.Image, prompt: str):
    inputs = processor(images=img, text=prompt.strip(), return_tensors="pt").to(device)

    # Convert dtype se necessario
    for key in inputs:
        if hasattr(inputs[key], "dtype") and inputs[key].dtype == torch.float32:
            inputs[key] = inputs[key].to(model.dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SCORE_THR,
        mask_threshold=MASK_THR,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]   # [N, H, W]
    scores = results["scores"] # [N]
    return masks, scores

def _combine_masks(masks: torch.Tensor) -> np.ndarray:
    """
    OR logico su tutte le mask (già filtrate dai threshold in post-process).
    """
    if masks is None or len(masks) == 0:
        return None
    combined = masks.cpu().numpy().any(axis=0).astype(np.uint8)
    return _to_uint8_mask(combined)


# --------------------------- LOAD MODEL ---------------------------

print("🔄 Loading SAM3 model...")
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


# --------------------------- LOAD PHOTOS MANIFEST ---------------------------

if not PHOTOS_MANIFEST.exists():
    print(f"❌ Missing {PHOTOS_MANIFEST}. Run step1_download_data.py first.")
    exit(1)

photos = json.loads(PHOTOS_MANIFEST.read_text(encoding="utf-8"))
print(f"📄 Loaded photos_manifest: {len(photos)} images")


# --------------------------- RUN ---------------------------

manifest_out = []
total_images = len(photos)

for idx, p in enumerate(photos, 1):
    img_path = Path(p["path"])
    if not img_path.exists():
        continue

    print(f"\n{'='*60}")
    print(f"📸 [{idx}/{total_images}] {img_path.name}")
    print(f"{'='*60}")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    print(f"📐 Size: {W}x{H} | projection={p.get('projection')} heading={p.get('heading')}")

    entry = {
        "photo_id": p.get("photo_id", img_path.stem),
        "image": str(img_path),
        "projection": p.get("projection"),
        "heading": p.get("heading"),
        "camera_prior": p.get("camera_prior", {}),
        "classes": {},
        "notes": "Structural masks extracted with SAM3; used as priors/conditioning (not material GT)."
    }

    for cls_name, prompt_list in PROMPTS.items():
        print(f"\n  🔍 {cls_name.upper()} prompts: {prompt_list}")

        try:
            # Multi-prompt: unione OR dei risultati
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            all_scores = []
            total_instances = 0

            for text_prompt in prompt_list:
                masks, scores = _infer_one_prompt(processor, model, device, img, text_prompt)

                n_masks = len(masks)
                total_instances += n_masks
                if n_masks > 0:
                    all_scores.extend(scores.cpu().numpy().tolist())
                    cm = _combine_masks(masks)
                    if cm is not None:
                        combined_mask = np.maximum(combined_mask, cm)

            # Post-processing conservativo
            if combined_mask is None:
                combined_mask = np.zeros((H, W), dtype=np.uint8)

            # 1) erosione leggera (precision > recall)
            combined_mask = _erode_mask_uint8(combined_mask, ERODE_K)

            # 2) rimozione componenti piccole
            combined_mask = _remove_small_components(combined_mask, MIN_AREA_PX)

            cov = _coverage_pct(combined_mask)
            print(f"  ✅ instances={total_instances} | coverage={cov:.2f}%")

            out_png = OUT_DIR / f"{img_path.stem}_{cls_name}.png"
            Image.fromarray(combined_mask).save(out_png)

            entry["classes"][cls_name] = {
                "prompts": prompt_list,
                "scores": all_scores[:50],  # evita json enormi
                "num_instances": int(total_instances),
                "coverage_pct": cov,
                "output_file": str(out_png),
            }

        except Exception as e:
            print(f"  ❌ Error on {cls_name}: {e}")
            entry["classes"][cls_name] = {"error": str(e)}

    # Roof visibility flag (utile dopo)
    roof_cov = entry["classes"].get("roof", {}).get("coverage_pct", 0.0) or 0.0
    entry["roof_visible"] = bool(roof_cov >= ROOF_MIN_COVERAGE_PCT)

    manifest_out.append(entry)

# Save manifest
manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'='*60}")
print("🎉 DONE!")
print(f"{'='*60}")
print(f"📁 Output: {OUT_DIR}")
print(f"📄 Manifest: {manifest_path}")
print(f"📊 Processed {len(manifest_out)} images, {len(PROMPTS)} classes each")
