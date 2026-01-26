import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore")

from transformers import Sam3Processor, Sam3Model


# -------------------------------------------------
# DEFAULT CONFIG (can be overridden via CLI)
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_THR = 0.6
MASK_THR  = 0.5

PROMPTS = {
    "building": ["building"],
    "wall": ["building facade", "wall", "building front"],
    "roof": ["building roof", "roof"],
}

ROAD_PROMPTS_MAIN = [
    "road", "street", "road surface", "asphalt road", "driveway",
    "sidewalk", "pavement", "footpath", "walkway", "pedestrian walkway"
]

ROAD_PROMPTS_FALLBACK = [
    "ground", "floor", "concrete ground", "stone ground"
]

ROAD_FALLBACK_MIN_COV_PCT = 1.0  # if road coverage < 1% => add fallback prompts

SAVE_ROAD_DEBUG_PER_PROMPT = True


# -------------------------------------------------
# UTILS
# -------------------------------------------------

def save_mask(mask_u8: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8).save(path)

def to_u8(mask_bool_or_01: np.ndarray) -> np.ndarray:
    return (mask_bool_or_01 > 0).astype(np.uint8) * 255

def intersect_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    return (np.logical_and(a_u8 > 0, b_u8 > 0).astype(np.uint8) * 255)

def coverage_pct(mask_u8: np.ndarray) -> float:
    if mask_u8.size == 0:
        return 0.0
    return float((mask_u8 > 0).mean() * 100.0)

def cast_inputs_to_model_dtype(inputs: dict, model: torch.nn.Module) -> dict:
    """Avoid FloatTensor vs HalfTensor mismatch."""
    target_dtype = next(model.parameters()).dtype
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype=target_dtype)
        else:
            out[k] = v
    return out

def union_masks_uint8(masks_t: torch.Tensor, H: int, W: int) -> np.ndarray:
    """OR across instances -> single u8 mask (0/255)."""
    if masks_t is None or len(masks_t) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    m = masks_t.detach().cpu().numpy()
    if m.ndim == 2:
        m = m[None, ...]
    return to_u8(m.any(axis=0))

def run_sam3_instance(img: Image.Image, text_prompt: str, processor, model, device: str):
    """
    Returns dict:
    - masks: torch.Tensor [N,H,W] (0/1)
    - scores: torch.Tensor [N]
    """
    inputs = processor(images=img, text=text_prompt, return_tensors="pt").to(device)
    inputs = cast_inputs_to_model_dtype(inputs, model)

    with torch.no_grad():
        outputs = model(**inputs)

    res = processor.post_process_instance_segmentation(
        outputs,
        threshold=SCORE_THR,
        mask_threshold=MASK_THR,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]
    return res

def union_from_prompts(
    img: Image.Image,
    prompts: list[str],
    H: int,
    W: int,
    processor,
    model,
    device: str,
    debug_dir: Path | None = None
):
    """Run SAM3 on multiple prompts and OR all masks. Optionally saves per-prompt masks."""
    acc = np.zeros((H, W), dtype=bool)
    per_prompt = {}

    for p in prompts:
        r = run_sam3_instance(img, p, processor, model, device)
        m_u8 = union_masks_uint8(r.get("masks", None), H, W)
        acc |= (m_u8 > 0)

        if debug_dir is not None:
            safe = "".join([c if c.isalnum() else "_" for c in p])[:80]
            out = debug_dir / f"road_prompt__{safe}.png"
            save_mask(m_u8, out)
            per_prompt[p] = str(out)

    return to_u8(acc), per_prompt


# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="SAM3 structural + building instance masks from a folder of images.")
    ap.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Folder containing frames (png/jpg). Example: C:\\Users\\Debora\\Desktop\\bryggenn\\v3\\data_input\\images"
    )
    ap.add_argument("--out-dir", type=str, default="data_output/sam3_instances", help="Output directory.")
    ap.add_argument("--pattern", type=str, default="*.png", help="Glob pattern: '*.png' or '*.jpg' etc.")
    ap.add_argument("--recursive", action="store_true", help="Search images recursively.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N images.")
    ap.add_argument("--score-thr", type=float, default=SCORE_THR, help="SAM3 score threshold.")
    ap.add_argument("--mask-thr", type=float, default=MASK_THR, help="SAM3 mask threshold.")
    ap.add_argument("--no-road-debug", action="store_true", help="Disable saving road per-prompt debug masks.")
    return ap.parse_args()


def main():
    global SCORE_THR, MASK_THR, SAVE_ROAD_DEBUG_PER_PROMPT

    args = parse_args()
    SCORE_THR = float(args.score_thr)
    MASK_THR = float(args.mask_thr)
    SAVE_ROAD_DEBUG_PER_PROMPT = (not args.no_road_debug)

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"images-dir not found: {images_dir}")

    # Collect images
    if args.recursive:
        image_paths = sorted(images_dir.rglob(args.pattern))
    else:
        image_paths = sorted(images_dir.glob(args.pattern))

    # If user has mixed formats, a common pattern is "*" and filter suffixes.
    # But we keep it simple and configurable via --pattern.
    image_paths = [p for p in image_paths if p.is_file()]

    if args.limit and args.limit > 0:
        image_paths = image_paths[:args.limit]

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found in {images_dir} with pattern '{args.pattern}'. "
            f"Try --pattern \"*.jpg\" or --pattern \"*.*\" --recursive"
        )

    print(f"🖼️  Found {len(image_paths)} images in: {images_dir}")
    print("🔄 Loading SAM3...")

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()

    print(f"✅ SAM3 ready | device={DEVICE} | dtype={next(model.parameters()).dtype}")
    print(f"⚙️  score_thr={SCORE_THR} | mask_thr={MASK_THR} | road_debug={SAVE_ROAD_DEBUG_PER_PROMPT}")

    manifest_out = []

    for img_path in image_paths:
        print(f"\n📸 Processing {img_path.name}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Skipping (cannot open): {img_path} | {e}")
            continue

        W, H = img.size  # PIL: (W,H)

        img_dir = out_dir / img_path.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "image": str(img_path),
            "global": {},
            "instances": []
        }

        # ---------- GLOBAL STRUCTURAL MASKS: WALL / ROOF ----------
        structural_masks = {}

        for cls in ["wall", "roof"]:
            prompts = PROMPTS[cls]
            m_u8, _ = union_from_prompts(
                img, prompts, H, W,
                processor=processor, model=model, device=DEVICE,
                debug_dir=None
            )
            structural_masks[cls] = m_u8

            out_path = img_dir / f"{cls}.png"
            save_mask(m_u8, out_path)

            entry["global"][f"{cls}_mask"] = str(out_path)
            entry["global"][f"{cls}_coverage_pct"] = round(coverage_pct(m_u8), 3)
            entry["global"][f"{cls}_prompts"] = prompts

        # ---------- GLOBAL STRUCTURAL MASK: ROAD (+SIDEWALK) ----------
        road_debug_dir = None
        if SAVE_ROAD_DEBUG_PER_PROMPT:
            road_debug_dir = img_dir / "road_debug"
            road_debug_dir.mkdir(parents=True, exist_ok=True)

        road_u8, road_prompt_outputs = union_from_prompts(
            img, ROAD_PROMPTS_MAIN, H, W,
            processor=processor, model=model, device=DEVICE,
            debug_dir=road_debug_dir
        )
        road_cov = coverage_pct(road_u8)
        used_prompts = list(ROAD_PROMPTS_MAIN)

        if road_cov < ROAD_FALLBACK_MIN_COV_PCT and len(ROAD_PROMPTS_FALLBACK) > 0:
            road_u8_fb, fb_prompt_outputs = union_from_prompts(
                img, ROAD_PROMPTS_FALLBACK, H, W,
                processor=processor, model=model, device=DEVICE,
                debug_dir=road_debug_dir
            )
            road_u8 = to_u8((road_u8 > 0) | (road_u8_fb > 0))
            road_cov = coverage_pct(road_u8)
            used_prompts += list(ROAD_PROMPTS_FALLBACK)
            road_prompt_outputs.update(fb_prompt_outputs)

        structural_masks["road"] = road_u8

        road_path = img_dir / "road.png"
        save_mask(road_u8, road_path)

        entry["global"]["road_mask"] = str(road_path)
        entry["global"]["road_coverage_pct"] = round(road_cov, 3)
        entry["global"]["road_prompts_used"] = used_prompts
        if SAVE_ROAD_DEBUG_PER_PROMPT:
            entry["global"]["road_prompt_masks"] = road_prompt_outputs

        # ---------- BUILDING INSTANCES ----------
        building_prompt = PROMPTS["building"][0]
        building_res = run_sam3_instance(img, building_prompt, processor, model, DEVICE)

        building_masks = building_res.get("masks", None)    # [N,H,W]
        building_scores = building_res.get("scores", None)  # [N]

        n_buildings = 0 if building_masks is None else len(building_masks)
        print(f"🏢 Found {n_buildings} buildings")

        if building_masks is not None and n_buildings > 0:
            for i in range(n_buildings):
                bmask_u8 = to_u8(building_masks[i].detach().cpu().numpy())

                bid = f"building_{i:02d}"
                bdir = img_dir / bid
                bdir.mkdir(parents=True, exist_ok=True)

                building_path = bdir / "building.png"
                save_mask(bmask_u8, building_path)

                wall_mask = intersect_u8(bmask_u8, structural_masks["wall"])
                roof_mask = intersect_u8(bmask_u8, structural_masks["roof"])

                wall_path = bdir / "wall.png"
                roof_path = bdir / "roof.png"
                save_mask(wall_mask, wall_path)
                save_mask(roof_mask, roof_path)

                score_i = float(building_scores[i].detach().cpu().item()) if building_scores is not None else 0.0

                entry["instances"].append({
                    "id": bid,
                    "score": score_i,
                    "building_mask": str(building_path),
                    "wall_mask": str(wall_path),
                    "roof_mask": str(roof_path),
                    "wall_coverage_pct": round(coverage_pct(wall_mask), 3),
                    "roof_coverage_pct": round(coverage_pct(roof_mask), 3)
                })

        manifest_out.append(entry)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n🎉 DONE")
    print(f"📄 Manifest: {manifest_path}")
    print(f"🗂️  Output folder: {out_dir}")


if __name__ == "__main__":
    main()
