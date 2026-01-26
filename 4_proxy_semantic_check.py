# 4_proxy_semantic_check.py
# Step 4: Two-proxy semantic sanity check (ADE -> road/wall/roof super-classes) vs SAM3 structure-first masks
#
# Proxy A: SegFormer ADE20K
# Proxy B: OneFormer ADE20K
#
# Key fix vs earlier versions:
# - NO "or" with numpy arrays (ambiguous truth value)
# - Proxy WALL is a "facade/vertical-built" superclass:
#     wall = (wall ∪ building ∪ house ∪ storefront ∪ door/window/etc)
#   so we don't mistakenly compare SAM3 facade masks with ADE "indoor wall" only.
# - Optional per-building restriction for wall/roof agreement.
#
# Usage (Windows):
#   python 4_proxy_semantic_check.py ^
#     --manifest data_output/sam3_instances/manifest.json ^
#     --out_dir data_output/proxy_check ^
#     --save_overlays
#
# Notes:
# - This is a semantic proxy sanity check, not material GT.
# - Roof agreement may be weak in street-level imagery.

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Defaults: 2 proxies
# ----------------------------
PROXY_A_MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
PROXY_B_MODEL = "shi-labs/oneformer_ade20k_swin_large"


# ----------------------------
# Mask I/O + metrics
# ----------------------------
def load_mask_bool(p: str | Path) -> Optional[np.ndarray]:
    """Load a PNG mask and return boolean array. Returns None if missing."""
    if not p:
        return None
    pp = Path(p)
    if not pp.exists():
        return None
    m = np.array(Image.open(pp).convert("L"))
    return (m > 127)


def ensure_bool_mask(m: Optional[np.ndarray], H: int, W: int) -> np.ndarray:
    """Return mask if present else zeros(H,W). Also fixes shape mismatch if needed."""
    if m is None:
        return np.zeros((H, W), dtype=bool)
    if m.dtype != np.bool_:
        m = (m > 0)
    if m.shape != (H, W):
        # best-effort resize (nearest) if shape mismatch
        im = Image.fromarray((m.astype(np.uint8) * 255))
        im = im.resize((W, H), resample=Image.NEAREST)
        m = (np.array(im) > 127)
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def save_overlay_3class(
    rgb: Image.Image,
    road: np.ndarray,
    wall: np.ndarray,
    roof: np.ndarray,
    out_path: Path
):
    """
    Debug overlay (QC only):
      road = yellow
      wall = red
      roof = cyan
    """
    img = np.array(rgb).astype(np.float32)
    overlay = img.copy()

    # road yellow
    overlay[road] = 0.55 * overlay[road] + 0.45 * np.array([255, 255, 0], dtype=np.float32)
    # wall red
    overlay[wall] = 0.55 * overlay[wall] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
    # roof cyan
    overlay[roof] = 0.55 * overlay[roof] + 0.45 * np.array([0, 255, 255], dtype=np.float32)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)


# ----------------------------
# Label mapping (ADE -> superclasses)
# ----------------------------
def normalize_label(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ")


def build_keyword_mapper(id2label: Dict[int, str]) -> Tuple[List[int], List[int], List[int]]:
    """
    Build permissive keyword-based mapping from ADE labels to:
      - road-like
      - wall/facade-like (vertical built structures + parts)
      - roof-like
    Priority: roof > wall > road (exclusivity enforced later).
    """
    # Road-like: include ground-like + pedestrian surfaces.
    road_kw = {
        "road", "street", "sidewalk", "pavement", "path", "walkway", "runway",
        "ground", "floor", "dirt", "earth", "sand", "field", "playing field",
        "parking", "driveway", "railroad track", "cobblestone", "asphalt"
    }

    # Roof-like: (street-level often weak, but keep)
    roof_kw = {"roof", "awning", "canopy"}

    # Wall/facade-like: IMPORTANT (facades often labeled as building/house in ADE)
    wall_kw = {
        "wall",
        "building", "house", "skyscraper", "tower", "apartment", "hotel",
        "shop", "storefront", "booth",
        "facade",
        "door", "window", "shutter",
        "balcony", "railing", "fence",
        "column", "pillar", "beam", "arch", "porch", "stair", "stairway",
        "sign", "signboard", "banner"
    }

    road_ids: List[int] = []
    wall_ids: List[int] = []
    roof_ids: List[int] = []

    for k, v in id2label.items():
        name = normalize_label(v)

        # roof first
        if any(kw in name for kw in roof_kw):
            roof_ids.append(int(k))
            continue

        # wall next (we want facades to fall here, not into road)
        if any(kw in name for kw in wall_kw):
            wall_ids.append(int(k))
            continue

        # road last
        if any(kw in name for kw in road_kw):
            road_ids.append(int(k))
            continue

    return road_ids, wall_ids, roof_ids


def argmax_to_3masks(
    pred_ids: np.ndarray,
    road_ids: List[int],
    wall_ids: List[int],
    roof_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert argmax label map -> 3 boolean masks, exclusive (roof > wall > road)."""
    road = np.isin(pred_ids, road_ids)
    wall = np.isin(pred_ids, wall_ids)
    roof = np.isin(pred_ids, roof_ids)

    # Exclusivity: roof > wall > road
    wall = np.logical_and(wall, ~roof)
    road = np.logical_and(road, ~(roof | wall))
    return road, wall, roof


# ----------------------------
# Proxy inference
# ----------------------------
def load_segformer(model_id: str, device: str):
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    proc = SegformerImageProcessor.from_pretrained(model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def infer_segformer(rgb: Image.Image, proc, model, device: str) -> np.ndarray:
    inputs = proc(images=rgb, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits  # [1, C, h, w]
    up = torch.nn.functional.interpolate(
        logits, size=rgb.size[::-1], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)
    return pred


def load_oneformer(model_id: str, device: str):
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    proc = OneFormerProcessor.from_pretrained(model_id)
    model = OneFormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def infer_oneformer(rgb: Image.Image, proc, model, device: str) -> np.ndarray:
    inputs = proc(images=rgb, task_inputs=["semantic"], return_tensors="pt").to(device)
    out = model(**inputs)
    sem = proc.post_process_semantic_segmentation(out, target_sizes=[rgb.size[::-1]])[0]
    pred = sem.detach().cpu().numpy().astype(np.int32)
    return pred


# ----------------------------
# Manifest loading
# ----------------------------
def load_manifest(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "items" in raw and isinstance(raw["items"], list):
        return raw["items"]
    raise ValueError("Manifest must be a list or a dict with key 'items'.")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Step 4: Two-proxy semantic sanity check (ADE->road/wall/roof super-classes) vs SAM3 structure-first masks."
    )
    ap.add_argument("--manifest", required=True, help="SAM3 manifest.json (data_output/sam3_instances/manifest.json)")
    ap.add_argument("--out_dir", default="data_output/proxy_check", help="Output dir for CSV + overlays")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--proxyA", default=PROXY_A_MODEL, help="Proxy A model id (SegFormer ADE)")
    ap.add_argument("--proxyB", default=PROXY_B_MODEL, help="Proxy B model id (OneFormer ADE)")
    ap.add_argument("--save_overlays", action="store_true", help="Save debug overlays (3-class)")
    ap.add_argument("--per_building", action="store_true", help="Also compute per-building IoU (if instances present)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest))

    print(f"🔌 device={device}")
    print(f"🅰️ Proxy A (SegFormer): {args.proxyA}")
    print(f"🅱️ Proxy B (OneFormer): {args.proxyB}")

    # Load models
    seg_proc, seg_model = load_segformer(args.proxyA, device)
    one_proc, one_model = load_oneformer(args.proxyB, device)

    # Build mappers from each model's id2label
    seg_id2label = {int(k): v for k, v in seg_model.config.id2label.items()}
    one_id2label = {int(k): v for k, v in one_model.config.id2label.items()}

    seg_road_ids, seg_wall_ids, seg_roof_ids = build_keyword_mapper(seg_id2label)
    one_road_ids, one_wall_ids, one_roof_ids = build_keyword_mapper(one_id2label)

    rows: List[dict] = []
    rows_buildings: List[dict] = []

    for item in manifest:
        rgb_path = item.get("rgb") or item.get("image")
        if not rgb_path:
            continue
        rgb_path = Path(rgb_path)
        if not rgb_path.exists():
            continue

        img_id = item.get("id", rgb_path.stem)
        rgb = Image.open(rgb_path).convert("RGB")
        W, H = rgb.size

        # Load SAM3 global masks
        g = item.get("global", {})
        sam_road = ensure_bool_mask(load_mask_bool(g.get("road_mask", "")), H, W)
        sam_wall = ensure_bool_mask(load_mask_bool(g.get("wall_mask", "")), H, W)
        sam_roof = ensure_bool_mask(load_mask_bool(g.get("roof_mask", "")), H, W)

        # Proxy inference
        predA = infer_segformer(rgb, seg_proc, seg_model, device)
        predB = infer_oneformer(rgb, one_proc, one_model, device)

        A_road, A_wall, A_roof = argmax_to_3masks(predA, seg_road_ids, seg_wall_ids, seg_roof_ids)
        B_road, B_wall, B_roof = argmax_to_3masks(predB, one_road_ids, one_wall_ids, one_roof_ids)

        # Global IoU
        rows.append({
            "img_id": img_id,
            "rgb": str(rgb_path),
            "iouA_road": iou(A_road, sam_road),
            "iouA_wall": iou(A_wall, sam_wall),
            "iouA_roof": iou(A_roof, sam_roof),
            "iouB_road": iou(B_road, sam_road),
            "iouB_wall": iou(B_wall, sam_wall),
            "iouB_roof": iou(B_roof, sam_roof),
        })

        if args.save_overlays:
            save_overlay_3class(rgb, A_road, A_wall, A_roof, overlays_dir / f"{img_id}__proxyA.png")
            save_overlay_3class(rgb, B_road, B_wall, B_roof, overlays_dir / f"{img_id}__proxyB.png")
            save_overlay_3class(rgb, sam_road, sam_wall, sam_roof, overlays_dir / f"{img_id}__sam3.png")

        # Optional per-building IoU
        if args.per_building:
            for inst in item.get("instances", []):
                bid = inst.get("id", "building")
                bmask = load_mask_bool(inst.get("building_mask", ""))
                if bmask is None:
                    continue
                bmask = ensure_bool_mask(bmask, H, W)

                sam_w = ensure_bool_mask(load_mask_bool(inst.get("wall_mask", "")), H, W)
                sam_r = ensure_bool_mask(load_mask_bool(inst.get("roof_mask", "")), H, W)

                A_w = np.logical_and(A_wall, bmask)
                A_r = np.logical_and(A_roof, bmask)
                B_w = np.logical_and(B_wall, bmask)
                B_r = np.logical_and(B_roof, bmask)

                rows_buildings.append({
                    "img_id": img_id,
                    "building_id": bid,
                    "iouA_wall_in_building": iou(A_w, sam_w),
                    "iouA_roof_in_building": iou(A_r, sam_r),
                    "iouB_wall_in_building": iou(B_w, sam_w),
                    "iouB_roof_in_building": iou(B_r, sam_r),
                })

    # Write CSVs
    out_csv = out_dir / "proxy_global_iou.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        out_csv.write_text("img_id,rgb,iouA_road,iouA_wall,iouA_roof,iouB_road,iouB_wall,iouB_roof\n", encoding="utf-8")

    out_csv_b = out_dir / "proxy_per_building_iou.csv"
    if rows_buildings:
        with out_csv_b.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_buildings[0].keys()))
            w.writeheader()
            for r in rows_buildings:
                w.writerow(r)
    else:
        out_csv_b.write_text(
            "img_id,building_id,iouA_wall_in_building,iouA_roof_in_building,iouB_wall_in_building,iouB_roof_in_building\n",
            encoding="utf-8"
        )

    print("✅ Done")
    print(f"📄 {out_csv}")
    print(f"📄 {out_csv_b}")
    if args.save_overlays:
        print(f"🖼️ overlays: {overlays_dir}")


if __name__ == "__main__":
    main()
