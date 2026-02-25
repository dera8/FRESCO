#!/usr/bin/env python3
import os
import json
import base64
import argparse
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image
from openai import OpenAI


# ============================================================
# VOCABS (SOFT CANONICALIZATION)
# ============================================================
MATERIAL_VOCAB = {
    # road-ish
    "asphalt": "asphalt",
    "tarmac": "asphalt",
    "bitumen": "asphalt",
    "concrete": "concrete",
    "cement": "concrete",

    # masonry / stone / brick (valid for both roads and walls)
    "brick": "brick",
    "bricks": "brick",
    "paver": "brick",
    "pavers": "brick",
    "cobblestone": "stone",
    "cobblestones": "stone",
    "stone": "stone",
    "granite": "stone",

    # wall-ish
    "wood": "wood",
    "timber": "wood",
    "plaster": "plaster",
    "stucco": "plaster",
    "render": "plaster",
    "glass": "glass",
    "metal": "metal",
    "steel": "metal",
    "copper": "metal",
    "zinc": "metal",
    "paint": "painted_surface",
    "painted": "painted_surface",

    # roof-ish
    "tile": "tile",
    "tiles": "tile",
    "clay tile": "tile",
    "terracotta": "tile",
    "slate": "slate",
    "shingle": "shingles",
    "shingles": "shingles",
    "asphalt shingles": "shingles",
}

COLOR_VOCAB = {
    "black": "black",
    "grey": "gray",
    "gray": "gray",
    "white": "white",
    "red": "red",
    "brown": "brown",
    "yellow": "yellow",
    "blue": "blue",
    "green": "green",
    "beige": "beige",
    "tan": "beige",
    "orange": "orange",
}

SURFACE_VOCAB = {
    "smooth": "smooth",
    "flat": "smooth",
    "polished": "smooth",
    "rough": "rough",
    "textured": "rough",
    "grainy": "rough",
    "patterned": "rough",
    "striped": "rough",
    "checkered": "rough",
    "solid": "unknown",
    "none": "unknown",
}

AGING_VOCAB = {
    "new": "new",
    "well-maintained": "well_maintained",
    "well maintained": "well_maintained",
    "maintained": "well_maintained",
    "weathered": "weathered",
    "worn": "worn",
    "aged": "aged",
    "dirty": "dirty",
    "stained": "dirty",
    "damaged": "damaged",
}

CONF_TEXT_TO_NUM = {"high": 0.85, "medium": 0.55, "low": 0.25}

CLASSES_GLOBAL = ["road", "wall", "roof", "door", "window"]
CLASSES_BUILDING = ["wall", "roof", "door", "window"]


# ============================================================
# PROMPT
# ============================================================
PROMPT = """
You are analyzing ONLY the {class_name} region in an urban street-level photo.
The image is masked: ONLY the target region is visible, everything else is black.

TASK:
Return a SINGLE structured material descriptor.

Rules:
- JSON only. No extra text.
- confidence MUST be a number in [0, 1].
- If you are uncertain or the best label is not in common categories, use "other:<short label>".

OUTPUT JSON:
{{
  "class": "asphalt | concrete | brick | stone | wood | plaster | glass | metal | painted_surface | tile | slate | shingles | other:<...> | unknown",
  "color": "black|gray|white|red|brown|yellow|blue|green|beige|orange|other:<...>|unknown",
  "surface": "smooth|rough|other:<...>|unknown",
  "aging": "new|well_maintained|weathered|worn|aged|dirty|damaged|other:<...>|unknown",
  "confidence": 0.0
}}
""".strip()


# ============================================================
# Utils
# ============================================================
def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_from_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    return "application/octet-stream"


def parse_json_from_text(raw: str) -> Dict[str, Any]:
    """Robustly extract the first JSON object from a model response.

    Handles:
    - markdown fences (```json ... ```)
    - leading/trailing commentary
    - multiple JSON blocks (takes the first balanced {...})
    """
    if raw is None:
        raise ValueError("Empty response (None).")

    s = str(raw).strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)

    # Find first balanced JSON object
    start = s.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found. Raw: {s[:200]}")

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i+1].strip()
                return json.loads(candidate)

    raise ValueError(f"Unbalanced braces; could not extract JSON. Raw: {s[:200]}")



def _norm(x: str) -> str:
    return (x or "").strip().lower()


def _pick_first(v: Union[str, list, None]) -> Optional[str]:
    if isinstance(v, list) and v:
        return v[0]
    if isinstance(v, str):
        return v
    return None


def canonicalize_value(
    raw_value: Any,
    vocab: Dict[str, str],
    default: str = "unknown",
    mode: str = "soft",
    keep_other: bool = True,
    other_prefix: str = "other:",
) -> str:
    raw = _pick_first(raw_value)
    if raw is None:
        return default
    tok = _norm(raw)
    if not tok:
        return default

    if mode == "off":
        return tok

    if keep_other and tok.startswith(other_prefix):
        tail = tok[len(other_prefix):].strip().replace(" ", "_")
        return f"{other_prefix}{tail}" if tail else default

    if tok in vocab:
        return vocab[tok]

    if tok in set(vocab.values()):
        return tok

    for k, v in vocab.items():
        if k in tok:
            return v

    if mode == "hard":
        return default

    if keep_other:
        short = tok.split(",")[0].split(".")[0]
        short = short.replace(" ", "_")
        return f"{other_prefix}{short[:32]}" if short else default

    return default


def canonicalize_confidence(raw_conf: Any) -> float:
    if isinstance(raw_conf, (int, float)):
        c = float(raw_conf)
        return max(0.0, min(1.0, c))
    if isinstance(raw_conf, str):
        t = _norm(raw_conf)
        if t in CONF_TEXT_TO_NUM:
            return CONF_TEXT_TO_NUM[t]
        try:
            c = float(t)
            return max(0.0, min(1.0, c))
        except Exception:
            return 0.5
    return 0.5


def make_descriptor(parsed: Dict[str, Any], canon_mode: str) -> Dict[str, Any]:
    raw_class = parsed.get("class", parsed.get("material"))
    raw_color = parsed.get("color")
    raw_surface = parsed.get("surface", parsed.get("pattern"))
    raw_aging = parsed.get("aging", parsed.get("condition"))
    raw_conf = parsed.get("confidence")

    return {
        "class": canonicalize_value(raw_class, MATERIAL_VOCAB, mode=canon_mode),
        "color": canonicalize_value(raw_color, COLOR_VOCAB, mode=canon_mode),
        "surface": canonicalize_value(raw_surface, SURFACE_VOCAB, mode=canon_mode),
        "aging": canonicalize_value(raw_aging, AGING_VOCAB, mode=canon_mode),
        "confidence": canonicalize_confidence(raw_conf),
    }


def load_mask_bool(p: Union[str, Path, None]) -> Optional[np.ndarray]:
    if not p:
        return None
    p = Path(p)
    if not p.exists():
        return None
    m = np.array(Image.open(p).convert("L"))
    return (m > 127)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns (x0,y0,x1,y1) in PIL crop convention (x1,y1 exclusive).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def bbox_height_from_mask(mask: np.ndarray) -> int:
    b = bbox_from_mask(mask)
    if b is None:
        return 0
    _, y0, _, y1 = b
    return int(y1 - y0)


def should_skip_instance_by_bbox(cls: str, mask: np.ndarray, args) -> Tuple[bool, str]:
    """Heuristic gating for distant buildings using bbox height."""
    if not getattr(args, "skip_far_buildings", False):
        return False, ""
    h = bbox_height_from_mask(mask)
    if h <= 0:
        return True, "empty_mask"
    if cls == "wall" and h < int(getattr(args, "min_inst_bbox_h_wall", 0)):
        return True, f"far_building_bbox_h<{args.min_inst_bbox_h_wall}px"
    if cls == "roof" and h < int(getattr(args, "min_inst_bbox_h_roof", 0)):
        return True, f"far_building_bbox_h<{args.min_inst_bbox_h_roof}px"
    return False, ""


def pad_bbox(b: Tuple[int, int, int, int], W: int, H: int, pad: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    return x0, y0, x1, y1


def masked_crop_from_mask(
    rgb_path: Path,
    mask_bool: np.ndarray,
    out_path: Path,
    crop_pad: int = 24,
    min_crop_side: int = 64,
    structure_class: str = "unknown",  # ✅ FIX 2: Add class-aware padding
) -> Dict[str, Any]:
    """
    Creates a PNG where only mask is visible (else black), then crops to bbox(+pad).
    Returns stats dict: coverage_pct, crop_box, crop_size, skipped flags.
    
    ✅ FIX 2: Uses adaptive padding based on structure class to reduce contamination.
    """
    rgb = Image.open(rgb_path).convert("RGB")
    W, H = rgb.size

    if mask_bool.shape[0] != H or mask_bool.shape[1] != W:
        # safety: resize mask with nearest
        m_img = Image.fromarray((mask_bool.astype(np.uint8) * 255))
        m_img = m_img.resize((W, H), resample=Image.NEAREST)
        mask_bool = (np.array(m_img) > 127)

    coverage_pct = float(mask_bool.mean() * 100.0)
    b = bbox_from_mask(mask_bool)

    if b is None:
        return {"coverage_pct": round(coverage_pct, 3), "skipped": True, "reason": "empty_mask"}

    # ✅ FIX 2: Adaptive padding based on structure class
    if structure_class in ["roof", "door", "window"]:
        effective_pad = max(8, crop_pad // 3)  # Roof: tight crop (8px min)
    elif structure_class == "wall":
        effective_pad = max(12, crop_pad // 2)  # Wall: medium crop (12px min)
    else:  # road
        effective_pad = crop_pad  # Road: full padding OK
    
    b = pad_bbox(b, W, H, effective_pad)
    x0, y0, x1, y1 = b
    cw, ch = (x1 - x0), (y1 - y0)
    
    if cw < min_crop_side or ch < min_crop_side:
        return {
            "coverage_pct": round(coverage_pct, 3),
            "skipped": True,
            "reason": f"crop_too_small({cw}x{ch})",
            "crop_box": [x0, y0, x1, y1],
        }

    rgb_np = np.array(rgb).astype(np.uint8)
    m3 = np.stack([mask_bool] * 3, axis=-1)
    masked = (rgb_np * m3).astype(np.uint8)

    masked_crop = masked[y0:y1, x0:x1, :]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked_crop).save(out_path)  # PNG

    return {
        "coverage_pct": round(coverage_pct, 3),
        "skipped": False,
        "crop_box": [x0, y0, x1, y1],
        "crop_size": [int(cw), int(ch)],
        "effective_padding": effective_pad,  # For debugging
    }


def analyze_with_nemotron(
    client: OpenAI,
    model: str,
    img_path: Path,
    class_name: str,
    max_retries: int = 3,
    retry_backoff: float = 0.8,
) -> Tuple[Dict[str, Any], str]:
    """Call Nemotron and robustly parse JSON with retries.

    Retries are useful for sporadic API failures, timeouts, or occasional non-JSON outputs.
    """
    prompt = PROMPT.format(class_name=class_name)
    mime = mime_from_path(img_path)

    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            img_b64 = encode_image_base64(img_path)
            resp = client.chat.completions.create(
                model=model,
                temperature=0.4,
                max_tokens=220,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    ]
                }]
            )

            raw = (resp.choices[0].message.content or "").strip()
            parsed = parse_json_from_text(raw)
            return parsed, raw

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = retry_backoff * (2 ** (attempt - 1))
                print(f"      ⚠️ Nemotron error ({attempt}/{max_retries}) on {class_name}: {type(e).__name__} — retry in {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                break

    raise ValueError(f"Nemotron failed after {max_retries} attempts for {class_name}: {type(last_err).__name__}: {last_err}")





# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Step 4: Nemotron material descriptors from SAM3 manifest (global + per-building).")
    ap.add_argument("--manifest", required=True, help="SAM3 manifest.json (data_output/sam3_instances/manifest.json)")
    ap.add_argument("--masked_dir", default="data_output/masked_rgb", help="Output dir for masked/cropped PNGs (debug + reproducibility)")
    ap.add_argument("--out_json", default="data_output/materials_from_nemotron.json", help="Output JSON file")
    ap.add_argument("--model", default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", help="NVIDIA Nemotron model id")

    # NOTE: min_cov is in PERCENT
    ap.add_argument("--min_cov", type=float, default=1.0, help="Skip if mask coverage < min_cov (%) on full image (GLOBAL for road/wall)")
    ap.add_argument("--min_cov_roof", type=float, default=1.5, help="Min coverage % for GLOBAL roof (street-level imagery often has small roofs)")
    ap.add_argument("--min_cov_building", type=float, default=0.5, help="Skip if mask coverage < min_cov_building (%) on full image (PER-BUILDING)")
    ap.add_argument("--roof_conf_threshold", type=float, default=0.6, help="✅ FIX 3: Minimum confidence for roof predictions")
    ap.add_argument("--min_cov_roof_building", type=float, default=None,
                    help="Optional: min coverage % for PER-BUILDING roof. If unset, uses --min_cov_building.")

    ap.add_argument("--max_retries", type=int, default=3, help="Max retries for Nemotron calls (parse/API errors).")
    ap.add_argument("--retry_backoff", type=float, default=0.8, help="Base seconds for exponential backoff between retries.")

    ap.add_argument("--skip_far_buildings", action="store_true",
                    help="Skip per-building wall/roof when bbox height is too small (distant buildings).")
    ap.add_argument("--min_inst_bbox_h_wall", type=int, default=70,
                    help="If --skip_far_buildings: skip instance wall if bbox height < this (px).")
    ap.add_argument("--min_inst_bbox_h_roof", type=int, default=110,
                    help="If --skip_far_buildings: skip instance roof if bbox height < this (px).")

    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N images in manifest")
    ap.add_argument("--per_building", action="store_true", help="Also run per-building wall/roof descriptors using instance masks")
    ap.add_argument("--canon_mode", choices=["off", "soft", "hard"], default="soft",
                    help="Canonicalization: off=raw only, soft=map when possible else other:<..>, hard=force closed-set")
    ap.add_argument("--save_raw_text", action="store_true", help="Also store raw model response text (bigger JSON).")

    ap.add_argument("--crop_pad", type=int, default=24, help="Padding (px) added around mask bbox before cropping (adaptive per class)")
    ap.add_argument("--min_crop_side", type=int, default=64, help="Skip if cropped bbox is smaller than this")

    args = ap.parse_args()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing NVIDIA_API_KEY env var. Set it before running.")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"❌ Manifest not found: {manifest_path}")

    masked_root = Path(args.masked_dir)
    masked_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("❌ Manifest must be a JSON list (as produced by the SAM3 script).")

    if args.limit and args.limit > 0:
        data = data[:args.limit]

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "canon_mode": args.canon_mode,
            "min_cov_global_pct": args.min_cov,
            "min_cov_roof_pct": args.min_cov_roof,  # ✅ FIX 1: Document roof threshold
            "min_cov_building_pct": args.min_cov_building,
            "roof_conf_threshold": args.roof_conf_threshold,  # ✅ FIX 3: Document confidence filter
            "min_cov_roof_building_pct": args.min_cov_roof_building if args.min_cov_roof_building is not None else args.min_cov_building,
            "max_retries": args.max_retries,
            "retry_backoff": args.retry_backoff,
            "skip_far_buildings": bool(args.skip_far_buildings),
            "min_inst_bbox_h_wall": args.min_inst_bbox_h_wall,
            "min_inst_bbox_h_roof": args.min_inst_bbox_h_roof,
            "manifest": str(manifest_path),
            "masked_dir": str(masked_root),
            "per_building": bool(args.per_building),
            "crop_pad": args.crop_pad,
            "min_crop_side": args.min_crop_side,
            "version": "v3.0",  # Mark this as improved version
        },
        "data": {}
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- SNIPPET 1: LOAD EXISTING PROGRESS ---
    if out_path.exists():
        print(f"🔄 Found existing output file at {out_path}. Loading progress...")
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Existing JSON is corrupted. Starting fresh.")
            #results = {"meta": {}, "data": {}}
    #else:
    #    results = {"meta": {}, "data": {}}

    results["meta"]["version"] = "v3.0"
    results["meta"]["manifest"] = str(manifest_path)
    if "data" not in results:
        results["data"] = {}
    # -----------------------------------------

    for i, item in enumerate(data, 1):
        img_id = item.get("id") or Path(item.get("rgb", item.get("image", ""))).stem
        # --- SNIPPET 2: SKIP IF ALREADY DONE ---
        if str(img_id) in results["data"]:
            print(f"[{i}/{len(data)}] ⏭️ {img_id} (Already processed, skipping)")
            continue
        # ---------------------------------------
        rgb_path = Path(item.get("rgb") or item.get("image") or "")
        if not rgb_path.exists():
            print(f"[{i}/{len(data)}] ⚠️ missing rgb: {rgb_path} (skip)")
            continue

        print(f"[{i}/{len(data)}] 📸 {img_id}")

        g = item.get("global", {})
        per_global: Dict[str, Any] = {}
        img_out_dir = masked_root / str(img_id)
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # ---------- GLOBAL ----------
        for cls in CLASSES_GLOBAL:
            mask_path = g.get(f"{cls}_mask", "")
            m = load_mask_bool(mask_path)
            if m is None:
                per_global[cls] = {"skipped": True, "reason": "missing_mask"}
                continue

            cov = float(m.mean() * 100.0)
            
            # ✅ FIX 1: Different coverage threshold for roof
            min_threshold = args.min_cov_roof if cls in ["roof", "window", "door"] else args.min_cov
            
            if cov < min_threshold:
                per_global[cls] = {
                    "skipped": True, 
                    "coverage_pct": round(cov, 3), 
                    "reason": f"low_coverage (threshold={min_threshold}%)"
                }
                continue

            out_png = img_out_dir / f"global__{cls}.png"
            stats = masked_crop_from_mask(
                rgb_path, m, out_png,
                crop_pad=args.crop_pad,
                min_crop_side=args.min_crop_side,
                structure_class=cls,  # ✅ FIX 2: Pass class for adaptive padding
            )
            if stats.get("skipped"):
                per_global[cls] = {"skipped": True, **stats, "mask": str(mask_path)}
                continue

            try:
                parsed_json, raw_text = analyze_with_nemotron(client, args.model, out_png, cls, max_retries=args.max_retries, retry_backoff=args.retry_backoff)
                descriptor = make_descriptor(parsed_json, args.canon_mode)

                # ✅ FIX 3: Post-filter low-confidence roof predictions
                if cls in ["roof", "window", "door"] and descriptor["confidence"] < args.roof_conf_threshold:
                    original_class = descriptor["class"]
                    descriptor["class"] = f"other:uncertain_{cls}"
                    descriptor["filtered"] = True
                    descriptor["original_class"] = original_class
                    descriptor["filter_reason"] = f"Low confidence ({descriptor['confidence']:.2f} < {args.roof_conf_threshold})"

                per_global[cls] = {
                    "skipped": False,
                    "mask": str(mask_path),
                    "masked_rgb": str(out_png),
                    **stats,
                    "raw_prediction": parsed_json,
                    "material_descriptor": descriptor,
                }
                if args.save_raw_text:
                    per_global[cls]["raw_text"] = raw_text

                # Print with filter indicator
                filtered_indicator = " [FILTERED]" if descriptor.get("filtered") else ""
                print(f"   ✓ global {cls}: {descriptor['class']} | conf={descriptor['confidence']:.2f} ({stats['coverage_pct']:.1f}%){filtered_indicator}")
            
            except Exception as e:
                per_global[cls] = {
                    "skipped": False,
                    "mask": str(mask_path),
                    "masked_rgb": str(out_png),
                    **stats,
                    "error": str(e),
                }
                print(f"   ❌ global {cls}: {e}")

        # ---------- PER BUILDING ----------
        per_instances: Dict[str, Any] = {}
        if args.per_building:
            instances = item.get("instances", []) or []
            for inst in instances:
                bid = inst.get("id", "building")
                inst_out_dir = img_out_dir / bid
                inst_out_dir.mkdir(parents=True, exist_ok=True)

                per_inst_cls: Dict[str, Any] = {}
                for cls in CLASSES_BUILDING:
                    mask_path = inst.get(f"{cls}_mask", "")
                    m = load_mask_bool(mask_path)
                    if m is None:
                        per_inst_cls[cls] = {"skipped": True, "reason": "missing_mask"}
                        continue

                    cov = float(m.mean() * 100.0)

                    # Optional: skip distant buildings by bbox height
                    skip_far, why_far = should_skip_instance_by_bbox(cls, m, args)
                    if skip_far:
                        per_inst_cls[cls] = {
                            "skipped": True,
                            "coverage_pct": round(cov, 3),
                            "reason": f"{why_far} (bbox-gating)",
                        }
                        continue

                    # Per-building: default uses --min_cov_building; you can override roof via --min_cov_roof_building
                    min_threshold = (args.min_cov_roof_building if (cls in ["roof", "window", "door"] and args.min_cov_roof_building is not None) else args.min_cov_building)
                    
                    if cov < min_threshold:
                        per_inst_cls[cls] = {
                            "skipped": True, 
                            "coverage_pct": round(cov, 3), 
                            "reason": f"low_coverage (threshold={min_threshold}%)"
                        }
                        continue

                    out_png = inst_out_dir / f"{cls}.png"
                    stats = masked_crop_from_mask(
                        rgb_path, m, out_png,
                        crop_pad=args.crop_pad,
                        min_crop_side=args.min_crop_side,
                        structure_class=cls,  # ✅ FIX 2: Class-aware padding for buildings too
                    )
                    if stats.get("skipped"):
                        per_inst_cls[cls] = {"skipped": True, **stats, "mask": str(mask_path)}
                        continue

                    try:
                        parsed_json, raw_text = analyze_with_nemotron(client, args.model, out_png, f"{cls} (building)", max_retries=args.max_retries, retry_backoff=args.retry_backoff)
                        descriptor = make_descriptor(parsed_json, args.canon_mode)

                        # ✅ FIX 3: Filter low-confidence building roofs
                        if cls in ["roof", "window", "door"] and descriptor["confidence"] < args.roof_conf_threshold:
                            original_class = descriptor["class"]
                            descriptor["class"] = f"other:uncertain_{cls}"
                            descriptor["filtered"] = True
                            descriptor["original_class"] = original_class
                            descriptor["filter_reason"] = f"Low confidence ({descriptor['confidence']:.2f} < {args.roof_conf_threshold})"

                        per_inst_cls[cls] = {
                            "skipped": False,
                            "mask": str(mask_path),
                            "masked_rgb": str(out_png),
                            **stats,
                            "raw_prediction": parsed_json,
                            "material_descriptor": descriptor,
                        }
                        if args.save_raw_text:
                            per_inst_cls[cls]["raw_text"] = raw_text

                        filtered_indicator = " [FILTERED]" if descriptor.get("filtered") else ""
                        print(f"   ✓ {bid} {cls}: {descriptor['class']} | conf={descriptor['confidence']:.2f} ({stats['coverage_pct']:.1f}%){filtered_indicator}")
                    
                    except Exception as e:
                        per_inst_cls[cls] = {
                            "skipped": False,
                            "mask": str(mask_path),
                            "masked_rgb": str(out_png),
                            **stats,
                            "error": str(e),
                        }
                        print(f"   ❌ {bid} {cls}: {e}")

                per_instances[bid] = per_inst_cls

        results["data"][str(img_id)] = {
            "rgb": str(rgb_path),
            "global": per_global,
            "instances": per_instances if args.per_building else {},
        }
        # --- SNIPPET 3: SAVE AFTER EVERY IMAGE ---
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        # -----------------------------------------

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
