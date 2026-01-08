#!/usr/bin/env python3
import os
import base64
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
from PIL import Image
from openai import OpenAI

# =========================
# CLASSES / MASKS
# =========================
CLASS_IDS = {"road": 1, "wall": 2, "roof": 3}
CLASSES = ["road", "wall", "roof"]

# =========================
# VOCABS (SOFT CANONICALIZATION)
# - values are ECCV macro-labels
# =========================
MATERIAL_VOCAB = {
    # road-ish
    "asphalt": "asphalt",
    "tarmac": "asphalt",
    "bitumen": "asphalt",
    "concrete": "concrete",
    "cement": "concrete",
    "paver": "brick_paver",
    "pavers": "brick_paver",
    "brick": "brick_paver",
    "bricks": "brick_paver",
    "cobblestone": "stone_paver",
    "cobblestones": "stone_paver",
    "stone": "stone_paver",
    "granite": "stone_paver",

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
    # patterns often used as texture descriptors → fold into rough
    "patterned": "rough",
    "striped": "rough",
    "checkered": "rough",
    "tile": "rough",
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

# =========================
# PROMPT
# - IMPORTANT: We DO NOT over-constrain the model.
# - We ask for a descriptor + confidence numeric.
# - We allow "other:<free_text>" for class/color/etc if unsure.
# =========================
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
  "class": "asphalt | concrete | brick | stone | wood | plaster | glass | metal | tile | slate | shingles | other:<...> | unknown",
  "color": "black|gray|white|red|brown|yellow|blue|green|beige|orange|other:<...>|unknown",
  "surface": "smooth|rough|other:<...>|unknown",
  "aging": "new|well_maintained|weathered|worn|aged|dirty|damaged|other:<...>|unknown",
  "confidence": 0.0
}}
""".strip()

# =========================
# Utils
# =========================
def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_json_from_text(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found. Raw: {raw[:200]}")
    return json.loads(raw[start:end + 1])

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
    other_prefix: str = "other:"
) -> str:
    """
    mode:
      - off: return raw (cleaned) or default
      - soft: map when possible, else keep "other:<token>" if any, else "other:<raw>" or default
      - hard: force to vocab values or default
    """
    raw = _pick_first(raw_value)
    if raw is None:
        return default
    tok = _norm(raw)
    if not tok:
        return default

    if mode == "off":
        return tok

    # already marked as other
    if keep_other and tok.startswith(other_prefix):
        # compress: other:foo bar -> other:foo_bar
        tail = tok[len(other_prefix):].strip().replace(" ", "_")
        return f"{other_prefix}{tail}" if tail else default

    # direct synonym mapping
    if tok in vocab:
        return vocab[tok]

    # allow if the model already outputs macro label
    if tok in set(vocab.values()):
        return tok

    # substring match (e.g. "asphalt road" contains asphalt)
    for k, v in vocab.items():
        if k in tok:
            return v

    if mode == "hard":
        return default

    # soft fallback
    if keep_other:
        # keep compact token, not a full sentence
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
    """
    Build ECCV descriptor from parsed JSON (supports legacy keys too).
    """
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

# =========================
# Masking
# =========================
def masked_from_semantic(
    rgb_path: Path,
    semantic_path: Path,
    class_name: str,
    out_path: Path
) -> float:
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    sem = np.array(Image.open(semantic_path).convert("L"))
    class_id = CLASS_IDS[class_name]
    mask = (sem == class_id)
    coverage = mask.mean() * 100.0
    if coverage <= 0:
        return 0.0
    masked_rgb = rgb * np.stack([mask] * 3, axis=-1)
    Image.fromarray(masked_rgb.astype(np.uint8)).save(out_path, quality=95)
    return coverage

# =========================
# Nemotron call
# =========================
def analyze_with_nemotron(client: OpenAI, model: str, img_path: Path, class_name: str) -> Tuple[Dict[str, Any], str]:
    img_b64 = encode_image_base64(img_path)
    prompt = PROMPT.format(class_name=class_name)

    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,  # lower = more stable
        max_tokens=220,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}} ,
            ]
        }]
    )

    raw = resp.choices[0].message.content
    parsed = parse_json_from_text(raw)
    return parsed, raw

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos_dir", default="photos2")
    ap.add_argument("--semantic_dir", default="semantic_masks", help="Folder with <id>_semantic.png (0/1/2/3)")
    ap.add_argument("--masked_dir", default="masked_rgb", help="Debug masked RGB outputs")
    ap.add_argument("--out_json", default="materials_per_image.json")
    ap.add_argument("--model", default="nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
    ap.add_argument("--min_cov", type=float, default=0.2, help="Skip if coverage < min_cov (%)")
    ap.add_argument("--limit", type=int, default=0, help="If >0, process only first N images")

    # NEW:
    ap.add_argument("--canon_mode", choices=["off", "soft", "hard"], default="soft",
                    help="Canonicalization: off=raw only, soft=map when possible else other:<..>, hard=force closed-set")
    ap.add_argument("--save_raw_text", action="store_true",
                    help="If set, also store the raw model response text (bigger JSON).")
    args = ap.parse_args()

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing NVIDIA_API_KEY env var. Set it before running.")

    root = Path.cwd()
    photos_dir = Path(args.photos_dir) if Path(args.photos_dir).is_absolute() else (root / args.photos_dir)
    semantic_dir = Path(args.semantic_dir) if Path(args.semantic_dir).is_absolute() else (root / args.semantic_dir)
    masked_dir = root / args.masked_dir
    masked_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(photos_dir.glob("*.jpg"))
    if args.limit and args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"❌ No .jpg found in {photos_dir}")

    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "canon_mode": args.canon_mode,
            "min_cov": args.min_cov,
            "photos_dir": str(photos_dir),
            "semantic_dir": str(semantic_dir),
        },
        "data": {}
    }

    for i, rgb_path in enumerate(images, 1):
        photo_id = rgb_path.stem
        sem_path = semantic_dir / f"{photo_id}_semantic.png"
        if not sem_path.exists():
            print(f"[{i}/{len(images)}] ⚠️ missing semantic mask: {sem_path.name}")
            continue

        print(f"[{i}/{len(images)}] 📸 {photo_id}")
        per_class: Dict[str, Any] = {}

        for cls in CLASSES:
            out_masked = masked_dir / f"{photo_id}_{cls}.jpg"
            cov = masked_from_semantic(rgb_path, sem_path, cls, out_masked)

            if cov < args.min_cov:
                per_class[cls] = {"skipped": True, "coverage": round(cov, 3)}
                continue

            try:
                parsed_json, raw_text = analyze_with_nemotron(client, args.model, out_masked, cls)

                # two-level output
                descriptor = make_descriptor(parsed_json, args.canon_mode)

                per_class[cls] = {
                    "skipped": False,
                    "coverage": round(cov, 3),
                    "masked_rgb": str(out_masked),

                    # ✅ RAW (for analysis + ablations)
                    "raw_prediction": parsed_json,

                    # ✅ CANONICAL (for metrics/plots in ECCV)
                    "material_descriptor": descriptor,
                }

                if args.save_raw_text:
                    per_class[cls]["raw_text"] = raw_text

                print(f"   ✓ {cls}: {descriptor['class']} | conf={descriptor['confidence']:.2f} ({cov:.1f}%)")

            except Exception as e:
                per_class[cls] = {
                    "skipped": False,
                    "coverage": round(cov, 3),
                    "masked_rgb": str(out_masked),
                    "error": str(e),
                }
                print(f"   ❌ {cls}: {e}")

        results["data"][photo_id] = {
            "image": str(rgb_path),
            "semantic_mask": str(sem_path),
            "classes": per_class,
        }

    out_path = root / args.out_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
