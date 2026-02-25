#!/usr/bin/env python3
"""
BASELINE: VLM on full image (no structural masks).
"""
import os
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union
from openai import OpenAI
from PIL import Image

# ============================================================
# COPIED VOCABS & UTILS (from 4_nemotron_per_imagev2.py)
# ============================================================
MATERIAL_VOCAB = {
    "asphalt": "asphalt",
    "tarmac": "asphalt",
    "bitumen": "asphalt",
    "concrete": "concrete",
    "cement": "concrete",
    "brick": "brick",
    "bricks": "brick",
    "paver": "brick",
    "pavers": "brick",
    "cobblestone": "stone",
    "cobblestones": "stone",
    "stone": "stone",
    "granite": "stone",
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

# ============================================================
# BASELINE PROMPT (deliberately weaker)
# ============================================================
PROMPT_FULL_IMAGE = """
Analyze this urban street-level photo and identify the materials visible for roads, walls, and roofs.

Return ONLY valid JSON:
{
  "road": {"class": "material", "color": "color", "confidence": 0.0},
  "wall": {"class": "material", "color": "color", "confidence": 0.0},
  "roof": {"class": "material", "color": "color", "confidence": 0.0}
}

If a category is not visible, set class to "not_visible".
"""

def analyze_full_image(client, model, img_path: Path):
    """VLM analysis without structural masks."""
    
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    resp = client.chat.completions.create(
        model=model,
        temperature=0.4,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_FULL_IMAGE},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }},
            ]
        }]
    )
    
    raw = resp.choices[0].message.content
    
    # Parse JSON
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON in response: {raw[:200]}")
    
    parsed = json.loads(raw[start:end+1])
    
    # Canonicalize predictions
    canonicalized = {}
    for cls in ["road", "wall", "roof", "door", "window"]:
        if cls in parsed:
            raw_pred = parsed[cls]
            canonicalized[cls] = {
                "class": canonicalize_value(
                    raw_pred.get("class", "unknown"),
                    MATERIAL_VOCAB,
                    mode="soft"
                ),
                "color": canonicalize_value(
                    raw_pred.get("color", "unknown"),
                    COLOR_VOCAB,
                    mode="soft"
                ),
                "surface": "unknown",  # Not in baseline prompt
                "aging": "unknown",     # Not in baseline prompt
                "confidence": canonicalize_confidence(raw_pred.get("confidence", 0.5))
            }
        else:
            canonicalized[cls] = {
                "class": "unknown",
                "color": "unknown",
                "surface": "unknown",
                "aging": "unknown",
                "confidence": 0.0
            }
    
    return parsed, canonicalized

def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing NVIDIA_API_KEY")
    
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    
    # Load manifest
    manifest_path = Path("data_output/sam3_instances/manifest.json")
    if not manifest_path.exists():
        raise SystemExit(f"❌ Manifest not found: {manifest_path}")
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    
    """results = {
        "meta": {
            "method": "baseline_full_image",
            "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "prompt_type": "unguided_multi_class",
            "note": "No structural masks, analyzes full image at once",
            "source_manifest": str(manifest_path)
        },
        "data": {}
    }"""

    # --- SNIPPET 1: LOAD EXISTING PROGRESS ---
    out_path = Path("data_output/baseline_full_image.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"🔄 Found existing output file at {out_path}. Loading progress...")
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Existing JSON is corrupted. Starting fresh.")
            results = {"meta": {}, "data": {}}
    else:
        results = {"meta": {}, "data": {}}

    results["meta"] = {
        "method": "baseline_full_image",
        "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "prompt_type": "unguided_multi_class",
        "note": "No structural masks, analyzes full image at once",
        "source_manifest": str(manifest_path)
    }
    if "data" not in results:
        results["data"] = {}
    # -----------------------------------------
    
    skipped = 0
    processed = 0
    errors = 0
    
    for i, item in enumerate(manifest, 1):
        # ✅ FIX: Safely check both keys and ensure it's a file
        img_id = item.get("id", f"image_{i}")

        # --- SNIPPET 2: SKIP IF ALREADY DONE ---
        if str(img_id) in results["data"]:
            print(f"[{i}/{len(manifest)}] ⏭️ {img_id} (Already processed, skipping)")
            continue
        # ---------------------------------------
        
        # Grab the path string, fallback to "image" if "rgb" is missing
        raw_path_str = item.get("rgb") or item.get("image") or ""
        rgb_path = Path(raw_path_str)
        
        # Validate that it's actually a file, not just an empty directory path
        if not raw_path_str or not rgb_path.is_file():
            skipped += 1
            print(f"[{i}/{len(manifest)}] ⚠️  {img_id}: missing or invalid file path -> '{raw_path_str}'")
            continue
        
        print(f"[{i}/{len(manifest)}] 📸 {img_id}")
        
        try:
            raw_pred, canon_pred = analyze_full_image(
                client,
                "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                rgb_path
            )
            
            results["data"][img_id] = {
                "rgb": str(rgb_path),
                "raw_predictions": raw_pred,
                "canonicalized_predictions": canon_pred
            }
            
            # Print summary
            for cls in ["road", "wall", "roof", "door", "window"]:
                mat = canon_pred[cls]["class"]
                conf = canon_pred[cls]["confidence"]
                print(f"   {cls}: {mat} (conf={conf:.2f})")
            
            processed += 1
        
        except Exception as e:
            errors += 1
            print(f"   ❌ Error: {e}")
            results["data"][img_id] = {
                "rgb": str(rgb_path),
                "error": str(e)
            }
        
        # --- SNIPPET 3: SAVE AFTER EVERY IMAGE ---
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        # -----------------------------------------
    
    # Save results
    out_path = Path("baseline_full_image.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"✅ Saved: {out_path}")
    print(f"📊 Processed: {processed}/{len(manifest)}")
    print(f"⚠️  Skipped (missing files): {skipped}")
    print(f"❌ Errors: {errors}")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()