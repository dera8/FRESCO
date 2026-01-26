import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# CONFIG / THRESHOLDS
# ============================================================

@dataclass
class Thresholds:
    # ---- Image / instance minimums ----
    min_building_area_px: int = 500
    min_wall_area_px: int = 400
    min_roof_area_px: int = 800  # used for "roof visibility"

    # ---- Roof visibility gating (validator-side) ----
    roof_min_frac_of_building: float = 0.03  # 3% of building => likely visible
    roof_visible_min_area_px: int = 800

    # ---- Coverage checks (visibility-aware) ----
    min_visible_structure_cov_warn: float = 0.15
    min_visible_structure_cov_ok: float = 0.35

    # ---- Overlap checks (wall vs roof) ----
    wr_iou_warn: float = 0.10
    wr_iou_fail: float = 0.25

    # fraction overlap thresholds
    wr_overlap_frac_warn: float = 0.25
    wr_overlap_frac_fail: float = 0.50

    # ---- Road sanity (optional) ----
    # if road exists and overlaps a lot with building -> warning
    road_building_overlap_frac_warn: float = 0.20

    # ---- Small mask warnings (optional) ----
    tiny_component_warn_px: int = 200


# ============================================================
# IO UTILITIES
# ============================================================

def load_mask_bool(path: Path) -> Optional[np.ndarray]:
    if path is None or (not path.exists()):
        return None
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 0)

def safe_zeros_like(ref: np.ndarray) -> np.ndarray:
    return np.zeros_like(ref, dtype=bool)

def area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int(mask.sum())

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0

def frac_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """fraction of a that overlaps b: |a∩b| / |a|"""
    denom = a.sum()
    if denom <= 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / denom)

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEBUG OVERLAY
# ============================================================

def overlay_debug(
    rgb_path: Path,
    building: np.ndarray,
    wall: np.ndarray,
    roof: np.ndarray,
    road: Optional[np.ndarray],
    out_path: Path,
    title: str,
    status: str,
    reasons: str = "",
):
    """
    Create a visual overlay for debugging:
    - building: red outline
    - wall: green fill (semi)
    - roof: blue fill (semi)
    - road: yellow fill (semi)
    """
    ensure_parent(out_path)

    img = Image.open(rgb_path).convert("RGB")
    W, H = img.size
    base = img.convert("RGBA")

    def colorize(mask_bool: np.ndarray, rgba: Tuple[int, int, int, int]) -> Image.Image:
        m = (mask_bool.astype(np.uint8) * 255)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # put alpha only where mask is true
        px = np.zeros((H, W, 4), dtype=np.uint8)
        px[..., 0] = rgba[0]
        px[..., 1] = rgba[1]
        px[..., 2] = rgba[2]
        px[..., 3] = (m > 0).astype(np.uint8) * rgba[3]
        return Image.fromarray(px, mode="RGBA")

    # semi-transparent fills
    if road is not None:
        base = Image.alpha_composite(base, colorize(road, (255, 215, 0, 70)))  # yellow
    base = Image.alpha_composite(base, colorize(wall, (0, 255, 0, 70)))        # green
    base = Image.alpha_composite(base, colorize(roof, (0, 128, 255, 70)))      # blue

    # building outline (red)
    # simple outline via edge detection from mask
    b = building.astype(np.uint8)
    # edge: pixels in building with at least one neighbor off
    edge = np.zeros_like(b, dtype=bool)
    edge[1:-1, 1:-1] = (
        (b[1:-1, 1:-1] == 1) &
        (
            (b[0:-2, 1:-1] == 0) |
            (b[2:  , 1:-1] == 0) |
            (b[1:-1, 0:-2] == 0) |
            (b[1:-1, 2:  ] == 0)
        )
    )
    base = Image.alpha_composite(base, colorize(edge, (255, 0, 0, 170)))

    draw = ImageDraw.Draw(base)
    txt = f"{title} | {status}"
    if reasons:
        txt2 = reasons
    else:
        txt2 = rgb_path.name

    # small readable box
    draw.rectangle([10, 10, 10 + 1200, 10 + 80], fill=(0, 0, 0, 140))
    draw.text((20, 18), txt, fill=(255, 255, 255, 255))
    draw.text((20, 48), txt2[:170], fill=(220, 220, 220, 255))

    base.convert("RGB").save(out_path)


# ============================================================
# MANIFEST PARSING
# ============================================================

def normalize_manifest(manifest_data) -> List[dict]:
    """
    Support:
    - list of entries (new style): [{"id","rgb","out_dir","global","instances":[...]}]
    - dict with "items"
    - old style: [{"image": "...", "global": {...}, "instances": [...]}]
    """
    if isinstance(manifest_data, dict) and "items" in manifest_data:
        return manifest_data["items"]
    if isinstance(manifest_data, list):
        return manifest_data
    raise ValueError("Manifest format not recognized (expected list or dict with 'items').")

def resolve_paths(entry: dict) -> Tuple[Path, Path, Optional[Path], List[dict]]:
    """
    Returns:
      rgb_path, img_dir, road_path, instances_list
    """
    # new style
    if "rgb" in entry:
        rgb_path = Path(entry["rgb"])
        img_dir = Path(entry.get("out_dir", rgb_path.parent))
        instances = entry.get("instances", [])
        road_path = Path(entry["global"]["road_mask"]) if entry.get("global", {}).get("road_mask") else (img_dir / "road.png")
        return rgb_path, img_dir, road_path, instances

    # old style
    rgb_path = Path(entry.get("image", entry.get("rgb", "")))
    # in your older scripts you used OUT_DIR / img_path.stem
    img_dir = Path(entry.get("out_dir", Path(entry.get("global", {}).get("road_mask", "")).parent if entry.get("global", {}).get("road_mask") else rgb_path.parent))
    instances = entry.get("instances", [])

    # road
    road_path = None
    if entry.get("global", {}).get("road_mask"):
        road_path = Path(entry["global"]["road_mask"])
    else:
        candidate = img_dir / "road.png"
        road_path = candidate if candidate.exists() else None

    return rgb_path, img_dir, road_path, instances


# ============================================================
# CORE VALIDATION
# ============================================================

def roof_visible(roof_mask: np.ndarray, building_mask: np.ndarray, th: Thresholds) -> bool:
    r_area = area(roof_mask)
    b_area = area(building_mask)
    if b_area <= 0:
        return False
    if r_area < th.roof_visible_min_area_px:
        return False
    if (r_area / b_area) < th.roof_min_frac_of_building:
        return False
    return True

def visible_structure_union(
    building: np.ndarray,
    wall: np.ndarray,
    roof: np.ndarray,
    th: Thresholds
) -> Tuple[np.ndarray, Dict[str, bool]]:
    """
    Visibility-aware union:
    - wall considered visible if area>=min_wall_area_px
    - roof considered visible if roof_visible(...) is True
    """
    vis = {"wall_visible": False, "roof_visible": False}

    w_ok = area(wall) >= th.min_wall_area_px
    r_ok = roof_visible(roof, building, th)

    vis["wall_visible"] = bool(w_ok)
    vis["roof_visible"] = bool(r_ok)

    union = safe_zeros_like(building)
    if w_ok:
        union |= wall
    if r_ok:
        union |= roof
    return union, vis

def classify_overlap_wall_roof(
    wall: np.ndarray,
    roof: np.ndarray,
    th: Thresholds
) -> Tuple[str, List[str], Dict[str, float]]:
    """
    Returns:
      overlap_level in {"OK","WARNING","FAIL"}
      reasons list
      metrics dict
    """
    reasons = []
    metrics = {}

    wA = area(wall)
    rA = area(roof)

    if wA == 0 or rA == 0:
        metrics["wr_iou"] = 0.0
        metrics["wr_overlap_frac_wall"] = 0.0
        metrics["wr_overlap_frac_roof"] = 0.0
        return "OK", reasons, metrics

    wr_iou = iou(wall, roof)
    of_w = frac_overlap(wall, roof)
    of_r = frac_overlap(roof, wall)

    metrics["wr_iou"] = wr_iou
    metrics["wr_overlap_frac_wall"] = of_w
    metrics["wr_overlap_frac_roof"] = of_r

    # heavy overlap for BOTH => FAIL
    if (wr_iou >= th.wr_iou_fail) and (of_w >= th.wr_overlap_frac_fail) and (of_r >= th.wr_overlap_frac_fail):
        reasons.append(f"wall_roof_overlap_heavy (iou={wr_iou:.3f}, of_w={of_w:.3f}, of_r={of_r:.3f})")
        return "FAIL", reasons, metrics

    # moderate overlap => WARNING (or FAIL if you want stricter)
    if (wr_iou >= th.wr_iou_warn) or (of_w >= th.wr_overlap_frac_warn) or (of_r >= th.wr_overlap_frac_warn):
        reasons.append(f"wall_roof_overlap (iou={wr_iou:.3f}, of_w={of_w:.3f}, of_r={of_r:.3f})")
        return "WARNING", reasons, metrics

    return "OK", reasons, metrics

def validate_instance(
    rgb_path: Path,
    building_path: Path,
    wall_path: Optional[Path],
    roof_path: Optional[Path],
    road_mask: Optional[np.ndarray],
    th: Thresholds,
) -> Tuple[str, List[str], Dict[str, float], Dict[str, bool], Dict[str, int]]:
    """
    Validate one building instance.
    Returns: status, reasons, metrics, flags, areas
    """
    reasons: List[str] = []
    metrics: Dict[str, float] = {}
    flags: Dict[str, bool] = {}
    areas: Dict[str, int] = {}

    b = load_mask_bool(building_path)
    if b is None:
        return "SKIP", ["missing_building_mask"], metrics, flags, areas

    w = load_mask_bool(wall_path) if wall_path else None
    r = load_mask_bool(roof_path) if roof_path else None

    # If wall/roof missing -> treat as empty (but keep note)
    empty = safe_zeros_like(b)
    if w is None:
        w = empty
        reasons.append("missing_wall_mask")
    if r is None:
        r = empty
        reasons.append("missing_roof_mask")

    # Areas
    bA = area(b); wA = area(w); rA = area(r)
    areas = {"building_area": bA, "wall_area": wA, "roof_area": rA}

    if bA < th.min_building_area_px:
        return "SKIP", [f"tiny_building<{th.min_building_area_px}px"], metrics, flags, areas

    # Clamp structure inside building (sanity)
    w_in = np.logical_and(w, b)
    r_in = np.logical_and(r, b)
    # if lots of structure outside building -> warning
    w_out = area(w) - area(w_in)
    r_out = area(r) - area(r_in)
    if w_out > th.tiny_component_warn_px:
        reasons.append("wall_outside_building")
    if r_out > th.tiny_component_warn_px:
        reasons.append("roof_outside_building")

    # Overlap check (wall vs roof)
    overlap_level, overlap_reasons, overlap_metrics = classify_overlap_wall_roof(w_in, r_in, th)
    reasons.extend(overlap_reasons)
    metrics.update(overlap_metrics)

    # Visibility-aware coverage
    union_vis, vis_flags = visible_structure_union(b, w_in, r_in, th)
    flags.update(vis_flags)

    vis_union_area = area(union_vis)
    structure_cov = float(vis_union_area / bA) if bA > 0 else 0.0
    metrics["visible_structure_cov"] = structure_cov

    # Coverage status
    cov_level = "OK"
    if structure_cov < th.min_visible_structure_cov_warn:
        cov_level = "WARNING"
        reasons.append(f"low_visible_structure_cov<{th.min_visible_structure_cov_warn:.2f} ({structure_cov:.3f})")
    elif structure_cov < th.min_visible_structure_cov_ok:
        # mild warning (optional)
        reasons.append(f"mid_visible_structure_cov<{th.min_visible_structure_cov_ok:.2f} ({structure_cov:.3f})")

    # Road/building overlap sanity (optional)
    if road_mask is not None:
        # fraction of building covered by road
        rb = frac_overlap(b, road_mask)
        metrics["road_building_overlap_frac"] = rb
        if rb >= th.road_building_overlap_frac_warn:
            reasons.append(f"road_overlaps_building ({rb:.3f})")

    # Decide final status:
    # Priority: FAIL if overlap heavy; else WARNING if any warnings; else OK
    if overlap_level == "FAIL":
        status = "FAIL"
    else:
        # if we have any serious warning tokens
        warning_tokens = [
            x for x in reasons
            if (
                x.startswith("low_visible_structure_cov")
                or x.startswith("wall_roof_overlap")
                or x.startswith("road_overlaps_building")
                or x in ("wall_outside_building", "roof_outside_building")
            )
        ]
        status = "WARNING" if len(warning_tokens) > 0 else "OK"

    return status, reasons, metrics, flags, areas


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="Path to manifest.json")
    ap.add_argument("--out_dir", type=str, default="qc_out", help="Output dir for reports")
    ap.add_argument("--save_overlays", action="store_true", help="Save debug overlays for WARNING/FAIL")
    ap.add_argument("--overlays_limit", type=int, default=200, help="Max overlays to save")
    args = ap.parse_args()

    th = Thresholds()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_csv = out_dir / "qc_report.csv"
    instances_csv = out_dir / "qc_instances.csv"
    summary_json = out_dir / "summary.json"
    overlays_dir = out_dir / "debug_overlays"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = normalize_manifest(data)

    # Stats
    totals = {"OK": 0, "WARNING": 0, "FAIL": 0, "SKIP": 0}
    fail_reasons_counter: Dict[str, int] = {}
    warn_reasons_counter: Dict[str, int] = {}

    rows_report = []
    rows_instances = []

    overlays_saved = 0

    for entry in items:
        rgb_path, img_dir, road_path, instances = resolve_paths(entry)
        if not rgb_path.exists():
            # skip entire image if rgb missing
            continue

        road_mask = load_mask_bool(road_path) if (road_path is not None and road_path.exists()) else None

        # If your manifest instances entries already contain mask paths
        for inst in instances:
            # Support both:
            # new: {"id","building_mask","wall_mask","roof_mask"}
            # old: same keys, maybe "id" differs
            bid = inst.get("id", inst.get("instance_id", "unknown"))

            building_mask_path = Path(inst.get("building_mask", inst.get("mask_building", ""))) if inst.get("building_mask") or inst.get("mask_building") else None
            wall_mask_path = Path(inst.get("wall_mask", inst.get("mask_wall", ""))) if inst.get("wall_mask") or inst.get("mask_wall") else None
            roof_mask_path = Path(inst.get("roof_mask", inst.get("mask_roof", ""))) if inst.get("roof_mask") or inst.get("mask_roof") else None

            # Fallback: if paths are missing, try standard folder layout:
            if building_mask_path is None or str(building_mask_path) == "":
                # try img_dir / bid / building.png
                building_mask_path = img_dir / bid / "building.png"
            if wall_mask_path is None or str(wall_mask_path) == "":
                wall_mask_path = img_dir / bid / "wall.png"
            if roof_mask_path is None or str(roof_mask_path) == "":
                roof_mask_path = img_dir / bid / "roof.png"

            status, reasons, metrics, flags, areas = validate_instance(
                rgb_path=rgb_path,
                building_path=building_mask_path,
                wall_path=wall_mask_path,
                roof_path=roof_mask_path,
                road_mask=road_mask,
                th=th
            )

            totals[status] += 1

            # count reasons
            if status == "FAIL":
                for r in reasons:
                    fail_reasons_counter[r] = fail_reasons_counter.get(r, 0) + 1
            elif status == "WARNING":
                for r in reasons:
                    warn_reasons_counter[r] = warn_reasons_counter.get(r, 0) + 1

            # save per-instance row
            row_inst = {
                "image_id": entry.get("id", rgb_path.stem),
                "instance_id": bid,
                "rgb": str(rgb_path),
                "building_mask": str(building_mask_path),
                "wall_mask": str(wall_mask_path),
                "roof_mask": str(roof_mask_path),
                "status": status,
                "reasons": ";".join(reasons[:25]),
                **areas,
                **metrics,
                **flags,
            }
            rows_instances.append(row_inst)

            # Save overlays
            if args.save_overlays and status in ("WARNING", "FAIL") and overlays_saved < args.overlays_limit:
                b = load_mask_bool(building_mask_path)

                w = load_mask_bool(wall_mask_path)
                if w is None:
                    w = safe_zeros_like(b)

                r = load_mask_bool(roof_mask_path)
                if r is None:
                    r = safe_zeros_like(b)

                out_path = overlays_dir / f"{entry.get('id', rgb_path.stem)}__{bid}__{status}.jpg"
                overlay_debug(
                    rgb_path=rgb_path,
                    building=b,
                    wall=w,
                    roof=r,
                    road=road_mask,
                    out_path=out_path,
                    title=f"{entry.get('id', rgb_path.stem)} / {bid}",
                    status=status,
                    reasons=", ".join(reasons[:3]),
                )
                overlays_saved += 1

        # image-level summary row (optional)
        rows_report.append({
            "image_id": entry.get("id", rgb_path.stem),
            "rgb": str(rgb_path),
            "n_instances": len(instances),
        })

    # Write CSVs
    def write_csv(path: Path, rows: List[dict]):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(report_csv, rows_report)
    write_csv(instances_csv, rows_instances)

    summary = {
        "thresholds": asdict(th),
        "totals": totals,
        "fail_reasons_top": sorted(fail_reasons_counter.items(), key=lambda x: x[1], reverse=True)[:30],
        "warn_reasons_top": sorted(warn_reasons_counter.items(), key=lambda x: x[1], reverse=True)[:30],
        "overlays_saved": overlays_saved if args.save_overlays else 0,
        "files": {
            "qc_report_csv": str(report_csv),
            "qc_instances_csv": str(instances_csv),
            "summary_json": str(summary_json),
            "overlays_dir": str(overlays_dir) if args.save_overlays else None,
        }
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ QC SUMMARY ================")
    print("Totals:", totals)
    if summary["fail_reasons_top"]:
        print("\nTop FAIL reasons:")
        for k, v in summary["fail_reasons_top"][:10]:
            print(f"  {v:>6}  {k}")
    if summary["warn_reasons_top"]:
        print("\nTop WARNING reasons:")
        for k, v in summary["warn_reasons_top"][:10]:
            print(f"  {v:>6}  {k}")
    print("===========================================")
    print("Wrote:", report_csv)
    print("Wrote:", instances_csv)
    print("Wrote:", summary_json)
    if args.save_overlays:
        print("Overlays:", overlays_dir)


if __name__ == "__main__":
    main()
