import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
import numpy as np


IGNORE_CLASSES = {
    "", "unknown", "none",
    "not_visible", "other:not_visible",
    "other:unknown", "other:none"
}


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_class(c: Optional[str]) -> str:
    if c is None:
        return ""
    c = str(c).strip().lower()
    return c


def extract_materials(prediction: dict, surface_type: str) -> Set[str]:
    """
    Returns a SET of material classes for a given surface_type in one image.
    Supports:
      - Hierarchical: prediction["global"][surface_type] and prediction["instances"][...][surface_type]
      - Baseline full-image: prediction["canonicalized_predictions"][surface_type]["class"] (fallback raw_predictions)
    """
    materials: Set[str] = set()

    # -------------------------
    # (A) Hierarchical structure
    # -------------------------
    if isinstance(prediction, dict):
        # global (road/wall/roof)
        global_block = prediction.get("global")
        if isinstance(global_block, dict):
            surf = global_block.get(surface_type, {})
            if isinstance(surf, dict) and not surf.get("skipped", False):
                md = surf.get("material_descriptor", {})
                if not isinstance(md, dict):
                    md = {}
                c = normalize_class(md.get("class") or surf.get("class"))
                if c and c not in IGNORE_CLASSES:
                    materials.add(c)

        # instances (per-building)
        instances = prediction.get("instances")
        if isinstance(instances, dict):
            for _bid, inst in instances.items():
                if not isinstance(inst, dict):
                    continue
                surf = inst.get(surface_type, {})
                if not isinstance(surf, dict) or surf.get("skipped", False):
                    continue
                md = surf.get("material_descriptor", {})
                if not isinstance(md, dict):
                    md = {}
                c = normalize_class(md.get("class") or surf.get("class"))
                if c and c not in IGNORE_CLASSES:
                    materials.add(c)

    # -------------------------
    # (B) Baseline full-image structure
    # -------------------------
    # baseline_full_image.json has:
    # prediction["canonicalized_predictions"][surface_type]["class"]
    # or prediction["raw_predictions"][surface_type]["class"]
    can = prediction.get("canonicalized_predictions")
    if isinstance(can, dict) and surface_type in can:
        c = normalize_class((can.get(surface_type) or {}).get("class"))
        if c and c not in IGNORE_CLASSES:
            materials.add(c)

    raw = prediction.get("raw_predictions")
    if not materials and isinstance(raw, dict) and surface_type in raw:
        c = normalize_class((raw.get(surface_type) or {}).get("class"))
        if c and c not in IGNORE_CLASSES:
            materials.add(c)

    return materials


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_stability_for_cluster(
    cluster_images: List[str],
    predictions: Dict[str, dict],
    surface_type: str
) -> Optional[float]:
    mats_per_image: List[Set[str]] = []
    for img_id in cluster_images:
        pred = predictions.get(str(img_id))
        if not isinstance(pred, dict):
            continue
        mats = extract_materials(pred, surface_type)
        if mats:
            mats_per_image.append(mats)

    if len(mats_per_image) < 2:
        return None

    sims = []
    for i in range(len(mats_per_image)):
        for j in range(i + 1, len(mats_per_image)):
            sims.append(jaccard_similarity(mats_per_image[i], mats_per_image[j]))

    return float(np.mean(sims)) if sims else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchical", required=True, help="materials_v2_filtered_FULL.json (AutoPBR structured)")
    parser.add_argument("--baseline", required=False, help="baseline_full_image.json (full-image Nemotron)")
    parser.add_argument("--clusters", required=True, help="building_clusters.json from gps_clustering.py")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    print("Loading data...")
    hierarchical_json = load_json(Path(args.hierarchical))
    hierarchical_data = hierarchical_json.get("data", {})

    baseline_data = {}
    if args.baseline:
        baseline_json = load_json(Path(args.baseline))
        baseline_data = baseline_json.get("data", {})

    clusters_info = load_json(Path(args.clusters))
    clusters = clusters_info.get("clusters", {})

    avg_views = clusters_info.get("avg_views_per_cluster", float("nan"))
    print(f"Loaded {len(clusters)} clusters, avg {avg_views:.1f} views/cluster")

    results = []
    for building_id, image_ids in clusters.items():
        if len(image_ids) < 2:
            continue

        row = {"building_id": building_id, "n_views": len(image_ids)}

        for surface in ["road", "wall", "roof"]:
            h = compute_stability_for_cluster(image_ids, hierarchical_data, surface)
            row[f"hierarchical_{surface}_jaccard"] = "" if h is None else h

            if baseline_data:
                b = compute_stability_for_cluster(image_ids, baseline_data, surface)
                row[f"baseline_{surface}_jaccard"] = "" if b is None else b

        results.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["building_id", "n_views"]
    fieldnames += [f"hierarchical_{s}_jaccard" for s in ["road", "wall", "roof"]]
    if baseline_data:
        fieldnames += [f"baseline_{s}_jaccard" for s in ["road", "wall", "roof"]]

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n✓ Saved {out}")

    # Summary
    print("\n" + "=" * 60)
    for surface in ["road", "wall", "roof"]:
        hv = [r[f"hierarchical_{surface}_jaccard"] for r in results if r[f"hierarchical_{surface}_jaccard"] != ""]
        if hv:
            hv = np.array(hv, dtype=float)
            print(f"{surface.upper()} (Hierarchical): {hv.mean():.3f} ± {hv.std():.3f}")

        if baseline_data:
            bv = [r.get(f"baseline_{surface}_jaccard", "") for r in results if r.get(f"baseline_{surface}_jaccard", "") != ""]
            if bv:
                bv = np.array(bv, dtype=float)
                print(f"{surface.upper()} (Baseline):     {bv.mean():.3f} ± {bv.std():.3f}")

    # Save JSON summary next to CSV
    summary = {"n_clusters_used": len(results), "avg_views_per_cluster": float(avg_views)}
    for surface in ["road", "wall", "roof"]:
        hv = [r[f"hierarchical_{surface}_jaccard"] for r in results if r[f"hierarchical_{surface}_jaccard"] != ""]
        if hv:
            summary[f"hierarchical_{surface}_mean"] = float(np.mean(np.array(hv, dtype=float)))

        if baseline_data:
            bv = [r.get(f"baseline_{surface}_jaccard", "") for r in results if r.get(f"baseline_{surface}_jaccard", "") != ""]
            if bv:
                summary[f"baseline_{surface}_mean"] = float(np.mean(np.array(bv, dtype=float)))

    summary_path = out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"✓ Saved {summary_path}")


if __name__ == "__main__":
    main()
