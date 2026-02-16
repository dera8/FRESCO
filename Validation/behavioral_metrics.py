#!/usr/bin/env python3
"""
behavioral_metrics.py

Compute paired "behavioral" metrics comparing:
- hierarchical predictions (mask-based, structured)
- baseline predictions (full-image, no masks)

Key idea:
- Use hierarchical coverage_pct as *visibility proxy* for BOTH methods.
  (Baseline has no masks; so you cannot infer visibility from baseline itself.)

Outputs a JSON report with:
- hallucination rate (predicts material when surface not visible)
- abstention rate (surface visible but model says none/unknown/skipped)
- low-confidence rate (visible but confidence < threshold)
- agreement rate (both predict non-unknown and classes match)
- McNemar p-values for paired comparisons
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


SURFACES = ["road", "wall", "roof"]


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_global_entry(pred: Dict[str, Any], image_id: str) -> Optional[Dict[str, Any]]:
    """
    Supports two common layouts:
    - {"data": {id: {"global": {...}}}}
    - {id: {"global": {...}}}
    """
    if "data" in pred and isinstance(pred["data"], dict):
        return pred["data"].get(image_id)
    return pred.get(image_id)


def get_surface(pred_entry: Dict[str, Any], surface: str) -> Dict[str, Any]:
    # expected: entry["global"][surface]
    g = pred_entry.get("global", {}) if isinstance(pred_entry, dict) else {}
    s = g.get(surface, {}) if isinstance(g, dict) else {}
    return s if isinstance(s, dict) else {}


def extract_pred_class_conf(surface_obj: Dict[str, Any]) -> Tuple[str, float, bool]:
    """
    Returns (pred_class, confidence, skipped_flag)
    Works for both:
    - hierarchical: surface_obj["material_descriptor"]["class"], ["confidence"], plus surface_obj["skipped"]
    - baseline: same structure in your FIXED pipeline
    """
    skipped = bool(surface_obj.get("skipped", False))
    md = surface_obj.get("material_descriptor", {}) if isinstance(surface_obj.get("material_descriptor", {}), dict) else {}
    cls = str(md.get("class", "unknown") or "unknown").strip().lower()
    conf = clamp01(md.get("confidence", 0.0))
    return cls, conf, skipped


def extract_visibility_from_hier(surface_obj: Dict[str, Any]) -> float:
    """
    Use hierarchical coverage_pct as visibility proxy.
    """
    try:
        return float(surface_obj.get("coverage_pct", 0.0))
    except Exception:
        return 0.0


def is_unknown_like(cls: str) -> bool:
    cls = (cls or "").strip().lower()
    return cls in ("unknown", "none", "")


def mcnemar_p_value(b: int, c: int) -> float:
    """
    Exact McNemar test (two-sided) using binomial distribution on discordant pairs.
    b = count where A=1, B=0
    c = count where A=0, B=1
    """
    # If no discordant pairs, p=1
    n = b + c
    if n == 0:
        return 1.0

    # Two-sided exact binomial with p=0.5
    # p = 2 * min( P(X<=min(b,c)), P(X>=max(b,c)) )
    from math import comb

    k = min(b, c)
    # cumulative P(X<=k) for X~Bin(n,0.5)
    cum = 0.0
    for i in range(0, k + 1):
        cum += comb(n, i) * (0.5 ** n)
    p = 2.0 * cum
    return min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hierarchical", required=True, help="materials_v2_filtered_FULL.json (mask-based)")
    ap.add_argument("--baseline", required=True, help="baseline_full_image.json (full-image)")
    ap.add_argument("--output", required=True, help="Output JSON report path")
    ap.add_argument("--vis_road", type=float, default=10.0, help="Visibility threshold (coverage%%) for road")
    ap.add_argument("--vis_wall", type=float, default=10.0, help="Visibility threshold (coverage%%) for wall")
    ap.add_argument("--vis_roof", type=float, default=5.0, help="Visibility threshold (coverage%%) for roof")
    ap.add_argument("--conf_th", type=float, default=0.3, help="Low-confidence threshold")
    args = ap.parse_args()

    hier = json.loads(Path(args.hierarchical).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    # Common image ids
    hier_ids = set(hier.get("data", hier).keys()) if isinstance(hier, dict) else set()
    base_ids = set(base.get("data", base).keys()) if isinstance(base, dict) else set()
    common_ids = sorted(list(hier_ids & base_ids))

    vis_th = {"road": args.vis_road, "wall": args.vis_wall, "roof": args.vis_roof}

    # Counters
    report: Dict[str, Any] = {
        "n_images_common": len(common_ids),
        "thresholds": {
            "visibility_pct": vis_th,
            "low_confidence": args.conf_th
        },
        "per_surface": {}
    }

    for surf in SURFACES:
        # For paired boolean events: (hier_event, base_event)
        # We'll keep discordant counts for McNemar.
        events = {
            "hallucination": {"A1_B0": 0, "A0_B1": 0, "A1": 0, "B1": 0, "N": 0},
            "abstention":   {"A1_B0": 0, "A0_B1": 0, "A1": 0, "B1": 0, "N": 0},
            "lowconf":      {"A1_B0": 0, "A0_B1": 0, "A1": 0, "B1": 0, "N": 0},
        }

        # Non-paired tallies
        visible_count = 0
        agree_count = 0
        agree_denom = 0

        for image_id in common_ids:
            h_entry = get_global_entry(hier, image_id)
            b_entry = get_global_entry(base, image_id)
            if not h_entry or not b_entry:
                continue

            h_s = get_surface(h_entry, surf)
            b_s = get_surface(b_entry, surf)

            # visibility from hierarchical only
            vis = extract_visibility_from_hier(h_s)
            is_visible = vis >= vis_th[surf]

            h_cls, h_conf, h_skipped = extract_pred_class_conf(h_s)
            b_cls, b_conf, b_skipped = extract_pred_class_conf(b_s)

            # Define events
            # Hallucination: NOT visible, but predicts something non-unknown with conf>=conf_th and not skipped
            h_hallu = (not is_visible) and (not h_skipped) and (not is_unknown_like(h_cls)) and (h_conf >= args.conf_th)
            b_hallu = (not is_visible) and (not b_skipped) and (not is_unknown_like(b_cls)) and (b_conf >= args.conf_th)

            # Abstention: visible, but model says unknown/none OR skipped OR conf==0
            h_abst = is_visible and (h_skipped or is_unknown_like(h_cls) or h_conf <= 0.0)
            b_abst = is_visible and (b_skipped or is_unknown_like(b_cls) or b_conf <= 0.0)

            # Low confidence: visible, predicts something (not unknown), but conf < conf_th
            h_low = is_visible and (not h_skipped) and (not is_unknown_like(h_cls)) and (h_conf < args.conf_th)
            b_low = is_visible and (not b_skipped) and (not is_unknown_like(b_cls)) and (b_conf < args.conf_th)

            # Agreement: visible AND both predict non-unknown and not skipped
            if is_visible:
                visible_count += 1
                if (not h_skipped) and (not b_skipped) and (not is_unknown_like(h_cls)) and (not is_unknown_like(b_cls)):
                    agree_denom += 1
                    if h_cls == b_cls:
                        agree_count += 1

            # Update paired counts
            def upd(ev_name: str, A: bool, B: bool):
                e = events[ev_name]
                e["N"] += 1
                if A: e["A1"] += 1
                if B: e["B1"] += 1
                if A and (not B): e["A1_B0"] += 1
                if (not A) and B: e["A0_B1"] += 1

            upd("hallucination", h_hallu, b_hallu)
            upd("abstention",   h_abst,  b_abst)
            upd("lowconf",      h_low,   b_low)

        # Build surface report
        surf_rep = {
            "visibility_count": visible_count,
            "agreement_rate_when_both_predict": (agree_count / agree_denom) if agree_denom else None,
            "agreement_denom": agree_denom,
            "events": {}
        }

        for ev_name, e in events.items():
            A_rate = e["A1"] / e["N"] if e["N"] else 0.0
            B_rate = e["B1"] / e["N"] if e["N"] else 0.0
            p = mcnemar_p_value(e["A1_B0"], e["A0_B1"])
            surf_rep["events"][ev_name] = {
                "hierarchical_rate": A_rate,
                "baseline_rate": B_rate,
                "mcnemar": {
                    "A1_B0": e["A1_B0"],
                    "A0_B1": e["A0_B1"],
                    "p_value": p
                }
            }

        report["per_surface"][surf] = surf_rep

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Pretty print
    print("\n" + "=" * 70)
    print("BEHAVIORAL METRICS (paired, mask-aware visibility)")
    print("=" * 70)
    print(f"Images (common): {report['n_images_common']}")
    print(f"Visibility thresholds: {report['thresholds']['visibility_pct']}")
    print(f"Low-conf threshold: {report['thresholds']['low_confidence']}")
    print("-" * 70)

    for surf in SURFACES:
        sr = report["per_surface"][surf]
        print(f"\n[{surf.upper()}] visible_count={sr['visibility_count']}, agreement={sr['agreement_rate_when_both_predict']}")
        for ev in ["hallucination", "abstention", "lowconf"]:
            er = sr["events"][ev]
            print(f"  {ev:13s}  hier={er['hierarchical_rate']:.3f}  base={er['baseline_rate']:.3f}  "
                  f"p={er['mcnemar']['p_value']:.6f}")

    print("\n✓ Saved:", str(out_path))


if __name__ == "__main__":
    main()
