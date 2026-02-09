#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


# ----------------------------
# IO helpers
# ----------------------------

def norm_path(p: str) -> str:
    return str(Path(p)) if p else ""


def load_predictions(json_path: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return data["data"] if isinstance(data, dict) and "data" in data else data


# ----------------------------
# Sampling (PER-BUILDING WALL)
# ----------------------------

def build_samples(predictions: dict, n_samples: int, min_wall_coverage_pct: float = 0.0):
    samples = []
    for img_id, img_node in predictions.items():
        img_path = norm_path(img_node.get("rgb", ""))
        instances = img_node.get("instances", {}) or {}

        if not img_path or not Path(img_path).exists():
            continue

        for inst_id, inst in instances.items():
            wall = (inst or {}).get("wall")
            if not wall or wall.get("skipped"):
                continue

            cov = float(wall.get("coverage_pct", 0.0) or 0.0)
            if cov < float(min_wall_coverage_pct):
                continue

            mask_path = norm_path(wall.get("mask", ""))
            if not mask_path or not Path(mask_path).exists():
                continue

            pred = wall.get("material_descriptor") or wall.get("raw_prediction") or {}
            pred_class = str(pred.get("class", "unknown"))

            samples.append({
                "image_id": str(img_id),
                "instance_id": str(inst_id),
                "surface": "wall",
                "pred_class": pred_class,
                "image_path": img_path,
                "mask_path": mask_path,
                "coverage_pct": cov,
            })

    random.shuffle(samples)
    return samples[:n_samples]


# ----------------------------
# Visualization
# ----------------------------

def pil_rgb_to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def compute_outline(mask_gray: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    mask_gray: uint8
    returns outline as binary mask 0/255
    """
    m = (mask_gray > 0).astype(np.uint8) * 255
    k = 3
    kernel = np.ones((k, k), np.uint8)

    outline = np.zeros_like(m)
    cur = m.copy()
    for _ in range(max(1, thickness)):
        eroded = cv2.erode(cur, kernel, iterations=1)
        edge = cv2.subtract(cur, eroded)
        outline = cv2.bitwise_or(outline, edge)
        cur = eroded
    return outline


def draw_text_readable(img_bgr: np.ndarray, text: str, org: tuple, scale: float = 0.8):
    """
    Black text with white outline (readable on bright images)
    """
    x, y = org
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)


def render_sample(sample: dict, target_w: int = 1200, target_h: int = 800) -> np.ndarray:
    img_pil = Image.open(sample["image_path"]).convert("RGB")
    img_bgr = pil_rgb_to_bgr(img_pil)

    mask_pil = Image.open(sample["mask_path"]).convert("L").resize(img_pil.size, resample=Image.NEAREST)
    mask = np.array(mask_pil).astype(np.uint8)

    outline = compute_outline(mask, thickness=3)
    vis = img_bgr.copy()
    vis[outline > 0] = (0, 0, 255)  # red outline (BGR)

    # Resize to a fixed comfortable window while preserving aspect
    h, w = vis.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Add readable HUD
    t1 = f"{sample['image_id']} | {sample['instance_id']} | cov={sample['coverage_pct']:.2f}%"
    t2 = f"Pred: {sample['pred_class']}"
    t3 = "Keys: 1=Correct  2=Incorrect  3=Ambiguous  4=Skip  (q=quit)"

    draw_text_readable(vis, t1, (20, 40), 0.9)
    draw_text_readable(vis, t2, (20, 80), 0.9)
    draw_text_readable(vis, t3, (20, 120), 0.8)

    return vis


def wait_for_label(window_name: str) -> str:
    """
    Wait ONLY for keypress in the cv2 window.
    Returns "1"/"2"/"3"/"4" or "q".
    """
    while True:
        key = cv2.waitKey(0) & 0xFF  # block until a key is pressed
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            return chr(key)
        if key in (ord('q'), ord('Q'), 27):  # q or ESC
            return "q"


# ----------------------------
# Summary
# ----------------------------

def compute_metrics(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, "r", encoding="utf-8")))

    valid = [r for r in rows if r["material_accuracy"] in ("1", "2")]
    ambiguous = [r for r in rows if r["material_accuracy"] == "3"]
    skip = [r for r in rows if r["material_accuracy"] == "4"]

    correct = sum(1 for r in valid if r["material_accuracy"] == "1")
    incorrect = sum(1 for r in valid if r["material_accuracy"] == "2")
    acc = correct / (correct + incorrect) if (correct + incorrect) else 0.0

    print("\n" + "-" * 60)
    print("HUMAN EVALUATION SUMMARY (ECCV)")
    print("-" * 60)
    print(f"CSV: {csv_path}")
    print(f"Total samples     : {len(rows)}")
    print(f"Valid (1/2)       : {len(valid)}")
    print(f"Ambiguous (3)     : {len(ambiguous)}")
    print(f"Skip (4)          : {len(skip)}")
    print(f"Material Accuracy : {acc:.3f}")
    print("-" * 60)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSON predictions (annotate) or CSV (summary).")
    parser.add_argument("--output", default="validation_results/human_eval_structured.csv")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_wall_coverage", type=float, default=0.0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.summary:
        compute_metrics(args.predictions)
        return

    preds = load_predictions(args.predictions)
    samples = build_samples(preds, args.sample, args.min_wall_coverage)

    if not samples:
        raise SystemExit("No samples found (check image/mask paths and coverage threshold).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id", "instance_id", "surface", "pred_class",
        "coverage_pct", "material_accuracy", "timestamp"
    ]
    write_header = not out_path.exists()

    window_name = "ECCV-HumanEval"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    try:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for i, sample in enumerate(samples, 1):
                vis = render_sample(sample, target_w=1200, target_h=800)
                cv2.imshow(window_name, vis)

                label = wait_for_label(window_name)
                if label == "q":
                    print("\n[INFO] Quit requested. Stopping.")
                    break

                row = {
                    "image_id": sample["image_id"],
                    "instance_id": sample["instance_id"],
                    "surface": sample["surface"],
                    "pred_class": sample["pred_class"],
                    "coverage_pct": f"{sample['coverage_pct']:.6f}",
                    "material_accuracy": label,
                    "timestamp": datetime.now().isoformat()
                }
                writer.writerow(row)
                f.flush()

                print(f"[{i}/{len(samples)}] saved label={label}")

        print(f"\n✅ Saved annotations to: {out_path}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
