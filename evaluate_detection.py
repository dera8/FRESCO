import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings

# Suppress warnings for zero-division when a class is rarely predicted
warnings.filterwarnings('ignore') 

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
GT_JSON_PATH = "ground_truth_eval.json"
PRED_JSON_PATH = "materials_full_filtered.json"
BASELINE_JSON_PATH = "baseline_full_image.json"

CLASSES_TO_EVAL = ["wall", "roof", "road", "door", "window"]

# ==========================================
# UTILS
# ==========================================
def load_mask(path_str):
    if not path_str: return None
    # Fix Windows backslashes for Linux environments
    clean_path = str(path_str).replace('\\', '/')
    p = Path(clean_path)
    if not p.exists(): return None
    m = np.array(Image.open(p).convert("L"))
    return (m > 127)

def get_iou_f1(mask_pred, mask_gt):
    if mask_pred is None and mask_gt is None:
        return 1.0, 1.0
    if mask_pred is None: mask_pred = np.zeros_like(mask_gt)
    if mask_gt is None: mask_gt = np.zeros_like(mask_pred)
    
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    
    iou = intersection / union if union > 0 else 0.0
    f1 = (2 * intersection) / (mask_pred.sum() + mask_gt.sum()) if (mask_pred.sum() + mask_gt.sum()) > 0 else 0.0
    return iou, f1

def extract_material(data_node):
    if not data_node or data_node.get("skipped", True):
        return None
    return data_node.get("material_descriptor", {}).get("class", "unknown")

def get_struct_items(data_node, is_global=False):
    """Smartly parses the JSON node to group items by their structural type."""
    items = defaultdict(list)
    if not data_node: return items
    
    if is_global:
        for struct in CLASSES_TO_EVAL:
            if struct in data_node:
                items[struct].append(data_node[struct])
    else:
        for bid, inst_data in data_node.items():
            if bid.startswith("building"):
                if "wall" in inst_data: items["wall"].append(inst_data["wall"])
                if "roof" in inst_data: items["roof"].append(inst_data["roof"])
                if "door" in inst_data: items["door"].append(inst_data["door"])
                if "roof" in inst_data: items["window"].append(inst_data["window"])
            elif bid.startswith("road"):
                items["road"].append(inst_data)
            elif bid.startswith("sidewalk"):
                items["sidewalk"].append(inst_data)
    return items

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("⏳ Loading JSON files...")
    with open(GT_JSON_PATH, 'r', encoding='utf-8') as f: gt_data = json.load(f).get("data", {})
    with open(PRED_JSON_PATH, 'r', encoding='utf-8') as f: pred_data = json.load(f).get("data", {})
    with open(BASELINE_JSON_PATH, 'r', encoding='utf-8') as f: base_data = json.load(f).get("data", {})

    # TRACKERS: y_true[struct], y_base[struct], y_glob[struct]
    y_true = defaultdict(list)
    y_base = defaultdict(list)
    y_glob = defaultdict(list)

    # TRACKERS: seg_metrics[struct][material] = {"global_iou": [], "global_f1": [], ...}
    seg_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print("🔍 Evaluating Images...")
    for img_id, gt_img in gt_data.items():
        base_img = base_data.get(img_id, {})
        pred_img = pred_data.get(img_id, {})
        
        # 1. Parse GT objects
        # FIX: We ONLY pull from instances to ensure GT is strictly instance-based!
        gt_items = get_struct_items(gt_img.get("instances", {}), is_global=False)
            
        # 2. Parse Prediction objects
        pred_items_glob = get_struct_items(pred_img.get("global", {}), is_global=True)
        pred_items_inst = get_struct_items(pred_img.get("instances", {}), is_global=False)
            
        # ==========================================
        # CLASSIFICATION EVALUATION
        # ==========================================
        for struct in CLASSES_TO_EVAL:
            for obj_gt in gt_items[struct]:
                gt_mat = extract_material(obj_gt)
                if not gt_mat: continue 
                
                # Baseline Prediction (Image-level fallback)
                base_target = "road" if struct == "sidewalk" else struct
                base_mat = base_img.get("canonicalized_predictions", {}).get(base_target, {}).get("class", "unknown")
                if base_mat is None: 
                    base_mat = "unknown"
                
                # Global Prediction (Image-level fallback)
                glob_objs = pred_items_glob[struct]
                glob_mat = "unknown"
                if glob_objs:
                    extracted = extract_material(glob_objs[0])
                    if extracted is not None:
                        glob_mat = extracted
                
                y_true[struct].append(gt_mat)
                y_base[struct].append(base_mat)
                y_glob[struct].append(glob_mat)

        # ==========================================
        # SEGMENTATION EVALUATION
        # ==========================================
        for struct in CLASSES_TO_EVAL:
            gt_materials_present = set(extract_material(obj) for obj in gt_items[struct] if extract_material(obj))
            
            for mat in gt_materials_present:
                # Build GT Mask (Strictly from verified Instances)
                gt_mask = None
                for obj in gt_items[struct]:
                    if extract_material(obj) == mat:
                        m = load_mask(obj.get("mask"))
                        if m is not None: gt_mask = m if gt_mask is None else np.logical_or(gt_mask, m)
                
                if gt_mask is None: continue

                # Build Global Pred Mask
                glob_mask = np.zeros_like(gt_mask)
                for obj in pred_items_glob[struct]:
                    if extract_material(obj) == mat:
                        m = load_mask(obj.get("mask"))
                        if m is not None: glob_mask = m if glob_mask is None else np.logical_or(glob_mask, m)
                
                # Build Instance Pred Mask
                inst_mask = np.zeros_like(gt_mask)
                for obj in pred_items_inst[struct]:
                    if extract_material(obj) == mat:
                        m = load_mask(obj.get("mask"))
                        if m is not None: inst_mask = m if inst_mask is None else np.logical_or(inst_mask, m)

                # Calculate Pixel Math
                g_iou, g_f1 = get_iou_f1(glob_mask, gt_mask)
                i_iou, i_f1 = get_iou_f1(inst_mask, gt_mask)
                
                seg_metrics[struct][mat]["global_iou"].append(g_iou)
                seg_metrics[struct][mat]["global_f1"].append(g_f1)
                seg_metrics[struct][mat]["inst_iou"].append(i_iou)
                seg_metrics[struct][mat]["inst_f1"].append(i_f1)

    # ==========================================
    # PRINT REPORT
    # ==========================================
    print("\n" + "="*70)
    print("📊 CLASSIFICATION METRICS (Baseline vs. Global vs. GT)")
    print("="*70)
    
    for struct in CLASSES_TO_EVAL:
        if len(y_true[struct]) == 0: continue
        print(f"\n🏗️  --- {struct.upper()} (Total Valid Objects: {len(y_true[struct])}) ---")
        
        classes = sorted(list(set(y_true[struct])))
        P_b, R_b, F1_b, _ = precision_recall_fscore_support(y_true[struct], y_base[struct], labels=classes, zero_division=0, average='macro')
        P_g, R_g, F1_g, _ = precision_recall_fscore_support(y_true[struct], y_glob[struct], labels=classes, zero_division=0, average='macro')
        acc_b = accuracy_score(y_true[struct], y_base[struct])
        acc_g = accuracy_score(y_true[struct], y_glob[struct])
        
        print("  [MACRO AVERAGES]")
        print(f"    BASELINE -> Acc: {acc_b:.2f} | Prec: {P_b:.2f} | Rec: {R_b:.2f} | F1: {F1_b:.2f}")
        print(f"    GLOBAL   -> Acc: {acc_g:.2f} | Prec: {P_g:.2f} | Rec: {R_g:.2f} | F1: {F1_g:.2f}")
        
        P_bc, R_bc, F1_bc, S_c = precision_recall_fscore_support(y_true[struct], y_base[struct], labels=classes, zero_division=0)
        P_gc, R_gc, F1_gc, _   = precision_recall_fscore_support(y_true[struct], y_glob[struct], labels=classes, zero_division=0)
        
        print("\n  [PER-CLASS RESULTS]")
        for i, c in enumerate(classes):
            print(f"    🔸 {c} (Support: {S_c[i]})")
            print(f"        Baseline -> Prec: {P_bc[i]:.2f} | Rec: {R_bc[i]:.2f} | F1: {F1_bc[i]:.2f}")
            print(f"        Global   -> Prec: {P_gc[i]:.2f} | Rec: {R_gc[i]:.2f} | F1: {F1_gc[i]:.2f}")

    print("\n\n" + "="*70)
    print("🗺️  SEGMENTATION METRICS (Global Masks vs. Instance Masks)")
    print("="*70)
    
    for struct in CLASSES_TO_EVAL:
        if not seg_metrics[struct]: continue
        print(f"\n🏗️  --- {struct.upper()} ---")
        
        all_g_iou, all_g_f1, all_i_iou, all_i_f1 = [], [], [], []
        for mat, metrics in seg_metrics[struct].items():
            all_g_iou.extend(metrics["global_iou"])
            all_g_f1.extend(metrics["global_f1"])
            all_i_iou.extend(metrics["inst_iou"])
            all_i_f1.extend(metrics["inst_f1"])
            
        print("  [MACRO AVERAGES]")
        print(f"    GLOBAL MASKS   -> Mean IoU: {np.mean(all_g_iou):.3f} | Mean F1: {np.mean(all_g_f1):.3f}")
        print(f"    INSTANCE MASKS -> Mean IoU: {np.mean(all_i_iou):.3f} | Mean F1: {np.mean(all_i_f1):.3f}")

        print("\n  [PER-CLASS RESULTS]")
        for mat in sorted(seg_metrics[struct].keys()):
            metrics = seg_metrics[struct][mat]
            print(f"    🔸 {mat} (Evaluated on {len(metrics['global_iou'])} images)")
            print(f"        Global Masks   -> Mean IoU: {np.mean(metrics['global_iou']):.3f} | Mean F1: {np.mean(metrics['global_f1']):.3f}")
            print(f"        Instance Masks -> Mean IoU: {np.mean(metrics['inst_iou']):.3f} | Mean F1: {np.mean(metrics['inst_f1']):.3f}")
            
    print("\n✅ Evaluation Complete.")

if __name__ == "__main__":
    main()