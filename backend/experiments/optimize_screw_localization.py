from pathlib import Path
import json
import csv
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "patchcore_screw_wideresnet50_l2_005.ckpt"
)
OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "screw_wideresnet50_l2_005_localization"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# LOAD MODEL & RUN PREDICTIONS
# ============================================================
print("[1/5] Initializing datamodule and running inference...")
datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="screw",
    num_workers=0,
)
datamodule.setup()

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2"],
    pre_trained=True,
    coreset_sampling_ratio=0.05,
    num_neighbors=9,
)

engine = Engine(
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)

prediction_batches = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

if not prediction_batches:
    raise RuntimeError("No predictions returned.")

# ============================================================
# EXTRACT ALL SAMPLES
# ============================================================
print("[2/5] Processing raw prediction batches...")

def to_numpy(val):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)

samples = []
for batch in prediction_batches:
    image_paths = getattr(batch, "image_path", None)
    if image_paths is None:
        continue

    images = to_numpy(getattr(batch, "image", None))
    anomaly_maps = to_numpy(getattr(batch, "anomaly_map", None))
    pred_masks = to_numpy(getattr(batch, "pred_mask", None))
    gt_masks = to_numpy(getattr(batch, "gt_mask", None))
    gt_labels = to_numpy(getattr(batch, "gt_label", None))

    batch_size = len(image_paths)
    for i in range(batch_size):
        path = Path(image_paths[i])
        defect_type = path.parent.name

        amap = anomaly_maps[i] if anomaly_maps is not None else None
        if amap is not None and amap.ndim == 3 and amap.shape[0] == 1:
            amap = amap[0]

        gmask = gt_masks[i] if gt_masks is not None else None
        if gmask is not None and gmask.ndim == 3 and gmask.shape[0] == 1:
            gmask = gmask[0]

        pmask = pred_masks[i] if pred_masks is not None else None
        if pmask is not None and pmask.ndim == 3 and pmask.shape[0] == 1:
            pmask = pmask[0]

        img = images[i] if images is not None else None
        if img is not None and img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        samples.append({
            "image_path": str(path),
            "defect_type": defect_type,
            "image": img,
            "anomaly_map": amap,
            "baseline_pred_mask": pmask,
            "gt_mask": gmask,
            "gt_label": bool(gt_labels[i]) if gt_labels is not None else (defect_type != "good"),
        })

# ============================================================
# DETERMINISTIC STRATIFIED SPLIT (SEED 42)
# ============================================================
print("[3/5] Creating 50/50 stratified validation / held-out test split (seed=42)...")
by_category = {}
for s in samples:
    cat = s["defect_type"]
    by_category.setdefault(cat, []).append(s)

val_samples = []
test_samples = []

split_info = {}
for cat, items in sorted(by_category.items()):
    items_copy = list(items)
    random.seed(SEED)
    random.shuffle(items_copy)
    n_val = len(items_copy) // 2
    val_subset = items_copy[:n_val]
    test_subset = items_copy[n_val:]
    val_samples.extend(val_subset)
    test_samples.extend(test_subset)
    split_info[cat] = {
        "total": len(items_copy),
        "val_count": len(val_subset),
        "test_count": len(test_subset),
    }

print("Split Summary:")
for cat, counts in split_info.items():
    print(f"  - {cat:<20}: Total {counts['total']:>2} | Val {counts['val_count']:>2} | Test {counts['test_count']:>2}")

# Compute global min and max of raw anomaly maps from validation set (no leakage)
all_val_maps = [s["anomaly_map"] for s in val_samples if s["anomaly_map"] is not None]
global_min = float(min(m.min() for m in all_val_maps))
global_max = float(max(m.max() for m in all_val_maps))

print(f"Global Anomaly Map Range (Validation set): min={global_min:.4f}, max={global_max:.4f}")

for s in samples:
    amap = s["anomaly_map"]
    if amap is not None:
        s["norm_map"] = np.clip((amap - global_min) / (global_max - global_min + 1e-8), 0.0, 1.0)

# ============================================================
# EVALUATION & POST-PROCESSING HELPERS
# ============================================================
def calculate_sample_metrics(gt_mask, pred_mask, is_defective):
    if pred_mask is None:
        pred_has_defect = False
    else:
        pred_has_defect = bool(np.any(pred_mask > 0))

    if is_defective:
        tp = 1 if pred_has_defect else 0
        fn = 0 if pred_has_defect else 1
        good_fp = 0
    else:
        tp = 0
        fn = 0
        good_fp = 1 if pred_has_defect else 0

    iou = None
    if gt_mask is not None and pred_mask is not None:
        gt = np.asarray(gt_mask).astype(bool)
        pred = np.asarray(pred_mask).astype(bool)
        intersection = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = float(intersection / union)

    return tp, fn, good_fp, iou

def post_process_map(norm_map, threshold, min_component_size=0, morph_op=None):
    if norm_map is None:
        return None

    binary = (norm_map >= threshold).astype(np.uint8)

    # Connected component filtering
    if min_component_size > 0 and np.any(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_component_size:
                filtered[labels == label] = 1
        binary = filtered

    # Morphological processing
    if morph_op is not None and np.any(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if morph_op == "opening":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif morph_op == "closing":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif morph_op == "open_close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

def evaluate_dataset(sample_list, threshold, min_component_size=0, morph_op=None, use_baseline_pred=False):
    total_tp = 0
    total_fn = 0
    good_total = 0
    good_fp = 0
    ious = []

    per_defect = {}

    for s in sample_list:
        is_defective = s["gt_label"]
        defect_cat = s["defect_type"]

        if use_baseline_pred:
            pred_m = (s["baseline_pred_mask"] > 0).astype(np.uint8) if s["baseline_pred_mask"] is not None else None
        else:
            pred_m = post_process_map(s["norm_map"], threshold, min_component_size, morph_op)

        tp, fn, gfp, iou = calculate_sample_metrics(s["gt_mask"], pred_m, is_defective)

        if is_defective:
            total_tp += tp
            total_fn += fn
            if defect_cat not in per_defect:
                per_defect[defect_cat] = {"total": 0, "tp": 0, "fn": 0, "ious": []}
            per_defect[defect_cat]["total"] += 1
            per_defect[defect_cat]["tp"] += tp
            per_defect[defect_cat]["fn"] += fn
            if iou is not None:
                per_defect[defect_cat]["ious"].append(iou)
                ious.append(iou)
        else:
            good_total += 1
            good_fp += gfp

    recall = (total_tp / (total_tp + total_fn)) * 100.0 if (total_tp + total_fn) > 0 else 0.0
    mean_iou = float(np.mean(ious)) * 100.0 if ious else 0.0
    fpr = (good_fp / good_total) * 100.0 if good_total > 0 else 0.0

    return {
        "recall": recall,
        "mean_iou": mean_iou,
        "fpr": fpr,
        "tp": total_tp,
        "fn": total_fn,
        "good_total": good_total,
        "good_fp": good_fp,
        "per_defect": per_defect,
    }

# ============================================================
# OPTIMIZATION SWEEPS ON VALIDATION SPLIT
# ============================================================
print("\n[4/5] Running Optimization Sweeps on Validation Set...")

# --- OPTIMIZATION A: Threshold Sweep ---
thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
thresh_rows = []

best_thresh = thresholds[0]
best_thresh_score = (-1.0, -1.0, -1.0) # (Satisfies Recall>=90, Satisfies FPR<5, Mean IoU)

for t in thresholds:
    res = evaluate_dataset(val_samples, threshold=t)
    thresh_rows.append({
        "threshold": t,
        "recall": res["recall"],
        "mean_iou": res["mean_iou"],
        "fpr": res["fpr"],
        "tp": res["tp"],
        "fn": res["fn"],
        "good_fp": res["good_fp"],
    })
    
    score_key = (1 if res["recall"] >= 90.0 else 0, 1 if res["fpr"] < 5.0 else 0, res["mean_iou"])
    if score_key > best_thresh_score:
        best_thresh_score = score_key
        best_thresh = t

with open(OUTPUT_DIR / "threshold_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=thresh_rows[0].keys())
    writer.writeheader()
    writer.writerows(thresh_rows)

print(f"-> Best Validation Threshold: {best_thresh} (Recall: {best_thresh_score[0]}, FPR: {best_thresh_score[1]}, Mean IoU: {best_thresh_score[2]:.2f}%)")

# --- OPTIMIZATION B: Connected Component Sweep ---
component_sizes = [0, 10, 25, 50, 100, 250, 500]
cc_rows = []

best_cc = 0
best_cc_score = (-1.0, -1.0, -1.0)

for cc in component_sizes:
    res = evaluate_dataset(val_samples, threshold=best_thresh, min_component_size=cc)
    cc_rows.append({
        "min_component_size": cc,
        "recall": res["recall"],
        "mean_iou": res["mean_iou"],
        "fpr": res["fpr"],
        "tp": res["tp"],
        "fn": res["fn"],
        "good_fp": res["good_fp"],
    })

    score_key = (1 if res["recall"] >= 90.0 else 0, 1 if res["fpr"] < 5.0 else 0, res["mean_iou"])
    if score_key > best_cc_score:
        best_cc_score = score_key
        best_cc = cc

with open(OUTPUT_DIR / "connected_component_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cc_rows[0].keys())
    writer.writeheader()
    writer.writerows(cc_rows)

print(f"-> Best Component Size: {best_cc} (Mean IoU: {best_cc_score[2]:.2f}%)")

# --- OPTIMIZATION C: Morphology Sweep ---
morph_ops = [None, "opening", "closing", "open_close"]
morph_rows = []

best_morph = None
best_morph_score = (-1.0, -1.0, -1.0)

for mop in morph_ops:
    res = evaluate_dataset(val_samples, threshold=best_thresh, min_component_size=best_cc, morph_op=mop)
    morph_rows.append({
        "morphology_op": str(mop),
        "recall": res["recall"],
        "mean_iou": res["mean_iou"],
        "fpr": res["fpr"],
        "tp": res["tp"],
        "fn": res["fn"],
        "good_fp": res["good_fp"],
    })

    score_key = (1 if res["recall"] >= 90.0 else 0, 1 if res["fpr"] < 5.0 else 0, res["mean_iou"])
    if score_key > best_morph_score:
        best_morph_score = score_key
        best_morph = mop

with open(OUTPUT_DIR / "morphology_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=morph_rows[0].keys())
    writer.writeheader()
    writer.writerows(morph_rows)

print(f"-> Best Morphology Operation: {best_morph} (Mean IoU: {best_morph_score[2]:.2f}%)")

# --- OPTIMIZATION D: Combination Search ---
candidate_threshs = list(set([best_thresh, max(0.10, best_thresh - 0.05), min(0.90, best_thresh + 0.05)]))
candidate_ccs = list(set([best_cc, 0, 25, 50]))
candidate_morphs = [None, "opening", "closing", "open_close"]

comb_rows = []
best_combo = None
best_combo_score = (-1, -1, -1.0)

for t in candidate_threshs:
    for cc in candidate_ccs:
        for mop in candidate_morphs:
            res = evaluate_dataset(val_samples, threshold=t, min_component_size=cc, morph_op=mop)
            comb_rows.append({
                "threshold": t,
                "min_component_size": cc,
                "morphology_op": str(mop),
                "recall": res["recall"],
                "mean_iou": res["mean_iou"],
                "fpr": res["fpr"],
                "tp": res["tp"],
                "fn": res["fn"],
                "good_fp": res["good_fp"],
            })

            score_key = (1 if res["recall"] >= 90.0 else 0, 1 if res["fpr"] < 5.0 else 0, res["mean_iou"])
            if score_key > best_combo_score:
                best_combo_score = score_key
                best_combo = {"threshold": t, "min_component_size": cc, "morphology_op": mop}

with open(OUTPUT_DIR / "combination_search.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comb_rows[0].keys())
    writer.writeheader()
    writer.writerows(comb_rows)

print("\nSELECTED LOCKED VALIDATION CONFIGURATION:")
print(json.dumps(best_combo, indent=2))

with open(OUTPUT_DIR / "selected_validation_config.json", "w") as f:
    json.dump(best_combo, f, indent=2)

# ============================================================
# PART 3 — FINAL HELD-OUT TEST EVALUATION
# ============================================================
print("\n[5/5] Evaluating Locked Config on Held-Out Test Split...")

baseline_test_eval = evaluate_dataset(test_samples, threshold=0.5, use_baseline_pred=True)
opt_test_eval = evaluate_dataset(
    test_samples,
    threshold=best_combo["threshold"],
    min_component_size=best_combo["min_component_size"],
    morph_op=best_combo["morphology_op"],
)

final_test_summary = {
    "split_info": split_info,
    "selected_config": best_combo,
    "baseline": {
        "recall": baseline_test_eval["recall"],
        "mean_iou": baseline_test_eval["mean_iou"],
        "fpr": baseline_test_eval["fpr"],
        "tp": baseline_test_eval["tp"],
        "fn": baseline_test_eval["fn"],
        "good_total": baseline_test_eval["good_total"],
        "good_fp": baseline_test_eval["good_fp"],
        "per_defect": {
            k: {
                "total": v["total"],
                "tp": v["tp"],
                "fn": v["fn"],
                "recall": (v["tp"]/v["total"])*100.0 if v["total"]>0 else 0.0,
                "mean_iou": float(np.mean(v["ious"]))*100.0 if v["ious"] else 0.0,
            } for k, v in baseline_test_eval["per_defect"].items()
        }
    },
    "optimized": {
        "recall": opt_test_eval["recall"],
        "mean_iou": opt_test_eval["mean_iou"],
        "fpr": opt_test_eval["fpr"],
        "tp": opt_test_eval["tp"],
        "fn": opt_test_eval["fn"],
        "good_total": opt_test_eval["good_total"],
        "good_fp": opt_test_eval["good_fp"],
        "per_defect": {
            k: {
                "total": v["total"],
                "tp": v["tp"],
                "fn": v["fn"],
                "recall": (v["tp"]/v["total"])*100.0 if v["total"]>0 else 0.0,
                "mean_iou": float(np.mean(v["ious"]))*100.0 if v["ious"] else 0.0,
            } for k, v in opt_test_eval["per_defect"].items()
        }
    },
    "improvements": {
        "recall_diff": opt_test_eval["recall"] - baseline_test_eval["recall"],
        "mean_iou_diff": opt_test_eval["mean_iou"] - baseline_test_eval["mean_iou"],
        "fpr_diff": opt_test_eval["fpr"] - baseline_test_eval["fpr"],
    },
    "prd_status": {
        "recall_pass": opt_test_eval["recall"] >= 90.0,
        "mean_iou_pass": opt_test_eval["mean_iou"] >= 80.0,
        "fpr_pass": opt_test_eval["fpr"] < 5.0,
        "overall_pass": (opt_test_eval["recall"] >= 90.0) and (opt_test_eval["mean_iou"] >= 80.0) and (opt_test_eval["fpr"] < 5.0),
    }
}

with open(OUTPUT_DIR / "final_heldout_test_results.json", "w") as f:
    json.dump(final_test_summary, f, indent=2)

with open(OUTPUT_DIR / "comparison_report.json", "w") as f:
    json.dump(final_test_summary, f, indent=2)

print("\nFinal Held-Out Test Evaluation Complete.")
print(f"Baseline  -> Recall: {baseline_test_eval['recall']:.2f}%, Mean IoU: {baseline_test_eval['mean_iou']:.2f}%, FPR: {baseline_test_eval['fpr']:.2f}%")
print(f"Optimized -> Recall: {opt_test_eval['recall']:.2f}%, Mean IoU: {opt_test_eval['mean_iou']:.2f}%, FPR: {opt_test_eval['fpr']:.2f}%")
print(f"Diff      -> Recall: {final_test_summary['improvements']['recall_diff']:+.2f}%, Mean IoU: {final_test_summary['improvements']['mean_iou_diff']:+.2f}%, FPR: {final_test_summary['improvements']['fpr_diff']:+.2f}%")

# ============================================================
# VISUAL ANALYSIS: SAVE 5-PANEL VISUALIZATIONS
# ============================================================
print("\nGenerating 5-panel visualizations for each category + false positive cases...")

categories = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top", "good"]

for cat in categories:
    cat_samples = [s for s in test_samples if s["defect_type"] == cat]
    if not cat_samples:
        continue

    # Pick representative samples
    if cat == "good":
        # Find false positives if any, plus normal good
        fp_samples = [s for s in cat_samples if calculate_sample_metrics(s["gt_mask"], post_process_map(s["norm_map"], best_combo["threshold"], best_combo["min_component_size"], best_combo["morphology_op"]), False)[2] > 0]
        selected_vis = fp_samples[:2] if fp_samples else cat_samples[:2]
    else:
        selected_vis = cat_samples[:1]

    for idx, s in enumerate(selected_vis):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # 1. Original Image
        img = s["image"]
        if img is not None:
            if img.max() <= 1.0:
                img_disp = (img * 255).astype(np.uint8)
            else:
                img_disp = img.astype(np.uint8)
            axes[0].imshow(img_disp)
        axes[0].set_title("1. Original Image")
        axes[0].axis("off")

        # 2. Anomaly Map
        amap = s["norm_map"]
        if amap is not None:
            axes[1].imshow(amap, cmap="jet")
        axes[1].set_title("2. Anomaly Map")
        axes[1].axis("off")

        # 3. Ground Truth Mask
        gt_m = s["gt_mask"]
        if gt_m is not None:
            axes[2].imshow(gt_m, cmap="gray")
        else:
            axes[2].imshow(np.zeros((256, 256)), cmap="gray")
        axes[2].set_title("3. Ground-Truth Mask")
        axes[2].axis("off")

        # 4. Baseline Predicted Mask
        base_p = s["baseline_pred_mask"]
        if base_p is not None:
            axes[3].imshow(base_p, cmap="gray")
        else:
            axes[3].imshow(np.zeros((256, 256)), cmap="gray")
        axes[3].set_title("4. Baseline Mask")
        axes[3].axis("off")

        # 5. Optimized Predicted Mask
        opt_p = post_process_map(s["norm_map"], best_combo["threshold"], best_combo["min_component_size"], best_combo["morphology_op"])
        if opt_p is not None:
            axes[4].imshow(opt_p, cmap="gray")
        else:
            axes[4].imshow(np.zeros((256, 256)), cmap="gray")
        axes[4].set_title("5. Optimized Mask")
        axes[4].axis("off")

        plt.suptitle(f"Category: {cat} (Sample {idx+1})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        fname = f"vis_{cat}_sample{idx+1}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved visualization: {fname}")

print("\nOptimization Pipeline Completed Successfully!")
