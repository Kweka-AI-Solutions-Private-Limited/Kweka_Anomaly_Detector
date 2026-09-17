"""
evaluate_efficientad.py

Runs the same deterministic threshold/CC/morphology sweep pipeline
as optimize_screw_localization.py, but loads an EfficientAD checkpoint.

Gate: Recall >= 90%, Mean IoU >= 80%, FPR < 5%
"""

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
from anomalib.models import EfficientAd

# ============================================================
# PATHS & CONFIG
# ============================================================
MODEL_SIZE = "medium"   # Must match what was used in train_efficientad.py
CATEGORY = "screw"

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT = BASE_DIR / "outputs" / "efficientad_baseline" / f"efficientad_screw_{MODEL_SIZE}.ckpt"
OUTPUT_DIR = BASE_DIR / "outputs" / "efficientad_baseline" / f"screw_{MODEL_SIZE}_localization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("EFFICIENTAD LOCALIZATION EVALUATION")
print("=" * 70)
print(f"Model size:  {MODEL_SIZE}")
print(f"Category:    {CATEGORY}")
print(f"Checkpoint:  {CHECKPOINT}")
print(f"Output dir:  {OUTPUT_DIR}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}\n"
        f"Run train_efficientad.py first."
    )

# ============================================================
# LOAD MODEL & RUN PREDICTIONS
# ============================================================
print("\n[1/5] Initializing datamodule and running inference ...")
datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category=CATEGORY,
    num_workers=0,
)
datamodule.setup()

model = EfficientAd(
    teacher_out_channels=384 if MODEL_SIZE == "medium" else 128,
    model_size=MODEL_SIZE,
    lr=0.0001,
    weight_decay=0.00001,
    padding=False,
    pad_maps=True,
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
    raise RuntimeError("No predictions returned from engine.predict().")

print(f"Got {len(prediction_batches)} prediction batch(es).")

# ============================================================
# EXTRACT ALL SAMPLES
# ============================================================
print("\n[2/5] Processing raw prediction batches ...")

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

    images      = to_numpy(getattr(batch, "image",       None))
    anomaly_maps = to_numpy(getattr(batch, "anomaly_map", None))
    pred_masks  = to_numpy(getattr(batch, "pred_mask",   None))
    gt_masks    = to_numpy(getattr(batch, "gt_mask",     None))
    gt_labels   = to_numpy(getattr(batch, "gt_label",    None))

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
            "image_path":         str(path),
            "defect_type":        defect_type,
            "image":              img,
            "anomaly_map":        amap,
            "baseline_pred_mask": pmask,
            "gt_mask":            gmask,
            "gt_label":           bool(gt_labels[i]) if gt_labels is not None else (defect_type != "good"),
        })

print(f"Total samples extracted: {len(samples)}")
defect_counts = {}
for s in samples:
    defect_counts[s["defect_type"]] = defect_counts.get(s["defect_type"], 0) + 1
for cat, cnt in sorted(defect_counts.items()):
    print(f"  {cat:<25}: {cnt}")

# ============================================================
# DETERMINISTIC STRATIFIED SPLIT (SEED 42)
# Identical split logic to optimize_screw_localization.py
# ============================================================
print("\n[3/5] 50/50 stratified val/test split (seed=42) ...")
by_category = {}
for s in samples:
    by_category.setdefault(s["defect_type"], []).append(s)

val_samples  = []
test_samples = []
split_info   = {}

for cat, items in sorted(by_category.items()):
    items_copy = list(items)
    random.seed(SEED)
    random.shuffle(items_copy)
    n_val          = len(items_copy) // 2
    val_subset     = items_copy[:n_val]
    test_subset    = items_copy[n_val:]
    val_samples.extend(val_subset)
    test_samples.extend(test_subset)
    split_info[cat] = {
        "total":      len(items_copy),
        "val_count":  len(val_subset),
        "test_count": len(test_subset),
    }

print("Split summary:")
for cat, counts in split_info.items():
    print(f"  {cat:<25}: total {counts['total']:>2} | val {counts['val_count']:>2} | test {counts['test_count']:>2}")

# ============================================================
# ANOMALY MAP NORMALIZATION
# ------------------------------------------------------------
# EfficientAD's anomaly_map is already normalized during validation
# via map_norm_quantiles() (q0.9 and q0.995 stored in the checkpoint).
# The output anomaly_map is in a calibrated range where values > 1
# indicate anomaly. We do NOT re-normalize globally — that would destroy
# the calibration.
#
# Instead, we set norm_map = anomaly_map directly, and search thresholds
# on the actual output scale. We also clip to [0, max_observed] for display.
# ============================================================
all_amaps = [s["anomaly_map"] for s in samples if s["anomaly_map"] is not None]
amap_global_min = float(min(m.min() for m in all_amaps))
amap_global_max = float(max(m.max() for m in all_amaps))
amap_val_max    = float(max(m.max() for m in
                            [s["anomaly_map"] for s in val_samples
                             if s["anomaly_map"] is not None]))

print(f"\nEfficientAD anomaly_map stats (all samples):")
print(f"  min={amap_global_min:.6f}, max={amap_global_max:.6f}")
print(f"  val_max={amap_val_max:.6f}")
print("  NOTE: values > 1.0 indicate anomalous regions (model-calibrated)")

for s in samples:
    amap = s["anomaly_map"]
    if amap is not None:
        # Use anomaly_map directly as norm_map — no re-normalization
        s["norm_map"] = amap.astype(np.float32)


# ============================================================
# EVALUATION & POST-PROCESSING HELPERS
# ============================================================
def calculate_sample_metrics(gt_mask, pred_mask, is_defective):
    pred_has_defect = bool(np.any(pred_mask > 0)) if pred_mask is not None else False

    if is_defective:
        tp, fn, good_fp = (1, 0, 0) if pred_has_defect else (0, 1, 0)
    else:
        tp, fn, good_fp = (0, 0, 1) if pred_has_defect else (0, 0, 0)

    iou = None
    if gt_mask is not None and pred_mask is not None:
        gt   = np.asarray(gt_mask).astype(bool)
        pred = np.asarray(pred_mask).astype(bool)
        intersection = np.logical_and(gt, pred).sum()
        union        = np.logical_or(gt, pred).sum()
        iou = float(intersection / union) if union > 0 else (1.0 if intersection == 0 else 0.0)

    return tp, fn, good_fp, iou


def post_process_map(norm_map, threshold, min_component_size=0, morph_op=None):
    if norm_map is None:
        return None

    binary = (norm_map >= threshold).astype(np.uint8)

    if min_component_size > 0 and np.any(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_size:
                filtered[labels == label] = 1
        binary = filtered

    if morph_op is not None and np.any(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if morph_op == "opening":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        elif morph_op == "closing":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif morph_op == "open_close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif morph_op == "dilate":
            binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

    return binary


def evaluate_dataset(sample_list, threshold, min_component_size=0, morph_op=None,
                     use_baseline_pred=False):
    total_tp = total_fn = good_total = good_fp = 0
    ious = []
    per_defect = {}

    for s in sample_list:
        is_defective = s["gt_label"]
        defect_cat   = s["defect_type"]

        if use_baseline_pred:
            pred_m = (s["baseline_pred_mask"] > 0).astype(np.uint8) \
                     if s["baseline_pred_mask"] is not None else None
        else:
            pred_m = post_process_map(s.get("norm_map"), threshold,
                                      min_component_size, morph_op)

        tp, fn, gfp, iou = calculate_sample_metrics(s["gt_mask"], pred_m, is_defective)

        if is_defective:
            total_tp += tp
            total_fn += fn
            per_defect.setdefault(defect_cat, {"total": 0, "tp": 0, "fn": 0, "ious": []})
            per_defect[defect_cat]["total"] += 1
            per_defect[defect_cat]["tp"]    += tp
            per_defect[defect_cat]["fn"]    += fn
            if iou is not None:
                per_defect[defect_cat]["ious"].append(iou)
                ious.append(iou)
        else:
            good_total += 1
            good_fp    += gfp

    recall   = (total_tp / (total_tp + total_fn)) * 100.0 if (total_tp + total_fn) > 0 else 0.0
    mean_iou = float(np.mean(ious)) * 100.0 if ious else 0.0
    fpr      = (good_fp / good_total) * 100.0 if good_total > 0 else 0.0

    return {
        "recall": recall, "mean_iou": mean_iou, "fpr": fpr,
        "tp": total_tp, "fn": total_fn,
        "good_total": good_total, "good_fp": good_fp,
        "per_defect": per_defect,
    }

# ============================================================
# [4/5] OPTIMIZATION SWEEPS ON VALIDATION SET
# ============================================================
print("\n[4/5] Optimization sweeps on validation set ...")

# --- A: Threshold sweep — dynamic range based on actual EfficientAD output scale ---
# After proper 70k-step training, EfficientAD's calibrated anomaly_map values:
#   - Good image pixels: typically cluster near ~0.5–1.5
#   - Anomalous image pixels: spike to 2–10+
# We search thresholds from the 10th percentile to 99th percentile of
# all validation map values to cover the actual decision boundary.

all_val_pixel_vals = np.concatenate([
    s["norm_map"].flatten() for s in val_samples if s.get("norm_map") is not None
])
t_min = float(np.percentile(all_val_pixel_vals, 10))
t_max = float(np.percentile(all_val_pixel_vals, 99))

# Also compute individual image max to see if good images have high peaks
good_val_maxes   = [s["norm_map"].max() for s in val_samples if s.get("norm_map") is not None and not s["gt_label"]]
defect_val_maxes = [s["norm_map"].max() for s in val_samples if s.get("norm_map") is not None and s["gt_label"]]
print(f"\nVal anomaly map thresholds:")
print(f"  Sweep range: [{t_min:.4f}, {t_max:.4f}]")
if good_val_maxes:
    print(f"  Good image max scores: mean={np.mean(good_val_maxes):.4f}, max={np.max(good_val_maxes):.4f}")
if defect_val_maxes:
    print(f"  Defect image max scores: mean={np.mean(defect_val_maxes):.4f}, max={np.max(defect_val_maxes):.4f}")

# 20-point sweep, plus the midpoint 1.0 (model's calibration center)
n_steps = 20
thresholds = sorted(set(
    [round(t_min + i * (t_max - t_min) / (n_steps - 1), 4) for i in range(n_steps)]
    + [round(t_min + 0.5 * (t_max - t_min), 4)]  # midpoint
))

thresh_rows = []
best_thresh = thresholds[0]
best_thresh_score = (-1, -1, -1.0)

for t in thresholds:
    res = evaluate_dataset(val_samples, threshold=t)
    thresh_rows.append({
        "threshold": t, "recall": res["recall"], "mean_iou": res["mean_iou"],
        "fpr": res["fpr"], "tp": res["tp"], "fn": res["fn"], "good_fp": res["good_fp"],
    })
    score_key = (1 if res["recall"] >= 90.0 else 0,
                 1 if res["fpr"] < 5.0       else 0,
                 res["mean_iou"])
    if score_key > best_thresh_score:
        best_thresh_score = score_key
        best_thresh       = t

with open(OUTPUT_DIR / "threshold_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=thresh_rows[0].keys())
    writer.writeheader()
    writer.writerows(thresh_rows)

print(f"\nThreshold sweep results:")
print(f"{'Thresh':>8} {'Recall':>8} {'IoU':>8} {'FPR':>8}")
for row in thresh_rows:
    flag = " <-- BEST" if row["threshold"] == best_thresh else ""
    print(f"  {row['threshold']:>6.2f}  {row['recall']:>7.2f}%  {row['mean_iou']:>7.2f}%  {row['fpr']:>7.2f}%{flag}")

print(f"\n-> Best Threshold: {best_thresh} (Recall: {best_thresh_score[0]}, FPR OK: {best_thresh_score[1]}, IoU: {best_thresh_score[2]:.2f}%)")

# --- B: Connected Component sweep ---
component_sizes = [0, 10, 25, 50, 100, 200, 500]
cc_rows = []
best_cc = 0
best_cc_score = (-1, -1, -1.0)

for cc in component_sizes:
    res = evaluate_dataset(val_samples, threshold=best_thresh, min_component_size=cc)
    cc_rows.append({
        "min_component_size": cc, "recall": res["recall"],
        "mean_iou": res["mean_iou"], "fpr": res["fpr"],
        "tp": res["tp"], "fn": res["fn"], "good_fp": res["good_fp"],
    })
    score_key = (1 if res["recall"] >= 90.0 else 0,
                 1 if res["fpr"] < 5.0       else 0,
                 res["mean_iou"])
    if score_key > best_cc_score:
        best_cc_score = score_key
        best_cc       = cc

with open(OUTPUT_DIR / "connected_component_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cc_rows[0].keys())
    writer.writeheader()
    writer.writerows(cc_rows)

print(f"-> Best Component Size: {best_cc} (IoU: {best_cc_score[2]:.2f}%)")

# --- C: Morphology sweep (includes dilate for EfficientAD) ---
morph_ops = [None, "opening", "closing", "open_close", "dilate"]
morph_rows = []
best_morph = None
best_morph_score = (-1, -1, -1.0)

for mop in morph_ops:
    res = evaluate_dataset(val_samples, threshold=best_thresh,
                           min_component_size=best_cc, morph_op=mop)
    morph_rows.append({
        "morphology_op": str(mop), "recall": res["recall"],
        "mean_iou": res["mean_iou"], "fpr": res["fpr"],
        "tp": res["tp"], "fn": res["fn"], "good_fp": res["good_fp"],
    })
    score_key = (1 if res["recall"] >= 90.0 else 0,
                 1 if res["fpr"] < 5.0       else 0,
                 res["mean_iou"])
    if score_key > best_morph_score:
        best_morph_score = score_key
        best_morph       = mop

with open(OUTPUT_DIR / "morphology_sweep.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=morph_rows[0].keys())
    writer.writeheader()
    writer.writerows(morph_rows)

print(f"-> Best Morphology: {best_morph} (IoU: {best_morph_score[2]:.2f}%)")

# --- D: Combo search ---
step = (t_max - t_min) / (n_steps - 1)
candidate_threshs  = sorted(set([
    best_thresh,
    max(round(t_min, 4), round(best_thresh - step, 4)),
    min(round(t_max, 4), round(best_thresh + step, 4)),
]))
candidate_ccs      = sorted(set([best_cc, 0, 25, 50]))
candidate_morphs   = [None, "opening", "closing", "open_close", "dilate"]

comb_rows  = []
best_combo: dict = {"threshold": best_thresh, "min_component_size": best_cc,
                    "morphology_op": best_morph}  # safe default
best_combo_score = (-1, -1, -1.0)

for t in candidate_threshs:
    for cc in candidate_ccs:
        for mop in candidate_morphs:
            res = evaluate_dataset(val_samples, threshold=t,
                                   min_component_size=cc, morph_op=mop)
            comb_rows.append({
                "threshold": t, "min_component_size": cc, "morphology_op": str(mop),
                "recall": res["recall"], "mean_iou": res["mean_iou"], "fpr": res["fpr"],
                "tp": res["tp"], "fn": res["fn"], "good_fp": res["good_fp"],
            })
            score_key = (1 if res["recall"] >= 90.0 else 0,
                         1 if res["fpr"] < 5.0       else 0,
                         res["mean_iou"])
            if score_key > best_combo_score:
                best_combo_score = score_key
                best_combo = {"threshold": t, "min_component_size": cc,
                              "morphology_op": mop}

with open(OUTPUT_DIR / "combination_search.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comb_rows[0].keys())
    writer.writeheader()
    writer.writerows(comb_rows)

print(f"\nSelected validation config:")
print(json.dumps({k: str(v) for k, v in best_combo.items()}, indent=2))

with open(OUTPUT_DIR / "selected_validation_config.json", "w") as f:
    json.dump(best_combo, f, indent=2)

# ============================================================
# [5/5] FINAL HELD-OUT TEST EVALUATION
# ============================================================
print("\n[5/5] Final held-out test evaluation ...")

baseline_test_eval = evaluate_dataset(test_samples, threshold=0.5, use_baseline_pred=True)
opt_test_eval = evaluate_dataset(
    test_samples,
    threshold=best_combo["threshold"],
    min_component_size=best_combo["min_component_size"],
    morph_op=best_combo["morphology_op"],
)

def per_defect_summary(per_defect_dict):
    return {
        k: {
            "total":    v["total"],
            "tp":       v["tp"],
            "fn":       v["fn"],
            "recall":   (v["tp"] / v["total"]) * 100.0 if v["total"] > 0 else 0.0,
            "mean_iou": float(np.mean(v["ious"])) * 100.0 if v["ious"] else 0.0,
        }
        for k, v in per_defect_dict.items()
    }

final_summary = {
    "model":         "EfficientAD",
    "model_size":    MODEL_SIZE,
    "split_info":    split_info,
    "selected_config": best_combo,
    "baseline": {
        "recall":      baseline_test_eval["recall"],
        "mean_iou":    baseline_test_eval["mean_iou"],
        "fpr":         baseline_test_eval["fpr"],
        "tp":          baseline_test_eval["tp"],
        "fn":          baseline_test_eval["fn"],
        "good_total":  baseline_test_eval["good_total"],
        "good_fp":     baseline_test_eval["good_fp"],
        "per_defect":  per_defect_summary(baseline_test_eval["per_defect"]),
    },
    "optimized": {
        "recall":      opt_test_eval["recall"],
        "mean_iou":    opt_test_eval["mean_iou"],
        "fpr":         opt_test_eval["fpr"],
        "tp":          opt_test_eval["tp"],
        "fn":          opt_test_eval["fn"],
        "good_total":  opt_test_eval["good_total"],
        "good_fp":     opt_test_eval["good_fp"],
        "per_defect":  per_defect_summary(opt_test_eval["per_defect"]),
    },
    "improvements": {
        "recall_diff":   opt_test_eval["recall"]   - baseline_test_eval["recall"],
        "mean_iou_diff": opt_test_eval["mean_iou"] - baseline_test_eval["mean_iou"],
        "fpr_diff":      opt_test_eval["fpr"]       - baseline_test_eval["fpr"],
    },
    "prd_gate": {
        "recall_pass":  opt_test_eval["recall"]   >= 90.0,
        "mean_iou_pass": opt_test_eval["mean_iou"] >= 80.0,
        "fpr_pass":     opt_test_eval["fpr"]       < 5.0,
        "overall_pass": (opt_test_eval["recall"]   >= 90.0
                         and opt_test_eval["mean_iou"] >= 80.0
                         and opt_test_eval["fpr"]       < 5.0),
    },
    "vs_patchcore": {
        "patchcore_optimized_iou":  24.68,
        "efficientad_optimized_iou": opt_test_eval["mean_iou"],
        "iou_improvement":          opt_test_eval["mean_iou"] - 24.68,
        "patchcore_recall":         86.89,
        "efficientad_recall":       opt_test_eval["recall"],
    },
}

with open(OUTPUT_DIR / "final_heldout_test_results.json", "w") as f:
    json.dump(final_summary, f, indent=2)

# ============================================================
# PRINT FINAL RESULTS
# ============================================================
print("\n" + "=" * 70)
print("EFFICIENTAD FINAL RESULTS")
print("=" * 70)
print(f"\n{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'PatchCore':>12}")
print("-" * 60)
print(f"{'Recall':<20} {baseline_test_eval['recall']:>11.2f}% {opt_test_eval['recall']:>11.2f}% {'86.89':>11}%")
print(f"{'Mean IoU':<20} {baseline_test_eval['mean_iou']:>11.2f}% {opt_test_eval['mean_iou']:>11.2f}% {'24.68':>11}%")
print(f"{'FPR':<20} {baseline_test_eval['fpr']:>11.2f}% {opt_test_eval['fpr']:>11.2f}% {'4.76':>11}%")

print("\nPRD GATE STATUS:")
gate = final_summary["prd_gate"]
print(f"  Recall >= 90%:   {'[PASS]' if gate['recall_pass']   else '[FAIL]'} ({opt_test_eval['recall']:.2f}%)")
print(f"  Mean IoU >= 80%: {'[PASS]' if gate['mean_iou_pass'] else '[FAIL]'} ({opt_test_eval['mean_iou']:.2f}%)")
print(f"  FPR < 5%:        {'[PASS]' if gate['fpr_pass']      else '[FAIL]'} ({opt_test_eval['fpr']:.2f}%)")
print(f"  OVERALL:         {'[ALL PASS]' if gate['overall_pass'] else '[NOT YET]'}")

print("\nPER-DEFECT BREAKDOWN (Optimized, Test Set):")
print(f"{'Category':<25} {'Recall':>8} {'Mean IoU':>10}")
print("-" * 46)
for cat, d in final_summary["optimized"]["per_defect"].items():
    print(f"  {cat:<23} {d['recall']:>7.2f}%  {d['mean_iou']:>8.2f}%")

print(f"\nResults saved to: {OUTPUT_DIR}")
print("=" * 70)

# ============================================================
# VISUALIZATIONS  (5-panel: original | anomaly map | GT | baseline | optimized)
# ============================================================
print("\nGenerating visualizations ...")

categories = ["manipulated_front", "scratch_head", "scratch_neck",
              "thread_side", "thread_top", "good"]

for cat in categories:
    cat_samples = [s for s in test_samples if s["defect_type"] == cat]
    if not cat_samples:
        continue

    if cat == "good":
        fp_samples = [
            s for s in cat_samples
            if calculate_sample_metrics(
                s["gt_mask"],
                post_process_map(s.get("norm_map"), best_combo["threshold"],
                                 best_combo["min_component_size"],
                                 best_combo["morphology_op"]),
                False
            )[2] > 0
        ]
        selected_vis = fp_samples[:2] if fp_samples else cat_samples[:2]
    else:
        selected_vis = cat_samples[:1]

    for idx, s in enumerate(selected_vis):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        img = s["image"]
        if img is not None:
            img_disp = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            axes[0].imshow(img_disp)
        axes[0].set_title("1. Original Image"); axes[0].axis("off")

        amap = s.get("norm_map")
        if amap is not None:
            im = axes[1].imshow(amap, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title("2. EfficientAD Anomaly Map"); axes[1].axis("off")

        gt_m = s["gt_mask"]
        axes[2].imshow(gt_m if gt_m is not None else np.zeros((256, 256)), cmap="gray")
        axes[2].set_title("3. Ground-Truth Mask"); axes[2].axis("off")

        base_p = s["baseline_pred_mask"]
        axes[3].imshow(base_p if base_p is not None else np.zeros((256, 256)), cmap="gray")
        axes[3].set_title("4. Baseline Mask"); axes[3].axis("off")

        opt_p = post_process_map(s.get("norm_map"), best_combo["threshold"],
                                 best_combo["min_component_size"],
                                 best_combo["morphology_op"])
        axes[4].imshow(opt_p if opt_p is not None else np.zeros((256, 256)), cmap="gray")
        axes[4].set_title("5. Optimized Mask"); axes[4].axis("off")

        # Compute IoU for title
        _, _, _, iou_val = calculate_sample_metrics(gt_m, opt_p, s["gt_label"])
        iou_str = f"IoU: {iou_val*100:.1f}%" if iou_val is not None else ""
        plt.suptitle(f"EfficientAD — {cat} (sample {idx+1}) {iou_str}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()

        fname = f"vis_{cat}_sample{idx+1}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname}")

print("\nEfficientAD Evaluation Pipeline Complete!")
