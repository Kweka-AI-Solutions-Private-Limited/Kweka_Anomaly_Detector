"""
evaluate_patchcore_l1l2.py

Controlled evaluation script for PatchCore Layer1 + Layer2 Experiment.
Guarantees 256x256 test image preprocessing BEFORE entering backbone.
Applies MultiLayerPooler for layer1 (stride=2 -> 32x32) and layer2 (stride=1 -> 32x32) spatial feature alignment.
Selects optimal threshold on validation split ONLY, then locks threshold to evaluate held-out test split.
Calculates Recall, Mean Pixel IoU, FPR, raw confusion counts, per-defect metrics, and saves visualizations.
"""

from pathlib import Path
import json
import csv
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ── Configuration ─────────────────────────────────────────────
CATEGORY               = "screw"
BACKBONE               = "wide_resnet50_2"
LAYERS                 = ["layer1", "layer2"]
CORESET_SAMPLING_RATIO = 0.05
NUM_NEIGHBORS          = 9
IMAGE_SIZE             = (256, 256)
SEED                   = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATASET_ROOT  = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
CHECKPOINT    = PROJECT_ROOT / "outputs" / "patchcore_baseline" / "patchcore_screw_wideresnet50_l1l2_005.ckpt"
RUN_INFO_FILE = PROJECT_ROOT / "outputs" / "patchcore_baseline" / "baseline_run_screw_wideresnet50_l1l2_005.json"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "patchcore_baseline" / "screw_wideresnet50_l1l2_005_localization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MultiLayerPooler for Layer1+Layer2 Spatial Alignment ─────
class MultiLayerPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_l1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_l2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        if feature_tensor.shape[-1] == 64:
            return self.pool_l1(feature_tensor)
        else:
            return self.pool_l2(feature_tensor)

# ── Dataset Pre-resizing Wrapper (Guarantees 256x256 Preprocessing) ────
class PreResizedDataset(Dataset):
    def __init__(self, dataset, target_size=(256, 256)):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if hasattr(item, "image") and isinstance(item.image, torch.Tensor):
            if item.image.shape[-2:] != self.target_size:
                item.image = F.interpolate(
                    item.image.unsqueeze(0),
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)

print("=" * 70)
print("PATCHCORE LAYER1+LAYER2 CONTROLLED EVALUATION")
print("=" * 70)
print(f"Checkpoint: {CHECKPOINT}")
if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Run run_patchcore_l1l2.py first.\nMissing: {CHECKPOINT}")

# Load training run metadata
build_time = 0.0
inference_time = 0.0
num_train_good = 320
if RUN_INFO_FILE.exists():
    with open(RUN_INFO_FILE, "r") as f:
        run_meta = json.load(f)
        build_time = run_meta.get("build_time_seconds", 0.0)
        inference_time = run_meta.get("inference_time_seconds", 0.0)
        num_train_good = run_meta.get("number_of_training_good_images", 320)

# ── 1. Inference ──────────────────────────────────────────────
print("\n[1/5] Running test inference with 256x256 dataset preprocessing ...")
datamodule = MVTecAD(root=str(DATASET_ROOT), category=CATEGORY, eval_batch_size=4, num_workers=0)
datamodule.setup()

# Pre-resize test dataset before passing to feature extractor
if hasattr(datamodule, "test_data"):
    datamodule.test_data = PreResizedDataset(datamodule.test_data, target_size=IMAGE_SIZE)

model = Patchcore(
    backbone=BACKBONE,
    layers=LAYERS,
    pre_trained=True,
    coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
    num_neighbors=NUM_NEIGHBORS,
)

# Apply MultiLayerPooler for layer1+layer2 spatial feature map alignment
model.model.feature_pooler = MultiLayerPooler()

engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)
prediction_batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(CHECKPOINT))

if not prediction_batches:
    raise RuntimeError("engine.predict() returned no prediction batches.")
print(f"Retrieved {len(prediction_batches)} prediction batch(es).")

# ── 2. Extract & Preprocess Samples ───────────────────────────
print("\n[2/5] Extracting & standardizing test samples ...")

def to_np(v):
    if v is None: return None
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

samples = []
for batch in prediction_batches:
    paths = getattr(batch, "image_path", None)
    if paths is None: continue
    images     = to_np(getattr(batch, "image",       None))
    amaps      = to_np(getattr(batch, "anomaly_map", None))
    pred_masks = to_np(getattr(batch, "pred_mask",   None))
    gt_masks   = to_np(getattr(batch, "gt_mask",     None))
    gt_labels  = to_np(getattr(batch, "gt_label",    None))

    for i, path in enumerate(paths):
        path = Path(path)
        def squeeze(arr, idx):
            if arr is None: return None
            a = arr[idx]
            return a[0] if a.ndim == 3 and a.shape[0] == 1 else a
        
        img = images[i] if images is not None else None
        if img is not None and img.ndim == 3 and img.shape[0] in [1,3]:
            img = np.transpose(img, (1,2,0))
        
        amap   = squeeze(amaps, i)
        pred_m = squeeze(pred_masks, i)
        gt_m   = squeeze(gt_masks, i)

        if img is not None and img.shape[:2] != IMAGE_SIZE:
            img = cv2.resize(img, IMAGE_SIZE)
        if amap is not None and amap.shape[:2] != IMAGE_SIZE:
            amap = cv2.resize(amap, IMAGE_SIZE)
        if pred_m is not None and pred_m.shape[:2] != IMAGE_SIZE:
            pred_m = cv2.resize(pred_m.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        if gt_m is not None and gt_m.shape[:2] != IMAGE_SIZE:
            gt_m = cv2.resize(gt_m.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        samples.append({
            "image_path":  str(path),
            "defect_type": path.parent.name,
            "image":       img,
            "anomaly_map": amap,
            "pred_mask":   pred_m,
            "gt_mask":     gt_m,
            "gt_label":    bool(gt_labels[i]) if gt_labels is not None else (path.parent.name != "good"),
        })

print(f"Total test samples extracted: {len(samples)}")
for cat, cnt in sorted({s["defect_type"]: 0 for s in samples}.items()):
    cnt = sum(1 for s in samples if s["defect_type"] == cat)
    print(f"  {cat:<25}: {cnt}")

# ── 3. Stratified Split ───────────────────────────────────────
print("\n[3/5] Performing 50/50 stratified validation/test split (seed=42) ...")
by_cat = {}
for s in samples:
    by_cat.setdefault(s["defect_type"], []).append(s)

val_samples = []; test_samples = []
for cat, items in sorted(by_cat.items()):
    random.seed(SEED); random.shuffle(items)
    n = len(items) // 2
    val_samples.extend(items[:n])
    test_samples.extend(items[n:])

print(f"Validation split: {len(val_samples)} samples | Held-out Test split: {len(test_samples)} samples")

# ── 4. Validation Sweep (Threshold selection ONLY on val) ─────
print("\n[4/5] Running threshold sweep on validation split ONLY ...")

all_val_amaps = [s["anomaly_map"] for s in val_samples if s["anomaly_map"] is not None]
min_val = float(min(a.min() for a in all_val_amaps))
max_val = float(max(a.max() for a in all_val_amaps))
print(f"Validation anomaly map value range: [{min_val:.4f}, {max_val:.4f}]")

def compute_metrics(sample_list, threshold):
    tp, fp, tn, fn = 0, 0, 0, 0
    ious = []

    for s in sample_list:
        is_defective = s["gt_label"]
        amap = s["anomaly_map"]
        if amap is None: continue

        pred_binary = (amap >= threshold).astype(np.uint8)
        img_has_anomaly = bool(pred_binary.max() > 0)

        if is_defective:
            if img_has_anomaly: tp += 1
            else: fn += 1

            if s["gt_mask"] is not None:
                gt_binary = (s["gt_mask"] > 0).astype(np.uint8)
                intersection = np.logical_and(gt_binary, pred_binary).sum()
                union        = np.logical_or(gt_binary, pred_binary).sum()
                iou          = intersection / union if union > 0 else 0.0
                ious.append(iou)
        else:
            if img_has_anomaly: fp += 1
            else: tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return recall, mean_iou, fpr, tp, fn, tn, fp, ious

best_thresh = min_val
best_val_iou = -1.0

print(f"  {'Thresh':>8} {'Recall':>10} {'Mean IoU':>12} {'FPR':>10}")
print("-" * 45)
for step in range(1, 20):
    t = min_val + (max_val - min_val) * (step / 20.0)
    rec, iou, fpr, _, _, _, _, _ = compute_metrics(val_samples, t)
    mark = ""
    if iou > best_val_iou:
        best_val_iou = iou
        best_thresh = t
        mark = " <-- BEST"
    print(f"  {t:8.4f} {rec*100:9.2f}% {iou*100:11.2f}% {fpr*100:9.2f}%{mark}")

print(f"\nLocked optimal threshold from validation set: {best_thresh:.4f} (Val Mean IoU: {best_val_iou*100:.2f}%)")

# ── 5. Final Evaluation on Held-Out Test Split ────────────────
print("\n[5/5] Evaluating locked configuration on HELD-OUT TEST split ...")

test_rec, test_iou, test_fpr, tp, fn, tn, fp, test_ious = compute_metrics(test_samples, best_thresh)

total_good = tn + fp
good_fp = fp

# Per-defect type metrics
per_defect = {}
test_by_cat = {}
for s in test_samples:
    test_by_cat.setdefault(s["defect_type"], []).append(s)

print("\nPER-DEFECT METRICS (HELD-OUT TEST SET):")
print(f"  {'Category':<25} {'Recall':>10} {'Mean IoU':>12}")
print("-" * 52)

for cat in sorted(test_by_cat.keys()):
    items = test_by_cat[cat]
    rec_c, iou_c, fpr_c, tp_c, fn_c, tn_c, fp_c, _ = compute_metrics(items, best_thresh)
    if cat != "good":
        per_defect[cat] = {
            "recall": round(rec_c * 100, 2),
            "mean_iou": round(iou_c * 100, 2),
            "tp": tp_c,
            "fn": fn_c,
        }
        print(f"  {cat:<25} {rec_c*100:9.2f}% {iou_c*100:11.2f}%")

print("\n" + "=" * 70)
print("PATCHCORE LAYER1+LAYER2 FINAL RESULTS")
print("=" * 70)
print(f"RAW CONFUSION COUNTS:")
print(f"  True Positives (TP) : {tp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Negatives (TN) : {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  Total GOOD Test     : {total_good}")
print(f"  GOOD False Positives: {good_fp}")
print("-" * 70)
print(f"IMAGE RECALL         : {test_rec * 100:.2f}%  (Formula: TP / (TP + FN))")
print(f"MEAN PIXEL IOU       : {test_iou * 100:.2f}%  (Formula: mean of Intersection / Union)")
print(f"FALSE POSITIVE RATE  : {test_fpr * 100:.2f}%  (Formula: FP / (FP + TN))")
print("=" * 70)

# ── 6. Visualizations ─────────────────────────────────────────
print("\nGenerating representative visualization figures ...")

categories_to_vis = ["good", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]
if good_fp > 0:
    categories_to_vis.append("good_fp")

for cat in categories_to_vis:
    if cat == "good_fp":
        candidates = [s for s in test_samples if s["defect_type"] == "good" and (s["anomaly_map"] >= best_thresh).max()]
    else:
        candidates = [s for s in test_samples if s["defect_type"] == cat]
    
    if not candidates: continue
    s = candidates[0]

    img = s["image"]
    amap = s["anomaly_map"]
    gt_m = s["gt_mask"]
    pred_binary = (amap >= best_thresh).astype(np.uint8) if amap is not None else np.zeros(IMAGE_SIZE)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Panel 1: Original Image
    if img is not None:
        if img.max() <= 1.0: img_show = (img * 255).astype(np.uint8)
        else: img_show = img.astype(np.uint8)
        axes[0].imshow(img_show)
    axes[0].set_title(f"Original ({cat})")
    axes[0].axis("off")

    # Panel 2: Ground Truth
    if gt_m is not None:
        axes[1].imshow(gt_m, cmap="gray")
    else:
        axes[1].imshow(np.zeros(IMAGE_SIZE), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Panel 3: Anomaly Map
    if amap is not None:
        im = axes[2].imshow(amap, cmap="jet")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Anomaly Map")
    axes[2].axis("off")

    # Panel 4: Predicted Mask
    axes[3].imshow(pred_binary, cmap="gray")
    axes[3].set_title(f"Pred Mask (T={best_thresh:.3f})")
    axes[3].axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"vis_{cat}_sample.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")

# ── 7. Save Evaluation Metadata JSON ──────────────────────────
eval_results_json = {
    "model": "PatchCore",
    "category": CATEGORY,
    "backbone": BACKBONE,
    "layers": LAYERS,
    "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
    "num_neighbors": NUM_NEIGHBORS,
    "image_size": list(IMAGE_SIZE),
    "embedding_channels": 1536,
    "spatial_reduction": "MultiLayerPooler (layer1: 64x64->32x32, layer2: 32x32->32x32)",
    "number_of_training_good_images": num_train_good,
    "build_time_seconds": build_time,
    "inference_time_seconds": inference_time,
    "selected_threshold": round(float(best_thresh), 4),
    "selected_post_processing_configuration": {
        "threshold": round(float(best_thresh), 4),
        "split": "50/50 stratified validation threshold selection",
    },
    "raw_confusion_counts": {
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
        "total_good_test_images": total_good,
        "good_false_positives": good_fp,
    },
    "image_recall": round(float(test_rec * 100), 2),
    "mean_pixel_iou": round(float(test_iou * 100), 2),
    "false_positive_rate": round(float(test_fpr * 100), 2),
    "per_defect_results": per_defect,
}

eval_json_path = OUTPUT_DIR / "evaluation_results.json"
with open(eval_json_path, "w") as f:
    json.dump(eval_results_json, f, indent=2)

print(f"\nSaved evaluation metadata JSON: {eval_json_path}")
print("\n" + "=" * 70)
print("PATCHCORE LAYER1+LAYER2 EVALUATION SCRIPT COMPLETE")
print("=" * 70)
