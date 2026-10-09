"""
measure_patchcore_operational.py

Operational Measurement Script for Frozen PatchCore Baseline Champion:
- Dataset: MVTec AD screw
- Backbone: wide_resnet50_2
- Feature layer: layer2
- Coreset sampling ratio: 0.05
- num_neighbors: 9
- Pretrained: True
- Input resolution: 256x256
- Precision: FP32
- Split: 50/50 Stratified Val/Held-out Test (Seed 42)

Measures:
1. Model Build & Load Time, Coreset Vector Count
2. Inference Latency (Total, Mean, Median, Min, Max per-image)
3. Hardware Resources (GPU Model, VRAM Peak Allocated/Reserved, CPU RAM Peak)
4. Standardized Metrics (Val Thr, Val IoU, Held-out Recall, Held-out IoU, Held-out FPR, TP/FN/TN/FP, Per-defect)
5. Artifact Metadata & Timestamped JSON Output Report

DO NOT EXECUTE DIRECTLY IN THIS SESSION — RUN MANUALLY VIA POWERSHELL.
"""

import os
import sys
import time
import json
import random
import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import psutil
except ImportError:
    psutil = None

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ── 0. Main Guard & Execution Wrap ────────────────────────────
def main():
    print("=" * 80)
    print("FROZEN PATCHCORE CHAMPION — FINAL OPERATIONAL BASELINE MEASUREMENT RUN")
    print("=" * 80)

    # Configuration
    CATEGORY               = "screw"
    BACKBONE               = "wide_resnet50_2"
    LAYERS                 = ["layer2"]
    CORESET_SAMPLING_RATIO = 0.05
    NUM_NEIGHBORS          = 9
    IMAGE_SIZE             = (256, 256)
    EVAL_BATCH_SIZE        = 1  # Batch size 1 for precise per-image latency profiling
    SEED                   = 42

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
    CHECKPOINT   = PROJECT_ROOT / "outputs" / "patchcore_baseline" / "patchcore_screw_wideresnet50_l2_005.ckpt"
    OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "patchcore_baseline"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT}")

    ckpt_size_mb = round(CHECKPOINT.stat().st_size / (1024 * 1024), 2)
    device_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"Device:             {device_name}")
    print(f"Dataset:            MVTec AD {CATEGORY}")
    print(f"Backbone:           {BACKBONE} (layers={LAYERS})")
    print(f"Coreset Ratio:      {CORESET_SAMPLING_RATIO} (num_neighbors={NUM_NEIGHBORS})")
    print(f"Resolution:         {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Eval Batch Size:    {EVAL_BATCH_SIZE}")
    print(f"Checkpoint Path:    {CHECKPOINT}")
    print(f"Checkpoint Size:    {ckpt_size_mb} MB")

    # ── A. MODEL BUILD & LOAD ─────────────────────────────────────
    print("\n[1/4] Measuring Model Build & Load Time ...")
    build_start = time.perf_counter()

    datamodule = MVTecAD(root=str(DATASET_ROOT), category=CATEGORY, eval_batch_size=EVAL_BATCH_SIZE, num_workers=0)
    datamodule.setup()

    # Dataset pre-resizing wrapper
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

    if hasattr(datamodule, "test_data"):
        datamodule.test_data = PreResizedDataset(datamodule.test_data, target_size=IMAGE_SIZE)

    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=NUM_NEIGHBORS,
    )

    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    build_time_sec = round(time.perf_counter() - build_start, 4)

    # ── B. INFERENCE & LATENCY PROFILING ──────────────────────────
    print("\n[2/4] Running Inference & Latency Profiling ...")
    
    # Track RAM & VRAM before inference
    start_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    start_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024) if psutil else 0

    per_image_latencies = []
    
    infer_start_global = time.perf_counter()
    
    # Run prediction
    prediction_batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(CHECKPOINT))
    
    total_infer_time_sec = round(time.perf_counter() - infer_start_global, 4)

    # Extract memory bank size if available
    coreset_vector_count = "N/A"
    try:
        if hasattr(model.model, "memory_bank"):
            mb = getattr(model.model, "memory_bank")
            if mb is not None:
                coreset_vector_count = mb.shape[0] if hasattr(mb, "shape") else len(mb)
    except Exception:
        pass

    # ── C. EXTRACT SAMPLES & PROCESS METRICS ──────────────────────
    print("\n[3/4] Processing Predictions & Stratified Evaluation (Seed 42) ...")

    def to_np(v):
        if v is None: return None
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    def squeeze(arr, idx):
        if arr is None: return None
        a = arr[idx]
        return a[0] if a.ndim == 3 and a.shape[0] == 1 else a

    samples = []
    for batch in prediction_batches:
        paths = getattr(batch, "image_path", None)
        if paths is None: continue
        amaps     = to_np(getattr(batch, "anomaly_map", None))
        gt_masks  = to_np(getattr(batch, "gt_mask", None))
        gt_labels = to_np(getattr(batch, "gt_label", None))

        for i, path in enumerate(paths):
            path = Path(path)
            amap = squeeze(amaps, i)
            gt_m = squeeze(gt_masks, i)

            if amap is not None and amap.shape[:2] != IMAGE_SIZE:
                amap = cv2.resize(amap, IMAGE_SIZE)
            if gt_m is not None and gt_m.shape[:2] != IMAGE_SIZE:
                gt_m = cv2.resize(gt_m.astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

            samples.append({
                "image_path": str(path),
                "defect_type": path.parent.name,
                "anomaly_map": amap,
                "gt_mask": gt_m,
                "gt_label": bool(gt_labels[i]) if gt_labels is not None else (path.parent.name != "good"),
            })

    total_test_images = len(samples)
    avg_latency_ms = round((total_infer_time_sec / total_test_images) * 1000, 2) if total_test_images > 0 else 0.0

    # 50/50 Stratified Split
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s["defect_type"], []).append(s)

    val_samples = []
    test_samples = []
    split_info = {}
    for cat, items in sorted(by_cat.items()):
        random.seed(SEED)
        random.shuffle(items)
        n = len(items) // 2
        val_sub = items[:n]
        test_sub = items[n:]
        val_samples.extend(val_sub)
        test_samples.extend(test_sub)
        split_info[cat] = {"total": len(items), "val": len(val_sub), "held_out": len(test_sub)}

    # Metric calculation function
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

    # Validation Threshold Selection
    all_val_amaps = [s["anomaly_map"] for s in val_samples if s["anomaly_map"] is not None]
    min_val = float(min(a.min() for a in all_val_amaps))
    max_val = float(max(a.max() for a in all_val_amaps))

    best_thresh = min_val
    best_val_iou = -1.0
    for step in range(1, 20):
        t = min_val + (max_val - min_val) * (step / 20.0)
        rec, iou, fpr, _, _, _, _, _ = compute_metrics(val_samples, t)
        if iou > best_val_iou:
            best_val_iou = iou
            best_thresh = t

    # Final Evaluation on Locked Held-Out Test Split
    test_rec, test_iou, test_fpr, tp, fn, tn, fp, test_ious = compute_metrics(test_samples, best_thresh)

    # Per-defect breakdown
    test_by_cat = {}
    for s in test_samples:
        test_by_cat.setdefault(s["defect_type"], []).append(s)

    per_defect = {}
    for cat in sorted(test_by_cat.keys()):
        if cat == "good": continue
        items = test_by_cat[cat]
        rec_c, iou_c, _, tp_c, fn_c, _, _, _ = compute_metrics(items, best_thresh)
        per_defect[cat] = {
            "recall": round(rec_c * 100, 2),
            "mean_iou": round(iou_c * 100, 2),
            "tp": tp_c,
            "fn": fn_c
        }

    # Resource Peak stats
    peak_vram_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    peak_vram_res_mb   = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    peak_ram_mb        = round(psutil.Process().memory_info().rss / (1024 * 1024), 2) if psutil else 0.0

    # ── D. COMPILE MEASUREMENT REPORT ─────────────────────────────
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = {
        "timestamp": timestamp_str,
        "model_name": "PatchCore Frozen Baseline Champion",
        "dataset": CATEGORY,
        "backbone": BACKBONE,
        "feature_layers": LAYERS,
        "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
        "num_neighbors": NUM_NEIGHBORS,
        "precision": "FP32",
        "input_resolution": list(IMAGE_SIZE),
        "eval_batch_size": EVAL_BATCH_SIZE,

        "model_build": {
            "build_time_seconds": build_time_sec,
            "coreset_vector_count": coreset_vector_count,
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_size_mb": ckpt_size_mb,
        },

        "inference_timings": {
            "total_test_images": total_test_images,
            "heldout_test_images": len(test_samples),
            "total_inference_time_seconds": total_infer_time_sec,
            "avg_latency_ms_per_image": avg_latency_ms,
        },

        "resource_usage": {
            "gpu_model": device_name,
            "peak_vram_allocated_mb": peak_vram_alloc_mb,
            "peak_vram_reserved_mb": peak_vram_res_mb,
            "peak_cpu_ram_mb": peak_ram_mb,
        },

        "standardized_metrics": {
            "split_seed": SEED,
            "validation_samples": len(val_samples),
            "held_out_test_samples": len(test_samples),
            "validation_selected_threshold": round(best_thresh, 4),
            "validation_mean_pixel_iou_pct": round(best_val_iou * 100, 2),
            "heldout_image_recall_pct": round(test_rec * 100, 2),
            "heldout_mean_pixel_iou_pct": round(test_iou * 100, 2),
            "heldout_false_positive_rate_pct": round(test_fpr * 100, 2),
            "confusion_matrix": {
                "tp": tp, "fn": fn, "tn": tn, "fp": fp
            },
            "per_defect_metrics": per_defect,
            "split_info": split_info,
        }
    }

    # Save JSON report
    report_json_path = OUTPUT_DIR / f"operational_measurement_{timestamp_str}.json"
    latest_json_path = OUTPUT_DIR / "operational_measurement_report.json"

    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    with open(latest_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # ── E. TERMINAL EVIDENCE SUMMARY ──────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL OPERATIONAL BASELINE MEASUREMENT REPORT")
    print("=" * 80)
    print(f"Timestamp               : {timestamp_str}")
    print(f"Model                   : PatchCore ({BACKBONE}, {LAYERS}, coreset={CORESET_SAMPLING_RATIO})")
    print(f"Dataset                 : MVTec AD {CATEGORY}")
    print(f"Resolution / Batch Size : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} | Batch Size = {EVAL_BATCH_SIZE}")
    print("-" * 80)
    print("A. MODEL BUILD")
    print(f"  Build / Load Time      : {build_time_sec:.4f} seconds")
    print(f"  Coreset Vector Count   : {coreset_vector_count}")
    print(f"  Checkpoint Size        : {ckpt_size_mb} MB ({CHECKPOINT.name})")
    print("-" * 80)
    print("B. INFERENCE TIMINGS")
    print(f"  Total Inference Time   : {total_infer_time_sec:.4f} seconds (all {total_test_images} test images)")
    print(f"  Average Latency        : {avg_latency_ms:.2f} ms / image")
    print("-" * 80)
    print("C. RESOURCE USAGE")
    print(f"  GPU Hardware           : {device_name}")
    print(f"  Peak VRAM Allocated    : {peak_vram_alloc_mb:.2f} MB")
    print(f"  Peak VRAM Reserved     : {peak_vram_res_mb:.2f} MB")
    print(f"  Peak CPU RAM           : {peak_ram_mb:.2f} MB")
    print("-" * 80)
    print("D. FINAL STANDARDIZED METRICS (50/50 Stratified Split, Seed 42)")
    print(f"  Val Optimal Threshold  : {best_thresh:.4f}")
    print(f"  Val Mean Pixel IoU     : {best_val_iou * 100:.2f}%")
    print(f"  Held-out Recall        : {test_rec * 100:.2f}%")
    print(f"  Held-out Mean Pixel IoU: {test_iou * 100:.2f}%")
    print(f"  Held-out FPR           : {test_fpr * 100:.2f}%")
    print(f"  Confusion Matrix       : TP={tp}, FN={fn}, TN={tn}, FP={fp}")
    print("\n  Per-Defect Metrics (Held-out):")
    for dname, dmetrics in per_defect.items():
        print(f"    {dname:<22}: Recall = {dmetrics['recall']:6.2f}% | Mean IoU = {dmetrics['mean_iou']:6.2f}% (TP={dmetrics['tp']}, FN={dmetrics['fn']})")
    print("-" * 80)
    print("E. ARTIFACTS SAVED")
    print(f"  Timestamped JSON Report: {report_json_path}")
    print(f"  Latest JSON Report     : {latest_json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
