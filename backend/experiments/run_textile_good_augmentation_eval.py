import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as T

from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (256, 256)

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2"]
CORESET_RATIO = 0.05
NUM_NEIGHBORS = 9
SEED = 42

# Baseline metrics for comparison
BASELINE_METRICS = {
    "calibration_method": "250-GOOD Memory / 30-GOOD Independent Calibration Split",
    "memory_image_count": 250,
    "locked_threshold": 21.3272,
    "true_positives": 49,
    "true_negatives": 2,
    "false_positives": 8,
    "false_negatives": 1,
    "false_positive_rate_percent": 80.0,
    "recall_sensitivity_percent": 98.0,
    "overall_accuracy_percent": 85.0,
    "runtime_seconds": 470.13
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Define Mild, Physically Plausible Augmentation Pipeline
mild_augmentation_transform = T.Compose([
    T.RandomRotation(degrees=(-5, 5)),
    T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    T.ColorJitter(brightness=0.05, contrast=0.05),
])


class AugmentableFolderDataset(Dataset):
    """
    Dataset that loads original images, and if augment=True, generates a mild augmented pair.
    """
    def __init__(self, image_paths, augment=False, target_size=DEFAULT_TARGET_SIZE, seed=SEED):
        self.image_paths = [Path(p) for p in image_paths]
        self.augment = augment
        self.target_size = target_size
        self.seed = seed

    def __len__(self):
        # If augment=True, return 2x samples (original + 1 mild augmented version per image)
        return len(self.image_paths) * (2 if self.augment else 1)

    def __getitem__(self, idx):
        orig_idx = idx % len(self.image_paths)
        is_augmented_sample = (idx >= len(self.image_paths))

        path = self.image_paths[orig_idx]
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)

        if is_augmented_sample:
            # Deterministic per-index augmentation seed for reproducibility
            torch.manual_seed(self.seed + idx)
            img_tensor = T.functional.to_image(img_resized)
            img_aug = mild_augmentation_transform(img_tensor)
            tensor = T.functional.to_dtype(img_aug, dtype=torch.float32, scale=True)
            path_str = f"{str(path)}#aug"
        else:
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            path_str = str(path)

        return {
            "image": tensor,
            "image_path": path_str,
        }


def generic_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    return ImageBatch(image=images, image_path=image_paths)


def run_experiment():
    start_total_time = time.time()
    set_seed(SEED)

    base_dir = Path("c:/dev/Anomaly_Detector/backend").resolve()
    data_dir = base_dir / "data" / "textile"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    output_dir = base_dir / "outputs" / "generic_patchcore" / "textile" / "good_augmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all 280 training GOOD images
    all_train = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"Discovered {len(all_train)} GOOD training reference images in {train_dir}")

    # Deterministic split: 250 memory / 30 untouched calibration
    rng = random.Random(SEED)
    shuffled_train = list(all_train)
    rng.shuffle(shuffled_train)

    train_memory_250 = shuffled_train[:250]
    calib_good_30 = shuffled_train[250:]

    print(f"Split GOOD training data: {len(train_memory_250)} Memory Images (to be augmented) + {len(calib_good_30)} Independent Untouched Calibration Images")

    # 2. Discover test set (10 Known GOOD + 50 Defective)
    all_test = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    known_good_paths = sorted([p for p in all_test if "copy 2" in p.name])
    defective_paths = sorted([p for p in all_test if p not in known_good_paths])

    print(f"Discovered {len(all_test)} test images ({len(known_good_paths)} Known GOOD, {len(defective_paths)} Defective)")

    # -------------------------------------------------------------
    # PHASE 1: CALIBRATION ON AUGMENTED MEMORY SET (250 x 2 = 500 images)
    # -------------------------------------------------------------
    print("\n[Phase 1/2] Building Calibration Model on 500 Augmented GOOD Memory Images...")
    start_calib_time = time.time()

    model_calib = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine_calib = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    calib_memory_dataset = AugmentableFolderDataset(train_memory_250, augment=True)
    print(f"  Calib Memory Dataset size: {len(calib_memory_dataset)} samples")
    tr_loader_500 = DataLoader(calib_memory_dataset, batch_size=4, shuffle=False, collate_fn=generic_collate_fn)

    engine_calib.fit(model=model_calib, train_dataloaders=tr_loader_500)
    model_calib.post_processor = None

    print("Scoring 30 Untouched GOOD Calibration Images...")
    calib_untouched_dataset = AugmentableFolderDataset(calib_good_30, augment=False)
    calib_loader_30 = DataLoader(calib_untouched_dataset, batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    calib_preds = engine_calib.predict(model=model_calib, dataloaders=calib_loader_30)

    calib_scores = [float(b.pred_score[0]) for b in calib_preds]
    calib_duration = time.time() - start_calib_time

    calib_arr = np.array(calib_scores)
    calib_stats = {
        "sample_size": len(calib_scores),
        "mean": float(np.mean(calib_arr)),
        "std": float(np.std(calib_arr)),
        "min": float(np.min(calib_arr)),
        "max": float(np.max(calib_arr)),
        "p90": float(np.percentile(calib_arr, 90)),
        "p95": float(np.percentile(calib_arr, 95)),
        "p97_5": float(np.percentile(calib_arr, 97.5)),
        "p99": float(np.percentile(calib_arr, 99)),
    }

    locked_threshold_aug = calib_stats["p95"]
    print(f"Calibration Complete in {calib_duration:.2f}s!")
    print(f"  Mean Raw Dist: {calib_stats['mean']:.4f} +/- {calib_stats['std']:.4f}")
    print(f"  Locked Augmented 95th Percentile Threshold (theta_aug_95): {locked_threshold_aug:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: EVALUATION OF FINAL MODEL (280 x 2 = 560 images) ON UNTOUCHED TEST SET
    # -------------------------------------------------------------
    print("\n[Phase 2/2] Building Final Production Model on ALL 560 Augmented GOOD Reference Images...")
    start_eval_time = time.time()

    model_final = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine_final = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    full_tr_dataset = AugmentableFolderDataset(all_train, augment=True)
    print(f"  Final Reference Dataset size: {len(full_tr_dataset)} samples")
    full_tr_loader = DataLoader(full_tr_dataset, batch_size=4, shuffle=False, collate_fn=generic_collate_fn)

    engine_final.fit(model=model_final, train_dataloaders=full_tr_loader)
    model_final.post_processor = None

    print("Evaluating 60 Untouched Test Images (10 Known GOOD + 50 Defective)...")
    test_dataset = AugmentableFolderDataset(all_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=generic_collate_fn)

    test_preds = engine_final.predict(model=model_final, dataloaders=test_loader)
    eval_duration = time.time() - start_eval_time

    results_per_image = []
    tp, fp, tn, fn = 0, 0, 0, 0
    known_good_results = []
    defective_results = []

    for b in test_preds:
        p_str = b.image_path[0].split("#")[0]
        p_path = Path(p_str)
        raw_dist = float(b.pred_score[0])
        calib_score = raw_dist / locked_threshold_aug
        pred_status = "ANOMALOUS" if raw_dist >= locked_threshold_aug else "NORMAL"
        is_known_good = "copy 2" in p_path.name
        gt_status = "NORMAL" if is_known_good else "ANOMALOUS"

        is_correct = (pred_status == gt_status)

        if gt_status == "ANOMALOUS" and pred_status == "ANOMALOUS":
            tp += 1
        elif gt_status == "NORMAL" and pred_status == "ANOMALOUS":
            fp += 1
        elif gt_status == "NORMAL" and pred_status == "NORMAL":
            tn += 1
        elif gt_status == "ANOMALOUS" and pred_status == "NORMAL":
            fn += 1

        res_item = {
            "filename": p_path.name,
            "relative_path": str(p_path.relative_to(data_dir)),
            "ground_truth": gt_status,
            "prediction": pred_status,
            "raw_anomaly_distance": round(raw_dist, 4),
            "calibrated_score": round(calib_score, 4),
            "is_correct": is_correct
        }
        results_per_image.append(res_item)
        if is_known_good:
            known_good_results.append(res_item)
        else:
            defective_results.append(res_item)

    fpr = fp / len(known_good_paths) if len(known_good_paths) > 0 else 0.0
    tpr = tp / len(defective_paths) if len(defective_paths) > 0 else 0.0
    accuracy = (tp + tn) / len(all_test)
    total_runtime = time.time() - start_total_time

    augmented_metrics = {
        "calibration_method": "250-GOOD Memory (Augmented 2x) / 30-GOOD Independent Calibration Split",
        "memory_image_count": 500,
        "final_reference_image_count": 560,
        "locked_threshold": round(locked_threshold_aug, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate_percent": round(fpr * 100, 2),
        "recall_sensitivity_percent": round(tpr * 100, 2),
        "overall_accuracy_percent": round(accuracy * 100, 2),
        "runtime_seconds": round(total_runtime, 2)
    }

    # Determine Verdict: RETAIN vs REJECT
    # Retain ONLY if FPR improves (decreases) AND Recall stays >= 90.0%
    fpr_improved = augmented_metrics["false_positive_rate_percent"] < BASELINE_METRICS["false_positive_rate_percent"]
    recall_maintained = augmented_metrics["recall_sensitivity_percent"] >= 90.0

    if fpr_improved and recall_maintained:
        verdict = "RETAIN"
        verdict_reason = f"Mild GOOD-data augmentation successfully reduced FPR from {BASELINE_METRICS['false_positive_rate_percent']}% to {augmented_metrics['false_positive_rate_percent']}% while maintaining high Recall ({augmented_metrics['recall_sensitivity_percent']}% >= 90.0%)."
    elif not fpr_improved:
        verdict = "REJECT"
        verdict_reason = f"Mild GOOD-data augmentation did NOT materially reduce False Positive Rate (Baseline FPR: {BASELINE_METRICS['false_positive_rate_percent']}%, Augmented FPR: {augmented_metrics['false_positive_rate_percent']}%)."
    else:
        verdict = "REJECT"
        verdict_reason = f"Mild GOOD-data augmentation reduced FPR to {augmented_metrics['false_positive_rate_percent']}%, but caused Recall to drop below 90.0% (Augmented Recall: {augmented_metrics['recall_sensitivity_percent']}%)."

    report = {
        "dataset": "textile",
        "experiment": "Controlled GOOD-Data Augmentation (2x Multiplier)",
        "augmentation_parameters": {
            "rotation_range_degrees": [-5, 5],
            "translation_fraction": [0.04, 0.04],
            "scale_range": [0.95, 1.05],
            "brightness_jitter": 0.05,
            "contrast_jitter": 0.05,
            "multiplier": "2x (Original + 1 Mild Augmented Copy)"
        },
        "calibration_statistics": calib_stats,
        "baseline_metrics": BASELINE_METRICS,
        "augmented_metrics": augmented_metrics,
        "experiment_verdict": {
            "decision": verdict,
            "reasoning": verdict_reason
        },
        "known_good_test_results": known_good_results,
        "defective_test_results": defective_results,
    }

    # Save JSON Report
    json_path = output_dir / "textile_good_augmentation_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown Report
    md = fr"""# Textile Controlled GOOD-Data Augmentation Experiment Report

## 1. Executive Summary & Experiment Verdict

- **Experiment Goal**: Test whether mild physical augmentation of GOOD reference images broadens the learned normal distribution and reduces the 80% False Positive Rate on textile test images.
- **Augmentation Configuration**: Mild Rotation ($\pm 5^\circ$), Translation ($\pm 4\%$), Scale ($0.95-1.05$), Brightness/Contrast ($\pm 5\%$) at **$2\times$ Multiplier** ($250 \to 500$ memory samples).
- **Decision Verdict**: **`{verdict}`**
- **Verdict Rationale**: {verdict_reason}

---

## 2. BASELINE vs. AUGMENTED Comparison Table

| Metric | BASELINE (Unaugmented) | AUGMENTED ($2\times$ Multiplier) | Delta / Change |
| :--- | :--- | :--- | :--- |
| **Reference Memory Count** | `250` images | **`500` images** | $+250$ augmented samples |
| **Final Reference Count** | `280` images | **`560` images** | $+280$ augmented samples |
| **Locked Threshold ($\theta$)** | `21.3272` | **`{locked_threshold_aug:.4f}`** | `{locked_threshold_aug - 21.3272:+.4f}` |
| **False Positive Rate (FPR)** | `80.0%` (`8/10`) | **`{augmented_metrics['false_positive_rate_percent']:.1f}%`** (`{fp}/10`) | `{augmented_metrics['false_positive_rate_percent'] - 80.0:+.1f}%` |
| **Recall / Sensitivity (TPR)**| `98.0%` (`49/50`) | **`{augmented_metrics['recall_sensitivity_percent']:.1f}%`** (`{tp}/50`) | `{augmented_metrics['recall_sensitivity_percent'] - 98.0:+.1f}%` |
| **Overall Accuracy** | `85.0%` (`51/60`) | **`{augmented_metrics['overall_accuracy_percent']:.1f}%`** (`{tp+tn}/60`) | `{augmented_metrics['overall_accuracy_percent'] - 85.0:+.1f}%` |
| **True Positives (TP)** | `49` | **`{tp}`** | `{tp - 49:+d}` |
| **True Negatives (TN)** | `2` | **`{tn}`** | `{tn - 2:+d}` |
| **False Positives (FP)** | `8` | **`{fp}`** | `{fp - 8:+d}` |
| **False Negatives (FN)** | `1` | **`{fn}`** | `{fn - 1:+d}` |
| **Total Runtime** | `470.13s` | **`{total_runtime:.2f}s`** | `{total_runtime - 470.13:+.2f}s` |

---

## 3. Calibration Statistics ($N=30$ Untouched GOOD Calibration Images)

- **Mean Raw Distance**: `{calib_stats['mean']:.4f}`
- **Standard Deviation ($\sigma$)**: `{calib_stats['std']:.4f}`
- **Min Raw Distance**: `{calib_stats['min']:.4f}`
- **Max Raw Distance**: `{calib_stats['max']:.4f}`
- **90th Percentile**: `{calib_stats['p90']:.4f}`
- **Locked 95th Percentile Threshold ($\theta_{{aug\_95}}$)**: **`{calib_stats['p95']:.4f}`**
- **97.5th Percentile**: `{calib_stats['p97_5']:.4f}`
- **99th Percentile**: `{calib_stats['p99']:.4f}`

---

## 4. 10 Known GOOD Test Image Results Under Augmentation

| Filename | Baseline Raw | Augmented Raw | Calibrated Score ($S = d / \\theta_{{aug}}$) | Ground Truth | Prediction | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for r in known_good_results:
        ok_icon = "[OK]" if r["is_correct"] else "[FAIL]"
        md += f"| `{r['filename']}` | `—` | `{r['raw_anomaly_distance']:.4f}` | `{r['calibrated_score']:.4f}` | `{r['ground_truth']}` | `{r['prediction']}` | {ok_icon} |\n"

    md_path = output_dir / "textile_good_augmentation_eval.md"
    md_path.write_text(md)

    print("\n" + "=" * 80)
    print("CONTROLLED GOOD-DATA AUGMENTATION EXPERIMENT COMPLETED")
    print(f"VERDICT: {verdict}")
    print(f"Baseline FPR: {BASELINE_METRICS['false_positive_rate_percent']}% -> Augmented FPR: {augmented_metrics['false_positive_rate_percent']}%")
    print(f"Baseline Recall: {BASELINE_METRICS['recall_sensitivity_percent']}% -> Augmented Recall: {augmented_metrics['recall_sensitivity_percent']}%")
    print(f"JSON Saved: {json_path}")
    print(f"Markdown Saved: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
