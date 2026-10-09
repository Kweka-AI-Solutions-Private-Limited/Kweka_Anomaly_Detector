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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GenericFolderDataset(Dataset):
    def __init__(self, image_paths, target_size=DEFAULT_TARGET_SIZE):
        self.image_paths = [Path(p) for p in image_paths]
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {
            "image": tensor,
            "image_path": str(path),
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

    output_dir = base_dir / "outputs" / "generic_patchcore" / "textile"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover and partition GOOD training images (280 total -> 250 memory, 30 calibration)
    all_train = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"Discovered {len(all_train)} GOOD training reference images in {train_dir}")
    if len(all_train) != 280:
        print(f"WARNING: Expected 280 training images, found {len(all_train)}")

    rng = random.Random(SEED)
    shuffled_train = list(all_train)
    rng.shuffle(shuffled_train)

    train_memory_250 = shuffled_train[:250]
    calib_good_30 = shuffled_train[250:]

    print(f"Partitioned GOOD training images: {len(train_memory_250)} Memory Training Images, {len(calib_good_30)} Independent Calibration Images")

    # 2. Discover test images (10 Known GOOD + 50 Defective = 60 total)
    all_test = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    known_good_paths = sorted([p for p in all_test if "copy 2" in p.name])
    defective_paths = sorted([p for p in all_test if p not in known_good_paths])

    print(f"Discovered {len(all_test)} test images ({len(known_good_paths)} Known GOOD, {len(defective_paths)} Defective)")

    # -------------------------------------------------------------
    # PHASE 1: CALIBRATION ON 30 INDEPENDENT GOOD IMAGES
    # -------------------------------------------------------------
    print("\n[Phase 1/2] Building Calibration PatchCore Model on 250 GOOD Memory Images...")
    start_calib_time = time.time()

    model_calib = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine_calib = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    tr_loader_250 = DataLoader(GenericFolderDataset(train_memory_250), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    engine_calib.fit(model=model_calib, train_dataloaders=tr_loader_250)
    model_calib.post_processor = None  # Disable MinMax clipping

    print("Scoring 30 Independent GOOD Calibration Images...")
    calib_loader_30 = DataLoader(GenericFolderDataset(calib_good_30), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    calib_preds = engine_calib.predict(model=model_calib, dataloaders=calib_loader_30)

    calib_scores_dict = {Path(b.image_path[0]).name: float(b.pred_score[0]) for b in calib_preds}
    calib_values = list(calib_scores_dict.values())
    calib_duration = time.time() - start_calib_time

    calib_arr = np.array(calib_values)
    calib_stats = {
        "sample_size": len(calib_values),
        "mean": float(np.mean(calib_arr)),
        "std": float(np.std(calib_arr)),
        "min": float(np.min(calib_arr)),
        "max": float(np.max(calib_arr)),
        "p90": float(np.percentile(calib_arr, 90)),
        "p95": float(np.percentile(calib_arr, 95)),
        "p97_5": float(np.percentile(calib_arr, 97.5)),
        "p99": float(np.percentile(calib_arr, 99)),
    }

    locked_threshold_95 = calib_stats["p95"]
    print(f"Calibration Complete in {calib_duration:.2f}s!")
    print(f"  Mean Raw Dist: {calib_stats['mean']:.4f} +/- {calib_stats['std']:.4f}")
    print(f"  Locked 95th Percentile Threshold (theta_95): {locked_threshold_95:.4f}")

    # -------------------------------------------------------------
    # PHASE 2: EVALUATION OF FULL MODEL (280 GOOD) ON 60 TEST IMAGES
    # -------------------------------------------------------------
    print("\n[Phase 2/2] Building Final Production Model on ALL 280 GOOD Reference Images...")
    start_eval_time = time.time()

    model_final = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine_final = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    full_tr_loader = DataLoader(GenericFolderDataset(all_train), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    engine_final.fit(model=model_final, train_dataloaders=full_tr_loader)
    model_final.post_processor = None  # Disable MinMax clipping

    print("Evaluating 60 Test Images (10 Known GOOD + 50 Defective)...")
    test_loader = DataLoader(GenericFolderDataset(all_test), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    test_preds = engine_final.predict(model=model_final, dataloaders=test_loader)
    eval_duration = time.time() - start_eval_time

    results_per_image = []
    tp, fp, tn, fn = 0, 0, 0, 0
    known_good_results = []
    defective_results = []

    for b in test_preds:
        p_str = b.image_path[0]
        p_path = Path(p_str)
        raw_dist = float(b.pred_score[0])
        calib_score = raw_dist / locked_threshold_95
        pred_status = "ANOMALOUS" if calib_score >= 1.0 else "NORMAL"
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
            "relative_path": str(p_path.relative_to(data_dir)),
            "filename": p_path.name,
            "ground_truth": gt_status,
            "prediction": pred_status,
            "raw_anomaly_distance": round(raw_dist, 4),
            "calibrated_score": round(calib_score, 4),
            "diagnostic_z_score": round((raw_dist - calib_stats["mean"]) / calib_stats["std"], 2),
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

    metrics = {
        "calibration_method": "250-GOOD Memory / 30-GOOD Independent Calibration Split",
        "n_train_total": len(all_train),
        "n_train_memory_ref": len(train_memory_250),
        "n_train_calib_good": len(calib_good_30),
        "n_test_total": len(all_test),
        "n_known_good_test": len(known_good_paths),
        "n_defective_test": len(defective_paths),
        "locked_95th_percentile_threshold": round(locked_threshold_95, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "rates": {
            "false_positive_rate": round(fpr, 4),
            "false_positive_rate_percent": round(fpr * 100, 2),
            "true_positive_rate_recall_sensitivity": round(tpr, 4),
            "true_positive_rate_percent": round(tpr * 100, 2),
            "overall_accuracy": round(accuracy, 4),
            "overall_accuracy_percent": round(accuracy * 100, 2),
        },
        "runtimes_seconds": {
            "calibration_phase": round(calib_duration, 2),
            "evaluation_phase": round(eval_duration, 2),
            "total_execution_runtime": round(total_runtime, 2),
        }
    }

    report = {
        "dataset": "textile",
        "detector_configuration": {
            "backbone": BACKBONE,
            "layers": LAYERS,
            "coreset_sampling_ratio": CORESET_RATIO,
            "num_neighbors": NUM_NEIGHBORS,
            "input_size": DEFAULT_TARGET_SIZE,
            "seed": SEED
        },
        "calibration_statistics": calib_stats,
        "calibration_raw_scores_dict": {k: round(v, 4) for k, v in calib_scores_dict.items()},
        "metrics": metrics,
        "known_good_test_results": known_good_results,
        "defective_test_results": defective_results,
    }

    # Save JSON Report
    json_path = output_dir / "textile_250_30_calibration_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Generate Markdown Report
    md = f"""# Textile 250-Memory / 30-Calibration Experiment Report

## 1. Executive Summary & Calibration Setup

- **Dataset**: `data/textile`
- **Detector Configuration**: `WideResNet50_2` + `Layer2` + `5% Coreset` + `9 Neighbors` + `256x256` (Seed: 42)
- **Calibration Protocol**: 250-GOOD Memory Training Images / 30-GOOD Independent Calibration Images Split
- **Locked 95th Percentile Threshold ($\theta_{{95}}$)**: **`{locked_threshold_95:.4f}`**
- **Total Execution Runtime**: **`{total_runtime:.2f} seconds`**

---

## 2. Independent Calibration Statistics (30 Out-of-Sample GOOD Images)

- **Number of Calibration Samples**: `30`
- **Mean Raw Distance**: `{calib_stats['mean']:.4f}`
- **Standard Deviation**: `{calib_stats['std']:.4f}`
- **Min Raw Distance**: `{calib_stats['min']:.4f}`
- **Max Raw Distance**: `{calib_stats['max']:.4f}`
- **90th Percentile**: `{calib_stats['p90']:.4f}`
- **95th Percentile Threshold ($\theta_{{95}}$)**: **`{calib_stats['p95']:.4f}`**
- **97.5th Percentile**: `{calib_stats['p97_5']:.4f}`
- **99th Percentile**: `{calib_stats['p99']:.4f}`

---

## 3. Core Evaluation Metrics & Confusion Matrix

| Metric | Value | Detail |
| :--- | :--- | :--- |
| **Total Test Images** | `60` | `10` Known GOOD + `50` Defective |
| **True Positives (TP)** | **`{tp}`** | Defective images correctly classified as ANOMALOUS |
| **True Negatives (TN)** | **`{tn}`** | Known GOOD test images correctly classified as NORMAL |
| **False Positives (FP)** | **`{fp}`** | Known GOOD test images misclassified as ANOMALOUS |
| **False Negatives (FN)** | **`{fn}`** | Defective images misclassified as NORMAL |
| **False Positive Rate (FPR)** | **`{fpr*100:.1f}%`** | `{fp} / 10` Known GOOD test images |
| **Recall / Sensitivity (TPR)** | **`{tpr*100:.1f}%`** | `{tp} / 50` Defective test images |
| **Overall Accuracy** | **`{accuracy*100:.1f}%`** | `{tp+tn} / 60` Total test images |

---

## 4. Performance on 10 Known GOOD Test Images

| Filename | Raw Distance | Calibrated Score ($S = d / \theta_{{95}}$) | Ground Truth | Prediction | Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for r in known_good_results:
        correct_icon = "[OK]" if r["is_correct"] else "[FAIL]"
        md += f"| `{r['filename']}` | `{r['raw_anomaly_distance']:.4f}` | `{r['calibrated_score']:.4f}` | `{r['ground_truth']}` | `{r['prediction']}` | {correct_icon} |\n"

    md += f"""
---

## 5. Performance on 50 Defective Test Images (Summary)

- **Total Defective Images Evaluated**: `50`
- **Correctly Identified as ANOMALOUS (TP)**: **`{tp}`**
- **Missed / Classified as NORMAL (FN)**: **`{fn}`**
- **Defective Test Image Distances Range**: `{min(r['raw_anomaly_distance'] for r in defective_results):.4f}` – `{max(r['raw_anomaly_distance'] for r in defective_results):.4f}`
"""

    md_path = output_dir / "textile_250_30_calibration_eval.md"
    md_path.write_text(md)

    print("\n" + "=" * 80)
    print("TEXTILE 250-MEMORY / 30-CALIBRATION EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"Locked Threshold (theta_95): {locked_threshold_95:.4f}")
    print(f"FPR on 10 Known GOOD: {fpr*100:.1f}% ({fp}/10)")
    print(f"Recall/TPR on 50 Defective: {tpr*100:.1f}% ({tp}/50)")
    print(f"Overall Accuracy: {accuracy*100:.1f}% ({tp+tn}/60)")
    print(f"Total Runtime: {total_runtime:.2f}s")
    print(f"JSON Report Saved: {json_path}")
    print(f"Markdown Report Saved: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
