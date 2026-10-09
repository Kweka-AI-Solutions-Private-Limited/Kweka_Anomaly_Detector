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


def set_seed(seed=42):
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


def run_evaluation():
    set_seed(42)

    base_dir = Path("c:/dev/Anomaly_Detector/backend").resolve()
    data_dir = base_dir / "data" / "textile"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    output_dir = base_dir / "outputs" / "generic_patchcore" / "textile"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover train images (280 GOOD)
    train_images = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"Discovered {len(train_images)} GOOD training reference images in {train_dir}")
    if len(train_images) != 280:
        print(f"WARNING: Expected 280 training images, found {len(train_images)}")

    # 2. Discover test images (10 Known GOOD + 50 Defective = 60 total)
    all_test = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    known_good_paths = sorted([p for p in all_test if "copy 2" in p.name])
    defective_paths = sorted([p for p in all_test if p not in known_good_paths])

    print(f"Discovered {len(all_test)} test images ({len(known_good_paths)} Known GOOD, {len(defective_paths)} Defective)")

    # -------------------------------------------------------------
    # 3. LEAVE-ONE-OUT (LOO) CALIBRATION ACROSS ALL N=280 GOOD IMAGES
    # -------------------------------------------------------------
    print("\n[1/3] Running Leave-One-Out (LOO) Calibration across N=280 GOOD reference images...")
    start_loo_time = time.time()
    loo_raw_scores = {}

    for i in range(len(train_images)):
        if (i + 1) % 40 == 0 or i == 0 or i == len(train_images) - 1:
            print(f"  LOO Progress: {i+1}/{len(train_images)} images processed...")

        loo_train = [train_images[j] for j in range(len(train_images)) if j != i]
        loo_val = [train_images[i]]

        m_loo = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
        e_loo = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

        tr_loader_loo = DataLoader(GenericFolderDataset(loo_train), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
        val_loader_loo = DataLoader(GenericFolderDataset(loo_val), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)

        e_loo.fit(model=m_loo, train_dataloaders=tr_loader_loo)
        m_loo.post_processor = None

        p_loo = e_loo.predict(model=m_loo, dataloaders=val_loader_loo)
        s_loo = float(p_loo[0].pred_score[0])
        loo_raw_scores[train_images[i].name] = s_loo

    loo_duration = time.time() - start_loo_time
    print(f"LOO Calibration Completed in {loo_duration:.2f}s")

    loo_arr = np.array(list(loo_raw_scores.values()))
    loo_stats = {
        "mean": float(np.mean(loo_arr)),
        "std": float(np.std(loo_arr)),
        "min": float(np.min(loo_arr)),
        "max": float(np.max(loo_arr)),
        "p90": float(np.percentile(loo_arr, 90)),
        "p95": float(np.percentile(loo_arr, 95)),
        "p97_5": float(np.percentile(loo_arr, 97.5)),
        "p99": float(np.percentile(loo_arr, 99)),
    }

    calibrated_threshold = loo_stats["p95"]
    print(f"\nDerived 95th Percentile LOO Threshold: {calibrated_threshold:.4f}")

    # -------------------------------------------------------------
    # 4. BUILD FINAL MODEL ON ALL 280 GOOD REFERENCE IMAGES & SCORE TEST SET
    # -------------------------------------------------------------
    print("\n[2/3] Fitting final PatchCore model on ALL 280 GOOD reference images...")
    final_model = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    final_engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    full_tr_loader = DataLoader(GenericFolderDataset(train_images), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    final_engine.fit(model=final_model, train_dataloaders=full_tr_loader)
    final_model.post_processor = None

    print("[3/3] Evaluating 60 test images (10 Known GOOD + 50 Defective)...")
    test_dataset = GenericFolderDataset(all_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=generic_collate_fn)

    test_preds = final_engine.predict(model=final_model, dataloaders=test_loader)

    results_per_image = []
    tp, fp, tn, fn = 0, 0, 0, 0

    for b in test_preds:
        p_str = b.image_path[0]
        p_path = Path(p_str)
        raw_dist = float(b.pred_score[0])
        calib_score = raw_dist / calibrated_threshold
        pred_status = "ANOMALOUS" if calib_score >= 1.0 else "NORMAL"
        is_known_good = "copy 2" in p_path.name
        gt_status = "NORMAL" if is_known_good else "ANOMALOUS"

        if gt_status == "ANOMALOUS" and pred_status == "ANOMALOUS":
            tp += 1
        elif gt_status == "NORMAL" and pred_status == "ANOMALOUS":
            fp += 1
        elif gt_status == "NORMAL" and pred_status == "NORMAL":
            tn += 1
        elif gt_status == "ANOMALOUS" and pred_status == "NORMAL":
            fn += 1

        results_per_image.append({
            "relative_path": str(p_path.relative_to(data_dir)),
            "filename": p_path.name,
            "ground_truth": gt_status,
            "prediction": pred_status,
            "raw_anomaly_distance": round(raw_dist, 4),
            "calibrated_score": round(calib_score, 4),
            "diagnostic_z_score": round((raw_dist - loo_stats["mean"]) / loo_stats["std"], 2),
            "is_correct": (pred_status == gt_status)
        })

    fpr = fp / len(known_good_paths) if len(known_good_paths) > 0 else 0.0
    tpr = tp / len(defective_paths) if len(defective_paths) > 0 else 0.0
    accuracy = (tp + tn) / len(all_test)

    summary_metrics = {
        "total_test_images": len(all_test),
        "n_known_good": len(known_good_paths),
        "n_defective": len(defective_paths),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "false_positive_rate": round(fpr, 4),
        "false_positive_rate_percent": round(fpr * 100, 2),
        "true_positive_rate_sensitivity": round(tpr, 4),
        "true_positive_rate_percent": round(tpr * 100, 2),
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
    }

    report = {
        "dataset": "textile",
        "n_train_good": len(train_images),
        "calibration_method": "Leave-One-Out (LOO) on N=280 GOOD images",
        "loo_calibration_statistics": loo_stats,
        "calibrated_95th_percentile_threshold": round(calibrated_threshold, 4),
        "evaluation_summary": summary_metrics,
        "per_image_results": results_per_image,
    }

    # Save JSON Report
    json_path = output_dir / "textile_280_loo_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Generate Markdown Report
    md = f"""# Textile 280-Image LOO Calibration & Test Set Evaluation Report

## 1. Executive Summary & Core Performance Metrics

- **Training Reference Dataset**: `280` GOOD reference images (`data/textile/train`)
- **Calibration Method**: Leave-One-Out (LOO) across all `N=280` training images
- **Derived 95th Percentile Threshold**: **`{calibrated_threshold:.4f}`**
- **Test Dataset Composition**: `60` total test images (`10` Known GOOD + `50` Defective)

### Summary Table
| Metric | Value | Breakdown |
| :--- | :--- | :--- |
| **Total Test Images** | `60` | `10` Known GOOD, `50` Defective |
| **True Positives (TP)** | `{tp}` | Correctly identified defective images |
| **True Negatives (TN)** | `{tn}` | Correctly identified GOOD test images |
| **False Positives (FP)** | `{fp}` | GOOD test images misclassified as defective |
| **False Negatives (FN)** | `{fn}` | Defective images misclassified as GOOD |
| **False Positive Rate (FPR)** | **`{fpr*100:.1f}%`** | `{fp} / 10` Known GOOD images |
| **True Positive Rate (TPR / Sensitivity)** | **`{tpr*100:.1f}%`** | `{tp} / 50` Defective images |
| **Overall Accuracy** | **`{accuracy*100:.1f}%`** | `{tp+tn} / 60` Total test images |

---

## 2. 280-Image Out-of-Sample LOO Calibration Statistics

- **Mean Raw Distance**: `{loo_stats['mean']:.4f}`
- **Standard Deviation**: `{loo_stats['std']:.4f}`
- **Min Raw Distance**: `{loo_stats['min']:.4f}`
- **Max Raw Distance**: `{loo_stats['max']:.4f}`
- **90th Percentile**: `{loo_stats['p90']:.4f}`
- **95th Percentile Threshold ($\theta_{{LOO\_95}}$)**: **`{loo_stats['p95']:.4f}`**
- **97.5th Percentile**: `{loo_stats['p97_5']:.4f}`
- **99th Percentile**: `{loo_stats['p99']:.4f}`

---

## 3. Detailed Results on 10 Known GOOD Test Images

| Filename | Raw Distance | Calibrated Score ($S = d / \theta_{{95}}$) | Ground Truth | Prediction | Correct? |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for r in results_per_image:
        if r["ground_truth"] == "NORMAL":
            correct_icon = "✅" if r["is_correct"] else "❌"
            md += f"| `{r['filename']}` | `{r['raw_anomaly_distance']:.4f}` | `{r['calibrated_score']:.4f}` | `{r['ground_truth']}` | `{r['prediction']}` | {correct_icon} |\n"

    md += f"""
---

## 4. Defective Test Image Summary (50 Defective Images)

- **Total Defective Images Evaluated**: `50`
- **Correctly Classified as ANOMALOUS (TP)**: `{tp}`
- **Misclassified as NORMAL (FN)**: `{fn}`
"""

    md_path = output_dir / "textile_280_loo_eval.md"
    md_path.write_text(md)

    print("\n" + "=" * 80)
    print("TEXTILE 280-IMAGE LOO EVALUATION COMPLETED")
    print(f"FPR on 10 GOOD: {fpr*100:.1f}% ({fp}/10)")
    print(f"TPR on 50 Defective: {tpr*100:.1f}% ({tp}/50)")
    print(f"Overall Accuracy: {accuracy*100:.1f}% ({tp+tn}/60)")
    print(f"JSON Report Saved: {json_path}")
    print(f"Markdown Report Saved: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
