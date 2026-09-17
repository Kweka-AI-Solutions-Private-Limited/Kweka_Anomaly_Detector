import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
import numpy as np
import cv2
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
DEFAULT_LOCKED_THRESHOLD = 21.3272


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_ground_truth(file_path: Path) -> str:
    """
    Generic ground truth inference based on path and filename semantics.
    Returns 'NORMAL' if image represents a GOOD reference sample, else 'ANOMALOUS'.
    """
    path_str = str(file_path).lower()
    parent_str = file_path.parent.name.lower()
    name_str = file_path.name.lower()

    # Common GOOD conventions: 'good', 'normal', 'ok', 'copy 2'
    is_good = (
        "good" in parent_str
        or "normal" in parent_str
        or "ok" in parent_str
        or "good" in name_str
        or "normal" in name_str
        or "copy 2" in name_str
        or "copy_2" in name_str
    )

    return "NORMAL" if is_good else "ANOMALOUS"


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


def run_failure_analysis(dataset_dir: str, threshold: float, output_dir: str):
    set_seed(SEED)
    start_time = time.time()

    dataset_path = Path(dataset_dir).resolve()
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    dataset_name = dataset_path.name

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Dataset directory must contain 'train/' and 'test/' subfolders: {dataset_path}")

    out_base = Path(output_dir).resolve() if output_dir else dataset_path.parent.parent / "outputs" / "generic_patchcore" / dataset_name / "failure_analysis"
    out_base.mkdir(parents=True, exist_ok=True)

    # 1. Discover training images
    train_images = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"[{dataset_name}] Discovered {len(train_images)} training reference images in {train_dir}")

    # 2. Discover test images
    test_images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"[{dataset_name}] Discovered {len(test_images)} test images in {test_dir}")

    if len(train_images) == 0 or len(test_images) == 0:
        raise ValueError(f"No valid images found in train ({len(train_images)}) or test ({len(test_images)})")

    # 3. Fit PatchCore model ONCE on all training images
    print(f"\n[1/3] Training final PatchCore model on {len(train_images)} GOOD reference images...")
    model = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    train_loader = DataLoader(GenericFolderDataset(train_images), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    engine.fit(model=model, train_dataloaders=train_loader)
    model.post_processor = None  # Disable min-max clipping to preserve raw distances

    # 4. Score all test images
    print(f"\n[2/3] Scoring {len(test_images)} test images against locked threshold {threshold:.4f}...")
    test_loader = DataLoader(GenericFolderDataset(test_images), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    predictions = engine.predict(model=model, dataloaders=test_loader)

    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    all_results = []
    gallery_items = []

    for pred in predictions:
        p_str = pred.image_path[0]
        p_path = Path(p_str)
        fname = p_path.name
        raw_score = float(pred.pred_score[0])
        calib_score = raw_score / threshold
        pred_status = "ANOMALOUS" if raw_score >= threshold else "NORMAL"
        gt_status = infer_ground_truth(p_path)

        is_correct = (pred_status == gt_status)

        if gt_status == "ANOMALOUS" and pred_status == "ANOMALOUS":
            category = "TP"
            tp_list.append(fname)
        elif gt_status == "NORMAL" and pred_status == "NORMAL":
            category = "TN"
            tn_list.append(fname)
        elif gt_status == "NORMAL" and pred_status == "ANOMALOUS":
            category = "FP"
            fp_list.append(fname)
        elif gt_status == "ANOMALOUS" and pred_status == "NORMAL":
            category = "FN"
            fn_list.append(fname)

        res_item = {
            "filename": fname,
            "relative_path": str(p_path.relative_to(dataset_path)),
            "ground_truth": gt_status,
            "prediction": pred_status,
            "category": category,
            "raw_anomaly_distance": round(raw_score, 4),
            "calibrated_score": round(calib_score, 4),
            "is_correct": is_correct
        }
        all_results.append(res_item)

        # Generate visualizations for FP and TN images (and optional TP/FN)
        if category in ("FP", "TN"):
            clean_stem = p_path.stem.replace(" ", "_")
            folder_name = f"{clean_stem}_{category}"

            cat_sub_dir = out_base / ("false_positives" if category == "FP" else "true_negatives") / folder_name
            cat_sub_dir.mkdir(parents=True, exist_ok=True)

            amap = pred.anomaly_map[0].squeeze().cpu().numpy()

            orig_pil = Image.open(p_path).convert("RGB").resize(DEFAULT_TARGET_SIZE, Image.BILINEAR)
            orig_rgb = np.array(orig_pil)
            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

            # original.png
            cv2.imwrite(str(cat_sub_dir / "original.png"), orig_bgr)

            # heatmap.png
            amap_norm = np.clip((amap - 15.0) / (32.0 - 15.0), 0.0, 1.0)
            amap_u8 = (amap_norm * 255.0).astype(np.uint8)
            heatmap_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
            cv2.imwrite(str(cat_sub_dir / "heatmap.png"), heatmap_bgr)

            # overlay.png
            overlay_bgr = cv2.addWeighted(orig_bgr, 0.5, heatmap_bgr, 0.5, 0)
            cv2.imwrite(str(cat_sub_dir / "overlay.png"), overlay_bgr)

            # bbox.png
            bbox_bgr = overlay_bgr.copy()
            mask = (amap >= threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                for cnt in contours:
                    if cv2.contourArea(cnt) >= 16:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(bbox_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        peak_val = amap[y:y+h, x:x+w].max()
                        cv2.putText(bbox_bgr, f"{peak_val:.1f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                max_y, max_x = np.unravel_index(np.argmax(amap), amap.shape)
                x1, y1 = max(0, max_x - 16), max(0, max_y - 16)
                x2, y2 = min(256, max_x + 16), min(256, max_y + 16)
                cv2.rectangle(bbox_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bbox_bgr, f"Peak: {amap.max():.1f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Header Banner
            header_h = 55
            padded_bbox = np.zeros((256 + header_h, 256, 3), dtype=np.uint8)
            padded_bbox[header_h:, :] = bbox_bgr
            status_color = (0, 0, 255) if pred_status == "ANOMALOUS" else (0, 200, 0)
            cv2.rectangle(padded_bbox, (0, 0), (256, header_h), (20, 20, 20), -1)

            cv2.putText(padded_bbox, f"File: {fname}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(padded_bbox, f"GT: {gt_status} | Pred: {pred_status}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)
            cv2.putText(padded_bbox, f"Dist: {raw_score:.2f} | Thresh: {threshold:.2f} ({category})", (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            cv2.imwrite(str(cat_sub_dir / "bbox.png"), padded_bbox)

            gallery_items.append({
                "filename": fname,
                "category": category,
                "bbox_img": padded_bbox
            })

    # 5. Build Contact Sheet Gallery
    print("\n[3/3] Generating Composite Contact Sheet Gallery...")
    n_gal = len(gallery_items)
    if n_gal > 0:
        cols = min(5, n_gal)
        rows = int(np.ceil(n_gal / cols))
        cell_w, cell_h = 256, 256 + 55
        margin = 10
        grid_w = cols * cell_w + (cols + 1) * margin
        grid_h = rows * cell_h + (rows + 1) * margin + 40

        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30
        cv2.putText(canvas, f"{dataset_name.upper()} GOOD TEST IMAGES FAILURE ANALYSIS ({n_gal} IMAGES)", (margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for idx, g_item in enumerate(gallery_items):
            r = idx // cols
            c = idx % cols
            x = margin + c * (cell_w + margin)
            y = 40 + margin + r * (cell_h + margin)
            canvas[y:y+cell_h, x:x+cell_w] = g_item["bbox_img"]

        gallery_path = out_base / f"{dataset_name}_failure_analysis_gallery.png"
        cv2.imwrite(str(gallery_path), canvas)
        print(f"Saved Contact Sheet Gallery: {gallery_path}")

    # Calculate summary metrics
    n_tp = len(tp_list)
    n_tn = len(tn_list)
    n_fp = len(fp_list)
    n_fn = len(fn_list)
    n_total = len(all_results)
    n_good = n_tn + n_fp
    n_def = n_tp + n_fn

    fpr = n_fp / n_good if n_good > 0 else 0.0
    tpr = n_tp / n_def if n_def > 0 else 0.0
    acc = (n_tp + n_tn) / n_total if n_total > 0 else 0.0
    duration = time.time() - start_time

    summary = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "locked_threshold": threshold,
        "total_test_images": n_total,
        "confusion_matrix": {
            "true_positives": n_tp,
            "true_negatives": n_tn,
            "false_positives": n_fp,
            "false_negatives": n_fn,
        },
        "rates": {
            "false_positive_rate": round(fpr, 4),
            "false_positive_rate_percent": round(fpr * 100, 2),
            "recall_sensitivity_tpr": round(tpr, 4),
            "recall_percent": round(tpr * 100, 2),
            "accuracy": round(acc, 4),
            "accuracy_percent": round(acc * 100, 2),
        },
        "discovered_false_positives": fp_list,
        "discovered_true_negatives": tn_list,
        "discovered_true_positives": tp_list,
        "discovered_false_negatives": fn_list,
        "all_results": all_results,
        "runtime_seconds": round(duration, 2)
    }

    # Save JSON Report
    json_path = out_base / f"{dataset_name}_failure_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save Markdown Report
    md = f"""# Generic PatchCore Failure Analysis Report: `{dataset_name}`

## 1. Executive Summary

- **Dataset Directory**: `{dataset_path}`
- **Locked Threshold**: **`{threshold:.4f}`**
- **Total Test Images Evaluated**: `{n_total}` (`{n_good}` GOOD + `{n_def}` Defective)
- **False Positive Rate (FPR)**: **`{fpr*100:.1f}%`** (`{n_fp} / {n_good}` GOOD test images)
- **Recall / Sensitivity (TPR)**: **`{tpr*100:.1f}%`** (`{n_tp} / {n_def}` Defective test images)
- **Overall Accuracy**: **`{acc*100:.1f}%`** (`{n_tp+n_tn} / {n_total}`)
- **Runtime**: **`{duration:.2f} seconds`**

---

## 2. Discovered Classification Breakdown

### Automatically Discovered False Positive Images (`{n_fp}` Images)
"""
    for fname in fp_list:
        res = next(r for r in all_results if r["filename"] == fname)
        md += f"- `{fname}` — Raw Distance: `{res['raw_anomaly_distance']:.4f}` (Score ratio: `{res['calibrated_score']:.4f}`)\n"

    md += f"""
### Automatically Discovered True Negative Images (`{n_tn}` Images)
"""
    for fname in tn_list:
        res = next(r for r in all_results if r["filename"] == fname)
        md += f"- `{fname}` — Raw Distance: `{res['raw_anomaly_distance']:.4f}` (Score ratio: `{res['calibrated_score']:.4f}`)\n"

    md += f"""
---

## 3. Comprehensive Test Results Table

| Filename | Ground Truth | Prediction | Category | Raw Distance | Calibrated Score ($S = d / \\theta$) | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for res in all_results:
        ok_label = "[OK]" if res["is_correct"] else "[FAIL]"
        md += f"| `{res['filename']}` | `{res['ground_truth']}` | `{res['prediction']}` | `{res['category']}` | `{res['raw_anomaly_distance']:.4f}` | `{res['calibrated_score']:.4f}` | {ok_label} |\n"

    md_path = out_base / f"{dataset_name}_failure_analysis.md"
    md_path.write_text(md)

    print("\n" + "=" * 80)
    print(f"GENERIC PATCHCORE FAILURE ANALYSIS COMPLETE FOR [{dataset_name.upper()}]")
    print(f"Discovered FPs: {fp_list}")
    print(f"Discovered TNs: {tn_list}")
    print(f"FPR: {fpr*100:.1f}% | Recall: {tpr*100:.1f}% | Accuracy: {acc*100:.1f}%")
    print(f"JSON Saved: {json_path}")
    print(f"Markdown Saved: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Dataset-Agnostic PatchCore Failure Analysis Tool")
    parser.add_argument("--dataset_dir", type=str, default="c:/dev/Anomaly_Detector/backend/data/textile", help="Path to dataset root containing train/ and test/")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LOCKED_THRESHOLD, help="Locked decision threshold")
    parser.add_argument("--output_dir", type=str, default="", help="Optional custom output directory")

    args = parser.parse_args()
    run_failure_analysis(args.dataset_dir, args.threshold, args.output_dir)
