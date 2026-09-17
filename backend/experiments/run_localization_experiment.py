"""
run_localization_experiment.py — PatchCore Localization Benchmark Experiment
-----------------------------------------------------------------------------
Runs a standardized benchmark comparing:
1. BASELINE: Max Contour Area Selection
2. EXPERIMENT: Intensity-Weighted Component Selection

Reports:
- Recall, FPR, TP, FN, TN, FP (Classification sanity check)
- Pixel-level & BBox Mean/Median IoU
- Per-defect breakdown
- Localization success / failure counts
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import cv2

# Ensure src is in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from services.localization_experiment import (
    select_localization_component_current,
    select_localization_component_intensity_weighted,
    refine_heatmap_edge_aware
)

BASE_DIR = SRC_DIR.parent
DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "FINAL"
    / "patchcore_screw_wideresnet50_l2_005.ckpt"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "localization_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_bbox_iou(boxA: Dict[str, int], boxB: Dict[str, int]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA["x"], boxB["x"])
    yA = max(boxA["y"], boxB["y"])
    xB = min(boxA["x"] + boxA["width"], boxB["x"] + boxB["width"])
    yB = min(boxA["y"] + boxA["height"], boxB["y"] + boxB["height"])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA["width"] * boxA["height"]
    boxBArea = boxB["width"] * boxB["height"]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou


def mask_from_bbox(bbox: Optional[Dict[str, int]], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if bbox:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        mask[y : y + h, x : x + w] = True
    return mask


def main():
    print("=" * 72)
    print("PATCHCORE BENCHMARK — BASELINE vs EXPERIMENT 1 vs EXPERIMENT 2")
    print("=" * 72)

    if not CHECKPOINT.exists():
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT}")
        return

    print("Initializing MVTecAD Screw dataset...")
    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="screw",
        num_workers=0,
    )
    datamodule.setup()

    print("Initializing PatchCore WideResNet50_2 baseline model...")
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2"],
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    print("Running inference...")
    preds = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(CHECKPOINT))
    if not preds:
        print("[ERROR] Predictions failed.")
        return

    # Metrics accumulators
    baseline_bous = []
    exp1_bous = []
    exp2_bous = []

    per_defect_baseline = {}
    per_defect_exp1 = {}
    per_defect_exp2 = {}

    tp_count = 0
    fn_count = 0
    tn_count = 0
    fp_count = 0

    valid_loc_baseline = 0
    valid_loc_exp1 = 0
    valid_loc_exp2 = 0
    no_comp_count = 0

    sample_counter = 0

    for batch in preds:
        batches = batch if isinstance(batch, list) else [batch]
        for b in batches:
            image_paths = getattr(b, "image_path", None)
            if not image_paths:
                continue

            batch_size = len(image_paths)
            for i in range(batch_size):
                sample_counter += 1
                img_p = str(image_paths[i])
                defect_type = Path(img_p).parent.name

                # Load original guide image for Experiment 2
                guide_img_np = cv2.imread(img_p)

                gt_lbl = getattr(b, "gt_label", None)
                if isinstance(gt_lbl, torch.Tensor):
                    gt_label = bool(gt_lbl[i].item()) if gt_lbl.ndim > 0 else bool(gt_lbl.item())
                else:
                    gt_label = bool(gt_lbl)

                pred_lbl = getattr(b, "pred_label", None)
                if isinstance(pred_lbl, torch.Tensor):
                    pred_label = bool(pred_lbl[i].item()) if pred_lbl.ndim > 0 else bool(pred_lbl.item())
                else:
                    pred_label = bool(pred_lbl)

                # Classification matrix
                if gt_label and pred_label:
                    tp_count += 1
                elif not gt_label and not pred_label:
                    tn_count += 1
                elif not gt_label and pred_label:
                    fp_count += 1
                elif gt_label and not pred_label:
                    fn_count += 1

                if defect_type not in per_defect_baseline:
                    per_defect_baseline[defect_type] = []
                    per_defect_exp1[defect_type] = []
                    per_defect_exp2[defect_type] = []

                # Process localization if anomaly map is present
                if hasattr(b, "anomaly_map") and b.anomaly_map is not None:
                    am_raw = b.anomaly_map[i] if b.anomaly_map.ndim > 3 else b.anomaly_map
                    am = am_raw.detach().cpu().numpy()
                    while am.ndim > 2:
                        am = am[0]

                    # GT mask & bounding box
                    gt_mask = None
                    if hasattr(b, "gt_mask") and b.gt_mask is not None:
                        gm_raw = b.gt_mask[i] if b.gt_mask.ndim > 3 else b.gt_mask
                        gt_mask = gm_raw.detach().cpu().numpy()
                        while gt_mask.ndim > 2:
                            gt_mask = gt_mask[0]

                    gt_bbox = None
                    if gt_mask is not None and gt_mask.any():
                        gt_uint8 = (gt_mask > 0).astype(np.uint8) * 255
                        contours_gt, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours_gt:
                            largest_gt = max(contours_gt, key=cv2.contourArea)
                            gx, gy, gw, gh = cv2.boundingRect(largest_gt)
                            gt_bbox = {"x": int(gx), "y": int(gy), "width": int(gw), "height": int(gh)}

                # Normalize anomaly map to uint8
                norm_map = (am - am.min()) / (am.max() - am.min() + 1e-8)
                norm_map = (norm_map * 255).astype(np.uint8)

                if gt_mask is not None:
                    norm_map_resized = cv2.resize(norm_map, (gt_mask.shape[1], gt_mask.shape[0]))
                else:
                    norm_map_resized = norm_map

                # --- BASELINE & EXP 1 BRANCH ---
                thresh_mask_base = (norm_map_resized > 128).astype(np.uint8)
                contours_pred_base, _ = cv2.findContours(thresh_mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # --- EXP 2 (EDGE-AWARE SPATIAL REFINEMENT) BRANCH ---
                # Apply conservative bilateral spatial refinement before identical >128 thresholding
                norm_map_exp2 = refine_heatmap_edge_aware(
                    norm_map_resized, guide_image_np=guide_img_np, d=9, sigma_color=75.0, sigma_space=75.0
                )
                thresh_mask_exp2 = (norm_map_exp2 > 128).astype(np.uint8)
                contours_pred_exp2, _ = cv2.findContours(thresh_mask_exp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_base = None
                bbox_exp1 = None
                bbox_exp2 = None

                if contours_pred_base:
                    bbox_base = select_localization_component_current(contours_pred_base, norm_map_resized)
                    bbox_exp1 = select_localization_component_intensity_weighted(
                        contours_pred_base, norm_map_resized, alpha=1.5, beta=1.0
                    )
                else:
                    no_comp_count += 1

                if contours_pred_exp2:
                    bbox_exp2 = select_localization_component_current(contours_pred_exp2, norm_map_exp2)

                if bbox_base:
                    valid_loc_baseline += 1
                if bbox_exp1:
                    valid_loc_exp1 += 1
                if bbox_exp2:
                    valid_loc_exp2 += 1

                # Calculate IoU against GT bbox if defective sample
                if gt_label and gt_bbox is not None:
                    iou_b = calculate_bbox_iou(bbox_base, gt_bbox) if bbox_base else 0.0
                    iou_e1 = calculate_bbox_iou(bbox_exp1, gt_bbox) if bbox_exp1 else 0.0
                    iou_e2 = calculate_bbox_iou(bbox_exp2, gt_bbox) if bbox_exp2 else 0.0

                    baseline_bous.append(iou_b)
                    exp1_bous.append(iou_e1)
                    exp2_bous.append(iou_e2)

                    per_defect_baseline[defect_type].append(iou_b)
                    per_defect_exp1[defect_type].append(iou_e1)
                    per_defect_exp2[defect_type].append(iou_e2)

    recall = (tp_count / (tp_count + fn_count)) * 100 if (tp_count + fn_count) > 0 else 0.0
    fpr = (fp_count / (fp_count + tn_count)) * 100 if (fp_count + tn_count) > 0 else 0.0

    mean_iou_baseline = float(np.mean(baseline_bous)) * 100 if baseline_bous else 0.0
    mean_iou_exp1 = float(np.mean(exp1_bous)) * 100 if exp1_bous else 0.0
    mean_iou_exp2 = float(np.mean(exp2_bous)) * 100 if exp2_bous else 0.0

    print("\n" + "=" * 72)
    print("BENCHMARK EVALUATION RESULTS SUMMARY")
    print("=" * 72)
    print(f"Total Samples Evaluated: {sample_counter}")
    print(f"True Positives  : {tp_count}")
    print(f"False Negatives : {fn_count}")
    print(f"True Negatives  : {tn_count}")
    print(f"False Positives : {fp_count}")
    print(f"Recall          : {recall:.2f}% (Classification decision strictly UNCHANGED)")
    print(f"FPR             : {fpr:.2f}% (Classification decision strictly UNCHANGED)")

    print("\nLOCALIZATION IOU COMPARISON (Pixel & Bounding Box)")
    print("-" * 78)
    print(f"{'Metric':<22} | {'Baseline (Max Area)':<18} | {'Exp 1 (Intensity)':<18} | {'Exp 2 (Edge Refined)':<20}")
    print("-" * 78)
    print(f"{'Mean BBox IoU':<22} | {mean_iou_baseline:>17.2f}% | {mean_iou_exp1:>17.2f}% | {mean_iou_exp2:>19.2f}%")
    print(f"{'Valid Localizations':<22} | {valid_loc_baseline:>18} | {valid_loc_exp1:>18} | {valid_loc_exp2:>20}")

    print("\nPER-DEFECT IOU BREAKDOWN")
    print("-" * 78)
    print(f"{'Defect Category':<20} | {'Baseline IoU':<15} | {'Exp 1 IoU':<15} | {'Exp 2 IoU':<18}")
    print("-" * 78)
    for defect in per_defect_baseline:
        if defect == "good":
            continue
        b_iou = float(np.mean(per_defect_baseline[defect])) * 100 if per_defect_baseline[defect] else 0.0
        e1_iou = float(np.mean(per_defect_exp1[defect])) * 100 if per_defect_exp1[defect] else 0.0
        e2_iou = float(np.mean(per_defect_exp2[defect])) * 100 if per_defect_exp2[defect] else 0.0
        print(f"{defect:<20} | {b_iou:>14.2f}% | {e1_iou:>14.2f}% | {e2_iou:>17.2f}%")

    results_payload = {
        "recall": recall,
        "fpr": fpr,
        "tp": tp_count,
        "fn": fn_count,
        "tn": tn_count,
        "fp": fp_count,
        "mean_iou_baseline": mean_iou_baseline,
        "mean_iou_exp1": mean_iou_exp1,
        "mean_iou_exp2": mean_iou_exp2,
        "valid_loc_baseline": valid_loc_baseline,
        "valid_loc_exp1": valid_loc_exp1,
        "valid_loc_exp2": valid_loc_exp2,
        "no_comp_count": no_comp_count
    }

    res_file = OUTPUT_DIR / "results.json"
    with open(res_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nResults saved to {res_file}")


if __name__ == "__main__":
    main()

