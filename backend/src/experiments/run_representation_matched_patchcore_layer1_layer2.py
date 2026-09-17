"""
Representation-Matched PatchCore -- Layer1 + Layer2 Experiment
===============================================================
Controlled experiment evaluating multi-layer feature fusion:
layers = ["layer1", "layer2"]

Strict Fair-Comparison Rules:
- Same 320 GOOD images
- Same seed = 42
- Same 250 train / 70 held-out GOOD split
- Same canonical product preprocessing (5% margin, RGB 202 background padding)
- Same 256x256 resolution
- Same 5% coreset ratio, 9 neighbors
- Same WideResNet50_2 backbone
- Same 119 defective test images & GT masks

EXPERIMENT ONLY -- NO PRODUCTION CHANGES MADE.
"""

import sys
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_representation_matched_patchcore import (
    compute_object_bbox_robust,
    preprocess_and_canonicalize_crop,
    PreprocessedImageDataset,
    collate_fn,
    compute_iou,
    DEFAULT_TARGET_SIZE,
    SAFETY_MARGIN_RATIO,
    REF_BG_COLOR_BGR,
    set_seed
)

try:
    from anomalib.data import ImageBatch
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False

PATCHCORE_LAYER1_LAYER2_CONFIG = {
    "name": "patchcore",
    "backbone": "wide_resnet50_2",
    "layers": ["layer1", "layer2"],
    "coreset_sampling_ratio": 0.05,
    "num_neighbors": 9,
    "image_size": [256, 256],
    "pretrained": True
}


def main():
    start_total_time = time.time()
    set_seed(42)

    run_id = f"run_{int(time.time())}"
    output_dir = Path(f"backend/storage/experiments/representation_matched_patchcore_layer1_layer2/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================================")
    print("  REPRESENTATION-MATCHED PATCHCORE -- LAYER1 + LAYER2 EXPERIMENT                         ")
    print("==========================================================================================")

    dataset_base = Path("backend/data/mvtec_anomaly_detection/screw")
    train_good_dir = dataset_base / "train" / "good"

    if not train_good_dir.exists():
        print(f"[ERROR] Dataset directory not found: {train_good_dir}")
        return

    good_paths = sorted([p for p in train_good_dir.glob("*.png")])
    total_good = len(good_paths)

    rng = random.Random(42)
    shuffled_good = list(good_paths)
    rng.shuffle(shuffled_good)

    train_count = 250
    heldout_count = total_good - train_count

    train_good_paths = shuffled_good[:train_count]
    heldout_good_paths = shuffled_good[train_count:]

    print(f"Dataset Split: {train_count} Training GOOD images | {heldout_count} Held-out Validation GOOD images")
    print(f"Configuration: Backbone = wide_resnet50_2 | Layers = ['layer1', 'layer2'] | Coreset = 5% | Neighbors = 9")
    print("------------------------------------------------------------------------------------------")

    # 1. Build PatchCore Memory Bank (Layer1 + Layer2)
    t_build_start = time.time()
    from anomalib.models.components.sampling import KCenterGreedy
    def fast_gpu_select_coreset_idxs(self) -> list[int]:
        self.model.fit(self.embedding)
        self.features = self.model.transform(self.embedding)
        if torch.cuda.is_available():
            self.features = self.features.to("cuda")

        selected_coreset_idxs: list[torch.Tensor] = []
        idx = torch.randint(high=self.n_observations, size=(1,), device=self.features.device)[0]
        selected_coreset_idxs.append(idx)
        self.reset_distances()
        self.update_distances(cluster_center=idx.item())

        self.min_distances.scatter_(0, idx.reshape(1, 1), 0.0)

        for _ in range(self.coreset_size - 1):
            idx = self.get_new_idx()
            self.update_distances(cluster_center=idx.item())
            self.min_distances.scatter_(0, idx.reshape(1, 1), 0.0)
            selected_coreset_idxs.append(idx)

        return torch.stack(selected_coreset_idxs).cpu().tolist()

    KCenterGreedy.select_coreset_idxs = fast_gpu_select_coreset_idxs

    model = Patchcore(
        backbone=PATCHCORE_LAYER1_LAYER2_CONFIG["backbone"],
        layers=PATCHCORE_LAYER1_LAYER2_CONFIG["layers"],
        pre_trained=PATCHCORE_LAYER1_LAYER2_CONFIG["pretrained"],
        coreset_sampling_ratio=PATCHCORE_LAYER1_LAYER2_CONFIG["coreset_sampling_ratio"],
        num_neighbors=PATCHCORE_LAYER1_LAYER2_CONFIG["num_neighbors"],
    )
    accelerator_mode = "gpu" if torch.cuda.is_available() else "auto"
    engine = Engine(accelerator=accelerator_mode, devices=1, enable_progress_bar=False)

    train_loader = DataLoader(
        PreprocessedImageDataset(train_good_paths),
        batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    engine.fit(model=model, train_dataloaders=train_loader)
    model.post_processor = None
    build_time_sec = round(time.time() - t_build_start, 2)
    print(f"Memory bank build completed in {build_time_sec:.2f} seconds.")

    ckpt_file = output_dir / "patchcore_layer1_layer2.ckpt"
    torch.save(model.state_dict(), ckpt_file)

    # 2. Out-of-Sample GOOD Calibration (70 Held-Out GOOD Images)
    print("\nEvaluating 70 Held-Out GOOD Validation images for calibration...")
    heldout_loader = DataLoader(
        PreprocessedImageDataset(heldout_good_paths),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    t_calib_start = time.time()
    heldout_preds = engine.predict(model=model, dataloaders=heldout_loader)
    heldout_infer_time_sec = round(time.time() - t_calib_start, 2)

    heldout_scores = []
    for b in heldout_preds:
        if hasattr(b, "pred_score"):
            heldout_scores.append(float(b.pred_score[0]))
        elif isinstance(b, list) and len(b) > 0 and hasattr(b[0], "pred_score"):
            heldout_scores.append(float(b[0].pred_score[0]))
    calib_mean = float(np.mean(heldout_scores))
    calib_median = float(np.median(heldout_scores))
    calib_std = float(np.std(heldout_scores))
    calib_min = float(np.min(heldout_scores))
    calib_max = float(np.max(heldout_scores))
    calibrated_threshold = round(float(np.percentile(heldout_scores, 95)), 2)

    good_fp = [s for s in heldout_scores if s > calibrated_threshold]
    good_pass = [s for s in heldout_scores if s <= calibrated_threshold]
    good_fpr = len(good_fp) / float(len(heldout_scores))

    print(f"Calibration (N={len(heldout_scores)}): Mean={calib_mean:.2f} | Median={calib_median:.2f} | Std={calib_std:.2f} | Range=[{calib_min:.2f}, {calib_max:.2f}]")
    print(f"Calibrated 95th Percentile Threshold: {calibrated_threshold}")
    print(f"Held-Out GOOD Validation: PASS = {len(good_pass)}/{len(heldout_scores)} | False Positives = {len(good_fp)}/{len(heldout_scores)} | FPR = {good_fpr:.2%}")

    # 3. Labeled Defective Test Image Evaluation (119 Images)
    test_dir = dataset_base / "test"
    gt_dir = dataset_base / "ground_truth"
    defect_categories = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]

    defect_results = []
    defect_scores_all = []
    total_tp = 0
    total_fn = 0

    iou_scores = []
    pred_areas = []
    gt_areas = []
    overlap_count = 0

    t_defect_start = time.time()

    print("\nEvaluating Labeled Defective Test Dataset (Layer1 + Layer2)...")

    for cat in defect_categories:
        cat_dir = test_dir / cat
        cat_gt_dir = gt_dir / cat
        if not cat_dir.exists(): continue

        cat_images = sorted([p for p in cat_dir.glob("*.png")])
        cat_scores = []
        cat_tps = 0
        cat_fns = 0

        for img_p in cat_images:
            img_loader = DataLoader(
                PreprocessedImageDataset([img_p]),
                batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
            )
            pred = engine.predict(model=model, dataloaders=img_loader)
            score = float(pred[0].pred_score[0])
            cat_scores.append(score)
            defect_scores_all.append(score)

            if score > calibrated_threshold:
                cat_tps += 1
                total_tp += 1
            else:
                cat_fns += 1
                total_fn += 1

            # Localization IoU
            gt_mask_p = cat_gt_dir / f"{img_p.stem}_mask.png"
            if gt_mask_p.exists() and hasattr(pred[0], "anomaly_map"):
                am = pred[0].anomaly_map[0].detach().cpu().numpy()
                if am.ndim == 3: am = am[0]

                gt_mask = cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    img_bgr = cv2.imread(str(img_p))
                    bbox = compute_object_bbox_robust(img_bgr)
                    mask_crop_256 = preprocess_and_canonicalize_crop(
                        cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR),
                        bbox, bg_color=(0, 0, 0)
                    )
                    mask_bin = (cv2.cvtColor(mask_crop_256, cv2.COLOR_BGR2GRAY) > 128)

                    am_norm = (am - am.min()) / (am.max() - am.min() + 1e-8)
                    am_bin = am_norm > 0.5

                    p_area = int(np.sum(am_bin))
                    g_area = int(np.sum(mask_bin))
                    pred_areas.append(p_area)
                    gt_areas.append(g_area)

                    inter = np.logical_and(am_bin, mask_bin).sum()
                    if inter > 0: overlap_count += 1

                    iou = compute_iou(am_bin, mask_bin)
                    iou_scores.append(iou)

        cat_recall = cat_tps / float(len(cat_images)) if cat_images else 0.0
        defect_results.append({
            "category": cat,
            "count": len(cat_images),
            "tp": cat_tps,
            "fn": cat_fns,
            "recall": round(cat_recall, 4),
            "mean_score": round(float(np.mean(cat_scores)), 2),
            "median_score": round(float(np.median(cat_scores)), 2),
            "min_score": round(float(np.min(cat_scores)), 2),
            "max_score": round(float(np.max(cat_scores)), 2),
        })
        print(f"  [{cat:18s}] Count={len(cat_images):2d} | TP={cat_tps:2d} | FN={cat_fns:2d} | Recall={cat_recall:6.2%} | Mean Score={np.mean(cat_scores):5.2f}")

    defect_infer_time_sec = round(time.time() - t_defect_start, 2)
    avg_infer_latency_ms = round((defect_infer_time_sec / float(len(defect_scores_all))) * 1000, 1)

    total_defects = total_tp + total_fn
    overall_recall = total_tp / float(total_defects) if total_defects > 0 else 0.0
    mean_defect_score = float(np.mean(defect_scores_all)) if defect_scores_all else 0.0

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    median_iou = float(np.median(iou_scores)) if iou_scores else 0.0

    mean_pred_area = float(np.mean(pred_areas)) if pred_areas else 0.0
    mean_gt_area = float(np.mean(gt_areas)) if gt_areas else 0.0
    overlap_rate = float(overlap_count / len(iou_scores)) if iou_scores else 0.0

    # Baseline (Layer2) for Direct Side-by-Side Comparison
    baseline_metrics = {
        "good_fpr": 0.0571,
        "calibrated_threshold": 25.63,
        "defect_recall": 0.8992,
        "mean_iou": 0.0369,
        "thread_side_recall": 0.6957,
        "build_time_sec": 76.5,
        "avg_infer_latency_ms": 115.0
    }

    # Side-by-Side Comparison Table
    ts_layer12 = next(r for r in defect_results if r["category"] == "thread_side")

    comparison_table = {
        "good_fpr": {"layer2": "5.71% (4/70)", "layer1_layer2": f"{good_fpr:.2%} ({len(good_fp)}/{heldout_count})"},
        "threshold": {"layer2": "25.63", "layer1_layer2": str(calibrated_threshold)},
        "defect_recall": {"layer2": "89.92% (107/119)", "layer1_layer2": f"{overall_recall:.2%} ({total_tp}/{total_defects})"},
        "mean_iou": {"layer2": "3.69%", "layer1_layer2": f"{mean_iou:.2%}"},
        "thread_side_recall": {"layer2": "69.57% (16/23)", "layer1_layer2": f"{ts_layer12['recall']:.2%} ({ts_layer12['tp']}/{ts_layer12['count']})"},
        "build_time_sec": {"layer2": "76.5 s", "layer1_layer2": f"{build_time_sec:.2f} s"},
        "avg_infer_latency_ms": {"layer2": "~115 ms", "layer1_layer2": f"{avg_infer_latency_ms:.1f} ms"}
    }

    # Save Machine-Readable JSON Report
    report_data = {
        "experiment_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 42,
        "patchcore": PATCHCORE_LAYER1_LAYER2_CONFIG,
        "calibration": {
            "method": "95th_percentile_out_of_sample_good",
            "threshold": calibrated_threshold,
            "mean": round(calib_mean, 2),
            "std": round(calib_std, 2),
            "min": round(calib_min, 2),
            "max": round(calib_max, 2)
        },
        "good_validation": {
            "total": heldout_count,
            "pass_count": len(good_pass),
            "fp_count": len(good_fp),
            "fpr": round(good_fpr, 4),
            "mean_score": round(calib_mean, 2),
            "median_score": round(calib_median, 2),
            "std": round(calib_std, 2)
        },
        "defect_evaluation": {
            "total_defects": total_defects,
            "tp": total_tp,
            "fn": total_fn,
            "recall": round(overall_recall, 4),
            "mean_score": round(mean_defect_score, 2),
            "categories": defect_results
        },
        "localization": {
            "evaluated_masks": len(iou_scores),
            "mean_iou": round(mean_iou, 4),
            "median_iou": round(median_iou, 4),
            "mean_pred_mask_area_px": round(mean_pred_area, 1),
            "mean_gt_mask_area_px": round(mean_gt_area, 1),
            "overlap_rate": round(overlap_rate, 4)
        },
        "performance": {
            "build_time_sec": build_time_sec,
            "total_inference_time_sec": defect_infer_time_sec,
            "avg_infer_latency_ms": avg_infer_latency_ms
        },
        "direct_comparison_against_layer2_baseline": comparison_table,
        "production_status": "UNTOUCHED -- EXPERIMENT ONLY"
    }

    report_file = output_dir / "representation_matched_layer1_layer2_report.json"
    report_file.write_text(json.dumps(report_data, indent=2))

    print("\n==========================================================================================")
    print("  DIRECT SIDE-BY-SIDE COMPARISON: LAYER2 BASELINE vs LAYER1+LAYER2                       ")
    print("==========================================================================================")
    print(f"  Metric                      Layer2 Baseline         Layer1 + Layer2 Experiment          ")
    print(f"  ----------------------------------------------------------------------------------------")
    print(f"  GOOD FPR                    {comparison_table['good_fpr']['layer2']:23s} {comparison_table['good_fpr']['layer1_layer2']}")
    print(f"  Calibrated Threshold        {comparison_table['threshold']['layer2']:23s} {comparison_table['threshold']['layer1_layer2']}")
    print(f"  Defect Recall               {comparison_table['defect_recall']['layer2']:23s} {comparison_table['defect_recall']['layer1_layer2']}")
    print(f"  Mean Pixel IoU              {comparison_table['mean_iou']['layer2']:23s} {comparison_table['mean_iou']['layer1_layer2']}")
    print(f"  Thread-Side Recall          {comparison_table['thread_side_recall']['layer2']:23s} {comparison_table['thread_side_recall']['layer1_layer2']}")
    print(f"  Build Time                  {comparison_table['build_time_sec']['layer2']:23s} {comparison_table['build_time_sec']['layer1_layer2']}")
    print(f"  Inference Latency           {comparison_table['avg_infer_latency_ms']['layer2']:23s} {comparison_table['avg_infer_latency_ms']['layer1_layer2']}")
    print("==========================================================================================")
    print(f"  Report written to: '{report_file}'.")
    print("\n==========================================================================================")
    print("  EXPERIMENT ONLY -- NO PRODUCTION CHANGES MADE                                           ")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
