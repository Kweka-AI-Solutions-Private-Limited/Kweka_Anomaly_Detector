"""
Comparative Experiment: Pipeline A Version #2 (Raw Baseline) vs Version #3 (Product-Centric)
---------------------------------------------------------------------------------------------
Evaluates the effect of product-centric localization on Pipeline A anomaly detection robustness:
  - Trains Version #2 memory bank on raw GOOD reference images + calibrates V2 threshold
  - Trains Version #3 memory bank on product-localized GOOD reference images + calibrates V3 threshold
  - Tests both pipelines on standard MVTec screw test set & position/background-shifted test set
  - Logs anomaly scores, predictions, thresholds, false positives, false negatives, localization failures, and latency
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2

# Add src to path
BACKEND_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BACKEND_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.patchcore_service import (
    build_patchcore_version,
    run_patchcore_inference,
    build_patchcore_v3_version,
    run_patchcore_v3_inference
)
from services.product_localization_service import detect_product_bbox


def create_perturbed_test_image(
    src_path: Path,
    out_dir: Path,
    shift_x: int = 40,
    shift_y: int = -30,
    bg_color: tuple = (50, 55, 60)
) -> Path:
    """Creates a spatial/background perturbed test image to test pipeline robustness."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return src_path

    h, w = img_bgr.shape[:2]
    # Create new background canvas
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = bg_color

    # Shift original image
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(img_bgr, M, (w, h), borderValue=bg_color)
    
    out_path = out_dir / f"perturbed_{src_path.name}"
    cv2.imwrite(str(out_path), shifted)
    return out_path


def run_experiment():
    print("=" * 80)
    print("PIPELINE A VERSION #2 vs VERSION #3 COMPARATIVE EXPERIMENT")
    print("=" * 80)

    dataset_screw_dir = BACKEND_DIR / "data" / "mvtec_anomaly_detection" / "screw"
    train_good_dir = dataset_screw_dir / "train" / "good"
    test_dir = dataset_screw_dir / "test"

    if not train_good_dir.exists():
        print(f"[ERROR] MVTec screw dataset not found at '{train_good_dir}'.")
        return

    # 1. Collect GOOD reference images (use up to 20 images)
    ref_paths = sorted(list(train_good_dir.glob("*.png")))[:20]
    print(f"[INFO] Using {len(ref_paths)} GOOD reference images for memory bank building...")

    exp_artifacts_dir = BACKEND_DIR / "outputs" / "experiments_v2_v3"
    v2_artifacts_dir = exp_artifacts_dir / "v2_artifacts"
    v3_artifacts_dir = exp_artifacts_dir / "v3_artifacts"
    perturbed_dir = exp_artifacts_dir / "perturbed_samples"

    # 2. Build Version #2 (Raw Baseline)
    print("\n--- Building Version #2 Baseline (Raw Full Images) ---")
    v2_thr, v2_build_ms, v2_uris = build_patchcore_version(ref_paths, v2_artifacts_dir)
    print(f"Version #2 Calibrated Threshold: {v2_thr} (Build Time: {v2_build_ms:.1f} ms)")

    # 3. Build Version #3 (Product-Centric Localized)
    print("\n--- Building Version #3 (Product-Centric Localized) ---")
    v3_thr, v3_build_ms, v3_uris = build_patchcore_v3_version(ref_paths, v3_artifacts_dir)
    print(f"Version #3 Calibrated Threshold: {v3_thr} (Build Time: {v3_build_ms:.1f} ms)")

    # 4. Collect Test Set
    test_samples: List[Dict[str, Any]] = []

    # Standard MVTec test categories
    for cat_dir in sorted(test_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        is_gt_good = (cat_name.lower() == "good")

        for img_p in sorted(cat_dir.glob("*.png"))[:5]:  # 5 per category
            test_samples.append({
                "path": img_p,
                "category": cat_name,
                "is_perturbed": False,
                "gt_label": "GOOD" if is_gt_good else "DEFECTIVE"
            })

    # Add perturbed samples (shifted position & altered background) to test robustness
    print("\nGenerating spatial/background perturbed test samples...")
    good_sample = test_samples[0]["path"]
    defect_sample = next(s["path"] for s in test_samples if s["gt_label"] == "DEFECTIVE")

    pert_good = create_perturbed_test_image(good_sample, perturbed_dir, shift_x=35, shift_y=-25, bg_color=(40, 40, 40))
    pert_defect = create_perturbed_test_image(defect_sample, perturbed_dir, shift_x=-30, shift_y=20, bg_color=(30, 35, 45))

    test_samples.append({
        "path": pert_good,
        "category": "good_perturbed",
        "is_perturbed": True,
        "gt_label": "GOOD"
    })
    test_samples.append({
        "path": pert_defect,
        "category": "defect_perturbed",
        "is_perturbed": True,
        "gt_label": "DEFECTIVE"
    })

    print(f"Total Test Samples: {len(test_samples)}")

    # 5. Execute Comparative Inference
    v2_results = []
    v3_results = []

    v2_fp = 0
    v2_fn = 0
    v3_fp = 0
    v3_fn = 0

    loc_failures = 0

    print("\n" + "=" * 105)
    print(f"{'Category':<18} | {'GT':<9} | {'V2 Score':<9} | {'V2 Pred':<8} | {'V3 Score':<9} | {'V3 Pred':<8} | {'Loc Status':<12} | {'V3 Latency':<10}")
    print("=" * 105)

    for item in test_samples:
        path = item["path"]
        gt = item["gt_label"]
        cat = item["category"]

        # Run Version #2
        t0 = time.time()
        res_v2 = run_patchcore_inference(path, v2_uris, v2_thr)
        v2_time = (time.time() - t0) * 1000

        # Run Version #3
        t0 = time.time()
        res_v3 = run_patchcore_v3_inference(path, v3_uris, v3_thr)
        v3_time = (time.time() - t0) * 1000

        v2_pred = "GOOD" if res_v2["status"] == "normal" else "DEFECTIVE"
        v3_pred = "GOOD" if res_v3["status"] == "normal" else "DEFECTIVE"

        loc_st = res_v3["localization_status"]
        if "FALLBACK" in loc_st:
            loc_failures += 1

        # Track error matrix
        if gt == "GOOD" and v2_pred == "DEFECTIVE":
            v2_fp += 1
        if gt == "DEFECTIVE" and v2_pred == "GOOD":
            v2_fn += 1

        if gt == "GOOD" and v3_pred == "DEFECTIVE":
            v3_fp += 1
        if gt == "DEFECTIVE" and v3_pred == "GOOD":
            v3_fn += 1

        v2_results.append({
            "sample": path.name,
            "category": cat,
            "gt": gt,
            "score": res_v2["anomaly_score"],
            "pred": v2_pred,
            "latency_ms": v2_time
        })

        v3_results.append({
            "sample": path.name,
            "category": cat,
            "gt": gt,
            "score": res_v3["anomaly_score"],
            "pred": v3_pred,
            "loc_status": loc_st,
            "latency_ms": v3_time
        })

        print(f"{cat:<18} | {gt:<9} | {res_v2['anomaly_score']:<9.2f} | {v2_pred:<8} | {res_v3['anomaly_score']:<9.2f} | {v3_pred:<8} | {loc_st:<12} | {v3_time:<10.1f}ms")

    print("=" * 105)

    # 6. Aggregate Summary
    total_samples = len(test_samples)
    good_count = sum(1 for s in test_samples if s["gt_label"] == "GOOD")
    defect_count = sum(1 for s in test_samples if s["gt_label"] == "DEFECTIVE")

    avg_v2_time = float(np.mean([r["latency_ms"] for r in v2_results]))
    avg_v3_time = float(np.mean([r["latency_ms"] for r in v3_results]))

    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Test Images evaluated : {total_samples} (GOOD: {good_count}, DEFECTIVE: {defect_count})")
    print(f"Version #2 Threshold       : {v2_thr}")
    print(f"Version #3 Threshold       : {v3_thr}")
    print("-" * 70)
    print(f"Version #2 False Positives : {v2_fp} / {good_count} (FPR: {v2_fp/good_count*100:.1f}%)")
    print(f"Version #2 False Negatives : {v2_fn} / {defect_count} (FNR: {v2_fn/defect_count*100:.1f}%)")
    print("-" * 70)
    print(f"Version #3 False Positives : {v3_fp} / {good_count} (FPR: {v3_fp/good_count*100:.1f}%)")
    print(f"Version #3 False Negatives : {v3_fn} / {defect_count} (FNR: {v3_fn/defect_count*100:.1f}%)")
    print(f"Version #3 Loc Fallbacks   : {loc_failures} / {total_samples}")
    print("-" * 70)
    print(f"Avg Inference Latency V2   : {avg_v2_time:.1f} ms")
    print(f"Avg Inference Latency V3   : {avg_v3_time:.1f} ms")
    print("=" * 70)

    # Save summary JSON
    results_json_path = exp_artifacts_dir / "v2_vs_v3_comparison.json"
    summary_data = {
        "v2": {
            "threshold": v2_thr,
            "false_positives": v2_fp,
            "false_negatives": v2_fn,
            "avg_latency_ms": round(avg_v2_time, 2)
        },
        "v3": {
            "threshold": v3_thr,
            "false_positives": v3_fp,
            "false_negatives": v3_fn,
            "localization_fallbacks": loc_failures,
            "avg_latency_ms": round(avg_v3_time, 2)
        },
        "test_sample_count": total_samples,
        "v2_results": v2_results,
        "v3_results": v3_results
    }
    with open(results_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[SUCCESS] Experiment results saved to: {results_json_path}")


if __name__ == "__main__":
    run_experiment()
