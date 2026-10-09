"""
Final Validation Experiment Script: Pipeline A Version #2 vs Version #3 (Product-Centric)
-----------------------------------------------------------------------------------------
Executes a clean, non-destructive comparative evaluation of:
  - Version #2 Production Baseline (Raw Full Images, Calibrated Threshold: 28.06)
  - Version #3 Product-Centric Pipeline (v3_product_centric_r2, Calibrated Threshold: 23.06)

Evaluates:
  1. Standard MVTec Screw Test Samples (GOOD + all defect categories: bent, defect_thread, head_damaged, scratch, manipulated_front)
  2. Perturbed GOOD Samples (position shift, scale change, background change, widescreen 16:9, lighting change, boundary touch)
  3. Perturbed DEFECT Samples (position shift, background change)

Logs per-sample anomaly scores, thresholds, score/threshold ratios, verdicts, localization metadata, latency breakdowns,
and saves visualization artifacts comparing [Original Image | V3 Crop | V3 Canonical 256x256].
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2

# Add src to sys.path
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
from services.product_localization_service import (
    localize_and_normalize_product,
    detect_product_bbox,
    estimate_border_background_color
)

# Threshold definitions
V2_CALIBRATED_THRESHOLD = 28.06
V2_DB_TEST_FIXTURE_THRESHOLD = 24.54
V3_CALIBRATED_THRESHOLD = 23.06


def create_perturbed_image(
    src_path: Path,
    out_dir: Path,
    mode: str = "shift",
    shift_x: int = 35,
    shift_y: int = -25,
    scale: float = 1.0,
    bg_color: tuple = (45, 50, 55),
    target_canvas_size: tuple = None,
    gamma: float = 1.0
) -> Path:
    """Creates synthetic spatial, scale, background, dimension, and lighting perturbed test images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(src_path))
    if img_bgr is None:
        return src_path

    h, w = img_bgr.shape[:2]

    if mode == "widescreen":
        canvas_w, canvas_h = (640, 360)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, :] = bg_color
        
        # Scale img_bgr to fit inside 300x300 canvas area while preserving aspect ratio
        scale_f = min(300.0 / float(w), 300.0 / float(h))
        nw, nh = int(round(w * scale_f)), int(round(h * scale_f))
        resized = cv2.resize(img_bgr, (nw, nh))

        start_x = (canvas_w - nw) // 2 + 60
        start_y = (canvas_h - nh) // 2
        canvas[start_y : start_y + nh, start_x : start_x + nw] = resized
        out_path = out_dir / f"perturbed_{mode}_{src_path.name}"
        cv2.imwrite(str(out_path), canvas)
        return out_path

    if mode == "lighting":
        # Apply gamma lighting shift
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        lit = cv2.LUT(img_bgr, table)
        out_path = out_dir / f"perturbed_{mode}_{src_path.name}"
        cv2.imwrite(str(out_path), lit)
        return out_path

    # Spatial shift / background change
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = bg_color

    M = np.array([[scale, 0, shift_x], [0, scale, shift_y]], dtype=np.float32)
    shifted = cv2.warpAffine(img_bgr, M, (w, h), borderValue=bg_color)

    out_path = out_dir / f"perturbed_{mode}_{src_path.name}"
    cv2.imwrite(str(out_path), shifted)
    return out_path


def create_visualization_collage(
    orig_path: Path,
    crop_rgb: np.ndarray,
    canonical_rgb: np.ndarray,
    out_path: Path
):
    """Saves a 3-panel visualization collage: [Original | V3 Crop | V3 Canonical 256x256]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_bgr = cv2.imread(str(orig_path))
    if orig_bgr is None:
        return

    # Resize panels to equal height 256px
    target_h = 256
    w_orig = int(round(orig_bgr.shape[1] * (target_h / float(orig_bgr.shape[0]))))
    panel_orig = cv2.resize(orig_bgr, (w_orig, target_h))

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    w_crop = int(round(crop_bgr.shape[1] * (target_h / float(crop_bgr.shape[0]))))
    panel_crop = cv2.resize(crop_bgr, (w_crop, target_h))

    canonical_bgr = cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2BGR)
    panel_canon = cv2.resize(canonical_bgr, (256, 256))

    # Add header labels
    cv2.putText(panel_orig, "1. Original Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(panel_crop, "2. V3 Localized Crop", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(panel_canon, "3. V3 Canonical 256x256", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    collage = np.hstack([panel_orig, panel_crop, panel_canon])
    cv2.imwrite(str(out_path), collage)


def run_validation():
    print("=" * 90)
    print("FINAL VALIDATION: PIPELINE A VERSION #2 vs VERSION #3 (PRODUCT-CENTRIC)")
    print("=" * 90)

    dataset_screw_dir = BACKEND_DIR / "data" / "mvtec_anomaly_detection" / "screw"
    train_good_dir = dataset_screw_dir / "train" / "good"
    test_dir = dataset_screw_dir / "test"

    out_base_dir = BACKEND_DIR / "outputs" / "experiments" / "pipeline_a_v2_v3_validation"
    v2_art_dir = out_base_dir / "v2_artifacts"
    v3_art_dir = out_base_dir / "v3_artifacts"
    pert_dir = out_base_dir / "perturbed_samples"
    viz_dir = out_base_dir / "visualizations"
    out_base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect GOOD reference paths (20 images)
    ref_paths = sorted(list(train_good_dir.glob("*.png")))[:20]
    print(f"[INFO] Reference images collected: {len(ref_paths)}")

    # 2. Build / Load V2 Model
    print("\n--- Initializing Version #2 Production Baseline ---")
    v2_thr, v2_build_ms, v2_uris = build_patchcore_version(ref_paths, v2_art_dir)
    print(f"V2 Reference-Calibrated Threshold: {v2_thr} (DB Fixture: {V2_DB_TEST_FIXTURE_THRESHOLD})")

    # 3. Build / Load V3 Model (v3_product_centric_r2)
    print("\n--- Initializing Version #3 Product-Centric Pipeline (v3_product_centric_r2) ---")
    v3_thr, v3_build_ms, v3_uris = build_patchcore_v3_version(ref_paths, v3_art_dir)
    print(f"V3 Calibrated Threshold: {v3_thr}")

    # 4. Construct Comprehensive Evaluation Test Set
    test_items: List[Dict[str, Any]] = []

    # Category A: Standard MVTec Test Images (5 per category)
    for cat_dir in sorted(test_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        is_gt_good = (cat_name.lower() == "good")

        for img_p in sorted(cat_dir.glob("*.png"))[:5]:
            test_items.append({
                "path": img_p,
                "category": cat_name,
                "gt_label": "GOOD" if is_gt_good else "DEFECTIVE",
                "variation": "standard_mvtec",
                "description": f"Standard MVTec {cat_name}"
            })

    # Category B: Perturbed Robustness Samples
    sample_good = train_good_dir / "000.png" if (train_good_dir / "000.png").exists() else ref_paths[0]
    sample_defect = test_items[5]["path"] if len(test_items) > 5 else ref_paths[0]

    p_shifted = create_perturbed_image(sample_good, pert_dir, mode="shift", shift_x=40, shift_y=-30, bg_color=(40, 45, 50))
    p_dark_bg = create_perturbed_image(sample_good, pert_dir, mode="dark_bg", shift_x=0, shift_y=0, bg_color=(20, 20, 25))
    p_scaled = create_perturbed_image(sample_good, pert_dir, mode="scaled", shift_x=10, shift_y=10, scale=0.85, bg_color=(50, 50, 50))
    p_widescreen = create_perturbed_image(sample_good, pert_dir, mode="widescreen", bg_color=(55, 60, 65))
    p_lighting = create_perturbed_image(sample_good, pert_dir, mode="lighting", gamma=1.4)
    p_boundary = create_perturbed_image(sample_good, pert_dir, mode="boundary", shift_x=-50, shift_y=-60, bg_color=(45, 45, 45))

    pert_good_specs = [
        (p_shifted, "good_shifted", "Position Shift (X: +40, Y: -30)"),
        (p_dark_bg, "good_dark_bg", "Altered Dark Background"),
        (p_scaled, "good_scaled", "Scale Change (0.85x)"),
        (p_widescreen, "good_widescreen", "Widescreen Aspect Ratio (640x360)"),
        (p_lighting, "good_lighting", "Lighting Illumination Shift (Gamma 1.4)"),
        (p_boundary, "good_boundary", "Near Image Boundary Touch")
    ]

    for p_path, p_cat, p_desc in pert_good_specs:
        test_items.append({
            "path": p_path,
            "category": p_cat,
            "gt_label": "GOOD",
            "variation": "robustness_perturbation",
            "description": p_desc
        })

    # Add perturbed defective sample
    p_def_shifted = create_perturbed_image(sample_defect, pert_dir, mode="defect_shifted", shift_x=-30, shift_y=20, bg_color=(35, 35, 40))
    test_items.append({
        "path": p_def_shifted,
        "category": "defect_shifted",
        "gt_label": "DEFECTIVE",
        "variation": "robustness_perturbation",
        "description": "Defective Screw Shifted + Dark Background"
    })

    print(f"\n[INFO] Total Evaluation Samples: {len(test_items)}")

    # 5. Run Comparative Inference
    v2_results = []
    v3_results = []
    metadata_records = []
    score_comparison_list = []

    v2_latencies = []
    v3_latencies = []
    v3_loc_times = []
    v3_norm_times = []
    v3_infer_times = []

    v2_fp = 0
    v2_fn = 0
    v3_fp = 0
    v3_fn = 0
    loc_fallbacks = 0

    print("\n" + "=" * 115)
    print(f"{'Category':<18} | {'GT':<9} | {'V2 Score (Thr:28.06)':<21} | {'V3 Score (Thr:23.06)':<21} | {'V3 Loc Status':<12} | {'V3 Latency':<10}")
    print("=" * 115)

    for idx, item in enumerate(test_items):
        p_path = item["path"]
        gt = item["gt_label"]
        cat = item["category"]

        # Measure V2 Inference Latency
        t0 = time.time()
        res_v2 = run_patchcore_inference(p_path, v2_uris, v2_thr)
        v2_time_ms = (time.time() - t0) * 1000
        v2_latencies.append(v2_time_ms)

        # Measure V3 Detailed Latency Components
        t_start = time.time()
        t_loc0 = time.time()
        tensor, crop_rgb, transform_meta, loc_st = localize_and_normalize_product(p_path)
        t_loc_ms = (time.time() - t_loc0) * 1000

        t_canon0 = time.time()
        # Preprocessing canonical step inside localize_and_normalize_product
        t_canon_ms = (time.time() - t_canon0) * 1000

        t_inf0 = time.time()
        res_v3 = run_patchcore_v3_inference(p_path, v3_uris, v3_thr)
        t_inf_ms = (time.time() - t_inf0) * 1000
        v3_total_ms = (time.time() - t_start) * 1000

        v3_latencies.append(v3_total_ms)
        v3_loc_times.append(t_loc_ms)
        v3_norm_times.append(t_canon_ms)
        v3_infer_times.append(t_inf_ms)

        v2_pred = "GOOD" if res_v2["status"] == "normal" else "DEFECTIVE"
        v3_pred = "GOOD" if res_v3["status"] == "normal" else "DEFECTIVE"

        if transform_meta["is_fallback"]:
            loc_fallbacks += 1

        # Track error matrix
        if gt == "GOOD":
            if v2_pred == "DEFECTIVE":
                v2_fp += 1
            if v3_pred == "DEFECTIVE":
                v3_fp += 1
        else:
            if v2_pred == "GOOD":
                v2_fn += 1
            if v3_pred == "GOOD":
                v3_fn += 1

        # Score relative to threshold
        v2_ratio = round(res_v2["anomaly_score"] / v2_thr, 4)
        v3_ratio = round(res_v3["anomaly_score"] / v3_thr, 4)

        record_v2 = {
            "sample_name": p_path.name,
            "category": cat,
            "gt_label": gt,
            "anomaly_score": res_v2["anomaly_score"],
            "threshold": v2_thr,
            "ratio_to_threshold": v2_ratio,
            "verdict": v2_pred,
            "latency_ms": round(v2_time_ms, 2)
        }

        record_v3 = {
            "sample_name": p_path.name,
            "category": cat,
            "gt_label": gt,
            "anomaly_score": res_v3["anomaly_score"],
            "threshold": v3_thr,
            "ratio_to_threshold": v3_ratio,
            "verdict": v3_pred,
            "localization_status": loc_st,
            "latency_ms": round(v3_total_ms, 2)
        }

        v2_results.append(record_v2)
        v3_results.append(record_v3)
        metadata_records.append({
            "sample_name": p_path.name,
            "category": cat,
            "gt_label": gt,
            "transformation_metadata": transform_meta
        })

        score_comparison_list.append({
            "sample_name": p_path.name,
            "category": cat,
            "gt_label": gt,
            "v2_score": res_v2["anomaly_score"],
            "v2_threshold": v2_thr,
            "v2_ratio": v2_ratio,
            "v2_verdict": v2_pred,
            "v3_score": res_v3["anomaly_score"],
            "v3_threshold": v3_thr,
            "v3_ratio": v3_ratio,
            "v3_verdict": v3_pred,
            "loc_status": loc_st
        })

        # Save visualization collage for representative samples
        if idx < 12 or "perturbed" in cat or gt == "DEFECTIVE":
            # Extract 256x256 canonical array from tensor for collage
            tensor_np = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            create_visualization_collage(p_path, crop_rgb, tensor_np, viz_dir / f"viz_{cat}_{p_path.stem}.png")

        print(f"{cat:<18} | {gt:<9} | {res_v2['anomaly_score']:<7.2f} ({v2_pred:<9}) | {res_v3['anomaly_score']:<7.2f} ({v3_pred:<9}) | {loc_st:<12} | {v3_total_ms:<10.1f}ms")

    print("=" * 115)

    # 6. Aggregate Statistical Summaries
    total_count = len(test_items)
    good_items = [r for r in score_comparison_list if r["gt_label"] == "GOOD"]
    defect_items = [r for r in score_comparison_list if r["gt_label"] == "DEFECTIVE"]

    good_n = len(good_items)
    defect_n = len(defect_items)

    # V2 GOOD metrics
    v2_good_scores = [r["v2_score"] for r in good_items]
    v2_defect_scores = [r["v2_score"] for r in defect_items]

    # V3 GOOD metrics
    v3_good_scores = [r["v3_score"] for r in good_items]
    v3_defect_scores = [r["v3_score"] for r in defect_items]

    def calc_stats(scores_list):
        if not scores_list:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
        return {
            "min": round(float(np.min(scores_list)), 2),
            "max": round(float(np.max(scores_list)), 2),
            "mean": round(float(np.mean(scores_list)), 2),
            "median": round(float(np.median(scores_list)), 2),
            "std": round(float(np.std(scores_list)), 2)
        }

    summary_payload = {
        "validation_metadata": {
            "v2_checkpoint": v2_uris.get("checkpoint_uri"),
            "v2_calibrated_threshold": v2_thr,
            "v2_db_fixture_threshold": V2_DB_TEST_FIXTURE_THRESHOLD,
            "v3_revision": "v3_product_centric_r2",
            "v3_checkpoint": v3_uris.get("checkpoint_uri"),
            "v3_calibrated_threshold": v3_thr,
            "total_images_tested": total_count,
            "good_count": good_n,
            "defective_count": defect_n
        },
        "classification_metrics": {
            "v2": {
                "good_tested": good_n,
                "false_positives": v2_fp,
                "fpr_percent": round((v2_fp / max(1, good_n)) * 100, 2),
                "defective_tested": defect_n,
                "false_negatives": v2_fn,
                "fnr_percent": round((v2_fn / max(1, defect_n)) * 100, 2),
                "recall_percent": round(((defect_n - v2_fn) / max(1, defect_n)) * 100, 2),
                "accuracy_percent": round(((total_count - (v2_fp + v2_fn)) / total_count) * 100, 2)
            },
            "v3": {
                "good_tested": good_n,
                "false_positives": v3_fp,
                "fpr_percent": round((v3_fp / max(1, good_n)) * 100, 2),
                "defective_tested": defect_n,
                "false_negatives": v3_fn,
                "fnr_percent": round((v3_fn / max(1, defect_n)) * 100, 2),
                "recall_percent": round(((defect_n - v3_fn) / max(1, defect_n)) * 100, 2),
                "accuracy_percent": round(((total_count - (v3_fp + v3_fn)) / total_count) * 100, 2),
                "localization_fallbacks": loc_fallbacks,
                "localization_fallback_percent": round((loc_fallbacks / total_count) * 100, 2)
            }
        },
        "score_statistics": {
            "good_samples": {
                "v2": calc_stats(v2_good_scores),
                "v3": calc_stats(v3_good_scores)
            },
            "defective_samples": {
                "v2": calc_stats(v2_defect_scores),
                "v3": calc_stats(v3_defect_scores)
            }
        },
        "latency_metrics_ms": {
            "v2": {
                "average": round(float(np.mean(v2_latencies)), 2),
                "median": round(float(np.median(v2_latencies)), 2),
                "max": round(float(np.max(v2_latencies)), 2)
            },
            "v3_total": {
                "average": round(float(np.mean(v3_latencies)), 2),
                "median": round(float(np.median(v3_latencies)), 2),
                "max": round(float(np.max(v3_latencies)), 2)
            },
            "v3_breakdown": {
                "localization_avg": round(float(np.mean(v3_loc_times)), 2),
                "normalization_avg": round(float(np.mean(v3_norm_times)), 2),
                "inference_avg": round(float(np.mean(v3_infer_times)), 2)
            }
        }
    }

    # 7. Write Output Files
    with open(out_base_dir / "comparison_results.json", "w") as f:
        json.dump(score_comparison_list, f, indent=2)

    with open(out_base_dir / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    with open(out_base_dir / "transformation_metadata.json", "w") as f:
        json.dump(metadata_records, f, indent=2)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"Total Images Evaluated : {total_count} (GOOD: {good_n}, DEFECTIVE: {defect_n})")
    print(f"V2 Threshold           : {v2_thr} (Ref-Calibrated) / {V2_DB_TEST_FIXTURE_THRESHOLD} (DB Fixture)")
    print(f"V3 Threshold           : {v3_thr} (Calibrated V3 R2)")
    print("-" * 70)
    print(f"V2 False Positives     : {v2_fp} / {good_n} ({summary_payload['classification_metrics']['v2']['fpr_percent']}%)")
    print(f"V2 False Negatives     : {v2_fn} / {defect_n} ({summary_payload['classification_metrics']['v2']['fnr_percent']}%)")
    print(f"V2 Defective Recall    : {summary_payload['classification_metrics']['v2']['recall_percent']}%")
    print(f"V2 Overall Accuracy    : {summary_payload['classification_metrics']['v2']['accuracy_percent']}%")
    print("-" * 70)
    print(f"V3 False Positives     : {v3_fp} / {good_n} ({summary_payload['classification_metrics']['v3']['fpr_percent']}%)")
    print(f"V3 False Negatives     : {v3_fn} / {defect_n} ({summary_payload['classification_metrics']['v3']['fnr_percent']}%)")
    print(f"V3 Defective Recall    : {summary_payload['classification_metrics']['v3']['recall_percent']}%")
    print(f"V3 Overall Accuracy    : {summary_payload['classification_metrics']['v3']['accuracy_percent']}%")
    print(f"V3 Loc Fallbacks       : {loc_fallbacks} / {total_count}")
    print("-" * 70)
    print(f"Avg Latency V2         : {summary_payload['latency_metrics_ms']['v2']['average']} ms")
    print(f"Avg Latency V3 Total   : {summary_payload['latency_metrics_ms']['v3_total']['average']} ms")
    print(f"  └─ Localization Avg  : {summary_payload['latency_metrics_ms']['v3_breakdown']['localization_avg']} ms")
    print(f"  └─ Inference Avg     : {summary_payload['latency_metrics_ms']['v3_breakdown']['inference_avg']} ms")
    print("=" * 70)
    print(f"\n[SUCCESS] Artifacts saved to: {out_base_dir}")


if __name__ == "__main__":
    run_validation()
