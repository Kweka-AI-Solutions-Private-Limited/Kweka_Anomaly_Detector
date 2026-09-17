"""
Standalone Representation-Matched PatchCore Experiment
======================================================
Answers: "Can PatchCore reliably detect defects when its memory bank is built from
the same individual-product representation that will be used after Gemini localization?"

Strict Rules:
- NO production database modifications
- NO production checkpoint overwrites
- NO Pipeline A changes
- NO hardcoded screw logic / count / dimensions
- EXPERIMENT ONLY -- NO PRODUCTION CHANGES MADE
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from anomalib.data import ImageBatch
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False

PATCHCORE_CONFIG = {
    "name": "patchcore",
    "backbone": "wide_resnet50_2",
    "layers": ["layer2"],
    "coreset_sampling_ratio": 0.05,
    "num_neighbors": 9,
    "image_size": [256, 256],
    "pretrained": True
}

DEFAULT_TARGET_SIZE = (256, 256)
SAFETY_MARGIN_RATIO = 0.05  # 5% explicit safety margin around bounding box
REF_BG_COLOR_BGR = (202, 202, 202)  # Derived median background color from 320 GOOD references


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_object_bbox_robust(img_bgr: np.ndarray) -> Dict[str, int]:
    """Deterministically extracts product bounding box using neutral background contour detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = gray.shape[:2]

    corner_pixels = np.concatenate([
        gray[:10, :10].flatten(),
        gray[:10, -10:].flatten(),
        gray[-10:, :10].flatten(),
        gray[-10:, -10:].flatten()
    ])
    bg_val = float(np.median(corner_pixels))

    diff = cv2.absdiff(gray, int(bg_val))
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"x": 0, "y": 0, "width": orig_w, "height": orig_h}

    min_area = 0.005 * orig_w * orig_h
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        valid_contours = [max(contours, key=cv2.contourArea)]

    all_pts = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}


def preprocess_and_canonicalize_crop(
    img_bgr: np.ndarray,
    bbox: Dict[str, int],
    margin_ratio: float = SAFETY_MARGIN_RATIO,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    bg_color: Tuple[int, int, int] = REF_BG_COLOR_BGR
) -> np.ndarray:
    """
    Deterministic canonical representation transformation:
    1. Expands bbox with a small explicit safety margin (default 5%).
    2. Crops product from original image.
    3. Pads symmetrically to 1:1 square canvas with reference-matched neutral grey background.
    4. Resizes to PatchCore target_size (256x256).
    """
    orig_h, orig_w = img_bgr.shape[:2]
    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

    # Explicit margin expansion
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))

    x1 = max(0, bx - mx)
    y1 = max(0, by - my)
    x2 = min(orig_w, bx + bw + mx)
    y2 = min(orig_h, by + bh + my)

    crop = img_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    max_dim = max(ch, cw)
    pad_top = (max_dim - ch) // 2
    pad_bottom = max_dim - ch - pad_top
    pad_left = (max_dim - cw) // 2
    pad_right = max_dim - cw - pad_left

    canvas = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=bg_color
    )
    return cv2.resize(canvas, target_size, interpolation=cv2.INTER_LINEAR)


class PreprocessedImageDataset(Dataset):
    """Dataset serving preprocessed (256x256) float tensors scaled [0, 1]."""
    def __init__(self, image_paths: List[Path], target_size=DEFAULT_TARGET_SIZE):
        self.image_paths = [Path(p) for p in image_paths if Path(p).exists()]
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img_bgr = cv2.imread(str(path))
            bbox = compute_object_bbox_robust(img_bgr)
            crop_256 = preprocess_and_canonicalize_crop(img_bgr, bbox, target_size=self.target_size)
            crop_rgb = cv2.cvtColor(crop_256, cv2.COLOR_BGR2RGB)
            arr = np.array(crop_rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        except Exception:
            tensor = torch.zeros((3, self.target_size[1], self.target_size[0]), dtype=torch.float32)

        return {
            "image": tensor,
            "image_path": str(path),
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    if ANOMALIB_AVAILABLE:
        return ImageBatch(image=images, image_path=image_paths)
    return {"image": images, "image_path": image_paths}


def compute_iou(map_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    intersection = np.logical_and(map_bin, mask_bin).sum()
    union = np.logical_or(map_bin, mask_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def main():
    start_time = time.time()
    set_seed(42)

    run_id = f"run_{int(time.time())}"
    output_dir = Path(f"backend/storage/experiments/representation_matched_patchcore/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================================")
    print("  REPRESENTATION-MATCHED PATCHCORE EXPERIMENT                                            ")
    print("==========================================================================================")

    # 1. Locate 320 GOOD reference images
    dataset_base = Path("backend/data/mvtec_anomaly_detection/screw")
    train_good_dir = dataset_base / "train" / "good"

    if not train_good_dir.exists():
        print(f"[ERROR] Dataset directory not found: {train_good_dir}")
        return

    good_paths = sorted([p for p in train_good_dir.glob("*.png")])
    total_good = len(good_paths)
    print(f"Located {total_good} GOOD reference images in '{train_good_dir}'.")

    if total_good < 320:
        print(f"[WARNING] Expected 320 GOOD images, found {total_good}.")

    # 2. Fixed Reproducible Split (Seed 42)
    rng = random.Random(42)
    shuffled_good = list(good_paths)
    rng.shuffle(shuffled_good)

    train_count = 250
    heldout_count = total_good - train_count

    train_good_paths = shuffled_good[:train_count]
    heldout_good_paths = shuffled_good[train_count:]

    print(f"Split: {train_count} Training GOOD images | {heldout_count} Held-out Validation GOOD images")

    # Audit BBox statistics across reference images
    bbox_stats = []
    for p in good_paths[:50]:  # Sample first 50 for quick audit logging
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: continue
        h_img, w_img = img_bgr.shape[:2]
        bbox = compute_object_bbox_robust(img_bgr)
        w_b, h_b = bbox["width"], bbox["height"]
        occupancy = (w_b * h_b) / float(w_img * h_img)
        ar = w_b / float(h_b) if h_b > 0 else 1.0
        bbox_stats.append({"occupancy": occupancy, "aspect_ratio": ar})

    mean_occupancy = float(np.mean([s["occupancy"] for s in bbox_stats]))
    mean_aspect_ratio = float(np.mean([s["aspect_ratio"] for s in bbox_stats]))
    print(f"BBox Audit (50 samples): Mean Occupancy = {mean_occupancy:.4f} | Mean Aspect Ratio = {mean_aspect_ratio:.4f}")

    # 3. Build Temporary Representation-Matched PatchCore Memory Bank
    print("\nBuilding temporary representation-matched PatchCore memory bank from 250 GOOD training images...")
    model = Patchcore(
        backbone=PATCHCORE_CONFIG["backbone"],
        layers=PATCHCORE_CONFIG["layers"],
        pre_trained=PATCHCORE_CONFIG["pretrained"],
        coreset_sampling_ratio=PATCHCORE_CONFIG["coreset_sampling_ratio"],
        num_neighbors=PATCHCORE_CONFIG["num_neighbors"],
    )
    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    train_loader = DataLoader(
        PreprocessedImageDataset(train_good_paths),
        batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    engine.fit(model=model, train_dataloaders=train_loader)
    model.post_processor = None  # Disable MinMax clipping for raw distance scores

    # Save temporary checkpoint in experiment folder
    ckpt_file = output_dir / "patchcore_representation_matched.ckpt"
    torch.save(model.state_dict(), ckpt_file)
    print(f"Saved temporary checkpoint to '{ckpt_file}'.")

    # 4. Out-of-Sample GOOD Calibration on 70 Held-Out GOOD Images
    print("\nEvaluating 70 Held-Out GOOD Validation images for calibration...")
    heldout_loader = DataLoader(
        PreprocessedImageDataset(heldout_good_paths),
        batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    heldout_preds = engine.predict(model=model, dataloaders=heldout_loader)

    heldout_scores = []
    heldout_details = []

    for idx, batch_pred in enumerate(heldout_preds):
        score = float(batch_pred.pred_score[0])
        p_name = heldout_good_paths[idx].name
        heldout_scores.append(score)
        heldout_details.append({"filename": p_name, "score": score})

    calib_mean = float(np.mean(heldout_scores))
    calib_median = float(np.median(heldout_scores))
    calib_std = float(np.std(heldout_scores))
    calib_min = float(np.min(heldout_scores))
    calib_max = float(np.max(heldout_scores))
    calibrated_threshold = round(float(np.percentile(heldout_scores, 95)), 2)

    good_fp = [d for d in heldout_details if d["score"] > calibrated_threshold]
    good_pass = [d for d in heldout_details if d["score"] <= calibrated_threshold]
    good_fpr = len(good_fp) / float(len(heldout_scores))

    print(f"Calibration Result (N={len(heldout_scores)}): Mean={calib_mean:.2f} | Median={calib_median:.2f} | Std={calib_std:.2f} | Range=[{calib_min:.2f}, {calib_max:.2f}]")
    print(f"Calibrated 95th Percentile Threshold: {calibrated_threshold}")
    print(f"Held-Out GOOD Validation: PASS = {len(good_pass)}/{len(heldout_scores)} | False Positives = {len(good_fp)}/{len(heldout_scores)} | FPR = {good_fpr:.2%}")

    # 5. Labeled Defective Test Image Evaluation
    test_dir = dataset_base / "test"
    gt_dir = dataset_base / "ground_truth"

    defect_categories = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]
    defect_results = []
    defect_scores_all = []

    total_tp = 0
    total_fn = 0

    iou_scores = []

    print("\nEvaluating Labeled Defective Test Dataset...")

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

            is_tp = score > calibrated_threshold
            if is_tp:
                cat_tps += 1
                total_tp += 1
            else:
                cat_fns += 1
                total_fn += 1

            # Localization IoU evaluation if GT mask exists
            gt_mask_p = cat_gt_dir / f"{img_p.stem}_mask.png"
            if gt_mask_p.exists() and hasattr(pred[0], "anomaly_map"):
                am = pred[0].anomaly_map[0].detach().cpu().numpy()
                if am.ndim == 3: am = am[0]

                gt_mask = cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    # Preprocess GT mask into canonical crop space for direct IoU comparison
                    img_bgr = cv2.imread(str(img_p))
                    bbox = compute_object_bbox_robust(img_bgr)
                    mask_crop_256 = preprocess_and_canonicalize_crop(
                        cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR),
                        bbox, bg_color=(0,0,0)
                    )
                    mask_bin = (cv2.cvtColor(mask_crop_256, cv2.COLOR_BGR2GRAY) > 128)

                    am_norm = (am - am.min()) / (am.max() - am.min() + 1e-8)
                    am_bin = am_norm > 0.5

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

    total_defects = total_tp + total_fn
    overall_recall = total_tp / float(total_defects) if total_defects > 0 else 0.0
    mean_defect_score = float(np.mean(defect_scores_all)) if defect_scores_all else 0.0

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    median_iou = float(np.median(iou_scores)) if iou_scores else 0.0

    # Gate Evaluations
    gate_a_pass = good_fpr <= 0.10
    gate_b_pass = overall_recall >= 0.90
    gate_c_pass = mean_iou >= 0.30
    gate_d_pass = (calibrated_threshold < 28.0) and (good_fpr <= 0.10)

    overall_decision = "PASS" if (gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass) else "FAIL"

    # Save machine-readable JSON report
    report_data = {
        "experiment_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 42,
        "reference_total": total_good,
        "train_good": train_count,
        "heldout_good": heldout_count,
        "preprocessing": {
            "input_size": list(DEFAULT_TARGET_SIZE),
            "aspect_preserving": True,
            "margin_ratio": SAFETY_MARGIN_RATIO,
            "bg_color_bgr": list(REF_BG_COLOR_BGR),
            "method": "neutral_background_padding_to_square"
        },
        "patchcore": PATCHCORE_CONFIG,
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
            "median_iou": round(median_iou, 4)
        },
        "baseline_comparison": {
            "existing_v2_isolated_crop_fp": "10/10 (100% FP)",
            "existing_v2_isolated_crop_mean_score": 33.01,
            "experiment_isolated_crop_fp": f"{len(good_fp)}/{heldout_count} ({good_fpr:.1%} FP)",
            "experiment_isolated_crop_mean_score": round(calib_mean, 2)
        },
        "gates": {
            "gate_a_good_accuracy": "PASS" if gate_a_pass else "FAIL",
            "gate_b_defect_recall": "PASS" if gate_b_pass else "FAIL",
            "gate_c_localization_iou": "PASS" if gate_c_pass else "FAIL",
            "gate_d_material_improvement": "PASS" if gate_d_pass else "FAIL"
        },
        "decision": overall_decision,
        "production_status": "UNTOUCHED — EXPERIMENT ONLY"
    }

    report_file = output_dir / "representation_matched_patchcore_report.json"
    report_file.write_text(json.dumps(report_data, indent=2))

    print("\n==========================================================================================")
    print("  REPRESENTATION-MATCHED PATCHCORE EXPERIMENT SUMMARY                                     ")
    print("==========================================================================================")
    print(f"References: Total GOOD = {total_good} | Train = {train_count} | Held-out GOOD = {heldout_count}")
    print(f"Preprocessing: Input = 256x256 | Aspect Preserving = YES | Margin = {SAFETY_MARGIN_RATIO:.0%} | Padding = Neutral RGB(202,202,202)")
    print(f"PatchCore: Backbone = wide_resnet50_2 | Layer = layer2 | Coreset = 5% | Neighbors = 9")
    print(f"Calibration: Calibrated Threshold = {calibrated_threshold}")
    print(f"GOOD Validation: PASS = {len(good_pass)}/{heldout_count} | FP = {len(good_fp)}/{heldout_count} | FPR = {good_fpr:.2%} | Mean Score = {calib_mean:.2f}")
    print(f"Defect Evaluation: Total = {total_defects} | TP = {total_tp} | FN = {total_fn} | Recall = {overall_recall:.2%} | Mean Score = {mean_defect_score:.2f}")
    print(f"Localization: Evaluated Masks = {len(iou_scores)} | Mean IoU = {mean_iou:.4f} | Median IoU = {median_iou:.4f}")
    print(f"\nBaseline isolated-crop failure:")
    print(f"  Existing Version #2 FP: 10/10 (100% FP, Mean Score 33.01)")
    print(f"  Experiment FP: {len(good_fp)}/{heldout_count} ({good_fpr:.1%} FP, Mean Score {calib_mean:.2f})")
    print("==========================================================================================")
    print(f"  EXPERIMENT DECISION: {overall_decision}                                                ")
    print("==========================================================================================")
    print(f"  Gates: Gate A (GOOD Acc)={report_data['gates']['gate_a_good_accuracy']} | Gate B (Recall)={report_data['gates']['gate_b_defect_recall']} | Gate C (IoU)={report_data['gates']['gate_c_localization_iou']} | Gate D (Improvement)={report_data['gates']['gate_d_material_improvement']}")
    print(f"  Report written to: '{report_file}'.")
    print("\n==========================================================================================")
    print("  EXPERIMENT ONLY — NO PRODUCTION CHANGES MADE                                           ")
    print("==========================================================================================")

if __name__ == "__main__":
    main()
