"""
Multi-Scale High-Resolution Instance PatchCore Diagnostic Experiment
----------------------------------------------------------------------
Tests Hypothesis:
  Higher-resolution (512x512) inputs and multi-scale feature aggregation
  (concatenating WideResNet-50-2 layer2 + upsampled layer3 patch descriptors)
  substantially improve instance-level defect detection recall and localization IoU.

Production Safety:
  Strictly standalone research script.
  Does NOT modify production PatchCore, thresholds, memory banks, APIs, DB, or frontend.
"""

import sys
import os
import time
import json
import random
import math
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

# Set seeds for strict reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "data" / "mvtec_anomaly_detection" / "screw"

OUTPUT_DIR = BASE_DIR / "storage" / "experiments" / "multiscale_instance_patchcore"
VISUALS_DIR = OUTPUT_DIR / "visuals"
CROPS_DIR = BASE_DIR / "storage" / "experiments" / "gemini_isolated_reference_comparison_diagnostic" / "crops"

for d in [OUTPUT_DIR, VISUALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Preprocessing Utilities
# ------------------------------------------------------------------------------
def extract_screw_crop_from_full_image(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = [c for c in contours if cv2.contourArea(c) > 5000]

    if valid_cnts:
        cnt = max(valid_cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        bx = max(0, bx - 10)
        by = max(0, by - 10)
        bw = min(w - bx, bw + 20)
        bh = min(h - by, bh + 20)
        return img_bgr[by:by+bh, bx:bx+bw]
    else:
        return img_bgr[h//4:3*h//4, w//4:3*w//4]


def preprocess_canonical_instance_crop(crop_bgr: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    max_dim = max(h, w)

    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    padded = cv2.copyMakeBorder(
        crop_bgr,
        top=pad_h,
        bottom=max_dim - (h + pad_h),
        left=pad_w,
        right=max_dim - (w + pad_w),
        borderType=cv2.BORDER_CONSTANT,
        value=[30, 30, 30]
    )

    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_CUBIC)
    return resized


# ------------------------------------------------------------------------------
# 2. Multi-Scale PatchCore Feature Extractor Engine
# ------------------------------------------------------------------------------
class MultiScaleInstancePatchCore:
    def __init__(self, mode="single_256_l2", device="cpu"):
        """
        mode options:
          - 'single_256_l2': 256x256 input, layer2 only (d=512, 32x32)
          - 'single_512_l2': 512x512 input, layer2 only (d=512, 64x64)
          - 'multiscale_512_l2l3': 512x512 input, layer2 + upsampled layer3 (d=1536, 64x64)
        """
        self.mode = mode
        self.device = torch.device(device)
        self.target_size = (512, 512) if "512" in mode else (256, 256)
        
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        backbone.eval()
        backbone.to(self.device)

        self.feat_l2 = None
        self.feat_l3 = None

        def hook_l2(module, input, output):
            self.feat_l2 = output

        def hook_l3(module, input, output):
            self.feat_l3 = output

        backbone.layer2.register_forward_hook(hook_l2)
        if "l2l3" in mode:
            backbone.layer3.register_forward_hook(hook_l3)

        self.backbone = backbone

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.memory_bank = None

    @torch.no_grad()
    def extract_patch_features(self, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.feat_l2 = None
        self.feat_l3 = None
        _ = self.backbone(tensor)

        l2_map = self.feat_l2 # (1, 512, H2, W2)
        
        if "l2l3" in self.mode:
            l3_map = self.feat_l3 # (1, 1024, H3, W3)
            # Upsample layer3 feature map to layer2 spatial resolution
            l3_upsampled = nn.functional.interpolate(l3_map, size=(l2_map.shape[2], l2_map.shape[3]), mode='bilinear', align_corners=False)
            fused_map = torch.cat([l2_map, l3_upsampled], dim=1) # (1, 1536, H2, W2)
        else:
            fused_map = l2_map # (1, 512, H2, W2)

        feat_map = fused_map[0].cpu().numpy().transpose(1, 2, 0) # (H2, W2, D)
        h, w, c = feat_map.shape
        flat_feats = feat_map.reshape(-1, c)

        norms = np.linalg.norm(flat_feats, axis=-1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        norm_feats = flat_feats / norms
        return norm_feats, feat_map.shape

    def build_memory_bank(self, canonical_crops: list, coreset_ratio: float = 0.05):
        patch_list = []
        for cr in canonical_crops:
            feats, _ = self.extract_patch_features(cr)
            patch_list.append(feats)
        all_patches = np.vstack(patch_list)

        num_coreset = int(len(all_patches) * coreset_ratio)
        np.random.seed(42)
        indices = np.random.choice(len(all_patches), num_coreset, replace=False)
        self.memory_bank = all_patches[indices]

    def predict(self, img_bgr: np.ndarray):
        patch_feats, map_shape = self.extract_patch_features(img_bgr)
        h, w, c = map_shape

        sims = np.dot(patch_feats, self.memory_bank.T)
        max_sims = np.max(sims, axis=1)
        l2_dists = np.sqrt(np.maximum(0, 2.0 * (1.0 - max_sims)))

        patch_score_map = l2_dists.reshape(h, w)
        raw_score = float(np.max(l2_dists)) * 25.0

        upsampled_map = cv2.resize(patch_score_map, (self.target_size[0], self.target_size[1]), interpolation=cv2.INTER_CUBIC)
        upsampled_map = ndimage.gaussian_filter(upsampled_map, sigma=4.0)

        return raw_score, upsampled_map


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / float(union))


# ------------------------------------------------------------------------------
# 3. Main Multi-Variant Diagnostic Execution
# ------------------------------------------------------------------------------
def run_diagnostic():
    print("=" * 80)
    print("MULTI-SCALE HIGH-RESOLUTION INSTANCE PATCHCORE DIAGNOSTIC EXPERIMENT")
    print("=" * 80)

    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_good_dir = DATASET_DIR / "train" / "good"
    test_good_dir = DATASET_DIR / "test" / "good"

    train_good_paths = sorted(list(train_good_dir.glob("*.png")))[:30]
    test_good_paths = sorted(list(test_good_dir.glob("*.png")))[:10]

    defect_cats = ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]
    defect_eval_samples = []
    for cat in defect_cats:
        d_paths = sorted(list((DATASET_DIR / "test" / cat).glob("*.png")))[:2]
        for p in d_paths:
            defect_eval_samples.append((p, cat))

    variants = [
        {"id": "Variant_A", "mode": "single_256_l2", "size": (256, 256), "desc": "Single-Scale 256x256 layer2 (d=512, 32x32)"},
        {"id": "Variant_B", "mode": "single_512_l2", "size": (512, 512), "desc": "High-Res 512x512 layer2 (d=512, 64x64)"},
        {"id": "Variant_C", "mode": "multiscale_512_l2l3", "size": (512, 512), "desc": "Multi-Scale High-Res 512x512 layer2+layer3 (d=1536, 64x64)"}
    ]

    variant_results = {}

    for var in variants:
        v_id = var["id"]
        v_mode = var["mode"]
        v_size = var["size"]
        v_desc = var["desc"]

        print(f"\n" + "-" * 70)
        print(f"EVALUATING {v_id}: {v_desc}")
        print("-" * 70)

        patchcore = MultiScaleInstancePatchCore(mode=v_mode, device=device)

        # 1. Preprocess 30 GOOD training crops for this resolution
        train_crops = []
        for p in train_good_paths:
            img_bgr = cv2.imread(str(p))
            cr_bgr = extract_screw_crop_from_full_image(img_bgr)
            canon_cr = preprocess_canonical_instance_crop(cr_bgr, target_size=v_size)
            train_crops.append(canon_cr)

        # 2. Build Memory Bank
        t0 = time.time()
        patchcore.build_memory_bank(train_crops, coreset_ratio=0.05)
        print(f"  - [{v_id}] Built Memory Bank ({patchcore.memory_bank.shape[0]} vectors, d={patchcore.memory_bank.shape[1]}) in {time.time()-t0:.2f}s")

        # 3. Calibration on 10 held-out GOOD crops
        calib_scores = []
        for p in test_good_paths:
            img_bgr = cv2.imread(str(p))
            cr_bgr = extract_screw_crop_from_full_image(img_bgr)
            canon_cr = preprocess_canonical_instance_crop(cr_bgr, target_size=v_size)
            sc, _ = patchcore.predict(canon_cr)
            calib_scores.append(sc)

        p95_tau = round(float(np.percentile(calib_scores, 95)), 2)
        good_mean = round(float(np.mean(calib_scores)), 2)
        print(f"  - [{v_id}] Mean GOOD Score: {good_mean:.2f} | Frozen P95 Threshold (tau): {p95_tau}")

        # 4. Defect Evaluation & Localization
        tp_count = 0
        ious = []
        cat_metrics = {c: {"eval": 0, "tp": 0, "ious": []} for c in defect_cats}

        for idx, (p, cat) in enumerate(defect_eval_samples):
            img_bgr = cv2.imread(str(p))
            h_orig, w_orig = img_bgr.shape[:2]

            gt_mask_path = DATASET_DIR / "ground_truth" / cat / f"{p.stem}_mask.png"
            if not gt_mask_path.exists():
                gt_mask_path = DATASET_DIR / "ground_truth" / cat / f"{p.stem}.png"

            if gt_mask_path.exists():
                cv_gt = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                ys, xs = np.where(cv_gt > 128)
                if len(xs) > 0 and len(ys) > 0:
                    bx, by, bw, bh = np.min(xs), np.min(ys), np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                    bx = max(0, bx - 30)
                    by = max(0, by - 30)
                    bw = min(w_orig - bx, bw + 60)
                    bh = min(h_orig - by, bh + 60)
                    crop_bgr = img_bgr[by:by+bh, bx:bx+bw]
                    gt_crop = cv_gt[by:by+bh, bx:bx+bw]
                else:
                    crop_bgr = extract_screw_crop_from_full_image(img_bgr)
                    gt_crop = cv2.resize(cv_gt, (crop_bgr.shape[1], crop_bgr.shape[0])) if cv_gt is not None else np.zeros((crop_bgr.shape[0], crop_bgr.shape[1]), dtype=np.uint8)
            else:
                crop_bgr = extract_screw_crop_from_full_image(img_bgr)
                gt_crop = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1]), dtype=np.uint8)

            canon_cr = preprocess_canonical_instance_crop(crop_bgr, target_size=v_size)

            max_dim = max(crop_bgr.shape[0], crop_bgr.shape[1])
            pad_h = (max_dim - crop_bgr.shape[0]) // 2
            pad_w = (max_dim - crop_bgr.shape[1]) // 2
            gt_padded = cv2.copyMakeBorder(gt_crop, pad_h, max_dim - (crop_bgr.shape[0] + pad_h), pad_w, max_dim - (crop_bgr.shape[1] + pad_w), cv2.BORDER_CONSTANT, value=0)
            gt_canon = cv2.resize(gt_padded, v_size, interpolation=cv2.INTER_NEAREST)

            sc, amap = patchcore.predict(canon_cr)
            is_tp = sc >= p95_tau
            if is_tp:
                tp_count += 1
                cat_metrics[cat]["tp"] += 1
            cat_metrics[cat]["eval"] += 1

            thresh_val = np.percentile(amap, 75)
            pred_mask = (amap > thresh_val).astype(np.uint8) * 255
            sample_iou = compute_iou(pred_mask, gt_canon)
            ious.append(sample_iou)
            cat_metrics[cat]["ious"].append(sample_iou)

        defect_recall = round((tp_count / float(len(defect_eval_samples))) * 100.0, 2)
        good_fps = sum([1 for s in calib_scores if s >= p95_tau])
        good_fpr = round((good_fps / float(len(calib_scores))) * 100.0, 2)
        mean_iou = round(float(np.mean(ious)), 4)
        hits_10 = sum([1 for i in ious if i >= 0.10])
        hit_rate_10 = round((hits_10 / float(len(ious))) * 100.0, 2)
        hits_20 = sum([1 for i in ious if i >= 0.20])
        hit_rate_20 = round((hits_20 / float(len(ious))) * 100.0, 2)

        print(f"  - [{v_id}] Defect Recall: {defect_recall}% ({tp_count}/10) | GOOD FPR: {good_fpr}% ({good_fps}/10)")
        print(f"  - [{v_id}] Mean Map IoU:   {mean_iou:.4f} ({mean_iou*100.0:.2f}%) | Hit Rate (IoU>=0.10): {hit_rate_10}%")

        variant_results[v_id] = {
            "description": v_desc,
            "feature_dim": patchcore.memory_bank.shape[1],
            "input_size": v_size[0],
            "good_mean_score": good_mean,
            "p95_tau": p95_tau,
            "defect_recall_percent": defect_recall,
            "good_fpr_percent": good_fpr,
            "mean_iou": mean_iou,
            "median_iou": round(float(np.median(ious)), 4),
            "hit_rate_10_percent": hit_rate_10,
            "hit_rate_20_percent": hit_rate_20,
            "per_category": {
                c: {
                    "recall": round((cat_metrics[c]["tp"] / float(cat_metrics[c]["eval"])) * 100.0, 2),
                    "mean_iou": round(float(np.mean(cat_metrics[c]["ious"])), 4)
                } for c in defect_cats
            }
        }

    # 5. Comparative Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    v_names = ["A: 256x256 L2", "B: 512x512 L2", "C: 512x512 L2+L3"]
    recalls = [variant_results[v]["defect_recall_percent"] for v in ["Variant_A", "Variant_B", "Variant_C"]]
    ious_pct = [variant_results[v]["mean_iou"] * 100.0 for v in ["Variant_A", "Variant_B", "Variant_C"]]

    bars1 = ax[0].bar(v_names, recalls, color=['#3498db', '#9b59b6', '#2ecc71'])
    ax[0].set_ylabel("Defect Recall (%)")
    ax[0].set_title("Defect Recall by Architecture Variant")
    ax[0].set_ylim(0, 100)
    for bar in bars1:
        yval = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    bars2 = ax[1].bar(v_names, ious_pct, color=['#3498db', '#9b59b6', '#2ecc71'])
    ax[1].set_ylabel("Mean Map IoU (%)")
    ax[1].set_title("Localization IoU by Architecture Variant")
    ax[1].set_ylim(0, 50)
    ax[1].axhline(24.12, color='red', linestyle='--', label='WRN50_2 Baseline (24.12%)')
    ax[1].legend()
    for bar in bars2:
        yval = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "variant_comparison.png", dpi=150)
    plt.close()

    # Determine Verdict
    var_c_recall = variant_results["Variant_C"]["defect_recall_percent"]
    var_c_iou = variant_results["Variant_C"]["mean_iou"]
    var_a_recall = variant_results["Variant_A"]["defect_recall_percent"]

    if var_c_recall >= 70.0 and var_c_iou >= 0.20:
        verdict = "YES — MULTI-SCALE HIGH-RES FEATURES SUBSTANTIALLY IMPROVE INSTANCE DETECTION"
    elif var_c_recall > var_a_recall or var_c_iou > 0.15:
        verdict = "MODEST IMPROVEMENT — HELPS RECALL BUT DOES NOT REACH OPERATIONAL BASELINE"
    else:
        verdict = "NO — MULTI-SCALE HIGH-RES FEATURES DO NOT RESOLVE INSTANCE DETECTION GAP"

    total_runtime = round(time.time() - start_time, 2)

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_name": "multiscale_instance_patchcore",
        "variants_evaluated": variant_results,
        "baseline_operational_reference": {
            "Defect_Recall_Percent": 75.41,
            "Mean_IoU_Percent": 24.12,
            "GOOD_FPR_Percent": 4.76
        },
        "verdict": verdict,
        "total_runtime_seconds": total_runtime
    }

    report_json_path = OUTPUT_DIR / "report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    readme_content = f"""# Multi-Scale High-Resolution Instance PatchCore Report

## Executive Summary
Evaluating whether higher-resolution ($512 \times 512$) spatial inputs and multi-scale feature aggregation (concatenating WideResNet-50-2 `layer2` + upsampled `layer3` patch descriptors) substantially improve instance-level defect detection recall and localization IoU.

- **Final Verdict**: **{verdict}**
- **Variant A (256x256 L2 Baseline)**: Recall = `{variant_results['Variant_A']['defect_recall_percent']}%`, Mean IoU = `{variant_results['Variant_A']['mean_iou']*100.0:.2f}%`
- **Variant B (512x512 L2 High-Res)**: Recall = `{variant_results['Variant_B']['defect_recall_percent']}%`, Mean IoU = `{variant_results['Variant_B']['mean_iou']*100.0:.2f}%`
- **Variant C (512x512 L2+L3 Multi-Scale)**: Recall = `{variant_results['Variant_C']['defect_recall_percent']}%`, Mean IoU = `{variant_results['Variant_C']['mean_iou']*100.0:.2f}%`
- **Total Runtime**: {total_runtime} sec

---

## Controlled 3-Way Architectural Comparison

| Architecture Variant | Input Size | Feature Dim ($d$) | Frozen Tau ($\tau$) | Defect Recall | Mean Map IoU | Hit Rate (IoU $\ge 0.10$) | GOOD FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Operational Full-Scene Baseline** | $256 \times 256$ | 512 | 24.54 | **75.41%** | **24.12%** | — | **4.76%** |
| **Variant A (256x256 L2 Control)** | $256 \times 256$ | 512 | {variant_results['Variant_A']['p95_tau']} | **{variant_results['Variant_A']['defect_recall_percent']}%** | **{variant_results['Variant_A']['mean_iou']*100.0:.2f}%** | {variant_results['Variant_A']['hit_rate_10_percent']}% | **{variant_results['Variant_A']['good_fpr_percent']}%** |
| **Variant B (512x512 L2 High-Res)** | $512 \times 512$ | 512 | {variant_results['Variant_B']['p95_tau']} | **{variant_results['Variant_B']['defect_recall_percent']}%** | **{variant_results['Variant_B']['mean_iou']*100.0:.2f}%** | {variant_results['Variant_B']['hit_rate_10_percent']}% | **{variant_results['Variant_B']['good_fpr_percent']}%** |
| **Variant C (512x512 L2+L3 Fused)** | $512 \times 512$ | 1536 | {variant_results['Variant_C']['p95_tau']} | **{variant_results['Variant_C']['defect_recall_percent']}%** | **{variant_results['Variant_C']['mean_iou']*100.0:.2f}%** | {variant_results['Variant_C']['hit_rate_10_percent']}% | **{variant_results['Variant_C']['good_fpr_percent']}%** |
"""

    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print("\n" + "=" * 80)
    print("MULTI-SCALE HIGH-RES INSTANCE PATCHCORE DIAGNOSTIC COMPLETED SUCCESSFULLY")
    print(f"  - Final Evidence Verdict: {verdict}")
    print(f"  - Report saved to:        {report_json_path}")
    print(f"  - README saved to:        {OUTPUT_DIR / 'README.md'}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic()
