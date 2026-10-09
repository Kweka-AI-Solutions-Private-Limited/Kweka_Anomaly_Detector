"""
Debug & Smoke Test for Gemini Crop Robustness Diagnostic
---------------------------------------------------------
1. Fixes image border clamping bug by padding input image with a 200px neutral canvas
   so loose bounding box padding (+5%, +10%, +20%, +30%) expands freely without clamping.
2. Saves debug transformed images for 2 GOOD and 2 DEFECT samples under:
   backend/storage/experiments/gemini_crop_robustness_debug/
3. Logs exact image difference metrics (dimensions, mean absolute pixel error, % pixels changed)
   against CONTROL to prove inputs are distinct.
4. Executes a small smoke test (2 GOOD, 2 DEFECT) on CONTROL vs LOOSE_10 using frozen checkpoint
   `full_val_dedicated_patchcore.ckpt` and frozen threshold tau = 18.01.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "data" / "mvtec_anomaly_detection" / "screw"
DEBUG_DIR = BASE_DIR / "storage" / "experiments" / "gemini_crop_robustness_debug"
CKPT_PATH = BASE_DIR / "storage" / "experiments" / "full_validation_multiscale_instance_patchcore" / "full_val_dedicated_patchcore.ckpt"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def extract_gt_screw_tight_bbox(img_bgr: np.ndarray):
    """Extracts tight bounding box without arbitrary initial padding."""
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
        return (bx, by, bw, bh)
    else:
        h, w = img_bgr.shape[:2]
        return (w//4, h//4, w//2, h//2)


def get_unclamped_perturbed_crop(img_bgr: np.ndarray, gt_bbox: tuple, condition: str):
    """
    Pads image with neutral canvas before cropping so expanding boxes (+10%, +20%, +30%)
    do NOT clamp to array boundaries and remain strictly distinct.
    """
    pad_len = 250
    # Pad original image with dark canvas [30, 30, 30] or replicate for original background
    if condition == "BACKGROUND_ORIG":
        padded_img = cv2.copyMakeBorder(img_bgr, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_REPLICATE)
    else:
        padded_img = cv2.copyMakeBorder(img_bgr, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_CONSTANT, value=[30, 30, 30])

    bx, by, bw, bh = gt_bbox
    # Offset bbox by pad_len
    pbx, pby = bx + pad_len, by + pad_len

    if condition == "CONTROL":
        return padded_img[pby:pby+bh, pbx:pbx+bw]

    elif condition.startswith("LOOSE_"):
        pct = float(condition.split("_")[1]) / 100.0
        pad_x = int(bw * pct)
        pad_y = int(bh * pct)
        nx = pbx - pad_x
        ny = pby - pad_y
        nw = bw + 2 * pad_x
        nh = bh + 2 * pad_y
        return padded_img[ny:ny+nh, nx:nx+nw]

    elif condition.startswith("TIGHT_"):
        pct = float(condition.split("_")[1]) / 100.0
        cut_x = int(bw * pct)
        cut_y = int(bh * pct)
        nx = pbx + cut_x
        ny = pby + cut_y
        nw = max(10, bw - 2 * cut_x)
        nh = max(10, bh - 2 * cut_y)
        return padded_img[ny:ny+nh, nx:nx+nw]

    elif condition == "TRANSLATION":
        shift_x = int(bw * 0.10)
        shift_y = int(bh * 0.10)
        nx = pbx + shift_x
        ny = pby + shift_y
        return padded_img[ny:ny+bh, nx:nx+bw]

    elif condition == "SCALE_80":
        cr = padded_img[pby:pby+bh, pbx:pbx+bw]
        scaled = cv2.resize(cr, (max(10, int(bw * 0.8)), max(10, int(bh * 0.8))), interpolation=cv2.INTER_LINEAR)
        pad_h = (bh - scaled.shape[0]) // 2
        pad_w = (bw - scaled.shape[1]) // 2
        padded = cv2.copyMakeBorder(scaled, pad_h, bh - scaled.shape[0] - pad_h, pad_w, bw - scaled.shape[1] - pad_w, cv2.BORDER_CONSTANT, value=[30, 30, 30])
        return padded

    elif condition == "ROTATION_5":
        h_p, w_p = padded_img.shape[:2]
        center = (pbx + bw // 2, pby + bh // 2)
        M = cv2.getRotationMatrix2D(center, 5.0, 1.0)
        rotated = cv2.warpAffine(padded_img, M, (w_p, h_p), borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))
        return rotated[pby:pby+bh, pbx:pbx+bw]

    elif condition == "BACKGROUND_ORIG":
        pad_x = int(bw * 0.25)
        pad_y = int(bh * 0.25)
        nx = pbx - pad_x
        ny = pby - pad_y
        nw = bw + 2 * pad_x
        nh = bh + 2 * pad_y
        return padded_img[ny:ny+nh, nx:nx+nw]

    else:
        return padded_img[pby:pby+bh, pbx:pbx+bw]


def preprocess_canonical(crop_bgr: np.ndarray, target_size=(512, 512)) -> np.ndarray:
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


class FastRobustnessPatchCoreEvaluator:
    def __init__(self, ckpt_path: Path, device="cpu"):
        self.device = torch.device(device)
        self.target_size = (512, 512)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        memory_bank_np = ckpt["memory_bank"]
        self.memory_bank_torch = torch.from_numpy(memory_bank_np).to(self.device)

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
        backbone.layer3.register_forward_hook(hook_l3)
        self.backbone = backbone

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.feat_l2 = None
        self.feat_l3 = None
        _ = self.backbone(tensor)

        l2_map = self.feat_l2
        l3_map = self.feat_l3
        l3_upsampled = nn.functional.interpolate(l3_map, size=(l2_map.shape[2], l2_map.shape[3]), mode='bilinear', align_corners=False)
        fused_map = torch.cat([l2_map, l3_upsampled], dim=1)

        b, c, h, w = fused_map.shape
        flat_feats = fused_map[0].permute(1, 2, 0).reshape(-1, c)
        norms = torch.norm(flat_feats, p=2, dim=-1, keepdim=True)
        norms[norms < 1e-8] = 1.0
        norm_feats = flat_feats / norms

        sims = torch.mm(norm_feats, self.memory_bank_torch.T)
        max_sims, _ = torch.max(sims, dim=1)
        l2_dists = torch.sqrt(torch.clamp(2.0 * (1.0 - max_sims), min=0.0))

        raw_score = float(torch.max(l2_dists).item()) * 25.0
        return raw_score


def main():
    print("=" * 80)
    print("DEBUG & SMOKE TEST FOR GEMINI CROP ROBUSTNESS DIAGNOSTIC")
    print("=" * 80)

    # 1. Select 2 GOOD and 2 DEFECT images
    good_dir = DATASET_DIR / "test" / "good"
    sample_good = sorted(list(good_dir.glob("*.png")))[:2]

    sample_defects = [
        DATASET_DIR / "test" / "manipulated_front" / "000.png",
        DATASET_DIR / "test" / "scratch_head" / "000.png"
    ]

    all_samples = [(p, "GOOD") for p in sample_good] + [(p, "DEFECT") for p in sample_defects]

    conditions = [
        "CONTROL",
        "LOOSE_5",
        "LOOSE_10",
        "LOOSE_20",
        "TIGHT_10",
        "TRANSLATION",
        "SCALE_80",
        "BACKGROUND_ORIG",
        "ROTATION_5"
    ]

    print("\n[STEP 1] Generating transformed images & checking pixel differences vs CONTROL...")
    diff_records = []

    for p, status in all_samples:
        img_bgr = cv2.imread(str(p))
        gt_bbox = extract_gt_screw_tight_bbox(img_bgr)

        # Generate CONTROL image
        ctrl_crop = get_unclamped_perturbed_crop(img_bgr, gt_bbox, "CONTROL")
        ctrl_canon = preprocess_canonical(ctrl_crop, target_size=(512, 512))

        # Save CONTROL sample
        cv2.imwrite(str(DEBUG_DIR / f"{p.stem}_{status}_CONTROL.png"), ctrl_canon)

        for cond in conditions:
            if cond == "CONTROL":
                continue
            cond_crop = get_unclamped_perturbed_crop(img_bgr, gt_bbox, cond)
            cond_canon = preprocess_canonical(cond_crop, target_size=(512, 512))

            # Save transformed debug image
            cv2.imwrite(str(DEBUG_DIR / f"{p.stem}_{status}_{cond}.png"), cond_canon)

            # Compute pixel differences vs CONTROL
            abs_diff = np.abs(cond_canon.astype(np.float32) - ctrl_canon.astype(np.float32))
            mean_abs_diff = float(np.mean(abs_diff))
            pct_changed = float(np.sum(abs_diff > 1.0) / abs_diff.size) * 100.0

            diff_records.append({
                "sample": f"{p.stem}_{status}",
                "condition": cond,
                "crop_shape": list(cond_crop.shape),
                "canon_shape": list(cond_canon.shape),
                "mean_abs_pixel_diff": round(mean_abs_diff, 2),
                "pct_pixels_changed": round(pct_changed, 2)
            })

            print(f"  - [{p.stem}_{status}] CONTROL vs {cond:16s} | Crop Shape: {str(cond_crop.shape):14s} | Mean Abs Pixel Diff: {mean_abs_diff:6.2f} | Pixels Changed: {pct_changed:5.1f}%")

    # 2. Run Small Smoke Test on PatchCore (2 GOOD, 2 DEFECT)
    print("\n[STEP 2] Running PatchCore Smoke Test on CONTROL vs LOOSE_10...")
    evaluator = FastRobustnessPatchCoreEvaluator(ckpt_path=CKPT_PATH, device="cpu")
    frozen_tau = 18.01

    smoke_results = []
    for p, status in all_samples:
        img_bgr = cv2.imread(str(p))
        gt_bbox = extract_gt_screw_tight_bbox(img_bgr)

        ctrl_crop = get_unclamped_perturbed_crop(img_bgr, gt_bbox, "CONTROL")
        ctrl_canon = preprocess_canonical(ctrl_crop)
        score_ctrl = evaluator.predict(ctrl_canon)

        loose10_crop = get_unclamped_perturbed_crop(img_bgr, gt_bbox, "LOOSE_10")
        loose10_canon = preprocess_canonical(loose10_crop)
        score_loose10 = evaluator.predict(loose10_canon)

        tight10_crop = get_unclamped_perturbed_crop(img_bgr, gt_bbox, "TIGHT_10")
        tight10_canon = preprocess_canonical(tight10_crop)
        score_tight10 = evaluator.predict(tight10_canon)

        smoke_results.append({
            "sample": f"{p.stem}_{status}",
            "status": status,
            "score_CONTROL": round(score_ctrl, 2),
            "score_LOOSE_10": round(score_loose10, 2),
            "score_TIGHT_10": round(score_tight10, 2),
            "score_diff_LOOSE10_minus_CTRL": round(score_loose10 - score_ctrl, 2)
        })

        print(f"  - [{p.stem}_{status:6s}] CONTROL: {score_ctrl:5.2f} | LOOSE_10: {score_loose10:5.2f} | TIGHT_10: {score_tight10:5.2f} | Diff (LOOSE10 - CTRL): {score_loose10 - score_ctrl:+5.2f}")

    # Verify that scores are DIFFERENT
    scores_diff = [r["score_diff_LOOSE10_minus_CTRL"] for r in smoke_results]
    is_distinct = any(abs(d) > 0.01 for d in scores_diff)

    print("\n" + "=" * 80)
    print("DEBUG & SMOKE TEST VERDICT")
    print("=" * 80)
    if is_distinct:
        print("PASS: CONTROL and LOOSE_10 produce DISTINCT transformed inputs and INDEPENDENT scores!")
        print("The unclamped canvas padding fix successfully resolves the identical score issue.")
    else:
        print("FAIL: Scores remain identical.")

    debug_output = {
        "pixel_difference_verification": diff_records,
        "smoke_test_scores": smoke_results,
        "smoke_test_passed": is_distinct
    }

    with open(DEBUG_DIR / "debug_report.json", "w") as f:
        json.dump(debug_output, f, indent=2)

    print(f"Debug report saved to: {DEBUG_DIR / 'debug_report.json'}")


if __name__ == "__main__":
    main()
