"""
demo_patchcore_visualization.py

PatchCore Demo Layer & Visualization Pipeline
----------------------------------------------
- Model Architecture: Patchcore (wide_resnet50_2, layer2, coreset_sampling_ratio=0.05, num_neighbors=9)
- Checkpoint: backend/outputs/patchcore_baseline/FINAL/patchcore_screw_wideresnet50_l2_005.ckpt
- Dataset: MVTec AD screw
- Fixed Test Subset: 6 deterministic samples (1 good + 1 of each of 5 defect types)
- Anomaly Threshold: Frozen at 0.4981 (raw anomaly map -> threshold -> binary mask -> contours)
- Output 1: Senior-Facing Annotated Visualizations (Original, Heatmap, Bounding Box Overlay, Status & Score; NO GT)
- Output 2: Diagnostic 5-Panel Visualizations (Original, Heatmap, GT Mask, Thresholded Mask, Bounding Box Overlay)
"""

import os
from pathlib import Path
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ============================================================
# PATHS & CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "FINAL"
    / "patchcore_screw_wideresnet50_l2_005.ckpt"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "demo_visualizations"
SENIOR_OUTPUT_DIR = OUTPUT_DIR / "senior_facing"
DIAGNOSTIC_OUTPUT_DIR = OUTPUT_DIR / "diagnostic_5panel"

SENIOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FROZEN_THRESHOLD = 0.4981

REQUIRED_CATEGORIES = [
    "good",
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(val):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def main():
    print("=" * 80)
    print("PATCHCORE DEMO LAYER — VISUALIZATION PIPELINE")
    print("=" * 80)
    print(f"Checkpoint Path:     {CHECKPOINT}")
    print(f"Frozen Threshold:    {FROZEN_THRESHOLD}")
    print(f"Target Categories:   {REQUIRED_CATEGORIES}")
    print("-" * 80)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT}")

    set_seed(SEED)

    # 1. Initialize Datamodule & Model
    print("[1/4] Setting up datamodule and PatchCore model...")
    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="screw",
        eval_batch_size=1,
        num_workers=0,
    )
    datamodule.setup()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2"],
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
    )

    # 2. Run Inference
    print("[2/4] Running model inference on test dataset...")
    prediction_batches = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    if not prediction_batches:
        raise RuntimeError("engine.predict() returned no prediction batches.")

    # 3. Extract All Samples
    print("[3/4] Extracting predictions and selecting 6 deterministic test images...")
    all_samples = []
    for batch in prediction_batches:
        image_paths = getattr(batch, "image_path", None)
        if image_paths is None:
            continue

        images = to_numpy(getattr(batch, "image", None))
        anomaly_maps = to_numpy(getattr(batch, "anomaly_map", None))
        gt_masks = to_numpy(getattr(batch, "gt_mask", None))
        gt_labels = to_numpy(getattr(batch, "gt_label", None))
        pred_scores = to_numpy(getattr(batch, "pred_score", None))

        batch_size = len(image_paths)
        for i in range(batch_size):
            path = Path(image_paths[i])
            defect_type = path.parent.name

            amap = anomaly_maps[i] if anomaly_maps is not None else None
            if amap is not None and amap.ndim == 3 and amap.shape[0] == 1:
                amap = amap[0]

            gmask = gt_masks[i] if gt_masks is not None else None
            if gmask is not None and gmask.ndim == 3 and gmask.shape[0] == 1:
                gmask = gmask[0]

            img = images[i] if images is not None else None
            if img is not None and img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            score = float(pred_scores[i].reshape(-1)[0]) if pred_scores is not None else (float(amap.max()) if amap is not None else 0.0)

            all_samples.append({
                "image_path": str(path),
                "defect_type": defect_type,
                "image": img,
                "anomaly_map": amap,
                "gt_mask": gmask,
                "gt_label": bool(gt_labels[i]) if gt_labels is not None else (defect_type != "good"),
                "pred_score": score,
            })

    # Select exactly 1 sample from each required category (deterministic pick)
    selected_samples = {}
    for cat in REQUIRED_CATEGORIES:
        matching = [s for s in all_samples if s["defect_type"] == cat]
        if not matching:
            raise RuntimeError(f"Could not find any test samples for category: {cat}")
        selected_samples[cat] = matching[0]

    print(f"Selected {len(selected_samples)} deterministic test images:")
    for cat, s in selected_samples.items():
        print(f"  - {cat:<20}: {Path(s['image_path']).name} | GT: {'DEFECT' if s['gt_label'] else 'GOOD'}")

    # 4. Generate Visualizations
    print("\n[4/4] Generating senior-facing and diagnostic visualizations...")

    summary_rows = []

    for cat in REQUIRED_CATEGORIES:
        sample = selected_samples[cat]
        img = sample["image"]
        amap = sample["anomaly_map"]
        gt_mask = sample["gt_mask"]
        gt_label = sample["gt_label"]
        pred_score = sample["pred_score"]

        # Raw Thresholding at 0.4981
        if amap is not None:
            binary_mask = (amap >= FROZEN_THRESHOLD).astype(np.uint8)
        else:
            binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Classification Status based on threshold
        has_defect = bool(binary_mask.max() > 0)
        status_str = "DEFECT DETECTED" if has_defect else "PASS (NORMAL)"

        # Prepare Anomaly Heatmap Overlay
        if amap is not None:
            # Min-Max normalize for colormap rendering
            amap_norm = cv2.normalize(amap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_overlay = cv2.addWeighted(img, 0.5, heatmap_rgb, 0.5, 0)
        else:
            heatmap_overlay = img.copy()

        # Prepare Bounding Box & Contour Overlay
        overlay_img = img.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2:  # filter negligible noise points
                box_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red Bounding Box
                cv2.drawContours(overlay_img, [cnt], -1, (255, 255, 0), 1)  # Yellow contour

        # Draw Status Badge on Overlay
        badge_color = (255, 0, 0) if has_defect else (0, 200, 0)
        cv2.rectangle(overlay_img, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(overlay_img, status_str, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

        # -------------------------------------------------------------
        # OUTPUT 1: SENIOR-FACING VISUALIZATION (NO GROUND TRUTH SHOWN)
        # -------------------------------------------------------------
        fig_senior, axes_s = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original Image
        axes_s[0].imshow(img)
        axes_s[0].set_title("1. Input Image", fontsize=12, fontweight="bold")
        axes_s[0].axis("off")

        # 2. Anomaly Heatmap Overlay
        axes_s[1].imshow(heatmap_overlay)
        axes_s[1].set_title("2. Anomaly Heatmap Overlay", fontsize=12, fontweight="bold")
        axes_s[1].axis("off")

        # 3. Detection Bounding Boxes & Mask Overlay
        axes_s[2].imshow(overlay_img)
        axes_s[2].set_title("3. Defect Detection & Bounding Boxes", fontsize=12, fontweight="bold")
        axes_s[2].axis("off")

        fig_senior.suptitle(
            f"Senior Overview — Category: {cat.upper()}\n"
            f"Status: {status_str} | Anomaly Score: {pred_score:.4f} | Threshold: {FROZEN_THRESHOLD}",
            fontsize=13,
            fontweight="bold",
            color="red" if has_defect else "green"
        )
        plt.tight_layout()
        senior_path = SENIOR_OUTPUT_DIR / f"senior_annotated_{cat}.png"
        plt.savefig(senior_path, dpi=150, bbox_inches="tight")
        plt.close(fig_senior)

        # -------------------------------------------------------------
        # OUTPUT 2: DIAGNOSTIC 5-PANEL VISUALIZATION (GT SHOWN ONLY HERE)
        # -------------------------------------------------------------
        fig_diag, axes_d = plt.subplots(1, 5, figsize=(22, 4.5))

        # Panel 1: Original Image
        axes_d[0].imshow(img)
        axes_d[0].set_title("1. Original Image", fontsize=11, fontweight="bold")
        axes_d[0].axis("off")

        # Panel 2: Anomaly Map (Jet)
        if amap is not None:
            axes_d[1].imshow(amap, cmap="jet")
        axes_d[1].set_title("2. Raw Anomaly Map", fontsize=11, fontweight="bold")
        axes_d[1].axis("off")

        # Panel 3: Ground Truth Mask (Comparison only)
        if gt_mask is not None:
            axes_d[2].imshow(gt_mask, cmap="gray")
        else:
            axes_d[2].imshow(np.zeros((img.shape[0], img.shape[1])), cmap="gray")
        axes_d[2].set_title("3. Ground-Truth Mask", fontsize=11, fontweight="bold")
        axes_d[2].axis("off")

        # Panel 4: Thresholded Binary Mask (0.4981)
        axes_d[3].imshow(binary_mask, cmap="gray")
        axes_d[3].set_title(f"4. Mask (Thr={FROZEN_THRESHOLD})", fontsize=11, fontweight="bold")
        axes_d[3].axis("off")

        # Panel 5: Bounding Box & Contour Overlay
        axes_d[4].imshow(overlay_img)
        axes_d[4].set_title("5. Bounding Box Overlay", fontsize=11, fontweight="bold")
        axes_d[4].axis("off")

        fig_diag.suptitle(
            f"Diagnostic 5-Panel View — Category: {cat} | GT: {'DEFECT' if gt_label else 'GOOD'} | Status: {status_str} | Score: {pred_score:.4f}",
            fontsize=13,
            fontweight="bold"
        )
        plt.tight_layout()
        diag_path = DIAGNOSTIC_OUTPUT_DIR / f"diagnostic_5panel_{cat}.png"
        plt.savefig(diag_path, dpi=150, bbox_inches="tight")
        plt.close(fig_diag)

        summary_rows.append({
            "category": cat,
            "gt_label": "DEFECT" if gt_label else "GOOD",
            "pred_status": status_str,
            "anomaly_score": round(pred_score, 4),
            "boxes_found": box_count,
            "senior_image": str(senior_path.relative_to(BASE_DIR)),
            "diagnostic_image": str(diag_path.relative_to(BASE_DIR)),
        })

    # Print Summary Table
    print("\n" + "=" * 80)
    print("PATCHCORE DEMO VISUALIZATION RUN COMPLETE — SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Category':<20} | {'GT Label':<8} | {'Pred Status':<17} | {'Score':<7} | {'Boxes':<5}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['category']:<20} | {r['gt_label']:<8} | {r['pred_status']:<17} | {r['anomaly_score']:<7.4f} | {r['boxes_found']:<5}")
    print("=" * 80)
    print(f"Senior-facing images saved to:     {SENIOR_OUTPUT_DIR}")
    print(f"Diagnostic 5-panel images saved: {DIAGNOSTIC_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
