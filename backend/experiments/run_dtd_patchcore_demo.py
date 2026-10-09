"""
run_dtd_patchcore_demo.py

Standalone DTD PatchCore Zero-Adaptation Inference Script (Diagnostic Pass)
-----------------------------------------------------------------------------
- Model Architecture: Patchcore (wide_resnet50_2, layer2, coreset_sampling_ratio=0.05, num_neighbors=9)
- Checkpoint: backend/outputs/patchcore_baseline/FINAL/patchcore_screw_wideresnet50_l2_005.ckpt
- Data Source: Direct PNG images from backend/data/dtd/ (NO MVTec AD Datamodule used)
- Anomaly Threshold: Frozen at 0.4981
- Diagnostics: Temporarily prints raw prediction statistics before visualization normalization.
"""

import os
import csv
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ============================================================
# PATHS & CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DTD_DATA_DIR = BASE_DIR / "data" / "dtd"
CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "FINAL"
    / "patchcore_screw_wideresnet50_l2_005.ckpt"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "dtd_patchcore_demo"
SENIOR_OUTPUT_DIR = OUTPUT_DIR / "senior_facing"
DIAGNOSTIC_OUTPUT_DIR = OUTPUT_DIR / "diagnostic_4panel"

SENIOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FROZEN_THRESHOLD = 0.4981
TARGET_SIZE = (256, 256)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CUSTOM PYTORCH DATASET FOR DIRECT DTD IMAGE LOADING
# ============================================================
class DTDDirectDataset(Dataset):
    def __init__(self, dtd_dir, target_size=TARGET_SIZE):
        self.image_paths = sorted(list(Path(dtd_dir).glob("*.png")))
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


def dtd_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    return ImageBatch(image=images, image_path=image_paths)


def to_numpy(val):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def main():
    print("=" * 80)
    print("STANDALONE DTD PATCHCORE ZERO-ADAPTATION INFERENCE RUN (DIAGNOSTIC PASS)")
    print("=" * 80)
    print(f"Data Directory:      {DTD_DATA_DIR}")
    print(f"Checkpoint Path:     {CHECKPOINT}")
    print(f"Frozen Threshold:    {FROZEN_THRESHOLD}")
    print("-" * 80)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT}")

    if not DTD_DATA_DIR.exists():
        raise FileNotFoundError(f"DTD dataset directory not found: {DTD_DATA_DIR}")

    set_seed(SEED)

    # 1. Instantiate Custom Dataset & DataLoader
    print("[1/4] Loading DTD images directly...")
    dataset = DTDDirectDataset(DTD_DATA_DIR, target_size=TARGET_SIZE)
    if len(dataset) == 0:
        raise RuntimeError(f"No .png images found in {DTD_DATA_DIR}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dtd_collate_fn)
    print(f"Loaded {len(dataset)} PNG images from {DTD_DATA_DIR}")

    # 2. Configure PatchCore Model Architecture
    print("[2/4] Initializing PatchCore model (wide_resnet50_2, layer2, 5% coreset, 9 neighbors)...")
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

    # 3. Run PatchCore Inference
    print("[3/4] Running zero-adaptation inference...")
    prediction_batches = engine.predict(
        model=model,
        dataloaders=dataloader,
        ckpt_path=str(CHECKPOINT),
    )

    if not prediction_batches:
        raise RuntimeError("engine.predict() returned no prediction batches.")

    # 4. Process Predictions and Print Diagnostic Raw Statistics
    print("\n" + "=" * 80)
    print("DIAGNOSTIC RAW PREDICTION STATISTICS (BEFORE VISUALIZATION SCALING)")
    print("=" * 80)
    
    summary_rows = []

    for idx, batch in enumerate(prediction_batches):
        paths = getattr(batch, "image_path", None)
        if paths is None:
            continue

        anomaly_maps = to_numpy(getattr(batch, "anomaly_map", None))
        pred_scores = to_numpy(getattr(batch, "pred_score", None))
        
        path_val = paths[0] if isinstance(paths, (list, tuple)) else paths
        path = Path(str(path_val))
        filename = path.name

        amap = anomaly_maps[0] if anomaly_maps is not None else None
        if amap is not None and amap.ndim == 3 and amap.shape[0] == 1:
            amap = amap[0]

        # Load raw RGB image for visualization
        orig_pil = Image.open(path).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
        img_rgb = np.array(orig_pil, dtype=np.uint8)

        score = float(pred_scores[0].reshape(-1)[0]) if pred_scores is not None else (float(amap.max()) if amap is not None else 0.0)

        # Raw Anomaly Map Statistics (BEFORE any visualization normalization)
        if amap is not None:
            amap_min = float(amap.min())
            amap_max = float(amap.max())
            amap_mean = float(amap.mean())
            amap_std = float(amap.std())
            amap_shape = amap.shape
            is_already_normalized = (amap_min >= 0.0) and (amap_max <= 1.0)
        else:
            amap_min, amap_max, amap_mean, amap_std = 0.0, 0.0, 0.0, 0.0
            amap_shape = (0, 0)
            is_already_normalized = False

        print(f"[{idx+1:02d}/15] File: {filename}")
        print(f"     - pred_score raw value : {score:.6f}")
        print(f"     - anomaly_map shape    : {amap_shape}")
        print(f"     - anomaly_map min      : {amap_min:.6f}")
        print(f"     - anomaly_map max      : {amap_max:.6f}")
        print(f"     - anomaly_map mean     : {amap_mean:.6f}")
        print(f"     - anomaly_map std      : {amap_std:.6f}")
        print(f"     - is_normalized [0,1]  : {is_already_normalized}")

        # Binary Prediction Mask at threshold 0.4981 (NO cv2.normalize used for decision)
        if amap is not None:
            binary_mask = (amap >= FROZEN_THRESHOLD).astype(np.uint8)
        else:
            binary_mask = np.zeros(TARGET_SIZE, dtype=np.uint8)

        has_defect = bool(binary_mask.max() > 0)
        status_str = "DEFECT DETECTED" if has_defect else "PASS (NORMAL)"

        # Prepare Anomaly Heatmap Overlay for VISUALIZATION ONLY
        if amap is not None:
            if is_already_normalized:
                amap_vis = (amap * 255.0).clip(0, 255).astype(np.uint8)
            else:
                amap_vis = cv2.normalize(amap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(amap_vis, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_rgb, 0.5, 0)
        else:
            heatmap_overlay = img_rgb.copy()

        # Bounding Box & Contour Overlay
        overlay_img = img_rgb.copy()
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        box_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2:
                box_count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.drawContours(overlay_img, [cnt], -1, (255, 255, 0), 1)

        # Status badge
        badge_color = (255, 0, 0) if has_defect else (0, 200, 0)
        cv2.rectangle(overlay_img, (5, 5), (210, 35), (0, 0, 0), -1)
        cv2.putText(overlay_img, status_str, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

        # Senior-Facing Annotated Output
        fig_senior, axes_s = plt.subplots(1, 3, figsize=(15, 5))
        axes_s[0].imshow(img_rgb)
        axes_s[0].set_title("1. DTD Input Image", fontsize=12, fontweight="bold")
        axes_s[0].axis("off")

        axes_s[1].imshow(heatmap_overlay)
        axes_s[1].set_title("2. Anomaly Heatmap Overlay", fontsize=12, fontweight="bold")
        axes_s[1].axis("off")

        axes_s[2].imshow(overlay_img)
        axes_s[2].set_title("3. Defect Detection & Bounding Boxes", fontsize=12, fontweight="bold")
        axes_s[2].axis("off")

        fig_senior.suptitle(
            f"DTD Zero-Adaptation Test — Image: {filename}\n"
            f"Status: {status_str} | Anomaly Score: {score:.4f} | Threshold: {FROZEN_THRESHOLD}",
            fontsize=13,
            fontweight="bold",
            color="red" if has_defect else "green"
        )
        plt.tight_layout()
        senior_path = SENIOR_OUTPUT_DIR / f"senior_annotated_{path.stem}.png"
        plt.savefig(senior_path, dpi=150, bbox_inches="tight")
        plt.close(fig_senior)

        # Diagnostic 4-Panel Output
        fig_diag, axes_d = plt.subplots(1, 4, figsize=(18, 4.5))
        axes_d[0].imshow(img_rgb)
        axes_d[0].set_title("1. Original Image", fontsize=11, fontweight="bold")
        axes_d[0].axis("off")

        if amap is not None:
            axes_d[1].imshow(amap, cmap="jet")
        axes_d[1].set_title("2. Raw Anomaly Map", fontsize=11, fontweight="bold")
        axes_d[1].axis("off")

        axes_d[2].imshow(binary_mask, cmap="gray")
        axes_d[2].set_title(f"3. Mask (Thr={FROZEN_THRESHOLD})", fontsize=11, fontweight="bold")
        axes_d[2].axis("off")

        axes_d[3].imshow(overlay_img)
        axes_d[3].set_title("4. Bounding Box Overlay", fontsize=11, fontweight="bold")
        axes_d[3].axis("off")

        fig_diag.suptitle(
            f"DTD Diagnostic View — Image: {filename} | Status: {status_str} | Score: {score:.4f}",
            fontsize=13,
            fontweight="bold"
        )
        plt.tight_layout()
        diag_path = DIAGNOSTIC_OUTPUT_DIR / f"diagnostic_4panel_{path.stem}.png"
        plt.savefig(diag_path, dpi=150, bbox_inches="tight")
        plt.close(fig_diag)

        summary_rows.append({
            "filename": filename,
            "anomaly_score": round(score, 4),
            "status": status_str,
            "bounding_boxes": box_count,
            "senior_image": str(senior_path.relative_to(BASE_DIR)),
            "diagnostic_image": str(diag_path.relative_to(BASE_DIR)),
        })

    # Save CSV Summary Report
    csv_path = OUTPUT_DIR / "dtd_inference_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "anomaly_score", "status", "bounding_boxes", "senior_image", "diagnostic_image"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("DTD STANDALONE INFERENCE RUN COMPLETE — SUMMARY")
    print("=" * 80)
    print(f"CSV Summary Report:            {csv_path}")
    print(f"Senior-facing images saved to: {SENIOR_OUTPUT_DIR}")
    print(f"Diagnostic images saved to:    {DIAGNOSTIC_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
