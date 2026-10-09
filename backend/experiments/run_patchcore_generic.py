"""
run_patchcore_generic.py

Generic Folder-Based PatchCore Anomaly Detector Runner
------------------------------------------------------
Dataset-agnostic PatchCore runner that trains on any folder containing train/
(all images treated as GOOD references) and evaluates test/ images (unlabeled inspection).

Features:
  - Out-of-sample GOOD calibration (LOO for N < 10, 80/20 split for N >= 10)
  - Percentile-based calibrated threshold (95th percentile of out-of-sample GOOD scores)
  - Unclipped raw L2 anomaly distance + Calibrated score scaling (S = raw / threshold)
  - Diagnostic Z-score logging (non-probabilistic)

CLI Usage:
    python src/run_patchcore_generic.py --dataset_dir "path/to/dataset" [--threshold 0.5] [--output_dir "custom/path"] [--seed 42]
"""

import os
import sys
import csv
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
import matplotlib.pyplot as plt

from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ============================================================
# CONSTANTS & CONFIGURATION
# ============================================================
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (256, 256)

# Finalized Model Parameters (Frozen)
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2"]
CORESET_RATIO = 0.05
NUM_NEIGHBORS = 9


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET DISCOVERY HELPERS
# ============================================================
def discover_images(directory: Path):
    """
    Recursively discovers all supported images under a directory.
    Returns a sorted list of absolute Path objects.
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    found_paths = []
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found_paths.append(p)
    return sorted(found_paths)


# ============================================================
# CUSTOM PYTORCH DATASET (UN-NORMALIZED FLOATS IN [0,1])
# ============================================================
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
        
        # Convert to float tensor in range [0.0, 1.0].
        # Single ImageNet normalization is handled internally by Anomalib's PreProcessor.
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


def to_numpy(val):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


# ============================================================
# OUT-OF-SAMPLE GOOD CALIBRATION ROUTINE
# ============================================================
def calibrate_out_of_sample_good(train_images, target_size=DEFAULT_TARGET_SIZE, seed=42):
    """
    Calibrates an out-of-sample GOOD threshold from reference training images.
      - If N < 10: Uses Leave-One-Out (LOO) cross-validation on GOOD data.
      - If N >= 10: Uses an 80/20 train/calibration split on GOOD data.
    Returns: (calibrated_threshold, calibration_mean, calibration_std, calibration_scores, calib_mode)
    """
    n_train = len(train_images)
    calib_scores = []

    if n_train < 10:
        calib_mode = f"Leave-One-Out (LOO, N={n_train})"
        print(f"Executing {calib_mode} calibration on GOOD reference data...")
        
        for i in range(n_train):
            sub_train = [train_images[j] for j in range(n_train) if j != i]
            val_sample = [train_images[i]]

            m = Patchcore(
                backbone=BACKBONE,
                layers=LAYERS,
                pre_trained=True,
                coreset_sampling_ratio=CORESET_RATIO,
                num_neighbors=NUM_NEIGHBORS,
            )
            e = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

            tr_loader = DataLoader(GenericFolderDataset(sub_train, target_size), batch_size=2, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)
            val_loader = DataLoader(GenericFolderDataset(val_sample, target_size), batch_size=1, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)

            e.fit(model=m, train_dataloaders=tr_loader)
            m.post_processor = None  # Disable MinMax clipping to inspect raw L2 distances

            preds = e.predict(model=m, dataloaders=val_loader)
            if preds and hasattr(preds[0], "pred_score"):
                score = float(preds[0].pred_score[0])
                calib_scores.append(score)
    else:
        calib_mode = f"Split-Sample (80/20, N={n_train})"
        print(f"Executing {calib_mode} calibration on GOOD reference data...")
        
        rng = random.Random(seed)
        shuffled = list(train_images)
        rng.shuffle(shuffled)

        split_idx = max(1, int(n_train * 0.8))
        ref_images = shuffled[:split_idx]
        val_images = shuffled[split_idx:]

        m = Patchcore(
            backbone=BACKBONE,
            layers=LAYERS,
            pre_trained=True,
            coreset_sampling_ratio=CORESET_RATIO,
            num_neighbors=NUM_NEIGHBORS,
        )
        e = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

        tr_loader = DataLoader(GenericFolderDataset(ref_images, target_size), batch_size=2, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)
        val_loader = DataLoader(GenericFolderDataset(val_images, target_size), batch_size=1, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)

        e.fit(model=m, train_dataloaders=tr_loader)
        m.post_processor = None  # Disable MinMax clipping

        preds = e.predict(model=m, dataloaders=val_loader)
        for b in preds:
            if hasattr(b, "pred_score"):
                for s in b.pred_score:
                    calib_scores.append(float(s))

    if not calib_scores:
        calib_scores = [10.0]

    mu_calib = float(np.mean(calib_scores))
    std_calib = float(np.std(calib_scores)) if len(calib_scores) > 1 else 1.0

    # 95th percentile out-of-sample GOOD threshold
    p95_thr = float(np.percentile(calib_scores, 95))

    return p95_thr, mu_calib, std_calib, calib_scores, calib_mode


# ============================================================
# MAIN RUNNER PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generic Folder-Based PatchCore Anomaly Detector")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root directory (containing train/ and test/)")
    parser.add_argument("--threshold", type=float, default=None, help="Optional explicit anomaly threshold")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional custom output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_root = Path(args.dataset_dir).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    dataset_name = dataset_root.name
    base_proj_dir = Path(__file__).resolve().parents[1]

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = base_proj_dir / "outputs" / "generic_patchcore" / dataset_name

    ckpt_dir = output_dir / "checkpoints"
    vis_dir = output_dir / "visualizations"
    senior_vis_dir = vis_dir / "senior_facing"
    diag_vis_dir = vis_dir / "diagnostic_4panel"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    senior_vis_dir.mkdir(parents=True, exist_ok=True)
    diag_vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERIC PATCHCORE ANOMALY DETECTOR")
    print("=" * 80)
    print(f"Dataset Name:        {dataset_name}")
    print(f"Dataset Directory:   {dataset_root}")
    print(f"Output Directory:    {output_dir}")
    print("-" * 80)

    # 1. Discover Images under train/ and test/
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    train_images = discover_images(train_dir)
    test_images = discover_images(test_dir)

    print(f"Training images discovered: {len(train_images)} (under {train_dir})")
    print(f"Test images discovered:     {len(test_images)} (under {test_dir})")

    if len(train_images) == 0:
        raise ValueError(
            f"Error: Training directory '{train_dir}' is missing or contains 0 supported images "
            f"(.png, .jpg, .jpeg, .bmp, .webp)."
        )

    if len(test_images) == 0:
        raise ValueError(
            f"Error: Test directory '{test_dir}' is missing or contains 0 supported images."
        )

    # 2. Perform Out-of-Sample Calibration on GOOD Reference Training Data
    print("\n[1/3] Calibrating out-of-sample GOOD threshold from reference training images...")
    calib_start = time.perf_counter()
    auto_p95_thr, mu_calib, std_calib, calib_scores, calib_mode = calibrate_out_of_sample_good(
        train_images, target_size=DEFAULT_TARGET_SIZE, seed=args.seed
    )
    calib_time = time.perf_counter() - calib_start

    if args.threshold is not None:
        effective_threshold = args.threshold
        threshold_source = f"CLI Argument (--threshold {args.threshold})"
    else:
        effective_threshold = auto_p95_thr
        threshold_source = f"Auto Calibrated 95th Percentile ({calib_mode})"

    print(f"Calibration Complete ({calib_time:.2f}s) | Mode: {calib_mode}")
    print(f"  Out-of-sample GOOD Mean Raw Distance: {mu_calib:.4f}")
    print(f"  Out-of-sample GOOD Std Dev:           {std_calib:.4f}")
    print(f"  Out-of-sample GOOD 95th Percentile:   {auto_p95_thr:.4f}")
    print(f"  Effective Decision Threshold:         {effective_threshold:.4f} ({threshold_source})")

    # 3. Build Final Model on ALL Training Reference Images
    print("\n[2/3] Building final PatchCore memory bank from ALL training reference images...")
    build_start = time.perf_counter()
    train_dataset = GenericFolderDataset(train_images, target_size=DEFAULT_TARGET_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)

    final_model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=NUM_NEIGHBORS,
    )

    final_engine = Engine(
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
    )

    final_engine.fit(model=final_model, train_dataloaders=train_loader)
    final_model.post_processor = None  # Disable MinMax clipping to retain unclipped raw L2 distances
    build_time_sec = time.perf_counter() - build_start

    # Extract Coreset Vector Count
    coreset_count = "N/A"
    try:
        if hasattr(final_model.model, "memory_bank"):
            mb = getattr(final_model.model, "memory_bank")
            if mb is not None:
                coreset_count = mb.shape[0] if hasattr(mb, "shape") else len(mb)
    except Exception:
        pass

    print(f"Build complete in {build_time_sec:.2f}s. Coreset vector count: {coreset_count}")

    # Save Model Checkpoint & Metadata
    ckpt_path = ckpt_dir / "patchcore_model.ckpt"
    torch.save({"state_dict": final_model.state_dict(), "hyper_parameters": final_model.hparams}, ckpt_path)

    metadata = {
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_root),
        "training_images_discovered": len(train_images),
        "test_images_discovered": len(test_images),
        "backbone": BACKBONE,
        "feature_layers": LAYERS,
        "coreset_sampling_ratio": CORESET_RATIO,
        "num_neighbors": NUM_NEIGHBORS,
        "input_resolution": list(DEFAULT_TARGET_SIZE),
        "calibration_mode": calib_mode,
        "calibration_good_mean_raw_distance": round(mu_calib, 4),
        "calibration_good_std_raw_distance": round(std_calib, 4),
        "calibration_good_95th_percentile": round(auto_p95_thr, 4),
        "effective_decision_threshold": round(effective_threshold, 4),
        "threshold_source": threshold_source,
        "build_time_seconds": round(build_time_sec, 4),
        "coreset_vector_count": coreset_count,
        "checkpoint_path": str(ckpt_path),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 4. Test Inference & Visualizations
    print("\n[3/3] Running PatchCore test inference...")
    test_dataset = GenericFolderDataset(test_images, target_size=DEFAULT_TARGET_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=generic_collate_fn)

    infer_start = time.perf_counter()
    prediction_batches = final_engine.predict(
        model=final_model,
        dataloaders=test_loader,
    )
    total_infer_time = time.perf_counter() - infer_start

    if not prediction_batches:
        raise RuntimeError("final_engine.predict() returned no prediction batches.")

    print(f"Inference complete in {total_infer_time:.2f}s. Processing visualizations...")

    summary_rows = []
    normal_count = 0
    anomalous_count = 0

    for idx, batch in enumerate(prediction_batches):
        paths = getattr(batch, "image_path", None)
        if paths is None:
            continue

        anomaly_maps = to_numpy(getattr(batch, "anomaly_map", None))
        pred_scores = to_numpy(getattr(batch, "pred_score", None))

        path_val = paths[0] if isinstance(paths, (list, tuple)) else paths
        abs_path = Path(str(path_val))
        
        # Preserve relative path under test/
        try:
            rel_path = str(abs_path.relative_to(test_dir))
        except ValueError:
            rel_path = abs_path.name

        rel_slug = rel_path.replace("\\", "_").replace("/", "_")

        amap = anomaly_maps[0] if anomaly_maps is not None else None
        if amap is not None and amap.ndim == 3 and amap.shape[0] == 1:
            amap = amap[0]

        # Load RGB image for visual overlay
        orig_pil = Image.open(abs_path).convert("RGB").resize(DEFAULT_TARGET_SIZE, Image.BILINEAR)
        img_rgb = np.array(orig_pil, dtype=np.uint8)

        raw_score = float(pred_scores[0].reshape(-1)[0]) if pred_scores is not None else (float(amap.max()) if amap is not None else 0.0)

        # Calibrated Score = raw L2 distance / effective threshold
        calibrated_score = raw_score / effective_threshold if effective_threshold > 0 else raw_score
        diagnostic_z_score = (raw_score - mu_calib) / std_calib if std_calib > 0 else 0.0

        # Classification Status
        has_defect = bool(raw_score >= effective_threshold)
        if has_defect:
            status_str = "ANOMALOUS"
            anomalous_count += 1
        else:
            status_str = "NORMAL"
            normal_count += 1

        # Binary Mask at effective decision threshold
        if amap is not None:
            binary_mask = (amap >= effective_threshold).astype(np.uint8)
        else:
            binary_mask = np.zeros(DEFAULT_TARGET_SIZE, dtype=np.uint8)

        # Anomaly Heatmap Overlay for Visualization
        if amap is not None:
            amap_norm = cv2.normalize(amap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
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

        # Draw Status Badge
        badge_color = (255, 0, 0) if status_str == "ANOMALOUS" else (0, 200, 0)
        cv2.rectangle(overlay_img, (5, 5), (200, 35), (0, 0, 0), -1)
        cv2.putText(overlay_img, status_str, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, badge_color, 2)

        # Senior-Facing Annotated Visualization
        fig_senior, axes_s = plt.subplots(1, 3, figsize=(15, 5))
        axes_s[0].imshow(img_rgb)
        axes_s[0].set_title(f"1. Input: {rel_path}", fontsize=11, fontweight="bold")
        axes_s[0].axis("off")

        axes_s[1].imshow(heatmap_overlay)
        axes_s[1].set_title("2. Anomaly Heatmap Overlay", fontsize=11, fontweight="bold")
        axes_s[1].axis("off")

        axes_s[2].imshow(overlay_img)
        axes_s[2].set_title("3. Bounding Boxes & Mask Overlay", fontsize=11, fontweight="bold")
        axes_s[2].axis("off")

        fig_senior.suptitle(
            f"Dataset: {dataset_name} | Image: {rel_path}\n"
            f"Status: {status_str} | Calibrated Score: {calibrated_score:.4f} | Raw Distance: {raw_score:.2f} | Thr: {effective_threshold:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        senior_path = senior_vis_dir / f"senior_annotated_{rel_slug}.png"
        plt.savefig(senior_path, dpi=150, bbox_inches="tight")
        plt.close(fig_senior)

        # Diagnostic 4-Panel Visualization
        fig_diag, axes_d = plt.subplots(1, 4, figsize=(18, 4.5))
        axes_d[0].imshow(img_rgb)
        axes_d[0].set_title("1. Original Image", fontsize=10, fontweight="bold")
        axes_d[0].axis("off")

        if amap is not None:
            axes_d[1].imshow(amap, cmap="jet")
        axes_d[1].set_title("2. Raw Anomaly Map", fontsize=10, fontweight="bold")
        axes_d[1].axis("off")

        axes_d[2].imshow(binary_mask, cmap="gray")
        axes_d[2].set_title(f"3. Binary Mask (Thr={effective_threshold:.2f})", fontsize=10, fontweight="bold")
        axes_d[2].axis("off")

        axes_d[3].imshow(overlay_img)
        axes_d[3].set_title("4. Bounding Box Overlay", fontsize=10, fontweight="bold")
        axes_d[3].axis("off")

        fig_diag.suptitle(
            f"Diagnostic 4-Panel View — Image: {rel_path} | Status: {status_str} | Calib Score: {calibrated_score:.4f} | Diagnostic Z-score: {diagnostic_z_score:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        diag_path = diag_vis_dir / f"diagnostic_4panel_{rel_slug}.png"
        plt.savefig(diag_path, dpi=150, bbox_inches="tight")
        plt.close(fig_diag)

        summary_rows.append({
            "relative_path": rel_path,
            "raw_anomaly_distance": round(raw_score, 4),
            "calibrated_score": round(calibrated_score, 4),
            "diagnostic_z_score": round(diagnostic_z_score, 2),
            "status": status_str,
            "bounding_boxes": box_count,
            "senior_image": str(senior_path.relative_to(base_proj_dir)),
            "diagnostic_image": str(diag_path.relative_to(base_proj_dir)),
        })

        print(f"[{idx+1:02d}/{len(test_images)}] {rel_path:<25} -> Raw Dist: {raw_score:<7.2f} | Calib Score: {calibrated_score:<6.4f} | Status: {status_str}")

    # Save Summary CSV Report
    csv_path = output_dir / "inference_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "raw_anomaly_distance", "calibrated_score", "diagnostic_z_score", "status", "bounding_boxes", "senior_image", "diagnostic_image"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Final Terminal Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"NORMAL:       {normal_count}")
    print(f"ANOMALOUS:    {anomalous_count}")
    print("=" * 80)
    print(f"Calibration Mode:      {calib_mode}")
    print(f"Effective Threshold:   {effective_threshold:.4f}")
    print(f"Checkpoint saved:      {ckpt_path}")
    print(f"Metadata saved:        {metadata_path}")
    print(f"Summary CSV saved:     {csv_path}")
    print(f"Senior visual outputs: {senior_vis_dir}")
    print(f"Diagnostic views:      {diag_vis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
