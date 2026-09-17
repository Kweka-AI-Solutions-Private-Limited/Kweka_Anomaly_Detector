import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Patchcore

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_TARGET_SIZE = (256, 256)

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2"]
CORESET_RATIO = 0.05
NUM_NEIGHBORS = 9
SEED = 42
LOCKED_THRESHOLD = 21.3272  # Operational baseline threshold on textile dataset


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_ground_truth(file_path: Path) -> str:
    path_str = str(file_path).lower()
    name_str = file_path.name.lower()
    is_good = (
        "good" in path_str
        or "normal" in path_str
        or "ok" in path_str
        or "copy 2" in name_str
        or "copy 3" in name_str
        or "copy 4" in name_str
        or "copy 5" in name_str
        or "copy.png" in name_str
    )
    return "NORMAL" if is_good else "ANOMALOUS"


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
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        return {
            "image": tensor,
            "image_path": str(path),
            "orig_size": img_pil.size  # (W, H)
        }


def generic_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    return ImageBatch(image=images, image_path=image_paths)


def run_technical_audit(dataset_dir: str, output_dir: str):
    set_seed(SEED)
    dataset_path = Path(dataset_dir).resolve()
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"

    out_base = Path(output_dir).resolve() if output_dir else dataset_path.parent.parent / "outputs" / "patchcore_technical_audit"
    out_base.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_base / "intermediate_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_images = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    test_images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])

    print(f"[Audit] Discovered {len(train_images)} training reference images and {len(test_images)} test images.")

    # 1. Fit PatchCore model
    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_RATIO,
        num_neighbors=NUM_NEIGHBORS
    )
    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    train_loader = DataLoader(GenericFolderDataset(train_images), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    engine.fit(model=model, train_dataloaders=train_loader)
    model.post_processor = None

    # 2. Predict on test dataset
    test_loader = DataLoader(GenericFolderDataset(test_images), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    predictions = engine.predict(model=model, dataloaders=test_loader)

    audit_records = []

    for idx, pred in enumerate(predictions):
        p_path = Path(pred.image_path[0])
        fname = p_path.name
        raw_score = float(pred.pred_score[0])
        gt_status = infer_ground_truth(p_path)
        pred_status = "ANOMALOUS" if raw_score >= LOCKED_THRESHOLD else "NORMAL"

        amap = pred.anomaly_map[0].squeeze().cpu().numpy()  # raw anomaly map tensor (H, W)
        amap_min, amap_max = float(amap.min()), float(amap.max())
        amap_mean, amap_std = float(amap.mean()), float(amap.std())

        orig_pil = Image.open(p_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        orig_np = np.array(orig_pil)

        # Resized 256x256 image used for model input
        resized_pil = orig_pil.resize((256, 256), Image.BILINEAR)
        resized_np = np.array(resized_pil)

        # -------------------------------------------------------------
        # Pipeline Analysis Step A: Current UI Heatmap Normalization (Per-Image Min-Max)
        # -------------------------------------------------------------
        ui_norm_map = (amap - amap_min) / (amap_max - amap_min + 1e-8)
        ui_norm_u8 = (ui_norm_map * 255.0).astype(np.uint8)
        ui_norm_resized = cv2.resize(ui_norm_u8, (orig_w, orig_h))
        ui_heatmap_bgr = cv2.applyColorMap(ui_norm_resized, cv2.COLORMAP_JET)
        ui_overlay_bgr = cv2.addWeighted(cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR), 0.6, ui_heatmap_bgr, 0.4, 0)

        # Current UI BBox logic (norm_map > 128 & largest contour)
        ui_thresh_mask = (ui_norm_resized > 128).astype(np.uint8)
        ui_contours, _ = cv2.findContours(ui_thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ui_bbox = None
        if ui_contours:
            largest_cnt = max(ui_contours, key=cv2.contourArea)
            ux, uy, uw, uh = cv2.boundingRect(largest_cnt)
            ui_bbox = {"x": int(ux), "y": int(uy), "width": int(uw), "height": int(uh)}

        # -------------------------------------------------------------
        # Pipeline Analysis Step B: Absolute Threshold Bounding Box (amap >= threshold)
        # -------------------------------------------------------------
        abs_thresh_mask_256 = (amap >= LOCKED_THRESHOLD).astype(np.uint8)
        abs_thresh_mask_orig = cv2.resize(abs_thresh_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        abs_contours, _ = cv2.findContours(abs_thresh_mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        abs_bboxes = []
        if abs_contours:
            for cnt in abs_contours:
                if cv2.contourArea(cnt) >= 16:
                    ax, ay, aw, ah = cv2.boundingRect(cnt)
                    abs_bboxes.append({"x": int(ax), "y": int(ay), "width": int(aw), "height": int(ah)})

        # Peak activation coordinates in 256x256 model space & mapped to original image
        max_y_256, max_x_256 = np.unravel_index(np.argmax(amap), amap.shape)
        peak_x_orig = int(round((max_x_256 / 256.0) * orig_w))
        peak_y_orig = int(round((max_y_256 / 256.0) * orig_h))

        # Save intermediate artifacts for detailed failure mode inspect
        img_sub_dir = artifacts_dir / f"{fname.replace(' ', '_')}"
        img_sub_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(img_sub_dir / "01_original.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_sub_dir / "02_resized_256x256.png"), cv2.cvtColor(resized_np, cv2.COLOR_RGB2BGR))

        # Save raw anomaly map normalized to absolute scale [10.0, 35.0]
        abs_amap_norm = np.clip((amap - 10.0) / (35.0 - 10.0), 0.0, 1.0)
        abs_amap_u8 = cv2.resize((abs_amap_norm * 255.0).astype(np.uint8), (orig_w, orig_h))
        abs_heatmap_bgr = cv2.applyColorMap(abs_amap_u8, cv2.COLORMAP_JET)
        cv2.imwrite(str(img_sub_dir / "03_raw_anomaly_heatmap_absolute_scale.png"), abs_heatmap_bgr)
        cv2.imwrite(str(img_sub_dir / "04_ui_per_image_minmax_heatmap.png"), ui_heatmap_bgr)
        cv2.imwrite(str(img_sub_dir / "05_ui_overlay.png"), ui_overlay_bgr)
        cv2.imwrite(str(img_sub_dir / "06_ui_thresh_mask_gt128.png"), ui_thresh_mask * 255)
        cv2.imwrite(str(img_sub_dir / "07_absolute_thresh_mask.png"), abs_thresh_mask_orig * 255)

        # Draw UI BBox vs Absolute BBoxes on diagnostic visual
        diag_img = ui_overlay_bgr.copy()
        if ui_bbox:
            cv2.rectangle(
                diag_img,
                (ui_bbox["x"], ui_bbox["y"]),
                (ui_bbox["x"] + ui_bbox["width"], ui_bbox["y"] + ui_bbox["height"]),
                (0, 0, 255), 2  # Red for UI largest contour box
            )
            cv2.putText(diag_img, "UI Largest Contour", (ui_bbox["x"], max(15, ui_bbox["y"] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.circle(diag_img, (peak_x_orig, peak_y_orig), 6, (0, 255, 255), -1)
        cv2.putText(diag_img, f"Peak ({amap_max:.1f})", (peak_x_orig + 8, peak_y_orig), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imwrite(str(img_sub_dir / "08_diagnostic_localization.png"), diag_img)

        record = {
            "index": idx,
            "filename": fname,
            "path": str(p_path),
            "ground_truth": gt_status,
            "prediction": pred_status,
            "raw_anomaly_score": round(raw_score, 4),
            "threshold": LOCKED_THRESHOLD,
            "amap_min": round(amap_min, 4),
            "amap_max": round(amap_max, 4),
            "amap_mean": round(amap_mean, 4),
            "amap_std": round(amap_std, 4),
            "peak_coords_256": [int(max_x_256), int(max_y_256)],
            "peak_coords_orig": [peak_x_orig, peak_y_orig],
            "ui_bbox": ui_bbox,
            "abs_bboxes": abs_bboxes,
            "orig_size": [orig_w, orig_h],
            "artifact_dir": str(img_sub_dir)
        }
        audit_records.append(record)

    audit_summary_path = out_base / "technical_audit_records.json"
    with open(audit_summary_path, "w") as f:
        json.dump(audit_records, f, indent=2)

    print(f"[Audit] Technical audit complete. Evaluated {len(audit_records)} images.")
    print(f"[Audit] Summary saved to: {audit_summary_path}")

    # Print out summary table of key test files
    print("\n" + "=" * 95)
    print(f"{'Filename':<25} | {'GT':<10} | {'Pred':<10} | {'Score':<8} | {'Min':<6} | {'Max':<6} | {'Peak (x,y)':<12}")
    print("=" * 95)
    for r in audit_records:
        print(f"{r['filename']:<25} | {r['ground_truth']:<10} | {r['prediction']:<10} | {r['raw_anomaly_score']:<8.2f} | {r['amap_min']:<6.2f} | {r['amap_max']:<6.2f} | {str(r['peak_coords_orig']):<12}")
    print("=" * 95)


if __name__ == "__main__":
    dataset_dir = "c:/dev/Anomaly_Detector/backend/data/textile"
    out_dir = "c:/dev/Anomaly_Detector/outputs/patchcore_technical_audit"
    run_technical_audit(dataset_dir, out_dir)
