import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont

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
LOCKED_THRESHOLD = 21.3272


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        }


def generic_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    return ImageBatch(image=images, image_path=image_paths)


def create_visualizations():
    set_seed(SEED)

    base_dir = Path("c:/dev/Anomaly_Detector/backend").resolve()
    data_dir = base_dir / "data" / "textile"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    output_base = base_dir / "outputs" / "generic_patchcore" / "textile" / "false_positive_analysis"
    output_base.mkdir(parents=True, exist_ok=True)

    # 1. Discover all 280 training images
    all_train = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    print(f"Discovered {len(all_train)} GOOD training reference images.")

    # 2. Discover the 10 Known GOOD test images
    all_test = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    known_good_test = sorted([p for p in all_test if "copy 2" in p.name])
    print(f"Targeting {len(known_good_test)} Known GOOD test images for visualization.")

    # 3. Fit PatchCore on ALL 280 GOOD reference images
    print("\n[1/3] Training final PatchCore model on all 280 GOOD reference images...")
    model = Patchcore(backbone=BACKBONE, layers=LAYERS, pre_trained=True, coreset_sampling_ratio=CORESET_RATIO, num_neighbors=NUM_NEIGHBORS)
    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    train_loader = DataLoader(GenericFolderDataset(all_train), batch_size=4, shuffle=False, collate_fn=generic_collate_fn)
    engine.fit(model=model, train_dataloaders=train_loader)
    model.post_processor = None  # Preserve raw feature distances

    # 4. Predict on the 10 Known GOOD test images
    print("\n[2/3] Extracting raw anomaly maps for 10 Known GOOD test images...")
    test_loader = DataLoader(GenericFolderDataset(known_good_test), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    predictions = engine.predict(model=model, dataloaders=test_loader)

    gallery_items = []

    for pred in predictions:
        p_str = pred.image_path[0]
        p_path = Path(p_str)
        fname = p_path.name
        raw_score = float(pred.pred_score[0])
        status = "ANOMALOUS" if raw_score >= LOCKED_THRESHOLD else "NORMAL"
        decision_label = "FALSE POSITIVE" if status == "ANOMALOUS" else "TRUE NEGATIVE"
        folder_suffix = "FP" if status == "ANOMALOUS" else "TN"

        # Extract 000_copy2_TN format folder name
        idx_str = fname.split()[0]  # e.g., '000'
        folder_name = f"{idx_str}_copy2_{folder_suffix}"
        img_out_dir = output_base / folder_name
        img_out_dir.mkdir(parents=True, exist_ok=True)

        # Raw anomaly map tensor (shape 256x256)
        amap = pred.anomaly_map[0].squeeze().cpu().numpy()

        # Load RGB original image (256x256)
        orig_pil = Image.open(p_path).convert("RGB").resize(DEFAULT_TARGET_SIZE, Image.BILINEAR)
        orig_rgb = np.array(orig_pil)
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

        # 1. Save original.png
        cv2.imwrite(str(img_out_dir / "original.png"), orig_bgr)

        # 2. Generate and save heatmap.png
        # Normalize anomaly map relative to threshold and dataset max distance (~35.0) for consistent jet coloring
        amap_norm = np.clip((amap - 15.0) / (32.0 - 15.0), 0.0, 1.0)
        amap_u8 = (amap_norm * 255.0).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
        cv2.imwrite(str(img_out_dir / "heatmap.png"), heatmap_bgr)

        # 3. Generate and save overlay.png
        overlay_bgr = cv2.addWeighted(orig_bgr, 0.5, heatmap_bgr, 0.5, 0)
        cv2.imwrite(str(img_out_dir / "overlay.png"), overlay_bgr)

        # 4. Generate and save bbox.png
        bbox_bgr = overlay_bgr.copy()

        # Bounding box thresholding at LOCKED_THRESHOLD
        mask = (amap >= LOCKED_THRESHOLD).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for cnt in contours:
                if cv2.contourArea(cnt) >= 16:  # Filter noise patches < 16 pixels
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(bbox_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # Label peak raw distance in box
                    peak_val = amap[y:y+h, x:x+w].max()
                    cv2.putText(bbox_bgr, f"{peak_val:.1f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            # If TN (no region > threshold), highlight top peak patch box (32x32 area)
            max_y, max_x = np.unravel_index(np.argmax(amap), amap.shape)
            x1, y1 = max(0, max_x - 16), max(0, max_y - 16)
            x2, y2 = min(256, max_x + 16), min(256, max_y + 16)
            cv2.rectangle(bbox_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bbox_bgr, f"Peak: {amap.max():.1f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Add Header Info Bar
        header_h = 55
        padded_bbox = np.zeros((256 + header_h, 256, 3), dtype=np.uint8)
        padded_bbox[header_h:, :] = bbox_bgr

        # Header background
        status_color = (0, 0, 255) if status == "ANOMALOUS" else (0, 200, 0)
        cv2.rectangle(padded_bbox, (0, 0), (256, header_h), (20, 20, 20), -1)

        # Draw Header Text
        cv2.putText(padded_bbox, f"File: {fname}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(padded_bbox, f"GT: GOOD | Pred: {status}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        cv2.putText(padded_bbox, f"Dist: {raw_score:.2f} | Thresh: {LOCKED_THRESHOLD:.2f} ({decision_label})", (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.imwrite(str(img_out_dir / "bbox.png"), padded_bbox)

        gallery_items.append({
            "filename": fname,
            "folder": folder_name,
            "raw_score": raw_score,
            "status": status,
            "decision_label": decision_label,
            "bbox_img": padded_bbox,
            "overlay_img": overlay_bgr,
        })

        print(f"  Generated visualizations for {fname:<16} -> {folder_name} (Dist: {raw_score:.4f}, Status: {decision_label})")

    # -------------------------------------------------------------
    # 5. GENERATE CONTACT SHEET (2x5 Grid)
    # -------------------------------------------------------------
    print("\n[3/3] Generating Composite Contact Sheet Gallery (textile_good_false_positive_gallery.png)...")

    rows, cols = 2, 5
    cell_w, cell_h = 256, 256 + 55
    margin = 10
    grid_w = cols * cell_w + (cols + 1) * margin
    grid_h = rows * cell_h + (rows + 1) * margin + 40  # Extra space for top title banner

    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    # Top Title Banner
    cv2.putText(canvas, "TEXTILE GOOD TEST IMAGES: FAILURE ANALYSIS GALLERY (10 IMAGES)", (margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, f"Locked Threshold: {LOCKED_THRESHOLD:.4f} | wide_resnet50_2 + layer2 + 5% coreset", (grid_w - 480, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    for idx, item in enumerate(gallery_items):
        r = idx // cols
        c = idx % cols
        x = margin + c * (cell_w + margin)
        y = 40 + margin + r * (cell_h + margin)

        canvas[y:y+cell_h, x:x+cell_w] = item["bbox_img"]

    gallery_path = output_base / "textile_good_false_positive_gallery.png"
    cv2.imwrite(str(gallery_path), canvas)
    print(f"Saved Composite Gallery Contact Sheet: {gallery_path}")


if __name__ == "__main__":
    create_visualizations()
