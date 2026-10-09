import os
import sys
import json
import time
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
TARGET_THRESHOLD = 29.96  # Target reported threshold for PCB transistor dataset


def set_seed(seed=SEED):
    import random
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
            "orig_size": img_pil.size
        }


def generic_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    image_paths = [b["image_path"] for b in batch]
    return ImageBatch(image=images, image_path=image_paths)


def run_pcb_transistor_audit(dataset_dir: str, output_dir: str):
    set_seed(SEED)
    dataset_path = Path(dataset_dir).resolve()
    train_dir = dataset_path / "train" / "good"
    test_dir = dataset_path / "test"
    gt_dir = dataset_path / "ground_truth"

    out_base = Path(output_dir).resolve() if output_dir else dataset_path.parent.parent.parent / "outputs" / "pcb_transistor_technical_audit"
    out_base.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_base / "intermediate_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_images = sorted([p for p in train_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    test_images = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS])

    print(f"[PCB Audit] Discovered {len(train_images)} GOOD training reference images.")
    print(f"[PCB Audit] Discovered {len(test_images)} test images across subfolders.")

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
    model.post_processor = None  # Raw distance score mode

    # Calculate actual 95th percentile threshold on GOOD calibration split
    val_loader = DataLoader(GenericFolderDataset(train_images[:10]), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    val_preds = engine.predict(model=model, dataloaders=val_loader)
    val_scores = [float(p.pred_score[0]) for p in val_preds]
    calib_p95 = round(float(np.percentile(val_scores, 95)), 2)
    effective_threshold = TARGET_THRESHOLD if TARGET_THRESHOLD else calib_p95
    print(f"[PCB Audit] Calibration GOOD scores mean={np.mean(val_scores):.2f}, std={np.std(val_scores):.2f}, p95={calib_p95:.2f}. Effective Threshold={effective_threshold:.2f}")

    # 2. Score full test set
    test_loader = DataLoader(GenericFolderDataset(test_images), batch_size=1, shuffle=False, collate_fn=generic_collate_fn)
    predictions = engine.predict(model=model, dataloaders=test_loader)

    audit_records = []
    category_counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    category_records = {"TP": [], "TN": [], "FP": [], "FN": []}

    target_002_record = None

    for idx, pred in enumerate(predictions):
        p_path = Path(pred.image_path[0])
        fname = p_path.name
        subfolder = p_path.parent.name
        raw_score = float(pred.pred_score[0])
        gt_status = "NORMAL" if subfolder == "good" else "ANOMALOUS"
        pred_status = "ANOMALOUS" if raw_score >= effective_threshold else "NORMAL"

        if gt_status == "ANOMALOUS" and pred_status == "ANOMALOUS":
            cat = "TP"
        elif gt_status == "NORMAL" and pred_status == "NORMAL":
            cat = "TN"
        elif gt_status == "NORMAL" and pred_status == "ANOMALOUS":
            cat = "FP"
        else:
            cat = "FN"

        category_counts[cat] += 1

        amap = pred.anomaly_map[0].squeeze().cpu().numpy()  # (256, 256)
        amap_min, amap_max = float(amap.min()), float(amap.max())
        amap_mean, amap_std = float(amap.mean()), float(amap.std())
        amap_p50, amap_p90, amap_p95, amap_p99 = [
            float(np.percentile(amap, p)) for p in [50, 90, 95, 99]
        ]

        orig_pil = Image.open(p_path).convert("RGB")
        orig_w, orig_h = orig_pil.size
        orig_np = np.array(orig_pil)

        # Ground truth mask check if available
        mask_path = gt_dir / subfolder / f"{p_path.stem}_mask.png"
        gt_mask_present = mask_path.exists()
        defect_region_max_score = None
        defect_region_mean_score = None
        bg_region_mean_score = None

        if gt_mask_present:
            mask_pil = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
            mask_np = np.array(mask_pil) > 128
            if np.any(mask_np):
                defect_region_max_score = float(amap[mask_np].max())
                defect_region_mean_score = float(amap[mask_np].mean())
            if np.any(~mask_np):
                bg_region_mean_score = float(amap[~mask_np].mean())

        # Coordinates of maximum activation
        max_y_256, max_x_256 = np.unravel_index(np.argmax(amap), amap.shape)
        peak_x_orig = int(round((max_x_256 / 256.0) * orig_w))
        peak_y_orig = int(round((max_y_256 / 256.0) * orig_h))

        record = {
            "index": idx,
            "filename": fname,
            "subfolder": subfolder,
            "path": str(p_path),
            "ground_truth": gt_status,
            "prediction": pred_status,
            "category": cat,
            "raw_anomaly_score": round(raw_score, 4),
            "threshold": effective_threshold,
            "amap_min": round(amap_min, 4),
            "amap_max": round(amap_max, 4),
            "amap_mean": round(amap_mean, 4),
            "amap_std": round(amap_std, 4),
            "amap_p50": round(amap_p50, 4),
            "amap_p90": round(amap_p90, 4),
            "amap_p95": round(amap_p95, 4),
            "amap_p99": round(amap_p99, 4),
            "peak_coords_256": [int(max_x_256), int(max_y_256)],
            "peak_coords_orig": [peak_x_orig, peak_y_orig],
            "defect_region_max_score": round(defect_region_max_score, 4) if defect_region_max_score is not None else None,
            "defect_region_mean_score": round(defect_region_mean_score, 4) if defect_region_mean_score is not None else None,
            "bg_region_mean_score": round(bg_region_mean_score, 4) if bg_region_mean_score is not None else None,
            "orig_size": [orig_w, orig_h]
        }

        audit_records.append(record)
        category_records[cat].append(record)

        if subfolder in ("misplaced", "bent_lead", "cut_lead", "damaged_case") and fname == "002.png":
            target_002_record = record

            # Save detailed visual artifacts for target 002.png
            sub_art_dir = artifacts_dir / f"{subfolder}_{fname.replace('.', '_')}"
            sub_art_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(sub_art_dir / "01_original.png"), cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))

            amap_norm_abs = np.clip((amap - 15.0) / (35.0 - 15.0), 0.0, 1.0)
            amap_u8 = cv2.resize((amap_norm_abs * 255.0).astype(np.uint8), (orig_w, orig_h))
            heat_bgr = cv2.applyColorMap(amap_u8, cv2.COLORMAP_JET)
            cv2.imwrite(str(sub_art_dir / "02_absolute_heatmap.png"), heat_bgr)

            overlay_bgr = cv2.addWeighted(cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR), 0.6, heat_bgr, 0.4, 0)
            cv2.imwrite(str(sub_art_dir / "03_overlay.png"), overlay_bgr)

            if gt_mask_present:
                gt_mask_orig = cv2.resize(np.array(Image.open(mask_path).convert("L")), (orig_w, orig_h))
                mask_visual = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR).copy()
                mask_visual[gt_mask_orig > 128] = (0, 0, 255)
                cv2.imwrite(str(sub_art_dir / "04_ground_truth_mask_overlay.png"), mask_visual)

    summary_payload = {
        "dataset": "mvtec_transistor",
        "effective_threshold": effective_threshold,
        "total_test_images": len(audit_records),
        "confusion_matrix": category_counts,
        "target_002_record": target_002_record,
        "category_summary": {
            cat: {
                "count": len(recs),
                "score_min": round(min(r["raw_anomaly_score"] for r in recs), 2) if recs else None,
                "score_max": round(max(r["raw_anomaly_score"] for r in recs), 2) if recs else None,
                "score_mean": round(float(np.mean([r["raw_anomaly_score"] for r in recs])), 2) if recs else None,
            }
            for cat, recs in category_records.items()
        },
        "records": audit_records
    }

    summary_path = out_base / "pcb_transistor_audit_records.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[PCB Audit] Completed evaluation. JSON saved to {summary_path}")
    print(f"[PCB Audit] Confusion Matrix: TP={category_counts['TP']}, TN={category_counts['TN']}, FP={category_counts['FP']}, FN={category_counts['FN']}")
    if target_002_record:
        print(f"[PCB Audit] Target 002.png record: Subfolder={target_002_record['subfolder']}, Score={target_002_record['raw_anomaly_score']}, Threshold={effective_threshold}, Defect Region Max Score={target_002_record['defect_region_max_score']}")


if __name__ == "__main__":
    dataset_dir = "c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/transistor"
    out_dir = "c:/dev/Anomaly_Detector/outputs/pcb_transistor_technical_audit"
    run_pcb_transistor_audit(dataset_dir, out_dir)
