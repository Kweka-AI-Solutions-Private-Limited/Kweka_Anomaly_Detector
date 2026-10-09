import sys
import io
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.patchcore_service import (
    Patchcore, Engine, generic_collate_fn, PATCHCORE_CONFIG, DEFAULT_TARGET_SIZE
)

prod_ckpt = Path("c:/dev/Anomaly_Detector/backend/storage/artifacts/6aa3aa1d4067031f17c51123/v2/patchcore_memory_bank.ckpt")

m = Patchcore(
    backbone=PATCHCORE_CONFIG["backbone"],
    layers=PATCHCORE_CONFIG["layers"],
    pre_trained=PATCHCORE_CONFIG["pretrained"],
    coreset_sampling_ratio=PATCHCORE_CONFIG["coreset_sampling_ratio"],
    num_neighbors=PATCHCORE_CONFIG["num_neighbors"],
)
st = torch.load(prod_ckpt, map_location="cpu")
m.load_state_dict(st)
m.post_processor = None
engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

def preprocess_aspect_preserving(img_pil: Image.Image, target_size=(256, 256)) -> torch.Tensor:
    w, h = img_pil.size
    target_w, target_h = target_size
    if w == h:
        img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # Pad using BORDER_REFLECT_101
    padded_arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    padded_arr = padded_arr / 255.0
    return torch.from_numpy(padded_arr).permute(2, 0, 1)

def preprocess_direct(img_pil: Image.Image, target_size=(256, 256)) -> torch.Tensor:
    img_resized = img_pil.resize(target_size, Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)

class ExperimentDataset(Dataset):
    def __init__(self, paths, mode="aspect"):
        self.paths = paths
        self.mode = mode
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img_pil = Image.open(p).convert("RGB")
        if self.mode == "aspect":
            t = preprocess_aspect_preserving(img_pil)
        else:
            t = preprocess_direct(img_pil)
        return {"image": t, "image_path": str(p)}

test_cases = [
    ("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/good/003.png", "768x256 Non-Square Good (003.png)"),
    ("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png", "1024x1024 Square Good (000.png)"),
    ("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head/000.png", "1024x1024 Square Defective (000.png)"),
]

print("="*85)
print("ASPECT-RATIO PRESERVATION PREPROCESSING EXPERIMENT")
print("Target Threshold = 24.54")
print("="*85)
print(f"{'DESCRIPTION':<40} | {'DIRECT SCORE':<12} | {'ASPECT SCORE':<12} | {'STATUS (ASPECT)':<15}")
print("-" * 85)

for path_str, desc in test_cases:
    p = Path(path_str)
    if not p.exists():
        continue
    # Direct
    dl_dir = DataLoader(ExperimentDataset([p], mode="direct"), batch_size=1, collate_fn=generic_collate_fn)
    p_dir = float(engine.predict(model=m, dataloaders=dl_dir)[0].pred_score[0])
    
    # Aspect
    dl_asp = DataLoader(ExperimentDataset([p], mode="aspect"), batch_size=1, collate_fn=generic_collate_fn)
    p_asp = float(engine.predict(model=m, dataloaders=dl_asp)[0].pred_score[0])

    status = "PASS (< 24.54)" if p_asp <= 24.54 else "REJECT (>= 24.54)"
    print(f"{desc:<40} | {p_dir:<12.2f} | {p_asp:<12.2f} | {status:<15}")

print("="*85)
