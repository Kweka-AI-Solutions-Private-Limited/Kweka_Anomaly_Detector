import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from services.patchcore_service import (
    Patchcore, Engine, generic_collate_fn, preprocess_image_aspect_preserving, PATCHCORE_CONFIG
)
from torch.utils.data import DataLoader

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

p = Path("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/good/003.png")
img_pil = Image.open(p).convert("RGB")

def exp_func(img_pil: Image.Image, target_size=(256, 256)) -> torch.Tensor:
    w, h = img_pil.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    padded_arr = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    padded_arr = padded_arr / 255.0
    return torch.from_numpy(padded_arr).permute(2, 0, 1)

t_exp = exp_func(img_pil)
t_prod, meta = preprocess_image_aspect_preserving(img_pil)

print("t_exp shape:", t_exp.shape, "min:", t_exp.min().item(), "max:", t_exp.max().item())
print("t_prod shape:", t_prod.shape, "min:", t_prod.min().item(), "max:", t_prod.max().item())
print("tensors equal:", torch.equal(t_exp, t_prod))

class SingleDL(torch.utils.data.Dataset):
    def __init__(self, t): self.t = t
    def __len__(self): return 1
    def __getitem__(self, i): return {"image": self.t, "image_path": str(p)}

s_exp = float(engine.predict(model=m, dataloaders=DataLoader(SingleDL(t_exp), collate_fn=generic_collate_fn))[0].pred_score[0])
s_prod = float(engine.predict(model=m, dataloaders=DataLoader(SingleDL(t_prod), collate_fn=generic_collate_fn))[0].pred_score[0])

print(f"Score exp : {s_exp:.2f}")
print(f"Score prod: {s_prod:.2f}")
