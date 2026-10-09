import os
import sys
import time
import inspect

os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import torchvision.transforms.v2 as T
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.models.components.sampling import KCenterGreedy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[0] device = {device}", flush=True)

print("[1] Loading data...", flush=True)
t0 = time.time()
dm = MVTecAD(
    root=str(DATA_ROOT),
    category="metal_nut",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
)
dm.setup()
train_loader = dm.train_dataloader()
print(f"    done in {time.time()-t0:.2f}s, {len(dm.train_data)} images", flush=True)

print("[2] Building PatchCore torch model...", flush=True)
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)
torch_model = model.model.to(device)
torch_model.train()
print("    forward signature:", inspect.signature(torch_model.forward), flush=True)

# Matches the model's own PreProcessor exactly (Resize 256x256 + ImageNet normalize)
preprocess = T.Compose([
    T.Resize([256, 256], antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("[3] Extracting patch embeddings batch-by-batch (plain prints, no Rich)...", flush=True)
t0 = time.time()
embeddings = []
with torch.no_grad():
    for i, batch in enumerate(train_loader):
        images = batch.image.to(device)
        images = preprocess(images)          # <-- the missing step
        out = torch_model(images)
        emb = out.embeddings if hasattr(out, "embeddings") else out
        embeddings.append(emb.detach().cpu())
        print(
            f"    batch {i}: {images.shape[0]} imgs {tuple(images.shape[2:])} -> {tuple(emb.shape)}  "
            f"[t={time.time()-t0:.1f}s]",
            flush=True,
        )
embeddings = torch.cat(embeddings)
print(
    f"[3] done in {time.time()-t0:.2f}s. Total embedding shape: {tuple(embeddings.shape)}",
    flush=True,
)

print("[4] Running k-center-greedy coreset selection...", flush=True)
t0 = time.time()
sampler = KCenterGreedy(embedding=embeddings.to(device), sampling_ratio=0.1)
idxs = sampler.select_coreset_idxs()
print(
    f"[4] done in {time.time()-t0:.2f}s. Selected {len(idxs)} / {embeddings.shape[0]}",
    flush=True,
)