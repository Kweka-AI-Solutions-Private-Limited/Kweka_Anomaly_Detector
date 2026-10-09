"""
run_patchcore_l1l2.py

Controlled PatchCore Layer1 + Layer2 Experiment on MVTec AD screw.
Config:
  Backbone:           wide_resnet50_2
  Layers:             ["layer1", "layer2"]
  Coreset ratio:      0.05 (5%)
  Category:           screw (GOOD training images ONLY - 320 images)
  Image Resolution:   (256, 256) — Pre-resized at PyTorch Dataset level
  Spatial Alignment:  MultiLayerPooler
                      - layer1: AvgPool2d(3, stride=2, padding=1) -> 32x32
                      - layer2: AvgPool2d(3, stride=1, padding=1) -> 32x32
                      Produces 32x32 spatial map (1,024 patches/image) with 1,536 channels.
  Total Train Patches: 320 * 1,024 = 327,680 patch vectors (5% coreset = 16,384 vectors)
"""

from pathlib import Path
import os
import time
import json
import random
import threading

os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import rich.console
rich.console._is_jupyter = lambda: False

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# ── Configuration ─────────────────────────────────────────────
CATEGORY               = "screw"
BACKBONE               = "wide_resnet50_2"
LAYERS                 = ["layer1", "layer2"]
CORESET_SAMPLING_RATIO = 0.05
NUM_NEIGHBORS          = 9
IMAGE_SIZE             = (256, 256)
BATCH_SIZE             = 16
SEED                   = 42

random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "patchcore_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "patchcore_screw_wideresnet50_l1l2_005.ckpt"
RESULTS_PATH     = OUTPUT_DIR / "baseline_run_screw_wideresnet50_l1l2_005.json"

# ── Heartbeat logger ──────────────────────────────────────────
def start_heartbeat(label, interval=15):
    stop_event = threading.Event()
    t0 = time.time()
    def _beat():
        while not stop_event.wait(interval):
            print(f"    ...[{label}] still running, {time.time()-t0:.0f}s elapsed", flush=True)
    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event

# ── MultiLayerPooler for Layer1+Layer2 Spatial Alignment ─────
class MultiLayerPooler(nn.Module):
    """
    Applies stride=2 pooling on layer1 (64x64 -> 32x32) and stride=1 on layer2 (32x32 -> 32x32)
    so both feature maps natively align to 32x32 without needing bilinear upsampling.
    """
    def __init__(self):
        super().__init__()
        self.pool_l1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_l2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        if feature_tensor.shape[-1] == 64:
            return self.pool_l1(feature_tensor)
        else:
            return self.pool_l2(feature_tensor)

# ── Dataset Pre-resizing Wrapper (Guarantees 256x256 Preprocessing) ────
class PreResizedDataset(Dataset):
    def __init__(self, dataset, target_size=(256, 256)):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if hasattr(item, "image") and isinstance(item.image, torch.Tensor):
            if item.image.shape[-2:] != self.target_size:
                item.image = F.interpolate(
                    item.image.unsqueeze(0),
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)

print("=" * 70)
print("PATCHCORE LAYER1+LAYER2 MULTI-SCALE EXPERIMENT (320 IMAGES, 5% CORESET)")
print("=" * 70)
print(f"Category:          {CATEGORY}")
print(f"Backbone:          {BACKBONE}")
print(f"Layers:            {LAYERS}")
print(f"Coreset Ratio:     {CORESET_SAMPLING_RATIO} (5%)")
print(f"Num Neighbors:     {NUM_NEIGHBORS}")
print(f"Pre-resized Size:  {IMAGE_SIZE} (Dataset Pre-processing)")
print(f"Spatial Alignment: MultiLayerPooler (layer1 2x pool -> 32x32, layer2 1x pool -> 32x32)")
print(f"Data root:         {DATA_ROOT}")
print(f"Output dir:        {OUTPUT_DIR}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")

# ── Dataset Loading ───────────────────────────────────────────
print("\n[1/4] Loading GOOD training images for MVTec AD 'screw'...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()

raw_train_data = datamodule.train_data
num_train_good = len(raw_train_data)

# Wrap train_data to guarantee 256x256 tensors before feature extraction
datamodule.train_data = PreResizedDataset(raw_train_data, target_size=IMAGE_SIZE)

patches_per_image = 32 * 32  # 32x32 = 1,024 patches
total_expected_patches = num_train_good * patches_per_image
expected_coreset_vectors = int(total_expected_patches * CORESET_SAMPLING_RATIO)
embedding_dim = 512 + 1024   # 1,536 feature channels

print(f"\n" + "-" * 70)
print(f"PRE-FIT MEMORY STATS & PATCH COUNT:")
print(f"  Training GOOD Images     : {num_train_good} (0 defective images used)")
print(f"  Feature Dimensions       : {embedding_dim} channels (layer1: 512 + layer2: 1024)")
print(f"  Aligned Spatial Map      : 32 x 32 = {patches_per_image} patches/image")
print(f"  Total Expected Patches   : {total_expected_patches:,} patch vectors")
print(f"  Expected Coreset Vectors : {expected_coreset_vectors:,} vectors (5% sampling ratio)")
print("-" * 70)

# ── PatchCore Model Setup & MultiLayerPooler Injection ───────
print(f"\n[2/4] Initializing PatchCore with {LAYERS} ...", flush=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = Patchcore(
    backbone=BACKBONE,
    layers=LAYERS,
    pre_trained=True,
    coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
    num_neighbors=NUM_NEIGHBORS,
)

# Inject MultiLayerPooler for technically valid layer1+layer2 spatial feature alignment
model.model.feature_pooler = MultiLayerPooler()
print(f"Configured feature_pooler: {model.model.feature_pooler}")

# ── Engine Setup ──────────────────────────────────────────────
print("\n[3/4] Initializing Anomalib Engine...", flush=True)
engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)
print("Engine created.")

# ── Memory Bank Fitting ───────────────────────────────────────
print("\n[4/4] Building PatchCore Layer1+Layer2 memory bank (320 images, 5% coreset)...", flush=True)
print("Extracting concatenated 1536-dim layer1+layer2 feature representations...")

start_time = time.time()
hb_stop = start_heartbeat("PatchCore layer1+layer2 memory bank construction")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb_stop.set()
build_time = time.time() - start_time

print("\n" + "=" * 70)
print("MEMORY BANK BUILD COMPLETE")
print("=" * 70)
print(f"Memory bank build time: {build_time:.2f} seconds")

# ── Save Checkpoint ───────────────────────────────────────────
print("\nSaving checkpoint...", flush=True)
engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
print(f"Saved checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")

# ── Run Inference on Test Set ─────────────────────────────────
print("\nMeasuring test inference time...", flush=True)

# Pre-resize test_data as well to guarantee 256x256 input tensors
if hasattr(datamodule, "test_data"):
    datamodule.test_data = PreResizedDataset(datamodule.test_data, target_size=IMAGE_SIZE)

start_time = time.time()
hb_stop = start_heartbeat("engine.predict")
try:
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT_PATH),
    )
finally:
    hb_stop.set()
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds ({len(predictions)} batches).")

# ── Save Run Metadata ─────────────────────────────────────────
metadata = {
    "model": "PatchCore",
    "category": CATEGORY,
    "backbone": BACKBONE,
    "layers": LAYERS,
    "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
    "num_neighbors": NUM_NEIGHBORS,
    "image_size": list(IMAGE_SIZE),
    "embedding_channels": embedding_dim,
    "spatial_reduction": "MultiLayerPooler (layer1: 64x64->32x32, layer2: 32x32->32x32)",
    "number_of_training_good_images": num_train_good,
    "expected_patches": total_expected_patches,
    "expected_coreset_vectors": expected_coreset_vectors,
    "build_time_seconds": round(build_time, 2),
    "inference_time_seconds": round(inference_time, 2),
    "checkpoint_path": str(CHECKPOINT_PATH),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata JSON: {RESULTS_PATH}")

print("\n" + "=" * 70)
print("PATCHCORE LAYER1+LAYER2 TRAINING COMPLETE — Ready for evaluate_patchcore_l1l2.py")
print("=" * 70)
