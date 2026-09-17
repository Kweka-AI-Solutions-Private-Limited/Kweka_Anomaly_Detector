from pathlib import Path
import os

# ============================================================
# ONEDRIVE & LOCAL PATH NOTE:
# This project is currently located under OneDrive. OneDrive sync locking
# and Windows 260-character path limits for nested venv/torch files can cause
# slow I/O during dataset loading and language server (Pyrefly) errors.
# Recommendation: Move repository to a short local path outside OneDrive
# (e.g., C:\dev\Anomaly_Detector).
# ============================================================

# ============================================================
# MUST run before importing rich/anomalib/lightning anywhere
# ============================================================
os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import rich.console
rich.console._is_jupyter = lambda: False

import time
import json
import threading

import torch
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# CONFIGURATION
# ============================================================
DEV_MODE = True  # Set False for full baseline run (wide_resnet50_2, 0.1 ratio, layer2+layer3)

if DEV_MODE:
    coreset_sampling_ratio = 0.05
    layers = ["layer2"]
    backbone = "wide_resnet50_2"
else:
    coreset_sampling_ratio = 0.05
    layers = ["layer2"]
    backbone = "wide_resnet50_2"

# Batch size configuration
batch_size = 32  # Default 32. Drop to 16 or 8 if CUDA Out-Of-Memory (OOM) errors occur on low VRAM GPUs (e.g. RTX 3050 Ti Laptop 4GB).


# ============================================================
# Heartbeat helper — proves the process is alive during long steps
# ============================================================
def start_heartbeat(label, interval=15):
    stop_event = threading.Event()
    t0 = time.time()

    def _beat():
        while not stop_event.wait(interval):
            print(f"    ...[{label}] still running, {time.time()-t0:.0f}s elapsed", flush=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "patchcore_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PATCHCORE METAL NUT BASELINE")
print("=" * 70)
print(f"Mode:        {'DEV_MODE (Fast Run)' if DEV_MODE else 'PRODUCTION (Full Run)'}")
print(f"Backbone:    {backbone}")
print(f"Layers:      {layers}")
print(f"Coreset:     {coreset_sampling_ratio}")
print(f"Batch Size:  {batch_size}")
print(f"Data root:   {DATA_ROOT}")
print(f"Output dir:  {OUTPUT_DIR}")

# ============================================================
# Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

# ============================================================
# Dataset
# ============================================================
print("\n[1/4] Loading MVTec AD...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category="screw",
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_workers=0,
)
datamodule.setup()

train_loader = datamodule.train_dataloader()
test_loader = datamodule.test_dataloader()

print(f"Train images: {len(datamodule.train_data)}")
print(f"Train workers: {train_loader.num_workers}")
print(f"Train batch size: {train_loader.batch_size}")

# ============================================================
# PatchCore
# ============================================================
print("\n[2/4] Creating PatchCore...", flush=True)
model = Patchcore(
    backbone=backbone,
    layers=layers,
    pre_trained=True,
    coreset_sampling_ratio=coreset_sampling_ratio,
    num_neighbors=9,
)
print("PatchCore created successfully.")

# ============================================================
# Engine
# ============================================================
print("\n[3/4] Creating Anomalib Engine...", flush=True)
engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)
print("Engine created successfully.")

# ============================================================
# Build memory bank
# ============================================================
print("\n[4/4] Building PatchCore memory bank...", flush=True)
print("This uses ONLY the GOOD training images.")
print("Please wait...\n", flush=True)

start_time = time.time()
hb_stop = start_heartbeat("engine.fit / coreset selection")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb_stop.set()
build_time = time.time() - start_time

print("\n" + "=" * 70)
print("MEMORY BANK BUILD COMPLETE")
print("=" * 70)
print(f"Build time: {build_time:.2f} seconds")

# ============================================================
# Save model checkpoint
# ============================================================
print("\nSaving model checkpoint...", flush=True)
checkpoint_path = OUTPUT_DIR / "patchcore_screw_wideresnet50_l2_005.ckpt"
engine.trainer.save_checkpoint(str(checkpoint_path))
print(f"Saved: {checkpoint_path}")

# ============================================================
# Prediction
# ============================================================
print("\nRunning inference on test images...", flush=True)
start_time = time.time()
hb_stop = start_heartbeat("engine.predict")
try:
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint_path),
    )
finally:
    hb_stop.set()
inference_time = time.time() - start_time
print(f"\nInference completed in {inference_time:.2f} seconds")

# ============================================================
# Inspect predictions
# ============================================================
print("\nPrediction results:")
if predictions is None:
    print("No predictions returned.")
else:
    print(f"Prediction batches/results: {len(predictions)}")
    for i, prediction in enumerate(predictions[:5]):
        print(f"\nPrediction {i}:")
        print(prediction)

# ============================================================
# Save run info
# ============================================================
results = {
    "dev_mode": DEV_MODE,
    "dataset": "MVTec AD",
    "category": "metal_nut",
    "train_good_images": len(datamodule.train_data),
    "backbone": backbone,
    "layers": layers,
    "coreset_sampling_ratio": coreset_sampling_ratio,
    "num_neighbors": 9,
    "train_batch_size": batch_size,
    "eval_batch_size": batch_size,
    "num_workers": 0,
    "device": device,
    "build_time_seconds": build_time,
    "inference_time_seconds": inference_time,
    "checkpoint": str(checkpoint_path),
}
results_path = OUTPUT_DIR / "baseline_run_screw_wideresnet50_l2_005.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved run information: {results_path}")

print("\n" + "=" * 70)
print("BASELINE SCRIPT FINISHED")
print("=" * 70)