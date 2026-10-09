"""
train_draem.py

Clean DRAEM Baseline Training Script for MVTec AD 'screw'.
Model:      DRÆM (Discriminatively Trained Reconstruction Embedding)
Backbone:   Reconstructive & Discriminative U-Net Subnetworks
Category:   screw (GOOD training images ONLY - 320 images)
Input Size: (256, 256) — Raw RGB [0, 1] without ImageNet normalization
Batch Size: 1 (Optimized for 4GB VRAM RTX 3050 Ti)
Max Epochs: 40
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
from anomalib.data import MVTecAD
from anomalib.models import Draem
from anomalib.engine import Engine

# ── Configuration ─────────────────────────────────────────────
CATEGORY   = "screw"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 1           # Batch size 1 to guarantee fits within 4GB VRAM
MAX_EPOCHS = 40
SEED       = 42

random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
DTD_ROOT     = PROJECT_ROOT / "data" / "dtd"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "draem_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DTD_ROOT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "draem_screw_256_001.ckpt"
RESULTS_PATH     = OUTPUT_DIR / "baseline_run_screw_draem_001.json"

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

print("=" * 70)
print("DRAEM BASELINE TRAINING EXPERIMENT (320 IMAGES, 40 EPOCHS)")
print("=" * 70)
print(f"Category:         {CATEGORY}")
print(f"Model:            DRÆM (Reconstructive + Discriminative Subnetworks)")
print(f"Input Size:       {IMAGE_SIZE}")
print(f"Batch Size:       {BATCH_SIZE} (Memory-conscious for 4GB VRAM RTX 3050 Ti)")
print(f"Max Epochs:       {MAX_EPOCHS}")
print(f"Data root:        {DATA_ROOT}")
print(f"DTD root:         {DTD_ROOT}")
print(f"Output dir:       {OUTPUT_DIR}")

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

num_train_good = len(datamodule.train_data)
print(f"  Training GOOD Images: {num_train_good} (0 defective images used)")

# ── DRAEM Model Initialization ────────────────────────────────
print("\n[2/4] Initializing DRÆM Model & DTD Anomaly Generator...", flush=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = Draem(dtd_dir=str(DTD_ROOT))
print("DRÆM model initialized successfully.")

# ── Engine Setup ──────────────────────────────────────────────
print("\n[3/4] Initializing Anomalib Engine...", flush=True)
engine = Engine(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)
print("Engine created.")

# ── Model Training ────────────────────────────────────────────
print(f"\n[4/4] Training DRÆM for {MAX_EPOCHS} epochs...", flush=True)

start_time = time.time()
hb_stop = start_heartbeat("DRÆM training")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb_stop.set()
train_time = time.time() - start_time

print("\n" + "=" * 70)
print("DRAEM TRAINING COMPLETE")
print("=" * 70)
print(f"Total training time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

# ── Save Checkpoint ───────────────────────────────────────────
print("\nSaving checkpoint...", flush=True)
engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
print(f"Saved checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")

# ── Run Inference on Test Set ─────────────────────────────────
print("\nMeasuring test inference time...", flush=True)
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
    "model": "DRAEM",
    "category": CATEGORY,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "number_of_training_good_images": num_train_good,
    "train_time_seconds": round(train_time, 2),
    "inference_time_seconds": round(inference_time, 2),
    "checkpoint_path": str(CHECKPOINT_PATH),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata JSON: {RESULTS_PATH}")

print("\n" + "=" * 70)
print("DRAEM TRAINING COMPLETE — Ready for evaluate_draem.py")
print("=" * 70)
