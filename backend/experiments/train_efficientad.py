import os

# ============================================================
# Must set before importing rich/lightning anywhere
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
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import EfficientAd
from anomalib.engine import Engine

# ============================================================
# CONFIGURATION
# ============================================================
# EfficientAD uses a student-teacher (PDN) architecture.
# It trains on GOOD images only, same as PatchCore.
# "medium" gives the best IoU; "small" is faster but slightly lower quality.
MODEL_SIZE = "medium"      # "small" or "medium"
CATEGORY = "screw"
# IMPORTANT: Anomalib's EfficientAD enforces train_batch_size=1 as a hard
# constraint. The PDN teacher-student architecture normalizes per sample.
# eval_batch_size can be higher (8-16) for faster inference.
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE  = 8

# ============================================================
# Heartbeat helper
# ============================================================
def start_heartbeat(label, interval=30):
    stop_event = threading.Event()
    t0 = time.time()

    def _beat():
        while not stop_event.wait(interval):
            elapsed = time.time() - t0
            print(f"    ...[{label}] still running, {elapsed:.0f}s elapsed", flush=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return stop_event


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "efficientad_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / f"efficientad_screw_{MODEL_SIZE}.ckpt"
RESULTS_PATH = OUTPUT_DIR / f"baseline_run_screw_{MODEL_SIZE}.json"

print("=" * 70)
print("EFFICIENTAD SCREW TRAINING")
print("=" * 70)
print(f"Model Size:       {MODEL_SIZE}")
print(f"Category:         {CATEGORY}")
print(f"Train Batch Size: {TRAIN_BATCH_SIZE} (EfficientAD requires 1)")
print(f"Eval Batch Size:  {EVAL_BATCH_SIZE}")
print(f"Data root:        {DATA_ROOT}")
print(f"Output dir:       {OUTPUT_DIR}")

# ============================================================
# Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:   {vram_gb:.1f} GB")
    if vram_gb < 4.0:
        print("WARNING: < 4 GB VRAM detected. Consider reducing BATCH_SIZE to 4.")

# ============================================================
# Dataset
# ============================================================
print(f"\n[1/4] Loading MVTec AD — category: {CATEGORY} ...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category=CATEGORY,
    train_batch_size=TRAIN_BATCH_SIZE,  # Must be 1 for EfficientAD
    eval_batch_size=EVAL_BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()

print(f"Train images (good only): {len(datamodule.train_data)}")
print(f"Test images:              {len(datamodule.test_data)}")

# ============================================================
# Model
# ============================================================
print(f"\n[2/4] Creating EfficientAD ({MODEL_SIZE}) ...", flush=True)

# EfficientAD key hyperparameters:
#   teacher_out_channels: PDN teacher output channels (384 for medium, 128 for small)
#   model_size: controls the PDN architecture size
#   lr: learning rate (default 1e-4)
#   weight_decay: default 1e-5
#   padding: whether to pad the PDN input — False gives slightly better localization
#   pad_maps: pads anomaly maps to original image size for accurate IoU
model = EfficientAd(
    teacher_out_channels=384 if MODEL_SIZE == "medium" else 128,
    model_size=MODEL_SIZE,
    lr=0.0001,
    weight_decay=0.00001,
    padding=False,
    pad_maps=True,
)

print("EfficientAD model created.")

# ============================================================
# Engine
# ============================================================
print("\n[3/4] Creating Anomalib Engine ...", flush=True)
engine = Engine(
    accelerator="auto",
    devices=1,
    # EfficientAD paper trains for 70,000 gradient steps (batch_size=1).
    # Use max_steps instead of max_epochs so Lightning doesn't stop at epoch 1.
    max_steps=70000,
    max_epochs=-1,   # -1 = no epoch limit; stop is controlled by max_steps
    enable_progress_bar=False,
)
print("Engine created.")

# ============================================================
# Train (builds student-teacher memory)
# ============================================================
print("\n[4/4] Training EfficientAD ...", flush=True)
print("This may take 5–20 minutes depending on GPU speed.")
print("EfficientAD converges quickly compared to flow-based models.\n", flush=True)

start_time = time.time()
hb_stop = start_heartbeat("EfficientAD training")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb_stop.set()

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.2f}s ({train_time/60:.1f} min)")

# ============================================================
# Save checkpoint
# ============================================================
print(f"\nSaving checkpoint ...", flush=True)
engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
print(f"Saved: {CHECKPOINT_PATH}")

# ============================================================
# Quick test inference
# ============================================================
print("\nRunning quick test inference ...", flush=True)
start_infer = time.time()
hb_stop2 = start_heartbeat("EfficientAD inference")
try:
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT_PATH),
    )
finally:
    hb_stop2.set()

infer_time = time.time() - start_infer
n_batches = len(predictions) if predictions else 0
print(f"Inference done: {n_batches} prediction batch(es) in {infer_time:.2f}s")

# ============================================================
# Save run info
# ============================================================
run_info = {
    "model": "EfficientAD",
    "model_size": MODEL_SIZE,
    "dataset": "MVTec AD",
    "category": CATEGORY,
    "train_good_images": len(datamodule.train_data),
    "test_images": len(datamodule.test_data),
    "teacher_out_channels": 384 if MODEL_SIZE == "medium" else 128,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "eval_batch_size": EVAL_BATCH_SIZE,
    "device": device,
    "train_time_seconds": train_time,
    "infer_time_seconds": infer_time,
    "checkpoint": str(CHECKPOINT_PATH),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(run_info, f, indent=2)
print(f"Run info saved: {RESULTS_PATH}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print("Next: run evaluate_efficientad.py to measure IoU / Recall / FPR")
print("=" * 70)
