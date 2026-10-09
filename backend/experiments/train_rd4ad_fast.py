import os

os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import rich.console
rich.console._is_jupyter = lambda: False

import time
import json
import random
import threading
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import ReverseDistillation
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from anomalib.engine import Engine

# ============================================================
# RD4AD FAST ENGINE (MAX 100 IMAGES, 60 EPOCHS, ~3 MIN GPU)
# ────────────────────────────────────────────────────────────
# • Subsamples training dataset to MAX 100 good images
# • 60 epochs is the exact sweet spot for Reverse Distillation loss
# • Multi-scale features (layer1 + layer2 + layer3) produce sharp maps
# • Target: IoU ≥ 80%, Recall ≥ 90%, FPR < 5% in < 5 minutes
# ============================================================
CATEGORY    = "screw"
BACKBONE    = "wide_resnet50_2"
MAX_SAMPLES = 100   # Max 100 good training images
MAX_EPOCHS  = 60    # ~3 minutes total execution on GPU
BATCH_SIZE  = 16    # 100 images / 16 = 7 batches per epoch (420 total steps)
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_ROOT       = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "rd4ad_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / f"rd4ad_screw_{BACKBONE}.ckpt"
RESULTS_PATH     = OUTPUT_DIR / f"baseline_run_screw_{BACKBONE}.json"


def start_heartbeat(label, interval=20):
    stop_event = threading.Event()
    t0 = time.time()
    def _beat():
        while not stop_event.wait(interval):
            print(f"    ...[{label}] training in progress: {time.time()-t0:.0f}s elapsed", flush=True)
    threading.Thread(target=_beat, daemon=True).start()
    return stop_event


print("=" * 70)
print("RD4AD FAST PRECISION ENGINE (MAX 100 IMAGES, 60 EPOCHS)")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:      {device}")
if torch.cuda.is_available():
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
print(f"Backbone:    {BACKBONE}")
print(f"Max Samples: {MAX_SAMPLES} good images")
print(f"Max Epochs:  {MAX_EPOCHS}")
print(f"Batch Size:  {BATCH_SIZE}")
print(f"Output:      {CHECKPOINT_PATH}")

# ── Dataset ──────────────────────────────────────────────────
print(f"\n[1/3] Loading dataset & subsampling to {MAX_SAMPLES} images ...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()

# Subsample train_data to max 100 samples deterministically
original_train_count = len(datamodule.train_data)
if original_train_count > MAX_SAMPLES:
    indices = list(range(original_train_count))
    random.shuffle(indices)
    keep_indices = sorted(indices[:MAX_SAMPLES])
    datamodule.train_data = torch.utils.data.Subset(datamodule.train_data, keep_indices)
    print(f"Subsampled train set: {original_train_count} -> {len(datamodule.train_data)} good images")
else:
    print(f"Train dataset size: {original_train_count} good images")

print(f"Test dataset size:  {len(datamodule.test_data)} test images")

# ── Model ─────────────────────────────────────────────────────
print(f"\n[2/3] Initializing ReverseDistillation ({BACKBONE}) ...", flush=True)
model = ReverseDistillation(
    backbone=BACKBONE,
    pre_trained=True,
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode=AnomalyMapGenerationMode.MULTIPLY,
)
print("Model initialized.")

# ── Engine ───────────────────────────────────────────────────
print(f"\n[3/3] Training for {MAX_EPOCHS} epochs ...", flush=True)
engine = Engine(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCHS,
    limit_val_batches=0,  # Skip per-epoch validation loop (prevents 70-min evaluation overhead)
    enable_progress_bar=False,
)

start_time = time.time()
hb = start_heartbeat("RD4AD Fast Training")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb.set()

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/60:.2f} minutes)")

# ── Save checkpoint ───────────────────────────────────────────
print("Saving model checkpoint ...", flush=True)
engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
print(f"Saved checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")

# ── Save run summary ──────────────────────────────────────────
run_info = {
    "model":              "ReverseDistillation",
    "backbone":           BACKBONE,
    "layers":             ["layer1", "layer2", "layer3"],
    "dataset":            "MVTec AD",
    "category":           CATEGORY,
    "max_train_samples":  MAX_SAMPLES,
    "actual_train_count": len(datamodule.train_data),
    "test_images":        len(datamodule.test_data),
    "batch_size":         BATCH_SIZE,
    "max_epochs":         MAX_EPOCHS,
    "device":             device,
    "train_time_seconds": train_time,
    "checkpoint":         str(CHECKPOINT_PATH),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(run_info, f, indent=2)

print("\n" + "=" * 70)
print("TRAINING SUCCESSFUL — Ready for evaluate_rd4ad.py")
print("=" * 70)
