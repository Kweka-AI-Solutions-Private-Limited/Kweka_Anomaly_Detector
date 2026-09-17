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
import threading
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine

# ============================================================
# Why ReverseDistillation (RD4AD)?
# ─────────────────────────────────
# • Trains in ~5–10 min (epoch-based, not 70k steps like EfficientAD)
# • Teacher → Student knowledge distillation produces SHARP, pixel-accurate
#   anomaly maps (vs PatchCore's blurry patch-votes)
# • MVTec pixel AUROC ~98%+ — comparable to EfficientAD
# • Wide ResNet-50 backbone with large_kernel_size for fine-grained detail
# ============================================================
CATEGORY   = "screw"
BACKBONE   = "wide_resnet50_2"  # strong backbone, same as our PatchCore baseline
BATCH_SIZE = 32
MAX_EPOCHS = 100   # ~5 min on GPU; RD4AD converges fast and reliably at 100 epochs

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_ROOT      = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR     = PROJECT_ROOT / "outputs" / "rd4ad_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / f"rd4ad_screw_{BACKBONE}.ckpt"
RESULTS_PATH    = OUTPUT_DIR / f"baseline_run_screw_{BACKBONE}.json"


def start_heartbeat(label, interval=30):
    stop_event = threading.Event()
    t0 = time.time()
    def _beat():
        while not stop_event.wait(interval):
            print(f"    ...[{label}] still running, {time.time()-t0:.0f}s elapsed", flush=True)
    threading.Thread(target=_beat, daemon=True).start()
    return stop_event


print("=" * 70)
print("REVERSE DISTILLATION (RD4AD) — SCREW TRAINING")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:     {device}")
if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Backbone:   {BACKBONE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Output dir: {OUTPUT_DIR}")

# ── Dataset ──────────────────────────────────────────────────
print(f"\n[1/3] Loading MVTec AD — {CATEGORY} ...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()
print(f"Train (good): {len(datamodule.train_data)}")
print(f"Test:         {len(datamodule.test_data)}")

from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode

# ── Model ─────────────────────────────────────────────────────
print(f"\n[2/3] Creating ReverseDistillation ({BACKBONE}) ...", flush=True)
model = ReverseDistillation(
    backbone=BACKBONE,
    pre_trained=True,
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode=AnomalyMapGenerationMode.MULTIPLY,
)
print("Model created.")

# ── Engine ───────────────────────────────────────────────────
print(f"\n[3/3] Training {MAX_EPOCHS} epochs ...", flush=True)
engine = Engine(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCHS,
    enable_progress_bar=False,
)

start_time = time.time()
hb = start_heartbeat("RD4AD training")
try:
    engine.fit(model=model, datamodule=datamodule)
finally:
    hb.set()

train_time = time.time() - start_time
print(f"\nTraining done in {train_time:.1f}s ({train_time/60:.1f} min)")

# ── Save checkpoint ───────────────────────────────────────────
print("Saving checkpoint ...", flush=True)
engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
print(f"Saved: {CHECKPOINT_PATH}")

# ── Save run info ─────────────────────────────────────────────
run_info = {
    "model":             "ReverseDistillation",
    "backbone":          BACKBONE,
    "layers":            ["layer1", "layer2", "layer3"],
    "dataset":           "MVTec AD",
    "category":          CATEGORY,
    "train_good_images": len(datamodule.train_data),
    "test_images":       len(datamodule.test_data),
    "batch_size":        BATCH_SIZE,
    "max_epochs":        MAX_EPOCHS,
    "device":            device,
    "train_time_seconds": train_time,
    "checkpoint":        str(CHECKPOINT_PATH),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(run_info, f, indent=2)
print(f"Run info: {RESULTS_PATH}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE — run evaluate_rd4ad.py next")
print("=" * 70)
