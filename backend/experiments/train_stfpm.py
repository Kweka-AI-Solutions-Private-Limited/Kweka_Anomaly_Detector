"""
train_stfpm.py

Fast Controlled STFPM Screening Experiment for MVTec AD 'screw'.
Applied Forensic Optimizations:
  - max_epochs:        15
  - limit_val_batches: 0 (Disables per-epoch validation loop during training)
  - num_workers:       4 (Multi-threaded asynchronous DataLoader)
  - batch_size:        4 (Optimized for 4GB VRAM RTX 3050 Ti)
  - backbone:          resnet18
  - layers:            ["layer1", "layer2", "layer3"]
  - input_size:        (256, 256)
  - training images:   320 GOOD images ONLY (0 defective images)

Logs: Total training time, time per epoch, batch counts, and GPU device metrics.
"""

from pathlib import Path
import os
import time
import json
import random

os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["FORCE_COLOR"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import rich.console
rich.console._is_jupyter = lambda: False

import torch
from lightning.pytorch.callbacks import Callback
from anomalib.data import MVTecAD
from anomalib.models import Stfpm
from anomalib.engine import Engine

# ── Configuration ─────────────────────────────────────────────
CATEGORY          = "screw"
BACKBONE          = "resnet18"
LAYERS            = ["layer1", "layer2", "layer3"]
IMAGE_SIZE        = (256, 256)
BATCH_SIZE        = 4
MAX_EPOCHS        = 15
NUM_WORKERS       = 4
LIMIT_VAL_BATCHES = 0   # Disable per-epoch validation loop to eliminate I/O overhead
SEED              = 42

random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT    = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "stfpm_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "stfpm_screw_resnet18_fast15.ckpt"
RESULTS_PATH     = OUTPUT_DIR / "baseline_run_screw_stfpm_fast15.json"

# ── Epoch Timing Callback ─────────────────────────────────────
class EpochTimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)
        epoch_num = trainer.current_epoch + 1
        total_epochs = trainer.max_epochs
        print(f"  [Epoch {epoch_num:02d}/{total_epochs:02d}] Completed in {elapsed:.2f}s (Avg: {sum(self.epoch_times)/len(self.epoch_times):.2f}s/epoch)", flush=True)


def main():
    print("=" * 70)
    print("FAST CONTROLLED STFPM SCREENING EXPERIMENT")
    print("=" * 70)
    print(f"Category:          {CATEGORY}")
    print(f"Backbone:          {BACKBONE}")
    print(f"Layers:            {LAYERS}")
    print(f"Input Size:        {IMAGE_SIZE}")
    print(f"Batch Size:        {BATCH_SIZE}")
    print(f"Max Epochs:        {MAX_EPOCHS}")
    print(f"Num Workers:       {NUM_WORKERS} (Multi-threaded DataLoader)")
    print(f"Limit Val Batches: {LIMIT_VAL_BATCHES} (Per-epoch val loop disabled)")
    print(f"Data root:         {DATA_ROOT}")
    print(f"Output dir:        {OUTPUT_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    print(f"\nExecution Device: {device.upper()}")
    print(f"GPU Name:         {gpu_name}")
    if torch.cuda.is_available():
        print(f"CUDA Version:     {torch.version.cuda}")

    # ── Dataset Loading ───────────────────────────────────────────
    print("\n[1/4] Loading GOOD training images for MVTec AD 'screw'...", flush=True)
    datamodule = MVTecAD(
        root=str(DATA_ROOT),
        category=CATEGORY,
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup()

    num_train_good = len(datamodule.train_data)
    batches_per_epoch = len(datamodule.train_dataloader())
    total_iterations = batches_per_epoch * MAX_EPOCHS

    print(f"\n" + "-" * 70)
    print(f"TRAINING BATCH & ITERATION SUMMARY:")
    print(f"  Training GOOD Images : {num_train_good} (0 defective images)")
    print(f"  Batches per Epoch    : {batches_per_epoch} batches ({BATCH_SIZE} images/batch)")
    print(f"  Total Max Epochs     : {MAX_EPOCHS}")
    print(f"  Total Optimizer Steps: {total_iterations:,} steps")
    print("-" * 70)

    # ── STFPM Model Initialization ────────────────────────────────
    print("\n[2/4] Initializing STFPM Model...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = Stfpm(
        backbone=BACKBONE,
        layers=LAYERS,
    )
    print("STFPM model initialized.")

    # ── Engine Setup ──────────────────────────────────────────────
    print("\n[3/4] Initializing Anomalib Engine...", flush=True)
    timing_callback = EpochTimingCallback()

    engine = Engine(
        max_epochs=MAX_EPOCHS,
        limit_val_batches=LIMIT_VAL_BATCHES,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        callbacks=[timing_callback],
    )
    print("Engine created with limit_val_batches=0 and EpochTimingCallback.")

    # ── Model Training ────────────────────────────────────────────
    print(f"\n[4/4] Training STFPM for {MAX_EPOCHS} epochs...", flush=True)

    start_time = time.time()
    engine.fit(model=model, datamodule=datamodule)
    total_train_time = time.time() - start_time

    avg_epoch_time = total_train_time / MAX_EPOCHS if MAX_EPOCHS > 0 else 0.0

    print("\n" + "=" * 70)
    print("STFPM TRAINING COMPLETE")
    print("=" * 70)
    print(f"Execution Device:      {device.upper()} ({gpu_name})")
    print(f"Total Training Batches:{batches_per_epoch} batches/epoch * {MAX_EPOCHS} epochs = {total_iterations:,} steps")
    print(f"Total Training Time:   {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)")
    print(f"Average Time per Epoch:{avg_epoch_time:.2f} seconds/epoch")
    print("=" * 70)

    # ── Save Checkpoint ───────────────────────────────────────────
    print("\nSaving checkpoint...", flush=True)
    engine.trainer.save_checkpoint(str(CHECKPOINT_PATH))
    print(f"Saved checkpoint: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")

    # ── Run Inference on Test Set ─────────────────────────────────
    print("\nMeasuring test inference time...", flush=True)
    start_time = time.time()
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT_PATH),
    )
    inference_time = time.time() - start_time
    num_pred_batches = len(predictions) if predictions is not None else 0
    print(f"Inference completed in {inference_time:.2f} seconds ({num_pred_batches} batches).")

    # ── Save Run Metadata ─────────────────────────────────────────
    metadata = {
        "model": "STFPM",
        "category": CATEGORY,
        "backbone": BACKBONE,
        "layers": LAYERS,
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "num_workers": NUM_WORKERS,
        "limit_val_batches": LIMIT_VAL_BATCHES,
        "device": device,
        "gpu_name": gpu_name,
        "number_of_training_good_images": num_train_good,
        "batches_per_epoch": batches_per_epoch,
        "total_optimizer_steps": total_iterations,
        "total_train_time_seconds": round(total_train_time, 2),
        "avg_epoch_time_seconds": round(avg_epoch_time, 2),
        "inference_time_seconds": round(inference_time, 2),
        "checkpoint_path": str(CHECKPOINT_PATH),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON: {RESULTS_PATH}")

    print("\n" + "=" * 70)
    print("FAST STFPM TRAINING COMPLETE — Ready for evaluate_stfpm.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
