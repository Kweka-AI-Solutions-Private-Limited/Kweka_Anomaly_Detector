import os
os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import time
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from anomalib.data import MVTecAD
from anomalib.models import ReverseDistillation
from anomalib.models.image.reverse_distillation.anomaly_map import AnomalyMapGenerationMode

CATEGORY    = "screw"
BACKBONE    = "wide_resnet50_2"
MAX_SAMPLES = 100
MAX_EPOCHS  = 40
BATCH_SIZE  = 1
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_ROOT       = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "rd4ad_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / f"rd4ad_screw_{BACKBONE}.ckpt"
RESULTS_PATH     = OUTPUT_DIR / f"baseline_run_screw_{BACKBONE}.json"

print("=" * 70)
print("DIRECT PYTORCH RD4AD ENGINE (100 IMAGES, 40 EPOCHS)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device:      {device}")
if torch.cuda.is_available():
    print(f"GPU:         {torch.cuda.get_device_name(0)}")

# ── Data ─────────────────────────────────────────────────────
print(f"\n[1/3] Loading dataset ({MAX_SAMPLES} good images @ 256x256, batch={BATCH_SIZE}) ...", flush=True)
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category=CATEGORY,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=0,
)
datamodule.setup()

full_train = datamodule.train_data
if len(full_train) > MAX_SAMPLES:
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    train_subset = Subset(full_train, sorted(indices[:MAX_SAMPLES]))
else:
    train_subset = full_train

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=datamodule.train_dataloader().collate_fn
)
print(f"Train images: {len(train_subset)} ({len(train_loader)} batches/epoch)")

# ── Model Initialization & Memory Cleanup ─────────────────────
print(f"\n[2/3] Initializing RD4AD ({BACKBONE}) ...", flush=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

model = ReverseDistillation(
    backbone=BACKBONE,
    pre_trained=True,
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode=AnomalyMapGenerationMode.MULTIPLY,
)
model.to(device)
model.train()

if torch.cuda.is_available():
    print(
        f"GPU memory after model load: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated / "
        f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved"
    )

optimizer = model.configure_optimizers()
if isinstance(optimizer, (list, tuple)):
    optimizer = optimizer[0]

# ── Direct Training Loop ──────────────────────────────────────
print(f"\n[3/3] Training for {MAX_EPOCHS} epochs ...", flush=True)
start_time = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    total_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        if hasattr(batch, "to"):
            batch = batch.to(device)
        if hasattr(batch, "image") and isinstance(batch.image, torch.Tensor):
            batch.image = batch.image.to(device)
            # Resize image on GPU to 256x256 to guarantee 4GB VRAM safety
            if batch.image.shape[-2:] != (256, 256):
                batch.image = F.interpolate(batch.image, size=(256, 256), mode="bilinear", align_corners=False)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                    if k == "image" and v.shape[-2:] != (256, 256):
                        batch[k] = F.interpolate(v, size=(256, 256), mode="bilinear", align_corners=False)
        
        optimizer.zero_grad()
        loss = model.training_step(batch, batch_count)
        if isinstance(loss, dict):
            loss = loss["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / max(1, batch_count)
    if epoch % 5 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:2d}/{MAX_EPOCHS:2d} | Loss: {avg_loss:.4f} | Elapsed: {elapsed:.1f}s", flush=True)

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.1f} seconds!")

# ── Save Checkpoint ───────────────────────────────────────────
print("Saving Lightning-compatible checkpoint ...", flush=True)
checkpoint_data = {
    "epoch": MAX_EPOCHS,
    "global_step": MAX_EPOCHS * len(train_loader),
    "pytorch-lightning_version": "2.2.0",
    "state_dict": model.state_dict(),
    "hyper_parameters": model.hparams if hasattr(model, "hparams") else {},
}
torch.save(checkpoint_data, str(CHECKPOINT_PATH))
print(f"Saved: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB)")

run_info = {
    "model": "ReverseDistillation",
    "backbone": BACKBONE,
    "train_images": len(train_subset),
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_time_seconds": train_time,
    "checkpoint": str(CHECKPOINT_PATH),
}
with open(RESULTS_PATH, "w") as f:
    json.dump(run_info, f, indent=2)

print("\n" + "=" * 70)
print("TRAINING SUCCESSFUL — Ready for evaluate_rd4ad.py")
print("=" * 70)
