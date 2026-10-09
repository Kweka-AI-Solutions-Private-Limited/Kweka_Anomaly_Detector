"""
build_layer1_checkpoint.py — Builds Layer1 PatchCore Memory Bank Checkpoint
-----------------------------------------------------------------------------
Trains PatchCore with layers=["layer1"] on MVTec AD Screw GOOD training set
and saves state_dict to outputs/patchcore_baseline/FINAL/patchcore_screw_wideresnet50_l1_005.ckpt.
Prints exact feature dimensions and memory bank coreset shape.
"""

import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

BASE_DIR = SRC_DIR.parent
DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
OUTPUT_CKPT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "FINAL"
    / "patchcore_screw_wideresnet50_l1_005.ckpt"
)

def main():
    print("=" * 72)
    print("BUILDING LAYER1 PATCHCORE MEMORY BANK CHECKPOINT")
    print("=" * 72)
    
    OUTPUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    start_t = time.time()

    print("1. Loading MVTecAD Screw training data...")
    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category="screw",
        num_workers=0,
    )
    datamodule.setup()

    print("2. Initializing PatchCore WideResNet50_2 with layers=['layer1']...")
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer1"],
        pre_trained=True,
        coreset_sampling_ratio=0.05,
        num_neighbors=9,
    )

    engine = Engine(accelerator="auto", devices=1, enable_progress_bar=False)

    print("3. Fitting memory bank on 320 GOOD training images...")
    engine.fit(model=model, datamodule=datamodule)

    print(f"4. Saving state dict to {OUTPUT_CKPT}...")
    torch.save(model.state_dict(), OUTPUT_CKPT)

    # Inspect model memory bank
    state = model.state_dict()
    print("\n--- LAYER1 MEMORY BANK DETAILS ---")
    if "model.memory_bank" in state:
        mb = state["model.memory_bank"]
        print(f"Memory Bank Shape: {mb.shape}")
        print(f"Total Coreset Vectors: {mb.shape[0]}")
        print(f"Feature Dimension (Channels): {mb.shape[1]}")
    elif hasattr(model, "memory_bank"):
        mb = getattr(model, "memory_bank")
        print(f"Memory Bank Shape: {mb.shape}")

    elapsed = round(time.time() - start_t, 2)
    print(f"✓ Layer1 Checkpoint successfully built and saved in {elapsed}s")

if __name__ == "__main__":
    main()
