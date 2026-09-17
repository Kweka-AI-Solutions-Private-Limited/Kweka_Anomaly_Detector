from pathlib import Path
import os
import sys
import time
import torch

os.environ["TQDM_DISABLE"] = "1"

print("Python executable:", sys.executable)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_anomaly_detection"

print("1. Initializing MVTecAD Datamodule...")
datamodule = MVTecAD(
    root=str(DATA_ROOT),
    category="metal_nut",
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,
)
datamodule.setup()
print("   Train dataloader length:", len(datamodule.train_dataloader()))

print("2. Initializing Patchcore model...")
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)
print("   Patchcore model initialized.")

print("3. Initializing Anomalib Engine...")
engine = Engine(
    max_epochs=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    enable_progress_bar=False,
)
print("   Engine initialized.")

print("4. Testing fit execution...")
t0 = time.time()
engine.fit(model=model, datamodule=datamodule)
print(f"   fit completed in {time.time() - t0:.2f}s")
