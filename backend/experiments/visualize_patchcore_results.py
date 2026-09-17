from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_ROOT = BASE_DIR / "data" / "mvtec_anomaly_detection"
CHECKPOINT = (
    BASE_DIR
    / "outputs"
    / "patchcore_baseline"
    / "patchcore_metal_nut.ckpt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "dev_evaluation"
    / "visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA
# ============================================================

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="metal_nut",
    num_workers=0,
)

datamodule.setup()


# ============================================================
# EXACT DEV MODEL
# ============================================================

model = Patchcore(
    backbone="resnet18",
    layers=["layer2"],
    pre_trained=True,
    coreset_sampling_ratio=0.02,
    num_neighbors=9,
)


# ============================================================
# ENGINE
# ============================================================

engine = Engine(
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)


# ============================================================
# PREDICTIONS
# ============================================================

print("Running PatchCore predictions...")

prediction_batches = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

if prediction_batches is None:
    raise RuntimeError("engine.predict() returned None.")

print(f"Prediction batches: {len(prediction_batches)}")


# ============================================================
# HELPER
# ============================================================

def to_numpy(value):
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def scalar_from_batch(value, index):
    """
    Extract one value from a batched tensor/list.
    """

    if value is None:
        return None

    array = to_numpy(value)

    if array is None:
        return None

    return array[index]


def path_from_batch(value, index):
    """
    Extract one image path from a batch.
    """

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return Path(str(value[index]))

    return Path(str(value))


# ============================================================
# CONVERT BATCHES INTO INDIVIDUAL PREDICTIONS
# ============================================================

samples = []

for batch in prediction_batches:

    # Number of images in this batch
    image_paths = getattr(batch, "image_path", None)

    if image_paths is None:
        continue

    batch_size = len(image_paths)

    for i in range(batch_size):

        sample = {
            "image_path": path_from_batch(
                image_paths,
                i,
            ),

            "gt_label": scalar_from_batch(
                getattr(batch, "gt_label", None),
                i,
            ),

            "pred_label": scalar_from_batch(
                getattr(batch, "pred_label", None),
                i,
            ),

            "pred_score": scalar_from_batch(
                getattr(batch, "pred_score", None),
                i,
            ),

            "gt_mask": scalar_from_batch(
                getattr(batch, "gt_mask", None),
                i,
            ),

            "pred_mask": scalar_from_batch(
                getattr(batch, "pred_mask", None),
                i,
            ),

            "anomaly_map": scalar_from_batch(
                getattr(batch, "anomaly_map", None),
                i,
            ),
        }

        samples.append(sample)


print(f"Individual images: {len(samples)}")


if len(samples) == 0:
    raise RuntimeError("No individual samples were extracted.")


# ============================================================
# SELECT REPRESENTATIVE CASES
# ============================================================

selected = {}

for sample in samples:

    image_path = sample["image_path"]

    defect_type = image_path.parent.name

    gt_label = bool(np.asarray(sample["gt_label"]).reshape(-1)[0])
    pred_label = bool(np.asarray(sample["pred_label"]).reshape(-1)[0])


    # --------------------------------------------------------
    # One sample from each defect type
    # --------------------------------------------------------

    if defect_type in [
        "bent",
        "color",
        "flip",
        "scratch",
    ]:

        if defect_type not in selected:
            selected[defect_type] = sample


    # --------------------------------------------------------
    # First false-positive GOOD
    # --------------------------------------------------------

    if (
        defect_type == "good"
        and not gt_label
        and pred_label
        and "false_positive_good" not in selected
    ):

        selected["false_positive_good"] = sample


print("\nSelected cases:")

for name, sample in selected.items():

    print(
        f"{name:22} -> "
        f"{sample['image_path']}"
    )


# ============================================================
# NORMALIZE HEATMAP
# ============================================================

def normalize_heatmap(anomaly_map):

    anomaly_map = anomaly_map.astype(np.float32)

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(anomaly_map)

    return (
        anomaly_map - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# VISUALIZE
# ============================================================

def visualize_prediction(name, sample):

    image_path = sample["image_path"]

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    image = np.array(
        Image.open(image_path).convert("RGB")
    )


    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    gt_mask = sample["gt_mask"]

    if gt_mask is not None:
        gt_mask = np.squeeze(gt_mask)


    pred_mask = sample["pred_mask"]

    if pred_mask is not None:
        pred_mask = np.squeeze(pred_mask)


    anomaly_map = sample["anomaly_map"]

    if anomaly_map is not None:

        anomaly_map = np.squeeze(anomaly_map)

        anomaly_map = normalize_heatmap(
            anomaly_map
        )


    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    gt_label = bool(
        np.asarray(sample["gt_label"])
        .reshape(-1)[0]
    )

    pred_label = bool(
        np.asarray(sample["pred_label"])
        .reshape(-1)[0]
    )

    pred_score = float(
        np.asarray(sample["pred_score"])
        .reshape(-1)[0]
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )


    fig.suptitle(
        f"{name.upper()} | "
        f"GT={'ANOMALY' if gt_label else 'GOOD'} | "
        f"PRED={'ANOMALY' if pred_label else 'GOOD'} | "
        f"Score={pred_score:.4f}",
        fontsize=14,
    )


    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0].imshow(image)

    axes[0].set_title("Original")

    axes[0].axis("off")


    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    axes[1].imshow(image)

    if gt_mask is not None:
        axes[1].imshow(
            gt_mask,
            cmap="Reds",
            alpha=0.55,
        )

    axes[1].set_title("Ground Truth")

    axes[1].axis("off")


    # --------------------------------------------------------
    # ANOMALY MAP
    # --------------------------------------------------------

    axes[2].imshow(image)

    if anomaly_map is not None:
        axes[2].imshow(
            anomaly_map,
            cmap="jet",
            alpha=0.50,
            vmin=0,
            vmax=1,
        )

    axes[2].set_title(
        "PatchCore Anomaly Map"
    )

    axes[2].axis("off")


    # --------------------------------------------------------
    # PREDICTED MASK
    # --------------------------------------------------------

    axes[3].imshow(image)

    if pred_mask is not None:
        axes[3].imshow(
            pred_mask,
            cmap="Reds",
            alpha=0.55,
        )

    axes[3].set_title(
        "Predicted Mask"
    )

    axes[3].axis("off")


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR / f"{name}.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

for name in [
    "bent",
    "color",
    "flip",
    "scratch",
    "false_positive_good",
]:

    if name in selected:

        visualize_prediction(
            name,
            selected[name],
        )

    else:

        print(
            f"WARNING: {name} not found"
        )


print("\nVisualization complete.")

print(
    f"Saved to:\n{OUTPUT_DIR}"
)