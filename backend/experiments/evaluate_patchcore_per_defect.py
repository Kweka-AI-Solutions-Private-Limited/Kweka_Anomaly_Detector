from pathlib import Path

import numpy as np
import torch

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
    / "patchcore_screw_wideresnet50_l2_005.ckpt"
)


# ============================================================
# DATA
# ============================================================

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="screw",
    num_workers=0,
)

datamodule.setup()


# ============================================================
# EXACT DEV MODEL
# ============================================================

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2"],
    pre_trained=True,
    coreset_sampling_ratio=0.05,
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
# PREDICT
# ============================================================

print("Running predictions...")

prediction_batches = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

if prediction_batches is None:
    raise RuntimeError("No predictions returned.")


# ============================================================
# HELPERS
# ============================================================

def to_numpy(value):

    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def scalar(value, index):

    array = to_numpy(value)

    if array is None:
        return None

    return np.asarray(array)[index]


def calculate_iou(gt_mask, pred_mask):

    if gt_mask is None or pred_mask is None:
        return None

    gt = np.asarray(gt_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:

        # Both empty = perfect agreement
        if intersection == 0:
            return 1.0

        return 0.0

    return float(intersection / union)


# ============================================================
# COLLECT INDIVIDUAL SAMPLES
# ============================================================

samples = []

for batch in prediction_batches:

    image_paths = getattr(batch, "image_path", None)

    if image_paths is None:
        continue

    batch_size = len(image_paths)

    for i in range(batch_size):

        samples.append(
            {
                "image_path": str(image_paths[i]),

                "gt_label": scalar(
                    getattr(batch, "gt_label", None),
                    i,
                ),

                "pred_label": scalar(
                    getattr(batch, "pred_label", None),
                    i,
                ),

                "gt_mask": scalar(
                    getattr(batch, "gt_mask", None),
                    i,
                ),

                "pred_mask": scalar(
                    getattr(batch, "pred_mask", None),
                    i,
                ),

                "pred_score": scalar(
                    getattr(batch, "pred_score", None),
                    i,
                ),
            }
        )


print(f"Total images: {len(samples)}")


# ============================================================
# METRICS STORAGE
# ============================================================

defect_types = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]

results = {}

for defect in defect_types:

    results[defect] = {
        "total": 0,
        "tp": 0,
        "fn": 0,
        "ious": [],
    }


good_total = 0
good_fp = 0


# ============================================================
# EVALUATE
# ============================================================

for sample in samples:

    path = Path(sample["image_path"])

    defect = path.parent.name

    gt_label = bool(
        np.asarray(sample["gt_label"])
        .reshape(-1)[0]
    )

    pred_label = bool(
        np.asarray(sample["pred_label"])
        .reshape(-1)[0]
    )


    # --------------------------------------------------------
    # DEFECT IMAGE
    # --------------------------------------------------------

    if defect in defect_types:

        results[defect]["total"] += 1

        if pred_label:
            results[defect]["tp"] += 1
        else:
            results[defect]["fn"] += 1


        # Localization IoU
        iou = calculate_iou(
            sample["gt_mask"],
            sample["pred_mask"],
        )

        if iou is not None:
            results[defect]["ious"].append(iou)


    # --------------------------------------------------------
    # GOOD IMAGE
    # --------------------------------------------------------

    elif defect == "good":

        good_total += 1

        if pred_label:
            good_fp += 1


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 72)
print("PATCHCORE PER-DEFECT EVALUATION")
print("=" * 72)

print()

print(
    f"{'Defect':<12}"
    f"{'Images':>10}"
    f"{'TP':>8}"
    f"{'FN':>8}"
    f"{'Recall':>12}"
    f"{'Mean IoU':>14}"
)

print("-" * 72)


all_ious = []

total_tp = 0
total_fn = 0

for defect in defect_types:

    r = results[defect]

    total = r["total"]
    tp = r["tp"]
    fn = r["fn"]

    recall = (
        tp / total
        if total > 0
        else 0.0
    )

    mean_iou = (
        float(np.mean(r["ious"]))
        if r["ious"]
        else 0.0
    )

    all_ious.extend(r["ious"])

    total_tp += tp
    total_fn += fn

    print(
        f"{defect:<12}"
        f"{total:>10}"
        f"{tp:>8}"
        f"{fn:>8}"
        f"{recall * 100:>11.2f}%"
        f"{mean_iou * 100:>13.2f}%"
    )


print("-" * 72)


overall_recall = (
    total_tp / (total_tp + total_fn)
    if (total_tp + total_fn) > 0
    else 0.0
)

overall_iou = (
    float(np.mean(all_ious))
    if all_ious
    else 0.0
)

fpr = (
    good_fp / good_total
    if good_total > 0
    else 0.0
)


print(
    f"{'OVERALL':<12}"
    f"{total_tp + total_fn:>10}"
    f"{total_tp:>8}"
    f"{total_fn:>8}"
    f"{overall_recall * 100:>11.2f}%"
    f"{overall_iou * 100:>13.2f}%"
)

print()
print(f"GOOD images       : {good_total}")
print(f"GOOD false positives: {good_fp}")
print(f"False Positive Rate : {fpr * 100:.2f}%")

print()
print("=" * 72)
