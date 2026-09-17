from pathlib import Path
import csv
import time
import statistics

import torch

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = ROOT / "data" / "mvtec_anomaly_detection"

CHECKPOINT = (
    ROOT
    / "outputs"
    / "patchcore_baseline"
    / "patchcore_metal_nut.ckpt"
)

OUTPUT_DIR = ROOT / "outputs" / "dev_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "results.csv"


# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("PATCHCORE DEV_MODE BASELINE EVALUATION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Dataset
# ============================================================

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="metal_nut",
    train_batch_size=32,
    eval_batch_size=1,
    num_workers=0,
)

datamodule.setup()

print(f"Test images: {len(datamodule.test_data)}")


# ============================================================
# DEV_MODE PatchCore
# ============================================================

model = Patchcore(
    backbone="resnet18",
    layers=["layer2"],
    pre_trained=True,
    coreset_sampling_ratio=0.02,
    num_neighbors=9,
)

print("PatchCore DEV_MODE created.")


# ============================================================
# Engine
# ============================================================

engine = Engine(
    accelerator="auto",
    devices=1,
    enable_progress_bar=False,
)

print("Engine created.")
print(f"Checkpoint: {CHECKPOINT}")


# ============================================================
# Prediction
# ============================================================

print()
print("Running inference...")

start_time = time.perf_counter()

predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=str(CHECKPOINT),
)

if torch.cuda.is_available():
    torch.cuda.synchronize()

total_time = time.perf_counter() - start_time

print("Inference complete.")


# ============================================================
# Validate prediction output
# ============================================================

if predictions is None:
    raise RuntimeError(
        "Inference failed: engine.predict() returned None."
    )

print(f"Prediction groups returned: {len(predictions)}")


# ============================================================
# Metrics
# ============================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

ious = []

rows = []

processed_images = 0


# ============================================================
# Process predictions
# ============================================================

for prediction_group in predictions:

    # --------------------------------------------------------
    # Anomalib normally returns ImageBatch objects here.
    # Some versions/types may represent a group as a list.
    # Normalize both cases.
    # --------------------------------------------------------

    if isinstance(prediction_group, list):
        batches = prediction_group
    else:
        batches = [prediction_group]

    for batch in batches:

        processed_images += 1

        # ----------------------------------------------------
        # Ground-truth and prediction labels
        # ----------------------------------------------------

        gt_label = bool(batch.gt_label.item())

        pred_label = bool(batch.pred_label.item())

        anomaly_score = float(batch.pred_score.item())

        image_path = batch.image_path[0]

        defect_type = Path(image_path).parent.name


        # ----------------------------------------------------
        # Classification confusion matrix
        # ----------------------------------------------------

        if gt_label and pred_label:

            true_positive += 1

        elif not gt_label and not pred_label:

            true_negative += 1

        elif not gt_label and pred_label:

            false_positive += 1

        elif gt_label and not pred_label:

            false_negative += 1


        # ----------------------------------------------------
        # Pixel-level IoU
        #
        # Only defective images have meaningful GT masks.
        # ----------------------------------------------------

        iou = None

        if gt_label:

            gt_mask = batch.gt_mask.squeeze().bool()

            pred_mask = batch.pred_mask.squeeze().bool()


            # ----------------------------------------------
            # Intersection
            # ----------------------------------------------

            intersection = torch.logical_and(
                gt_mask,
                pred_mask,
            ).sum().item()


            # ----------------------------------------------
            # Union
            # ----------------------------------------------

            union = torch.logical_or(
                gt_mask,
                pred_mask,
            ).sum().item()


            # ----------------------------------------------
            # IoU
            # ----------------------------------------------

            if union > 0:

                iou = intersection / union

            else:

                iou = 0.0


            ious.append(iou)


        # ----------------------------------------------------
        # Store per-image result
        # ----------------------------------------------------

        rows.append(
            {
                "image_path": image_path,
                "defect_type": defect_type,
                "gt_label": int(gt_label),
                "pred_label": int(pred_label),
                "anomaly_score": anomaly_score,
                "iou": "" if iou is None else iou,
            }
        )


# ============================================================
# Sanity check
# ============================================================

print()
print(f"Processed images: {processed_images}")

if processed_images != 115:

    print(
        "WARNING: Expected 115 test images "
        f"but processed {processed_images}."
    )


# ============================================================
# Calculate metrics
# ============================================================

# ------------------------------------------------------------
# Recall
#
# Recall = TP / (TP + FN)
# ------------------------------------------------------------

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)


# ------------------------------------------------------------
# False Positive Rate
#
# FPR = FP / (FP + TN)
# ------------------------------------------------------------

fpr = (
    false_positive / (false_positive + true_negative)
    if (false_positive + true_negative) > 0
    else 0.0
)


# ------------------------------------------------------------
# IoU
# ------------------------------------------------------------

average_iou = (
    statistics.mean(ious)
    if ious
    else 0.0
)

median_iou = (
    statistics.median(ious)
    if ious
    else 0.0
)


# ------------------------------------------------------------
# Inference time
# ------------------------------------------------------------

if processed_images > 0:

    average_inference_time = (
        total_time / processed_images
    )

else:

    average_inference_time = 0.0


# ============================================================
# Save per-image CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path",
            "defect_type",
            "gt_label",
            "pred_label",
            "anomaly_score",
            "iou",
        ],
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# Print final results
# ============================================================

print()
print("=" * 70)
print("DEV_MODE RESULTS")
print("=" * 70)

print()
print("CONFUSION MATRIX")
print("-" * 70)

print(f"True Positives : {true_positive}")
print(f"False Negatives: {false_negative}")
print(f"True Negatives : {true_negative}")
print(f"False Positives: {false_positive}")


print()
print("IMAGE-LEVEL METRICS")
print("-" * 70)

print(f"Recall         : {recall * 100:.2f}%")
print(f"False Positive : {fpr * 100:.2f}%")


print()
print("PIXEL-LEVEL LOCALIZATION")
print("-" * 70)

print(f"Average IoU    : {average_iou * 100:.2f}%")
print(f"Median IoU     : {median_iou * 100:.2f}%")


print()
print("INFERENCE TIME")
print("-" * 70)

print(f"Total time     : {total_time:.3f} sec")

print(
    f"Average/image  : "
    f"{average_inference_time:.4f} sec"
)

print(
    f"Average/image  : "
    f"{average_inference_time * 1000:.2f} ms"
)


print()
print("OUTPUT")
print("-" * 70)

print(f"CSV saved to:")
print(CSV_PATH)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)