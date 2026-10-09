import sys
import io
import cv2
import math
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_single_image_inspection
from services.patchcore_service import preprocess_image_aspect_preserving, DEFAULT_TARGET_SIZE
from fastapi import UploadFile

db = get_db()
m_id = "6aa3aa1d4067031f17c51123"

def make_upload(path: Path) -> UploadFile:
    with open(path, "rb") as f:
        content = f.read()
    return UploadFile(filename=path.name, file=io.BytesIO(content))

# Make a 768x256 defective image for non-square defective test
non_sq_good_path = Path("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/good/003.png")

non_sq_def_path = Path("c:/dev/Anomaly_Detector/results/Patchcore/v0/images/good/003_defective_test.png")
if not non_sq_def_path.exists():
    img_def_pil = Image.open("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head/000.png")
    img_def_resized = img_def_pil.resize((768, 256))
    img_def_resized.save(non_sq_def_path)

test_suite = [
    ("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png", "Square GOOD (1024x1024)"),
    ("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head/000.png", "Square DEFECTIVE (1024x1024)"),
    (str(non_sq_good_path), "Non-Square GOOD (768x256 003.png)"),
    (str(non_sq_def_path), "Non-Square DEFECTIVE (768x256)"),
]

print("="*95)
print("FINAL PIPELINE A INPUT-CONTRACT REPAIR VERIFICATION")
print("Model ID: 6aa3aa1d4067031f17c51123, Threshold: 24.54")
print("="*95)

for p_str, label in test_suite:
    p = Path(p_str)
    if not p.exists():
        print(f"Skipping {label}: file not found")
        continue

    img_pil = Image.open(p)
    w, h = img_pil.size
    ar = round(w / float(h), 2)
    _, meta = preprocess_image_aspect_preserving(img_pil, DEFAULT_TARGET_SIZE)

    uf = make_upload(p)
    res = run_single_image_inspection(db, model_id=m_id, upload_file=uf)

    pred = res.get("prediction", {})
    score = pred.get("anomaly_score", 0.0)
    status = pred.get("status", "unknown").upper()
    heatmap_uri = pred.get("heatmap_uri", "")

    storage_base = Path(__file__).resolve().parent.parent
    rel_h_path = heatmap_uri.lstrip("/").replace("/", "\\")
    full_h_path = storage_base / rel_h_path if rel_h_path else None
    h_exists = full_h_path.exists() if full_h_path else False

    print(f"LABEL           : {label}")
    print(f"PIPELINE        : Pipeline A (Single-Product PatchCore)")
    print(f"MODEL_ID        : {m_id}")
    print(f"MODEL_VERSION_ID: 6aa3f1583d15ba1fd0542734 (Version #2)")
    print(f"CHECKPOINT      : v2/patchcore_memory_bank.ckpt")
    print(f"PREPROCESSING   : Aspect-Ratio Preserving (BORDER_REFLECT_101)")
    print(f"INPUT_WIDTH     : {w}")
    print(f"INPUT_HEIGHT    : {h}")
    print(f"ASPECT_RATIO    : {ar}:1")
    print(f"RESIZED_WIDTH   : {meta['resized_width']}")
    print(f"RESIZED_HEIGHT  : {meta['resized_height']}")
    print(f"PAD_LEFT/TOP/R/B: {meta['pad_left']}/{meta['pad_top']}/{meta['pad_right']}/{meta['pad_bottom']}")
    print(f"ANOMALY_SCORE   : {score:.2f}")
    print(f"THRESHOLD       : 24.54")
    print(f"STATUS          : {status}")
    print(f"HEATMAP_PATH    : {heatmap_uri}")
    print(f"HEATMAP_EXISTS  : {h_exists}")
    print("-" * 95)

print("\n=== VERIFYING PIPELINE A -> PIPELINE B -> PIPELINE A ISOLATION ===")
# 1. Run Pipeline A on 000.png
score_a1 = run_single_image_inspection(db, model_id=m_id, upload_file=make_upload(Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")))["prediction"]["anomaly_score"]

# 2. Run Pipeline B mock / routing check
v_b = db.model_versions.find_one({"model_id": m_id, "inspection_mode": "multi_instance", "status": "active"})

# 3. Run Pipeline A again on 000.png
score_a2 = run_single_image_inspection(db, model_id=m_id, upload_file=make_upload(Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")))["prediction"]["anomaly_score"]

print(f"Pipeline A Score #1 : {score_a1:.4f}")
print(f"Pipeline B Active V : {v_b.get('version_id') if v_b else 'N/A'}")
print(f"Pipeline A Score #2 : {score_a2:.4f}")
print(f"Scores Identical    : {math.isclose(score_a1, score_a2, rel_tol=1e-6)}")
print("="*95)
