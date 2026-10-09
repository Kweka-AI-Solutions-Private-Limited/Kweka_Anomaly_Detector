import sys
import math
from pathlib import Path
import io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_single_image_inspection
from fastapi import UploadFile

db = get_db()
m_id = "6aa3aa1d4067031f17c51123"

def make_upload(path: Path) -> UploadFile:
    with open(path, "rb") as f:
        content = f.read()
    return UploadFile(filename=path.name, file=io.BytesIO(content))

good_path = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")
defective_path = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head/000.png")

thresholds = [10.0, 20.0, 24.54, 40.0, 60.0, 100.0]

print("="*80)
print("PIPELINE A THRESHOLD INDEPENDENCE REGRESSION MATRIX")
print("Model ID: 6aa3aa1d4067031f17c51123")
print("="*80)
print(f"{'Image':<15} | {'Threshold':<10} | {'Raw Score':<12} | {'Classification':<15}")
print("-" * 80)

results = {}

for label, img_path in [("GOOD", good_path), ("DEFECTIVE", defective_path)]:
    results[label] = []
    for thr in thresholds:
        uf = make_upload(img_path)
        res = run_single_image_inspection(db, model_id=m_id, upload_file=uf, threshold_override=thr)
        pred = res.get("prediction", {})
        score = float(pred.get("anomaly_score", 0.0))
        status = str(pred.get("status", "unknown")).upper()
        results[label].append((thr, score, status))
        print(f"{label:<15} | {thr:<10.2f} | {score:<12.2f} | {status:<15}")
    print("-" * 80)

# Invariance Check
print("\nVERIFYING RAW SCORE INVARIANCE:")
all_passed = True
for label in ["GOOD", "DEFECTIVE"]:
    scores = [r[1] for r in results[label]]
    first_score = scores[0]
    is_invariant = all(math.isclose(s, first_score, abs_tol=1e-3) for s in scores)
    if not is_invariant:
        all_passed = False
    print(f"{label:<10} Image Raw Scores across all thresholds: {scores} -> INVARIANT: {is_invariant}")

print("="*80)
if not all_passed:
    print("FAILED: Raw score changed when threshold changed!")
    sys.exit(1)
else:
    print("PASSED: Raw score is 100% threshold-independent!")
