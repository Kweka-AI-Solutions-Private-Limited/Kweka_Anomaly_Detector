import sys
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db.connection import get_db
from services.inspection_service import run_single_image_inspection
from fastapi import UploadFile

db = get_db()
m_id = "6aa3aa1d4067031f17c51123"

test_good_dir = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good")
test_good_files = sorted(list(test_good_dir.glob("0*.png")))[:10]  # first 10 good test images (excluding _v3_crop)

def make_upload_file(path: Path) -> UploadFile:
    with open(path, "rb") as f:
        content = f.read()
    return UploadFile(filename=path.name, file=io.BytesIO(content))

print(f"=== TESTING {len(test_good_files)} MVTEC TEST GOOD IMAGES ===")
print("IMAGE | LABEL | SCORE | THRESHOLD | STATUS")
print("-" * 50)
good_scores = []
for p in test_good_files:
    if "_v3_crop" in p.name:
        continue
    uf = make_upload_file(p)
    res = run_single_image_inspection(db, model_id=m_id, upload_file=uf)
    score = res.get("prediction", {}).get("anomaly_score")
    status = res.get("prediction", {}).get("status")
    good_scores.append(score)
    print(f"{p.name} | GOOD | {score:.2f} | 24.54 | {status}")

print(f"\nAverage GOOD Test Image Score: {sum(good_scores)/len(good_scores):.2f}")

# Defective test images
defective_dirs = [
    Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/scratch_head"),
    Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/thread_top"),
    Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/thread_side"),
    Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/manipulated_front")
]

print("\n=== TESTING MVTEC TEST DEFECTIVE IMAGES ===")
print("IMAGE | LABEL | SCORE | THRESHOLD | STATUS")
print("-" * 50)
for d in defective_dirs:
    if not d.exists():
        continue
    files = [f for f in sorted(list(d.glob("0*.png"))) if "_v3_crop" not in f.name and "_heatmap" not in f.name][:5]
    for p in files:
        uf = make_upload_file(p)
        res = run_single_image_inspection(db, model_id=m_id, upload_file=uf)
        score = res.get("prediction", {}).get("anomaly_score")
        status = res.get("prediction", {}).get("status")
        print(f"{d.name}/{p.name} | DEFECTIVE | {score:.2f} | 24.54 | {status}")
