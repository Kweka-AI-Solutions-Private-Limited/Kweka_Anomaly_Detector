import cv2
import json
from pathlib import Path
from bson import ObjectId
from services.patchcore_service import run_patchcore_inference
from db.connection import get_db

db = get_db()
model = db.models.find_one({"_id": ObjectId("6aa3aa1d4067031f17c51124")})
ver = db.model_versions.find_one({"_id": ObjectId("6aa3cc83f7dc0e82a7453aed")})

artifacts = ver["artifacts"]
thresh = float(ver.get("calibration", {}).get("threshold", 27.34))
print("Model:", model["name"], "Version:", ver["_id"], "Threshold:", thresh)

storage_root = Path("/app/backend/storage") if Path("/app/backend/storage").exists() else Path("C:/dev/Anomaly_Detector/backend/storage")
insp_doc = list(db.inspections.find({"inspection_mode": "multi_instance"}).sort("_id", -1).limit(1))[0]
insp_res = list(db.inspection_results.find({"inspection_mode": "multi_instance"}).sort("_id", -1).limit(1))[0]

for inst in insp_res["instances"]:
    crop_rel = inst["crop_storage_uri"]
    clean_rel = crop_rel.replace("storage/", "") if crop_rel.startswith("storage/") else crop_rel
    crop_path = storage_root / clean_rel
    if crop_path.exists():
        res = run_patchcore_inference(crop_path, artifacts, thresh)
        print(f"PADDED Inst #{inst['instance_id']} -> Score: {res['anomaly_score']:.2f}, Status: {res['status']}")
    else:
        print(f"Inst #{inst['instance_id']} crop path NOT found: {crop_path}")

# Test exact crop for Tile #3 without neighboring tile overlap
# Bbox for Inst #3: x=0, y=590, w=612, h=664
input_rel = insp_doc.get("input", {}).get("storage_uri", "")
clean_input = input_rel.replace("storage/", "") if input_rel.startswith("storage/") else input_rel
img_path = storage_root / clean_input
print("Loading composite image:", img_path)
orig_bgr = cv2.imread(str(img_path))
if orig_bgr is not None:
    # Crop Tile #3 without 5% padding expansion into separator line
    tile3_exact = orig_bgr[590:590+664, 0:612]
    exact_path = storage_root / "test_tile3_exact.png"
    cv2.imwrite(str(exact_path), tile3_exact)
    res_exact = run_patchcore_inference(exact_path, artifacts, thresh)
    print(f"EXACT  Inst #3 -> Score: {res_exact['anomaly_score']:.2f}, Status: {res_exact['status']}")
else:
    print("Could not read original composite image:", img_path)
