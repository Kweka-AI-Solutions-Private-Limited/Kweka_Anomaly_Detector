import cv2
import numpy as np
from pathlib import Path
from bson import ObjectId
from services.patchcore_service import run_patchcore_inference
from db.connection import get_db

db = get_db()
ver = db.model_versions.find_one({"_id": ObjectId("6aa3cc83f7dc0e82a7453aed")})
artifacts = ver["artifacts"]
thresh = float(ver.get("calibration", {}).get("threshold", 27.34))

storage_root = Path("/app/backend/storage") if Path("/app/backend/storage").exists() else Path("C:/dev/Anomaly_Detector/backend/storage")
exact_path = storage_root / "test_tile3_exact.png"

res = run_patchcore_inference(exact_path, artifacts, thresh)
print("Score:", res["anomaly_score"], "Status:", res["status"])
print("BBox in crop:", res.get("bbox"))
print("Heatmap URI:", res.get("heatmap_uri"))
