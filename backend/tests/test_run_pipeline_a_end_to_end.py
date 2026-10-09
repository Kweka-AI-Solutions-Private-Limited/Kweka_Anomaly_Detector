import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from api_server import app
from bson import ObjectId
from db.connection import get_db

client = TestClient(app)
db = get_db()

# Model ID: 6aa3aa1d4067031f17c51123 (Screw Quality Detector)
model_id = "6aa3aa1d4067031f17c51123"

# Upload 000_heatmap.png
with open("c:/dev/Anomaly_Detector/backend/data/dtd/000_heatmap.png", "rb") as f:
    res = client.post(
        "/api/inspections",
        data={"model_id": model_id},
        files={"image": ("000_heatmap.png", f.read(), "image/png")}
    )

print("STATUS CODE:", res.status_code)
if res.status_code == 201:
    data = res.json()
    print("INSPECTION ID:", data.get("id"))
    print("PREDICTION:", data.get("prediction"))
    print("LOCALIZATION:", data.get("localization"))
    print("VLM ANALYSIS:", data.get("vlm_analysis"))
else:
    print("ERROR:", res.text)
