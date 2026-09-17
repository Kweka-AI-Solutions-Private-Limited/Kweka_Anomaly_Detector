import requests
import json
from pathlib import Path

url = "http://127.0.0.1:8000/api/inspections"
img_path = Path("C:/dev/Anomaly_Detector/backend/storage/inspections/6aa3cde8f7dc0e82a7453af1/001.png")

form_data = {
    "model_id": "6aa3aa1d4067031f17c51124",
    "inspection_mode": "multi_instance"
}

with open(img_path, "rb") as f:
    files = {"image": ("001.png", f, "image/png")}
    response = requests.post(url, data=form_data, files=files)

print("HTTP Status:", response.status_code)
if response.status_code in (200, 201):
    data = response.json()
    print("Inspection ID:", data.get("inspection_id"))
    print("Mode:", data.get("inspection_mode"))
    overall = data.get("overall_prediction", {})
    print("\n--- OVERALL PREDICTION ---")
    print(f"Status: {overall.get('status')}")
    print(f"Total Instances: {overall.get('total_instances')}")
    print(f"Pass Count: {overall.get('pass_count')}")
    print(f"Reject Count: {overall.get('reject_count')}")
    print(f"Max Anomaly Score: {overall.get('max_anomaly_score')}")
    print(f"Threshold: {overall.get('threshold')}")
    
    instances = data.get("instances", [])
    print(f"\n--- PER-INSTANCE RESULTS ({len(instances)}) ---")
    for inst in instances:
        iid = inst.get("instance_id")
        crop_uri = inst.get("crop_storage_uri")
        heat_uri = inst.get("localization", {}).get("heatmap_uri")
        pred = inst.get("prediction", {})
        bbox = inst.get("bbox")
        print(f"Instance #{iid}: Status={pred.get('status')}, Score={pred.get('anomaly_score')}, BBox={bbox}")
        print(f"  Crop URI: {crop_uri}")
        print(f"  Heatmap URI: {heat_uri}")
        
        # Test HTTP 200 fetching crop URL
        crop_http = f"http://127.0.0.1:8000/{crop_uri if crop_uri.startswith('storage/') else 'storage/' + crop_uri}"
        c_res = requests.get(crop_http)
        print(f"    -> Crop HTTP GET: {c_res.status_code} ({len(c_res.content)} bytes)")
        
        if heat_uri:
            heat_http = f"http://127.0.0.1:8000/{heat_uri if heat_uri.startswith('storage/') else 'storage/' + heat_uri}"
            h_res = requests.get(heat_http)
            print(f"    -> Heatmap HTTP GET: {h_res.status_code} ({len(h_res.content)} bytes)")
else:
    print("Error:", response.text)
