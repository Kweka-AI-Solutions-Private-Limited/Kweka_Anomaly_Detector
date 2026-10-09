from db.connection import get_db

db = get_db()

print("--- MODELS ---")
for m in db.models.find():
    print(f"ID: {m['_id']} | Name: {m.get('name')} | Status: {m.get('status')} | Active Version ID: {m.get('active_version_id')}")

print("\n--- MODEL VERSIONS ---")
for v in db.model_versions.find():
    print(f"ID: {v['_id']} | Model ID: {v.get('model_id')} | Version #: {v.get('version_number')} | Status: {v.get('status')}")
    print(f"  Calibration: {v.get('calibration')}")
    print(f"  Artifacts: {v.get('artifacts')}")
