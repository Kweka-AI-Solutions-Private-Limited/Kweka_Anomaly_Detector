from bson import ObjectId
from db.connection import get_db

db = get_db()
m = db.models.find_one({"_id": ObjectId("6aa3aa1d4067031f17c51123")})
print(f"MODEL: {m['name']} ({m['_id']}) | Active Version: {m.get('active_version_id')}")

versions = list(db.model_versions.find({"model_id": ObjectId("6aa3aa1d4067031f17c51123")}))
for v in versions:
    print("=" * 60)
    print(f"Version ID: {v['_id']} | Version #: {v.get('version_number')} | Status: {v.get('status')}")
    print(f"Calibration: {v.get('calibration')}")
    print(f"Artifacts: {v.get('artifacts')}")
