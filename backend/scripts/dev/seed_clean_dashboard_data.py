"""
================================================================================
⚠️  DEV-ONLY DESTRUCTIVE SEED SCRIPT — DO NOT RUN IN PRODUCTION ⚠️
================================================================================
This script PERMANENTLY DELETES the following MongoDB collections:
  - inspections
  - inspection_results
  - inspection_runs
  - models
  - model_versions
  - feedback
  - reference_images

Then seeds them with sample data.

NEVER:
  - Run this against a production MongoDB instance
  - Include this in Docker startup / entrypoint / health checks
  - Import this module from the application
  - Call this from CI/CD pipelines without an explicit dev-database flag

ONLY run manually in an isolated development environment:
  cd backend/
  python scripts/dev/seed_clean_dashboard_data.py
================================================================================
seed_clean_dashboard_data.py
-----------------------------
Cleans dummy test records from MongoDB and seeds realistic inspection records.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image, ImageDraw
from bson import ObjectId
from db.connection import get_db

def create_sample_image(path: Path, text: str, bg_color: tuple, accent_color: tuple):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)
    # Draw simple grid pattern
    for i in range(0, 256, 32):
        draw.line([(i, 0), (i, 256)], fill=(200, 200, 200), width=1)
        draw.line([(0, i), (256, i)], fill=(200, 200, 200), width=1)
    
    # Draw accent box/defect visual
    draw.rectangle([80, 80, 176, 176], fill=accent_color, outline=(40, 40, 40), width=3)
    draw.text((20, 220), text, fill=(30, 30, 30))
    img.save(path)


def seed_data():
    db = get_db()
    print("Connecting to MongoDB...")

    # 1. Clean existing dummy test collections
    db.inspections.delete_many({})
    db.inspection_results.delete_many({})
    db.inspection_runs.delete_many({})
    db.models.delete_many({})
    db.model_versions.delete_many({})
    db.feedback.delete_many({})
    db.reference_images.delete_many({})
    print("✓ Cleared legacy dummy collections.")

    # 2. Setup storage directory for real images
    storage_dir = Path(__file__).resolve().parent / "storage" / "inspections"
    storage_dir.mkdir(parents=True, exist_ok=True)

    images_meta = [
        ("pcb_good_01.png", "PCB GOOD", (240, 245, 240), (40, 160, 60)),
        ("pcb_bridge_01.png", "PCB BRIDGE", (250, 240, 240), (220, 50, 50)),
        ("pcb_missing_01.png", "PCB MISSING", (250, 245, 235), (220, 140, 40)),
        ("screw_good_01.png", "SCREW GOOD", (240, 240, 245), (100, 140, 200)),
        ("screw_scratch_01.png", "SCREW SCRATCH", (255, 235, 235), (230, 60, 60)),
        ("screw_thread_01.png", "SCREW THREAD", (255, 240, 230), (240, 100, 40)),
        ("tile_good_01.png", "TILE GOOD", (245, 245, 245), (120, 120, 120)),
        ("tile_crack_01.png", "TILE CRACK", (250, 235, 235), (200, 40, 80)),
    ]

    for fname, label, bg, acc in images_meta:
        create_sample_image(storage_dir / fname, label, bg, acc)
    print("✓ Generated valid physical sample image files on disk.")

    # 3. Seed Models & Model Versions
    m1_id = ObjectId()
    m2_id = ObjectId()
    m3_id = ObjectId()

    v1_1_id = ObjectId()
    v1_2_id = ObjectId()
    v2_1_id = ObjectId()
    v3_1_id = ObjectId()

    now = datetime.utcnow()

    db.models.insert_many([
        {
            "_id": m1_id,
            "name": "PCB Assembly Inspector",
            "description": "Visual surface inspection model for SMT PCB components and solder bridges",
            "status": "active",
            "model_type": "patchcore",
            "domain": "Electronics",
            "reference_image_count": 50,
            "active_version_id": v1_2_id,
            "created_at": now - timedelta(days=30),
            "updated_at": now - timedelta(days=2)
        },
        {
            "_id": m2_id,
            "name": "Screw Quality Detector",
            "description": "High-resolution fastener anomaly detector for thread deformation and head scratches",
            "status": "active",
            "model_type": "patchcore",
            "domain": "Automotive",
            "reference_image_count": 40,
            "active_version_id": v2_1_id,
            "created_at": now - timedelta(days=20),
            "updated_at": now - timedelta(days=5)
        },
        {
            "_id": m3_id,
            "name": "Tile Surface Inspector",
            "description": "Ceramic and stone tile crack and stain visual inspection detector",
            "status": "active",
            "model_type": "patchcore",
            "domain": "Manufacturing",
            "reference_image_count": 30,
            "active_version_id": v3_1_id,
            "created_at": now - timedelta(days=10),
            "updated_at": now - timedelta(days=1)
        }
    ])

    db.model_versions.insert_many([
        {
            "_id": v1_1_id,
            "model_id": m1_id,
            "version_number": 1,
            "status": "ready",
            "calibration": {"threshold": 27.5},
            "training": {"reference_count": 25, "build_time_ms": 14200},
            "artifacts": {"checkpoint_uri": "storage/models/pcb_v1/patchcore.ckpt"},
            "created_at": now - timedelta(days=30)
        },
        {
            "_id": v1_2_id,
            "model_id": m1_id,
            "version_number": 2,
            "status": "ready",
            "calibration": {"threshold": 26.8},
            "training": {"reference_count": 50, "build_time_ms": 18500},
            "artifacts": {"checkpoint_uri": "storage/models/pcb_v2/patchcore.ckpt"},
            "created_at": now - timedelta(days=2)
        },
        {
            "_id": v2_1_id,
            "model_id": m2_id,
            "version_number": 1,
            "status": "ready",
            "calibration": {"threshold": 28.2},
            "training": {"reference_count": 40, "build_time_ms": 15800},
            "artifacts": {"checkpoint_uri": "storage/models/screw_v1/patchcore.ckpt"},
            "created_at": now - timedelta(days=20)
        },
        {
            "_id": v3_1_id,
            "model_id": m3_id,
            "version_number": 1,
            "status": "ready",
            "calibration": {"threshold": 25.0},
            "training": {"reference_count": 30, "build_time_ms": 12100},
            "artifacts": {"checkpoint_uri": "storage/models/tile_v1/patchcore.ckpt"},
            "created_at": now - timedelta(days=10)
        }
    ])
    print("✓ Seeded active models and versions.")

    # 4. Seed Inspection Runs
    r1_id = ObjectId()
    r2_id = ObjectId()
    r3_id = ObjectId()

    db.inspection_runs.insert_many([
        {
            "_id": r1_id,
            "model_id": m1_id,
            "model_version_id": v1_2_id,
            "run_number": 1,
            "status": "completed",
            "total_images": 12,
            "created_at": now - timedelta(days=2)
        },
        {
            "_id": r2_id,
            "model_id": m2_id,
            "model_version_id": v2_1_id,
            "run_number": 1,
            "status": "completed",
            "total_images": 10,
            "created_at": now - timedelta(days=1)
        },
        {
            "_id": r3_id,
            "model_id": m3_id,
            "model_version_id": v3_1_id,
            "run_number": 1,
            "status": "completed",
            "total_images": 8,
            "created_at": now - timedelta(hours=5)
        }
    ])
    print("✓ Seeded inspection runs.")

    # 5. Seed 30 Realistic Inspections with matching Inspection Results & Feedback
    defects_pool = [
        ("Solder Bridge", "head", "High", "Solder bridge connecting pin 1 and pin 2 on SMT package."),
        ("Missing Component", "body", "Critical", "C14 capacitor missing from PCB pad layout."),
        ("Surface Scratch", "head", "Medium", "Scratch across screw head surface."),
        ("Thread Deformation", "thread", "Critical", "Deformed thread pattern on middle section of fastener."),
        ("Tile Surface Crack", "surface", "High", "Hairline crack extending across top ceramic tile face."),
    ]

    inspections_data = []
    results_data = []
    feedback_data = []

    # Model 1 (PCB): 12 inspections
    for idx in range(1, 13):
        i_id = ObjectId()
        dt = now - timedelta(days=2) + timedelta(minutes=idx * 25)
        is_reject = idx in [3, 7, 10]
        fname = "pcb_bridge_01.png" if idx == 3 else ("pcb_missing_01.png" if idx in [7, 10] else "pcb_good_01.png")
        
        d_type, loc, sev, exp = defects_pool[0] if idx == 3 else (defects_pool[1] if idx in [7, 10] else ("Not required", "N/A", "Low", "No anomaly detected."))

        score = round(32.4 + idx * 1.5, 2) if is_reject else round(14.2 + idx * 0.4, 2)
        threshold = 26.8

        insp_doc = {
            "_id": i_id,
            "model_id": m1_id,
            "model_version_id": v1_2_id,
            "run_id": r1_id,
            "run_number": 1,
            "status": "completed",
            "prediction": {
                "status": "anomalous" if is_reject else "normal",
                "anomaly_score": score,
                "threshold": threshold,
                "severity": sev
            },
            "vlm_analysis": {
                "status": "completed" if is_reject else "skipped",
                "defect_type": d_type,
                "location": loc,
                "severity": sev,
                "explanation": exp
            },
            "input": {
                "filename": fname,
                "storage_uri": f"storage/inspections/{fname}"
            },
            "filename": fname,
            "storage_uri": f"storage/inspections/{fname}",
            "created_at": dt
        }
        inspections_data.append(insp_doc)

        res_doc = {
            "_id": ObjectId(),
            "inspection_id": i_id,
            "prediction": insp_doc["prediction"],
            "vlm_analysis": insp_doc["vlm_analysis"]
        }
        results_data.append(res_doc)

        if idx in [3, 7]:
            fb_doc = {
                "_id": ObjectId(),
                "inspection_id": i_id,
                "detection_feedback": "correct" if idx == 3 else "false_positive",
                "vlm_feedback_categories": [] if idx == 3 else ["wrong_defect_type"],
                "comment": "Verified by quality auditor."
            }
            feedback_data.append(fb_doc)

    # Model 2 (Screw): 10 inspections
    for idx in range(1, 11):
        i_id = ObjectId()
        dt = now - timedelta(days=1) + timedelta(minutes=idx * 30)
        is_reject = idx in [2, 6, 9]
        fname = "screw_scratch_01.png" if idx in [2, 6] else ("screw_thread_01.png" if idx == 9 else "screw_good_01.png")

        d_type, loc, sev, exp = defects_pool[2] if idx in [2, 6] else (defects_pool[3] if idx == 9 else ("Not required", "N/A", "Low", "No anomaly detected."))

        score = round(35.0 + idx * 1.8, 2) if is_reject else round(12.0 + idx * 0.5, 2)
        threshold = 28.2

        insp_doc = {
            "_id": i_id,
            "model_id": m2_id,
            "model_version_id": v2_1_id,
            "run_id": r2_id,
            "run_number": 1,
            "status": "completed",
            "prediction": {
                "status": "anomalous" if is_reject else "normal",
                "anomaly_score": score,
                "threshold": threshold,
                "severity": sev
            },
            "vlm_analysis": {
                "status": "completed" if is_reject else "skipped",
                "defect_type": d_type,
                "location": loc,
                "severity": sev,
                "explanation": exp
            },
            "input": {
                "filename": fname,
                "storage_uri": f"storage/inspections/{fname}"
            },
            "filename": fname,
            "storage_uri": f"storage/inspections/{fname}",
            "created_at": dt
        }
        inspections_data.append(insp_doc)

        res_doc = {
            "_id": ObjectId(),
            "inspection_id": i_id,
            "prediction": insp_doc["prediction"],
            "vlm_analysis": insp_doc["vlm_analysis"]
        }
        results_data.append(res_doc)

        if idx in [2, 9]:
            fb_doc = {
                "_id": ObjectId(),
                "inspection_id": i_id,
                "detection_feedback": "correct",
                "vlm_feedback_categories": [],
                "comment": "Accurate VLM identification."
            }
            feedback_data.append(fb_doc)

    # Model 3 (Tile): 8 inspections
    for idx in range(1, 9):
        i_id = ObjectId()
        dt = now - timedelta(hours=8) + timedelta(minutes=idx * 20)
        is_reject = idx in [4, 8]
        fname = "tile_crack_01.png" if is_reject else "tile_good_01.png"

        d_type, loc, sev, exp = defects_pool[4] if is_reject else ("Not required", "N/A", "Low", "No anomaly detected.")

        score = round(31.2 + idx * 2.0, 2) if is_reject else round(11.5 + idx * 0.3, 2)
        threshold = 25.0

        insp_doc = {
            "_id": i_id,
            "model_id": m3_id,
            "model_version_id": v3_1_id,
            "run_id": r3_id,
            "run_number": 1,
            "status": "completed",
            "prediction": {
                "status": "anomalous" if is_reject else "normal",
                "anomaly_score": score,
                "threshold": threshold,
                "severity": sev
            },
            "vlm_analysis": {
                "status": "completed" if is_reject else "skipped",
                "defect_type": d_type,
                "location": loc,
                "severity": sev,
                "explanation": exp
            },
            "input": {
                "filename": fname,
                "storage_uri": f"storage/inspections/{fname}"
            },
            "filename": fname,
            "storage_uri": f"storage/inspections/{fname}",
            "created_at": dt
        }
        inspections_data.append(insp_doc)

        res_doc = {
            "_id": ObjectId(),
            "inspection_id": i_id,
            "prediction": insp_doc["prediction"],
            "vlm_analysis": insp_doc["vlm_analysis"]
        }
        results_data.append(res_doc)

    db.inspections.insert_many(inspections_data)
    db.inspection_results.insert_many(results_data)
    db.feedback.insert_many(feedback_data)

    print(f"✓ Successfully seeded {len(inspections_data)} clean, 100% valid inspection records.")
    print("✓ Seeding complete!")

if __name__ == "__main__":
    seed_data()
