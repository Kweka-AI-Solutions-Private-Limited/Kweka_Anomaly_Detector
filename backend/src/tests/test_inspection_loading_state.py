import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from bson import ObjectId
from datetime import datetime, timezone
from services.inspection_run_service import get_inspection_run

def test_inspection_item_loading_state_structure():
    """
    Verifies that an inspection run document distinguishes completed items
    from processing items, ensuring processing items contain no stale prediction results.
    """
    run_oid = ObjectId()
    insp_oid1 = ObjectId()
    insp_oid2 = ObjectId()
    res_oid1 = ObjectId()

    mock_run_doc = {
        "_id": run_oid,
        "model_id": "model_carpet_v1",
        "status": "running",
        "total_images": 2,
        "completed_images": 1,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    
    mock_inspections = [
        {
            "_id": insp_oid1,
            "run_id": run_oid,
            "filename": "cut_001.png",
            "status": "completed",
            "result_id": res_oid1
        },
        {
            "_id": insp_oid2,
            "run_id": run_oid,
            "filename": "cut_002.png",
            "status": "processing"
        }
    ]
    
    mock_results_dict = {
        insp_oid1: {
            "_id": res_oid1,
            "inspection_id": insp_oid1,
            "prediction": {
                "status": "anomalous",
                "anomaly_score": 38.35,
                "threshold": 21.43
            }
        }
    }

    class MockCursor:
        def __init__(self, data):
            self.data = data
        def sort(self, key, direction):
            return self
        def __iter__(self):
            return iter(self.data)

    class MockDB:
        class inspection_runs:
            @staticmethod
            def find_one(query):
                return mock_run_doc
        class inspections:
            @staticmethod
            def find(query):
                return MockCursor(mock_inspections)
        class inspection_results:
            @staticmethod
            def find_one(query):
                return mock_results_dict.get(query.get("inspection_id"))

    formatted = get_inspection_run(MockDB(), str(run_oid))
    
    assert formatted["status"] == "running"
    assert len(formatted["inspections"]) == 2
    
    # Image A: Completed
    item_a = formatted["inspections"][0]
    assert item_a["filename"] == "cut_001.png"
    assert item_a["status"] == "completed"
    assert item_a["prediction"]["status"] == "anomalous"
    assert item_a["prediction"]["anomaly_score"] == 38.35
    
    # Image B: Processing
    item_b = formatted["inspections"][1]
    assert item_b["filename"] == "cut_002.png"
    assert item_b["status"] == "processing"
    assert "prediction" not in item_b


