"""
InspectAI Real PatchCore ML Integration Test Suite
--------------------------------------------------
End-to-end integration test verifying real PatchCore ML execution:
  1. Creates real 256x256 RGB image fixtures.
  2. Executes real PatchCore model build & 80/20 LOO out-of-sample GOOD calibration.
  3. Verifies physical creation of PyTorch checkpoint & metadata JSON artifacts.
  4. Executes real inference on GOOD and DEFECTIVE test samples.
  5. Verifies physical creation of PNG heatmap visualization files on disk.
  6. Verifies bounding box reticle extraction for anomalous samples.
  7. Verifies model version immutability & independent reload of v1 and v2 artifacts.
"""

import os
import sys
import shutil
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from services.patchcore_service import (
    build_patchcore_version, run_patchcore_inference, PATCHCORE_CONFIG
)

TEST_DIR = Path(__file__).resolve().parent / "tmp_ml_fixtures"
REFERENCES_DIR = TEST_DIR / "references"
ARTIFACTS_DIR = TEST_DIR / "artifacts"
INSPECTIONS_DIR = TEST_DIR / "inspections"


class TestPatchCoreRealMLIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        INSPECTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Generate 4 GOOD reference images (256x256 RGB with smooth texture)
        cls.ref_paths = []
        for i in range(1, 5):
            arr = np.ones((256, 256, 3), dtype=np.uint8) * 180
            arr += np.random.randint(-5, 5, (256, 256, 3), dtype=np.int16).astype(np.uint8)
            img_path = REFERENCES_DIR / f"good_ref_{i:02d}.png"
            Image.fromarray(arr).save(img_path)
            cls.ref_paths.append(img_path)

        # 2. Generate 1 GOOD test image
        good_arr = np.ones((256, 256, 3), dtype=np.uint8) * 180
        cls.test_good_path = INSPECTIONS_DIR / "test_good_sample.png"
        Image.fromarray(good_arr).save(cls.test_good_path)

        # 3. Generate 1 ANOMALOUS test image (with bright defect patch)
        defect_arr = np.ones((256, 256, 3), dtype=np.uint8) * 180
        defect_arr[90:150, 90:150, :] = 255  # Distinct bright defect patch
        cls.test_defect_path = INSPECTIONS_DIR / "test_defect_sample.png"
        Image.fromarray(defect_arr).save(cls.test_defect_path)

    @classmethod
    def tearDownClass(cls):
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR, ignore_errors=True)

    def test_01_real_patchcore_build_and_calibration(self):
        v1_dir = ARTIFACTS_DIR / "v1"
        threshold, build_time_ms, artifacts = build_patchcore_version(
            reference_image_paths=self.ref_paths,
            artifacts_dir=v1_dir,
            seed=42
        )

        # Assertions
        self.assertGreater(threshold, 0)
        self.assertGreater(build_time_ms, 0)
        self.assertIn("checkpoint_uri", artifacts)
        self.assertIn("metadata_uri", artifacts)

        # Verify physical file creation on disk
        ckpt_path = v1_dir / "patchcore_memory_bank.ckpt"
        meta_path = v1_dir / "version_metadata.json"
        self.assertTrue(ckpt_path.exists(), f"Checkpoint file missing: {ckpt_path}")
        self.assertTrue(meta_path.exists(), f"Metadata file missing: {meta_path}")

        self.__class__.v1_threshold = threshold
        self.__class__.v1_artifacts = artifacts

    def test_02_real_inference_on_good_sample(self):
        threshold = self.__class__.v1_threshold
        artifacts = self.__class__.v1_artifacts

        result = run_patchcore_inference(
            test_image_path=self.test_good_path,
            artifacts=artifacts,
            threshold=threshold
        )

        expected_status = "anomalous" if result["anomaly_score"] >= threshold else "normal"
        self.assertEqual(result["status"], expected_status)
        self.assertIn("heatmap_uri", result)

        # Verify physical heatmap PNG file creation on disk
        root_base = Path(__file__).resolve().parent.parent.parent
        heatmap_path = root_base / result["heatmap_uri"]
        self.assertTrue(heatmap_path.exists() or (INSPECTIONS_DIR / "heatmaps" / "test_good_sample_heatmap.png").exists(), "Heatmap PNG file missing on disk!")

        self.__class__.good_score = result["anomaly_score"]

    def test_03_real_inference_on_defective_sample(self):
        threshold = self.__class__.v1_threshold
        artifacts = self.__class__.v1_artifacts

        result = run_patchcore_inference(
            test_image_path=self.test_defect_path,
            artifacts=artifacts,
            threshold=threshold
        )

        # Defect sample should have higher anomaly score than good sample
        self.assertGreater(result["anomaly_score"], self.__class__.good_score)
        self.assertIsNotNone(result["bbox"])
        self.assertIn("x", result["bbox"])
        self.assertIn("y", result["bbox"])

    def test_04_version_immutability_and_reload(self):
        v2_dir = ARTIFACTS_DIR / "v2"
        v2_threshold, v2_build_time, v2_artifacts = build_patchcore_version(
            reference_image_paths=self.ref_paths[:3],
            artifacts_dir=v2_dir,
            seed=99
        )

        # Verify v1 and v2 artifacts exist independently
        self.assertTrue((ARTIFACTS_DIR / "v1" / "patchcore_memory_bank.ckpt").exists())
        self.assertTrue((ARTIFACTS_DIR / "v2" / "patchcore_memory_bank.ckpt").exists())

        # Verify inference on v1 still works using v1 threshold
        res_v1 = run_patchcore_inference(self.test_good_path, self.__class__.v1_artifacts, self.__class__.v1_threshold)
        self.assertEqual(res_v1["threshold"], self.__class__.v1_threshold)

        # Verify inference on v2 uses v2 threshold
        res_v2 = run_patchcore_inference(self.test_good_path, v2_artifacts, v2_threshold)
        self.assertEqual(res_v2["threshold"], v2_threshold)

    def test_05_disk_reload_after_cache_clear(self):
        from services.patchcore_service import _MODEL_CACHE
        ckpt_uri = self.__class__.v1_artifacts["checkpoint_uri"]

        # 1. Clear in-memory model cache
        _MODEL_CACHE.clear()
        self.assertNotIn(ckpt_uri, _MODEL_CACHE)

        # 2. Run inference on test image - should reload checkpoint from disk into _MODEL_CACHE
        res = run_patchcore_inference(
            test_image_path=self.test_good_path,
            artifacts=self.__class__.v1_artifacts,
            threshold=self.__class__.v1_threshold
        )

        # 3. Verify model was reloaded from disk into cache and inference succeeded
        self.assertIn(ckpt_uri, _MODEL_CACHE)
        self.assertEqual(res["threshold"], self.__class__.v1_threshold)
        self.assertGreater(res["anomaly_score"], 0)
        self.assertIn("heatmap_uri", res)

        # 4. Verify invalid/missing checkpoint URI raises FileNotFoundError instead of silent mock scoring
        invalid_artifacts = {"checkpoint_uri": "storage/models/non_existent_v999/patchcore_memory_bank.ckpt"}
        with self.assertRaises(FileNotFoundError):
            run_patchcore_inference(self.test_good_path, invalid_artifacts, 27.0)


if __name__ == "__main__":
    unittest.main()
