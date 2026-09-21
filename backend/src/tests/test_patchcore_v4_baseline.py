import unittest
import torch
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.patchcore_service import (
    GenericFolderDataset,
    load_patchcore_model,
    run_patchcore_inference,
    resolve_checkpoint_path,
    PATCHCORE_CONFIG,
    ANOMALIB_AVAILABLE
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class TestPatchCoreV4Baseline(unittest.TestCase):

    def test_v4_dataset_preprocessing(self):
        """Verify RGB conversion and direct 256x256 tensor scaling [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_img = Path(tmp_dir) / "test_rgb.png"
            Image.new("RGB", (500, 300), color=(100, 150, 200)).save(test_img)

            ds = GenericFolderDataset([test_img], target_size=(256, 256))
            self.assertEqual(len(ds), 1)
            sample = ds[0]
            tensor = sample["image"]
            
            self.assertEqual(tensor.shape, (3, 256, 256))
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(tensor.min() >= 0.0 and tensor.max() <= 1.0)

    def test_v4_patchcore_config(self):
        """Verify PatchCore baseline configuration constants."""
        self.assertEqual(PATCHCORE_CONFIG["backbone"], "wide_resnet50_2")
        self.assertEqual(PATCHCORE_CONFIG["layers"], ["layer2"])
        self.assertEqual(PATCHCORE_CONFIG["coreset_sampling_ratio"], 0.05)
        self.assertEqual(PATCHCORE_CONFIG["num_neighbors"], 9)
        self.assertEqual(PATCHCORE_CONFIG["image_size"], [256, 256])

    def test_v4_model_loading(self):
        """Verify loading V4 checkpoint memory bank artifact."""
        if not ANOMALIB_AVAILABLE:
            self.skipTest("Anomalib not available")

        ckpt_path = resolve_checkpoint_path("artifacts/6aa3aa1d4067031f17c51123/v4/patchcore_memory_bank.ckpt", BASE_DIR)
        self.assertIsNotNone(ckpt_path, "V4 memory bank checkpoint artifact path must resolve")
        self.assertTrue(ckpt_path.exists(), "V4 memory bank checkpoint artifact file must exist")

        model = load_patchcore_model(ckpt_path)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(hasattr(model.model, "memory_bank"))
        self.assertEqual(model.model.memory_bank.shape[1], 512)  # layer2 embedding dimension

    def test_v4_real_score_and_map_generation(self):
        """Verify V4 inference returns real score and spatial anomaly map visualization."""
        if not ANOMALIB_AVAILABLE:
            self.skipTest("Anomalib not available")

        art = {
            "checkpoint_uri": "artifacts/6aa3aa1d4067031f17c51123/v4/patchcore_memory_bank.ckpt"
        }
        sample_img = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")
        self.assertTrue(sample_img.exists())

        res = run_patchcore_inference(sample_img, art)
        self.assertIn("raw_score", res)
        self.assertIn("anomaly_score", res)
        self.assertIsInstance(res["raw_score"], float)
        self.assertGreater(res["raw_score"], 0)
        self.assertIsNone(res["bbox"])  # Localization reticle disabled for V4 baseline
        self.assertTrue(res["heatmap_uri"].startswith("storage/inspections/heatmaps/"))

        # Check generated heatmap file
        hm_path = BASE_DIR / res["heatmap_uri"]
        self.assertTrue(hm_path.exists())
        hm_np = np.array(Image.open(hm_path))
        self.assertEqual(hm_np.shape, (1024, 1024, 3))  # Resampled to original test image resolution

    def test_v4_absence_of_synthetic_fallback(self):
        """Verify system raises explicit error on missing checkpoint instead of fallback fake scores."""
        fake_art = {"checkpoint_uri": "artifacts/non_existent_v4_checkpoint.ckpt"}
        sample_img = Path("c:/dev/Anomaly_Detector/backend/data/mvtec_anomaly_detection/screw/test/good/000.png")

        with self.assertRaises((FileNotFoundError, RuntimeError)):
            run_patchcore_inference(sample_img, fake_art)


if __name__ == "__main__":
    unittest.main()
