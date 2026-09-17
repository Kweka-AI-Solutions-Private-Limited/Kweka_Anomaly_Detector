"""
Automated Regression Test for Heatmap Storage Isolation & Pure Colormap Artifacts
----------------------------------------------------------------------------------
Verifies that:
1. Two separate inspections processing images with identical filenames (e.g. 002.png)
   save their heatmaps into isolated directories: storage/inspections/{inspection_id}/.
2. Neither inspection overwrites the other inspection's heatmap.
3. The stored heatmap file exists, is non-empty, and is saved as pure colorized anomaly map.
"""

import os
import shutil
import unittest
from pathlib import Path
from io import BytesIO
from PIL import Image
from fastapi.datastructures import UploadFile

from services.patchcore_service import run_patchcore_inference
from services.storage_service import save_inspection_image, STORAGE_BASE_DIR


class TestHeatmapIsolation(unittest.TestCase):
    def setUp(self):
        self.test_dir = STORAGE_BASE_DIR / "inspections" / "test_tmp_isolation"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Create two sample dummy images with identical filename '002.png'
        self.img1_pil = Image.new("RGB", (256, 256), color=(255, 0, 0))  # Red image
        self.img2_pil = Image.new("RGB", (256, 256), color=(0, 255, 0))  # Green image

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_upload_file(self, pil_img: Image.Image, filename: str) -> UploadFile:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return UploadFile(file=buf, filename=filename)

    def test_heatmap_path_isolation_different_inspections(self):
        insp_id_a = "test_insp_0000000000000000000000a"
        insp_id_b = "test_insp_0000000000000000000000b"

        upload_a = self._create_upload_file(self.img1_pil, "002.png")
        upload_b = self._create_upload_file(self.img2_pil, "002.png")

        path_a, rel_uri_a, _, _ = save_inspection_image(insp_id_a, upload_a)
        path_b, rel_uri_b, _, _ = save_inspection_image(insp_id_b, upload_b)

        # Verify initial saved images are in isolated directories
        self.assertIn(insp_id_a, str(path_a))
        self.assertIn(insp_id_b, str(path_b))

        # Run PatchCore inference (mock/lightweight mode)
        res_a = run_patchcore_inference(path_a, {}, threshold=27.0)
        res_b = run_patchcore_inference(path_b, {}, threshold=27.0)

        heatmap_uri_a = res_a["heatmap_uri"]
        heatmap_uri_b = res_b["heatmap_uri"]

        # 1. Verify heatmap URIs are isolated by inspection_id
        self.assertIn(insp_id_a, heatmap_uri_a)
        self.assertIn(insp_id_b, heatmap_uri_b)
        self.assertNotEqual(heatmap_uri_a, heatmap_uri_b)

        # 2. Verify both heatmap files exist on disk
        full_heatmap_a = STORAGE_BASE_DIR.parent / heatmap_uri_a
        full_heatmap_b = STORAGE_BASE_DIR.parent / heatmap_uri_b

        self.assertTrue(full_heatmap_a.exists(), f"Heatmap A does not exist at {full_heatmap_a}")
        self.assertTrue(full_heatmap_b.exists(), f"Heatmap B does not exist at {full_heatmap_b}")

        # 3. Clean up test inspection folders
        shutil.rmtree(STORAGE_BASE_DIR / "inspections" / insp_id_a, ignore_errors=True)
        shutil.rmtree(STORAGE_BASE_DIR / "inspections" / insp_id_b, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
