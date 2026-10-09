import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def analyze_heatmap(filepath):
    p = Path(filepath)
    if not p.exists():
        print(f"File not found: {filepath}")
        return
    
    img_bgr = cv2.imread(str(p))
    if img_bgr is None:
        print(f"Could not load image: {filepath}")
        return
    pil_img = Image.open(str(p))
    
    h, w, c = img_bgr.shape
    min_val = int(img_bgr.min())
    max_val = int(img_bgr.max())
    mean_val = float(img_bgr.mean())
    std_val = float(img_bgr.std())
    
    print(f"\n--- Heatmap File Analysis: {p.name} ---")
    print(f"Path: {p}")
    print(f"Dimensions: {w}x{h}")
    print(f"Mode/Channels: {pil_img.mode} ({c} channels)")
    print(f"Min Pixel Value: {min_val}")
    print(f"Max Pixel Value: {max_val}")
    print(f"Mean Pixel Value: {mean_val:.2f}")
    print(f"Std Dev (Spatial Variation): {std_val:.2f}")
    print(f"Contains Spatial Variation: {std_val > 5.0}")
    print(f"Is Colorized (RGB Variance): {not (img_bgr[:,:,0] == img_bgr[:,:,1]).all()}")

if __name__ == "__main__":
    test_files = [
        "storage/temp_uploads/test/rough/heatmaps/000_heatmap.png",
        "data/dtd/000_heatmap.png",
        "outputs/experiments/pipeline_a_v2_v3_validation/visualizations/viz_scratch_neck_000_heatmap.png"
    ]
    for tf in test_files:
        analyze_heatmap(tf)
