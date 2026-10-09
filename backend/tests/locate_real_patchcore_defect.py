import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from pathlib import Path

# Load WideResNet50 backbone (standard PatchCore feature extractor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT).to(device)
backbone.eval()

# Extract layer2 features
features = []
def hook(module, input, output):
    features.append(output)
backbone.layer2.register_forward_hook(hook)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for fname in ["000.png", "001.png", "004.png"]:
    p = Path("c:/dev/Anomaly_Detector/backend/data/dtd") / fname
    if not p.exists():
        continue
    img = Image.open(p).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    features.clear()
    with torch.no_grad():
        backbone(tensor)
    
    feat = features[0][0] # Shape (C, H, W) e.g. (512, 32, 32)
    # Feature norm map across channels
    spatial_map = torch.norm(feat, dim=0).cpu().numpy()
    
    # Resize spatial_map to 256x256
    spatial_map_resized = cv2.resize(spatial_map, (256, 256))
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(spatial_map_resized)
    print(f"Image {fname}: Peak WRN50 layer2 feature location = {max_loc}, Peak value = {max_val:.2f}")
