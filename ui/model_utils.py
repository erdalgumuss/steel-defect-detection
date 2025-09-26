# src/ui/model_utils.py
import os
import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


# Sabit config – eğitimdeki değerlerle aynı olmalı
CFG = {
    "HEIGHT": 256,
    "WIDTH": 1600,
    "NUM_CLASSES": 4,
    "DECODER_MODE": "add",  # veya "concat"
}


# -----------------------------
# Model Tanımı (UNetResNet18)
# -----------------------------
class UNetResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, decoder_mode="add", dropout=0.0):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        del base.fc, base.avgpool

        self.enc1 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.enc2 = nn.Sequential(base.maxpool, base.layer1)
        self.enc3 = base.layer2
        self.enc4 = base.layer3
        self.enc5 = base.layer4

        self.mode = decoder_mode

        def up_block(in_ch, out_ch, use_concat=False):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            if use_concat:
                layers += [nn.Conv2d(out_ch*2, out_ch, 3, padding=1),
                           nn.BatchNorm2d(out_ch),
                           nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        if self.mode == "add":
            self.up4 = up_block(512,256)
            self.up3 = up_block(256,128)
            self.up2 = up_block(128,64)
            self.up1 = up_block(64,64)
        else:
            self.up4 = up_block(512,256, use_concat=True)
            self.up3 = up_block(256,128, use_concat=True)
            self.up2 = up_block(128,64, use_concat=True)
            self.up1 = up_block(64,64, use_concat=True)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        if self.mode == "add":
            d4 = self.up4(e5) + e4
            d3 = self.up3(d4) + e3
            d2 = self.up2(d3) + e2
            d1 = self.up1(d2) + e1
        else:
            d4 = self.up4(torch.cat([F.interpolate(e5, size=e4.shape[2:], mode="bilinear", align_corners=False), e4],1))
            d3 = self.up3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False), e3],1))
            d2 = self.up2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False), e2],1))
            d1 = self.up1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False), e1],1))

        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# -----------------------------
# Yardımcı Fonksiyonlar
# -----------------------------
def resize_to_target(np_img: np.ndarray, h: int, w: int, keep_aspect: bool = False) -> np.ndarray:
    """Resize image to target size."""
    if not keep_aspect:
        return cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)

    orig_h, orig_w = np_img.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def load_model(checkpoint: str = "model/best_model.pth"):
    """Load trained UNetResNet18 model from outputs/ directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNetResNet18(
        num_classes=CFG["NUM_CLASSES"],
        pretrained=False,
        decoder_mode=CFG["DECODER_MODE"],
    ).to(device)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"❌ Model checkpoint bulunamadı: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)

    # hem direkt state_dict hem de {"model": state} destekle
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    model.load_state_dict(state)
    model.eval()
    return model, CFG, device


def predict(model, cfg, device, pil_img: Image.Image, threshold: float = 0.5):
    """Run inference on a single PIL image. Returns resized np_img and binary masks."""
    H, W = cfg["HEIGHT"], cfg["WIDTH"]

    np_img = np.array(pil_img.convert("RGB"))
    np_img_resized = resize_to_target(np_img, H, W, keep_aspect=False)
    tensor_img = torch.from_numpy(np_img_resized).permute(2,0,1).unsqueeze(0).float()/255.0
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)[0]  # (C, H, W)
        masks = (probs > threshold).float().cpu().numpy()

    return np_img_resized, masks
