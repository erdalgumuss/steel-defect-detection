# src/ui/model_utils.py
import os
import torch
import numpy as np
from PIL import Image
import cv2

from config import Config
from models.unet import UNetResNet18
from data.transforms import get_infer_transforms


def resize_to_target(np_img: np.ndarray, h: int, w: int, keep_aspect: bool = False) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        np_img: (H, W, 3)
        h, w: target size
        keep_aspect: if True, keep aspect ratio and pad; if False, direct resize
    """
    if not keep_aspect:
        return cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)

    # keep aspect ratio with padding
    orig_h, orig_w = np_img.shape[:2]
    scale = min(w / orig_w, h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # pad to target
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def load_model(config_path: str = "config.yaml", checkpoint: str = None):
    """
    Load trained UNetResNet18 model.
    Supports both raw state_dict and dict with 'model' key.
    """
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        decoder_mode=cfg["model"].get("decoder_mode", "add"),
    ).to(device)

    if checkpoint is None:
        checkpoint = os.path.join(cfg["logging"]["output_dir"], "model_final.pth")

    state = torch.load(checkpoint, map_location=device)

    # handle both direct state_dict and wrapped dict
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    model.load_state_dict(state)
    model.eval()
    return model, cfg, device


def predict(model, cfg, device, pil_img: Image.Image, threshold: float = 0.5):
    """
    Run inference on a single PIL image.
    Returns resized np_img and binary masks.
    """
    H, W = cfg["data"]["image_height"], cfg["data"]["image_width"]
    tfms = get_infer_transforms(H, W)

    np_img = np.array(pil_img.convert("RGB"))
    out = tfms(
        image=np_img,
        mask=np.zeros((H, W, cfg["data"]["num_classes"]), dtype=np.uint8)
    )
    tensor_img = out["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)[0]  # (C, H, W)
        masks = (probs > threshold).float().cpu().numpy()

    np_img_resized = resize_to_target(np_img, H, W, keep_aspect=False)
    return np_img_resized, masks
