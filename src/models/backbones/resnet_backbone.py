# src/ui/app.py
import os
import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image

from config import Config
from models.unet import UNetResNet18
from data.transforms import get_infer_transforms


# -----------------------
# Load model (cache)
# -----------------------
@st.cache_resource
def load_model(config_path: str = "config.yaml", checkpoint: str = None):
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        decoder_mode=cfg["model"].get("decoder_mode", "add")
    ).to(device)

    if checkpoint is None:
        checkpoint = os.path.join(cfg["logging"]["output_dir"], "model_final.pth")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model, cfg, device


# -----------------------
# Inference function
# -----------------------
def predict(model, cfg, device, pil_img, threshold: float = 0.5):
    # Albumentations transforms
    tfms = get_infer_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    np_img = np.array(pil_img.convert("RGB"))
    out = tfms(image=np_img, mask=np.zeros((cfg["data"]["image_height"], cfg["data"]["image_width"], cfg["data"]["num_classes"])))
    tensor_img = out["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)[0]  # (C, H, W)
        masks = (probs > threshold).float().cpu().numpy()
    return np_img, masks


# -----------------------
# Overlay visualization
# -----------------------
def overlay_masks(image_np, masks, alpha=0.4):
    """
    Args:
        image_np: (H, W, 3) RGB
        masks: (C, H, W) binary
    """
    overlay = image_np.copy()
    colors = [
        (255, 0, 0),    # Class 1 → Red
        (0, 255, 0),    # Class 2 → Green
        (0, 0, 255),    # Class 3 → Blue
        (255, 255, 0)   # Class 4 → Yellow
    ]
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        colored_mask = np.zeros_like(image_np)
        for c in range(3):
            colored_mask[..., c] = mask * colors[i][c]
        overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
    return overlay


# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="Steel Defect Segmentation", layout="wide")
    st.title("🔍 Steel Defect Segmentation Demo")
    st.write("Upload an image and the trained model will predict segmentation masks.")

    # Load model
    model, cfg, device = load_model()

    # Sidebar
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05)
    alpha = st.sidebar.slider("Mask Transparency", 0.1, 1.0, 0.4, 0.05)

    # Upload image
    uploaded_file = st.file_uploader("Upload a steel surface image (.jpg/.png)", type=["jpg", "png"])
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_column_width=True)

        # Predict
        np_img, masks = predict(model, cfg, device, pil_img, threshold=threshold)

        # Show overlay
        overlay = overlay_masks(np_img, masks, alpha=alpha)
        st.image(overlay, caption="Prediction Overlay", use_column_width=True)

        # Per-class masks
        st.subheader("Per-Class Predictions")
        n_classes = masks.shape[0]
        cols = st.columns(n_classes)
        for i in range(n_classes):
            with cols[i]:
                st.image(masks[i] * 255, caption=f"Class {i+1}", clamp=True)


if __name__ == "__main__":
    main()
