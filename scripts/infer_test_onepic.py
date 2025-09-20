# scripts/test_inference.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.models.unet_effnet import UNet

# --- helpers ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denormalize_tensor(img_tensor):
    img = img_tensor.cpu().numpy()
    img = img.transpose(1,2,0)  
    img = (img * IMAGENET_STD) + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img

def overlay_masks_on_image(image_rgb, masks, colors=None, alpha=0.5):
    H, W, _ = image_rgb.shape
    canvas = (image_rgb * 255).astype(np.uint8).copy()
    if colors is None:
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0)]
    overlay = canvas.copy().astype(np.float32)
    for c in range(masks.shape[0]):
        mask = (masks[c] > 0.5).astype(np.uint8)
        if mask.sum() == 0:
            continue
        color = np.array(colors[c]) * 255.0
        for ch in range(3):
            overlay[..., ch] = np.where(mask==1,
                                        (1-alpha)*overlay[..., ch] + alpha*color[ch],
                                        overlay[..., ch])
    return overlay.astype(np.uint8)

def visualize_and_save(image_tensor, pred_masks, out_path):
    img = denormalize_tensor(image_tensor)
    masks_np = pred_masks.cpu().numpy() if torch.is_tensor(pred_masks) else pred_masks
    overlay = overlay_masks_on_image(img, masks_np, alpha=0.45)

    C = masks_np.shape[0]
    plt.figure(figsize=(4*(2+C), 4))
    plt.subplot(1, 2+C, 1)
    plt.imshow(img)
    plt.title("Image (denorm)")
    plt.axis("off")

    plt.subplot(1, 2+C, 2)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    for i in range(C):
        plt.subplot(1, 2+C, 3+i)
        plt.imshow(masks_np[i], cmap="gray")
        plt.title(f"Pred Class {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# --- main ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = UNet(in_channels=3, out_channels=4).to(device)
    ckpt = "outputs/checkpoints/unet_best.pth"
    if not os.path.exists(ckpt):
        ckpt = "outputs/checkpoints/unet_baseline.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError("No checkpoint found at expected locations")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Loaded checkpoint:", ckpt)

    # Tek görsel
    img_path = "data/raw/test_images/ff740a9be.jpg"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(device).float()

    with torch.no_grad():
        logits = model(img)
        probs = torch.sigmoid(logits)[0].cpu()
        preds = (probs > 0.5).float()

    os.makedirs("outputs/inference", exist_ok=True)
    out_path = "outputs/inference/pred_single.png"
    visualize_and_save(img.squeeze(0).cpu(), preds, out_path)
    print("Saved ->", out_path)

if __name__ == "__main__":
    main()
