# scripts/predict.py
import os
import yaml
import torch
import cv2
import matplotlib.pyplot as plt

from src.data.transforms import get_val_transforms
from src.models.unet import UNet


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config dosyası boş: {path}")
    return config


def visualize_prediction(image, preds, out_path):
    """
    Orijinal görsel + tahmin maskelerini çizdir ve kaydet
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(1, preds.shape[0] + 1, 1)
    plt.imshow(image)
    plt.title("Image")

    for i in range(preds.shape[0]):
        plt.subplot(1, preds.shape[0] + 1, i + 2)
        plt.imshow(preds[i], cmap="gray")
        plt.title(f"Class {i+1}")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Config
    config = load_config("config.yaml")

    # Device
    device = config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # ------------------------
    # Model yükle
    # ------------------------
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"].get("features", [64, 128, 256, 512]),
        norm=config["model"].get("norm", "batch"),
        dropout=config["model"].get("dropout", 0.0),
    )
    ckpt_path = os.path.join(config["training"]["checkpoint_dir"], "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # ------------------------
    # Transforms & threshold
    # ------------------------
    transforms = get_val_transforms(config["data"].get("img_size", (256, 1600)))
    threshold = config["metrics"].get("dice", {}).get("threshold", 0.5)

    # ------------------------
    # Prediction
    # ------------------------
    test_dir = config["data"]["img_dir"]  # test için raw/train_images klasörü
    out_dir = os.path.join(config["paths"]["outputs"], "inference")
    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(test_dir):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(test_dir, fname)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        augmented = transforms(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0]
            preds = (probs > threshold).float().cpu().numpy()

        # Overlay görsel kaydet
        out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_overlay.png")
        visualize_prediction(image, preds, out_path)

        # Maskeleri ayrı kaydet
        for i in range(preds.shape[0]):
            mask_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_c{i+1}.png")
            cv2.imwrite(mask_path, (preds[i] * 255).astype(np.uint8))

        print(f"[INFO] Prediction saved -> {out_path}")


if __name__ == "__main__":
    main()
