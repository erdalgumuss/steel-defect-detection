import os
import torch
import matplotlib.pyplot as plt

from src.models.unet_effnet import UNet
from src.data.transforms import get_val_transforms
from src.data.dataset import SteelDefectDataset

def visualize_prediction(image, mask_pred, out_path):
    """
    Orijinal görsel + tahmin edilen maskeleri çizdir ve kaydet
    """
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Image")

    for i in range(mask_pred.shape[0]):
        plt.subplot(1, 5, i+2)
        plt.imshow(mask_pred[i], cmap="gray")
        plt.title(f"Pred Class {i+1}")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Model yükle ---
    model = UNet(in_channels=3, out_channels=4).to(device)
    model.load_state_dict(torch.load("outputs/checkpoints/unet_baseline.pth", map_location=device))
    model.eval()

    # --- Dataset hazırla (val verisinden birkaç örnek alalım) ---
    dataset = SteelDefectDataset(
        split_file="data/processed/splits/val_mini.txt",
        img_dir="data/raw/train_images",
        mask_dir="data/processed/masks_png",
        augmentations=get_val_transforms()
    )

    # --- Birkaç örnek test et ---
    out_dir = "outputs/inference"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx in range(3):  # 3 örnek test edelim
            image, _ = dataset[idx]   # (img, mask) ama maskeyi burada kullanmıyoruz
            input_tensor = image.unsqueeze(0).to(device)

            # Model tahmini
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)[0]       # (C, H, W)
            preds = (probs > 0.5).float().cpu()    # threshold

            # Görselleştir
            out_path = os.path.join(out_dir, f"pred_{idx}.png")
            visualize_prediction(image, preds, out_path)
            print(f"Saved prediction -> {out_path}")

if __name__ == "__main__":
    main()
