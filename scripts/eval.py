# scripts/eval.py
import os
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.dataset import SteelDefectDataset
from src.data.transforms import get_val_transforms
from src.models.unet import UNet
from src.training.losses import get_loss_from_config
from src.training.metrics import build_metrics


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config dosyası boş: {path}")
    return config


def save_overlay_example(img, mask_true, mask_pred, save_path, threshold=0.5):
    """Input, GT, Prediction overlay kaydeder."""
    img_np = img.permute(1, 2, 0).cpu().numpy()
    mask_true_np = mask_true.cpu().numpy()
    mask_pred_np = (torch.sigmoid(mask_pred).cpu().numpy() > threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np.astype("uint8"))
    axes[0].set_title("Input")
    axes[1].imshow(np.max(mask_true_np, axis=0), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(np.max(mask_pred_np, axis=0), cmap="gray")
    axes[2].set_title("Prediction")
    for ax in axes: 
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # ------------------------
    # Config & Device
    # ------------------------
    config = load_config("config.yaml")
    device_str = config["training"].get("device", "cuda")
    device = torch.device("cuda" if (torch.cuda.is_available() and device_str == "cuda") else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------
    # Dataset & DataLoader
    # ------------------------
    val_dataset = SteelDefectDataset(
        split_file=config["data"]["val_split"],
        img_dir=config["data"]["img_dir"],
        mask_dir=config["data"]["mask_dir"],
        augmentations=get_val_transforms(config["data"].get("img_size", (256, 1600)))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4)
    )

    # ------------------------
    # Model & Checkpoint
    # ------------------------
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"].get("features", [64, 128, 256, 512]),
        norm=config["model"].get("norm", "batch"),
        dropout=config["model"].get("dropout", 0.0)
    )
    ckpt_path = os.path.join(config["training"]["checkpoint_dir"], "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # ------------------------
    # Loss & Metrics
    # ------------------------
    criterion = get_loss_from_config(config)
    metrics_fn = build_metrics(config)

    # ------------------------
    # Evaluation Loop
    # ------------------------
    total_loss = 0.0
    all_metrics = []

    os.makedirs(os.path.join(config["paths"]["outputs"], "eval_examples"), exist_ok=True)

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device).float()
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # metrics dict
            batch_metrics = metrics_fn["summary"](logits, masks)
            all_metrics.append(batch_metrics)

            # save a few overlay examples
            if idx < 3:  # ilk 3 batch
                save_overlay_example(
                    imgs[0].cpu(), masks[0].cpu(), logits[0].cpu(),
                    save_path=os.path.join(config["paths"]["outputs"], "eval_examples", f"example_{idx}.png"),
                    threshold=config["metrics"]["dice"].get("threshold", 0.5)
                )

    avg_loss = total_loss / len(val_loader)
    # metrikleri ortala
    keys = all_metrics[0].keys()
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}

    # ------------------------
    # Results
    # ------------------------
    results = {"val_loss": avg_loss, **avg_metrics}
    print(f"\nValidation Loss: {avg_loss:.4f}")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    # save json
    save_path = os.path.join(config["paths"]["outputs"], "eval_metrics.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Metrics saved to {save_path}")


if __name__ == "__main__":
    main()
