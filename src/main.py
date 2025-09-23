import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import Config   # <-- bizim class
from data.dataset import SteelDefectDataset, build_full_dataframe
from data.transforms import get_train_transforms, get_valid_transforms
from models.unet import UNetResNet18
from engines.training_engine import train_one_epoch, validate_one_epoch


def main(config_path: str = "config.yaml"):
    # ---- Config ----
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"

    # ---- Dataset ----
    train_csv = pd.read_csv(cfg["data"]["train_csv"])
    image_dir = cfg["data"]["train_images_dir"]

    full_df = build_full_dataframe(train_csv, image_dir)

    train_ds = SteelDefectDataset(
        full_df,
        image_dir,
        shape=(cfg["data"]["image_height"], cfg["data"]["image_width"]),
        num_classes=cfg["data"]["num_classes"],
        transforms=get_train_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    )

    valid_ds = SteelDefectDataset(
        full_df,
        image_dir,
        shape=(cfg["data"]["image_height"], cfg["data"]["image_width"]),
        num_classes=cfg["data"]["num_classes"],
        transforms=get_valid_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,           # GPU transferi için hızlandırma
        prefetch_factor=2,         # CPU önceden batch hazırlar
        persistent_workers=True,    # worker süreçleri tekrar tekrar kapanıp açılmaz
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            [x[2] for x in b]
        )
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,           # GPU transferi için hızlandırma
        prefetch_factor=2,         # CPU önceden batch hazırlar
        persistent_workers=True,    # worker süreçleri tekrar tekrar kapanıp açılmaz
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            [x[2] for x in b]
        )
    )

    # ---- Model ----
    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"]
    ).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # ---- Training Loop ----
    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        valid_metrics = validate_one_epoch(model, valid_loader, device)

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}")
        print(f"Valid Loss: {valid_metrics['loss']:.4f} | Valid Dice: {valid_metrics['dice']:.4f}")

        # Checkpoint kaydetme
        if (epoch + 1) % cfg["logging"]["save_every"] == 0:
            os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
            ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # ---- Save Final Model ----
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    final_path = os.path.join(cfg["logging"]["output_dir"], "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
