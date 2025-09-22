import torch
from torch.utils.data import DataLoader

from src.data.dataset import SteelDefectDataset
from src.data.transforms import build_transforms
from src.models.unet_resnet import UNet
from src.training.losses import get_loss_from_config

import yaml


def load_config(path="config_resnet.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Config
    cfg = load_config()

    # Dataset (1 batch yeterli)
    dataset = SteelDefectDataset(
        split_file=cfg["data"]["val_split"],
        img_dir=cfg["data"]["img_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        augmentations=build_transforms("val", tuple(cfg["data"]["img_size"])),  # ✅ düzeltildi
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    images, masks = next(iter(loader))
    print(f"✅ Input batch: {images.shape}, masks: {masks.shape}")

    # Model
    model = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        encoder_name=cfg["model"].get("encoder_name", "resnet18"),
        encoder_weights=cfg["model"].get("encoder_weights", "imagenet"),
        norm=cfg["model"].get("norm", "batch"),
        dropout=cfg["model"].get("dropout", 0.0),
        up_mode=cfg["model"].get("up_mode", "transpose"),
    )

    logits = model(images)
    print(f"✅ Model output: {logits.shape}")

    # Loss
    criterion = get_loss_from_config(cfg)
    loss = criterion(logits, masks)
    print(f"✅ Loss computed: {loss.item():.4f}")

    # Backward
    loss.backward()
    print("🎉 Smoke test (model + loss + backward) successful!")


if __name__ == "__main__":
    main()
