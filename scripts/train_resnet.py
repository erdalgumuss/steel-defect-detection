import os
import yaml
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import SteelDefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet_resnet import UNet  # ✅ unet_resnet'ten import
from src.training.losses import get_loss_from_config
from src.training.metrics import build_metrics
from src.training.trainer import Trainer


# ------------------------
# Utils
# ------------------------
def load_config(path="config_resnet.yaml"): # ✅ Varsayılan config dosyasını değiştirin
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config dosyası boş: {path}")
    return config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Main
# ------------------------
def main():
    # Config
    config = load_config()
    set_seed(config["experiment"].get("seed", 42))

    # Device
    device = config["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # ------------------------
    # Dataset & DataLoader
    # ------------------------
    for split_file in [config["data"]["train_split"], config["data"]["val_split"]]:
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

    train_dataset = SteelDefectDataset(
        split_file=config["data"]["train_split"],
        img_dir=config["data"]["img_dir"],
        mask_dir=config["data"]["mask_dir"],
        augmentations=get_train_transforms(config["data"].get("img_size", (256, 1600))),
    )
    val_dataset = SteelDefectDataset(
        split_file=config["data"]["val_split"],
        img_dir=config["data"]["img_dir"],
        mask_dir=config["data"]["mask_dir"],
        augmentations=get_val_transforms(config["data"].get("img_size", (256, 1600))),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
    )

    # ------------------------
    # Model
    # ------------------------
    # ✅ Güncellenen UNet sınıfına yeni parametreler ekleyin
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        norm=config["model"].get("norm", "batch"),
        dropout=config["model"].get("dropout", 0.0),
        num_groups=config["model"].get("num_groups", 8),
        init=config["model"].get("init", "kaiming"),
        up_mode=config["model"].get("up_mode", "transpose"),
        encoder_name=config["model"].get("encoder_name", "resnet18"),
        encoder_weights=config["model"].get("encoder_weights", "imagenet"),
    )


    # ------------------------
    # Loss
    # ------------------------
    criterion = get_loss_from_config(config)

    # ------------------------
    # Optimizer
    # ------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
    )

    # ------------------------
    # Metrics
    # ------------------------
    metrics = build_metrics(config, class_names=[f"class_{i+1}" for i in range(config["model"]["out_channels"])])

    # ------------------------
    # Trainer
    # ------------------------
    visualize_out_dir = os.path.join(config["paths"]["outputs"], "visualizations")
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        metrics=metrics,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        use_amp=config["training"].get("use_amp", True),
        out_dir=config["paths"]["outputs"],
        clip_grad_norm=config["training"].get("clip_grad_norm", None),
        monitor="dice_mean",
        class_names=[f"class_{i+1}" for i in range(config["model"]["out_channels"])],
        visualize_out_dir=visualize_out_dir,
    )
    # ------------------------
    # Training Loop
    # ------------------------
    trainer.fit(
        num_epochs=config["training"]["epochs"],
        start_epoch=config["training"].get("start_epoch", 1),
        save_every=config["training"].get("save_every", 1),
        early_stopping=config["training"].get("early_stopping", None),
    )


if __name__ == "__main__":
    main()
