import os
import json
import torch
import pandas as pd
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from data.dataset import SteelDefectDataset, build_full_dataframe
from data.transforms import get_train_transforms, get_valid_transforms
from models.unet import UNetResNet18
from engines.training_engine import train_one_epoch, validate_one_epoch


class EarlyStopping:
    def __init__(self, patience=5, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score <= self.best_score) or \
             (self.mode == "min" and score >= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def main(config_path: str = "config.yaml"):
    # ---- Config ----
    cfg = Config(config_path).dict
    device = cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- Dataset ----
    train_csv = pd.read_csv(cfg["data"]["train_csv"])
    image_dir = cfg["data"]["train_images_dir"]

    full_df = build_full_dataframe(train_csv, image_dir)

    # Train / Valid Split
    train_ids, valid_ids = train_test_split(
        full_df["ImageId"].unique(),
        test_size=cfg["data"]["val_split"],
        random_state=42,
        shuffle=True
    )

    train_df = full_df[full_df["ImageId"].isin(train_ids)].reset_index(drop=True)
    valid_df = full_df[full_df["ImageId"].isin(valid_ids)].reset_index(drop=True)

    train_ds = SteelDefectDataset(
        train_df,
        image_dir,
        shape=(cfg["data"]["image_height"], cfg["data"]["image_width"]),
        num_classes=cfg["data"]["num_classes"],
        transforms=get_train_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    )

    valid_ds = SteelDefectDataset(
        valid_df,
        image_dir,
        shape=(cfg["data"]["image_height"], cfg["data"]["image_width"]),
        num_classes=cfg["data"]["num_classes"],
        transforms=get_valid_transforms(cfg["data"]["image_height"], cfg["data"]["image_width"])
    )

    # ---- DataLoaders ----
    num_workers = cfg["training"]["num_workers"]
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=persistent,
        collate_fn=lambda b: (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            [x[2] for x in b]
        )
    )

    # ---- Model ----
    model = UNetResNet18(
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        decoder_mode=cfg["model"].get("decoder_mode", "add")
    ).to(device)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # ---- Scheduler ----
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg.get("mode", "max"),
        factor=scheduler_cfg.get("factor", 0.5),
        patience=scheduler_cfg.get("patience", 2),
        verbose=scheduler_cfg.get("verbose", True)
    )

    # ---- Early Stopping ----
    early_cfg = cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=early_cfg.get("patience", 5),
        mode=early_cfg.get("mode", "max")
    )

    # ---- Training Loop ----
    history = {"train": [], "valid": []}
    best_dice = 0.0
    best_path = os.path.join(cfg["logging"]["output_dir"], "best_model.pth")

    for epoch in range(cfg["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['training']['epochs']}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        valid_metrics = validate_one_epoch(model, valid_loader, device)

        # Add LR info to metrics
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics["lr"] = current_lr
        valid_metrics["lr"] = current_lr

        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}")
        print(f"Valid Loss: {valid_metrics['loss']:.4f} | Valid Dice: {valid_metrics['dice']:.4f}")
        print(f"Current LR: {current_lr:.6f}")
        print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")

        history["train"].append(train_metrics)
        history["valid"].append(valid_metrics)

        # Best model kaydı
        if valid_metrics["dice"] > best_dice:
            best_dice = valid_metrics["dice"]
            os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"Best model updated at epoch {epoch+1}: {best_path}")

        # LR scheduler step
        scheduler.step(valid_metrics["dice"])

        # Early stopping
        early_stopper(valid_metrics["dice"])
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

        # Periyodik checkpoint kaydetme
        if (epoch + 1) % cfg["logging"]["save_every"] == 0:
            os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)
            ckpt_path = os.path.join(cfg["logging"]["checkpoint_dir"], f"model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # ---- Save Final Model (copy of best) ----
    os.makedirs(cfg["logging"]["output_dir"], exist_ok=True)
    final_path = os.path.join(cfg["logging"]["output_dir"], "model_final.pth")
    if os.path.exists(best_path):
        shutil.copy(best_path, final_path)
        print(f"Final model saved as copy of best: {final_path}")
    else:
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved (no best found): {final_path}")

    # ---- Save History ----
    history_path = Path(cfg["logging"]["output_dir"]) / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
