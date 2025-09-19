import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataset import SteelDefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet_effnet import UNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import dice_mean



def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Config dosyası boş: {path}")
    return config

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            dice_scores.append(dice_mean(outputs, masks))
    return total_loss / len(loader), sum(dice_scores) / len(dice_scores)


def main():
    # Config oku
    config = load_config("config.yaml")

    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    # Dataset
    train_dataset = SteelDefectDataset(
        split_file=config["data"]["train_split"],
        img_dir=config["data"]["img_dir"],
        mask_dir=config["data"]["mask_dir"],
        augmentations=get_train_transforms()
    )
    val_dataset = SteelDefectDataset(
        split_file=config["data"]["val_split"],
        img_dir=config["data"]["img_dir"],
        mask_dir=config["data"]["mask_dir"],
        augmentations=get_val_transforms()
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    # Model
    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    ).to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    num_epochs = config["training"]["epochs"]
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    # Çıktılar
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config["training"]["checkpoint_dir"], "unet_baseline.pth"))

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.savefig("outputs/train_val_loss.png")
    plt.close()

    plt.figure()
    plt.plot(history["val_dice"], label="Val Dice")
    plt.legend()
    plt.savefig("outputs/val_dice.png")
    plt.close()


if __name__ == "__main__":
    main()
