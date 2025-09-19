import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataset import SteelDefectDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.training.metrics import dice_coeff

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
            dice_scores.append(dice_coeff(outputs, masks).item())
    return total_loss / len(loader), sum(dice_scores) / len(dice_scores)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = SteelDefectDataset(
        split_file="data/processed/splits/train.txt",
        img_dir="data/raw/train_images",
        mask_dir="data/processed/masks_png",
        augmentations=get_train_transforms()
    )
    val_dataset = SteelDefectDataset(
        split_file="data/processed/splits/val.txt",
        img_dir="data/raw/train_images",
        mask_dir="data/processed/masks_png",
        augmentations=get_val_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=4).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3  # duman testi için az
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

    # Çıktıları kaydet
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/unet_baseline.pth")

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
