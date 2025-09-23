import torch
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from metrics.dice_coefficient import dice_coefficient
from losses.weighted_focal_dice_loss import WeightedFocalDiceLoss


def train_one_epoch(model, loader: DataLoader, optimizer, device: str) -> Dict[str, float]:
    model.train()
    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()

        # Dataset class distribution'a göre ağırlıklar (örnek)
    class_weights = [0.12, 0.03, 0.72, 0.11]  # yüzdeliklere yakın
    criterion = WeightedFocalDiceLoss(class_weights=class_weights, gamma=2.0, 
                                      lambda_focal=0.7, lambda_dice=0.3)

    running_loss, running_dice = 0.0, 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks)


        loss.backward()
        optimizer.step()

        # Metric
        dice_score = dice_coefficient(outputs, masks)

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score.item() * images.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice_score.item():.4f}"})

    n = len(loader.dataset)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n
    }


@torch.no_grad()
def validate_one_epoch(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()

    
    class_weights = [0.12, 0.03, 0.72, 0.11]
    criterion = WeightedFocalDiceLoss(class_weights=class_weights, gamma=2.0, 
                                      lambda_focal=0.7, lambda_dice=0.3)


    running_loss, running_dice = 0.0, 0.0

    pbar = tqdm(loader, desc="Valid", leave=False)
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        loss = criterion(outputs, masks)
        dice_score = dice_coefficient(outputs, masks)


        running_loss += loss.item() * images.size(0)
        running_dice += dice_score.item() * images.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice_score.item():.4f}"})

    n = len(loader.dataset)
    return {
        "loss": running_loss / n,
        "dice": running_dice / n
    }
