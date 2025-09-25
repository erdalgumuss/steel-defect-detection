import torch
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from metrics.dice_coefficient import dice_coefficient, dice_per_class
from losses.weighted_focal_dice_loss import WeightedFocalDiceLoss


def train_one_epoch(model, loader: DataLoader, optimizer, device: str) -> Dict[str, float]:
    model.train()
    criterion = WeightedFocalDiceLoss(
        class_weights=[0.12, 0.03, 0.72, 0.11],
        gamma=2.0,
        lambda_focal=0.7,
        lambda_dice=0.3
    )

    running_loss, running_dice = 0.0, 0.0
    per_class_accum = None

    pbar = tqdm(loader, desc="Train", leave=False)
    for step, (images, masks, _) in enumerate(pbar, 1):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # Metrics
        dice_score = dice_coefficient(outputs, masks)
        class_scores = dice_per_class(outputs, masks)

        # Accumulate
        if per_class_accum is None:
            per_class_accum = torch.tensor(class_scores) * images.size(0)
        else:
            per_class_accum += torch.tensor(class_scores) * images.size(0)

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score.item() * images.size(0)

        # Step log (her batch için)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice_score.item():.4f}",
            "dice_c": [f"{c:.3f}" for c in class_scores]
        })

    n = len(loader.dataset)
    per_class_avg = (per_class_accum / n).tolist()

    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "per_class_dice": per_class_avg
    }

@torch.no_grad()


def validate_one_epoch(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    criterion = WeightedFocalDiceLoss(
        class_weights=[0.12, 0.03, 0.72, 0.11],
        gamma=2.0,
        lambda_focal=0.7,
        lambda_dice=0.3
    )

    running_loss, running_dice = 0.0, 0.0
    per_class_accum = None

    pbar = tqdm(loader, desc="Valid", leave=False)
    for step, (images, masks, _) in enumerate(pbar, 1):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)

        loss = criterion(outputs, masks)
        dice_score = dice_coefficient(outputs, masks)
        class_scores = dice_per_class(outputs, masks)

        if per_class_accum is None:
            per_class_accum = torch.tensor(class_scores) * images.size(0)
        else:
            per_class_accum += torch.tensor(class_scores) * images.size(0)

        running_loss += loss.item() * images.size(0)
        running_dice += dice_score.item() * images.size(0)

        # Step log
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice_score.item():.4f}",
            "dice_c": [f"{c:.3f}" for c in class_scores]
        })

    n = len(loader.dataset)
    per_class_avg = (per_class_accum / n).tolist()

    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "per_class_dice": per_class_avg
    }
