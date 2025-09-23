import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


class WeightedFocalDiceLoss(nn.Module):
    """
    Combo Loss: Weighted Focal + Dice
    Class imbalance için alpha ağırlıkları eklenebilir.
    """

    def __init__(self, class_weights=None, gamma: float = 2.0, lambda_focal: float = 0.7, lambda_dice: float = 0.3):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.dice_loss = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W)
        preds = torch.sigmoid(preds)
        eps = 1e-8

        # BCE
        bce = -(targets * torch.log(preds + eps) + (1 - targets) * torch.log(1 - preds + eps))

        # Focal Weight
        pt = torch.where(targets == 1, preds, 1 - preds)
        focal_weight = (1 - pt) ** self.gamma

        if self.class_weights is not None:
            class_weights = preds.new_tensor(self.class_weights).view(1, -1, 1, 1)
            focal_weight = focal_weight * class_weights

        focal_loss = (focal_weight * bce).mean()

        # Dice
        dice_loss = self.dice_loss(preds, targets)

        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss
