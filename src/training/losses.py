import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, C, H, W)
        """
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        loss = 1 - (num / den)
        # hem batch hem class üzerinde ortalama
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=0.75, dice_weight=0.25):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss
