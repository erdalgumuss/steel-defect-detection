import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (SOTA for imbalanced segmentation)
    - alpha: FN cezası
    - beta: FP cezası
    - gamma: Focal parametresi
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)

        preds = preds.view(preds.size(0), preds.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        TP = (preds * targets).sum(dim=2)
        FP = ((1 - targets) * preds).sum(dim=2)
        FN = (targets * (1 - preds)).sum(dim=2)

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky.mean()
