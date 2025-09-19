# src/training/metrics.py
import torch

def dice_coef_per_class(logits, targets, threshold=0.5, smooth=1.0):
    """
    logits: (B, C, H, W)
    targets: (B, C, H, W)
    Returns: dict -> {"class_1": dice, ..., "class_n": dice}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dices = {}
    C = targets.shape[1]

    for c in range(C):
        pred_c = preds[:, c]
        target_c = targets[:, c]

        num = 2 * (pred_c * target_c).sum() + smooth
        den = pred_c.sum() + target_c.sum() + smooth
        dices[f"class_{c+1}"] = (num / den).item()

    return dices


def dice_mean(logits, targets, threshold=0.5, smooth=1.0):
    """
    Ortalama Dice skoru (tüm class'ların ortalaması).
    """
    per_class = dice_coef_per_class(logits, targets, threshold, smooth)
    return sum(per_class.values()) / len(per_class)


def iou_per_class(logits, targets, threshold=0.5, smooth=1.0):
    """
    Intersection-over-Union (IoU) metriği.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    ious = {}
    C = targets.shape[1]

    for c in range(C):
        pred_c = preds[:, c]
        target_c = targets[:, c]

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        ious[f"class_{c+1}"] = ((intersection + smooth) / (union + smooth)).item()

    return ious


def iou_mean(logits, targets, threshold=0.5, smooth=1.0):
    """
    Ortalama IoU skoru.
    """
    per_class = iou_per_class(logits, targets, threshold, smooth)
    return sum(per_class.values()) / len(per_class)
