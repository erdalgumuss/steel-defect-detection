import torch

def dice_coef_per_class(logits, targets, threshold=0.5, smooth=1.0):
    """
    logits: (B, C, H, W), targets: (B, C, H, W)
    returns: [dice_c1, dice_c2, dice_c3, dice_c4]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dices = []
    for c in range(targets.shape[1]):
        pred_c = preds[:, c]
        target_c = targets[:, c]
        num = 2 * (pred_c * target_c).sum() + smooth
        den = pred_c.sum() + target_c.sum() + smooth
        dices.append((num / den).item())
    return dices

def dice_mean(logits, targets, threshold=0.5):
    per_class = dice_coef_per_class(logits, targets, threshold)
    return sum(per_class) / len(per_class)
