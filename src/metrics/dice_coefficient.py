import torch

def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Dice coefficient metric.
    
    Args:
        preds: (B, C, H, W) raw logits or probabilities
        targets: (B, C, H, W) ground truth mask (0/1)
        threshold: probability threshold for binarization
        epsilon: smoothing term to avoid division by zero

    Returns:
        Dice score averaged over batch and classes (scalar tensor)
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

    preds = preds.float()
    targets = targets.float()

    # Apply sigmoid if preds not in [0,1]
    if preds.max() > 1.0 or preds.min() < 0.0:
        preds = torch.sigmoid(preds)

    preds = (preds > threshold).float()

    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.mean()  # mean over batch and classes


def dice_per_class(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> list[float]:
    """
    Compute Dice score per class separately.

    Returns:
        list of dice values (len = C)
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.cpu().tolist()