import torch
from src.training.losses import DiceLoss, BCEDiceLoss, get_loss_from_config


def run_smoke_test():
    B, C, H, W = 2, 4, 16, 16
    logits = torch.randn(B, C, H, W)
    targets = (torch.rand(B, C, H, W) > 0.5).float()

    print("=== DiceLoss ===")
    dice_loss = DiceLoss(smooth=1.0)
    out1 = dice_loss(logits, targets)
    print(f"DiceLoss output: {out1.item():.4f}")

    print("=== BCEDiceLoss (no pos_weight) ===")
    bce_dice_loss = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
    out2 = bce_dice_loss(logits, targets)
    print(f"BCEDiceLoss output: {out2.item():.4f}")

    print("=== BCEDiceLoss (with pos_weight list) ===")
    pos_weight = [1.0, 2.0, 3.0, 4.0]
    bce_dice_loss_pw = BCEDiceLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float),
        bce_weight=0.6, dice_weight=0.4
    )
    out3 = bce_dice_loss_pw(logits, targets)
    print(f"BCEDiceLoss with pos_weight output: {out3.item():.4f}")

    print("=== get_loss_from_config ===")
    config = {
        "training": {"device": "cpu"},
        "model": {"out_channels": 4},
        "loss": {
            "type": "BCEDiceLoss",
            "params": {
                "bce_weight": 0.7,
                "dice_weight": 0.3,
                "smooth": 1.0,
                "pos_weight": [1.0, 2.0, 3.0, 4.0],
            }
        }
    }
    criterion = get_loss_from_config(config)
    out4 = criterion(logits, targets)
    print(f"get_loss_from_config output: {out4.item():.4f}")

    print("🎉 All loss smoke tests passed!")


if __name__ == "__main__":
    run_smoke_test()
