import unittest
import torch
from src.training import losses


class TestLosses(unittest.TestCase):

    def test_dice_loss_perfect_match(self):
        B, C, H, W = 1, 2, 8, 8
        targets = (torch.rand(B, C, H, W) > 0.5).float()
        # logits = yüksek pozitif (mask=1), yüksek negatif (mask=0)
        logits = targets * 20.0 + (1 - targets) * -20.0

        loss_fn = losses.DiceLoss(smooth=1e-6)
        loss = loss_fn(logits, targets)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_dice_loss_total_mismatch(self):
        B, C, H, W = 1, 2, 8, 8
        targets = (torch.rand(B, C, H, W) > 0.5).float()
        # logits = ters mask: sigmoid çıktısı 1 - targets olacak
        logits = (1 - targets) * 20.0 + targets * -20.0

        loss_fn = losses.DiceLoss(smooth=1e-6)
        loss = loss_fn(logits, targets)

        self.assertGreaterEqual(loss.item(), 0.99)  # neredeyse 1.0 olmalı

    def test_bce_dice_loss_weights(self):
        B, C, H, W = 2, 3, 16, 16
        logits = torch.randn(B, C, H, W)
        targets = (torch.rand(B, C, H, W) > 0.5).float()

        pos_weight = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = losses.BCEDiceLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, targets)

        self.assertTrue(torch.isfinite(loss).item())

    def test_get_loss_from_config_dice(self):
        cfg = {
            "loss": {"type": "DiceLoss", "params": {"smooth": 1e-6}},
            "model": {"out_channels": 2},
            "training": {"device": "cpu"},
        }
        loss_fn = losses.get_loss_from_config(cfg)
        self.assertIsInstance(loss_fn, losses.DiceLoss)

    def test_get_loss_from_config_bce_dice(self):
        cfg = {
            "loss": {
                "type": "BCEDiceLoss",
                "params": {"pos_weight": [1.0, 2.0]},
            },
            "model": {"out_channels": 2},
            "training": {"device": "cpu"},
        }
        loss_fn = losses.get_loss_from_config(cfg)
        self.assertIsInstance(loss_fn, losses.BCEDiceLoss)


if __name__ == "__main__":
    unittest.main()
