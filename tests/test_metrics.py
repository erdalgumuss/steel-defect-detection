import unittest
import torch
from src.training import metrics

###
####run: python test/test_metrics.py ###
####

class TestMetrics(unittest.TestCase):

    def test_per_class_shapes_and_range(self):
        B, C, H, W = 2, 4, 16, 16
        logits = torch.randn(B, C, H, W)
        targets = (torch.rand(B, C, H, W) > 0.5).float()

        dice_scores = metrics.dice_per_class(logits, targets)
        iou_scores = metrics.iou_per_class(logits, targets)

        # doğru sınıf sayısı döndürüyor mu?
        self.assertEqual(len(dice_scores), C)
        self.assertEqual(len(iou_scores), C)

        # skor aralığı [0,1]
        for v in list(dice_scores.values()) + list(iou_scores.values()):
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_dice_iou_perfect_match(self):
        B, C, H, W = 1, 3, 8, 8
        targets = (torch.rand(B, C, H, W) > 0.5).float()
        logits = targets.clone() * 20.0  # yüksek değer → sigmoid ≈1

        dice_scores = metrics.dice_per_class(logits, targets)
        iou_scores = metrics.iou_per_class(logits, targets)

        for v in dice_scores.values():
            self.assertAlmostEqual(v, 1.0, places=6)
        for v in iou_scores.values():
            self.assertAlmostEqual(v, 1.0, places=6)

    def test_metrics_summary_consistency(self):
        B, C, H, W = 2, 2, 8, 8
        logits = torch.randn(B, C, H, W)
        targets = (torch.rand(B, C, H, W) > 0.5).float()

        dice_cfg = {"threshold": 0.5}
        iou_cfg = {"threshold": 0.5}

        summary = metrics.metrics_summary(logits, targets, dice_cfg, iou_cfg)

        self.assertIn("dice_mean", summary)
        self.assertIn("iou_mean", summary)

        dice_values = [v for k, v in summary.items() if k.startswith("dice_class")]
        iou_values = [v for k, v in summary.items() if k.startswith("iou_class")]

        if dice_values:
            self.assertAlmostEqual(summary["dice_mean"], sum(dice_values) / len(dice_values), places=6)
        if iou_values:
            self.assertAlmostEqual(summary["iou_mean"], sum(iou_values) / len(iou_values), places=6)


if __name__ == "__main__":
    unittest.main()
