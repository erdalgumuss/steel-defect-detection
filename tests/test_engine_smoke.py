import unittest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SteelDefectDataset
from src.data.transforms import build_transforms
from src.models.unet_resnet import UNet
from src.training import engine, losses


class TestEngineSmoke(unittest.TestCase):

    def setUp(self):
        # Küçük bir config taklidi
        self.cfg = {
            "data": {
                "train_split": "data/processed/splits/train_mini.txt",
                "val_split": "data/processed/splits/val_mini.txt",
                "img_dir": "data/raw/train_images",
                "mask_dir": "data/processed/masks_png",
                "img_size": [128, 800],  # küçük tut
            },
            "model": {
                "in_channels": 3,
                "out_channels": 4,
                "encoder_name": "resnet18",
                "encoder_weights": None,
            },
            "training": {
                "device": "cpu",
                "batch_size": 2,
                "num_workers": 0,
                "use_amp": False,
                "clip_grad_norm": 1.0,
            },
            "loss": {
                "type": "BCEDiceLoss",
                "params": {"bce_weight": 0.7, "dice_weight": 0.3},
            },
            "metrics": {
                "dice": {"threshold": 0.5},
                "iou": {"threshold": 0.5},
            },
        }

        # Dataset & Loader
        train_dataset = SteelDefectDataset(
            split_file=self.cfg["data"]["train_split"],
            img_dir=self.cfg["data"]["img_dir"],
            mask_dir=self.cfg["data"]["mask_dir"],
            augmentations=build_transforms("train", tuple(self.cfg["data"]["img_size"])),
        )
        val_dataset = SteelDefectDataset(
            split_file=self.cfg["data"]["val_split"],
            img_dir=self.cfg["data"]["img_dir"],
            mask_dir=self.cfg["data"]["mask_dir"],
            augmentations=build_transforms("val", tuple(self.cfg["data"]["img_size"])),
        )

        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        # Model
        self.device = torch.device(self.cfg["training"]["device"])
        self.model = UNet(
            in_channels=self.cfg["model"]["in_channels"],
            out_channels=self.cfg["model"]["out_channels"],
            encoder_name=self.cfg["model"]["encoder_name"],
            encoder_weights=self.cfg["model"]["encoder_weights"],
        ).to(self.device)

        # Loss
        self.criterion = losses.get_loss_from_config(self.cfg)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def test_train_and_val_epoch(self):
        # Train loop
        train_stats = engine.train_one_epoch(
            self.model, self.train_loader, self.optimizer,
            self.criterion, self.device,
            use_amp=self.cfg["training"]["use_amp"],
            clip_grad_norm=self.cfg["training"]["clip_grad_norm"],
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            metrics_cfg=self.cfg["metrics"],
        )
        print("Train stats:", train_stats)
        self.assertIn("loss", train_stats)
        self.assertIn("metrics", train_stats)

        # Val loop
        val_stats = engine.validate_one_epoch(
            self.model, self.val_loader, self.criterion,
            metrics_cfg=self.cfg["metrics"], device=self.device,
            use_amp=self.cfg["training"]["use_amp"],
            scaler=torch.cuda.amp.GradScaler(enabled=False),
            visualize_out_dir="outputs/visualizations_test"
        )
        print("Val stats:", val_stats)
        self.assertIn("loss", val_stats)
        self.assertIn("metrics", val_stats)


if __name__ == "__main__":
    unittest.main()
