import os
import shutil
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.unet_resnet import UNet
from src.training.losses import DiceLoss
from src.training.trainer import Trainer


class TestTrainerSmoke(unittest.TestCase):

    def setUp(self):
        # Çalışma dizini (geçici)
        self.out_dir = "tmp_trainer_test"
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

        os.makedirs(self.out_dir, exist_ok=True)

        # Mini dataset (4 örnek → 2 train, 2 val)
        B, C, H, W = 4, 3, 64, 64
        X = torch.randn(B, C, H, W)
        y = (torch.rand(B, 2, H, W) > 0.5).float()

        train_ds = TensorDataset(X[:2], y[:2])
        val_ds = TensorDataset(X[2:], y[2:])

        self.train_loader = DataLoader(train_ds, batch_size=1)
        self.val_loader = DataLoader(val_ds, batch_size=1)

        # Model
        self.model = UNet(
            in_channels=3,
            out_channels=2,
            encoder_name="resnet18",
            encoder_weights=None
        )

        # Optimizer + Loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = DiceLoss()

    def tearDown(self):
        # Test çıktısını temizle
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def test_trainer_fit(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            metrics=None,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device="cpu",   # hızlı test
            out_dir=self.out_dir,
            use_amp=False   # CPU’da amp kapalı
        )

        # Fit 2 epoch, early stopping patience=1
        trainer.fit(num_epochs=2, early_stopping=1)

        # Checkpoint ve history kaydı kontrolü
        ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "epoch_1.pth")))
        self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "last.pth")))
        self.assertTrue(os.path.exists(os.path.join(self.out_dir, "training_history.json")))

        print("🎉 Trainer smoke test passed!")


if __name__ == "__main__":
    unittest.main()
