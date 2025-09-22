import os
import json
import time
import torch
import logging
from typing import Optional, Dict, Any, Callable, List
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from src.training import engine


class Trainer:
    """
    Yüksek seviye eğitim yöneticisi.

    - train/val döngülerini çalıştırır
    - checkpoint kaydı / yükleme
    - history.json incremental kaydı
    - early stopping, scheduler desteği
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        metrics: Optional[Dict[str, Callable]],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: str = "cuda",
        scheduler: Optional[_LRScheduler] = None,
        scheduler_step_on: str = "epoch",  # ✅ "epoch" | "batch"
        use_amp: bool = True,
        out_dir: str = "outputs",
        clip_grad_norm: Optional[float] = None,
        monitor: str = "dice_mean",
        monitor_mode: str = "max",  # ✅ "max" | "min"
        class_names: Optional[List[str]] = None,
        visualize_out_dir: Optional[str] = None,
    ):
        # core
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics or {}
        self.train_loader = train_loader
        self.val_loader = val_loader

        # config
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.scheduler_step_on = scheduler_step_on
        self.use_amp = use_amp and self.device.type == "cuda"
        self.clip_grad_norm = clip_grad_norm
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.class_names = class_names
        self.visualize_out_dir = visualize_out_dir

        # state
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = -float("inf") if monitor_mode == "max" else float("inf")
        self.history: Dict[str, Any] = {"train_loss": [], "val_loss": []}

        # dirs
        self.out_dir = out_dir
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        if self.visualize_out_dir:
            os.makedirs(self.visualize_out_dir, exist_ok=True)
            logging.info(f"Görsel çıktı dizini oluşturuldu: {self.visualize_out_dir}")

        # logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    # ------------------------
    # Checkpoint
    # ------------------------
    def save_checkpoint(self, epoch: int, is_best: bool = False, name: str = "checkpoint.pth"):
        """Eğitim durumunu kaydeder."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
        }
        path = os.path.join(self.out_dir, "checkpoints", name)
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, os.path.join(self.out_dir, "checkpoints", "best.pth"))

    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """Kaydedilmiş bir eğitim durumunu yükler."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass
        return ckpt

    # ------------------------
    # Training Loop
    # ------------------------
    def fit(
        self,
        num_epochs: int,
        start_epoch: int = 1,
        save_every: int = 1,
        early_stopping: Optional[int] = None,
    ):
        patience = 0
        self.start_time = time.time()

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            logging.info(f"Epoch {epoch}/{num_epochs}")

            # ---- train ----
            train_stats = engine.train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                scaler=self.scaler,
                use_amp=self.use_amp,
                clip_grad_norm=self.clip_grad_norm,
                metrics_cfg=self.metrics,
                class_names=self.class_names,
            )

            # ---- validate ----
            val_stats = engine.validate_one_epoch(
                model=self.model,
                loader=self.val_loader,
                criterion=self.criterion,
                metrics_cfg=self.metrics,
                device=self.device,
                use_amp=self.use_amp,
                class_names=self.class_names,
                visualize_out_dir=self.visualize_out_dir,
            )

            val_loss = val_stats["loss"]
            val_metrics = val_stats["metrics"]

            # ---- scheduler ----
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(self.monitor, val_loss))
                elif self.scheduler_step_on == "epoch":
                    self.scheduler.step()

            # ---- history ----
            self.history["train_loss"].append(float(train_stats["loss"]))
            self.history["val_loss"].append(float(val_loss))
            for k, v in train_stats.get("metrics", {}).items():
                self.history.setdefault(f"train_{k}", []).append(float(v))
            for k, v in val_metrics.items():
                self.history.setdefault(f"val_{k}", []).append(float(v))

            elapsed = time.time() - t0
            logging.info(
                f"Train Loss: {train_stats['loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"{self.monitor}: {val_metrics.get(self.monitor, 0.0):.4f} | "
                f"time: {elapsed:.1f}s"
            )

            # ---- checkpoint ----
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False, name=f"epoch_{epoch}.pth")

            # ---- best + early stopping ----
            key_metric = val_metrics.get(self.monitor, val_loss)
            is_better = (
                (self.monitor_mode == "max" and key_metric > self.best_metric)
                or (self.monitor_mode == "min" and key_metric < self.best_metric)
            )
            if is_better:
                self.best_metric = key_metric
                self.save_checkpoint(epoch, is_best=True, name="best_epoch.pth")
                logging.info(f"Yeni en iyi {self.monitor}: {self.best_metric:.4f}")
                patience = 0
            else:
                patience += 1
                if early_stopping is not None and patience >= early_stopping:
                    logging.info(f"Erken durdurma, epoch {epoch} sonunda.")
                    break

            # ---- incremental history save ----
            self._save_history()

        # ---- final save ----
        self.save_checkpoint(num_epochs, is_best=False, name="last.pth")
        self._save_history()
        logging.info("Eğitim tamamlandı.")

    # ------------------------
    # Helpers
    # ------------------------
    def _save_history(self):
        """Eğitim geçmişini JSON dosyasına kaydeder (incremental)."""
        history_path = os.path.join(self.out_dir, "training_history.json")
        with open(history_path, "w") as f:
            serializable_history = {k: [float(x) for x in v] for k, v in self.history.items()}
            json.dump(serializable_history, f, indent=4)
