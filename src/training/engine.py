import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Callable
from src.training.metrics import metrics_summary
from src.data.preprocess_utils import overlay_mask_on_image
import logging


class MetricTracker:
    """
    Epoch boyunca metrikleri RAM şişirmeden toplar.
    """
    def __init__(self, metrics_cfg: Optional[Dict], class_names: Optional[list] = None):
        self.metrics_cfg = metrics_cfg
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.all_logits = []
        self.all_masks = []

    def update(self, logits: torch.Tensor, masks: torch.Tensor):
        # CPU’ya taşı, detach et
        self.all_logits.append(logits.detach().cpu())
        self.all_masks.append(masks.detach().cpu())

    def compute(self) -> Dict[str, float]:
        if not self.metrics_cfg or not self.all_logits:
            return {}

        logits = torch.cat(self.all_logits, dim=0)
        masks = torch.cat(self.all_masks, dim=0)

        return metrics_summary(
            logits, masks,
            dice_cfg=self.metrics_cfg.get("dice", {}),
            iou_cfg=self.metrics_cfg.get("iou", {}),
            class_names=self.class_names
        )


def _run_one_epoch(model: torch.nn.Module,
                   loader: torch.utils.data.DataLoader,
                   optimizer: Optional[torch.optim.Optimizer],
                   criterion: Callable,
                   device: torch.device,
                   use_amp: bool,
                   scaler: Optional[torch.cuda.amp.GradScaler],
                   clip_grad_norm: Optional[float],
                   metrics_cfg: Optional[Dict],
                   class_names: Optional[list],
                   visualize_out_dir: Optional[str],
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   scheduler_step_on: str,
                   train: bool) -> Dict[str, float]:
    """
    Ortak epoch runner.
    train=True → train loop
    train=False → validation loop
    """
    model.train(mode=train)
    total_loss = 0.0
    n_batches = len(loader)

    metric_tracker = MetricTracker(metrics_cfg, class_names)
    has_visualized = False

    pbar = tqdm(enumerate(loader), total=n_batches, desc="Train" if train else "Val", leave=False)

    for batch_idx, (imgs, masks) in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        optimizer and optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)

        if train:
            scaler.scale(loss).backward()
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if scheduler and scheduler_step_on == "batch":
                scheduler.step()

        total_loss += loss.item()
        avg_loss_so_far = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss" if train else "val_loss": f"{avg_loss_so_far:.4f}"})

        # update metrics
        metric_tracker.update(logits, masks)

        # optional visualization (val loop, only first batch)
        if not train and visualize_out_dir and not has_visualized:
            try:
                _save_visualization(imgs[0], masks[0], logits[0], visualize_out_dir)
                has_visualized = True
            except Exception as e:
                logging.error(f"Görselleştirme hatası: {e}")

    avg_loss = total_loss / n_batches if n_batches else 0.0
    metrics_out = metric_tracker.compute()

    return {"loss": avg_loss, "metrics": metrics_out}


def train_one_epoch(model, loader, optimizer, criterion,
                    device, use_amp=True, clip_grad_norm=None,
                    scaler=None, metrics_cfg=None,
                    class_names=None, scheduler=None, scheduler_step_on="epoch"):
    return _run_one_epoch(model, loader, optimizer, criterion,
                          device, use_amp, scaler, clip_grad_norm,
                          metrics_cfg, class_names, None,
                          scheduler, scheduler_step_on, train=True)


def validate_one_epoch(model, loader, criterion,
                       metrics_cfg, device, use_amp=True,
                       scaler=None, class_names=None,
                       visualize_out_dir=None):
    return _run_one_epoch(model, loader, optimizer=None, criterion=criterion,
                          device=device, use_amp=use_amp, scaler=scaler,
                          clip_grad_norm=None, metrics_cfg=metrics_cfg,
                          class_names=class_names, visualize_out_dir=visualize_out_dir,
                          scheduler=None, scheduler_step_on="epoch", train=False)


def _save_visualization(img: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor, out_dir: str):
    """
    Gerçek maske ve tahmin maskelerini overlay olarak kaydeder.
    """
    os.makedirs(out_dir, exist_ok=True)

    img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    mask_np = mask.cpu().numpy()
    logits_np = logits.detach().cpu().numpy()

    pred_mask_np = (torch.sigmoid(torch.from_numpy(logits_np)) > 0.5).numpy().astype(np.uint8)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # BGR

    overlay_gt = img_np.copy()
    overlay_pred = img_np.copy()

    for c in range(mask_np.shape[0]):
        overlay_gt = overlay_mask_on_image(overlay_gt, mask_np[c], color=colors[c], alpha=0.35)
        overlay_pred = overlay_mask_on_image(overlay_pred, pred_mask_np[c], color=colors[c], alpha=0.35)

    cv2.imwrite(os.path.join(out_dir, "val_ground_truth_overlay.jpg"), overlay_gt)
    cv2.imwrite(os.path.join(out_dir, "val_prediction_overlay.jpg"), overlay_pred)
    logging.info(f"Görseller kaydedildi: {out_dir}")
