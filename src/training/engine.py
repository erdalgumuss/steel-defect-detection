import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Callable
from src.training.metrics import metrics_summary
# ✅ Yeni: Görselleştirme için gerekli importlar
from src.data.preprocess_utils import overlay_mask_on_image 


def train_one_epoch(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: Callable,
                    device: torch.device,
                    use_amp: bool = True,
                    clip_grad_norm: Optional[float] = None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    metrics_cfg: Optional[Dict] = None,
                    class_names: Optional[list] = None) -> Dict[str, float]:
    """
    Train for one epoch — only loss per batch, metrics once at end.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    all_logits, all_masks = [], []

    pbar = tqdm(enumerate(loader), total=n_batches, desc="Train", leave=False)
    for batch_idx, (imgs, masks) in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device).float()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)

            if batch_idx == 0:
                print(f"[DEBUG][TRAIN] logits shape: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
                print(f"[DEBUG][TRAIN] masks  shape: {masks.shape}, dtype: {masks.dtype}, device: {masks.device}")

            loss = criterion(logits, masks)

        # backward
        scaler.scale(loss).backward()
        if clip_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        avg_loss_so_far = total_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

        # save outputs for metrics
        all_logits.append(logits.detach().cpu())
        all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / n_batches if n_batches else 0.0

    # ✅ tek seferde metrik hesapla
    metrics_out = {}
    if metrics_cfg:
        all_logits = torch.cat(all_logits, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        metrics_out = metrics_summary(
            all_logits, all_masks,
            dice_cfg=metrics_cfg.get("dice", {}),
            iou_cfg=metrics_cfg.get("iou", {}),
            class_names=class_names
        )

    return {"loss": avg_loss, "metrics": metrics_out}


def validate_one_epoch(model: torch.nn.Module,
                       loader: torch.utils.data.DataLoader,
                       criterion: Callable,
                       metrics_cfg: Optional[Dict],
                       device: torch.device,
                       use_amp: bool = True,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       class_names: Optional[list] = None,
                       visualize_out_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Validate for one epoch — only loss per batch, metrics once at end.
    """
    model.eval()
    total_loss = 0.0
    n_batches = len(loader)

    all_logits, all_masks = [], []
    has_visualized = False

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=n_batches, desc="Val", leave=False)
        for batch_idx, (imgs, masks) in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            total_loss += loss.item()
            avg_loss_so_far = total_loss / (batch_idx + 1)
            pbar.set_postfix({"val_loss": f"{avg_loss_so_far:.4f}"})

            # ✅ Görselleştirme: Sadece ilk batç için
            if visualize_out_dir and not has_visualized:
                try:
                    # Görüntü, maske ve tahminleri CPU'ya ve numpy formatına çevir
                    img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    mask_np = masks[0].cpu().numpy()
                    logits_np = logits[0].detach().cpu().numpy()

                    # Tahminleri binarize et (threshold: 0.5)
                    pred_mask_np = (torch.sigmoid(torch.from_numpy(logits_np)) > 0.5).numpy().astype(np.uint8)
                    
                    # Gerçek maske overlay'i oluştur
                    overlay_gt = img_np.copy()
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR
                    for c in range(mask_np.shape[0]):
                        overlay_gt = overlay_mask_on_image(overlay_gt, mask_np[c], color=colors[c], alpha=0.35)

                    # Tahmin maskesi overlay'i oluştur
                    overlay_pred = img_np.copy()
                    for c in range(pred_mask_np.shape[0]):
                        overlay_pred = overlay_mask_on_image(overlay_pred, pred_mask_np[c], color=colors[c], alpha=0.35)

                    # Görselleri kaydet
                    cv2.imwrite(os.path.join(visualize_out_dir, "val_ground_truth_overlay.jpg"), overlay_gt)
                    cv2.imwrite(os.path.join(visualize_out_dir, "val_prediction_overlay.jpg"), overlay_pred)
                    print(f"\n[INFO] Görsel örnekler kaydedildi: {visualize_out_dir}")
                    has_visualized = True
                except Exception as e:
                    print(f"\n[HATA] Görselleştirme hatası: {e}")

            # save outputs for metrics
            all_logits.append(logits.detach().cpu())
            all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / n_batches if n_batches else 0.0

    metrics_out = {}
    if metrics_cfg:
        all_logits = torch.cat(all_logits, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        metrics_out = metrics_summary(
            all_logits, all_masks,
            dice_cfg=metrics_cfg.get("dice", {}),
            iou_cfg=metrics_cfg.get("iou", {}),
            class_names=class_names
        )

    return {"loss": avg_loss, "metrics": metrics_out}