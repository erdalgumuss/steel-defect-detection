import torch
import numpy as np
from typing import Dict, Any, List, Optional


def _binarize(
    logits: torch.Tensor,
    threshold: float = 0.5,
    mode: str = "sigmoid"
) -> torch.Tensor:
    """
    Binarize predictions for segmentation.
    mode="sigmoid" -> multi-label (independent classes)
    mode="softmax" -> multi-class (exclusive class)
    """
    if mode == "sigmoid":
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()
    elif mode == "softmax":
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1, keepdim=True)
        return torch.nn.functional.one_hot(
            preds, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()
    else:
        raise ValueError(f"Unknown mode {mode}")


def _flatten_batch_spatial(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    return x.view(B, C, -1)


def dice_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    class_names: Optional[List[str]] = None,
    mode: str = "sigmoid",
    weighted: bool = False
) -> Dict[str, float]:
    out_scores: Dict[str, float] = {}
    try:
        if logits.numel() == 0:
            return {}

        preds = _binarize(logits, threshold, mode)
        preds_f = _flatten_batch_spatial(preds)
        targets_f = _flatten_batch_spatial(targets.float())

        inter = (preds_f * targets_f).sum(dim=2)
        sums = preds_f.sum(dim=2) + targets_f.sum(dim=2)

        dice_per_sample = (2.0 * inter + smooth) / (sums + smooth)  # (B, C)

        if weighted:
            weights = targets_f.sum(dim=2).sum(dim=0).cpu().numpy() + 1e-6
            scores = (dice_per_sample.sum(dim=0).cpu().numpy()) / (weights / weights.mean())
        else:
            scores = dice_per_sample.mean(dim=0).cpu().numpy()

        C = logits.shape[1]
        names = class_names if class_names is not None else [f"class_{i+1}" for i in range(C)]
        out_scores = {names[i]: float(scores[i]) for i in range(C)}

    except Exception as e:
        print(f"[ERROR] dice_per_class metrik hesaplamasında hata: {e}")
        # Hata durumunda tüm sınıflar için 0.0 değeri döndür
        C = logits.shape[1] if logits.numel() > 0 else 4 # varsayılan 4
        names = class_names if class_names is not None else [f"class_{i+1}" for i in range(C)]
        out_scores = {names[i]: 0.0 for i in range(C)}
    
    return out_scores


def dice_mean(*args, **kwargs) -> float:
    per = dice_per_class(*args, **kwargs)
    return float(np.mean(list(per.values()))) if per else 0.0


def iou_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    class_names: Optional[List[str]] = None,
    mode: str = "sigmoid",
    weighted: bool = False
) -> Dict[str, float]:
    out_scores: Dict[str, float] = {}
    try:
        if logits.numel() == 0:
            return {}

        preds = _binarize(logits, threshold, mode)
        preds_f = _flatten_batch_spatial(preds)
        targets_f = _flatten_batch_spatial(targets.float())

        inter = (preds_f * targets_f).sum(dim=2)
        union = preds_f.sum(dim=2) + targets_f.sum(dim=2) - inter

        iou_per_sample = (inter + smooth) / (union + smooth)

        if weighted:
            weights = targets_f.sum(dim=2).sum(dim=0).cpu().numpy() + 1e-6
            scores = (iou_per_sample.sum(dim=0).cpu().numpy()) / (weights / weights.mean())
        else:
            scores = iou_per_sample.mean(dim=0).cpu().numpy()

        C = logits.shape[1]
        names = class_names if class_names is not None else [f"class_{i+1}" for i in range(C)]
        out_scores = {names[i]: float(scores[i]) for i in range(C)}

    except Exception as e:
        print(f"[ERROR] iou_per_class metrik hesaplamasında hata: {e}")
        # Hata durumunda tüm sınıflar için 0.0 değeri döndür
        C = logits.shape[1] if logits.numel() > 0 else 4 # varsayılan 4
        names = class_names if class_names is not None else [f"class_{i+1}" for i in range(C)]
        out_scores = {names[i]: 0.0 for i in range(C)}
    
    return out_scores


def iou_mean(*args, **kwargs) -> float:
    per = iou_per_class(*args, **kwargs)
    return float(np.mean(list(per.values()))) if per else 0.0


def metrics_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_cfg: Dict[str, Any],
    iou_cfg: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Dice ve IoU için ayrı config parametreleri ile özet"""
    out: Dict[str, float] = {}
    try:
        dice_c = dice_per_class(
            logits, targets,
            threshold=dice_cfg.get("threshold", 0.5),
            smooth=dice_cfg.get("smooth", 1e-6),
            class_names=class_names,
            mode=dice_cfg.get("mode", "sigmoid"),
            weighted=dice_cfg.get("weighted", False)
        )
        for k, v in dice_c.items():
            out[f"dice_{k}"] = float(v)
        out["dice_mean"] = float(np.mean(list(dice_c.values()))) if dice_c else 0.0

        iou_c = iou_per_class(
            logits, targets,
            threshold=iou_cfg.get("threshold", 0.5),
            smooth=iou_cfg.get("smooth", 1e-6),
            class_names=class_names,
            mode=iou_cfg.get("mode", "sigmoid"),
            weighted=iou_cfg.get("weighted", False)
        )
        for k, v in iou_c.items():
            out[f"iou_{k}"] = float(v)
        out["iou_mean"] = float(np.mean(list(iou_c.values()))) if iou_c else 0.0

    except Exception as e:
        print(f"[ERROR] metrics_summary fonksiyonunda hata: {e}")
        # Hata durumunda 0.0 değeri döndür
        C = logits.shape[1] if logits.numel() > 0 else 4
        names = class_names if class_names is not None else [f"class_{i+1}" for i in range(C)]
        
        out["dice_mean"] = 0.0
        out["iou_mean"] = 0.0
        for name in names:
            out[f"dice_{name}"] = 0.0
            out[f"iou_{name}"] = 0.0

    # ✅ garanti: her şey float
    return {str(k): float(v) for k, v in out.items()}


def build_metrics(cfg: Dict[str, Any], class_names: Optional[List[str]] = None):
    dice_cfg = cfg.get("metrics", {}).get("dice", {})
    iou_cfg = cfg.get("metrics", {}).get("iou", {})

    return {
        "dice_mean": lambda logits, targets: dice_mean(logits, targets, **dice_cfg, class_names=class_names),
        "iou_mean": lambda logits, targets: iou_mean(logits, targets, **iou_cfg, class_names=class_names),
        "summary": lambda logits, targets: metrics_summary(logits, targets, dice_cfg, iou_cfg, class_names)
    }