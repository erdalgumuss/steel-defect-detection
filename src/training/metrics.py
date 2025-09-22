import torch
from typing import Dict, Any, List, Optional

def _binarize(
    logits: torch.Tensor,
    threshold: float = 0.5,
    mode: str = "sigmoid"
) -> torch.Tensor:
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
    return x.view(x.shape[0], x.shape[1], -1)

def _compute_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    mode: str = "sigmoid",
    weighted: bool = False,
    metric: str = "dice",
) -> Dict[str, float]:
    if logits.numel() == 0:
        return {}

    preds = _binarize(logits, threshold, mode)
    preds_f = _flatten_batch_spatial(preds)
    targets_f = _flatten_batch_spatial(targets.float())

    inter = (preds_f * targets_f).sum(dim=2)

    if metric == "dice":
        denom = preds_f.sum(dim=2) + targets_f.sum(dim=2)
        scores = (2 * inter + smooth) / (denom + smooth)
    elif metric == "iou":
        union = preds_f.sum(dim=2) + targets_f.sum(dim=2) - inter
        scores = (inter + smooth) / (union + smooth)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if weighted:
        weights = targets_f.sum(dim=2).sum(dim=0) + 1e-6
        per_class = (scores.sum(dim=0) / weights)
    else:
        per_class = scores.mean(dim=0)

    C = logits.shape[1]
    names = class_names or [f"class_{i+1}" for i in range(C)]

    # GPU'da kalacak, CPU'ya sadece float() döndürürken çekiyoruz
    return {names[i]: float(per_class[i].detach().cpu()) for i in range(C)}

# ---- Public API ----

def dice_per_class(*args, **kwargs) -> Dict[str, float]:
    return _compute_per_class(*args, **kwargs, metric="dice")

def dice_mean(*args, **kwargs) -> float:
    per = dice_per_class(*args, **kwargs)
    return sum(per.values()) / len(per) if per else 0.0

def iou_per_class(*args, **kwargs) -> Dict[str, float]:
    return _compute_per_class(*args, **kwargs, metric="iou")

def iou_mean(*args, **kwargs) -> float:
    per = iou_per_class(*args, **kwargs)
    return sum(per.values()) / len(per) if per else 0.0

def metrics_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_cfg: Dict[str, Any],
    iou_cfg: Dict[str, Any],
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dice_c = dice_per_class(
        logits, targets,
        threshold=dice_cfg.get("threshold", 0.5),
        smooth=dice_cfg.get("smooth", 1e-6),
        class_names=class_names,
        mode=dice_cfg.get("mode", "sigmoid"),
        weighted=dice_cfg.get("weighted", False),
    )
    out.update({f"dice_{k}": v for k, v in dice_c.items()})
    out["dice_mean"] = sum(dice_c.values()) / len(dice_c) if dice_c else 0.0

    iou_c = iou_per_class(
        logits, targets,
        threshold=iou_cfg.get("threshold", 0.5),
        smooth=iou_cfg.get("smooth", 1e-6),
        class_names=class_names,
        mode=iou_cfg.get("mode", "sigmoid"),
        weighted=iou_cfg.get("weighted", False),
    )
    out.update({f"iou_{k}": v for k, v in iou_c.items()})
    out["iou_mean"] = sum(iou_c.values()) / len(iou_c) if iou_c else 0.0

    return out

def build_metrics(cfg: Dict[str, Any], class_names: Optional[List[str]] = None):
    dice_cfg = cfg.get("metrics", {}).get("dice", {})
    iou_cfg = cfg.get("metrics", {}).get("iou", {})

    return {
        "dice_mean": lambda logits, targets: dice_mean(logits, targets, **dice_cfg, class_names=class_names),
        "iou_mean": lambda logits, targets: iou_mean(logits, targets, **iou_cfg, class_names=class_names),
        "summary": lambda logits, targets: metrics_summary(logits, targets, dice_cfg, iou_cfg, class_names),
    }
        #