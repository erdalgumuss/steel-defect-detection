import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union


class DiceLoss(nn.Module):
    """
    Soft Dice Loss. İkili (binary) veya çoklu-sınıf (multi-class) segmentasyon için.
    
    Çoklu-sınıf senaryolarında her bir sınıfın Dice skorunu ayrı ayrı hesaplayıp
    ortalamasını alır, bu da dengesiz veri kümelerinde daha kararlı sonuçlar verir.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # İkili (binary) ve çoklu-sınıf (multi-class) senaryolarına uyum sağlar
        num_classes = logits.shape[1]
        probs = torch.sigmoid(logits)
        
        # Her sınıf için ayrı ayrı hesapla ve ortalamasını al
        dice_scores = []
        for i in range(num_classes):
            intersection = (probs[:, i, ...] * targets[:, i, ...]).sum()
            cardinality = probs[:, i, ...].sum() + targets[:, i, ...].sum()
            dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice_score)
        
        return 1.0 - torch.stack(dice_scores).mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss ve DiceLoss'un ağırlıklı kombinasyonu.
    """
    def __init__(self,
                 pos_weight: Optional[torch.Tensor] = None,
                 bce_weight: float = 0.75,
                 dice_weight: float = 0.25,
                 smooth: float = 1.0):
        super().__init__()

        # BCEWithLogitsLoss sadece 1 boyutlu pos_weight bekler.
        # Bu yüzden tensörün şeklini burada ayarlıyoruz.
        if pos_weight is not None:
            # pos_weight'in BCEWithLogitsLoss'a geçirilmesi için uygun boyutta olduğundan emin ol
            if pos_weight.ndim == 1:
                # Tek boyutluysa, broadcast için uygun hale getir
                # BCEWithLogitsLoss her sınıf için ayrı ağırlık bekler, bu nedenle
                # (1, num_classes, 1, 1) şekline getirilmelidir.
                pos_weight = pos_weight.view(1, len(pos_weight), 1, 1)

        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
def get_loss_from_config(config: Dict[str, Any]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "BCEDiceLoss")
    params = loss_cfg.get("params", {})
    
    device = torch.device(config.get("training", {}).get("device", "cpu"))

    if loss_type == "DiceLoss":
        return DiceLoss(smooth=float(params.get("smooth", 1.0)))

    elif loss_type == "BCEDiceLoss":
        pos_weight_cfg: Union[list, float, None] = params.get("pos_weight", None)
        pos_weight = None
        out_channels = config.get("model", {}).get("out_channels", 1)

        if isinstance(pos_weight_cfg, list):
            if len(pos_weight_cfg) != out_channels:
                 raise ValueError(f"pos_weight listesi boyutu ({len(pos_weight_cfg)}) model çıkış kanalı sayısına ({out_channels}) uymuyor.")
            # pos_weight tensörünü tek boyutlu olarak oluştur
            pos_weight = torch.tensor(pos_weight_cfg, dtype=torch.float, device=device)
            
        elif isinstance(pos_weight_cfg, (float, int)):
            pos_weight_val = float(pos_weight_cfg)
            # Tek bir float değerini out_channels boyutunda bir tensöre genişlet
            pos_weight = torch.full((out_channels,), pos_weight_val, dtype=torch.float, device=device)

        return BCEDiceLoss(
            pos_weight=pos_weight,
            bce_weight=float(params.get("bce_weight", 0.75)),
            dice_weight=float(params.get("dice_weight", 0.25)),
            smooth=float(params.get("smooth", 1.0))
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
