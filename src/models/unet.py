import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv(nn.Module):
    """(Conv -> Norm -> ReLU) * 2 with optional Dropout."""

    def __init__(self, in_channels: int, out_channels: int,
                 norm: Optional[str] = "batch", dropout: float = 0.0,
                 num_groups: int = 8):
        super().__init__()
        bias = norm is None  # norm varsa bias gereksiz
        layers = []

        # 1. conv
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))
        layers.append(nn.ReLU(inplace=True))

        # 2. conv
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias))
        if norm == "batch":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "group":
            layers.append(nn.GroupNorm(num_groups, out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    Configurable U-Net implementation.

    Args:
        in_channels (int): giriş kanal sayısı (ör. RGB için 3)
        out_channels (int): çıkış kanal sayısı (sınıf sayısı)
        features (list[int]): encoder feature sayıları
        norm (str): "batch", "group" veya None
        dropout (float): Dropout oranı
        num_groups (int): GroupNorm için grup sayısı
        init (str): "kaiming" | "xavier" | None (ağırlık başlatma yöntemi)
        up_mode (str): "transpose" (ConvTranspose2d) | "bilinear" (interpolate + conv)
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 4,
                 features: List[int] = [64, 128, 256, 512],
                 norm: Optional[str] = "batch",
                 dropout: float = 0.0,
                 num_groups: int = 8,
                 init: str = "kaiming",
                 up_mode: str = "transpose"):
        super().__init__()

        self.features = features
        self.norm = norm
        self.dropout = dropout
        self.num_groups = num_groups
        self.init = init
        self.up_mode = up_mode

        # -------- Encoder --------
        self.downs = nn.ModuleList()
        cur_in = in_channels
        for feat in features:
            self.downs.append(DoubleConv(cur_in, feat, norm=norm, dropout=dropout, num_groups=num_groups))
            cur_in = feat

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------- Bottleneck --------
        bottleneck_channels = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_channels, norm=norm,
                                     dropout=dropout, num_groups=num_groups)

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        in_ch = bottleneck_channels
        for feat in reversed(features):
            if self.up_mode == "transpose":
                self.ups.append(nn.ConvTranspose2d(in_ch, feat, kernel_size=2, stride=2))
            else:  # bilinear upsampling
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, feat, kernel_size=1)
                ))
            self.ups.append(DoubleConv(feat * 2, feat, norm=norm, dropout=dropout, num_groups=num_groups))
            in_ch = feat

        # -------- Final layer --------
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # -------- Init weights --------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.init == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif self.init == "xavier":
                    nn.init.xavier_normal_(m.weight)
                else:
                    # Varsayılan PyTorch init
                    pass
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, apply_activation: Optional[str] = None) -> torch.Tensor:
        # --- Encoder ---
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder ---
        skip_connections = skip_connections[::-1]
        up_idx = 0
        for i in range(0, len(self.ups), 2):
            upsample = self.ups[i]
            double_conv = self.ups[i + 1]

            x = upsample(x)
            skip = skip_connections[up_idx]
            up_idx += 1

            if x.shape[2:] != skip.shape[2:]:  # boyut uyuşmazlığı varsa
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)
            x = double_conv(x)

        x = self.final_conv(x)

        # Aktivasyon (opsiyonel)
        if apply_activation == "sigmoid":
            return torch.sigmoid(x)
        elif apply_activation == "softmax":
            return torch.softmax(x, dim=1)
        return x  # logits
