import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class DoubleConv(nn.Module):
    """(Conv -> Norm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels,
                 norm: Optional[str] = "batch",
                 dropout: float = 0.0,
                 num_groups: int = 8):
        super().__init__()
        bias = norm is None
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

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net with ResNet encoder
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 4,
                 norm: str = "batch",
                 dropout: float = 0.0,
                 num_groups: int = 8,
                 init: str = "kaiming",
                 up_mode: str = "transpose",
                 encoder_name: str = "resnet18",
                 encoder_weights: str = "imagenet"):
        super().__init__()

        # -------- ResNet Encoder --------
        if encoder_name == "resnet18":
            if encoder_weights == "imagenet":
                weights_enum = models.ResNet18_Weights.IMAGENET1K_V1
            elif encoder_weights is None:
                weights_enum = None
            else:
                weights_enum = encoder_weights

            self.encoder = models.resnet18(weights=weights_enum)
            self.encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # ResNet katmanlarını ayır
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        self.layer1 = self.encoder.layer1
        self.layer2 = self.encoder.layer2
        self.layer3 = self.encoder.layer3
        self.layer4 = self.encoder.layer4

        # -------- Decoder --------
        self.ups = nn.ModuleList()
        decoder_channels = self.encoder_channels[:-1][::-1]  # [256,128,64,64]

        in_ch = self.encoder_channels[-1]  # 512
        skip_channels = self.encoder_channels[:-1][::-1]     # [256,128,64,64]

        for out_ch, skip_ch in zip(decoder_channels, skip_channels):
            if up_mode == "transpose":
                self.ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            else:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1)
                ))
            self.ups.append(DoubleConv(out_ch + skip_ch, out_ch, norm=norm,
                                       dropout=dropout, num_groups=num_groups))
            in_ch = out_ch

        # Final conv
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

        # Init weights
        self._init_weights(init)

    def _init_weights(self, init_type="kaiming"):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, apply_activation: Optional[str] = None):
        # --- Encoder ---
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)  # 64
        x3 = self.layer2(x2)  # 128
        x4 = self.layer3(x3)  # 256
        x5 = self.layer4(x4)  # 512

        skips = [x4, x3, x2, x0]  # en derinden yüzeye

        # --- Decoder ---
        up_idx = 0
        for i in range(0, len(self.ups), 2):
            up = self.ups[i]
            conv = self.ups[i + 1]

            x5 = up(x5)
            skip = skips[up_idx]
            up_idx += 1

            if x5.shape[2:] != skip.shape[2:]:
                x5 = F.interpolate(x5, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x5 = torch.cat([skip, x5], dim=1)
            x5 = conv(x5)

        x = self.final_conv(x5)

        # 🔑 Çıkışı giriş çözünürlüğüne zorla
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2),
                          mode="bilinear", align_corners=False)

        if apply_activation == "sigmoid":
            return torch.sigmoid(x)
        elif apply_activation == "softmax":
            return torch.softmax(x, dim=1)
        return x
