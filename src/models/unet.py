import torch
import torch.nn as nn
import torchvision.models as models


class UNetResNet18(nn.Module):
    """
    U-Net style segmentation model with ResNet-18 encoder.
    Input:  (B, 3, H, W)
    Output: (B, num_classes, H, W)
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, decoder_mode: str = "add"):
        super().__init__()
        self.decoder_mode = decoder_mode
        base_model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # ----- Encoder -----
        self.enc1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)  # (64, H/2, W/2)
        self.enc2 = nn.Sequential(base_model.maxpool, base_model.layer1)              # (64, H/4, W/4)
        self.enc3 = base_model.layer2                                                # (128, H/8, W/8)
        self.enc4 = base_model.layer3                                                # (256, H/16, W/16)
        self.enc5 = base_model.layer4                                                # (512, H/32, W/32)

        # ----- Decoder -----
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256, 128)
        self.up2 = self._up_block(128, 64)
        self.up1 = self._up_block(64, 64)

        # Channel reducers (sadece concat modunda gerekli)
        if self.decoder_mode == "concat":
            self.red4 = nn.Conv2d(256 + 256, 256, kernel_size=1)
            self.red3 = nn.Conv2d(128 + 128, 128, kernel_size=1)
            self.red2 = nn.Conv2d(64 + 64, 64, kernel_size=1)
            self.red1 = nn.Conv2d(64 + 64, 64, kernel_size=1)

        # ----- Final conv -----
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # Ek upsample → output'u input boyutuna eşitle
        self.upsample_out = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def _skip_connect(self, up: torch.Tensor, enc: torch.Tensor, reducer: nn.Module | None = None) -> torch.Tensor:
        if self.decoder_mode == "add":
            return up + enc
        else:  # concat
            x = torch.cat([up, enc], dim=1)
            return reducer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Encoder -----
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # ----- Decoder (skip connections) -----
        d4 = self._skip_connect(self.up4(e5), e4, getattr(self, "red4", None))
        d3 = self._skip_connect(self.up3(d4), e3, getattr(self, "red3", None))
        d2 = self._skip_connect(self.up2(d3), e2, getattr(self, "red2", None))
        d1 = self._skip_connect(self.up1(d2), e1, getattr(self, "red1", None))

        out = self.final(d1)          # (B, C, H/2, W/2)
        out = self.upsample_out(out)  # (B, C, H, W)
        return out
