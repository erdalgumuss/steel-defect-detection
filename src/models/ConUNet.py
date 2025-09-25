import torch
import torch.nn as nn
import torchvision.models as models


class UNetResNet18(nn.Module):
    """
    U-Net style segmentation model with ResNet-18 encoder.
    Supports two decoder modes: "add" (lightweight) or "concat" (richer features).
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, decoder_mode: str = "add"):
        super().__init__()
        base_model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # ----- Encoder -----
        self.enc1 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)  # (64, H/2, W/2)
        self.enc2 = nn.Sequential(base_model.maxpool, base_model.layer1)              # (64, H/4, W/4)
        self.enc3 = base_model.layer2                                                # (128, H/8, W/8)
        self.enc4 = base_model.layer3                                                # (256, H/16, W/16)
        self.enc5 = base_model.layer4                                                # (512, H/32, W/32)

        self.decoder_mode = decoder_mode

        # ----- Decoder -----
        if decoder_mode == "add":
            # Basit toplama (hafif)
            self.up4 = self._up_block(512, 256)
            self.up3 = self._up_block(256, 128)
            self.up2 = self._up_block(128, 64)
            self.up1 = self._up_block(64, 64)

        elif decoder_mode == "concat":
            # Concatenation sonrası Conv2d
            self.up4 = self._up_block_concat(512, 256)
            self.up3 = self._up_block_concat(256, 128)
            self.up2 = self._up_block_concat(128, 64)
            self.up1 = self._up_block_concat(64, 64)

        else:
            raise ValueError(f"Unsupported decoder_mode: {decoder_mode}")

        # ----- Final conv -----
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.upsample_out = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _up_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Add mode: ConvTranspose + ReLU"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def _up_block_concat(self, in_ch: int, out_ch: int) -> nn.Module:
        """Concat mode: ConvTranspose + Conv2d(BN+ReLU)"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----- Encoder -----
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # ----- Decoder -----
        if self.decoder_mode == "add":
            d4 = self.up4(e5) + e4
            d3 = self.up3(d4) + e3
            d2 = self.up2(d3) + e2
            d1 = self.up1(d2) + e1

        elif self.decoder_mode == "concat":
            d4 = self.up4[0](e5)  # ConvTranspose
            d4 = torch.cat([d4, e4], dim=1)
            d4 = self.up4[1:](d4)

            d3 = self.up3[0](d4)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.up3[1:](d3)

            d2 = self.up2[0](d3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.up2[1:](d2)

            d1 = self.up1[0](d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.up1[1:](d1)

        out = self.final(d1)
        out = self.upsample_out(out)
        return out
