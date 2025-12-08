import torch
import torch.nn as nn
import torch.nn.functional as F

class UND_Block(nn.Module):
    """
    U-shape Noise Denoising block (UND block)
    in/out: [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # [B,C,H,W]
        x2 = self.enc2(self.pool1(x1))  # [B,C,H/2,W/2]
        x3 = self.enc3(self.pool2(x2))  # [B,C,H/4,W/4]
        x4 = self.enc4(self.pool3(x3))  # [B,C,H/8,W/8]

        # Decoder
        d3 = self.up3(x4) + x3
        d3 = self.dec3(d3)
        d2 = self.up2(d3) + x2
        d2 = self.dec2(d2)
        d1 = self.up1(d2) + x1
        d1 = self.dec1(d1)

        out = x + d1
        return out  # [B,C,H,W]