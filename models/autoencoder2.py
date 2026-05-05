# models/autoencoder.py
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d(stride=1) + ReLU 블록 (해상도 유지)"""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AutoEncoder(nn.Module):
    """
    Convolutional AutoEncoder (bottleneck 제거 버전)

    Encoder:
      256 -> 128 (stride=2 1회 다운샘플)
      이후 128x128에서 stride=1 conv block 4개 (해상도 유지)

    Decoder:
      128 -> 256 (ConvTranspose stride=2 1회)
      out_conv로 3채널 복원
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, img_size: int = 256):
        super().__init__()

        if img_size % 2 != 0:
            raise ValueError("img_size는 2의 배수여야 합니다. (stride=2 x1 다운샘플 때문)")

        self.img_size = img_size
        self.base_channels = base_channels

        # -------------------------
        # Encoder: 256 -> 128 (1회 다운샘플)
        # -------------------------
        self.enc_down = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),  # 256->128
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Encoder refine: stride=1 블록 4개 (128 유지)
        # -------------------------
        self.enc_refine = nn.Sequential(
            ConvBlock(base_channels),
            ConvBlock(base_channels),
            ConvBlock(base_channels),
            ConvBlock(base_channels),
        )

        # -------------------------
        # Decoder: 128 -> 256 (1회 업샘플)
        # -------------------------
        self.dec_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 32, kernel_size=4, stride=2, padding=1),  # 128->256
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc_down(x)       # [B, C, 128, 128]
        feat = self.enc_refine(feat)  # [B, C, 128, 128]
        out = self.dec_up(feat)       # [B, 32, 256, 256]
        out = torch.sigmoid(self.out_conv(out))  # [B, 3, 256, 256]
        return out


def build_autoencoder(in_channels: int = 3, base_channels: int = 64, img_size: int = 256):
    return AutoEncoder(in_channels=in_channels, base_channels=base_channels, img_size=img_size)
