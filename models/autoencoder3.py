# models/autoencoder.py
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Spatial Bottleneck Convolutional AutoEncoder (img_size=256 기준)

    Encoder:
      256 -> 128 -> 64 -> 32 -> 16 (stride=2 x 4)
      feature: [B, 256, 16, 16]

    Bottleneck (spatial):
      [B, 256, 16, 16] -> [B, latent_ch, 16, 16]  (1x1 conv)

    Decoder:
      16 -> 32 -> 64 -> 128 -> 256

    장점:
      - (1x1) 벡터로 완전 붕괴시키지 않아 공간 정보를 유지 → "덜 강한" 압축
      - latent_ch로 병목 강도를 쉽게 조절
    """

    def __init__(self, in_channels: int = 3, latent_ch: int = 128, img_size: int = 256):
        super().__init__()

        if img_size % 16 != 0:
            raise ValueError("img_size는 16의 배수여야 합니다. (stride=2 x4 다운샘플 때문)")

        self.img_size = img_size

        # -------------------------
        # Encoder
        # -------------------------
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 256->128
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),            # 128->64
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),           # 64->32
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),          # 32->16
        )

        # -------------------------
        # Spatial bottleneck (채널만 축소)
        # -------------------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, latent_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # -------------------------
        # Decoder (16->32->64->128->256)
        # bottleneck의 채널(latent_ch)에서 시작
        # -------------------------
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),         # 32->64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),          # 64->128
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),          # 128->256
        )
        self.out_conv = nn.Conv2d(32, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)               # [B,256,16,16]
        z = self.bottleneck(feat)        # [B,latent_ch,16,16]
        out = self.dec(z)                # [B,32,256,256]
        out = torch.sigmoid(self.out_conv(out))
        return out


def build_autoencoder(in_channels: int = 3, latent_ch: int = 128, img_size: int = 256):
    """
    덜 강하게: latent_ch를 키우세요. (예: 128 -> 192/256)
    더 강하게: latent_ch를 줄이세요. (예: 128 -> 64/32)
    """
    return AutoEncoder(in_channels=in_channels, latent_ch=latent_ch, img_size=img_size)
