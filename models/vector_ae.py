# models/autoencoder.py
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Vector Bottleneck Convolutional AutoEncoder (img_size=256 기준)

    Encoder:
      256 -> 128 -> 64 -> 32 -> 16 (stride=2 x 4)
      feature: [B, 256, 16, 16]

    Bottleneck:
      [B, 256, 16, 16] -> AdaptiveAvgPool2d(1) -> [B, 256]
      -> latent vector [B, latent_dim]

    Decoder:
      latent vector -> [B, 256, 16, 16] seed -> upsample x4 -> [B, 3, 256, 256]

    목적:
      공간 정보를 강하게 압축하여 정상 manifold에 더 강하게 맞추고,
      이상치 재구성 실패(=MSE 증가, heatmap 선명)를 유도.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, img_size: int = 256):
        super().__init__()

        if img_size % 16 != 0:
            raise ValueError("img_size는 16의 배수여야 합니다. (stride=2 x4 다운샘플 때문)")

        self.img_size = img_size
        self.seed_hw = img_size // 16  # 256 -> 16

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
        # Bottleneck (spatial -> vector)
        # -------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_latent = nn.Linear(256, latent_dim)

        # -------------------------
        # Vector -> spatial seed
        # -------------------------
        self.fc_up = nn.Linear(latent_dim, 256 * self.seed_hw * self.seed_hw)

        # -------------------------
        # Decoder (16->32->64->128->256)
        # -------------------------
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),   # 32->64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),    # 64->128
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),    # 128->256
        )
        self.out_conv = nn.Conv2d(32, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)  # [B,256,16,16] for img_size=256
        pooled = self.pool(feat).flatten(1)  # [B,256]
        z = self.fc_latent(pooled)           # [B,latent_dim]

        seed = self.fc_up(z).view(x.size(0), 256, self.seed_hw, self.seed_hw)  # [B,256,16,16]
        out = self.dec(seed)
        out = torch.sigmoid(self.out_conv(out))
        return out


def build_autoencoder(in_channels: int = 3, latent_dim: int = 128, img_size: int = 256): #더 강하게 하려면 latent_sim 16, 32로 낮춰서 진행
    return AutoEncoder(in_channels=in_channels, latent_dim=latent_dim, img_size=img_size)
