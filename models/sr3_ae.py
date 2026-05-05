import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNNx2(nn.Module):
    """
    Pretrained SRCNN(x2) to upscale 256->512.
    Architecture: bicubic upsample + (9x9)->(5x5)->(5x5) convs.
    """
    def __init__(self, scale: int = 2, in_channels: int = 3, fn: int = 32, dfn: int = 64):
        super().__init__()
        self.scale = scale

        self.conv1 = nn.Conv2d(in_channels, dfn, kernel_size=9, stride=1, padding=9 // 2)
        self.conv2   = nn.Conv2d(dfn, fn,  kernel_size=5, stride=1, padding=5 // 2)
        self.conv3  = nn.Conv2d(fn,  in_channels, kernel_size=5, stride=1, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

        # init (same as original)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                std = math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=0.0, std=std)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 256, 256]
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)  # 256->512
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # [B, 3, 512, 512]
        return x


class AE_512_32(nn.Module):
    """
    AutoEncoder:
      Encoder: 512 -> 256 -> 128 -> 64 -> 32  (4 downsamples, stride=2)
      Decoder: 32  -> 64  -> 128 -> 256 -> 512 (4 upsamples, ConvTranspose2d)
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, max_channels: int = 256):
        super().__init__()

        c1 = min(base_channels, max_channels)          # 64
        c2 = min(base_channels * 2, max_channels)      # 128
        c3 = min(base_channels * 4, max_channels)      # 256
        c4 = min(base_channels * 8, max_channels)      # 256 (clamped by max_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 512->256
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),          # 256->128
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),          # 128->64
            nn.Conv2d(c3, c4, 3, stride=2, padding=1), nn.ReLU(inplace=True),          # 64->32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 32->64
            nn.ConvTranspose2d(c3, c2, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 64->128
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1), nn.ReLU(inplace=True),  # 128->256
            nn.ConvTranspose2d(c1, in_channels, 3, stride=2, padding=1, output_padding=1),                # 256->512
            nn.Sigmoid(),  # 입력이 [0,1]일 때만 유지 (아래 참고)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class SR2_AE(nn.Module):
    """
    Pipeline model:
      input 256 -> SRCNN(x2) -> 512 -> AutoEncoder -> 512 recon
    """
    def __init__(
        self,
        in_channels: int = 3,
        scale: int = 2,
        srcnn_fn: int = 32,
        srcnn_dfn: int = 64,
        ae_base_channels: int = 64,
        ae_max_channels: int = 256,
    ):
        super().__init__()
        self.srcnn = SRCNNx2(scale=scale, in_channels=in_channels, fn=srcnn_fn, dfn=srcnn_dfn)
        self.ae = AE_512_32(in_channels=in_channels, base_channels=ae_base_channels, max_channels=ae_max_channels)

    def forward(self, x_256: torch.Tensor):
        sr_512 = self.srcnn(x_256)
        recon_512 = self.ae(sr_512)
        return sr_512, recon_512