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
        self.conv2 = nn.Conv2d(dfn, fn, kernel_size=5, stride=1, padding=5 // 2)
        self.conv3 = nn.Conv2d(fn, in_channels, kernel_size=5, stride=1, padding=5 // 2)

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


class VAE_512_32(nn.Module):
    """
    Variational AutoEncoder
    Input:  [B, 3, 512, 512]
    Encode: 512 -> 256 -> 128 -> 64 -> 32
    Latent: feature map based latent (mu, logvar) at [B, latent_channels, 32, 32]
    Decode: 32 -> 64 -> 128 -> 256 -> 512
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 256,
        latent_channels: int = 256,
    ):
        super().__init__()

        c1 = min(base_channels, max_channels)          # 64
        c2 = min(base_channels * 2, max_channels)      # 128
        c3 = min(base_channels * 4, max_channels)      # 256
        c4 = min(base_channels * 8, max_channels)      # 256

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1),  # 512->256
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),           # 256->128
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),           # 128->64
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),           # 64->32
            nn.ReLU(inplace=True),
        )

        # latent distribution
        self.mu_layer = nn.Conv2d(c4, latent_channels, kernel_size=3, stride=1, padding=1)
        self.logvar_layer = nn.Conv2d(c4, latent_channels, kernel_size=3, stride=1, padding=1)

        # map sampled z back to decoder input channels
        self.latent_to_dec = nn.Sequential(
            nn.Conv2d(latent_channels, c4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 3, stride=2, padding=1, output_padding=1),  # 32->64
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 3, stride=2, padding=1, output_padding=1),  # 64->128
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1),  # 128->256
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(c1, in_channels, 3, stride=2, padding=1, output_padding=1),  # 256->512
            nn.Sigmoid(),  # input이 [0,1] 범위일 때 사용
        )

    def encode(self, x: torch.Tensor):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)  # [B, c4, 32, 32]

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor):
        h = self.latent_to_dec(z)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        out = self.dec4(h)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


class SR2_VAE(nn.Module):
    """
    Pipeline model:
      input 256 -> SRCNN(x2) -> 512 -> VAE -> 512 recon
    """
    def __init__(
        self,
        in_channels: int = 3,
        scale: int = 2,
        srcnn_fn: int = 32,
        srcnn_dfn: int = 64,
        vae_base_channels: int = 64,
        vae_max_channels: int = 256,
        vae_latent_channels: int = 256,
    ):
        super().__init__()
        self.srcnn = SRCNNx2(
            scale=scale,
            in_channels=in_channels,
            fn=srcnn_fn,
            dfn=srcnn_dfn
        )
        self.vae = VAE_512_32(
            in_channels=in_channels,
            base_channels=vae_base_channels,
            max_channels=vae_max_channels,
            latent_channels=vae_latent_channels
        )

    def forward(self, x_256: torch.Tensor):
        sr_512 = self.srcnn(x_256)
        recon_512, mu, logvar = self.vae(sr_512)
        return sr_512, recon_512, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_type: str = "mse",
    kld_weight: float = 1e-4,
):
    """
    recon_x: VAE output
    x: target
    mu, logvar: latent distribution params
    recon_type: "mse" or "l1"
    kld_weight: KL loss weight (beta)
    """
    if recon_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    elif recon_type == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_type: {recon_type}")

    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss
