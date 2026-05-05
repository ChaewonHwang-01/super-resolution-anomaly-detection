import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """x,y: [1,C,H,W] 또는 [C,H,W] 텐서."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    return F.mse_loss(x, y).item()


def compute_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """SSIM 계산 (0~1). x,y: [1,C,H,W] or [C,H,W], 값 [0,1]."""
    if x.dim() == 4:
        x = x[0]
    if y.dim() == 4:
        y = y[0]

    # [C,H,W] → [H,W,C]
    x_np = x.detach().cpu().permute(1, 2, 0).numpy()
    y_np = y.detach().cpu().permute(1, 2, 0).numpy()

    return ssim(x_np, y_np, channel_axis=2, data_range=1.0)
