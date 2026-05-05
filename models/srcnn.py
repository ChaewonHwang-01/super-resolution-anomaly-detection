import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, scale: int = 2, im_c: int = 3, fn: int = 32, dfn: int = 64):
        super().__init__()
        self.scale = scale

        # SRCNN-style conv blocks
        self.first_part = nn.Conv2d(im_c, dfn, kernel_size=9, stride=1, padding=9 // 2)
        self.mid_part   = nn.Conv2d(dfn, fn,  kernel_size=5, stride=1, padding=5 // 2)
        self.last_part  = nn.Conv2d(fn,  im_c, kernel_size=5, stride=1, padding=5 // 2)

        self.relu = nn.ReLU(inplace=True)

        # weight init (same as your original)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                std = math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel()))
                nn.init.normal_(m.weight.data, mean=0.0, std=std)
                nn.init.zeros_(m.bias.data)

    def forward(self, x, cut: int | None = None):
        # bicubic upsample inside the model
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # optional border crop (kept from original)
        if cut is not None and cut > 0:
            x = x[:, :, cut:-cut, cut:-cut]

        x = self.relu(self.first_part(x))
        x = self.relu(self.mid_part(x))
        x = self.last_part(x)
        return x
