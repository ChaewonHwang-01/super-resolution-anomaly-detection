import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from .file_utils import ensure_dir


def plot_loss_curve(loss_list, save_path: str, title: str = "Training Loss"):
    ensure_dir(os.path.dirname(save_path), empty=False)
    plt.figure()
    plt.plot(range(1, len(loss_list) + 1), loss_list, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_recon_with_heatmap(input_tensor: torch.Tensor,
                            recon_tensor: torch.Tensor,
                            save_path: str,
                            suptitle: str = ""):
    """입력 / 복원 / 오차 heatmap 한 장으로 저장."""
    ensure_dir(os.path.dirname(save_path), empty=False)

    if input_tensor.dim() == 4:
        input_tensor = input_tensor[0]
    if recon_tensor.dim() == 4:
        recon_tensor = recon_tensor[0]

    inp = input_tensor.detach().cpu().permute(1, 2, 0).numpy()
    rec = recon_tensor.detach().cpu().permute(1, 2, 0).numpy()
    diff = np.abs(inp - rec).mean(axis=2)

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(inp)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(rec)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="hot")
    plt.title("Error Heatmap")
    plt.axis("off")

    if suptitle:
        plt.suptitle(suptitle)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
