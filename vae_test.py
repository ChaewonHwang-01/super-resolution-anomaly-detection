"""
SR+VAE Pixel-wise difference + Image score test (시각화 전부 유지)

- Input: 256 이미지 (재귀 탐색)
- SR: SR2_VAE 내부 SRCNN(x2) -> sr_512
- VAE: sr_512 -> recon_512
- anomaly map: L2 = mean_c (sr - recon)^2
- pixel-diff histogram: mean_c |sr - recon|
- overlay: amap >= threshold (+ dilation)
- image score: top-k mean (top 1%) or max
- CSV/TXT 저장, heatmap/quad/pixel-diff hist 저장, score hist 저장
- 깨진 이미지 자동 스킵 + bad_images.txt 저장
- AUROC(image-level) + ROC curve 저장
"""

import os
import csv
import math
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.vae import SR2_VAE
from utils.file_utils import load_config, ensure_dir


# =======================================================
#  AUROC / ROC utilities (sklearn 있으면 사용, 없으면 fallback)
# =======================================================
def _try_import_sklearn():
    try:
        from sklearn.metrics import roc_auc_score, roc_curve, auc
        return roc_auc_score, roc_curve, auc
    except Exception:
        return None, None, None


def compute_auroc_fallback(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC = P(score_pos > score_neg) + 0.5*P(equal)
    rank 기반 (Mann–Whitney U) 계산
    """
    scores = scores.astype(np.float64)
    labels = labels.astype(np.int32)

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # tie 평균 랭크
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    pos_ranks_sum = ranks[labels == 1].sum()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    U = pos_ranks_sum - n_pos * (n_pos + 1) / 2.0
    auc_val = U / (n_pos * n_neg)
    return float(auc_val)


def roc_curve_fallback(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    thresholds를 unique score로 sweep해서 ROC curve (FPR, TPR) 생성
    """
    scores = scores.astype(np.float64)
    labels = labels.astype(np.int32)

    if labels.max() == labels.min():
        return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    thresholds = np.unique(scores)[::-1]
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())

    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        pred_pos = scores >= thr
        TP = int(np.logical_and(pred_pos, labels == 1).sum())
        FP = int(np.logical_and(pred_pos, labels == 0).sum())
        tpr = TP / P if P > 0 else 0.0
        fpr = FP / N if N > 0 else 0.0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr = np.array([0.0] + fpr_list + [1.0], dtype=np.float64)
    tpr = np.array([0.0] + tpr_list + [1.0], dtype=np.float64)
    return fpr, tpr


def plot_roc_curve_and_get_auroc(scores: List[float], labels: List[int], save_path: str) -> float:
    """
    ROC curve 저장 + AUROC 반환
    """
    scores_np = np.array(scores, dtype=np.float64)
    labels_np = np.array(labels, dtype=np.int32)

    roc_auc_score, roc_curve, auc = _try_import_sklearn()
    if roc_auc_score is not None:
        auroc = float(roc_auc_score(labels_np, scores_np))
        fpr, tpr, _ = roc_curve(labels_np, scores_np)
        auc_label = float(auc(fpr, tpr))
    else:
        auroc = compute_auroc_fallback(scores_np, labels_np)
        fpr, tpr = roc_curve_fallback(scores_np, labels_np)
        auc_label = float(np.trapz(tpr, fpr))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc_label:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Image-level)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return auroc


# =======================================================
#  Label inference
# =======================================================
def infer_label_from_path(path: str) -> int:
    p = path.replace("\\", "/").lower()
    if "/good/" in p or p.endswith("/good") or p.endswith("/good/"):
        return 0
    return 1


# -------------------------------------------------------
#  Recursive image finder
# -------------------------------------------------------
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".gif", ".ppm", ".pgm"
}


def find_image_paths_recursive(root: str, exts=IMAGE_EXTS) -> List[str]:
    rootp = Path(root)
    if not rootp.exists():
        raise FileNotFoundError(f"test_dir does not exist: {root}")
    paths = []
    for p in rootp.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
    paths.sort()
    return paths


# -------------------------------------------------------
#  Test Dataset
# -------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root: str, transform=None, recursive: bool = True):
        self.transform = transform
        self.root = root
        if recursive:
            self.paths = find_image_paths_recursive(root)
        else:
            self.paths = sorted([str(p) for p in Path(root).glob("*") if p.is_file()])
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            return None, path, str(e)
        if self.transform:
            img = self.transform(img)
        return img, path, None


def collate_skip_bad(batch):
    imgs, paths, errs = [], [], []
    for img, path, err in batch:
        if img is None:
            errs.append((path, err))
            continue
        imgs.append(img)
        paths.append(path)
    if len(imgs) == 0:
        return None, None, errs
    imgs = torch.stack(imgs, dim=0)
    return imgs, paths, errs


def safe_filename_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


# -------------------------------------------------------
# Binning utilities
# -------------------------------------------------------
def make_bins(bin_width: float = 0.001, max_val: float = 0.20):
    if bin_width <= 0:
        raise ValueError("bin_width는 0보다 커야 합니다.")
    if max_val <= 0:
        raise ValueError("max_val은 0보다 커야 합니다.")
    n_bins = int(math.ceil(max_val / bin_width))
    edges = np.linspace(0.0, n_bins * bin_width, n_bins + 1, dtype=np.float32)
    return edges


def hist_counts(values: np.ndarray, bin_edges: np.ndarray):
    counts, _ = np.histogram(values, bins=bin_edges)
    return counts


def plot_hist_bar(counts: np.ndarray, bin_edges: np.ndarray, title: str, save_path: str, tick_step: int = 1):
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges) - 1)]
    x = np.arange(len(counts))
    plt.figure(figsize=(18, 6))
    plt.bar(x, counts)

    tick_step = max(1, int(tick_step))
    tick_idx = x[::tick_step]
    tick_labels = [labels[i] for i in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=60, ha="right")

    plt.ylabel("Pixel Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Maps + Scores
# -------------------------------------------------------
def make_anomaly_map_l2(img_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    diff = img_np - recon_np
    amap = (diff * diff).mean(axis=2)
    return amap


def make_pixel_diff_map_abs(img_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    diff = np.abs(img_np - recon_np)
    dmap = diff.mean(axis=2)
    return dmap


def compute_image_score(amap: np.ndarray, mode: str = "topk", topk_ratio: float = 0.01) -> float:
    flat = amap.reshape(-1)
    if mode == "max":
        return float(flat.max())
    if mode == "topk":
        k = max(1, int(len(flat) * topk_ratio))
        topk_vals = np.partition(flat, -k)[-k:]
        return float(topk_vals.mean())
    raise ValueError(f"Unknown mode: {mode}")


def threshold_mask_from_amap(amap: np.ndarray, threshold: float = 0.003) -> np.ndarray:
    return amap >= float(threshold)


def binary_dilation_np(mask: np.ndarray, iterations: int = 1, kernel_size: int = 3) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size는 3 이상의 홀수여야 합니다. 예: 3,5,7")
    pad = kernel_size // 2
    out = mask.copy()

    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
        dil = np.zeros_like(out, dtype=bool)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                dil |= padded[dy:dy + out.shape[0], dx:dx + out.shape[1]]
        out = dil
    return out


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------
def save_heatmap(amap: np.ndarray, title: str, save_path: str, vmax: float = None):
    plt.figure(figsize=(6, 6))
    if vmax is None:
        plt.imshow(amap, cmap="jet")
    else:
        plt.imshow(amap, cmap="jet", vmin=0.0, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _to_rgb_for_vis(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unexpected image shape: {img.shape}")


def make_red_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img_rgb = _to_rgb_for_vis(img)
    overlay = img_rgb.copy()
    overlay[mask, 0] = 1.0
    overlay[mask, 1] = overlay[mask, 1] * (1 - alpha)
    overlay[mask, 2] = overlay[mask, 2] * (1 - alpha)
    blended = img_rgb * (1 - alpha) + overlay * alpha
    return np.clip(blended, 0.0, 1.0)


def save_quad(sr: np.ndarray,
              recon: np.ndarray,
              amap: np.ndarray,
              overlay: np.ndarray,
              title: str,
              save_path: str,
              vmax: float = None):
    sr_vis = _to_rgb_for_vis(sr)
    recon_vis = _to_rgb_for_vis(recon)
    overlay_vis = _to_rgb_for_vis(overlay)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(sr_vis)
    axes[0].set_title("SR (SRCNN output)")
    axes[0].axis("off")

    axes[1].imshow(recon_vis)
    axes[1].set_title("VAE Reconstructed")
    axes[1].axis("off")

    if vmax is None:
        axes[2].imshow(amap, cmap="jet")
    else:
        axes[2].imshow(amap, cmap="jet", vmin=0.0, vmax=vmax)
    axes[2].set_title("Anomaly Map (L2)")
    axes[2].axis("off")

    axes[3].imshow(overlay_vis)
    axes[3].set_title("Loss>=Threshold (Dilated Overlay)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_score_hist(scores: list, title: str, save_path: str, bins: int = 50):
    arr = np.array(scores, dtype=np.float32)
    plt.figure(figsize=(10, 4))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("Image Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
def _load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=True)


def main():
    cfg = load_config("config.yaml")
    paths = cfg["paths"]
    model_cfg = cfg["model"]
    img_size = cfg["train"]["img_size"]

    # =========================
    # 체크포인트
    # =========================
    SRCNN_CKPT = model_cfg.get("srcnn_pretrained_path", "./weights/train_srcnn.pth")
    VAE_CKPT = cfg.get("test", {}).get("checkpoint_path", "").strip()
    LOAD_MODE = cfg.get("train", {}).get("save_mode", "vae_only")   # "vae_only" or "full"

    if not os.path.isfile(SRCNN_CKPT):
        raise FileNotFoundError(f"SRCNN checkpoint not found: {SRCNN_CKPT}")

    if not VAE_CKPT:
        ckpt_dir = os.path.join(paths["results_dir"], "checkpoints")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        ckpt_candidates = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        )
        if not ckpt_candidates:
            raise FileNotFoundError(f"No .pth checkpoint found in: {ckpt_dir}")
        VAE_CKPT = ckpt_candidates[-1]

    if not os.path.isfile(VAE_CKPT):
        raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CKPT}")

    # =========================
    # histogram / score / overlay 설정
    # =========================
    BIN_WIDTH = 0.001
    BIN_MAX = 0.20
    bin_edges = make_bins(bin_width=BIN_WIDTH, max_val=BIN_MAX)

    IMAGE_SCORE_MODE = "topk"
    SCORE_TOPK_RATIO = 0.01

    LOSS_THRESHOLD = 0.005
    DILATE_ITERS = 2
    DILATE_KSIZE = 3

    HEATMAP_VMAX = None
    OVERLAY_ALPHA = 0.35
    HIST_TICK_STEP = 1
    RECURSIVE_SEARCH = True

    # -------------------------
    # DEVICE
    # -------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # -------------------------
    # Model load
    # -------------------------
    in_channels = int(model_cfg.get("in_channels", 3))
    model = SR2_VAE(
        in_channels=in_channels,
        scale=int(model_cfg.get("scale", 2)),
        srcnn_fn=int(model_cfg.get("srcnn_fn", 32)),
        srcnn_dfn=int(model_cfg.get("srcnn_dfn", 64)),
        vae_base_channels=int(model_cfg.get("vae_base_channels", 64)),
        vae_max_channels=int(model_cfg.get("vae_max_channels", 256)),
        vae_latent_channels=int(model_cfg.get("vae_latent_channels", 256)),
    ).to(device).float()

    if LOAD_MODE == "vae_only":
        _load_state_dict_flexible(model.srcnn, SRCNN_CKPT, device)
        _load_state_dict_flexible(model.vae, VAE_CKPT, device)
    elif LOAD_MODE == "full":
        _load_state_dict_flexible(model, VAE_CKPT, device)
    else:
        raise ValueError(f"Unsupported LOAD_MODE: {LOAD_MODE}. Use 'vae_only' or 'full'.")

    model.eval()

    print(f"[test] device            = {device}")
    print(f"[test] in_channels       = {in_channels}")
    print(f"[test] SRCNN ckpt        = {SRCNN_CKPT}")
    print(f"[test] VAE ckpt          = {VAE_CKPT} (mode={LOAD_MODE})")
    print(f"[test] test_dir          = {paths['test_dir']}")
    print(f"[test] img_size(in)      = {img_size}")
    print(f"[test] overlay thr(amap) = {LOSS_THRESHOLD}, dilate(iters={DILATE_ITERS}, k={DILATE_KSIZE})")

    # -------------------------
    # Transform
    # -------------------------
    if in_channels == 1:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    # -------------------------
    # Dataset / Loader
    # -------------------------
    test_dataset = TestDataset(paths["test_dir"], transform, recursive=RECURSIVE_SEARCH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_skip_bad,
        num_workers=0,
    )

    # -------------------------
    # 결과 저장 디렉토리
    # -------------------------
    result_dir = os.path.join(paths["results_dir"], "sr_vae_test_zipper")
    ensure_dir(result_dir, empty=True)

    histplot_dir = os.path.join(result_dir, "pixel_diff_hist_plots")
    heatmap_dir = os.path.join(result_dir, "heatmaps")
    quad_dir = os.path.join(result_dir, "quads")
    ensure_dir(histplot_dir, empty=True)
    ensure_dir(heatmap_dir, empty=True)
    ensure_dir(quad_dir, empty=True)

    # -------------------------
    # 루프 결과
    # -------------------------
    results = []
    all_scores: List[float] = []
    all_labels: List[int] = []
    bad_images: List[Tuple[str, str]] = []

    for imgs, img_paths, errs in test_loader:
        if errs:
            for p, e in errs:
                bad_images.append((p, e))
        if imgs is None:
            continue

        x_256 = imgs.to(device).float()
        img_path = img_paths[0]

        label = infer_label_from_path(img_path)
        all_labels.append(int(label))

        with torch.no_grad():
            sr_512, recon_512, mu, logvar = model(x_256)

        H = min(sr_512.shape[-2], recon_512.shape[-2])
        W = min(sr_512.shape[-1], recon_512.shape[-1])
        sr_512 = sr_512[:, :, :H, :W]
        recon_512 = recon_512[:, :, :H, :W]

        sr_np = sr_512[0].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = recon_512[0].detach().cpu().permute(1, 2, 0).numpy()

        # anomaly map / score
        amap = make_anomaly_map_l2(sr_np, recon_np)
        image_score = compute_image_score(amap, mode=IMAGE_SCORE_MODE, topk_ratio=SCORE_TOPK_RATIO)
        all_scores.append(float(image_score))

        # threshold + dilation + overlay
        loss_mask = threshold_mask_from_amap(amap, threshold=LOSS_THRESHOLD)
        if DILATE_ITERS > 0:
            loss_mask = binary_dilation_np(loss_mask, iterations=DILATE_ITERS, kernel_size=DILATE_KSIZE)
        overlay_np = make_red_overlay(sr_np, loss_mask, alpha=OVERLAY_ALPHA)

        # pixel-diff histogram
        dmap = make_pixel_diff_map_abs(sr_np, recon_np)
        flat_d = dmap.reshape(-1)
        overflow_cnt = int((flat_d > BIN_MAX).sum())
        inrange = flat_d[flat_d <= BIN_MAX]
        counts = hist_counts(inrange, bin_edges)

        # stats
        flat_a = amap.reshape(-1)
        amap_mean = float(flat_a.mean())
        amap_max = float(flat_a.max())
        amap_p95 = float(np.percentile(flat_a, 95))

        thr_cnt = int(loss_mask.sum())
        thr_ratio = float(thr_cnt) / float(loss_mask.size)

        # optional: VAE latent stats
        mu_mean = float(mu.mean().item())
        mu_std = float(mu.std().item())
        logvar_mean = float(logvar.mean().item())
        logvar_std = float(logvar.std().item())

        base = safe_filename_from_path(img_path)

        histplot_path = os.path.join(histplot_dir, f"{base}_pixel_diff_hist.png")
        heatmap_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
        quad_path = os.path.join(quad_dir, f"{base}_quad.png")

        plot_hist_bar(
            counts=counts,
            bin_edges=bin_edges,
            title=f"{base} pixel-diff(|sr-recon|) hist (bin={BIN_WIDTH}, max={BIN_MAX})",
            save_path=histplot_path,
            tick_step=HIST_TICK_STEP,
        )

        save_heatmap(
            amap=amap,
            title=f"{base} | label={label} | score={image_score:.6f} | dil_thr(amap>={LOSS_THRESHOLD})={thr_ratio*100:.2f}%",
            save_path=heatmap_path,
            vmax=HEATMAP_VMAX
        )

        quad_title = (
            f"{base} | label={label} | score={image_score:.6f} | "
            f"amap>= {LOSS_THRESHOLD} (dilate iters={DILATE_ITERS}, k={DILATE_KSIZE})"
        )

        save_quad(
            sr=sr_np,
            recon=recon_np,
            amap=amap,
            overlay=overlay_np,
            title=quad_title,
            save_path=quad_path,
            vmax=HEATMAP_VMAX
        )

        results.append({
            "path": img_path,
            "label": int(label),
            "image_score": float(image_score),
            "amap_mean": amap_mean,
            "amap_p95": amap_p95,
            "amap_max": amap_max,
            "dil_thr_ratio(%)": thr_ratio * 100.0,
            "pixel_diff_overflow(>BIN_MAX)": overflow_cnt,
            "mu_mean": mu_mean,
            "mu_std": mu_std,
            "logvar_mean": logvar_mean,
            "logvar_std": logvar_std,
        })

    # score hist 저장
    score_hist_path = os.path.join(result_dir, "image_score_hist.png")
    save_score_hist(
        all_scores,
        title=f"Image Score Histogram ({IMAGE_SCORE_MODE}, topk={SCORE_TOPK_RATIO})",
        save_path=score_hist_path
    )

    # AUROC + ROC curve 저장
    roc_path = os.path.join(result_dir, "roc_curve.png")
    if len(all_scores) > 0 and len(all_scores) == len(all_labels) and len(set(all_labels)) >= 2:
        auroc_val = plot_roc_curve_and_get_auroc(all_scores, all_labels, save_path=roc_path)
    else:
        auroc_val = float("nan")

    # sort
    results_sorted = sorted(results, key=lambda x: x["image_score"], reverse=True)

    # CSV
    csv_path = os.path.join(result_dir, "results.csv")
    fieldnames = [
        "path",
        "label",
        "image_score",
        "amap_mean",
        "amap_p95",
        "amap_max",
        "dil_thr_ratio(%)",
        "pixel_diff_overflow(>BIN_MAX)",
        "mu_mean",
        "mu_std",
        "logvar_mean",
        "logvar_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    # TXT
    txt_path = os.path.join(result_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== SR+VAE Pixel-diff Histogram + Dilated Threshold Overlay + Image Score Results (+ AUROC) ===\n")
        f.write(f"SRCNN checkpoint : {SRCNN_CKPT}\n")
        f.write(f"VAE checkpoint   : {VAE_CKPT}\n")
        f.write(f"LOAD_MODE        : {LOAD_MODE}\n")
        f.write(f"test_dir         : {paths['test_dir']}\n")
        f.write(f"recursive        : {RECURSIVE_SEARCH}\n")
        f.write(f"img_size(in)     : {img_size}\n\n")

        f.write("[labels]\n")
        f.write("  rule           : path contains '/good/' => label=0 else label=1\n\n")

        f.write("[pixel histogram]\n")
        f.write("  metric         : mean_c |sr - recon|\n")
        f.write(f"  bin_width      : {BIN_WIDTH}\n")
        f.write(f"  bin_max        : {BIN_MAX}\n\n")

        f.write("[anomaly map]\n")
        f.write("  metric         : mean_c (sr - recon)^2\n")
        f.write(f"  overlay_thr    : {LOSS_THRESHOLD}\n")
        f.write(f"  dilation       : iters={DILATE_ITERS}, ksize={DILATE_KSIZE}\n\n")

        f.write("[image score]\n")
        f.write(f"  mode           : {IMAGE_SCORE_MODE}\n")
        f.write(f"  topk_ratio     : {SCORE_TOPK_RATIO}\n")
        f.write(f"  score_hist_png : {score_hist_path}\n\n")

        f.write("[AUROC]\n")
        f.write(f"  image-level AUROC: {auroc_val}\n")
        if not (isinstance(auroc_val, float) and np.isnan(auroc_val)):
            f.write(f"  roc_curve_png    : {roc_path}\n")
        else:
            f.write("  roc_curve_png    : (skipped; only one class present or no valid samples)\n")
        f.write("\n")

        if bad_images:
            f.write("[bad images skipped]\n")
            for p, e in bad_images:
                f.write(f"- {p}\t{e}\n")
            f.write("\n")

        f.write("rank\tlabel\timage_score\tamap_p95\tamap_max\tdil_thr_ratio(%)\toverflow\tmu_mean\tmu_std\tlogvar_mean\tlogvar_std\tpath\n")
        for i, r in enumerate(results_sorted):
            f.write(
                f"{i:04d}\t{r['label']}\t{r['image_score']:.6f}\t{r['amap_p95']:.6f}\t{r['amap_max']:.6f}\t"
                f"{r['dil_thr_ratio(%)']:.2f}\t{r['pixel_diff_overflow(>BIN_MAX)']}\t"
                f"{r['mu_mean']:.6f}\t{r['mu_std']:.6f}\t{r['logvar_mean']:.6f}\t{r['logvar_std']:.6f}\t"
                f"{r['path']}\n"
            )

    # bad images
    if bad_images:
        bad_path = os.path.join(result_dir, "bad_images.txt")
        with open(bad_path, "w", encoding="utf-8") as f:
            for p, e in bad_images:
                f.write(f"{p}\t{e}\n")

    print("\n=== 완료 ===")
    print(f"[saved] result_dir       : {result_dir}")
    print(f"[saved] results.csv      : {csv_path}")
    print(f"[saved] results.txt      : {txt_path}")
    print(f"[saved] image_score_hist : {score_hist_path}")
    if not (isinstance(auroc_val, float) and np.isnan(auroc_val)):
        print(f"[saved] roc_curve        : {roc_path}")
        print(f"[info ] AUROC            : {auroc_val:.6f}")
    else:
        print(f"[info ] AUROC            : NaN (only one class present or no samples)")
    print(f"[saved] pixel hist dir   : {histplot_dir}")
    print(f"[saved] heatmaps dir     : {heatmap_dir}")
    print(f"[saved] quads dir        : {quad_dir}")
    print(f"[info ] total images     : {len(test_dataset)}")
    print(f"[info ] processed        : {len(results)}")
    if bad_images:
        print(f"[info ] bad skipped      : {len(bad_images)}")


if __name__ == "__main__":
    main()
