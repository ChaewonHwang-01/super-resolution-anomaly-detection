"""
Pixel-wise difference + Top-k image score test 코드 (둘 다 유지)
- pixel-diff histogram (bin 기반) 결과는 그대로 저장/시각화 유지
- anomaly map: L2 = mean_c (x - x_hat)^2  (pixel-wise)
- image score: top-k mean (top 1%) 또는 max
- 모든 이미지 결과를 CSV/TXT로 저장
- heatmap / triplet / (기존) pixel-diff histogram 저장
- 전체 이미지 score 분포(hist)도 저장
- 체크포인트는 사용자가 직접 지정

[이번 수정]
- pixel histogram 범위: BIN_WIDTH=0.001, BIN_MAX=0.20
- histogram 라벨 소수점: 3자리로 표기 (0.001 단위가 보이게)
- AUROC는 아직 미포함
"""

import os
import glob
import csv
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.vector_ae import AutoEncoder
from utils.file_utils import load_config, ensure_dir


# -------------------------------------------------------
#  Test Dataset
# -------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


def safe_filename_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


# -------------------------------------------------------
# Binning utilities (픽셀-diff histogram 유지용)
# -------------------------------------------------------
def make_bins(bin_width: float = 0.001, max_val: float = 0.10):
    """
    [0, max_val] 구간을 bin_width 간격으로 분할.
    예: max_val=0.20, bin_width=0.001 -> 0.000-0.001, ..., 0.199-0.200
    """
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


def plot_hist_bar(counts: np.ndarray, bin_edges: np.ndarray, title: str, save_path: str):
    # BIN_WIDTH=0.001이면 소수 3자리로 표기해야 구간이 의미있게 보임
    labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges) - 1)]
    x = np.arange(len(counts))

    plt.figure(figsize=(16, 5))
    plt.bar(x, counts)
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Pixel Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------
# Anomaly map + image score (MVTec 스타일)
# -------------------------------------------------------
def make_anomaly_map_l2(img_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    """
    img_np, recon_np: HxWx3 (대개 [0,1])
    return: HxW anomaly map = mean_c (x - x_hat)^2
    """
    diff = img_np - recon_np
    amap = (diff * diff).mean(axis=2)
    return amap


def compute_image_score(amap: np.ndarray, mode: str = "topk", topk_ratio: float = 0.01) -> float:
    """
    amap: HxW anomaly map
    mode: "topk" or "max"
    topk_ratio: 상위 몇 % 평균낼지 (0.01 = top 1%)
    """
    flat = amap.reshape(-1)

    if mode == "max":
        return float(flat.max())

    if mode == "topk":
        k = max(1, int(len(flat) * topk_ratio))
        topk_vals = np.partition(flat, -k)[-k:]
        return float(topk_vals.mean())

    raise ValueError(f"Unknown mode: {mode}")


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


def save_triplet(img: np.ndarray, recon: np.ndarray, amap: np.ndarray, title: str, save_path: str, vmax: float = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    if vmax is None:
        axes[2].imshow(amap, cmap="jet")
    else:
        axes[2].imshow(amap, cmap="jet", vmin=0.0, vmax=vmax)
    axes[2].set_title("Anomaly Map (L2)")
    axes[2].axis("off")

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
def main():
    cfg = load_config("config.yaml")
    paths = cfg["paths"]
    img_size = cfg["train"]["img_size"]

    # =========================
    # 사용자 입력: 체크포인트 경로
    # =========================
    CKPT_PATH = "./results/checkpoints/autoencoder_20260121_212619.pth"
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"지정한 체크포인트 파일이 존재하지 않습니다: {CKPT_PATH}")

    # =========================
    # 사용자 입력: pixel-diff histogram 설정 (이번 요청: 0.001 / 0.2)
    # =========================
    BIN_WIDTH = 0.001
    BIN_MAX = 0.1
    bin_edges = make_bins(bin_width=BIN_WIDTH, max_val=BIN_MAX)

    # =========================
    # 사용자 입력: image score 설정
    # =========================
    IMAGE_SCORE_MODE = "topk"   # "topk" or "max"
    TOPK_RATIO = 0.01           # top 1%

    # =========================
    # (선택) heatmap 대비용 vmax
    # =========================
    HEATMAP_VMAX = None  # 예: 0.02로 고정하면 대비가 훨씬 좋아짐. 모르겠으면 None

    # -------------------------
    # DEVICE
    # -------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[test] device = {device}")
    print(f"[test] checkpoint = {CKPT_PATH}")
    print(f"[test] pixel_hist: bin={BIN_WIDTH}, max={BIN_MAX}")
    print(f"[test] image_score: {IMAGE_SCORE_MODE} (topk_ratio={TOPK_RATIO})")

    # -------------------------
    # Load AE model
    # -------------------------
    model = AutoEncoder().to(device).float()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # -------------------------
    # Transform
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Dataset / Loader
    # -------------------------
    test_dataset = TestDataset(paths["test_dir"], transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------
    # 결과 저장 디렉토리
    # -------------------------
    result_dir = os.path.join(paths["results_dir"], "vector_ae_test_500")
    ensure_dir(result_dir, empty=True)

    # histogram 저장 폴더
    histplot_dir = os.path.join(result_dir, "pixel_diff_hist_plots")
    ensure_dir(histplot_dir, empty=True)

    heatmap_dir = os.path.join(result_dir, "heatmaps")
    triplet_dir = os.path.join(result_dir, "triplets")
    ensure_dir(heatmap_dir, empty=True)
    ensure_dir(triplet_dir, empty=True)

    # -------------------------
    # 루프 결과
    # -------------------------
    results = []
    all_scores = []

    print("\n=== Step 1: pixel histogram + anomaly map + topk score 계산 ===")

    for img, path in test_loader:
        img = img.to(device).float()
        with torch.no_grad():
            recon = model(img)

        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = recon[0].detach().cpu().permute(1, 2, 0).numpy()

        # 1) anomaly map (L2)
        amap = make_anomaly_map_l2(img_np, recon_np)  # HxW

        # 2) image score
        image_score = compute_image_score(amap, mode=IMAGE_SCORE_MODE, topk_ratio=TOPK_RATIO)
        all_scores.append(float(image_score))

        # 3) (유지) pixel-diff histogram (범위: 0~0.2, bin=0.001)
        flat = amap.reshape(-1)
        overflow_cnt = int((flat > BIN_MAX).sum())
        inrange = flat[flat <= BIN_MAX]
        counts = hist_counts(inrange, bin_edges)

        # 통계(참고)
        amap_mean = float(flat.mean())
        amap_max = float(flat.max())
        amap_p95 = float(np.percentile(flat, 95))

        img_path = path[0]
        base = safe_filename_from_path(img_path)

        # 저장
        histplot_path = os.path.join(histplot_dir, f"{base}_pixel_diff_hist.png")
        heatmap_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
        triplet_path = os.path.join(triplet_dir, f"{base}_triplet.png")

        plot_hist_bar(
            counts=counts,
            bin_edges=bin_edges,
            title=f"{base} pixel-diff histogram (bin={BIN_WIDTH}, max={BIN_MAX})",
            save_path=histplot_path
        )

        save_heatmap(
            amap=amap,
            title=f"{base} | score={image_score:.6f} | overflow(>{BIN_MAX})={overflow_cnt}",
            save_path=heatmap_path,
            vmax=HEATMAP_VMAX
        )

        save_triplet(
            img=img_np,
            recon=recon_np,
            amap=amap,
            title=f"{base} | score={image_score:.6f}",
            save_path=triplet_path,
            vmax=HEATMAP_VMAX
        )

        results.append({
            "path": img_path,
            "image_score": float(image_score),
            "amap_mean": amap_mean,
            "amap_p95": amap_p95,
            "amap_max": amap_max,
            "overflow(>BIN_MAX)": overflow_cnt,
        })

    # score 분포 히스토그램 저장
    score_hist_path = os.path.join(result_dir, "image_score_hist.png")
    save_score_hist(
        all_scores,
        title=f"Image Score Histogram ({IMAGE_SCORE_MODE}, topk={TOPK_RATIO})",
        save_path=score_hist_path
    )

    # 결과 정렬 (score 큰 순)
    results_sorted = sorted(results, key=lambda x: x["image_score"], reverse=True)

    # CSV 저장
    csv_path = os.path.join(result_dir, "results.csv")
    fieldnames = ["path", "image_score", "amap_mean", "amap_p95", "amap_max", "overflow(>BIN_MAX)"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    # TXT 저장
    txt_path = os.path.join(result_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Pixel Histogram + Top-k Image Score Results ===\n")
        f.write(f"checkpoint      : {CKPT_PATH}\n")
        f.write(f"test_dir        : {paths['test_dir']}\n")
        f.write(f"img_size        : {img_size}\n\n")

        f.write(f"[pixel histogram]\n")
        f.write(f"  bin_width     : {BIN_WIDTH}\n")
        f.write(f"  bin_max       : {BIN_MAX}\n\n")

        f.write(f"[image score]\n")
        f.write(f"  mode          : {IMAGE_SCORE_MODE}\n")
        f.write(f"  topk_ratio    : {TOPK_RATIO}\n")
        f.write(f"  score_hist_png: {score_hist_path}\n\n")

        f.write("rank\timage_score\tamap_p95\tamap_max\toverflow\tpath\n")
        for i, r in enumerate(results_sorted):
            f.write(
                f"{i:04d}\t{r['image_score']:.6f}\t{r['amap_p95']:.6f}\t{r['amap_max']:.6f}\t"
                f"{r['overflow(>BIN_MAX)']}\t{r['path']}\n"
            )

    print("\n=== 완료 ===")
    print(f"[saved] result_dir          : {result_dir}")
    print(f"[saved] results.csv         : {csv_path}")
    print(f"[saved] results.txt         : {txt_path}")
    print(f"[saved] image_score_hist    : {score_hist_path}")
    print(f"[saved] pixel hist dir      : {histplot_dir}")
    print(f"[saved] heatmaps dir        : {heatmap_dir}")
    print(f"[saved] triplets dir        : {triplet_dir}")


if __name__ == "__main__":
    main()
