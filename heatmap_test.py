"""
Pixel-wise difference + Image score test (둘 다 유지)
- pixel-diff histogram (bin 기반) 결과 저장/시각화 유지
- anomaly map: L2 = mean_c (x - x_hat)^2 (pixel-wise)
- image score: top-k mean (top 1%) 또는 max
- 모든 이미지 결과를 CSV/TXT로 저장
- heatmap / (4-panel) quad / (기존) pixel-diff histogram 저장
- 전체 이미지 score 분포(hist)도 저장
- 체크포인트는 사용자가 직접 지정

[이번 수정]
- pixel histogram 범위: BIN_WIDTH=0.001, BIN_MAX=0.20
- histogram 라벨 소수점: 3자리로 표기 (0.001 단위가 보이게)
- "로스(=anomaly map 값) >= 0.01" 인 픽셀을 빨갛게 표시 (threshold overlay)
- triplet 대신 4장(Original / Reconstructed / Anomaly Map / Threshold Overlay) 저장
- AUROC는 아직 미포함

[추가 수정: 폴더 안의 폴더까지 전부 들어가서 이미지 찾기]
- 기존 glob(root/*.*) 대신 재귀 탐색(root 이하 전부)으로 이미지 파일 경로 수집
- 허용 확장자: jpg/jpeg/png/bmp/tif/tiff/webp/gif/ppm/pgm (필요시 수정)
- 비이미지/깨진 파일은 자동 스킵(로그 출력)

[주의]
- pixel-diff histogram은 "진짜 pixel-wise difference"로 유지: |x - x_hat| (채널 평균)
- overlay는 anomaly map(threshold on mean_c (x-x_hat)^2 ) 기반
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

from models.autoencoder3 import AutoEncoder
from utils.file_utils import load_config, ensure_dir


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
#  Test Dataset (recursive)
# -------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root: str, transform=None, recursive: bool = True):
        self.transform = transform
        self.root = root

        if recursive:
            self.paths = find_image_paths_recursive(root)
        else:
            # fallback: non-recursive
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
            # DataLoader에서 이 샘플을 "스킵"하려면 None 반환하고 collate에서 거르면 됨
            return None, path, str(e)

        if self.transform:
            img = self.transform(img)
        return img, path, None


def collate_skip_bad(batch):
    """
    batch: List[(img_or_None, path, err_or_None)]
    - img가 None(깨진 이미지)이면 제외하고 batch 구성
    - 전부 깨졌으면 빈 batch 반환 -> 루프에서 continue 처리
    """
    imgs = []
    paths = []
    errs = []
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
# Binning utilities (픽셀-diff histogram 유지용)
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


def threshold_mask_from_amap(amap: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    return amap >= float(threshold)


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


def make_red_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    overlay = img.copy()

    overlay[mask, 0] = 1.0
    overlay[mask, 1] = overlay[mask, 1] * (1 - alpha)
    overlay[mask, 2] = overlay[mask, 2] * (1 - alpha)

    blended = img * (1 - alpha) + overlay * alpha
    return np.clip(blended, 0.0, 1.0)


def save_quad(img: np.ndarray,
              recon: np.ndarray,
              amap: np.ndarray,
              overlay: np.ndarray,
              title: str,
              save_path: str,
              vmax: float = None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(np.clip(img, 0.0, 1.0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(recon, 0.0, 1.0))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    if vmax is None:
        axes[2].imshow(amap, cmap="jet")
    else:
        axes[2].imshow(amap, cmap="jet", vmin=0.0, vmax=vmax)
    axes[2].set_title("Anomaly Map (L2)")
    axes[2].axis("off")

    axes[3].imshow(np.clip(overlay, 0.0, 1.0))
    axes[3].set_title("Loss>=Threshold (Overlay)")
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
def main():
    cfg = load_config("config.yaml")
    paths = cfg["paths"]
    img_size = cfg["train"]["img_size"]

    # =========================
    # 사용자 입력: 체크포인트 경로
    # =========================
    CKPT_PATH = "./results/checkpoints/autoencoder_20260123_032640.pth"
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"지정한 체크포인트 파일이 존재하지 않습니다: {CKPT_PATH}")

    # =========================
    # 사용자 입력: pixel-diff histogram 설정
    # =========================
    BIN_WIDTH = 0.001
    BIN_MAX = 0.20
    bin_edges = make_bins(bin_width=BIN_WIDTH, max_val=BIN_MAX)

    # =========================
    # 사용자 입력: image score 설정 (amap 기반)
    # =========================
    IMAGE_SCORE_MODE = "topk"   # "topk" or "max"
    SCORE_TOPK_RATIO = 0.01     # top 1%

    # =========================
    # 사용자 입력: overlay threshold (로스 0.01 이상 빨강)
    # =========================
    LOSS_THRESHOLD = 0.01

    # =========================
    # (선택) heatmap 대비용 vmax
    # =========================
    HEATMAP_VMAX = None  # 예: 0.02로 고정하면 대비가 좋아짐. 모르겠으면 None

    # =========================
    # overlay alpha
    # =========================
    OVERLAY_ALPHA = 0.35

    # =========================
    # histogram tick step (라벨 너무 많으면 5~10 추천)
    # =========================
    HIST_TICK_STEP = 1

    # =========================
    # 폴더 재귀 탐색 ON
    # =========================
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

    print(f"[test] device = {device}")
    print(f"[test] checkpoint = {CKPT_PATH}")
    print(f"[test] pixel_hist: bin={BIN_WIDTH}, max={BIN_MAX}")
    print(f"[test] image_score: {IMAGE_SCORE_MODE} (topk_ratio={SCORE_TOPK_RATIO})")
    print(f"[test] overlay: loss(amap) >= {LOSS_THRESHOLD}")
    print(f"[test] recursive image search = {RECURSIVE_SEARCH}")
    print(f"[test] test_dir = {paths['test_dir']}")

    # -------------------------
    # Load AE model
    # -------------------------
    model = AutoEncoder().to(device).float()
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -------------------------
    # Transform
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Dataset / Loader (recursive)
    # -------------------------
    test_dataset = TestDataset(paths["test_dir"], transform, recursive=RECURSIVE_SEARCH)
    # batch_size=1 유지 (원래 코드 의도)
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
    result_dir = os.path.join(paths["results_dir"], "pixel_threshold_test_auto3")
    ensure_dir(result_dir, empty=True)

    # histogram 저장 폴더
    histplot_dir = os.path.join(result_dir, "pixel_diff_hist_plots")
    ensure_dir(histplot_dir, empty=True)

    heatmap_dir = os.path.join(result_dir, "heatmaps")
    quad_dir = os.path.join(result_dir, "quads")
    ensure_dir(heatmap_dir, empty=True)
    ensure_dir(quad_dir, empty=True)

    # -------------------------
    # 루프 결과
    # -------------------------
    results = []
    all_scores = []
    bad_images: List[Tuple[str, str]] = []

    print("\n=== Step 1: recursive image scan + pixel histogram + anomaly map + image score 계산 ===")

    for imgs, img_paths, errs in test_loader:
        # errs는 항상 리스트
        if errs:
            for p, e in errs:
                bad_images.append((p, e))

        # 전부 깨져서 batch가 비어있으면 스킵
        if imgs is None:
            continue

        # batch_size=1 기준
        img = imgs.to(device).float()  # [1,3,H,W]
        img_path = img_paths[0]

        with torch.no_grad():
            recon = model(img)

        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = recon[0].detach().cpu().permute(1, 2, 0).numpy()

        # 1) anomaly map (L2)
        amap = make_anomaly_map_l2(img_np, recon_np)  # HxW

        # 2) image score (amap 기반)
        image_score = compute_image_score(amap, mode=IMAGE_SCORE_MODE, topk_ratio=SCORE_TOPK_RATIO)
        all_scores.append(float(image_score))

        # 3) threshold mask + overlay (로스>=0.01 빨강)
        loss_mask = threshold_mask_from_amap(amap, threshold=LOSS_THRESHOLD)
        overlay_np = make_red_overlay(img_np, loss_mask, alpha=OVERLAY_ALPHA)

        # 4) pixel-diff histogram: mean_c |x - x_hat|
        dmap = make_pixel_diff_map_abs(img_np, recon_np)  # HxW
        flat_d = dmap.reshape(-1)

        overflow_cnt = int((flat_d > BIN_MAX).sum())
        inrange = flat_d[flat_d <= BIN_MAX]
        counts = hist_counts(inrange, bin_edges)

        # 통계(참고: amap 통계)
        flat_a = amap.reshape(-1)
        amap_mean = float(flat_a.mean())
        amap_max = float(flat_a.max())
        amap_p95 = float(np.percentile(flat_a, 95))

        # threshold 픽셀 비율(참고)
        thr_cnt = int(loss_mask.sum())
        thr_ratio = float(thr_cnt) / float(loss_mask.size)

        base = safe_filename_from_path(img_path)

        # 저장 경로
        histplot_path = os.path.join(histplot_dir, f"{base}_pixel_diff_hist.png")
        heatmap_path = os.path.join(heatmap_dir, f"{base}_heatmap.png")
        quad_path = os.path.join(quad_dir, f"{base}_quad.png")

        # histogram plot
        plot_hist_bar(
            counts=counts,
            bin_edges=bin_edges,
            title=f"{base} pixel-diff(|x-x_hat|) hist (bin={BIN_WIDTH}, max={BIN_MAX})",
            save_path=histplot_path,
            tick_step=HIST_TICK_STEP,
        )

        # heatmap 저장
        save_heatmap(
            amap=amap,
            title=f"{base} | score={image_score:.6f} | thr(amap>={LOSS_THRESHOLD})={thr_ratio*100:.2f}%",
            save_path=heatmap_path,
            vmax=HEATMAP_VMAX
        )

        # 4-panel 저장
        save_quad(
            img=img_np,
            recon=recon_np,
            amap=amap,
            overlay=overlay_np,
            title=f"{base} | score={image_score:.6f} | loss(amap)>={LOSS_THRESHOLD}",
            save_path=quad_path,
            vmax=HEATMAP_VMAX
        )

        results.append({
            "path": img_path,
            "image_score": float(image_score),
            "amap_mean": amap_mean,
            "amap_p95": amap_p95,
            "amap_max": amap_max,
            "thr_ratio(%)": thr_ratio * 100.0,
            "pixel_diff_overflow(>BIN_MAX)": overflow_cnt,
        })

    # score 분포 히스토그램 저장
    score_hist_path = os.path.join(result_dir, "image_score_hist.png")
    save_score_hist(
        all_scores,
        title=f"Image Score Histogram ({IMAGE_SCORE_MODE}, topk={SCORE_TOPK_RATIO})",
        save_path=score_hist_path
    )

    # 결과 정렬 (score 큰 순)
    results_sorted = sorted(results, key=lambda x: x["image_score"], reverse=True)

    # CSV 저장
    csv_path = os.path.join(result_dir, "results.csv")
    fieldnames = [
        "path",
        "image_score",
        "amap_mean",
        "amap_p95",
        "amap_max",
        "thr_ratio(%)",
        "pixel_diff_overflow(>BIN_MAX)"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    # TXT 저장
    txt_path = os.path.join(result_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Pixel-diff Histogram + Threshold Overlay + Image Score Results ===\n")
        f.write(f"checkpoint      : {CKPT_PATH}\n")
        f.write(f"test_dir        : {paths['test_dir']}\n")
        f.write(f"recursive       : {RECURSIVE_SEARCH}\n")
        f.write(f"img_size        : {img_size}\n\n")

        f.write(f"[pixel histogram]\n")
        f.write(f"  metric        : mean_c |x - x_hat|\n")
        f.write(f"  bin_width     : {BIN_WIDTH}\n")
        f.write(f"  bin_max       : {BIN_MAX}\n\n")

        f.write(f"[anomaly map]\n")
        f.write(f"  metric        : mean_c (x - x_hat)^2\n")
        f.write(f"  overlay_thr   : {LOSS_THRESHOLD}\n\n")

        f.write(f"[image score]\n")
        f.write(f"  mode          : {IMAGE_SCORE_MODE}\n")
        f.write(f"  topk_ratio    : {SCORE_TOPK_RATIO}\n")
        f.write(f"  score_hist_png: {score_hist_path}\n\n")

        if bad_images:
            f.write("[bad images skipped]\n")
            for p, e in bad_images:
                f.write(f"- {p}\t{e}\n")
            f.write("\n")

        f.write("rank\timage_score\tamap_p95\tamap_max\tthr_ratio(%)\toverflow\tpath\n")
        for i, r in enumerate(results_sorted):
            f.write(
                f"{i:04d}\t{r['image_score']:.6f}\t{r['amap_p95']:.6f}\t{r['amap_max']:.6f}\t"
                f"{r['thr_ratio(%)']:.2f}\t{r['pixel_diff_overflow(>BIN_MAX)']}\t{r['path']}\n"
            )

    # bad 이미지 목록도 별도 저장(원하면)
    if bad_images:
        bad_path = os.path.join(result_dir, "bad_images.txt")
        with open(bad_path, "w", encoding="utf-8") as f:
            for p, e in bad_images:
                f.write(f"{p}\t{e}\n")
        print(f"[saved] bad_images.txt      : {bad_path}")

    print("\n=== 완료 ===")
    print(f"[saved] result_dir          : {result_dir}")
    print(f"[saved] results.csv         : {csv_path}")
    print(f"[saved] results.txt         : {txt_path}")
    print(f"[saved] image_score_hist    : {score_hist_path}")
    print(f"[saved] pixel hist dir      : {histplot_dir}")
    print(f"[saved] heatmaps dir        : {heatmap_dir}")
    print(f"[saved] quads dir           : {quad_dir}")
    print(f"[info ] total images found  : {len(test_dataset)}")
    print(f"[info ] total processed     : {len(results)}")
    if bad_images:
        print(f"[info ] bad images skipped : {len(bad_images)}")


if __name__ == "__main__":
    main()
