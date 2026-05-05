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
- test_dir 아래 하위 폴더까지 포함해서 모든 이미지 재귀 로딩
- pixel histogram 범위: BIN_WIDTH=0.001, BIN_MAX=0.20
- histogram 라벨 소수점: 3자리로 표기 (0.001 단위가 보이게)
- MSE 결과(image_mse) 저장
- 차이가 0~0.001인 픽셀(amap<=0.001) 제외한 평균 저장 (현재 eps=0.003 사용)
- AUROC는 아직 미포함
- 시각화 저장은 score 상위 3개 + 하위 3개만 저장
- (추가) 입력 이미지는 항상 256x256으로 변환 후 실행
"""

import os
import csv
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.autoencoder import AutoEncoder
from utils.file_utils import load_config, ensure_dir


# -------------------------------------------------------
#  Test Dataset (재귀 로딩)
# -------------------------------------------------------
class TestDataset(Dataset):
    def __init__(self, root, transform=None, exts=None):
        self.root = root
        self.transform = transform
        self.exts = exts or {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        self.paths = self._collect_paths_recursive(self.root)

        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"[TestDataset] '{root}' 아래에서 이미지 파일을 찾지 못했습니다. "
                f"지원 확장자: {sorted(list(self.exts))}"
            )

    def _collect_paths_recursive(self, root: str):
        collected = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in self.exts:
                    collected.append(os.path.join(dirpath, fn))
        collected.sort()
        return collected

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


def unique_name_from_path(root_dir: str, path: str) -> str:
    """
    test_dir 기준 상대경로를 이용해 파일명 충돌 방지.
    예: test/abnormal/glue/0001.png -> abnormal__glue__0001
    """
    rel = os.path.relpath(path, root_dir)
    rel = rel.replace("\\", "/")
    rel_no_ext = os.path.splitext(rel)[0]
    safe = rel_no_ext.replace("/", "__")
    return safe


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


def plot_hist_bar(counts: np.ndarray, bin_edges: np.ndarray, title: str, save_path: str):
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
# Anomaly map + image score
# -------------------------------------------------------
def make_anomaly_map_l2(img_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    diff = img_np - recon_np
    amap = (diff * diff).mean(axis=2)
    return amap


def compute_image_score(amap: np.ndarray, mode: str = "topk", topk_ratio: float = 0.01) -> float:
    flat = amap.reshape(-1)

    if mode == "max":
        return float(flat.max())

    if mode == "topk":
        k = max(1, int(len(flat) * topk_ratio))
        topk_vals = np.partition(flat, -k)[-k:]
        return float(topk_vals.mean())

    raise ValueError(f"Unknown mode: {mode}")


def compute_image_mse(img_np: np.ndarray, recon_np: np.ndarray) -> float:
    diff = img_np - recon_np
    return float((diff * diff).mean())


def mean_excluding_small(amap: np.ndarray, eps: float = 0.003) -> tuple[float, int, float]:
    flat = amap.reshape(-1)
    excl_mask = (flat <= eps)
    excluded_pixels = int(excl_mask.sum())
    excluded_ratio = float(excluded_pixels / len(flat)) if len(flat) > 0 else 0.0

    remain = flat[~excl_mask]
    if remain.size == 0:
        return 0.0, excluded_pixels, excluded_ratio

    return float(remain.mean()), excluded_pixels, excluded_ratio


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
    img_size_cfg = cfg["train"]["img_size"]  # 참고용 (실제 실행은 256 고정)

    # =========================
    # 사용자 입력: 체크포인트 경로
    # =========================
    CKPT_PATH = "./results/checkpoints/autoencoder_20260109_014539.pth"
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"지정한 체크포인트 파일이 존재하지 않습니다: {CKPT_PATH}")

    # =========================
    # 사용자 입력: pixel-diff histogram 설정
    # =========================
    BIN_WIDTH = 0.001
    BIN_MAX = 0.20
    bin_edges = make_bins(bin_width=BIN_WIDTH, max_val=BIN_MAX)

    # =========================
    # 사용자 입력: image score 설정
    # =========================
    IMAGE_SCORE_MODE = "topk"   # "topk" or "max"
    TOPK_RATIO = 0.01           # top 1%

    # =========================
    # 0~0.003 픽셀 제외 평균 계산용 threshold
    # =========================
    EXCL_EPS = 0.003

    # =========================
    # (선택) heatmap 대비용 vmax
    # =========================
    HEATMAP_VMAX = None

    # =========================
    # 시각화 저장 개수
    # =========================
    TOPN_VIS = 3
    BOTTOMN_VIS = 3

    # =========================
    # (추가) 입력 이미지는 항상 256x256으로 변환
    # =========================
    FORCE_IMG_SIZE = 256

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
    print(f"[test] excl_mean: exclude amap <= {EXCL_EPS}")
    print(f"[test] test_dir (recursive) = {paths['test_dir']}")
    print(f"[test] img_size(cfg)={img_size_cfg}  -> FORCE resize = {FORCE_IMG_SIZE}")

    # -------------------------
    # Load AE model
    # -------------------------
    model = AutoEncoder().to(device).float()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # -------------------------
    # Transform (항상 256)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((FORCE_IMG_SIZE, FORCE_IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # Dataset / Loader
    # -------------------------
    test_dataset = TestDataset(paths["test_dir"], transform)
    print(f"[test] found images = {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------
    # 결과 저장 디렉토리
    # -------------------------
    result_dir = os.path.join(paths["results_dir"], "haz_auto1_mse_003")
    ensure_dir(result_dir, empty=True)

    histplot_dir = os.path.join(result_dir, "pixel_diff_hist_plots")
    heatmap_dir = os.path.join(result_dir, "heatmaps")
    triplet_dir = os.path.join(result_dir, "triplets")
    ensure_dir(histplot_dir, empty=True)
    ensure_dir(heatmap_dir, empty=True)
    ensure_dir(triplet_dir, empty=True)

    # =====================================================
    # PASS 1) 전체 이미지: score/mse/통계 계산 (CSV/TXT용)
    # =====================================================
    results = []
    all_scores = []

    print("\n=== Pass 1: 모든 이미지 score/mse/통계 계산 (시각화는 아직 안 함) ===")

    for img, path in test_loader:
        img = img.to(device).float()
        with torch.no_grad():
            recon = model(img)

        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = recon[0].detach().cpu().permute(1, 2, 0).numpy()

        image_mse = compute_image_mse(img_np, recon_np)
        amap = make_anomaly_map_l2(img_np, recon_np)
        amap_mean_excl, excl_pixels, excl_ratio = mean_excluding_small(amap, eps=EXCL_EPS)

        image_score = compute_image_score(amap, mode=IMAGE_SCORE_MODE, topk_ratio=TOPK_RATIO)
        all_scores.append(float(image_score))

        flat = amap.reshape(-1)
        overflow_cnt = int((flat > BIN_MAX).sum())

        amap_mean = float(flat.mean())
        amap_max = float(flat.max())
        amap_p95 = float(np.percentile(flat, 95))

        img_path = path[0]

        results.append({
            "path": img_path,
            "image_score": float(image_score),
            "image_mse": float(image_mse),
            "amap_mean": amap_mean,
            "amap_mean_excl_0_0p003": float(amap_mean_excl),
            "excl_pixels(<=0.003)": int(excl_pixels),
            "excl_ratio(<=0.003)": float(excl_ratio),
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

    # =====================================================
    # Top3 + Bottom3 선택
    # =====================================================
    top_k = results_sorted[:min(TOPN_VIS, len(results_sorted))]
    bottom_k = results_sorted[-min(BOTTOMN_VIS, len(results_sorted)):] if len(results_sorted) > 0 else []
    selected = top_k + bottom_k
    selected_paths = set(r["path"] for r in selected)

    print("\n=== 시각화 저장 대상 (Top/Bottom) ===")
    for r in top_k:
        print(f"[TOP]    score={r['image_score']:.6f}  path={r['path']}")
    for r in bottom_k:
        print(f"[BOTTOM] score={r['image_score']:.6f}  path={r['path']}")

    # =====================================================
    # PASS 2) 선택된 6장만: hist/heatmap/triplet 저장
    # =====================================================
    print("\n=== Pass 2: Top3 + Bottom3만 시각화 저장 ===")

    rank_tag = {}
    for i, r in enumerate(top_k):
        rank_tag[r["path"]] = f"TOP{i+1}"
    for i, r in enumerate(bottom_k):
        rank_tag[r["path"]] = f"BOT{i+1}"

    test_loader2 = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for img, path in test_loader2:
        img_path = path[0]
        if img_path not in selected_paths:
            continue

        img = img.to(device).float()
        with torch.no_grad():
            recon = model(img)

        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
        recon_np = recon[0].detach().cpu().permute(1, 2, 0).numpy()

        image_mse = compute_image_mse(img_np, recon_np)
        amap = make_anomaly_map_l2(img_np, recon_np)
        image_score = compute_image_score(amap, mode=IMAGE_SCORE_MODE, topk_ratio=TOPK_RATIO)

        flat = amap.reshape(-1)
        overflow_cnt = int((flat > BIN_MAX).sum())
        inrange = flat[flat <= BIN_MAX]
        counts = hist_counts(inrange, bin_edges)

        amap_mean_excl, _, _ = mean_excluding_small(amap, eps=EXCL_EPS)

        tag = rank_tag.get(img_path, "SEL")
        base = unique_name_from_path(paths["test_dir"], img_path)

        histplot_path = os.path.join(histplot_dir, f"{tag}__{base}_pixel_diff_hist.png")
        heatmap_path = os.path.join(heatmap_dir, f"{tag}__{base}_heatmap.png")
        triplet_path = os.path.join(triplet_dir, f"{tag}__{base}_triplet.png")

        plot_hist_bar(
            counts=counts,
            bin_edges=bin_edges,
            title=f"[{tag}] {base} pixel-diff histogram (bin={BIN_WIDTH}, max={BIN_MAX})",
            save_path=histplot_path
        )

        save_heatmap(
            amap=amap,
            title=(
                f"[{tag}] {base} | score={image_score:.6f} | mse={image_mse:.6f} | "
                f"mean_excl(<= {EXCL_EPS})={amap_mean_excl:.6f} | overflow(>{BIN_MAX})={overflow_cnt}"
            ),
            save_path=heatmap_path,
            vmax=HEATMAP_VMAX
        )

        save_triplet(
            img=img_np,
            recon=recon_np,
            amap=amap,
            title=f"[{tag}] {base} | score={image_score:.6f} | mse={image_mse:.6f}",
            save_path=triplet_path,
            vmax=HEATMAP_VMAX
        )

    # =====================================================
    # CSV 저장 (전체 결과)
    # =====================================================
    csv_path = os.path.join(result_dir, "results.csv")
    fieldnames = [
        "path",
        "image_score",
        "image_mse",
        "amap_mean",
        "amap_mean_excl_0_0p003",
        "excl_pixels(<=0.003)",
        "excl_ratio(<=0.003)",
        "amap_p95",
        "amap_max",
        "overflow(>BIN_MAX)",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow(r)

    # =====================================================
    # TXT 저장 (전체 결과)
    # =====================================================
    txt_path = os.path.join(result_dir, "results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Pixel Histogram + Top-k Image Score Results ===\n")
        f.write(f"checkpoint      : {CKPT_PATH}\n")
        f.write(f"test_dir        : {paths['test_dir']}\n")
        f.write(f"img_size(cfg)   : {img_size_cfg}\n")
        f.write(f"resize(force)   : {FORCE_IMG_SIZE}\n\n")

        f.write(f"[pixel histogram]\n")
        f.write(f"  bin_width     : {BIN_WIDTH}\n")
        f.write(f"  bin_max       : {BIN_MAX}\n\n")

        f.write(f"[image score]\n")
        f.write(f"  mode          : {IMAGE_SCORE_MODE}\n")
        f.write(f"  topk_ratio    : {TOPK_RATIO}\n")
        f.write(f"  score_hist_png: {score_hist_path}\n\n")

        f.write(f"[mse / excl-mean]\n")
        f.write(f"  image_mse     : mean((x - x_hat)^2) over H,W,C\n")
        f.write(f"  excl_eps      : {EXCL_EPS} (exclude amap <= eps)\n\n")

        f.write("[visualization saved]\n")
        f.write(f"  top_n         : {TOPN_VIS}\n")
        f.write(f"  bottom_n      : {BOTTOMN_VIS}\n\n")

        f.write("rank\timage_score\timage_mse\tamap_mean\tamap_mean_excl\tamap_p95\tamap_max\toverflow\texcl_ratio\tpath\n")
        for i, r in enumerate(results_sorted):
            f.write(
                f"{i:04d}\t{r['image_score']:.6f}\t{r['image_mse']:.6f}\t{r['amap_mean']:.6f}\t"
                f"{r['amap_mean_excl_0_0p003']:.6f}\t{r['amap_p95']:.6f}\t{r['amap_max']:.6f}\t"
                f"{r['overflow(>BIN_MAX)']}\t{r['excl_ratio(<=0.003)']:.4f}\t{r['path']}\n"
            )

    print("\n=== 완료 ===")
    print(f"[saved] result_dir          : {result_dir}")
    print(f"[saved] results.csv         : {csv_path}")
    print(f"[saved] results.txt         : {txt_path}")
    print(f"[saved] image_score_hist    : {score_hist_path}")
    print(f"[saved] pixel hist dir      : {histplot_dir}  (Top/Bottom만 저장)")
    print(f"[saved] heatmaps dir        : {heatmap_dir}   (Top/Bottom만 저장)")
    print(f"[saved] triplets dir        : {triplet_dir}   (Top/Bottom만 저장)")


if __name__ == "__main__":
    main()
