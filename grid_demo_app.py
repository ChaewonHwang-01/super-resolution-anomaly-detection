import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import cv2
import gradio as gr
import torch
import numpy as np

from torchvision import transforms
from models.vae import SR2_VAE
from utils.file_utils import load_config


# =======================================================
# 기본 설정
# =======================================================
cfg = load_config("config.yaml")
img_size = cfg["train"]["img_size"]
model_cfg = cfg["model"]

DEFAULT_SRCNN_CKPT = model_cfg.get("srcnn_pretrained_path", "./weights/train_srcnn.pth")
DEFAULT_VAE_CKPT = "./results/checkpoints/sr2_vae_only_grid.pth"
DEFAULT_LOAD_MODE = "vae_only"   # sr2_vae_only_grid.pth 이므로 vae_only 사용

# 이미지 단위 anomaly 판별용
DEFAULT_SCORE_GATE = 0.0035

# 점 마스크 생성용
DEFAULT_SCORE_BORDER = 4
DEFAULT_BORDER_MARGIN = 6

# 점 마스크 threshold 파라미터
DEFAULT_POINT_Z_THRESH = 1.6
DEFAULT_POINT_PERCENTILE = 98.5


# =======================================================
# defect bbox 생성용 파라미터
# =======================================================
# 핵심:
# - bbox 좌표는 high anomaly pixel 기준으로 계산한다.
# - low/support 영역은 가까운 seed 조각을 연결하는 용도로만 사용한다.
#
# 너무 많이 잡히면:
#   DEFAULT_BBOX_HIGH_PERCENTILE ↑
#   DEFAULT_BBOX_MIN_SEED_PIXELS ↑
#   DEFAULT_BBOX_MAX_BOXES ↓
#
# 너무 적게 잡히면:
#   DEFAULT_BBOX_HIGH_PERCENTILE ↓
#   DEFAULT_BBOX_MIN_SEED_PIXELS ↓
#   DEFAULT_BBOX_MERGE_GAP ↑
DEFAULT_BBOX_HIGH_PERCENTILE = 99.50
DEFAULT_BBOX_LOW_PERCENTILE = 98.20
DEFAULT_BBOX_Z_THRESH = 2.1

DEFAULT_BBOX_CLOSE_KSIZE = 11
DEFAULT_BBOX_DILATE_KSIZE = 5

DEFAULT_BBOX_MIN_SEED_PIXELS = 23
DEFAULT_BBOX_MIN_BOX_AREA = 120

DEFAULT_BBOX_MAX_BOXES = 5
DEFAULT_BBOX_PAD = 16
DEFAULT_BBOX_EDGE_MARGIN = 8

# 가까운 high seed box 병합용
DEFAULT_BBOX_MERGE_GAP = 45

# 너무 큰 오검출 박스 제거용
DEFAULT_BBOX_MAX_AREA_RATIO = 0.18


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[app] device = {device}")
print(f"[app] img_size = {img_size}")


# =======================================================
# 경로 유틸
# =======================================================
def resolve_path(path_str: str) -> str:
    path_str = (path_str or "").strip()
    if not path_str:
        return ""
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(BASE_DIR, path_str))


# =======================================================
# 모델 캐시
# =======================================================
_model_cache = {}


def _load_state_dict_flexible(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device
):
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


def load_model(
    srcnn_ckpt_path: str,
    vae_ckpt_path: str,
    load_mode: str = "vae_only"
):
    global _model_cache

    srcnn_ckpt_path = resolve_path(srcnn_ckpt_path)
    vae_ckpt_path = resolve_path(vae_ckpt_path)

    cache_key = (srcnn_ckpt_path, vae_ckpt_path, load_mode)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if load_mode not in ["vae_only", "full"]:
        raise ValueError("load_mode는 'vae_only' 또는 'full' 이어야 합니다.")

    if load_mode == "vae_only" and not os.path.isfile(srcnn_ckpt_path):
        raise FileNotFoundError(f"SRCNN checkpoint 파일이 없습니다: {srcnn_ckpt_path}")

    if not os.path.isfile(vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint 파일이 없습니다: {vae_ckpt_path}")

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

    if load_mode == "vae_only":
        _load_state_dict_flexible(model.srcnn, srcnn_ckpt_path, device)
        _load_state_dict_flexible(model.vae, vae_ckpt_path, device)
    else:
        _load_state_dict_flexible(model, vae_ckpt_path, device)

    model.eval()
    _model_cache[cache_key] = model
    return model


# =======================================================
# 전처리
# =======================================================
def make_transform():
    in_channels = int(model_cfg.get("in_channels", 3))

    if in_channels == 1:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


transform = make_transform()


# =======================================================
# 유틸
# =======================================================
def _to_rgb_for_vis(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)

    if img.ndim == 3 and img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    raise ValueError(f"Unexpected image shape: {img.shape}")


def tensor_to_numpy_image(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    return x.detach().cpu().permute(1, 2, 0).numpy()


# =======================================================
# vae_test.py 기준 anomaly map / image score
# =======================================================
def make_anomaly_map_l2(sr_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    diff = sr_np - recon_np
    amap = (diff * diff).mean(axis=2)
    return amap.astype(np.float32)


def make_pixel_diff_map_abs(sr_np: np.ndarray, recon_np: np.ndarray) -> np.ndarray:
    diff = np.abs(sr_np - recon_np)
    dmap = diff.mean(axis=2)
    return dmap.astype(np.float32)


def make_local_map_from_amap(amap: np.ndarray) -> np.ndarray:
    raw_map = cv2.GaussianBlur(amap.astype(np.float32), (3, 3), 0)
    bg = cv2.GaussianBlur(raw_map, (31, 31), 0)
    local_map = raw_map - bg
    local_map = np.clip(local_map, 0.0, None)
    local_map = cv2.GaussianBlur(local_map, (3, 3), 0)
    return local_map.astype(np.float32)


def compute_image_score(
    amap: np.ndarray,
    mode: str = "topk",
    topk_ratio: float = 0.01,
    border: int = DEFAULT_SCORE_BORDER,
) -> float:
    h, w = amap.shape

    if h <= 2 * border or w <= 2 * border:
        core = amap
    else:
        core = amap[border:h - border, border:w - border]

    flat = core.reshape(-1)

    if mode == "max":
        return float(flat.max())

    if mode == "topk":
        k = max(1, int(len(flat) * topk_ratio))
        topk_vals = np.partition(flat, -k)[-k:]
        return float(topk_vals.mean())

    raise ValueError(f"Unknown mode: {mode}")


# =======================================================
# Heatmap
# =======================================================
def normalize_map_percentile(
    amap: np.ndarray,
    low: float = 96.0,
    high: float = 99.8
) -> np.ndarray:
    lo = float(np.percentile(amap, low))
    hi = float(np.percentile(amap, high))

    if hi <= lo:
        return np.zeros_like(amap, dtype=np.float32)

    norm = (amap - lo) / (hi - lo + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(np.float32)


def render_heatmap_overlay(
    base_img: np.ndarray,
    amap: np.ndarray,
    alpha: float = 0.60
):
    base_rgb = _to_rgb_for_vis(base_img)
    base_u8 = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8)

    norm = normalize_map_percentile(amap, low=96.0, high=99.8)
    heat_u8 = (norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(base_u8, 1.0 - alpha, heat_color, alpha, 0)
    return overlay


# =======================================================
# 점 기반 anomaly mask
# =======================================================
def make_point_mask_from_amap(
    amap: np.ndarray,
    border_margin: int = DEFAULT_BORDER_MARGIN,
    z_thresh: float = DEFAULT_POINT_Z_THRESH,
    percentile_thr: float = DEFAULT_POINT_PERCENTILE,
    morph_ksize: int = 3,
):
    h, w = amap.shape
    work = amap.astype(np.float32).copy()

    if border_margin > 0:
        work[:border_margin, :] = 0.0
        work[h - border_margin:, :] = 0.0
        work[:, :border_margin] = 0.0
        work[:, w - border_margin:] = 0.0

    valid = work[border_margin:h - border_margin, border_margin:w - border_margin] \
        if (h > 2 * border_margin and w > 2 * border_margin) else work

    if valid.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med))) + 1e-8
    robust_sigma = 1.4826 * mad + 1e-8
    zmap = (work - med) / robust_sigma

    p_thr = float(np.percentile(valid, percentile_thr))
    binary = np.logical_and(zmap > z_thresh, work > p_thr).astype(np.uint8) * 255

    k = max(1, int(morph_ksize))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


# =======================================================
# High seed 기준 bbox 추출
# =======================================================
def _boxes_should_merge(
    box_a,
    box_b,
    gap: int = DEFAULT_BBOX_MERGE_GAP
):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ax1_g = ax1 - gap
    ay1_g = ay1 - gap
    ax2_g = ax2 + gap
    ay2_g = ay2 + gap

    return not (
        ax2_g < bx1 or
        bx2 < ax1_g or
        ay2_g < by1 or
        by2 < ay1_g
    )


def _merge_seed_boxes(
    box_items,
    gap: int = DEFAULT_BBOX_MERGE_GAP
):
    """
    high anomaly seed 기반 box들을 가까운 것끼리 병합한다.
    """
    if not box_items:
        return []

    merged = []

    for item in box_items:
        x1, y1, x2, y2 = item["box"]
        merged_flag = False

        for m in merged:
            mx1, my1, mx2, my2 = m["box"]

            if _boxes_should_merge(
                (x1, y1, x2, y2),
                (mx1, my1, mx2, my2),
                gap=gap
            ):
                new_box = (
                    min(x1, mx1),
                    min(y1, my1),
                    max(x2, mx2),
                    max(y2, my2),
                )

                m["box"] = new_box
                m["seed_pixels"] += item["seed_pixels"]
                m["max_score"] = max(m["max_score"], item["max_score"])
                m["mean_score"] = max(m["mean_score"], item["mean_score"])
                m["box_area"] = (
                    (new_box[2] - new_box[0] + 1) *
                    (new_box[3] - new_box[1] + 1)
                )
                merged_flag = True
                break

        if not merged_flag:
            merged.append(item.copy())

    return merged


def extract_hysteresis_defect_boxes_from_amap(
    amap: np.ndarray,
    border_margin: int = DEFAULT_BORDER_MARGIN,
    high_percentile: float = DEFAULT_BBOX_HIGH_PERCENTILE,
    low_percentile: float = DEFAULT_BBOX_LOW_PERCENTILE,
    z_thresh: float = DEFAULT_BBOX_Z_THRESH,
    close_ksize: int = DEFAULT_BBOX_CLOSE_KSIZE,
    dilate_ksize: int = DEFAULT_BBOX_DILATE_KSIZE,
    min_seed_pixels: int = DEFAULT_BBOX_MIN_SEED_PIXELS,
    min_box_area: int = DEFAULT_BBOX_MIN_BOX_AREA,
    max_boxes: int = DEFAULT_BBOX_MAX_BOXES,
    pad: int = DEFAULT_BBOX_PAD,
    edge_margin: int = DEFAULT_BBOX_EDGE_MARGIN,
    merge_gap: int = DEFAULT_BBOX_MERGE_GAP,
    max_area_ratio: float = DEFAULT_BBOX_MAX_AREA_RATIO,
):
    """
    bbox는 high anomaly pixel 기준으로 계산하고,
    low/support mask는 seed 조각 연결을 보조하는 데만 사용한다.
    """
    h, w = amap.shape
    work = amap.astype(np.float32).copy()

    if border_margin > 0:
        work[:border_margin, :] = 0.0
        work[h - border_margin:, :] = 0.0
        work[:, :border_margin] = 0.0
        work[:, w - border_margin:] = 0.0

    valid = work[border_margin:h - border_margin, border_margin:w - border_margin] \
        if (h > 2 * border_margin and w > 2 * border_margin) else work

    if valid.size == 0:
        return []

    # 너무 날카로운 점 노이즈를 줄이기 위해 약하게 smoothing
    smooth = cv2.GaussianBlur(work, (3, 3), 0)

    valid_smooth = smooth[border_margin:h - border_margin, border_margin:w - border_margin] \
        if (h > 2 * border_margin and w > 2 * border_margin) else smooth

    med = float(np.median(valid_smooth))
    mad = float(np.median(np.abs(valid_smooth - med))) + 1e-8
    robust_sigma = 1.4826 * mad + 1e-8
    zmap = (smooth - med) / robust_sigma

    high_thr = float(np.percentile(valid_smooth, high_percentile))
    low_thr = float(np.percentile(valid_smooth, low_percentile))

    high_mask = np.logical_and(
        smooth > high_thr,
        zmap > z_thresh
    ).astype(np.uint8) * 255

    low_mask = np.logical_and(
        smooth > low_thr,
        zmap > max(1.0, z_thresh * 0.45)
    ).astype(np.uint8) * 255

    # high seed를 조금만 연결
    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    high_mask = cv2.morphologyEx(high_mask, cv2.MORPH_CLOSE, seed_kernel)

    # low mask는 연결 판단용으로만 사용
    if close_ksize > 1:
        if close_ksize % 2 == 0:
            close_ksize += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_ksize, close_ksize)
        )
        low_mask = cv2.morphologyEx(low_mask, cv2.MORPH_CLOSE, close_kernel)

    if dilate_ksize > 1:
        if dilate_ksize % 2 == 0:
            dilate_ksize += 1

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_ksize, dilate_ksize)
        )
        low_mask = cv2.dilate(low_mask, dilate_kernel, iterations=1)

    # low support component 추출
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        low_mask,
        connectivity=8
    )

    raw_boxes = []

    for i in range(1, num_labels):
        component_region = labels == i

        # 이 support component 내부의 high seed만 사용
        seed_region = np.logical_and(component_region, high_mask > 0)
        ys, xs = np.where(seed_region)

        seed_pixels = int(len(xs))

        if seed_pixels < min_seed_pixels:
            continue

        # bbox 좌표는 low component가 아니라 high seed 좌표 기준
        x1_seed = int(xs.min())
        y1_seed = int(ys.min())
        x2_seed = int(xs.max())
        y2_seed = int(ys.max())

        if (
            x1_seed <= edge_margin or
            y1_seed <= edge_margin or
            x2_seed >= w - 1 - edge_margin or
            y2_seed >= h - 1 - edge_margin
        ):
            continue

        x1 = max(0, x1_seed - pad)
        y1 = max(0, y1_seed - pad)
        x2 = min(w - 1, x2_seed + pad)
        y2 = min(h - 1, y2_seed + pad)

        box_area = int((x2 - x1 + 1) * (y2 - y1 + 1))

        if box_area < min_box_area:
            continue

        # 너무 큰 박스는 격자 배경까지 먹은 오검출일 가능성이 높음
        if box_area > int(h * w * max_area_ratio):
            continue

        seed_values = work[seed_region]
        if seed_values.size == 0:
            continue

        mean_score = float(seed_values.mean())
        max_score = float(seed_values.max())

        raw_boxes.append({
            "box": (x1, y1, x2, y2),
            "seed_pixels": seed_pixels,
            "box_area": box_area,
            "mean_score": mean_score,
            "max_score": max_score,
            "rank_score": seed_pixels * max_score,
        })

    # high seed box 기준으로 가까운 결함 조각 병합
    boxes = _merge_seed_boxes(raw_boxes, gap=merge_gap)

    # 병합 후 너무 큰 박스 다시 제거
    filtered = []

    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        box_area = int((x2 - x1 + 1) * (y2 - y1 + 1))
        b["box_area"] = box_area

        if box_area < min_box_area:
            continue

        if box_area > int(h * w * max_area_ratio):
            continue

        filtered.append(b)

    boxes = sorted(
        filtered,
        key=lambda b: (
            b["seed_pixels"],
            b["max_score"],
            -b["box_area"]
        ),
        reverse=True
    )

    return boxes[:max_boxes]


def render_bbox_overlay(
    base_img: np.ndarray,
    boxes,
    line_thickness: int = 3,
):
    base_rgb = _to_rgb_for_vis(base_img)
    vis = (np.clip(base_rgb, 0.0, 1.0) * 255).astype(np.uint8).copy()

    for idx, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = item["box"]

        cv2.rectangle(
            vis,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            line_thickness
        )

        cv2.putText(
            vis,
            f"defect {idx}",
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return vis.astype(np.float32) / 255.0


def render_binary_mask(binary_mask: np.ndarray):
    mask_u8 = np.clip(binary_mask, 0, 255).astype(np.uint8)
    return np.stack([mask_u8, mask_u8, mask_u8], axis=2)


# =======================================================
# 실행 함수
# =======================================================
def run_demo(
    input_image,
    srcnn_ckpt_path,
    vae_ckpt_path,
    load_mode,
    score_mode,
    topk_ratio,
    heatmap_alpha,
    morph_ksize,
):
    if input_image is None:
        raise gr.Error("이미지를 업로드해줘.")

    srcnn_ckpt_path = srcnn_ckpt_path.strip()
    vae_ckpt_path = vae_ckpt_path.strip()

    if load_mode == "vae_only" and not srcnn_ckpt_path:
        raise gr.Error("vae_only 모드에서는 SRCNN checkpoint 경로를 입력해줘.")

    if not vae_ckpt_path:
        raise gr.Error("VAE 체크포인트가 비어 있습니다.")

    model = load_model(srcnn_ckpt_path, vae_ckpt_path, load_mode)

    in_channels = int(model_cfg.get("in_channels", 3))

    if in_channels == 1:
        pil_img = input_image.convert("L")
    else:
        pil_img = input_image.convert("RGB")

    img_tensor = transform(pil_img).unsqueeze(0).to(device).float()

    with torch.no_grad():
        sr_512, recon_512, mu, logvar = model(img_tensor)

    H = min(sr_512.shape[-2], recon_512.shape[-2])
    W = min(sr_512.shape[-1], recon_512.shape[-1])
    sr_512 = sr_512[:, :, :H, :W]
    recon_512 = recon_512[:, :, :H, :W]

    sr_np = tensor_to_numpy_image(sr_512)
    recon_np = tensor_to_numpy_image(recon_512)

    amap = make_anomaly_map_l2(sr_np, recon_np)
    pixel_diff_map = make_pixel_diff_map_abs(sr_np, recon_np)
    local_map = make_local_map_from_amap(amap)

    image_score = compute_image_score(
        amap,
        mode=score_mode,
        topk_ratio=float(topk_ratio),
        border=DEFAULT_SCORE_BORDER,
    )

    heatmap_img = render_heatmap_overlay(
        sr_np,
        amap,
        alpha=float(heatmap_alpha)
    )

    # Binary Mask 탭용 point mask
    point_mask = make_point_mask_from_amap(
        amap=amap,
        border_margin=DEFAULT_BORDER_MARGIN,
        z_thresh=DEFAULT_POINT_Z_THRESH,
        percentile_thr=DEFAULT_POINT_PERCENTILE,
        morph_ksize=int(morph_ksize),
    )

    # Defect Detection 탭용 bbox
    # bbox 좌표는 high seed 기준으로 계산하고,
    # low/support mask는 가까운 조각을 연결하는 용도로만 사용한다.
    boxes = extract_hysteresis_defect_boxes_from_amap(
        amap=amap,
        border_margin=DEFAULT_BORDER_MARGIN,
        high_percentile=DEFAULT_BBOX_HIGH_PERCENTILE,
        low_percentile=DEFAULT_BBOX_LOW_PERCENTILE,
        z_thresh=DEFAULT_BBOX_Z_THRESH,
        close_ksize=DEFAULT_BBOX_CLOSE_KSIZE,
        dilate_ksize=DEFAULT_BBOX_DILATE_KSIZE,
        min_seed_pixels=DEFAULT_BBOX_MIN_SEED_PIXELS,
        min_box_area=DEFAULT_BBOX_MIN_BOX_AREA,
        max_boxes=DEFAULT_BBOX_MAX_BOXES,
        pad=DEFAULT_BBOX_PAD,
        edge_margin=DEFAULT_BBOX_EDGE_MARGIN,
        merge_gap=DEFAULT_BBOX_MERGE_GAP,
        max_area_ratio=DEFAULT_BBOX_MAX_AREA_RATIO,
    )

    bbox_overlay = render_bbox_overlay(
        base_img=sr_np,
        boxes=boxes,
        line_thickness=3,
    )

    binary_img = render_binary_mask(point_mask)

    point_count = int((point_mask > 0).sum())
    defect_count = len(boxes)

    box_debug = ""
    if defect_count > 0:
        box_debug = " | boxes: " + ", ".join([
            f"{i + 1}(seed={b['seed_pixels']}, area={b['box_area']})"
            for i, b in enumerate(boxes)
        ])

    amap_mean = float(amap.mean())
    amap_max = float(amap.max())
    amap_p95 = float(np.percentile(amap, 95))
    local_max = float(local_map.max())

    pixel_diff_mean = float(pixel_diff_map.mean())
    pixel_diff_max = float(pixel_diff_map.max())

    mu_mean = float(mu.mean().item())
    mu_std = float(mu.std().item())
    logvar_mean = float(logvar.mean().item())
    logvar_std = float(logvar.std().item())

    status = "Anomaly Detected" if (
        image_score >= DEFAULT_SCORE_GATE and defect_count > 0
    ) else "No Strong Defect"

    if status == "No Strong Defect":
        bbox_overlay = _to_rgb_for_vis(sr_np).astype(np.float32)
        binary_img = render_binary_mask(
            np.zeros((sr_np.shape[0], sr_np.shape[1]), dtype=np.uint8)
        )
        point_count = 0
        defect_count = 0
        box_debug = ""

    summary_text = (
        f"{status} | "
        f"defect_regions: {defect_count} | "
        f"anomaly_points: {point_count} | "
        f"image_score: {image_score:.6f} | "
        f"amap_mean: {amap_mean:.6f} | "
        f"amap_p95: {amap_p95:.6f} | "
        f"amap_max: {amap_max:.6f} | "
        f"local_max: {local_max:.6f} | "
        f"pixel_diff_mean: {pixel_diff_mean:.6f} | "
        f"pixel_diff_max: {pixel_diff_max:.6f} | "
        f"mu_mean: {mu_mean:.6f} | "
        f"mu_std: {mu_std:.6f} | "
        f"logvar_mean: {logvar_mean:.6f} | "
        f"logvar_std: {logvar_std:.6f}"
        f"{box_debug}"
    )

    return (
        heatmap_img,
        np.clip(bbox_overlay, 0.0, 1.0),
        binary_img,
        f"Score: {image_score:.6f}",
        summary_text
    )


# =======================================================
# UI
# =======================================================
custom_css = """
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
}
h1 {
    margin-bottom: 10px !important;
}
.compact-box textarea,
.compact-box input {
    font-size: 14px !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Anomaly Detection Demo (SR2_VAE)</h1>")

    with gr.Row(equal_height=True):
        input_image = gr.Image(
            type="pil",
            label="Upload Image",
            height=380
        )

        with gr.Tabs():
            with gr.Tab("VAE Anomaly Map"):
                heatmap_image = gr.Image(label=None, height=380)

            with gr.Tab("Defect Detection"):
                defect_image = gr.Image(label=None, height=380)

            with gr.Tab("Binary Mask"):
                binary_image = gr.Image(label=None, height=380)

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            run_btn = gr.Button("Run", variant="primary")

        with gr.Column(scale=2):
            score_text = gr.Textbox(
                label="Image Score",
                interactive=False,
                elem_classes=["compact-box"]
            )

            summary_text = gr.Textbox(
                label="Summary",
                interactive=False,
                elem_classes=["compact-box"]
            )

    with gr.Accordion("Options", open=False):
        srcnn_ckpt_path = gr.Textbox(
            label="SRCNN Checkpoint Path",
            value=DEFAULT_SRCNN_CKPT
        )

        vae_ckpt_path = gr.Textbox(
            label="VAE Checkpoint Path",
            value=DEFAULT_VAE_CKPT
        )

        load_mode = gr.Radio(
            choices=["vae_only", "full"],
            value=DEFAULT_LOAD_MODE,
            label="Load Mode"
        )

        with gr.Row():
            score_mode = gr.Radio(
                choices=["topk", "max"],
                value="topk",
                label="Score Mode"
            )

            topk_ratio = gr.Slider(
                minimum=0.001,
                maximum=0.1,
                value=0.01,
                step=0.001,
                label="Top-k Ratio"
            )

        with gr.Row():
            heatmap_alpha = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.60,
                step=0.05,
                label="Heatmap Overlay Alpha"
            )

        gr.Markdown("### Defect detection options")

        with gr.Row():
            morph_ksize = gr.Slider(
                minimum=1,
                maximum=11,
                value=3,
                step=1,
                label="Binary Mask Morph Kernel Size"
            )

    run_btn.click(
        fn=run_demo,
        inputs=[
            input_image,
            srcnn_ckpt_path,
            vae_ckpt_path,
            load_mode,
            score_mode,
            topk_ratio,
            heatmap_alpha,
            morph_ksize,
        ],
        outputs=[
            heatmap_image,
            defect_image,
            binary_image,
            score_text,
            summary_text
        ]
    )


if __name__ == "__main__":
    demo.launch()