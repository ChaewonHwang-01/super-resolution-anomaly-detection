import os
import random
import cv2
import numpy as np
from glob import glob
from .file_utils import ensure_dir


def make_anomaly(img, cfg_anom: dict):
    """정상 이미지를 이상 이미지로 변환 (blur + text ONLY)."""

    # Blur
    if random.random() < cfg_anom.get("blur_prob", 0.5):
        img = cv2.GaussianBlur(img, (7, 7), 0)

    # 텍스트 삽입
    if random.random() < cfg_anom.get("text_prob", 0.3):
        cv2.putText(
            img,
            "SALE!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def generate_abnormal_tests(test_dir: str,
                            cfg_anom: dict,
                            resize: int = 128):
    """
    test 폴더에 있는 이미지 중 50%만 abnormal 생성.
    abnormal 이미지도 test_xxxxx.png 형식으로 이어서 저장.
    """

    # test 폴더 내 모든 normal 이미지 읽기
    img_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        img_paths.extend(glob(os.path.join(test_dir, ext)))

    img_paths.sort()  # 번호순 정렬
    n_total = len(img_paths)

    print(f"[ab-generate] 현재 test 이미지 개수: {n_total}")

    # -----------------------------
    # 절반만 abnormal 대상
    # -----------------------------
    indices = list(range(n_total))
    random.shuffle(indices)
    abnormal_targets = indices[: n_total // 2]

    print(f"[ab-generate] 비정상 생성 대상: {len(abnormal_targets)}")

    # abnormal 이미지 번호 시작값 = 기존 test 개수
    next_idx = n_total

    for idx in abnormal_targets:
        img = cv2.imread(img_paths[idx])
        if img is None:
            continue
        img = cv2.resize(img, (resize, resize))

        abnormal = make_anomaly(img, cfg_anom)

        save_name = f"test_{next_idx:05d}.png"
        next_idx += 1

        save_path = os.path.join(test_dir, save_name)
        cv2.imwrite(save_path, abnormal)

    print(f"[ab-generate] 비정상 이미지 생성 완료 → {test_dir}")
