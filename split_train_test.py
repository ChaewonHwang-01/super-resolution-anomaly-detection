"""
raw 데이터를 train(95%), test(5%) 으로 나누되
test 이미지는 test_xxxxx.png 형식으로 저장한다.
"""

import os
import random
import cv2
from utils.file_utils import load_config, ensure_dir, get_all_image_paths


def split_train_test(cfg: dict):

    paths_cfg = cfg["paths"]

    raw_dir = paths_cfg["raw_dir"]
    train_dir = os.path.join(paths_cfg["train_dir"], "normal")
    test_dir  = paths_cfg["test_dir"]   # normal/abnormal 분리 없음, 단일 디렉토리

    # ----------------------------------------------------
    # 폴더 초기화
    # ----------------------------------------------------
    ensure_dir(train_dir, empty=True)
    ensure_dir(test_dir, empty=True)

    # ----------------------------------------------------
    # raw 이미지 전체 불러오기
    # ----------------------------------------------------
    img_paths = get_all_image_paths(raw_dir)
    random.shuffle(img_paths)

    n_total = len(img_paths)
    train_ratio = 0.95
    n_train = int(n_total * train_ratio)

    print(f"[split] 전체 이미지: {n_total}")
    print(f"[split] → train: {n_train}, test: {n_total - n_train}")

    # ----------------------------------------------------
    # 1) train 저장
    # ----------------------------------------------------
    for idx, path in enumerate(img_paths[:n_train]):
        img = cv2.imread(path)
        if img is None:
            continue
        filename = f"train_{idx:05d}.png"
        save_path = os.path.join(train_dir, filename)
        cv2.imwrite(save_path, img)

    # ----------------------------------------------------
    # 2) test 저장 (이름: test_00000.png)
    # ----------------------------------------------------
    test_imgs = img_paths[n_train:]
    for idx, path in enumerate(test_imgs):
        img = cv2.imread(path)
        if img is None:
            continue
        filename = f"test_{idx:05d}.png"
        save_path = os.path.join(test_dir, filename)
        cv2.imwrite(save_path, img)

    print(f"[split] 데이터 분할 완료!")
    print(f" - train → {train_dir}")
    print(f" - test  → {test_dir}")
