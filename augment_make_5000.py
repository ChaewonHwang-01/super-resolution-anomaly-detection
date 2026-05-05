import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# --------- 설정(필요하면 여기만 바꿔도 됨) ----------
SRC_DIR = "wood/train/good"               # 원본 1024x1024 정상 이미지 폴더
OUT_DIR = "data/train/normal_aug_5000/good"       # 생성될 폴더
N_TARGET = 5000                                  # 만들 장수
OUT_SIZE = 256                                   # 최종 저장 크기 (256x256)
SEED = 42                                        # 재현성
SAVE_FORMAT = "png"                              # "png" or "jpg"
JPG_QUALITY = 95                                 # jpg일 때만 사용
# ---------------------------------------------------


def random_augment(img: Image.Image) -> Image.Image:
    """패턴 데이터에 안전한 가벼운 증강"""
    # 랜덤 플립
    if random.random() < 0.5:
        img = ImageOps.mirror(img)  # horizontal
    if random.random() < 0.5:
        img = ImageOps.flip(img)    # vertical

    # 90도 단위 회전(격자 패턴에 안전)
    k = random.choice([0, 1, 2, 3])
    if k:
        img = img.rotate(90 * k, expand=False)

    # 밝기/대비 약하게
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.15))

    # 아주 약한 가우시안 노이즈(선택)
    if random.random() < 0.3:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 3.0, arr.shape)  # 표준편차 3 정도(약하게)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def resize_to_256(img: Image.Image, out_size: int) -> Image.Image:
    """자르지 않고 전체 이미지를 out_size x out_size로 리사이즈"""
    # 원본이 정사각(1024x1024)이면 그냥 리사이즈가 가장 깔끔함
    # 다운샘플은 LANCZOS 추천
    return img.resize((out_size, out_size), resample=Image.Resampling.LANCZOS)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    src = Path(SRC_DIR)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not files:
        raise RuntimeError(f"No images found in {SRC_DIR}")

    print(f"Found {len(files)} source images in {SRC_DIR}")
    print(f"Generating {N_TARGET} augmented samples to {OUT_DIR} (resize={OUT_SIZE}x{OUT_SIZE})")

    for i in range(N_TARGET):
        p = random.choice(files)
        img = Image.open(p).convert("RGB")  # grid는 grayscale처럼 보여도 RGB로 저장된 경우 많음

        # 1024 -> 256 (자르지 않고 리사이즈)
        img_256 = resize_to_256(img, OUT_SIZE)

        # 256 이미지에 증강 적용
        aug_img = random_augment(img_256)

        # 저장
        name = f"aug_{i:05d}.{SAVE_FORMAT}"
        save_path = out / name

        if SAVE_FORMAT.lower() == "jpg":
            aug_img.save(save_path, quality=JPG_QUALITY)
        else:
            aug_img.save(save_path)

        if (i + 1) % 200 == 0:
            print(f"  saved {i+1}/{N_TARGET}")

    print("✅ Done.")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()