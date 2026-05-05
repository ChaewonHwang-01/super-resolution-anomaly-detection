import os
from pathlib import Path
from PIL import Image

from utils.file_utils import ensure_dir

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def rgb_to_y(img: Image.Image) -> Image.Image:
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    return y  # 'L'


def convert_folder_rgb_to_y(src_dir: str, out_dir: str):
    ensure_dir(out_dir, empty=False)

    paths = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(root, f))
    paths.sort()

    print(f"[make_y] 총 이미지 수: {len(paths)}")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        y_img = rgb_to_y(img)

        name = Path(path).stem
        save_path = os.path.join(out_dir, f"{name}.png")
        y_img.save(save_path)

        if (i + 1) % 50 == 0:
            print(f"[make_y] {i+1}/{len(paths)} 처리 완료")

    print(f"[make_y] 완료 → {out_dir}")


def main(cfg: dict):
    """
    train.py에서 호출됨.
    config.yaml의 paths를 사용해서 src/out 경로를 결정.
    """
    paths = cfg["paths"]
    src_dir = paths["train_rgb_dir"]   # RGB 원본 폴더
    out_dir = paths["train_y_dir"]     # Y 결과 폴더

    convert_folder_rgb_to_y(src_dir, out_dir)


if __name__ == "__main__":
    # 단독 실행도 가능하게 해두고 싶으면 여기서 config 로드해도 됨
    # 하지만 지금은 train.py가 호출하는 게 메인이라 비워둬도 OK
    pass
