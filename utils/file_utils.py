import os
import shutil
import glob
from datetime import datetime
import yaml


IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str, empty: bool = False):
    """디렉토리 없으면 생성, empty=True면 안의 내용 싹 비우기."""
    if empty and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def get_all_image_paths(root_dir: str):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext),
                               recursive=True))
    return paths


def get_timestamp() -> str:
    # 실행 시간 문자열 (파일 이름에 박기)
    return datetime.now().strftime("%Y%m%d_%H%M%S")
