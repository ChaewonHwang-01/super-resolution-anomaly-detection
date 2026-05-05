import os
import glob
import torch
import cv2
from torchvision import transforms
from PIL import Image
from utils.file_utils import ensure_dir
from models.sr_model import SRCNN   # 너의 SR 모델 이름에 맞게 수정

def apply_sr_to_raw(cfg):
    paths = cfg["paths"]
    raw_dir = paths["raw_dir"]
    sr_out_dir = paths["sr_out_dir"]
    img_size = cfg["train"]["img_size"]

    ensure_dir(sr_out_dir, empty=True)

    # SR 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YAML에서 가중치 경로 가져오기
    sr_weight_path = cfg["paths"]["sr_weights"]

    if not os.path.exists(sr_weight_path):
        raise FileNotFoundError(f"SR 모델 가중치를 찾을 수 없습니다: {sr_weight_path}")

    # 모델 생성
    model = SRCNN().to(device)

    # 가중치 로드
    state = torch.load(sr_weight_path, map_location=device)
    model.load_state_dict(state)

    model.eval()


    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    print("[SR] SR 생성 시작")

    img_paths = sorted(glob.glob(os.path.join(raw_dir, "*.*")))

    for idx, path in enumerate(img_paths):
        img = Image.open(path).convert("RGB")

        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(tensor)

        sr_img = sr_tensor[0].cpu().permute(1, 2, 0).numpy() * 255
        sr_img = sr_img.astype("uint8")

        save_name = f"sr_{idx:05d}.png"
        cv2.imwrite(os.path.join(sr_out_dir, save_name), sr_img)

    print("[SR] SR 생성 완료 →", sr_out_dir)
