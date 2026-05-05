"""
train.py 실행 시 동작:
1) raw 데이터를 train / test/normal 로 split
2) test/normal → test/abnormal 이상 이미지 생성
3) AutoEncoder 학습
4) weight / 로그 / loss 그래프 저장
"""

from utils.file_utils import load_config
from split_train_test import split_train_test
from make_anomalies import main as make_anomalies_main
from trainer import train_autoencoder
#from apply_sr import apply_sr_to_raw

def main():
    cfg = load_config("config.yaml")
    
    #print("===SR적용===")
    #apply_sr_to_raw(cfg)

    #print("==== 1) Train/Test Split ====")
    #split_train_test(cfg)

    #print("\n==== 2) Abnormal Image 생성 ====")
    #make_anomalies_main()

    print("\n==== 3) AutoEncoder 학습 ====")
    ckpt_path, log_path, plot_path = train_autoencoder(cfg)

    print("\n==== 완료 요약 ====")
    print(f"  - Checkpoint : {ckpt_path}")
    print(f"  - Train Log  : {log_path}")
    print(f"  - Loss Plot  : {plot_path}")


if __name__ == "__main__":
    main()
