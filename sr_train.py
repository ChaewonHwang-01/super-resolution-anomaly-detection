"""
train.py 실행 시 동작:
0) train/normal RGB 폴더 → Y 폴더 생성 (make_y)
1) raw 데이터를 train / test/normal 로 split (옵션)
2) test/normal → test/abnormal 이상 이미지 생성 (옵션)
3) SR2 + AutoEncoder 학습 (SRCNN pretrained 사용)
4) weight / 로그 / loss 그래프 저장
"""

from utils.file_utils import load_config

# (옵션) 네가 쓰던 파이프라인 유지
# from split_train_test import split_train_test
# from make_anomalies import main as make_anomalies_main

from utils.make_y import main as make_y_main          # <- 너가 만든 Y 변환 코드
from sr_trainer import train_sr2_ae                 # <- SR2_AE trainer


def main():
    cfg = load_config("config.yaml")

    print("\n==== 0) RGB → Y dataset 생성 ====")
    # make_y_dataset.py에서 cfg 기반으로 SRC_DIR/OUT_DIR를 잡아 저장하도록
    #make_y_main(cfg)

    # print("==== 1) Train/Test Split ====")
    # split_train_test(cfg)

    # print("\n==== 2) Abnormal Image 생성 ====")
    # make_anomalies_main()

    print("\n==== 3) SR2 + AutoEncoder 학습 ====")
    ckpt_path, log_path, plot_path = train_sr2_ae(cfg)

    print("\n==== 완료 요약 ====")
    print(f"  - Checkpoint : {ckpt_path}")
    print(f"  - Train Log  : {log_path}")
    print(f"  - Loss Plot  : {plot_path}")


if __name__ == "__main__":
    main()
