"""
data/test/ 안의 이미지 중 일부(50%)를 abnormal로 변환하여
동일한 data/test/ 안에 test_xxxxx.png 형식으로 저장.
"""

from utils.file_utils import load_config
from utils.anormaly_utils import generate_abnormal_tests


def main():
    cfg = load_config("config.yaml")

    test_dir = cfg["paths"]["test_dir"]     # normal/abnormal 분리된 폴더 아님!
    anomaly_cfg = cfg["anomaly"]

    generate_abnormal_tests(
        test_dir=test_dir,
        cfg_anom=anomaly_cfg,
        resize=128,
    )


if __name__ == "__main__":
    main()
