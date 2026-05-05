"""
실제 학습(Training)을 담당하는 모듈.
train.py 에서 이 함수를 호출해서 사용.
"""

import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

from models.autoencoder2 import AutoEncoder
from utils.file_utils import ensure_dir, get_timestamp
from utils.visualization import plot_loss_curve


def train_autoencoder(cfg: dict):
    paths = cfg["paths"]
    train_cfg = cfg["train"]

    img_size = train_cfg["img_size"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]

    train_dir = paths["train_dir"]
    results_dir = paths["results_dir"]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 결과 저장 폴더
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    log_dir = os.path.join(results_dir, "logs")
    plot_dir = os.path.join(results_dir, "plots")
    ensure_dir(ckpt_dir, empty=False)
    ensure_dir(log_dir, empty=False)
    ensure_dir(plot_dir, empty=False)

    timestamp = get_timestamp()

    # 데이터
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"[train] 학습용 이미지 개수: {len(train_dataset)}")

    # 모델, 손실, 옵티마이저 + ★CosineAnnealingLR 추가★
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🔥 Cosine Annealing Scheduler 추가 (중요)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1  # 최소 learning rate 1% 수준
    )

    loss_history = []

    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ★Epoch 종료 후 스케줄러 업데이트★
        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"[Epoch {epoch:03d}/{epochs}] Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # weight 저장
    ckpt_path = os.path.join(ckpt_dir, f"autoencoder_{timestamp}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[train] 가중치 저장 완료 → {ckpt_path}")

    # 로그 CSV 저장
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(loss_history, start=1):
            writer.writerow([i, l])
    print(f"[train] 로그 저장 완료 → {log_path}")

    # loss 그래프 저장
    plot_path = os.path.join(plot_dir, f"loss_curve_{timestamp}.png")
    plot_loss_curve(loss_history, plot_path)
    print(f"[train] Loss 그래프 저장 완료 → {plot_path}")

    return ckpt_path, log_path, plot_path
