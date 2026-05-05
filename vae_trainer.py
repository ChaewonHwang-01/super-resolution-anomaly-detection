"""
실제 학습(Training)을 담당하는 모듈.
train.py 에서 이 함수를 호출해서 사용.

- 입력(256) -> SRCNN(x2, pretrained) -> 512
- VAE: 512 -> 256 -> 128 -> 64 -> 32 -> 64 -> 128 -> 256 -> 512
- loss: recon_loss(recon_512, sr_512) + kld_weight * KL(mu, logvar)
- 기본: sr_512 detach 해서 VAE만 학습
"""

import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.vae import SR2_VAE   # <- 네 VAE 모델 파일 경로에 맞게 수정
from utils.file_utils import ensure_dir, get_timestamp
from utils.visualization import plot_loss_curve


def _load_state_dict_flexible(model: nn.Module, ckpt_path: str):
    """
    지원:
    - state_dict만 저장한 .pth
    - {"model": state_dict} / {"state_dict": state_dict} 형태
    - DataParallel(module.) prefix 제거
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    model.load_state_dict(new_state, strict=True)


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    recon_type: str = "mse",
    kld_weight: float = 1e-4,
):
    """
    recon_x: VAE output
    x: reconstruction target
    mu, logvar: latent distribution params
    recon_type: "mse" or "l1"
    kld_weight: beta for KL divergence
    """
    if recon_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    elif recon_type == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_type: {recon_type}")

    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kld_weight * kld_loss
    return total_loss, recon_loss, kld_loss


def train_sr2_vae(cfg: dict):
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    img_size = train_cfg["img_size"]          # 256
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["lr"]

    train_dir = paths["train_dir"]
    results_dir = paths["results_dir"]

    # VAE 전용 옵션
    recon_type = train_cfg.get("recon_type", "mse")         # "mse" or "l1"
    kld_weight = float(train_cfg.get("kld_weight", 1e-4))  # beta
    grad_clip = train_cfg.get("grad_clip", None)           # 예: 1.0
    save_mode = train_cfg.get("save_mode", "vae_only")     # "vae_only" or "full"

    # device
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

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"[train] 학습용 이미지 개수: {len(train_dataset)}")

    # -------------------------
    # 모델 구성
    # -------------------------
    # YAML 예시:
    # model:
    #   in_channels: 3
    #   scale: 2
    #   srcnn_fn: 32
    #   srcnn_dfn: 64
    #   vae_base_channels: 64
    #   vae_max_channels: 256
    #   vae_latent_channels: 256
    #   srcnn_pretrained_path: "/path/to/srcnn.pth"
    #   freeze_srcnn: true
    in_channels = model_cfg.get("in_channels", 3)
    scale = model_cfg.get("scale", 2)

    model = SR2_VAE(
        in_channels=in_channels,
        scale=scale,
        srcnn_fn=model_cfg.get("srcnn_fn", 32),
        srcnn_dfn=model_cfg.get("srcnn_dfn", 64),
        vae_base_channels=model_cfg.get("vae_base_channels", 64),
        vae_max_channels=model_cfg.get("vae_max_channels", 256),
        vae_latent_channels=model_cfg.get("vae_latent_channels", 256),
    ).to(device)

    # SRCNN pretrained 로드
    srcnn_pretrained_path = model_cfg.get("srcnn_pretrained_path", "")
    if srcnn_pretrained_path:
        print(f"[train] SRCNN pretrained 로드: {srcnn_pretrained_path}")
        _load_state_dict_flexible(model.srcnn, srcnn_pretrained_path)
    else:
        print("[train][WARN] srcnn_pretrained_path가 비어있음. (SRCNN이 랜덤 초기화 상태)")

    # SRCNN freeze
    freeze_srcnn = bool(model_cfg.get("freeze_srcnn", True))
    if freeze_srcnn:
        for p in model.srcnn.parameters():
            p.requires_grad = False
        print("[train] SRCNN frozen: VAE만 학습합니다.")
        optim_params = model.vae.parameters()
    else:
        print("[train] SRCNN trainable: end-to-end로 함께 학습합니다.")
        optim_params = model.parameters()

    # optimizer / scheduler
    optimizer = optim.Adam(optim_params, lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.1
    )

    loss_history = []
    recon_history = []
    kld_history = []

    # -------------------------
    # 학습 루프
    # -------------------------
    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0

        for imgs_256, _ in train_loader:
            imgs_256 = imgs_256.to(device)

            optimizer.zero_grad()

            sr_512, recon_512, mu, logvar = model(imgs_256)

            # VAE만 학습하는 기본 모드에서는 sr_512 detach
            target = sr_512.detach() if freeze_srcnn else sr_512

            loss, recon_loss, kld_loss = vae_loss_function(
                recon_x=recon_512,
                x=target,
                mu=mu,
                logvar=logvar,
                recon_type=recon_type,
                kld_weight=kld_weight,
            )

            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=grad_clip)

            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld_loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        avg_recon = running_recon / len(train_loader)
        avg_kld = running_kld / len(train_loader)

        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
        kld_history.append(avg_kld)

        print(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"Loss: {avg_loss:.6f}, "
            f"Recon: {avg_recon:.6f}, "
            f"KLD: {avg_kld:.6f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # -------------------------
    # 저장
    # -------------------------
    if save_mode == "vae_only":
        ckpt_path = os.path.join(ckpt_dir, f"sr2_vae_only_{timestamp}.pth")
        torch.save(model.vae.state_dict(), ckpt_path)
        print(f"[train] VAE 가중치만 저장 완료 → {ckpt_path}")
    else:
        ckpt_path = os.path.join(ckpt_dir, f"sr2_vae_full_{timestamp}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[train] 전체 모델 가중치 저장 완료 → {ckpt_path}")

    # 로그 CSV 저장
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "recon_loss", "kld_loss"])
        for i, (l, r, k) in enumerate(zip(loss_history, recon_history, kld_history), start=1):
            writer.writerow([i, l, r, k])
    print(f"[train] 로그 저장 완료 → {log_path}")

    # total loss 그래프 저장
    plot_path = os.path.join(plot_dir, f"loss_curve_{timestamp}.png")
    plot_loss_curve(loss_history, plot_path)
    print(f"[train] Loss 그래프 저장 완료 → {plot_path}")

    return ckpt_path, log_path, plot_path
