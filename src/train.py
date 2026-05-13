"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_transformer import CNNTransformer
from models.two_stream_transformer import TwoStreamTransformer
from utils import build_transforms, set_seed, split_train_val

# Track B (open world) models — lazy imports inside build_model so Track A users
# don't need to install the extra transformers/huggingface dependency.


def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "cnn_transformer":
        return CNNTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            num_heads=int(cfg.model.get("num_heads", 8)),
            num_layers=int(cfg.model.get("num_layers", 2)),
            dim_feedforward=int(cfg.model.get("dim_feedforward", 1024)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )
    if name == "two_stream_transformer":
        return TwoStreamTransformer(
            num_classes=num_classes,
            pretrained=pretrained,
            base_channels=int(cfg.model.get("base_channels", 32)),
            spatial_attn_heads=int(cfg.model.get("spatial_attn_heads", 4)),
            d_model=int(cfg.model.get("d_model", 256)),
            num_heads=int(cfg.model.get("num_heads", 8)),
            num_layers=int(cfg.model.get("num_layers", 4)),
            dim_feedforward=int(cfg.model.get("dim_feedforward", 1024)),
            dropout=float(cfg.model.get("dropout", 0.2)),
        )

    # Track B (open world) models. Prefixed with "a_" by convention.
    # Lazy import keeps the transformers dependency optional for Track A.
    if name == "a_videomae":
        from models.a_videomae import VideoMAEClassifier

        return VideoMAEClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(
                cfg.model.get("model_id", "MCG-NJU/videomae-base-finetuned-ssv2")
            ),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
        )
    if name == "a_timesformer":
        from models.a_timesformer import TimeSformerClassifier

        return TimeSformerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            model_id=str(
                cfg.model.get("model_id", "facebook/timesformer-base-finetuned-ssv2")
            ),
            num_frames=int(cfg.model.get("num_frames", cfg.dataset.num_frames)),
            freeze_backbone=bool(cfg.model.get("freeze_backbone", False)),
        )

    raise ValueError(f"Unknown model.name: {name}")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        # video_batch: (B, T, C, H, W), labels: (B,)
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(video_batch)  # (B, num_classes)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    # Match normalization to pretrained flag (ImageNet stats when using pretrained weights).
    use_imagenet_norm = bool(cfg.model.pretrained)
    train_transform = build_transforms(
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        use_horizontal_flip=bool(cfg.training.get("use_horizontal_flip", False)),
        use_random_crop=bool(cfg.training.get("use_random_crop", False)),
        random_crop_scale=tuple(cfg.training.get("random_crop_scale", (0.7, 1.0))),
        use_color_jitter=bool(cfg.training.get("use_color_jitter", False)),
        color_jitter_strength=float(cfg.training.get("color_jitter_strength", 0.2)),
    )
    eval_transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    num_workers = int(cfg.training.num_workers)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            cfg.training.get("persistent_workers", False)
        )
        loader_kwargs["prefetch_factor"] = int(
            cfg.training.get("prefetch_factor", 4)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(cfg).to(device)

    best_val_accuracy = 0.0
    resume_path = cfg.training.get("resume_checkpoint")
    if resume_path:
        resume_path = Path(resume_path).resolve()
        print(f"Resuming weights from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_accuracy = float(ckpt.get("val_accuracy", 0.0))
        print(f"Resuming best val accuracy from checkpoint: {best_val_accuracy:.4f}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_name = str(cfg.training.get("optimizer", "adam")).lower()
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    lr = float(cfg.training.lr)
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown training.optimizer: {optimizer_name}")
    print(
        f"Optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})"
    )
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    use_amp = bool(cfg.training.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    for epoch in range(int(cfg.training.epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, loss_fn, device, use_amp
        )

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
