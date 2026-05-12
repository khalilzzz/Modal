"""
CNN + Temporal Transformer (Track A — trained from scratch).

Forward:
    Input:  (B, T, C, H, W)
    Frame CNN (ResNet18): (B*T, C, H, W) -> (B*T, 512)
    Sequence: (B, T, 512)
    + sinusoidal positional encoding
    Transformer encoder: (B, T, 512) -> (B, T, 512)
    Mean pool over time: (B, 512)
    Linear: (B, num_classes)

See cnn_transformer_design.md for the full rationale.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models


class _SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal position signal to a (B, T, d_model) tensor."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        d_model = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.pos_enc = _SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video_batch.shape

        frames = video_batch.reshape(B * T, C, H, W)
        frame_features = self.backbone(frames)              # (B*T, 512)
        frame_features = torch.flatten(frame_features, 1)  # ensures (B*T, 512)

        sequence = frame_features.view(B, T, -1)           # (B, T, 512)
        sequence = self.pos_enc(sequence)                   # (B, T, 512)

        attended = self.transformer(sequence)               # (B, T, 512)

        pooled = attended.mean(dim=1)                       # (B, 512)
        return self.classifier(pooled)                      # (B, num_classes)
