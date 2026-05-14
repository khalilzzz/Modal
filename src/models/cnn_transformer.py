"""
CNN + Temporal Transformer for action anticipation (Track A — from scratch).

Forward:
    Input:  (B, T, C, H, W)
    Frame CNN (ResNet18):
        - if use_spatial_tokens: keep the 7x7 spatial grid -> (B*T, 512, 7, 7)
        - else:                  global average pool       -> (B*T, 512)
    Sequence: (B, T*S + 1, 512)  where S = 49 (spatial tokens) or 1 (global)
    + learnable spatial + temporal positional embeddings
    + optional learnable anticipation token appended at the end
    + optional causal mask (each frame only attends to its past)
    Transformer encoder -> LayerNorm -> Linear(num_classes)

Design choices that matter for anticipation (vs. recognition):
- Mean-pooling over time drowns the most recent (most predictive) frame in
  earlier context. We default to pooling from a learnable anticipation token
  appended at the end of the sequence — this lets the model learn the right
  per-frame weights instead of forcing a uniform mean.
- Keeping the 7x7 spatial grid lets the transformer reason about where the
  hand and the object are across frames, which matters a lot on SSv2.
- The dataset only provides the early frames of each clip, so bidirectional
  attention across these observed frames does not leak the future. A causal
  mask is exposed as an opt-in for experiments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_frames: int = 4,
        use_spatial_tokens: bool = True,
        causal: bool = False,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        if pool not in {"cls", "last", "mean"}:
            raise ValueError(f"pool must be one of cls/last/mean, got {pool!r}")

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        d_model = backbone.fc.in_features  # 512

        if use_spatial_tokens:
            # Drop avgpool + fc to keep the (512, 7, 7) feature map per frame.
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            spatial_size = 7  # 224x224 input -> 7x7 after ResNet18
        else:
            backbone.fc = nn.Identity()
            self.backbone = backbone
            spatial_size = 1

        self.d_model = d_model
        self.num_frames = num_frames
        self.use_spatial_tokens = use_spatial_tokens
        self.causal = causal
        self.pool = pool
        self.num_spatial_tokens = spatial_size * spatial_size

        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, d_model))
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, 1, self.num_spatial_tokens, d_model)
        )
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def _causal_mask(
        self, T: int, S: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean attention mask. True entries are blocked.

        Sequence layout:
            pool == "cls":  [frame_0(S tokens), ..., frame_{T-1}(S tokens), CLS]
            otherwise:      [frame_0(S tokens), ..., frame_{T-1}(S tokens)]

        Within a frame: full attention (all S tokens see each other).
        Across frames: token at time t only attends to tokens at times <= t.
        CLS (if present, last position) attends to all frames; no frame token
        may attend back to CLS.
        """
        L_frames = T * S
        t_idx = torch.arange(L_frames, device=device) // S          # (L_frames,)
        blocked = t_idx[None, :] > t_idx[:, None]                   # True = future of i
        if self.pool == "cls":
            L = L_frames + 1
            mask = torch.zeros((L, L), dtype=torch.bool, device=device)
            mask[:L_frames, :L_frames] = blocked
            mask[:L_frames, L_frames] = True   # frames cannot see CLS
            return mask
        return blocked

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """video_batch: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video_batch.shape
        if T > self.num_frames:
            raise ValueError(
                f"Got T={T} input frames, but the model was built for num_frames="
                f"{self.num_frames}. Set dataset.num_frames accordingly."
            )

        frames = video_batch.reshape(B * T, C, H, W)

        if self.use_spatial_tokens:
            feat = self.backbone(frames)                  # (B*T, D, h, w)
            _, D, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2)        # (B*T, h*w, D)
            seq = feat.view(B, T, h * w, D)               # (B, T, S, D)
        else:
            feat = self.backbone(frames)                  # (B*T, D)
            feat = torch.flatten(feat, 1)
            seq = feat.view(B, T, 1, -1)                  # (B, T, 1, D)

        S = seq.size(2)
        D = seq.size(3)

        seq = seq + self.temporal_pos[:, :T] + self.spatial_pos[:, :, :S]
        seq = seq.reshape(B, T * S, D)

        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            seq = torch.cat([seq, cls], dim=1)            # (B, T*S + 1, D)

        seq = self.pos_drop(seq)

        attn_mask: Optional[torch.Tensor] = None
        if self.causal:
            attn_mask = self._causal_mask(T, S, seq.device)
        attended = self.transformer(seq, mask=attn_mask)

        attended = self.final_norm(attended)

        if self.pool == "cls":
            pooled = attended[:, -1]                       # anticipation token
        elif self.pool == "last":
            pooled = attended[:, -S:].mean(dim=1)          # last observed frame
        else:  # "mean"
            pooled = attended.mean(dim=1)

        return self.classifier(pooled)
