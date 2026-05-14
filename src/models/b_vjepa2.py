"""
V-JEPA 2 wrapper for Track B (open world).

Loads a V-JEPA 2 SSv2-finetuned checkpoint (default
``facebook/vjepa2-vitl-fpc16-256-ssv2``), swaps the 174-class head to the
challenge's ``num_classes`` (33), and exposes the standard
``(B, T, C, H, W) -> (B, num_classes)`` interface used everywhere in this repo.

Why V-JEPA 2:
    Meta's V-JEPA 2 (2025) is the current public state-of-the-art on
    Something-Something V2. The pretraining is self-supervised on a very large
    video corpus, and the finetune is on the exact data distribution as the
    challenge. Expect a clear lift vs TimeSformer-base / VideoMAE-base.

Input adaptation:
    The fpc16 / 256 checkpoint expects 16 frames at 256x256. The repo's
    ``processed_data`` provides 4 frames at 224x224. The wrapper handles both:

    - Temporal: replicates each input frame ``native_frames // T`` times along
      the time axis. Order is preserved (no shuffle). With T=4 and native=16,
      ``[f0, f1, f2, f3] -> [f0,f0,f0,f0, f1,f1,f1,f1, f2,f2,f2,f2, f3,f3,f3,f3]``.
      This is the safest "downscale": no need to monkey-patch internal HF
      attributes (which have moved between transformers versions).

    - Spatial: bilinearly upsamples ``(B*T, C, 224, 224) -> (B*T, C, 256, 256)``
      if the input size doesn't match ``image_size``.

Note: ImageNet normalization (``build_transforms(use_imagenet_norm=True)``) is
correct for V-JEPA 2 checkpoints.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


class VJEPA2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
        num_frames: int = 4,
        image_size: int = 256,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Local import so Track A users don't need transformers installed.
        try:
            from transformers import AutoModelForVideoClassification
        except ImportError as e:
            raise ImportError(
                "V-JEPA 2 requires a recent `transformers` (>= 4.55). "
                "Run `uv sync` or `uv pip install -U transformers`."
            ) from e

        if not pretrained:
            raise ValueError(
                "V-JEPA 2 from-scratch is not useful here — set "
                "`model.pretrained=true` or use cnn_baseline / cnn_transformer "
                "for from-scratch ablations."
            )

        self.model = AutoModelForVideoClassification.from_pretrained(
            model_id,
            num_labels=int(num_classes),
            ignore_mismatched_sizes=True,
        )

        # The native frame count attribute name has varied across V-JEPA 2
        # transformers releases ("frames_per_clip", "num_frames"). Check both.
        cfg = self.model.config
        native_frames = int(
            getattr(cfg, "frames_per_clip", getattr(cfg, "num_frames", 16))
        )
        self.native_frames = native_frames
        self.target_frames = int(num_frames)
        self.image_size = int(image_size)

        if self.native_frames % self.target_frames != 0:
            raise ValueError(
                f"V-JEPA 2 native num_frames={self.native_frames} is not a "
                f"multiple of dataset num_frames={self.target_frames}. Pick a "
                "num_frames that divides the native count (e.g. 4 or 8 for 16)."
            )
        self._repeat = self.native_frames // self.target_frames

        # The forward kwarg for video has moved from `pixel_values` to
        # `pixel_values_videos` in newer transformers. Detect once at init.
        params = inspect.signature(self.model.forward).parameters
        self._video_kwarg = (
            "pixel_values_videos" if "pixel_values_videos" in params else "pixel_values"
        )

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits (B, num_classes)."""
        B, T, C, H, W = video.shape

        # Temporal: replicate frames to reach the native count, preserving order.
        if T != self.native_frames:
            video = video.repeat_interleave(self._repeat, dim=1)

        # Spatial: resize to the model's expected input size.
        if H != self.image_size or W != self.image_size:
            _, T2, C2, _, _ = video.shape
            flat = video.reshape(B * T2, C2, H, W)
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            video = flat.reshape(B, T2, C2, self.image_size, self.image_size)

        outputs = self.model(**{self._video_kwarg: video})
        return outputs.logits
