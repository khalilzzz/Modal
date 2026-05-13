"""
TimeSformer wrapper for Track B (open world).

Loads ``facebook/timesformer-base-finetuned-ssv2`` and swaps its 174-class head
to the challenge's ``num_classes`` (33).

Why TimeSformer for this challenge:
    First transformer model designed specifically for video (Facebook AI, 2021).
    Uses "divided space-time attention": each block does spatial attention then
    temporal attention. Its base variant on Hugging Face is fine-tuned on the
    Something-Something V2 task — same data family as our challenge.

Notable difference vs VideoMAE:
    TimeSformer expects exactly ``num_frames=8`` (matches the dataset default),
    while VideoMAE expects 16. So this model is the natural choice if you want
    to keep ``dataset.num_frames=8``.

Input / output:
    forward(video) takes ``(B, T, C, H, W)`` with ImageNet-normalized RGB frames
    and returns logits of shape ``(B, num_classes)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeSformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_id: str = "facebook/timesformer-base-finetuned-ssv2",
        num_frames: int = 8,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        from transformers import TimesformerConfig, TimesformerForVideoClassification

        if pretrained:
            self.model = TimesformerForVideoClassification.from_pretrained(
                model_id,
                num_labels=int(num_classes),
                num_frames=int(num_frames),
                ignore_mismatched_sizes=True,
            )
        else:
            config = TimesformerConfig(
                num_labels=int(num_classes), num_frames=int(num_frames)
            )
            self.model = TimesformerForVideoClassification(config)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """video: (B, T, C, H, W) -> logits (B, num_classes)."""
        outputs = self.model(pixel_values=video)
        return outputs.logits
