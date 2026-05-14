"""
Temporal Shift Module (TSM) — Track A, trained from scratch.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding" (ICCV 2019). The key idea: inside a 2D ResNet, shift a
small fraction of the channels along the temporal axis. This gives the
2D CNN temporal reasoning ability for **zero extra parameters** and
negligible FLOPs.

Pipeline:
    Input:  (B, T, 3, 224, 224)
    Reshape -> (B*T, 3, 224, 224)
    ResNet-style stages, each residual block prefixed with a temporal shift
    Global average pool -> (B*T, C)
    Reshape -> (B, T, C)
    Mean pool over T -> (B, C)
    Linear -> (B, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _temporal_shift(x: torch.Tensor, num_segments: int, fold_div: int = 8) -> torch.Tensor:
    """Shift `1/fold_div` of channels forward in time, another `1/fold_div`
    backward in time, leave the rest untouched.

    x: (B*T, C, H, W)
    """
    nt, c, h, w = x.shape
    b = nt // num_segments
    x = x.view(b, num_segments, c, h, w)

    fold = c // fold_div
    out = torch.zeros_like(x)
    # shift left  (frame t gets channels from frame t+1)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    # shift right (frame t gets channels from frame t-1)
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
    # leave the rest untouched
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

    return out.view(nt, c, h, w)


class _TSMResidualBlock(nn.Module):
    """ResNet basic block with a temporal shift applied to the first conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_segments: int,
        stride: int = 1,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        self.num_segments = num_segments
        self.fold_div = fold_div

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        identity = self.shortcut(x)
        shifted = _temporal_shift(x, self.num_segments, self.fold_div)  # the magic
        out = self.relu(self.bn1(self.conv1(shifted)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class TSM(nn.Module):
    """ResNet18-style backbone with Temporal Shift Modules + temporal mean pool."""

    def __init__(
        self,
        num_classes: int,
        num_segments: int = 4,
        pretrained: bool = False,  # accepted for API compatibility — Track A ignores it
        base_channels: int = 64,
        dropout: float = 0.5,
        fold_div: int = 8,
    ) -> None:
        super().__init__()
        del pretrained
        self.num_segments = num_segments

        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def make_layer(in_c: int, out_c: int, blocks: int, stride: int) -> nn.Sequential:
            layers = [_TSMResidualBlock(in_c, out_c, num_segments, stride=stride, fold_div=fold_div)]
            for _ in range(1, blocks):
                layers.append(_TSMResidualBlock(out_c, out_c, num_segments, stride=1, fold_div=fold_div))
            return nn.Sequential(*layers)

        # ResNet18 layout: [2, 2, 2, 2]
        self.layer1 = make_layer(c, c, blocks=2, stride=1)
        self.layer2 = make_layer(c, c * 2, blocks=2, stride=2)
        self.layer3 = make_layer(c * 2, c * 4, blocks=2, stride=2)
        self.layer4 = make_layer(c * 4, c * 8, blocks=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c * 8, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, C, H, W)   with T == self.num_segments
        returns logits: (B, num_classes)
        """
        B, T, C, H, W = video.shape
        assert T == self.num_segments, (
            f"TSM was built for num_segments={self.num_segments} but got T={T}"
        )

        x = video.reshape(B * T, C, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)                   # (B*T, C_out)
        x = x.view(B, T, -1).mean(dim=1)             # temporal mean pool -> (B, C_out)
        x = self.dropout(x)
        return self.classifier(x)
