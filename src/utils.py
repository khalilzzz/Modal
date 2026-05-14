"""
Small helpers: reproducibility, clip-level transforms, and metrics.

The transforms returned by ``build_transforms`` operate on a **list of PIL
frames** rather than a single image. All random parameters (crop, flip, color
jitter, random erasing) are sampled ONCE per clip and applied identically to
every frame. This preserves temporal consistency, which matters a lot for
Something-Something where most of the signal is in inter-frame motion.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# A clip transform maps a list of PIL frames to a list of (C, H, W) tensors.
ClipTransform = Callable[[List[Image.Image]], List[torch.Tensor]]


def _make_normalize(use_imagenet_norm: bool) -> transforms.Normalize:
    if use_imagenet_norm:
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def _sample_erase_rect(
    tensor: torch.Tensor,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    max_attempts: int = 10,
) -> Optional[Tuple[int, int, int, int]]:
    """Sample (i, j, h, w) for a random-erasing rectangle, or None if no valid
    rectangle was found within max_attempts. Same shape returned per clip,
    applied identically to every frame.
    """
    _, H, W = tensor.shape
    area = H * W
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    for _ in range(max_attempts):
        target_area = area * float(torch.empty(()).uniform_(scale[0], scale[1]))
        aspect = math.exp(float(torch.empty(()).uniform_(log_lo, log_hi)))
        h = int(round(math.sqrt(target_area * aspect)))
        w = int(round(math.sqrt(target_area / aspect)))
        if 0 < h < H and 0 < w < W:
            i = int(torch.randint(0, H - h + 1, (1,)).item())
            j = int(torch.randint(0, W - w + 1, (1,)).item())
            return i, j, h, w
    return None


class _TrainClipTransform:
    """Train-time augmentation. All random params sampled once per clip and
    applied identically to every frame.
    """

    def __init__(
        self,
        image_size: int,
        use_imagenet_norm: bool,
        use_horizontal_flip: bool,
        use_random_crop: bool,
        random_crop_scale: Tuple[float, float],
        random_crop_ratio: Tuple[float, float],
        use_color_jitter: bool,
        color_jitter_strength: float,
        use_random_erasing: bool,
        random_erasing_p: float,
        random_erasing_scale: Tuple[float, float],
        random_erasing_ratio: Tuple[float, float],
    ) -> None:
        self.image_size = image_size
        self.use_horizontal_flip = use_horizontal_flip
        self.use_random_crop = use_random_crop
        self.random_crop_scale = tuple(random_crop_scale)
        self.random_crop_ratio = tuple(random_crop_ratio)
        self.use_color_jitter = use_color_jitter
        self.color_jitter_strength = float(color_jitter_strength)
        self.use_random_erasing = use_random_erasing
        self.random_erasing_p = float(random_erasing_p)
        self.random_erasing_scale = tuple(random_erasing_scale)
        self.random_erasing_ratio = tuple(random_erasing_ratio)

        self.normalize = _make_normalize(use_imagenet_norm)

        if use_color_jitter:
            s = self.color_jitter_strength
            self._color_jitter = transforms.ColorJitter(
                brightness=s, contrast=s, saturation=s, hue=s * 0.5
            )
        else:
            self._color_jitter = None

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        size = self.image_size
        first = frames[0]

        # ---- Sample shared parameters once per clip ----
        if self.use_random_crop:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                first,
                scale=self.random_crop_scale,
                ratio=self.random_crop_ratio,
            )
        else:
            i = j = 0
            w_px, h_px = first.size  # PIL: (W, H)
            h, w = h_px, w_px

        do_flip = self.use_horizontal_flip and (torch.rand(()).item() < 0.5)

        if self._color_jitter is not None:
            fn_idx, b_f, c_f, s_f, h_f = self._color_jitter.get_params(
                self._color_jitter.brightness,
                self._color_jitter.contrast,
                self._color_jitter.saturation,
                self._color_jitter.hue,
            )
        else:
            fn_idx = ()
            b_f = c_f = s_f = h_f = None

        # ---- Apply per frame with the SHARED params ----
        out: List[torch.Tensor] = []
        for img in frames:
            if self.use_random_crop:
                img = TF.resized_crop(img, i, j, h, w, [size, size])
            else:
                img = TF.resize(img, [size, size])

            if do_flip:
                img = TF.hflip(img)

            if self._color_jitter is not None:
                for fn_id in fn_idx:
                    fn_id_int = int(fn_id)
                    if fn_id_int == 0 and b_f is not None:
                        img = TF.adjust_brightness(img, b_f)
                    elif fn_id_int == 1 and c_f is not None:
                        img = TF.adjust_contrast(img, c_f)
                    elif fn_id_int == 2 and s_f is not None:
                        img = TF.adjust_saturation(img, s_f)
                    elif fn_id_int == 3 and h_f is not None:
                        img = TF.adjust_hue(img, h_f)

            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)
            out.append(tensor)

        # ---- Random erasing: shared rectangle across the whole clip ----
        if (
            self.use_random_erasing
            and torch.rand(()).item() < self.random_erasing_p
        ):
            rect = _sample_erase_rect(
                out[0], self.random_erasing_scale, self.random_erasing_ratio
            )
            if rect is not None:
                ei, ej, eh, ew = rect
                for t in out:
                    t[:, ei : ei + eh, ej : ej + ew] = 0.0

        return out


class _EvalClipTransform:
    """Eval-time transform: deterministic resize + ToTensor + Normalize, same
    on every frame.
    """

    def __init__(self, image_size: int, use_imagenet_norm: bool) -> None:
        self.image_size = image_size
        self.normalize = _make_normalize(use_imagenet_norm)

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        size = self.image_size
        out: List[torch.Tensor] = []
        for img in frames:
            img = TF.resize(img, [size, size])
            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)
            out.append(tensor)
        return out


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    use_horizontal_flip: bool = True,
    use_random_crop: bool = False,
    random_crop_scale: Tuple[float, float] = (0.7, 1.0),
    random_crop_ratio: Tuple[float, float] = (0.85, 1.15),
    use_color_jitter: bool = False,
    color_jitter_strength: float = 0.2,
    use_random_erasing: bool = False,
    random_erasing_p: float = 0.25,
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2),
    random_erasing_ratio: Tuple[float, float] = (0.3, 3.3),
) -> ClipTransform:
    """Build a clip-level augmentation pipeline.

    Returns a callable mapping ``List[PIL.Image] -> List[torch.Tensor]``. All
    random parameters are sampled once per clip and applied identically to
    every frame, so the temporal motion within a clip is preserved.
    """
    if is_training:
        return _TrainClipTransform(
            image_size=image_size,
            use_imagenet_norm=use_imagenet_norm,
            use_horizontal_flip=use_horizontal_flip,
            use_random_crop=use_random_crop,
            random_crop_scale=random_crop_scale,
            random_crop_ratio=random_crop_ratio,
            use_color_jitter=use_color_jitter,
            color_jitter_strength=color_jitter_strength,
            use_random_erasing=use_random_erasing,
            random_erasing_p=random_erasing_p,
            random_erasing_scale=random_erasing_scale,
            random_erasing_ratio=random_erasing_ratio,
        )
    return _EvalClipTransform(
        image_size=image_size, use_imagenet_norm=use_imagenet_norm
    )


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    """Compute top-k correctness for each k in topk.

    logits: (batch_size, num_classes)
    targets: (batch_size,) integer class indices
    Returns a tuple of tensors, each shape (1,) with accuracy in [0, 1].
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """Shuffle then split (video_path, label) pairs into train/val portions."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples
