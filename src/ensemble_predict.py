#!/usr/bin/env python3
"""
Ensemble prediction / evaluation: combine the softmax outputs of N trained
checkpoints. Two modes:

  - **Submission mode** (default): runs over ``dataset.test_dir`` (no labels)
    and writes a CSV in the same format as ``create_submission.py``
    (``video_name,predicted_class``).

  - **Eval mode** (``--eval-dir PATH``): runs over a labelled directory
    (same layout as ``dataset.val_dir``) and reports **top-1 / top-5**
    accuracy for each individual model AND for the ensemble. No CSV.

Combination strategies (``--strategy``):

  - ``uniform``     — equal weights (1/N per model).
  - ``scalar``      — use ``--weights`` (1 scalar per model, same across classes).
  - ``temp_scalar`` — fit a temperature ``T_i`` per model on ``--calibration-dir``
                      (L-BFGS on NLL), then weight by ``(val_acc_i ** alpha)``.
                      Captures most of the per-class-weighting gain without the
                      overfit risk.

Reuses helpers from ``create_submission.py`` by import — no edits to that file.

Examples:
    # Uniform mean of 3 checkpoints
    uv run python src/ensemble_predict.py \\
        --checkpoints a.pt b.pt c.pt \\
        --strategy uniform

    # Manual scalar weights
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm.pt cnn_t.pt vjepa2.pt \\
        --strategy scalar --weights 1.0 1.0 4.0

    # Temperature + power weights, calibrated on val, submit on test
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm.pt cnn_t.pt vjepa2.pt \\
        --strategy temp_scalar --alpha 2.0 \\
        --calibration-dir processed_data/val \\
        --output submission_ensemble.csv

    # Same but in eval mode (note: same dir for cal+eval → biased estimate)
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm.pt cnn_t.pt vjepa2.pt \\
        --strategy temp_scalar --alpha 2.0 \\
        --calibration-dir processed_data/val \\
        --eval-dir processed_data/val
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, compute_prior_logits

# Reuse helpers from create_submission without modifying it.
from create_submission import (
    discover_all_test_videos,
    load_manifest_video_names,
    resolve_video_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to one or more .pt checkpoints to ensemble.",
    )
    p.add_argument(
        "--strategy",
        choices=["uniform", "scalar", "temp_scalar"],
        default="uniform",
        help="How to combine per-model softmaxes. See module docstring.",
    )
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-model scalar weights (required for --strategy scalar). "
        "Ignored for uniform and temp_scalar.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Power exponent applied to per-model calibration accuracy when "
        "--strategy temp_scalar. Higher α favors the stronger models more "
        "aggressively. Typical: 1.0–3.0. Default: 2.0.",
    )
    p.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Labelled directory used to fit temperatures and compute per-model "
        "weights (required for --strategy temp_scalar). Same layout as val_dir. "
        "If equal to --eval-dir, the calibration set IS the eval set — reported "
        "accuracies will be optimistic.",
    )
    p.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="If set: evaluate on this labelled directory (same layout as "
        "dataset.val_dir), print top-1 / top-5 per model AND ensemble. "
        "No CSV is written in this mode.",
    )
    p.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Submission mode: path to test root. Defaults to dataset.test_dir "
        "from the FIRST checkpoint's config. Ignored if --eval-dir is set.",
    )
    p.add_argument(
        "--test-manifest",
        type=str,
        default=None,
        help="Submission mode: optional manifest CSV with a 'video_name' column "
        "for clip ordering. Ignored if --eval-dir is set.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="submission_ensemble.csv",
        help="Submission mode: output CSV path.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--calibrate-prior",
        action="store_true",
        help="Subtract log P(y) (train-set class frequency) from each model's "
        "raw logits before softmax. Removes the inherited class-imbalance "
        "bias; recovers recall on under-represented twins (e.g. 011 vs 014). "
        "Pure post-hoc transform; no checkpoint or model change needed.",
    )
    p.add_argument(
        "--prior-train-dir",
        type=str,
        default=None,
        help="Source of class counts for --calibrate-prior. Defaults to "
        "dataset.train_dir from the first checkpoint's saved config.",
    )
    p.add_argument(
        "--prior-alpha",
        type=float,
        default=1.0,
        help="Strength of the prior calibration in [0, 1]. 1.0 = full Bayes "
        "(subtract log P(y_train)); 0.5 = half-strength; 0.0 = no calibration. "
        "Use a lower value (e.g. 0.5) if full calibration over-corrects and "
        "top-1 drops too much. Ignored unless --calibrate-prior is set.",
    )
    p.add_argument(
        "--save-per-model-softmax",
        action="store_true",
        help="Also save each model's individual softmax tensor (B, num_classes) "
        "to disk under a per_model/ folder next to --output (submission mode) "
        "or in the current directory (eval mode).",
    )
    return p.parse_args()


def _load_model_and_meta(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a checkpoint and return (model, meta) where meta carries the runtime
    knobs needed to build a matching DataLoader (num_frames, pretrained, ...).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' entry. "
            "Re-train with the current train.py so the Hydra config is saved."
        )
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    meta = {
        "num_frames": int(ckpt.get("num_frames", cfg.dataset.num_frames)),
        "pretrained": bool(ckpt.get("pretrained", cfg.model.pretrained)),
        "config": cfg,
        "model_name": str(cfg.model.name),
    }
    return model, meta


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
    tag: str = "",
    prior_logits: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference; return (softmax, labels) both on CPU.

    Labels are whatever the dataset's sample_list contains (true labels for a
    labelled set, dummy 0s in submission mode).

    If ``prior_logits`` is provided, it is subtracted from the raw logits
    before softmax — i.e. inference-time prior calibration.
    """
    model.eval()
    probs_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    prefix = f"    [{tag}] " if tag else "    "
    for batch_idx, (video_batch, labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        logits = model(video_batch)
        if prior_logits is not None:
            logits = logits - prior_logits
        probs = F.softmax(logits, dim=-1).cpu()
        probs_chunks.append(probs)
        label_chunks.append(labels.cpu())
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(f"{prefix}inference batch {batch_idx}/{n_batches}", flush=True)
    out = torch.cat(probs_chunks, dim=0)
    labels_out = torch.cat(label_chunks, dim=0)
    if out.size(0) != total_videos:
        raise RuntimeError(
            f"Got {out.size(0)} softmax rows but expected {total_videos} clips."
        )
    return out, labels_out


def _topk_accuracy(softmax: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Top-k accuracy in [0, 1]."""
    _, topk = softmax.topk(k, dim=-1, largest=True, sorted=True)
    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
    return float(correct.float().mean().item())


def _fit_temperature(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single scalar temperature T on (probs, labels) via L-BFGS to
    minimize cross-entropy.

    Trick: softmax is translation-invariant in the logit space, so using
    log(probs) as 'pseudo-logits' is equivalent to using the real model
    logits (they differ by a per-sample constant that softmax discards).
    This lets us calibrate temperature without re-running inference to get
    raw logits.

    Returns:
        Optimal temperature T. T > 1 softens overconfident outputs;
        T < 1 sharpens an underconfident model.
    """
    eps = 1e-12
    log_probs = probs.clamp_min(eps).log().double()  # (N, C)
    labels_l = labels.long()
    # Optimize log_T so T = exp(log_T) stays strictly positive.
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_T], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(log_probs / T, labels_l)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def _apply_temperature(probs: torch.Tensor, T: float) -> torch.Tensor:
    """Apply temperature T to a softmax distribution.

    softmax(log(p) / T) gives the tempered distribution, equivalent to
    softmax(logits / T) up to a per-sample constant in the logits (which
    softmax discards).
    """
    if abs(T - 1.0) < 1e-9:
        return probs
    eps = 1e-12
    return F.softmax(probs.clamp_min(eps).log() / float(T), dim=-1)


def _resolve_test_root(args: argparse.Namespace) -> Path:
    """Get test_dir from --test-dir if given, else from first checkpoint's saved config."""
    if args.test_dir is not None:
        return Path(args.test_dir).resolve()
    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu")
    cfg = OmegaConf.create(first_ckpt["config"])
    return Path(cfg.dataset.test_dir).resolve()


def _print_agreement_stats(
    per_model_preds: Dict[str, List[int]],
    ensemble_preds: List[int],
    num_classes: int,
) -> None:
    """Diagnostic: how often models agree among themselves and with the ensemble."""
    n_clips = len(ensemble_preds)
    n_models = len(per_model_preds)

    print(f"\n=== Ensemble agreement stats ({n_models} model(s), {n_clips} clip(s)) ===")
    if n_models >= 2:
        per_model_tensor = torch.tensor(list(per_model_preds.values()))
        first_row = per_model_tensor[0:1]
        all_agree = (per_model_tensor == first_row).all(dim=0).sum().item()
        print(
            f"All models agree on:      {all_agree}/{n_clips} clips "
            f"({100*all_agree/n_clips:.1f}%)"
        )
        ensemble_tensor = torch.tensor(ensemble_preds)
        for i, name in enumerate(per_model_preds.keys()):
            match = (per_model_tensor[i] == ensemble_tensor).sum().item()
            print(
                f"  {name:30s} agrees with ensemble on {100*match/n_clips:5.1f}% of clips"
            )
    bins = torch.bincount(torch.tensor(ensemble_preds), minlength=num_classes)
    distinct = (bins > 0).sum().item()
    top_class = int(bins.argmax().item())
    top_count = int(bins.max().item())
    print(f"Distinct classes in ensemble predictions: {distinct}/{num_classes}")
    print(f"Most-predicted class index: {top_class}  ({top_count} clips)")


def _print_accuracy_metrics(
    per_model_softmax: Dict[str, torch.Tensor],
    ensemble_softmax: torch.Tensor,
    true_labels: torch.Tensor,
) -> None:
    """Print top-1 / top-5 for each individual model and for the ensemble."""
    n_clips = int(true_labels.size(0))
    print(f"\n=== Accuracy on {n_clips} labelled clips ===")
    print(f"  {'model':30s}   top-1     top-5")
    print(f"  {'-'*30}   ------    ------")
    for name, probs in per_model_softmax.items():
        t1 = _topk_accuracy(probs, true_labels, 1)
        t5 = _topk_accuracy(probs, true_labels, 5)
        print(f"  {name:30s}   {t1:.4f}    {t5:.4f}")
    e_t1 = _topk_accuracy(ensemble_softmax, true_labels, 1)
    e_t5 = _topk_accuracy(ensemble_softmax, true_labels, 5)
    print(f"  {'-'*30}   ------    ------")
    print(f"  {'ENSEMBLE':30s}   {e_t1:.4f}    {e_t5:.4f}")


def _build_loader(
    clip_root: Path,
    sample_list: List[Tuple[Path, int]],
    num_frames: int,
    use_imagenet_norm: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    transform = build_transforms(
        is_training=False, use_imagenet_norm=use_imagenet_norm
    )
    dataset = VideoFrameDataset(
        root_dir=clip_root,
        num_frames=num_frames,
        transform=transform,
        sample_list=sample_list,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def _compute_weights(
    strategy: str,
    n_models: int,
    user_weights: Optional[List[float]],
    per_model_cal_softmax: Optional[Dict[str, torch.Tensor]],
    cal_labels: Optional[torch.Tensor],
    alpha: float,
) -> Tuple[List[float], Optional[Dict[str, float]]]:
    """Resolve (weights, temperatures) according to the chosen strategy.

    Returns:
        weights: list of float, one per checkpoint (in the input order).
        temperatures: dict {ckpt_name -> T}, or None if not applicable.
    """
    if strategy == "uniform":
        return [1.0] * n_models, None

    if strategy == "scalar":
        assert user_weights is not None, "scalar strategy requires --weights"
        return list(user_weights), None

    if strategy == "temp_scalar":
        assert per_model_cal_softmax is not None and cal_labels is not None, (
            "temp_scalar strategy requires calibration softmaxes and labels"
        )
        print("\n=== Temperature calibration (per-model L-BFGS on NLL) ===")
        print(f"  {'model':30s}   T        raw_top1   temp_top1")
        print(f"  {'-'*30}   ------   --------   ---------")
        temperatures: Dict[str, float] = {}
        accuracies: List[float] = []
        for name, probs in per_model_cal_softmax.items():
            T = _fit_temperature(probs, cal_labels)
            raw_acc = _topk_accuracy(probs, cal_labels, 1)
            temp_probs = _apply_temperature(probs, T)
            temp_acc = _topk_accuracy(temp_probs, cal_labels, 1)
            temperatures[name] = T
            accuracies.append(temp_acc)
            print(
                f"  {name:30s}   {T:6.3f}   {raw_acc:.4f}    {temp_acc:.4f}"
            )

        raw_weights = [a ** alpha for a in accuracies]
        total = sum(raw_weights)
        norm_weights = [w / total for w in raw_weights]
        print(f"\n  alpha = {alpha}")
        for name, w_n in zip(per_model_cal_softmax.keys(), norm_weights):
            print(f"  {name:30s}   weight = {w_n:.4f}")
        return raw_weights, temperatures

    raise ValueError(f"Unknown strategy: {strategy}")


def _combine(
    per_model_softmax: Dict[str, torch.Tensor],
    weights: List[float],
    temperatures: Optional[Dict[str, float]],
) -> torch.Tensor:
    """Weighted average of (optionally tempered) per-model softmaxes.

    Order is taken from per_model_softmax.keys(); weights must match that order.
    """
    names = list(per_model_softmax.keys())
    assert len(weights) == len(names), "weights length must match number of models"

    acc: Optional[torch.Tensor] = None
    total_w = 0.0
    for name, w in zip(names, weights):
        probs = per_model_softmax[name]
        if temperatures is not None:
            probs = _apply_temperature(probs, temperatures[name])
        contrib = probs * float(w)
        acc = contrib if acc is None else acc + contrib
        total_w += float(w)
    assert acc is not None
    return acc / total_w


def main() -> None:
    args = parse_args()

    # ---- Validate strategy / arg combos ----------------------------------
    if args.strategy == "scalar":
        if args.weights is None:
            raise SystemExit("--strategy scalar requires --weights")
    elif args.strategy == "temp_scalar":
        if args.calibration_dir is None:
            raise SystemExit("--strategy temp_scalar requires --calibration-dir")
        if args.weights is not None:
            print(
                "INFO: --weights ignored when --strategy temp_scalar (weights are "
                "derived from per-model calibration accuracy).",
                flush=True,
            )
    elif args.strategy == "uniform":
        if args.weights is not None:
            print(
                "INFO: --weights provided with --strategy uniform — auto-promoting "
                "to --strategy scalar.",
                flush=True,
            )
            args.strategy = "scalar"

    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise SystemExit(
            f"--weights ({len(args.weights)}) must have the same length as "
            f"--checkpoints ({len(args.checkpoints)})"
        )

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    eval_mode = args.eval_dir is not None

    # ---- Resolve target clip list (labelled for eval, unlabelled for sub) -
    if eval_mode:
        eval_root = Path(args.eval_dir).resolve()
        print(f"Eval mode: {eval_root}", flush=True)
        samples = collect_video_samples(eval_root)
        sample_list: List[Tuple[Path, int]] = list(samples)
        video_names = [p.name for p, _ in sample_list]
        clip_root = eval_root
        print(f"Labelled clips: {len(sample_list)}", flush=True)
    else:
        clip_root = _resolve_test_root(args)
        print(f"Submission mode — test root: {clip_root}", flush=True)
        if args.test_manifest:
            manifest_path = Path(args.test_manifest).resolve()
            print(f"Reading manifest: {manifest_path}", flush=True)
            video_names = load_manifest_video_names(manifest_path)
            video_dirs = resolve_video_dirs(clip_root, video_names)
        else:
            print(
                "No manifest provided; using all video_* folders under test_dir.",
                flush=True,
            )
            video_names, video_dirs = discover_all_test_videos(clip_root)
        sample_list = [(p, 0) for p in video_dirs]
        print(f"Test clips: {len(sample_list)}", flush=True)

    # ---- Resolve calibration clip list (temp_scalar only) ----------------
    calibration_root: Optional[Path] = None
    cal_sample_list: List[Tuple[Path, int]] = []
    reuse_target_for_calibration = False
    if args.strategy == "temp_scalar":
        calibration_root = Path(args.calibration_dir).resolve()
        if eval_mode and calibration_root == clip_root:
            reuse_target_for_calibration = True
            print(
                "WARNING: --calibration-dir equals --eval-dir. Temperatures will "
                "be fit on the same set you evaluate — reported accuracies will "
                "be optimistic. Use a held-out split for an unbiased estimate.",
                flush=True,
            )
        else:
            cal_samples = collect_video_samples(calibration_root)
            cal_sample_list = list(cal_samples)
            print(
                f"Calibration set: {calibration_root}  "
                f"({len(cal_sample_list)} labelled clips)",
                flush=True,
            )

    # ---- Optional prior calibration: load class counts from the first
    # checkpoint's saved train_dir (unless overridden). Materialized lazily
    # once we know num_classes (after the first model loads).
    prior_logits: Optional[torch.Tensor] = None
    prior_source_dir: Optional[Path] = None
    if args.calibrate_prior:
        first_ckpt = torch.load(args.checkpoints[0], map_location="cpu")
        first_cfg = OmegaConf.create(first_ckpt["config"])
        prior_source_dir = Path(
            args.prior_train_dir
            if args.prior_train_dir is not None
            else first_cfg.dataset.train_dir
        ).resolve()
        print(
            f"\nPrior calibration: ENABLED  (counts from {prior_source_dir})",
            flush=True,
        )

    # ---- Run each checkpoint sequentially: target + (optional) calibration
    per_model_softmax: Dict[str, torch.Tensor] = {}
    per_model_cal_softmax: Dict[str, torch.Tensor] = {}
    cal_labels: Optional[torch.Tensor] = None
    true_labels: Optional[torch.Tensor] = None
    num_classes: Optional[int] = None

    for idx, ckpt_path_str in enumerate(args.checkpoints, start=1):
        ckpt_path = Path(ckpt_path_str).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

        print(
            f"\n[{idx}/{len(args.checkpoints)}] Loading {ckpt_path.name}",
            flush=True,
        )
        model, meta = _load_model_and_meta(ckpt_path, device)
        print(
            f"    model={meta['model_name']}  num_frames={meta['num_frames']}  "
            f"pretrained={meta['pretrained']}",
            flush=True,
        )

        # Target inference.
        target_loader = _build_loader(
            clip_root=clip_root,
            sample_list=sample_list,
            num_frames=meta["num_frames"],
            use_imagenet_norm=meta["pretrained"],
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        print(
            f"    target: {len(sample_list)} clips, batch_size={args.batch_size}",
            flush=True,
        )

        # Lazily materialize the prior once we know num_classes from the
        # first model's output dim.
        if args.calibrate_prior and prior_logits is None:
            # `model` is loaded — use its head to learn num_classes without
            # paying the cost of a dummy forward.
            head_out_dim = None
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    head_out_dim = module.out_features
            if head_out_dim is None:
                # Fall back to one forward pass.
                head_out_dim = int(meta["config"].model.num_classes)
            assert prior_source_dir is not None
            prior_logits = compute_prior_logits(
                train_dir=prior_source_dir,
                num_classes=head_out_dim,
                alpha=float(args.prior_alpha),
            ).to(device)
            print(
                f"    prior_logits ready: alpha={args.prior_alpha}, "
                f"shape={tuple(prior_logits.shape)}",
                flush=True,
            )

        probs, labels = _run_inference(
            model, target_loader, device, total_videos=len(sample_list),
            tag="target", prior_logits=prior_logits,
        )
        per_model_softmax[ckpt_path.name] = probs
        if num_classes is None:
            num_classes = int(probs.size(1))
        if true_labels is None:
            true_labels = labels  # constant across models

        if args.save_per_model_softmax:
            base_dir = Path(args.output).resolve().parent if not eval_mode else Path.cwd()
            out_dir = base_dir / "per_model"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(probs, out_dir / f"{ckpt_path.stem}_softmax.pt")
            print(
                f"    saved softmax -> {out_dir / (ckpt_path.stem + '_softmax.pt')}",
                flush=True,
            )

        # Calibration inference (only if temp_scalar AND a separate set).
        if args.strategy == "temp_scalar" and not reuse_target_for_calibration:
            assert calibration_root is not None
            cal_loader = _build_loader(
                clip_root=calibration_root,
                sample_list=cal_sample_list,
                num_frames=meta["num_frames"],
                use_imagenet_norm=meta["pretrained"],
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
            )
            print(
                f"    calibration: {len(cal_sample_list)} clips",
                flush=True,
            )
            cal_probs, cal_labs = _run_inference(
                model, cal_loader, device, total_videos=len(cal_sample_list),
                tag="calib", prior_logits=prior_logits,
            )
            per_model_cal_softmax[ckpt_path.name] = cal_probs
            if cal_labels is None:
                cal_labels = cal_labs

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert num_classes is not None and true_labels is not None

    # ---- Resolve weights and temperatures --------------------------------
    if args.strategy == "temp_scalar":
        if reuse_target_for_calibration:
            # eval_mode + cal_dir == eval_dir: reuse target softmaxes.
            per_model_cal_softmax = per_model_softmax
            cal_labels = true_labels
        assert cal_labels is not None
        weights_resolved, temperatures = _compute_weights(
            strategy=args.strategy,
            n_models=len(args.checkpoints),
            user_weights=None,
            per_model_cal_softmax=per_model_cal_softmax,
            cal_labels=cal_labels,
            alpha=float(args.alpha),
        )
    else:
        weights_resolved, temperatures = _compute_weights(
            strategy=args.strategy,
            n_models=len(args.checkpoints),
            user_weights=args.weights,
            per_model_cal_softmax=None,
            cal_labels=None,
            alpha=float(args.alpha),
        )

    # ---- Combine ---------------------------------------------------------
    ensemble_softmax = _combine(per_model_softmax, weights_resolved, temperatures)
    ensemble_preds = ensemble_softmax.argmax(dim=-1).tolist()

    # ---- Diagnostics + outputs -------------------------------------------
    per_model_preds = {
        name: probs.argmax(dim=-1).tolist() for name, probs in per_model_softmax.items()
    }
    _print_agreement_stats(per_model_preds, ensemble_preds, num_classes)

    if eval_mode:
        _print_accuracy_metrics(per_model_softmax, ensemble_softmax, true_labels)
        print("\nEval mode — no submission CSV written.")
    else:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting submission to: {output_path}")
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "predicted_class"])  # match create_submission.py
            for name, pred in zip(video_names, ensemble_preds):
                writer.writerow([name, pred])
        print(f"Done. Wrote {len(ensemble_preds)} rows.")


if __name__ == "__main__":
    main()
