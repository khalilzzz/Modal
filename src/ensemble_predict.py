#!/usr/bin/env python3
"""
Ensemble prediction / evaluation: average the softmax outputs of N trained
checkpoints. Two modes:

  - **Submission mode** (default): runs over ``dataset.test_dir`` (no labels)
    and writes a CSV in the same format as ``create_submission.py``
    (``video_name,predicted_class``).

  - **Eval mode** (``--eval-dir PATH``): runs over a labelled directory
    (same layout as ``dataset.val_dir``) and reports **top-1 / top-5**
    accuracy for each individual model AND for the ensemble. No CSV.

Reuses helpers from ``create_submission.py`` by import — no edits to that file.

Examples:
    # Submission: average 3 TSM checkpoints, write a CSV
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm_a.pt tsm_b.pt tsm_c.pt \\
        --output submission_ensemble.csv

    # Eval on the validation set — prints top-1 / top-5 per model AND ensemble
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm_a.pt tsm_b.pt tsm_c.pt \\
        --eval-dir processed_data/val

    # Weighted ensemble in eval mode
    uv run python src/ensemble_predict.py \\
        --checkpoints tsm_weak.pt tsm_strong.pt \\
        --weights 1.0 2.0 \\
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
from utils import build_transforms

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
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-checkpoint weights (defaults to uniform). "
        "Must match --checkpoints length.",
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference; return (softmax, labels) both on CPU.

    Labels are whatever the dataset's sample_list contains (true labels in eval
    mode, dummy 0s in submission mode).
    """
    model.eval()
    probs_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    for batch_idx, (video_batch, labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        logits = model(video_batch)
        probs = F.softmax(logits, dim=-1).cpu()
        probs_chunks.append(probs)
        label_chunks.append(labels.cpu())
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(f"    inference batch {batch_idx}/{n_batches}", flush=True)
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


def main() -> None:
    args = parse_args()
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise SystemExit(
            f"--weights ({len(args.weights)}) must have the same length as "
            f"--checkpoints ({len(args.checkpoints)})"
        )
    weights = args.weights or [1.0] * len(args.checkpoints)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    eval_mode = args.eval_dir is not None

    # ---- Resolve clip list (and labels, if eval mode) --------------------
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

    # ---- Run each checkpoint sequentially --------------------------------
    softmax_accumulator: Optional[torch.Tensor] = None
    per_model_softmax: Dict[str, torch.Tensor] = {}
    true_labels: Optional[torch.Tensor] = None
    num_classes: Optional[int] = None

    for idx, (ckpt_path_str, w) in enumerate(
        zip(args.checkpoints, weights), start=1
    ):
        ckpt_path = Path(ckpt_path_str).resolve()
        if not ckpt_path.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_path}")

        print(
            f"\n[{idx}/{len(args.checkpoints)}] Loading {ckpt_path.name} "
            f"(weight={w})",
            flush=True,
        )
        model, meta = _load_model_and_meta(ckpt_path, device)
        print(
            f"    model={meta['model_name']}  num_frames={meta['num_frames']}  "
            f"pretrained={meta['pretrained']}",
            flush=True,
        )

        transform = build_transforms(
            is_training=False, use_imagenet_norm=meta["pretrained"]
        )
        dataset = VideoFrameDataset(
            root_dir=clip_root,
            num_frames=meta["num_frames"],
            transform=transform,
            sample_list=sample_list,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )

        print(
            f"    running inference: {len(dataset)} clips, "
            f"batch_size={args.batch_size}",
            flush=True,
        )
        probs, labels = _run_inference(
            model, loader, device, total_videos=len(dataset)
        )  # (N_clips, num_classes), (N_clips,)

        if num_classes is None:
            num_classes = int(probs.size(1))
        if true_labels is None:
            true_labels = labels  # constant across models

        per_model_softmax[ckpt_path.name] = probs

        if args.save_per_model_softmax:
            base_dir = Path(args.output).resolve().parent if not eval_mode else Path.cwd()
            out_dir = base_dir / "per_model"
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(probs, out_dir / f"{ckpt_path.stem}_softmax.pt")
            print(
                f"    saved softmax -> {out_dir / (ckpt_path.stem + '_softmax.pt')}",
                flush=True,
            )

        if softmax_accumulator is None:
            softmax_accumulator = probs * float(w)
        else:
            softmax_accumulator = softmax_accumulator + probs * float(w)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert softmax_accumulator is not None and num_classes is not None
    assert true_labels is not None

    total_weight = float(sum(weights))
    ensemble_softmax = softmax_accumulator / total_weight  # (N_clips, num_classes)
    ensemble_preds = ensemble_softmax.argmax(dim=-1).tolist()

    # ---- Diagnostics + outputs -------------------------------------------
    per_model_preds = {
        name: probs.argmax(dim=-1).tolist() for name, probs in per_model_softmax.items()
    }
    _print_agreement_stats(per_model_preds, ensemble_preds, num_classes)

    if eval_mode:
        _print_accuracy_metrics(per_model_softmax, ensemble_softmax, true_labels)
        # No CSV in eval mode — the user asked for pure accuracy.
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
