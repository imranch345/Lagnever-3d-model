"""Calibrate per-class logit offsets so decoded volumes match true ones (Level 2b).

**Why.** Training weights rare classes more than "nothing" (the trachea about 1.7, "nothing"
about 0.2). Where a structure borders other structures, the weights barely move its
boundary. A thin tube that borders mostly "nothing" swells instead: on held-out subjects
the chest model's trachea decodes 1.49× its true volume and the IVC 1.12×, while every
other structure is within 3%.

**How.**
1. For training subjects, using their own learned codes, decode the logits at grid points
   inside each subject's lung slab. Inside the slab the data's crops do not bite.
2. Look up the true labels at the same points (``fields.data.labels_on_grid``).
3. Find one offset per class, "nothing" fixed at 0, so that the argmax counts match the
   true counts:

       b_k ← b_k − rate · log(predicted_k / true_k)

   repeated until the counts agree.

The offsets are fitted on training subjects only. Held-out subjects, which evaluation
uses, play no part. The result is a copy of the checkpoint with ``logit_offsets``;
``fields.evaluate.load_checkpoint`` wraps the field in an ``OffsetField``, so evaluation,
generation, resize and the viewer use it unchanged.

    python -m fields.calibrate --checkpoint runs/gpu/chest-a-b8/checkpoint.pt
        --data datasets/processed/level2b-chest-v2 --raw datasets/raw/totalsegmentator_v1
        --out runs/gpu/chest-a-b8/checkpoint-calibrated.pt
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fields.data import labels_on_grid
from fields.evaluate import Grid, load_checkpoint
from fields.model import Field, pick_device

__all__ = ["collect", "fit_offsets"]


def collect(
    field: Field,
    codes: torch.Tensor,
    subjects: Sequence[str],
    stats: dict[str, Any],
    raw_root: Path,
    grid: Grid,
    classes: Sequence[str],
    *,
    points_per_subject: int = 200_000,
    seed: int = 0,
    chunk: int = 65_536,
) -> tuple[torch.Tensor, np.ndarray]:
    """Logits (N, classes) and true labels (N,) at random grid points in each lung slab."""
    device = next(field.parameters()).device
    rng = np.random.default_rng(seed)
    everywhere = grid.points()
    rows = everywhere[:, 1].reshape(grid.shape)[0, :, 0]
    lobes = [k for k, c in enumerate(classes, start=1) if "lobe" in c or c.endswith("_lung")]
    all_logits, all_truth = [], []
    for position, subject in enumerate(subjects):
        truth = labels_on_grid(raw_root / subject, stats[subject]["frame"], everywhere, classes)
        in_lungs = np.isin(truth.reshape(grid.shape), lobes).any(axis=(0, 2))
        low, high = rows[in_lungs].min(), rows[in_lungs].max()
        inside = np.flatnonzero((everywhere[:, 1] >= low) & (everywhere[:, 1] <= high))
        chosen = rng.choice(inside, size=min(points_per_subject, len(inside)), replace=False)
        code = codes[position : position + 1].to(device)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(chosen), chunk):
                part = torch.from_numpy(everywhere[chosen[start : start + chunk]]).unsqueeze(0)
                pieces.append(field(part.to(device), code)[0].to("cpu", torch.float16))
        all_logits.append(torch.cat(pieces))
        all_truth.append(truth[chosen])
        print(f"collected {subject} ({position + 1}/{len(subjects)})", flush=True)
    return torch.cat(all_logits), np.concatenate(all_truth)


def fit_offsets(
    logits: torch.Tensor, truth: np.ndarray, *, iterations: int = 200, rate: float = 0.5
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Offsets (classes,), "nothing" fixed at 0, that make argmax counts match true counts."""
    n_classes = logits.shape[1]
    true_counts = np.bincount(truth, minlength=n_classes).astype(np.float64)
    offsets = np.zeros(n_classes)
    work = logits.float()
    trace = []
    for iteration in range(iterations):
        predicted = (work + torch.from_numpy(offsets).float()).argmax(-1).numpy()
        counts = np.bincount(predicted, minlength=n_classes).astype(np.float64)
        ratio = (counts + 1.0) / (true_counts + 1.0)
        if iteration % 20 == 0 or iteration == iterations - 1:
            trace.append({"iteration": iteration, "ratio": [round(float(r), 4) for r in ratio]})
        step = -rate * np.log(ratio)
        step[0] = 0.0
        offsets += step
    return offsets, trace


def main(argv: Sequence[str] | None = None) -> int:
    """Calibrate a checkpoint from the command line."""
    parser = argparse.ArgumentParser(description="Calibrate Level 2b logit offsets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=20, help="training subjects used")
    parser.add_argument("--points", type=int, default=200_000, help="grid points per subject")
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    field, checkpoint = load_checkpoint(args.checkpoint, pick_device(args.device))
    if "logit_offsets" in checkpoint:
        raise SystemExit(f"{args.checkpoint} is already calibrated")
    classes = list(checkpoint["classes"][1:])
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    train = list(checkpoint["subjects"])
    grid = Grid.around(args.data, train, args.spacing)
    used = train[: args.subjects]
    logits, truth = collect(
        field,
        checkpoint["codes"][: args.subjects],
        used,
        manifest["stats"],
        args.raw,
        grid,
        classes,
        points_per_subject=args.points,
    )
    offsets, trace = fit_offsets(logits, truth)
    calibrated = dict(checkpoint)
    calibrated["logit_offsets"] = torch.from_numpy(offsets).float()
    calibrated["calibration"] = {
        "subjects": used,
        "points_per_subject": args.points,
        "data": str(args.data),
        "trace": trace,
        "offsets": {
            c: round(float(o), 4) for c, o in zip(["nothing", *classes], offsets, strict=True)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, args.out)
    before, after = trace[0]["ratio"], trace[-1]["ratio"]
    for name, b, a, o in zip(["nothing", *classes], before, after, offsets, strict=True):
        print(f"{name:36s} predicted/true {b:5.2f} → {a:5.2f}  offset {o:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
