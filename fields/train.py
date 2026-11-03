"""Train the Level 2b label field on the labelled points (``fields.data``).

Each step draws ``subjects_per_step`` training subjects and ``points_per_subject`` of their
points, and minimises:
- class-weighted cross-entropy, because chambers are 1–2% of the points and the lungs about
  40%; weights are the inverse square root of class frequency, normalised to mean 1;
- plus ``code_prior`` × the mean squared code, which keeps the codes in a ball that a
  Gaussian can later sample.

With ``focus_fraction`` above 0, that share of each subject's points is drawn from the box
around its heart structures (classes named ``focus_prefix``…, several prefixes joined by
``|``, plus ``focus_margin_mm``). The
heart's five structures hold under 9% of the points, against over 90% for the lungs and
"nothing"; focusing gives their boundaries more of every step without changing the data.

Every ``eval_every`` steps it records loss, accuracy, and per-class intersection-over-union on
a fixed probe of training points. A checkpoint holds the network, the codes, the config, the
subject list and the history, so a run can be resumed, evaluated or sampled from.

    python -m fields.train --data datasets/processed/level2b --out runs/level2b/m1 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from fields.model import DEFAULT_FIELD, AutoDecoder, FieldConfig, pick_device

__all__ = [
    "TrainConfig",
    "class_weights",
    "focus_points",
    "load_points",
    "part_indices",
    "train",
    "weights_from_counts",
]


@dataclass(frozen=True)
class TrainConfig:
    """The run's settings; saved with every checkpoint."""

    steps: int = 2000
    subjects_per_step: int = 8
    points_per_subject: int = 8192
    lr_field: float = 5e-4
    lr_codes: float = 1e-3
    code_prior: float = 1e-4
    eval_every: int = 200
    save_every: int = 500
    seed: int = 0
    max_subjects: int | None = None
    focus_fraction: float = 0.0
    focus_prefix: str = "heart."
    focus_margin_mm: float = 10.0


#: The default run; frozen, so sharing one instance is safe.
DEFAULT_TRAIN = TrainConfig()


def load_points(data_dir: Path, subjects: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """All subjects' points (S, N, 3) and labels (S, N), as CPU tensors."""
    points, labels = [], []
    for subject in subjects:
        stored = np.load(data_dir / "points" / f"{subject}.npz")
        points.append(torch.from_numpy(stored["points"].astype(np.float32)))
        labels.append(torch.from_numpy(stored["labels"].astype(np.int64)))
    return torch.stack(points), torch.stack(labels)


def class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse square-root frequency, mean 1; a class absent from the data gets weight 1."""
    return weights_from_counts(torch.bincount(labels.flatten(), minlength=n_classes))


def weights_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """``class_weights`` from per-class point counts."""
    counts = counts.double()
    weights = torch.where(counts > 0, counts.clamp(min=1).rsqrt(), torch.ones_like(counts))
    return (weights / weights.mean()).float()


def focus_points(
    points: torch.Tensor, labels: torch.Tensor, focus: Sequence[int], margin_mm: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per subject, the indices of its points in the box around its ``focus`` classes ± margin.

    Returned padded, (S, M), with each subject's count, (S,). A subject without any focus class
    keeps all its points.
    """
    wanted = torch.tensor(list(focus), dtype=labels.dtype)
    rows = []
    for subject_points, subject_labels in zip(points, labels, strict=True):
        member = torch.isin(subject_labels, wanted)
        if not bool(member.any()):
            rows.append(torch.arange(len(subject_points)))
            continue
        low = subject_points[member].min(0).values - margin_mm
        high = subject_points[member].max(0).values + margin_mm
        inside = ((subject_points >= low) & (subject_points <= high)).all(-1)
        rows.append(torch.nonzero(inside).squeeze(1))
    counts = torch.tensor([len(row) for row in rows])
    padded = torch.zeros(len(rows), int(counts.max()), dtype=torch.int64)
    for i, row in enumerate(rows):
        padded[i, : len(row)] = row
    return padded, counts


def _iou(predicted: torch.Tensor, truth: torch.Tensor, n_classes: int) -> list[float | None]:
    out: list[float | None] = []
    for k in range(n_classes):
        union = ((predicted == k) | (truth == k)).sum().item()
        out.append(None if union == 0 else ((predicted == k) & (truth == k)).sum().item() / union)
    return out


def train(
    data_dir: Path,
    out_dir: Path,
    config: TrainConfig = DEFAULT_TRAIN,
    field_config: FieldConfig = DEFAULT_FIELD,
    *,
    device: str | None = None,
    subjects: Sequence[str] | None = None,
    stop_after: int | None = None,
    tf32: bool = False,
) -> dict[str, Any]:
    """Train, resumable from ``out_dir/state.pt``; returns the final record.

    Every ``save_every`` steps the whole state is written atomically: network, codes,
    optimiser, schedule, step and both random generators. A run stopped for any reason (a flat
    battery, a reclaimed cloud machine) continues from there when started again with the same
    settings, and draws the same batches it would have drawn. ``stop_after`` stops early on
    purpose; it exists for the test that proves this. ``tf32`` lets an NVIDIA GPU multiply in
    TensorFloat-32 (about twice as fast on an L4); the checkpoint records it.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if subjects is None:
        split = manifest.get("split")
        subjects = [s for s in (split["train"] if split else manifest["subjects"])]
        subjects = [s for s in subjects if (data_dir / "points" / f"{s}.npz").exists()]
    if config.max_subjects is not None:
        subjects = list(subjects)[: config.max_subjects]
    field_config = replace(field_config, n_classes=len(manifest["classes"]))
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    run_device = pick_device(device)

    if tf32 and run_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    points, labels = load_points(data_dir, subjects)
    weights = class_weights(labels, field_config.n_classes).to(run_device)
    n_focus = int(round(config.points_per_subject * config.focus_fraction))
    focus_group = tuple(config.focus_prefix.split("|"))
    focus = [k for k, name in enumerate(manifest["classes"]) if name.startswith(focus_group)]
    focus_index, focus_counts = focus_points(points, labels, focus, config.focus_margin_mm)
    # All points live on the device (~0.4 GB for 105 subjects); a step then indexes there.
    points, labels = points.to(run_device), labels.to(run_device)
    model = AutoDecoder(len(subjects), field_config).to(run_device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.field.parameters(), "lr": config.lr_field},
            {"params": model.codes.parameters(), "lr": config.lr_codes},
        ]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    probe_subjects = torch.arange(min(4, len(subjects)), device=run_device)
    probe_index = torch.from_numpy(rng.integers(points.shape[1], size=16384)).to(run_device)
    history: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    first = 1
    state_path = out_dir / "state.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["train_config"] != asdict(config) or saved["subjects"] != list(subjects):
            raise SystemExit(f"{state_path} is from a different run; move it to start afresh")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        history = saved["history"]
        first = int(saved["step"]) + 1
        print(f"resuming at step {first}", flush=True)

    def save_state(step: int) -> None:
        payload = {
            "step": step,
            "train_config": asdict(config),
            "subjects": list(subjects),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "history": history,
        }
        partial = state_path.with_suffix(".tmp")
        torch.save(payload, partial)
        partial.replace(state_path)

    for step in range(first, config.steps + 1):
        chosen = torch.from_numpy(rng.choice(len(subjects), size=config.subjects_per_step))
        index = torch.from_numpy(
            rng.integers(
                points.shape[1], size=(config.subjects_per_step, config.points_per_subject)
            )
        )
        if n_focus:
            draw = torch.from_numpy(rng.random((config.subjects_per_step, n_focus)))
            slots = (draw * focus_counts[chosen].unsqueeze(1)).long()
            index[:, :n_focus] = torch.gather(focus_index[chosen], 1, slots)
        chosen, index = chosen.to(run_device), index.to(run_device)
        batch_points = points[chosen.unsqueeze(1), index]
        batch_labels = labels[chosen.unsqueeze(1), index]
        logits = model(batch_points, chosen)
        loss = functional.cross_entropy(
            logits.reshape(-1, field_config.n_classes),
            batch_labels.reshape(-1),
            weight=weights,
        )
        prior = model.codes(chosen).pow(2).sum(-1).mean()
        total = loss + config.code_prior * prior
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        schedule.step()

        if step % config.eval_every == 0 or step == config.steps:
            with torch.no_grad():
                probe_points = points[probe_subjects][:, probe_index]
                probe_truth = labels[probe_subjects][:, probe_index]
                predicted = model(probe_points, probe_subjects).argmax(-1)
                record = {
                    "step": step,
                    "loss": round(float(loss), 5),
                    "code_norm": round(float(prior.sqrt()), 4),
                    "accuracy": round(float((predicted == probe_truth).float().mean()), 4),
                    "iou": [
                        None if v is None else round(v, 4)
                        for v in _iou(predicted, probe_truth, field_config.n_classes)
                    ],
                    "seconds": round(time.time() - started, 1),
                }
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % config.save_every == 0 or step == config.steps:
            save_state(step)
        if stop_after is not None and step >= stop_after:
            return {"stopped_at": step}

    checkpoint = {
        "field_config": field_config.to_dict(),
        "train_config": asdict(config),
        "subjects": list(subjects),
        "classes": manifest["classes"],
        "field": model.field.state_dict(),
        "codes": model.codes.weight.detach().cpu(),
        "history": history,
        "device": str(run_device),
        "tf32": bool(tf32 and run_device.type == "cuda"),
        "torch": torch.__version__,
    }
    torch.save(checkpoint, out_dir / "checkpoint.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history[-1]


def part_indices(data_dir: Path, prefixes: Sequence[str]) -> tuple[int, ...]:
    """For classes 1..K of the data, the index of the first part each name belongs to.

    A part is a prefix, or several joined by ``|``: ``heart.aorta|heart.pulmonary_trunk``.
    """
    if not prefixes:
        return ()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    groups = [tuple(part.split("|")) for part in prefixes]
    part_of = []
    for name in manifest["classes"][1:]:
        matches = [i for i, group in enumerate(groups) if name.startswith(group)]
        if not matches:
            raise SystemExit(f"class {name} matches none of the parts {list(prefixes)}")
        part_of.append(matches[0])
    return tuple(part_of)


def main(argv: Sequence[str] | None = None) -> int:
    """Train from the command line."""
    parser = argparse.ArgumentParser(description="Train the Level 2b label field.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--subjects-per-step", type=int, default=TrainConfig.subjects_per_step)
    parser.add_argument("--points-per-subject", type=int, default=TrainConfig.points_per_subject)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=FieldConfig.hidden)
    parser.add_argument("--layers", type=int, default=FieldConfig.layers)
    parser.add_argument("--latent", type=int, default=FieldConfig.latent_dim)
    parser.add_argument("--bands", type=int, default=FieldConfig.bands)
    parser.add_argument("--lr-field", type=float, default=TrainConfig.lr_field)
    parser.add_argument("--lr-codes", type=float, default=TrainConfig.lr_codes)
    parser.add_argument("--code-prior", type=float, default=TrainConfig.code_prior)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", action="store_true", help="TensorFloat-32 matmuls on CUDA")
    parser.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="class-name prefixes, one per part with its own code (e.g. heart. respiratory.)",
    )
    parser.add_argument("--focus-fraction", type=float, default=TrainConfig.focus_fraction)
    parser.add_argument("--focus-margin", type=float, default=TrainConfig.focus_margin_mm)
    parser.add_argument(
        "--focus-prefix",
        default=TrainConfig.focus_prefix,
        help="classes to focus on: a prefix, or several joined by |",
    )
    args = parser.parse_args(argv)
    config = TrainConfig(
        steps=args.steps,
        subjects_per_step=args.subjects_per_step,
        points_per_subject=args.points_per_subject,
        lr_field=args.lr_field,
        lr_codes=args.lr_codes,
        code_prior=args.code_prior,
        eval_every=args.eval_every,
        save_every=args.save_every,
        max_subjects=args.max_subjects,
        seed=args.seed,
        focus_fraction=args.focus_fraction,
        focus_margin_mm=args.focus_margin,
        focus_prefix=args.focus_prefix,
    )
    field = FieldConfig(
        hidden=args.hidden,
        layers=args.layers,
        latent_dim=args.latent,
        bands=args.bands,
        part_of=part_indices(args.data, args.parts),
    )
    final = train(args.data, args.out, config, field, device=args.device, tf32=args.tf32)
    print(f"final: {json.dumps(final)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
