"""Step 10 Change 2 §5: choose the rotation loss weight, on training data, before any test.

The rotation term was degenerate on the Change 1 corpus — identically zero — so Step 8's
declared emphasis for it, a quarter of the translation term, was never actually realised.
On the rotated corpus the term is real, and the chordal distance it now uses is on a
different scale from an L1 sum of coordinates, so the coefficient has to be measured rather
than inherited.

The rule, fixed before the measurement was taken
------------------------------------------------

    lambda_rotation = 0.25 * mean(translation term) / mean(rotation term)

so that the rotation term's expected contribution to the frame loss is a quarter of the
translation term's: **meaningful, and not dominant**. A quarter is not a new choice; it is
the emphasis Step 8 wrote down for rotation and could not deliver.

Two things keep this honest. The measurement runs on ``train`` only — no test split is
touched, and none is touched again until the weight is frozen. And it runs with the rotation
term's weight set to **zero**, so the rotation term is measured, not optimised: a calibration
that let the term be trained by the weight it was choosing would be circular.

The terms are read from the trainer's own ``_frame_loss_parts``, so what is calibrated is
exactly what is optimised.

    python -m experiments.step10.calibrate_rotation_weight
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_manifest, load_step8_split
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = ["ARMS", "TARGET_RATIO", "calibrate", "main"]

ARMS: tuple[str, ...] = ("A1", "A1M", "A3Lite", "A3L", "A3")

#: Rotation's intended share of the translation term. Step 8's declared emphasis.
TARGET_RATIO = 0.25


def _significant(value: float, digits: int = 2) -> float:
    """Round to ``digits`` significant figures, so the frozen weight is a plain number."""
    if value == 0.0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(value))))
    return float(round(value, -exponent + (digits - 1)))


@torch.no_grad()
def _terms(trainer: Step10Trainer, batch: Any) -> dict[str, float]:
    """The three frame terms, masked by presence, exactly as the loss averages them."""
    output = trainer.model(batch, teacher_forcing=1.0)
    parts = trainer._frame_loss_parts(output.frames, batch.entity_frames)  # noqa: SLF001
    present = batch.entity_present.unsqueeze(-1).to(output.frames.dtype)
    divisor = present.sum().clamp_min(1)
    return {
        name: float((value * present).sum() / divisor) for name, value in parts.items()
    }


def calibrate(
    corpus_dir: str | Path,
    *,
    arms: Sequence[str] = ARMS,
    steps: int = 50,
    seed: int = 0,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Measure the three frame terms over early training, and derive the weight."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation", limit=32)

    per_arm: dict[str, Any] = {}
    for arm in arms:
        config = replace(
            with_variant(
                default_config(arm), placement_target="global", hierarchy="spatial", seed=seed
            ),
            rotation_objective="chordal",
            # Zero: the term is measured here, never optimised. Anything else is circular.
            rotation_loss_weight=0.0,
            steps=steps,
            batch_size=batch_size,
            device="cpu",
            eval_every=10**6,
        )
        trainer = Step10Trainer(config, ontology, domain, train, validation)
        collected: dict[str, list[float]] = {"translation": [], "scale": [], "rotation": []}
        first: dict[str, float] = {}
        for index, whole in zip(range(steps), trainer.train_loader.epoch(0), strict=False):
            batch = whole.batch.to(trainer.device)
            measured = _terms(trainer, batch)
            if index == 0:
                first = dict(measured)
            for name, value in measured.items():
                collected[name].append(value)
            trainer.train_step(whole, index)
        per_arm[arm] = {
            "steps_measured": len(collected["translation"]),
            "at_initialisation": first,
            "mean": {name: float(np.mean(values)) for name, values in collected.items()},
            "std": {name: float(np.std(values)) for name, values in collected.items()},
        }
        print(
            f"[calibrate] {arm}: translation {per_arm[arm]['mean']['translation']:.4f}  "
            f"scale {per_arm[arm]['mean']['scale']:.4f}  "
            f"rotation {per_arm[arm]['mean']['rotation']:.4f}",
            flush=True,
        )

    translation = float(np.mean([entry["mean"]["translation"] for entry in per_arm.values()]))
    rotation = float(np.mean([entry["mean"]["rotation"] for entry in per_arm.values()]))
    scale = float(np.mean([entry["mean"]["scale"] for entry in per_arm.values()]))
    raw = TARGET_RATIO * translation / rotation
    weight = _significant(raw)

    return {
        "corpus_id": manifest.corpus_id,
        "corpus_dir": str(corpus_dir),
        "rule": (
            "lambda_rotation = 0.25 * mean(translation term) / mean(rotation term), so the "
            "rotation term contributes a quarter of what translation does — Step 8's "
            "declared emphasis, realisable for the first time on this corpus"
        ),
        "fixed_before_measurement": True,
        "measured_on": "train only",
        "rotation_weight_during_measurement": 0.0,
        "why_zero": "the term is measured, not optimised; otherwise the calibration is circular",
        "steps_per_arm": steps,
        "seed": seed,
        "arms": per_arm,
        "means_over_arms": {
            "translation": translation,
            "scale": scale,
            "rotation": rotation,
        },
        "scale_weight": 0.5,
        "raw_weight": raw,
        "rotation_loss_weight": weight,
        "resulting_shares_of_frame_loss": {
            "translation": translation / (translation + 0.5 * scale + weight * rotation),
            "scale": 0.5 * scale / (translation + 0.5 * scale + weight * rotation),
            "rotation": weight * rotation / (translation + 0.5 * scale + weight * rotation),
        },
        "notes": [
            "Rounded to two significant figures so the frozen weight is a plain number.",
            "Frozen after this run; no test split was read to produce it.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Calibrate the Change 2 rotation weight.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step10-change2/rotation_weight_calibration.json"),
    )
    args = parser.parse_args(argv)

    report = calibrate(args.corpus, steps=args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shares = report["resulting_shares_of_frame_loss"]
    print(
        f"\nlambda_rotation = {report['rotation_loss_weight']} "
        f"(raw {report['raw_weight']:.4f})\n"
        f"shares of the frame loss: translation {shares['translation']:.2f}, "
        f"scale {shares['scale']:.2f}, rotation {shares['rotation']:.2f}\n"
        f"written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
