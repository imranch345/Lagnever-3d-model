"""Step 19: gradient direction across the transition, which magnitude cannot see.

Steps 16, 17 and 18 searched for a precursor to seed 9's escape at 50-step, 5-step and
corrected-baseline resolution. None was found, and the one apparent positive dissolved once the
variance was estimated at the resolution of the measurement. Every gradient number in that search
was an aggregate L2 norm per module.

A norm is blind to direction. A transition that occupies ten steps with no change in gradient
magnitude is exactly what a change in *where the gradient points* would look like, so this
measures that and nothing else.

Two quantities, and they differ in what they are allowed to claim.

``cosine_successive``
    cos(g_t, g_{t-1}), per pathway and overall. Uses only information available at step t. If the
    gradient direction destabilises or realigns before the transition, it shows up here, and the
    measurement is honest as a candidate precursor.

``cosine_escape_axis``
    cos(g_t, w_end - w_window_start), the alignment of the gradient with the displacement the
    weights actually undergo across the window. **This axis is defined using the end of the run,
    so it is not available at training time and cannot be a predictive signal.** It answers a
    different and weaker question: was the gradient already pointing along the eventual
    displacement before the geometry moved? It is reported as descriptive, never as a precursor.

Gradients are taken on the same fixed training batches Steps 14-18 used, so magnitudes remain
comparable with those reports, and the flattened vectors are held in memory for the window only
(about 13 MB per checkpoint) rather than written to disk.

Nothing is trained differently. The recorder wraps Step 16's, which restores RNG, gradients and
training mode around every measurement, and is verified by test to leave training bit-identical.

    python -m experiments.step19.gradient_direction --seed 9 --window 780 860
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step14.objective_audit import PATHWAYS, _total_loss
from experiments.step16.trajectory import (
    DENSE_EVERY,
    TrajectoryRecorder,
    _required_entities,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["DirectionRecorder", "record", "main"]

WEIGHT = 3.0
GRADIENT_BATCHES = 4


def _partition(named: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten a name-keyed tensor map into one vector per pathway, plus the whole model.

    Built over a **sorted, fixed** name list so that a gradient vector and a weight vector are
    element-for-element comparable. A cosine between two differently-ordered or differently-sized
    flattenings is meaningless, and an earlier version produced exactly that by skipping
    parameters whose gradient was ``None`` on one side but not the other.
    """
    order = sorted(named)
    out: dict[str, torch.Tensor] = {}
    for pathway, prefixes in PATHWAYS.items():
        parts = [named[name] for name in order if name.startswith(prefixes)]
        if parts:
            out[pathway] = torch.cat(parts)
    out["all_parameters"] = torch.cat([named[name] for name in order])
    return out


def _flat_gradient(trainer: Any, batches: Sequence[Any], step: int) -> dict[str, torch.Tensor]:
    """The summed gradient over the fixed batches, flattened per pathway and overall.

    Summed rather than averaged: the direction of a sum over a fixed batch set is the direction
    of their mean, and summing avoids a division that would differ from how Step 14's
    ``_gradient_norms`` accumulates.

    A parameter that received no gradient contributes zeros rather than being dropped, so the
    vector's layout matches :func:`_flat_weights` exactly.
    """
    trainer.model.zero_grad(set_to_none=True)
    for whole in batches[:GRADIENT_BATCHES]:
        computed = _total_loss(trainer, whole, step)
        total: Any = computed["total"]
        total.backward()
    named = {
        name: (
            parameter.grad.detach().flatten().clone()
            if parameter.grad is not None
            else torch.zeros(parameter.numel())
        )
        for name, parameter in trainer.model.named_parameters()
    }
    trainer.model.zero_grad(set_to_none=True)
    return _partition(named)


def _flat_weights(trainer: Any) -> dict[str, torch.Tensor]:
    """Weights flattened identically, so a displacement is comparable with a gradient."""
    return _partition(
        {
            name: parameter.detach().flatten().clone()
            for name, parameter in trainer.model.named_parameters()
        }
    )


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    """Cosine similarity, guarding the degenerate zero-vector case."""
    denominator = first.norm() * second.norm()
    if float(denominator) <= 0.0:
        return float("nan")
    return float(torch.dot(first, second) / denominator)


class DirectionRecorder:
    """Step 16's recorder, plus gradient direction inside the window."""

    def __init__(self, inner: TrajectoryRecorder, *, window: tuple[int, int]) -> None:
        self.inner = inner
        self.window = window
        self.rows: list[dict[str, Any]] = []
        self._previous: dict[str, torch.Tensor] | None = None
        self._gradients: dict[int, dict[str, torch.Tensor]] = {}
        self._weights_at_start: dict[str, torch.Tensor] | None = None
        self._weights_at_end: dict[str, torch.Tensor] | None = None

    def _in_window(self, step: int) -> bool:
        low, high = self.window
        return low <= step <= high and step % DENSE_EVERY == 0

    def __call__(self, trainer: Any, step: int) -> None:
        """Delegate to the inner recorder, then measure direction inside the window."""
        self.inner(trainer, step)
        if step == trainer.config.steps:
            # The run's end, needed for the escape axis. Taken outside the window on purpose.
            # Reading weights consumes no RNG, but the guard costs nothing and keeps the
            # invariant uniform.
            with torch.random.fork_rng(devices=[]):
                self._weights_at_end = _flat_weights(trainer)
            return
        if not self._in_window(step):
            return

        # The same protection Step 16's recorder applies internally, and for the same reason:
        # _total_loss runs a forward pass and the nested-LOD term, which consume RNG. Measuring
        # outside it made an instrumented run diverge from its twin -- caught by test, not by
        # inspection.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        was_training = trainer.model.training
        with torch.random.fork_rng(devices=[]):
            trainer.model.train()
            gradients = _flat_gradient(trainer, self.inner._train_batches, step)
            if self._weights_at_start is None:
                self._weights_at_start = _flat_weights(trainer)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        trainer.model.train(was_training)
        self._gradients[step] = gradients

        row: dict[str, Any] = {"step": step}
        for pathway, vector in gradients.items():
            row[f"norm_{pathway}"] = float(vector.norm())
            if self._previous is not None and pathway in self._previous:
                row[f"cosine_successive_{pathway}"] = _cosine(vector, self._previous[pathway])
        self._previous = gradients
        self.rows.append(row)
        printable = row.get("cosine_successive_all_parameters")
        print(
            f"[step19]   step {step:>5}  |g| {row['norm_all_parameters']:8.3f}  "
            f"cos(g_t, g_t-1) "
            f"{'n/a' if printable is None else format(printable, '+.4f')}",
            flush=True,
        )

    def finalise(self) -> dict[str, Any]:
        """Escape-axis alignment, computable only once the run has ended."""
        if self._weights_at_start is None or self._weights_at_end is None:
            return {"escape_axis": None, "reason": "window never opened, or run did not finish"}
        axis = {
            pathway: self._weights_at_end[pathway] - vector
            for pathway, vector in self._weights_at_start.items()
            if pathway in self._weights_at_end
        }
        for row in self.rows:
            gradients = self._gradients[int(row["step"])]
            for pathway, vector in axis.items():
                if pathway in gradients:
                    # Negated: gradient descent moves along -g, so a positive value means the
                    # step the optimiser would take points along the eventual displacement.
                    row[f"cosine_escape_axis_{pathway}"] = _cosine(-gradients[pathway], vector)
        return {
            "escape_axis": {
                "defined_as": "weights at the final step minus weights at the window's first "
                "checkpoint",
                "uses_future_information": True,
                "interpretation": "descriptive only; not available at training time and not a "
                "predictive signal",
                "norms": {p: float(v.norm()) for p, v in axis.items()},
            }
        }


def record(
    *, corpus_dir: Path, seed: int, window: tuple[int, int], out_dir: Path
) -> dict[str, Any]:
    """Train one instrumented run and return its direction trajectory."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")

    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
        ),
        rotation_objective="chordal",
        rotation_loss_weight=WEIGHT,
        device="cpu",
    )
    assert config.steps == 1200
    assert config.graph_recurrence == 1
    assert config.frame_scene_context is False
    assert config.relation_values is False

    trainer = Step10Trainer(config, ontology, domain, train, validation)
    fixed = WholeOrganLoader(
        load_step8_split(corpus_dir, "train", limit=64),
        builder,
        batch_size=config.batch_size,
        seed=0,
        shuffle=False,
    )
    inner = TrajectoryRecorder(
        corpus_dir=corpus_dir,
        builder=builder,
        required=_required_entities(corpus_dir, builder),
        validation=validation,
        train_batches=list(fixed.epoch(0)),
        dense_window=window,
    )
    recorder = DirectionRecorder(inner, window=window)
    print(f"[step19] seed {seed}, direction window {window[0]}-{window[1]}", flush=True)
    trainer.fit(checkpoint_dir=out_dir / "checkpoints", on_step=recorder)
    axis = recorder.finalise()

    successive = [
        row["cosine_successive_all_parameters"]
        for row in recorder.rows
        if "cosine_successive_all_parameters" in row
    ]
    return {
        "experiment_id": "step19-gradient-direction",
        "status": "diagnostic; instrumentation only, no intervention",
        "preregistration": "docs/STEP_19_GRADIENT_DIRECTION_PLAN.md",
        "seed": seed,
        "window": list(window),
        "dense_every": DENSE_EVERY,
        "gradient_batches": GRADIENT_BATCHES,
        "split_trained": "train",
        "split_measured": "validation",
        "test_splits_read": [],
        **axis,
        "successive_cosine_summary": {
            "mean": statistics.fmean(successive) if successive else None,
            "std": statistics.stdev(successive) if len(successive) > 1 else None,
            "min": min(successive) if successive else None,
            "max": max(successive) if successive else None,
        },
        "direction": recorder.rows,
        "trajectory": inner.rows,
        "notes": [
            "A gradient norm is blind to direction; this measures direction only.",
            "cosine_successive uses information available at step t and may be a precursor.",
            "cosine_escape_axis uses the run's end and is descriptive, never predictive.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 19 gradient direction.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--window", type=int, nargs=2, default=[780, 860])
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step19"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = record(
        corpus_dir=args.corpus,
        seed=args.seed,
        window=(args.window[0], args.window[1]),
        out_dir=args.out,
    )
    target = args.out / f"direction_seed{args.seed}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsuccessive cosine: {report['successive_cosine_summary']}")
    print(f"written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
