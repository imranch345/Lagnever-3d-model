"""Step 16: dense instrumentation of three A3 runs, to find what precedes escape.

Step 15 established that A3's outcome is bimodal and that escape from the identity-only
attractor can happen as late as the final third of training. It did not say what changes first.
Three validation points per run could not: this records twenty-five.

The instrumentation is a callback on the frozen training loop. It changes nothing about
training, and that claim is load-bearing rather than decorative, so three things are restored
around every measurement:

* **RNG.** Building a batch samples points. A diagnostic that advanced the generator would make
  an instrumented run diverge from its frozen twin for a reason that has nothing to do with
  escape. Python, NumPy and Torch generators are all saved and restored.
* **Gradients.** The gradient-pathway measurement calls ``backward``, which writes ``.grad``.
  It is zeroed afterwards, and no optimiser step runs inside the callback.
* **Mode.** The model is put in ``eval`` for measurement and returned to ``train``.

A test asserts that an instrumented run reproduces its uninstrumented result bit for bit.

What is recorded at each checkpoint, all on **validation** except the objective and gradient
terms, which need training batches to be comparable with Step 14:

``graph_usage_deg``
    Step 15's definition exactly: rotation with every relationship removed, minus rotation with
    the graph intact. Not a new metric.
``relational_subset_deg``
    rotation on Step 14's preregistered relationally-required entities, so the trajectory can be
    read where the graph is known to matter rather than only on average.
representation
    entity latent before and after the graph encoder, how much the encoder writes, the relation
    bias scale, and the frame head's output spread.
objective and gradients
    Step 14's decomposition and pathway norms, reused rather than reimplemented.

    python -m experiments.step16.trajectory --seed 9
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step7.perturbations import restrict_graphs
from experiments.step8.frame_metrics import rotation_error
from experiments.step10.rotation_metrics import DEGREES
from experiments.step14.objective_audit import _gradient_norms, _total_loss
from experiments.step14.relational_identifiability import (
    IDENTITY_ONLY_VALIDATION_DEG,
    MINIMUM_GRAPH_ADVANTAGE_DEG,
    _as_frame,
    _fit_tables,
    _scene_records,
)
from experiments.step15.seed_modes import GRAPH_USING_DEG, NON_USING_DEG, classify
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["EVERY", "SEED_ROLES", "TrajectoryRecorder", "record_seed", "main"]

#: Checkpoint interval, fixed in the brief.
EVERY = 50

#: The three seeds and why each was chosen, from the frozen Step 15 report.
SEED_ROLES: dict[int, str] = {
    9: "late escaper: 23.54 deg at step 800, 21.11 deg at 1200, graph usage 2.52",
    5: "non-escaper: graph usage 0.09, the lowest of ten, rotation 23.53 against a 23.51 floor",
    3: "intermediate: the single seed in the unlabelled band, graph usage 1.30, rotation 22.17",
}

WEIGHT = 3.0
OBJECTIVE_BATCHES = 4
GRADIENT_BATCHES = 4


def _maybe(value: float | None, spec: str) -> str:
    """Format a number that a reduced sample may legitimately not have."""
    return "n/a" if value is None else format(value, spec)


def _ordered_validation_batches(
    scenes: Sequence[Any], builder: WholeOrganBatchBuilder, batch_size: int = 16
) -> list[tuple[Any, list[Any]]]:
    """Validation batches with their scene lists, in a deterministic order.

    ``WholeOrganLoader`` groups by level of detail and, unshuffled, yields a fixed order -- but
    it does not hand back which scenes went into which batch, and the relationally-required
    subset is keyed on ``(relation signature, slot)``. Mirroring the loader's grouping here keeps
    batched speed and the alignment at once.
    """
    buckets: dict[int, list[Any]] = {}
    for scene in scenes:
        buckets.setdefault(scene.active_lod, []).append(scene)
    out: list[tuple[Any, list[Any]]] = []
    for lod in sorted(buckets):
        items = buckets[lod]
        for start in range(0, len(items), batch_size):
            group = items[start : start + batch_size]
            if group:
                out.append((builder.build(group).batch, group))
    return out


class TrajectoryRecorder:
    """Measures one run every ``EVERY`` steps without perturbing it."""

    def __init__(
        self,
        *,
        corpus_dir: Path,
        builder: WholeOrganBatchBuilder,
        required: set[tuple[int, int]],
        validation: Sequence[Any],
        train_batches: Sequence[Any],
    ) -> None:
        self.corpus_dir = corpus_dir
        self.required = required
        self.rows: list[dict[str, Any]] = []
        self._batches = _ordered_validation_batches(validation, builder)
        self._train_batches = list(train_batches)
        # Scene position is the stable identity of a validation scene, and the key the
        # relationally-required set is expressed in.
        self._position = {id(scene): index for index, scene in enumerate(validation)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _rotation_by_entity(self, model: Any, *, bare: bool) -> dict[tuple[int, int], float]:
        """Per-entity rotation error, keyed by ``(scene position, slot)``.

        Keyed on the scene's position rather than its relation signature: validation scenes
        share signatures, so keying on one would make two entities collide and silently drop
        from every average. Position is also the key Step 14's required set uses.
        """
        out: dict[tuple[int, int], float] = {}
        for batch, group in self._batches:
            used = restrict_graphs(batch, keep=()) if bare else batch
            output = model(used, use_predicted_frames=True)
            present = used.entity_present.bool()
            errors = rotation_error(output.frames, used.entity_frames) * DEGREES
            for index, scene in enumerate(group):
                position = self._position[id(scene)]
                for slot in range(present.shape[1]):
                    if bool(present[index, slot]):
                        out[(position, slot)] = float(errors[index, slot])
        return out

    @torch.no_grad()
    def _representation(self, model: Any) -> dict[str, float]:
        """How much the graph encoder writes, and what the head does with it."""
        batch = self._batches[0][0]
        staged = model.stage(batch)
        present = batch.entity_present.bool()
        before = staged.entity_latent_in[present]
        after = staged.entity_latent[present]
        identity_width = model.graph_encoder.config.identity_width
        written = (after[:, identity_width:] - before[:, identity_width:]).norm(dim=-1)
        frames = model.predict_frames(staged.entity_latent, present)[present]
        bias = model.graph_encoder.relation_bias.weight.detach()
        return {
            "entity_latent_in_std": float(before.std()),
            "entity_latent_std": float(after.std()),
            "context_written_norm": float(written.mean()),
            "context_written_share": float(
                (written / after[:, identity_width:].norm(dim=-1).clamp_min(1e-9)).mean()
            ),
            "relation_bias_abs_max": float(bias.abs().max()),
            "relation_bias_std": float(bias.std()),
            "frame_rotation_output_std": float(frames[:, 6:12].std()),
            "frame_translation_output_std": float(frames[:, 0:3].std()),
        }

    # ------------------------------------------------------------------
    def __call__(self, trainer: Any, step: int) -> None:
        """One checkpoint. Restores RNG, gradients and training mode before returning."""
        if step % EVERY and step != trainer.config.steps:
            return
        started = time.time()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        was_training = trainer.model.training
        with torch.random.fork_rng(devices=[]):
            trainer.model.eval()
            intact = self._rotation_by_entity(trainer.model, bare=False)
            bare = self._rotation_by_entity(trainer.model, bare=True)
            representation = self._representation(trainer.model)

            overall = statistics.fmean(intact.values())
            no_graph = statistics.fmean(bare.values())
            subset_keys = [key for key in intact if key in self.required]
            # A run measured against an empty required set records absence rather than
            # failing: the subset is a preregistered selection and may legitimately be empty
            # for a reduced validation sample.
            subset = statistics.fmean(intact[key] for key in subset_keys) if subset_keys else None

            trainer.model.train()
            objective = self._objective(trainer, step)
            gradients = self._gradients(trainer, step)
            trainer.model.zero_grad(set_to_none=True)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        trainer.model.train(was_training)

        usage = no_graph - overall
        self.rows.append(
            {
                "step": step,
                "validation_rotation_deg": overall,
                "rotation_without_any_graph_deg": no_graph,
                "graph_usage_deg": usage,
                "band": classify(usage),
                "relational_subset_deg": subset,
                "relational_subset_entities": len(subset_keys),
                "identity_only_reference_deg": IDENTITY_ONLY_VALIDATION_DEG,
                "representation": representation,
                "objective": objective,
                "gradients": gradients,
                "seconds": round(time.time() - started, 2),
            }
        )
        print(
            f"[step16]   step {step:>5}  rotation {overall:6.2f}  usage {usage:+5.2f}  "
            f"subset {_maybe(subset, '6.2f')}  "
            f"graph-grad {_maybe(gradients.get('graph_encoder'), '.4f')}  "
            f"({self.rows[-1]['seconds']:.0f}s)",
            flush=True,
        )

    def _objective(self, trainer: Any, step: int) -> dict[str, float]:
        """Step 14's decomposition, on fixed training batches so steps are comparable."""
        terms: dict[str, list[float]] = {}
        for whole in self._train_batches[:OBJECTIVE_BATCHES]:
            computed = _total_loss(trainer, whole, max(step, 0))
            terms.setdefault("total", []).append(float(computed["total"].detach()))
            for name, value in computed["module_terms"].items():
                terms.setdefault(name, []).append(float(value.detach()))
            terms.setdefault("frame_chordal", []).append(float(computed["frame_term"].detach()))
            terms.setdefault("nested_lod", []).append(float(computed["lod_term"].detach()))
            for name, value in computed["frame_split_raw"].items():
                terms.setdefault(f"frame_{name}", []).append(value)
        return {name: statistics.fmean(values) for name, values in terms.items()}

    def _gradients(self, trainer: Any, step: int) -> dict[str, float]:
        """Step 14's pathway norms, on the same fixed batches."""
        collected: dict[str, list[float]] = {}
        for whole in self._train_batches[:GRADIENT_BATCHES]:
            for key, value in _gradient_norms(trainer, whole, max(step, 0)).items():
                collected.setdefault(key, []).append(value)
        return {key: statistics.fmean(values) for key, values in collected.items()}


def _required_entities(corpus_dir: Path, builder: WholeOrganBatchBuilder) -> set[tuple[int, int]]:
    """Step 14's preregistered relationally-required set, rebuilt from train tables.

    Keyed on ``(scene position, slot)`` -- one entry per *entity* -- because that is how Step 14
    evaluated the rule. Keying on ``(relation signature, slot)`` instead would let one
    qualifying entity enrol every entity in every other validation scene sharing its signature,
    which inflated the set from 652 entities to 991 and would have been a different, broader
    subset wearing the preregistered name.
    """
    entity_table, pair_table, _ = _fit_tables(_scene_records(corpus_dir, "train", builder))
    required: set[tuple[int, int]] = set()
    for position, (signature, entities) in enumerate(
        _scene_records(corpus_dir, "validation", builder)
    ):
        for slot, frame in entities.items():
            key = (signature, slot)
            if key not in pair_table:
                continue
            identity = float(rotation_angle(_as_frame(entity_table[slot]), frame)) * DEGREES
            graph = float(rotation_angle(_as_frame(pair_table[key]), frame)) * DEGREES
            if (
                identity >= IDENTITY_ONLY_VALIDATION_DEG
                and (identity - graph) >= MINIMUM_GRAPH_ADVANTAGE_DEG
            ):
                required.add((position, slot))
    return required


def record_seed(*, corpus_dir: Path, seed: int, out_dir: Path) -> dict[str, Any]:
    """Train one instrumented run and return its trajectory."""
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
    # The frozen configuration, asserted rather than assumed.
    assert config.steps == 1200
    assert config.graph_recurrence == 1
    assert config.frame_scene_context is False
    assert config.relation_values is False
    assert config.rotation_loss_weight == WEIGHT

    trainer = Step10Trainer(config, ontology, domain, train, validation)
    fixed = WholeOrganLoader(
        load_step8_split(corpus_dir, "train", limit=64),
        builder,
        batch_size=config.batch_size,
        seed=0,
        shuffle=False,
    )
    recorder = TrajectoryRecorder(
        corpus_dir=corpus_dir,
        builder=builder,
        required=_required_entities(corpus_dir, builder),
        validation=validation,
        train_batches=list(fixed.epoch(0)),
    )
    print(f"[step16] seed {seed} ({SEED_ROLES.get(seed, 'unlisted')})", flush=True)
    started = time.time()
    trainer.fit(checkpoint_dir=out_dir / "checkpoints", on_step=recorder)
    rows = recorder.rows

    crossings = [r for r in rows if r["graph_usage_deg"] >= GRAPH_USING_DEG]
    escape: dict[str, Any] = {"escaped": bool(crossings)}
    if crossings:
        first = crossings[0]
        previous = [r for r in rows if r["step"] < first["step"]]
        escape.update(
            {
                "first_crossing_step": first["step"],
                "previous_checkpoint_step": previous[-1]["step"] if previous else None,
                "interval": (
                    f"{previous[-1]['step']} < escape <= {first['step']}"
                    if previous
                    else f"escape <= {first['step']}"
                ),
                "graph_usage_at_crossing": first["graph_usage_deg"],
            }
        )
    return {
        "experiment_id": "step16-escape-trajectory",
        "status": "diagnostic; instrumentation only, no intervention",
        "preregistration": "docs/STEP_16_ESCAPE_TRIGGER_PLAN.md",
        "seed": seed,
        "role": SEED_ROLES.get(seed, "unlisted"),
        "split_trained": "train",
        "split_measured": "validation",
        "test_splits_read": [],
        "checkpoint_every": EVERY,
        "steps": config.steps,
        "thresholds": {"graph_using": GRAPH_USING_DEG, "non_using": NON_USING_DEG},
        "escape": escape,
        "trajectory": rows,
        "runtime_seconds": round(time.time() - started, 2),
        "notes": [
            "Graph usage is Step 15's definition, unchanged.",
            "The relationally-required subset is Step 14's, rebuilt from train tables.",
            "RNG, gradients and training mode are restored around every measurement.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 16 dense trajectory.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step16"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = record_seed(corpus_dir=args.corpus, seed=args.seed, out_dir=args.out)
    target = args.out / f"trajectory_seed{args.seed}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nescape: {report['escape']}")
    print(f"written to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
