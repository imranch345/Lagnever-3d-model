"""Step 20: branch one frozen mid-training state under different data orderings.

Four observational searches (Steps 16-19) found no precursor to seed 9's escape. The remaining
question is causal -- at step 805, just before the transition, is the outcome already fixed by the
weights, or does the subsequent data ordering decide it -- and that needs an intervention.

The design holds everything constant except the one factor, which the audit showed is cleanly
separable:

* ``WholeOrganLoader`` draws scene order and batch order from **its own** ``torch.Generator``,
  not the global RNG, so a branch seed cannot reach the weights.
* ``build()`` was measured not to consume global RNG, and to return the same batch under a
  different global seed, so batch *contents* are a deterministic function of the scene group.
* ``train_step`` was measured not to consume global RNG.

So changing the branch seed changes which scenes are grouped together and in what order, and
nothing else. That is the intervention this step claims to make, and the claim is measured rather
than asserted.

Three kinds of run:

``capture``
    trains seed 9 from scratch to step 805 and saves a full continuable state. Needed because no
    such checkpoint exists: the project's checkpoints carry no optimiser moments, no step and no
    stream position. Verified against the frozen Step 18/19 dense trajectories.

``reference``
    continues from 805 with the **original** stream position restored exactly. This is the control
    that validates the whole setup: if it does not reproduce the frozen escape at 830-835, the
    branching machinery is wrong and no branch result means anything.

``branch``
    continues from 805 with a fresh loader seeded per branch. Eight of these.

    python -m experiments.step20.branching capture
    python -m experiments.step20.branching branch --branch 0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step15.seed_modes import GRAPH_USING_DEG
from experiments.step16.trajectory import TrajectoryRecorder, _required_entities
from experiments.step20.state import ResumableStream, load_state, save_state
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["BRANCH_POINT", "BRANCH_SEEDS", "SEED", "capture", "continue_run", "main"]

#: Fixed by the brief: exactly seed 9 at step 805.
SEED = 9
BRANCH_POINT = 805

#: Eight branches, seeds declared before execution and never extended afterwards. Offset well
#: clear of any seed used elsewhere in the programme so a branch cannot be confused with a run.
BRANCH_SEEDS: dict[str, int] = {f"branch_{index:02d}": 2000 + index for index in range(8)}

#: Dense measurement over the transition region, so escape timing is resolvable at 5-step
#: spacing. The coarse grid outside is Step 16's and keeps every run comparable.
WINDOW = (805, 900)
WEIGHT = 3.0


def _build(corpus_dir: Path) -> tuple[Any, Any, dict[int, int]]:
    """A trainer at the frozen configuration, and its recorder.

    The trainer's own loader is kept. A branch later reseeds *only* that loader's private
    generator, which is the narrowest possible intervention: no object is replaced, so there is
    nothing else that could differ between branches by construction.
    """
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")

    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=SEED
        ),
        rotation_objective="chordal",
        rotation_loss_weight=WEIGHT,
        device="cpu",
    )
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
        dense_window=WINDOW,
    )
    return trainer, recorder, {id(scene): index for index, scene in enumerate(train)}


def _order_fingerprint(trainer: Any, positions: dict[int, int], count: int = 6) -> list[list[int]]:
    """The first ``count`` batches' scene identities, to prove orderings really differ.

    Peeking would consume the stream, so this rebuilds the same epoch from the loader's current
    generator state, reads the groups without training on them, and restores the generator.

    Scenes carry no identifier of their own, so identity is their position in the training list --
    the same device Step 16 used to join validation entities to a subset.
    """
    loader = trainer.train_loader
    saved = loader.generator.get_state().clone()
    buckets = {lod: list(items) for lod, items in loader.buckets.items()}
    import torch as _torch

    batches: list[list[Any]] = []
    for lod in sorted(buckets):
        items = buckets[lod]
        order = _torch.randperm(len(items), generator=loader.generator).tolist()
        items = [items[i] for i in order]
        size = loader.batch_size
        stop = len(items) - size + 1 if loader.drop_last else len(items)
        for start in range(0, max(stop, 0), size):
            group = items[start : start + size]
            if group:
                batches.append(group)
    order = _torch.randperm(len(batches), generator=loader.generator).tolist()
    batches = [batches[i] for i in order]
    loader.generator.set_state(saved)
    return [[positions[id(scene)] for scene in group] for group in batches[:count]]


def capture(*, corpus_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Train seed 9 to the branch point and save a continuable state."""
    trainer, recorder, _positions = _build(corpus_dir)
    stream = ResumableStream(trainer.train_loader)
    iterator = stream.iterate()
    print(f"[step20] capturing seed {SEED} state at step {BRANCH_POINT}", flush=True)
    for step in range(BRANCH_POINT):
        recorder(trainer, step)
        trainer.train_step(next(iterator), step)
    recorder(trainer, BRANCH_POINT)

    path = out_dir / f"state_seed{SEED}_step{BRANCH_POINT}.pt"
    marks = save_state(
        trainer=trainer,
        stream=stream,
        step=BRANCH_POINT,
        path=path,
        label=f"seed{SEED}_step{BRANCH_POINT}",
    )
    return {
        "experiment_id": "step20-capture",
        "seed": SEED,
        "branch_point": BRANCH_POINT,
        "state_path": str(path),
        "fingerprint": marks,
        "trajectory": recorder.rows,
        "test_splits_read": [],
    }


def continue_run(
    *,
    corpus_dir: Path,
    state_path: Path,
    out_dir: Path,
    branch: str,
    loader_seed: int | None,
) -> dict[str, Any]:
    """Continue from the saved state to the end of the budget.

    ``loader_seed=None`` restores the original stream position exactly -- the reference control.
    Any integer starts a fresh ordering from the branch point, which is the intervention.
    """
    trainer, recorder, positions = _build(corpus_dir)
    restored = load_state(trainer=trainer, path=state_path)
    start = restored["step"]
    stream = ResumableStream(trainer.train_loader)

    if loader_seed is None:
        iterator = stream.restore(restored["stream"])
        ordering = "original stream position restored exactly"
    else:
        trainer.train_loader.generator.manual_seed(loader_seed)
        iterator = stream.iterate()
        ordering = f"fresh ordering from loader seed {loader_seed}"
    first_batches = _order_fingerprint(trainer, positions)

    print(
        f"[step20] {branch}: continuing from {start} to {trainer.config.steps} ({ordering})",
        flush=True,
    )
    for step in range(start, trainer.config.steps):
        recorder(trainer, step)
        trainer.train_step(next(iterator), step)
    recorder(trainer, trainer.config.steps)

    rows = recorder.rows
    usage = {int(r["step"]): float(r["graph_usage_deg"]) for r in rows}
    steps = sorted(usage)
    crossings = [s for s in steps if usage[s] >= GRAPH_USING_DEG]
    permanent = next(
        (s for s in crossings if all(usage[x] >= GRAPH_USING_DEG for x in steps if x >= s)), None
    )
    before = [s for s in steps if permanent is not None and s < permanent]
    final = rows[-1]
    return {
        "experiment_id": "step20-branch",
        "branch": branch,
        "loader_seed": loader_seed,
        "ordering": ordering,
        "branch_point": start,
        "steps": trainer.config.steps,
        "starting_fingerprint": restored["fingerprint"],
        "first_batches_after_branch": first_batches,
        "escape": {
            "escaped": permanent is not None,
            "permanent_crossing_step": permanent,
            "last_checkpoint_before": before[-1] if before else None,
            "interval": (
                f"{before[-1]} < escape <= {permanent}" if before and permanent else "never"
            ),
            "transient_crossings": [s for s in crossings if permanent is None or s < permanent],
        },
        "final": {
            "graph_usage_deg": final["graph_usage_deg"],
            "validation_rotation_deg": final["validation_rotation_deg"],
            "relational_subset_deg": final["relational_subset_deg"],
        },
        "trajectory": rows,
        "test_splits_read": [],
        "notes": [
            "Only the data stream's generator differs between branches; the audit in this "
            "module's docstring is what licenses that claim.",
            "Measurement definitions are Step 15/16's, unchanged.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 20 checkpoint branching.")
    parser.add_argument("mode", choices=("capture", "branch", "reference"))
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step20"))
    parser.add_argument("--branch", type=int, default=None)
    parser.add_argument(
        "--replay", action="store_true", help="determinism control: same seed again"
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / f"state_seed{SEED}_step{BRANCH_POINT}.pt"

    if args.mode == "capture":
        report = capture(corpus_dir=args.corpus, out_dir=args.out)
        target = args.out / "capture.json"
    elif args.mode == "reference":
        report = continue_run(
            corpus_dir=args.corpus,
            state_path=state_path,
            out_dir=args.out,
            branch="reference",
            loader_seed=None,
        )
        target = args.out / "branch_reference.json"
    else:
        if args.branch is None:
            raise SystemExit("--branch is required for mode 'branch'")
        name = f"branch_{args.branch:02d}"
        report = continue_run(
            corpus_dir=args.corpus,
            state_path=state_path,
            out_dir=args.out,
            branch=name,
            loader_seed=BRANCH_SEEDS[name],
        )
        target = args.out / (f"{name}_replay.json" if args.replay else f"{name}.json")

    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")
    if args.mode != "capture":
        print(f"escape: {report['escape']}")
        print(f"final: {report['final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
