"""Step 15: whether A3's outcome is bimodal in graph usage or a continuum.

Steps 11, 13 and 14 each measured the same bimodality from a different angle, always on the
same three seeds. Three samples cannot distinguish a two-mode distribution from one wide
unimodal one, and every interpretation since Step 11 has leaned on the word "mode". This
measures ten.

Two quantities per seed, both on validation, both on frozen weights:

**validation rotation** -- the canonical ``rotation.validation.mean_deg``, the same metric the
frozen references use, recomputed here rather than read from each run's own record so all ten
seeds pass through one code path.

**graph usage** -- the Step 11 measure: what removing every relationship costs at evaluation
(``entities_only`` minus ``intact``). This is the quantity the classification rule is written
against, because it is the one that separated the seeds in the first place.

Seeds 0 to 2 are the frozen Step 10 Change 2 runs; seeds 3 to 9 are Step 15's. They differ only
in seed, and the loader asserts the configuration matches before scoring.

Classification and the bimodality rule are preregistered in
``docs/STEP_15_SEED_MODE_PLAN.md`` and are applied here without adjustment.

No test split is read. Nothing is trained.

    python -m experiments.step15.seed_modes
"""

from __future__ import annotations

import argparse
import json
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
from experiments.step7.perturbations import restrict_graphs
from experiments.step8.frame_metrics import rotation_error
from experiments.step10.rotation_metrics import DEGREES
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["GRAPH_USING_DEG", "NON_USING_DEG", "SEEDS", "classify", "survey", "main"]

#: Preregistered classification thresholds (plan section 4).
GRAPH_USING_DEG = 1.5
NON_USING_DEG = 0.8

#: Preregistered shape rule (plan section 4.1): bimodal only if the middle band is a strict
#: minority, at most this many of ten.
MAXIMUM_INTERMEDIATE = 2

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
STEM = "s10-A3-noctx-l1-globalframe"
WEIGHT = 3.0

#: seed -> which runs directory holds its checkpoint.
FROZEN_RUNS = Path("experiments/runs/step10-change2-w3")
STEP15_RUNS = Path("experiments/runs/step15")


def _runs_dir(seed: int) -> Path:
    """Seeds 0-2 came from Step 10 Change 2; the rest are Step 15's."""
    return FROZEN_RUNS if seed < 3 else STEP15_RUNS


def classify(graph_usage_deg: float) -> str:
    """The preregistered band for one seed's graph usage."""
    if graph_usage_deg >= GRAPH_USING_DEG:
        return "graph_using"
    if graph_usage_deg <= NON_USING_DEG:
        return "non_using"
    return "intermediate"


def _load(seed: int, ontology: Any, domain: Any, small: Sequence[Any]) -> Any:
    """One seed's frozen model, with its configuration asserted to be the frozen one."""
    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
        ),
        rotation_objective="chordal",
        rotation_loss_weight=WEIGHT,
        device="cpu",
    )
    # Anything varying but the seed would make the survey measure two things at once.
    assert config.graph_recurrence == 1
    assert config.frame_scene_context is False
    assert config.relation_values is False
    assert config.rotation_loss_weight == WEIGHT
    trainer = Step10Trainer(config, ontology, domain, small, small)
    state = torch.load(
        _runs_dir(seed) / "checkpoints" / f"{STEM}-seed{seed}.pt", map_location="cpu"
    )
    trainer.model.load_state_dict(state["model"])
    trainer.model.eval()
    return trainer


@torch.no_grad()
def _rotation(model: Any, batches: Sequence[Any], *, bare: bool) -> float:
    """Mean geodesic rotation error, with the graph intact or every relationship removed."""
    errors: list[np.ndarray] = []
    for batch in batches:
        used = restrict_graphs(batch, keep=()) if bare else batch
        output = model(used, use_predicted_frames=True)
        present = used.entity_present.bool()
        errors.append(
            (rotation_error(output.frames, used.entity_frames) * DEGREES)[present].numpy()
        )
    return float(np.concatenate(errors).mean())


def _trajectory(seed: int) -> dict[str, float]:
    """Validation rotation at each recorded checkpoint, from the run's own manifest."""
    path = _runs_dir(seed) / "checkpoints" / f"{STEM}-seed{seed}.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for row in manifest.get("validation_history", []):
        value = row.get("rotation_error")
        if value is not None:
            out[str(int(row["step"]))] = float(value) * DEGREES
    return out


def survey(
    *, corpus_dir: Path, seeds: Sequence[int] = SEEDS, batch_size: int = 16
) -> dict[str, Any]:
    """Validation rotation and graph usage for every seed, plus the shape verdict."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    scenes = load_step8_split(corpus_dir, "validation")

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        trainer = _load(seed, ontology, domain, small)
        loader = WholeOrganLoader(
            scenes, trainer.builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
        )
        batches = [whole.batch.to(trainer.device) for whole in loader.epoch(0)]
        intact = _rotation(trainer.model, batches, bare=False)
        bare = _rotation(trainer.model, batches, bare=True)
        usage = bare - intact
        per_seed[str(seed)] = {
            "validation_rotation_deg": intact,
            "rotation_without_any_graph_deg": bare,
            "graph_usage_deg": usage,
            "band": classify(usage),
            "validation_trajectory_deg": _trajectory(seed),
        }
        print(
            f"[step15] seed {seed}: rotation {intact:6.2f}  no-graph {bare:6.2f}  "
            f"usage {usage:+5.2f}  -> {classify(usage)}",
            flush=True,
        )

    bands = {
        band: sorted(int(s) for s, e in per_seed.items() if e["band"] == band)
        for band in ("graph_using", "intermediate", "non_using")
    }
    intermediate = len(bands["intermediate"])
    outer_occupied = bool(bands["graph_using"]) and bool(bands["non_using"])
    if not outer_occupied:
        shape = "near-deterministic"
    elif intermediate <= MAXIMUM_INTERMEDIATE:
        shape = "bimodal"
    else:
        shape = "continuum"

    usages = [e["graph_usage_deg"] for e in per_seed.values()]
    rotations = [e["validation_rotation_deg"] for e in per_seed.values()]
    order = sorted(per_seed, key=lambda s: per_seed[s]["graph_usage_deg"])
    monotone = all(
        per_seed[order[i]]["validation_rotation_deg"]
        >= per_seed[order[i + 1]]["validation_rotation_deg"]
        for i in range(len(order) - 1)
    )
    return {
        "experiment_id": "step15-seed-modes",
        "status": "diagnostic; only the seed varied, nothing was intervened on",
        "preregistration": "docs/STEP_15_SEED_MODE_PLAN.md",
        "split": "validation",
        "test_splits_read": [],
        "seeds": len(per_seed),
        "rule": (
            f"graph_using if usage >= {GRAPH_USING_DEG} deg, non_using if <= {NON_USING_DEG}, "
            f"intermediate otherwise; bimodal only if intermediate <= {MAXIMUM_INTERMEDIATE} "
            "of ten and both outer bands are occupied"
        ),
        "per_seed": per_seed,
        "bands": bands,
        "band_counts": {band: len(members) for band, members in bands.items()},
        "shape": shape,
        "graph_usage": {
            "mean": statistics.fmean(usages),
            "std": statistics.stdev(usages),
            "min": min(usages),
            "max": max(usages),
            "sorted": sorted(usages),
        },
        "validation_rotation": {
            "mean": statistics.fmean(rotations),
            "std": statistics.stdev(rotations),
            "min": min(rotations),
            "max": max(rotations),
        },
        "usage_and_rotation_monotone": monotone,
        "notes": [
            "Graph usage is the Step 11 measure: the cost of removing every relationship.",
            "All ten seeds scored through one code path, not read from their own records.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 15 seed-mode survey.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step15/seed_modes.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv)
    report = survey(corpus_dir=args.corpus, seeds=args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'seed':>5}{'rotation':>11}{'no graph':>11}{'usage':>9}  band")
    for seed in sorted(report["per_seed"], key=int):
        entry = report["per_seed"][seed]
        print(
            f"{seed:>5}{entry['validation_rotation_deg']:11.2f}"
            f"{entry['rotation_without_any_graph_deg']:11.2f}"
            f"{entry['graph_usage_deg']:9.2f}  {entry['band']}"
        )
    print(f"\nbands: {report['band_counts']}")
    print(f"sorted usage: {[round(u, 2) for u in report['graph_usage']['sorted']]}")
    print(f"shape: {report['shape'].upper()}")
    print(f"usage and rotation monotonically related: {report['usage_and_rotation_monotone']}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
