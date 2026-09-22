"""Step 11 E1: which relational information the trained A3 model actually uses.

A lookup keyed on entity identity and the relationship graph reaches 11.57 degrees of rotation
error. A3, reading the same inputs, reaches 21.38. This asks the cheapest question that can
narrow the gap: **damage the graph at evaluation and see whether A3 notices.**

Nothing is retrained. The three frozen A3 checkpoints at lambda=3.0 are loaded, the batch's
relationship graph is perturbed with the Step 7 machinery, and the rotation error is measured
again. A perturbation the model does not notice is information the model was not using,
whatever its latent may contain.

Scored on **validation**. No test split is read.

Conditions, declared in the plan before any of this ran
-------------------------------------------------------

``intact``
    the control.
``drop_spatial`` / ``drop_functional`` / ``drop_structure``
    one typed graph removed at a time; what that graph was worth.
``shuffle_spatial_types``
    endpoints kept, relation **types** permuted among them. Isolates relation type, which in
    this architecture is a scalar attention bias and nothing else.
``randomise_spatial_endpoints``
    types kept, endpoints rewired at random. Isolates *which entities* are related, and
    doubles as the permutation control: the tensor is structurally intact and semantically
    meaningless.
``entities_only``
    every relationship removed. The floor of the ablation series.

    python -m experiments.step11.graph_ablation
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
from experiments.step7.perturbations import describe, perturb_relations, restrict_graphs
from experiments.step8.frame_metrics import position_error, rotation_error
from experiments.step10.rotation_metrics import DEGREES
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["CONDITIONS", "MINIMUM_MEANINGFUL_DEG", "ablate", "main"]

#: Declared in the Step 11 plan before any measurement.
CONDITIONS: tuple[str, ...] = (
    "intact",
    "drop_spatial",
    "drop_functional",
    "drop_structure",
    "shuffle_spatial_types",
    "randomise_spatial_endpoints",
    "entities_only",
)

#: A difference smaller than A3's own seed spread (1.03 deg) cannot be distinguished from it.
MINIMUM_MEANINGFUL_DEG = 1.0

SPLIT = "validation"
WEIGHT = 3.0


@torch.no_grad()
def _score(
    model: Any, batches: Sequence[Any], condition: str, inverse: torch.Tensor
) -> dict[str, float]:
    """Rotation and position error under one perturbation of the graph."""
    rotations: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    generator = torch.Generator().manual_seed(11)
    for batch in batches:
        if condition == "entities_only":
            damaged = restrict_graphs(batch, keep=())
        else:
            damaged = perturb_relations(
                batch, condition, inverse_relations=inverse, generator=generator
            )
        output = model(damaged, use_predicted_frames=True)
        present = damaged.entity_present.bool()
        truth = damaged.entity_frames
        rotations.append((rotation_error(output.frames, truth) * DEGREES)[present].numpy())
        positions.append(position_error(output.frames, truth)[present].numpy())
    rotation = np.concatenate(rotations)
    position = np.concatenate(positions)
    return {
        "rotation_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "position": float(position.mean()),
        "entities_scored": int(rotation.size),
    }


def ablate(
    *,
    corpus_dir: Path,
    runs_dir: Path,
    conditions: Sequence[str] = CONDITIONS,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score every condition on every frozen A3 seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    scenes = load_step8_split(corpus_dir, SPLIT)
    small = load_step8_split(corpus_dir, "train", limit=48)
    report = json.loads((runs_dir / "change2_report.json").read_text(encoding="utf-8"))
    a3 = [run for run in report["runs"] if run["arm"] == "A3" and run["cell"] == "T0_global"]
    if not a3:
        raise SystemExit("no frozen A3 runs found")

    per_seed: dict[str, dict[str, Any]] = {}
    for run in sorted(a3, key=lambda r: int(r["seed"])):
        seed = int(run["seed"])
        config = replace(
            with_variant(
                default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
            ),
            rotation_objective="chordal",
            rotation_loss_weight=WEIGHT,
            device="cpu",
        )
        trainer = Step10Trainer(config, ontology, domain, small, small)
        checkpoint = torch.load(
            runs_dir / "checkpoints" / f"{run['run_id']}.pt", map_location="cpu"
        )
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.model.eval()
        inverse = trainer.builder.inverse_relation_table()
        loader = WholeOrganLoader(
            scenes,
            trainer.builder,
            batch_size=batch_size,
            seed=0,
            shuffle=False,
            drop_last=False,
        )
        batches = [whole.batch.to(trainer.device) for whole in loader.epoch(0)]
        for condition in conditions:
            entry = _score(trainer.model, batches, condition, inverse)
            per_seed.setdefault(condition, {})[str(seed)] = entry
            print(
                f"[step11] A3 seed {seed} {condition:<28} "
                f"rotation {entry['rotation_deg']:6.2f}  position {entry['position']:.4f}",
                flush=True,
            )

    intact = [per_seed["intact"][s]["rotation_deg"] for s in sorted(per_seed["intact"])]
    summary: dict[str, Any] = {}
    for condition, seeds in per_seed.items():
        keys = sorted(seeds)
        rotation = [seeds[k]["rotation_deg"] for k in keys]
        position = [seeds[k]["position"] for k in keys]
        deltas = [value - base for value, base in zip(rotation, intact, strict=True)]
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        summary[condition] = {
            "what": describe("E_entities_only" if condition == "entities_only" else condition),
            "per_seed_rotation_deg": dict(zip(keys, rotation, strict=True)),
            "rotation_deg": statistics.fmean(rotation),
            "rotation_std": statistics.stdev(rotation) if len(rotation) > 1 else 0.0,
            "position": statistics.fmean(position),
            "delta_rotation_deg": statistics.fmean(deltas),
            "per_seed_delta": dict(zip(keys, deltas, strict=True)),
            "all_seeds_same_sign": same_sign,
            "exceeds_minimum": bool(
                abs(statistics.fmean(deltas)) >= MINIMUM_MEANINGFUL_DEG and same_sign
            ),
        }
    return {
        "experiment_id": "step11-graph-ablation",
        "status": "diagnostic; no model was trained, no architecture changed",
        "split": SPLIT,
        "test_splits_read": [],
        "arm": "A3",
        "rotation_loss_weight": WEIGHT,
        "minimum_meaningful_deg": MINIMUM_MEANINGFUL_DEG,
        "rule": (
            "a condition changes what the model uses only if the mean rotation delta is at "
            f"least {MINIMUM_MEANINGFUL_DEG} deg and all three seeds agree in sign"
        ),
        "conditions": summary,
        "notes": [
            "Perturbation happens at evaluation on frozen weights; nothing is retrained.",
            "A perturbation the model does not notice is information it was not using.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 11 graph ablation on frozen A3.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--runs", type=Path, default=Path("experiments/runs/step10-change2-w3")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step11/graph_ablation.json")
    )
    args = parser.parse_args(argv)

    report = ablate(corpus_dir=args.corpus, runs_dir=args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\ncondition                     rotation   delta   same sign   meaningful")
    for name, entry in report["conditions"].items():
        print(
            f"{name:<28} {entry['rotation_deg']:7.2f} {entry['delta_rotation_deg']:+7.2f}   "
            f"{str(entry['all_seeds_same_sign']):<9}   {entry['exceeds_minimum']}"
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
