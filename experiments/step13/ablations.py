"""Step 13: how sensitive each arm is to the relationship graph it was given.

Section 14 of the brief asks for the established ablations, repeated where they remain valid,
on every arm. The question is not whether an arm scores well but whether it *notices* damage to
the graph: a perturbation the model does not notice is information the model was not using,
whatever its latent may contain.

This is the H2 prediction's third clause. If extra propagation makes the graph matter more, the
ablations should become more discriminative -- removing all relationships should cost more than
the control's 1.83 degrees, not the same.

Conditions are the five section 14 names. The two Step 11 conditions that isolate relation type
and endpoint identity are deliberately left out: Step 12 settled relation type, and neither is
what H2 or H6 is about.

Nothing is retrained. Frozen checkpoints, perturbed batches, validation only.

    python -m experiments.step13.ablations
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step7.perturbations import perturb_relations, restrict_graphs
from experiments.step8.frame_metrics import position_error, rotation_error
from experiments.step10.rotation_metrics import DEGREES
from experiments.step13.evaluate import ARMS, MINIMUM_MEANINGFUL_DEG, SEEDS, build_arm
from training.whole_organ import WholeOrganLoader

__all__ = ["CONDITIONS", "ablate_arms", "main"]

#: The section 14 list, in the brief's order.
CONDITIONS: tuple[str, ...] = (
    "intact",
    "entities_only",
    "drop_spatial",
    "drop_structure",
    "drop_functional",
)


@torch.no_grad()
def _score(
    model: Any, batches: Sequence[Any], condition: str, inverse: torch.Tensor
) -> dict[str, Any]:
    """Rotation and position error under one perturbation of the graph."""
    rotations: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    generator = torch.Generator().manual_seed(11)
    for batch in batches:
        damaged = (
            restrict_graphs(batch, keep=())
            if condition == "entities_only"
            else perturb_relations(batch, condition, inverse_relations=inverse, generator=generator)
            if condition != "intact"
            else batch
        )
        output = model(damaged, use_predicted_frames=True)
        present = damaged.entity_present.bool()
        truth = damaged.entity_frames
        rotations.append((rotation_error(output.frames, truth) * DEGREES)[present].numpy())
        positions.append(position_error(output.frames, truth)[present].numpy())
    rotation = np.concatenate(rotations)
    return {
        "rotation_deg": float(rotation.mean()),
        "position": float(np.concatenate(positions).mean()),
        "entities_scored": int(rotation.size),
    }


def ablate_arms(
    *,
    corpus_dir: Path,
    arms: Sequence[str] = tuple(ARMS),
    conditions: Sequence[str] = CONDITIONS,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Score every condition on every arm and seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    scenes = load_step8_split(corpus_dir, "validation")

    raw: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        for seed in SEEDS:
            trainer = build_arm(arm, seed, ontology, domain, small)
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
                raw.setdefault(arm, {}).setdefault(condition, {})[str(seed)] = entry
                print(
                    f"[step13] {arm:<10} seed {seed} {condition:<16} "
                    f"rotation {entry['rotation_deg']:6.2f}",
                    flush=True,
                )

    summary: dict[str, Any] = {}
    for arm, by_condition in raw.items():
        intact = [by_condition["intact"][str(s)]["rotation_deg"] for s in SEEDS]
        for condition, seeds in by_condition.items():
            values = [seeds[str(s)]["rotation_deg"] for s in SEEDS]
            deltas = [v - base for v, base in zip(values, intact, strict=True)]
            summary.setdefault(arm, {})[condition] = {
                "per_seed_rotation_deg": {str(s): seeds[str(s)]["rotation_deg"] for s in SEEDS},
                "rotation_deg": statistics.fmean(values),
                "rotation_std": statistics.stdev(values),
                "position": statistics.fmean([seeds[str(s)]["position"] for s in SEEDS]),
                "cost_of_damage_deg": statistics.fmean(deltas),
                "per_seed_cost_deg": {str(s): d for s, d in zip(SEEDS, deltas, strict=True)},
                "all_seeds_same_sign": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
            }

    discrimination = {
        arm: entry["entities_only"]["cost_of_damage_deg"] for arm, entry in summary.items()
    }
    return {
        "experiment_id": "step13-graph-ablations",
        "status": "diagnostic; no model was trained, nothing was perturbed at training time",
        "split": "validation",
        "test_splits_read": [],
        "minimum_meaningful_deg": MINIMUM_MEANINGFUL_DEG,
        "conditions": summary,
        "what_the_whole_graph_is_worth_deg": discrimination,
        "notes": [
            "A perturbation the model does not notice is information it was not using.",
            "H2 predicts a treatment makes these costs larger, not merely its score better.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 13 graph ablations across arms.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step13/graph_ablations.json")
    )
    args = parser.parse_args(argv)
    report = ablate_arms(corpus_dir=args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'condition':<16} " + "".join(f"{a:>22}" for a in report["conditions"]))
    for condition in CONDITIONS:
        row = f"{condition:<16} "
        for arm in report["conditions"]:
            entry = report["conditions"][arm][condition]
            row += f"{entry['rotation_deg']:10.2f} ({entry['cost_of_damage_deg']:+6.2f})"
        print(row)
    print("\nwhat the whole graph is worth:")
    for arm, value in report["what_the_whole_graph_is_worth_deg"].items():
        print(f"  {arm:<10} {value:+.2f} deg")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
