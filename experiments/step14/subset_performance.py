"""Step 14: where A3 actually fails, and whether it is where the graph is needed.

Section 12 of the brief makes the point that global averages can hide a relational learning
problem: a model could be at the identity-only floor precisely on the entities the graph would
fix, while looking respectable overall. So the preregistered
``relationally_required_validation`` subset is scored directly.

Each validation scene is built on its own, one scene per batch, so that a model prediction can
be joined to a lookup prediction by ``(scene index, entity slot)`` with no possibility of a
reordering silently misaligning the two. The loader batches by level of detail and would not
preserve corpus order.

Three quantities per entity, all on validation:

* the identity-only lookup error -- what identity alone can do;
* the graph-aware lookup error -- what the graph makes available;
* A3's error -- what was actually learned.

No test split is read. No model is trained.

    python -m experiments.step14.subset_performance
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step10.rotation_metrics import DEGREES
from experiments.step13.evaluate import SEEDS, build_arm
from experiments.step14.relational_identifiability import (
    IDENTITY_ONLY_VALIDATION_DEG,
    MINIMUM_GRAPH_ADVANTAGE_DEG,
    MINIMUM_SAMPLES,
    _as_frame,
    _fit_tables,
    _scene_records,
)
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

__all__ = ["measure_subset", "main"]


@torch.no_grad()
def _model_errors(trainer: Any, corpus_dir: Path) -> dict[tuple[int, int], float]:
    """A3's rotation error per (scene index, slot), one scene per batch to keep alignment."""
    out: dict[tuple[int, int], float] = {}
    for index, scene in enumerate(load_step8_split(corpus_dir, "validation")):
        whole = trainer.builder.build([scene])
        batch = whole.batch.to(trainer.device)
        output = trainer.model(batch, use_predicted_frames=True)
        present = batch.entity_present.bool()[0]
        predicted = output.frames.double()[0]
        truth = batch.entity_frames.double()[0]
        for slot in range(truth.shape[0]):
            if bool(present[slot]):
                out[(index, slot)] = float(rotation_angle(predicted[slot], truth[slot])) * DEGREES
    return out


def measure_subset(*, corpus_dir: Path, arm: str = "A3") -> dict[str, Any]:
    """Identity, graph-aware and A3 errors, on all of validation and on the subset."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    small = load_step8_split(corpus_dir, "train", limit=48)

    entity_table, pair_table, _ = _fit_tables(_scene_records(corpus_dir, "train", builder))
    validation = _scene_records(corpus_dir, "validation", builder)

    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for index, (signature, entities) in enumerate(validation):
        for slot, frame in entities.items():
            key = (signature, slot)
            covered = key in pair_table
            identity = _as_frame(entity_table[slot])
            conditioned = _as_frame(pair_table[key]) if covered else identity
            lookup[(index, slot)] = {
                "covered": covered,
                "identity_deg": float(rotation_angle(identity, frame)) * DEGREES,
                "graph_deg": float(rotation_angle(conditioned, frame)) * DEGREES,
            }

    required = {
        key
        for key, row in lookup.items()
        if row["covered"]
        and row["identity_deg"] >= IDENTITY_ONLY_VALIDATION_DEG
        and (row["identity_deg"] - row["graph_deg"]) >= MINIMUM_GRAPH_ADVANTAGE_DEG
    }

    per_seed: dict[str, dict[str, float]] = {}
    for seed in SEEDS:
        trainer = build_arm(arm, seed, ontology, domain, small)
        errors = _model_errors(trainer, corpus_dir)
        missing = set(lookup) - set(errors)
        if missing:
            raise SystemExit(f"{len(missing)} entities have no model prediction; alignment broke")
        everything = [errors[k] for k in lookup]
        subset = [errors[k] for k in required]
        rest = [errors[k] for k in lookup if k not in required]
        per_seed[str(seed)] = {
            "all_validation_deg": float(np.mean(everything)),
            "relationally_required_deg": float(np.mean(subset)),
            "remainder_deg": float(np.mean(rest)),
        }
        print(
            f"[step14] {arm} seed {seed}: all {per_seed[str(seed)]['all_validation_deg']:6.2f}  "
            f"required {per_seed[str(seed)]['relationally_required_deg']:6.2f}  "
            f"rest {per_seed[str(seed)]['remainder_deg']:6.2f}",
            flush=True,
        )

    def band(keys: Sequence[tuple[int, int]], field: str) -> float:
        return float(np.mean([lookup[k][field] for k in keys]))

    everything_keys = list(lookup)
    required_keys = sorted(required)
    rest_keys = [k for k in lookup if k not in required]

    def summarise(field: str) -> dict[str, Any]:
        values = [per_seed[str(s)][field] for s in SEEDS]
        return {
            "per_seed": {str(s): per_seed[str(s)][field] for s in SEEDS},
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
        }

    slot_counts: dict[str, int] = defaultdict(int)
    for _, slot in required_keys:
        slot_counts[str(slot)] += 1

    table: dict[str, dict[str, Any]] = {
        "all_validation": {
            "entities": len(everything_keys),
            "identity_only_deg": band(everything_keys, "identity_deg"),
            "graph_aware_deg": band(everything_keys, "graph_deg"),
            "a3": summarise("all_validation_deg"),
        },
        "relationally_required": {
            "entities": len(required_keys),
            "share": len(required_keys) / len(everything_keys),
            "identity_only_deg": band(required_keys, "identity_deg"),
            "graph_aware_deg": band(required_keys, "graph_deg"),
            "a3": summarise("relationally_required_deg"),
        },
        "remainder": {
            "entities": len(rest_keys),
            "identity_only_deg": band(rest_keys, "identity_deg"),
            "graph_aware_deg": band(rest_keys, "graph_deg"),
            "a3": summarise("remainder_deg"),
        },
    }
    for entry in table.values():
        identity_deg = float(entry["identity_only_deg"])
        graph_deg = float(entry["graph_aware_deg"])
        learned_deg = float(entry["a3"]["mean"])
        available = identity_deg - graph_deg
        entry["recovered_share_of_available"] = (
            (identity_deg - learned_deg) / available if available > 0 else None
        )
        entry["a3_versus_identity_deg"] = identity_deg - learned_deg
        entry["a3_versus_graph_deg"] = learned_deg - graph_deg
    return {
        "experiment_id": "step14-subset-performance",
        "status": "diagnostic; nothing trained",
        "preregistration": "docs/STEP_14_OBJECTIVE_DIAGNOSTIC_PLAN.md",
        "arm": arm,
        "fitted_on": "train",
        "measured_on": "validation",
        "test_splits_read": [],
        "subset_rule": (
            f"identity-only >= {IDENTITY_ONLY_VALIDATION_DEG} deg AND graph advantage "
            f">= {MINIMUM_GRAPH_ADVANTAGE_DEG} deg AND signature seen >= {MINIMUM_SAMPLES} times"
        ),
        "bands": table,
        "subset_slot_counts": dict(sorted(slot_counts.items(), key=lambda kv: -kv[1])),
        "notes": [
            "One scene per batch, so model and lookup predictions are joined by (scene, slot).",
            "recovered_share_of_available is (identity - A3) / (identity - graph_aware).",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 14 subset performance.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step14/subset_performance.json")
    )
    args = parser.parse_args(argv)
    report = measure_subset(corpus_dir=args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'band':<24}{'n':>6}{'identity':>10}{'A3':>9}{'graph':>9}{'recovered':>11}")
    for name, entry in report["bands"].items():
        share = entry["recovered_share_of_available"]
        recovered = "n/a" if share is None else f"{share * 100:.1f}%"
        print(
            f"{name:<24}{entry['entities']:6d}{entry['identity_only_deg']:10.2f}"
            f"{entry['a3']['mean']:9.2f}{entry['graph_aware_deg']:9.2f}{recovered:>11}"
        )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
