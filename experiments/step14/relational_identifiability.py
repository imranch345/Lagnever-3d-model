"""Step 14: whether relational information is in the target and the objective rewards it.

The brief's decision rule has two clauses, and they are answered by different measurements.

**Clause 1 -- is relationship-dependent target variation present and measurable?** A variance
decomposition. For each entity slot, the rotation target varies across scenes. Some of that
variation is explained by which graph the scene has and some is not. If the between-graph share
is negligible, there is nothing relational to learn and H7 is moot.

**Clause 2 -- does the objective supervise that variation?** This is the measurement that must
not be taken from the model. Two predictions are constructed from the training data alone:

``identity_only``
    the best rotation for this entity slot, ignoring the graph -- a chordal mean over every
    training scene containing the slot.
``graph_conditioned``
    the best rotation for this slot *given its scene's relation signature*.

Both are then scored with **the objective's own rotation term**, ``rotation_chordal``, exactly
as the loss computes it. If the graph-conditioned prediction receives a materially lower loss,
the objective already rewards relational conditioning and the supervision exists whether or not
A3 exploits it. This is model-independent by construction: no checkpoint is loaded.

A3 enters only afterwards, to answer where it actually fails (brief sections 12 and 13).

Lookup tables are fitted on **train** only. Everything else is validation. No test split is
read.

    python -m experiments.step14.relational_identifiability
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
from datasets.whole_organ.relations import relation_signature
from experiments.step10.placement_floor import _mean_rotation
from experiments.step10.rotation_metrics import DEGREES
from generation.neural.nn.transforms import rotation_angle, rotation_chordal
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

__all__ = [
    "IDENTITY_ONLY_VALIDATION_DEG",
    "MINIMUM_GRAPH_ADVANTAGE_DEG",
    "OBJECTIVE_SEPARATION_THRESHOLD",
    "analyse",
    "main",
]

#: Preregistered subset rule (plan section 3.1).
IDENTITY_ONLY_VALIDATION_DEG = 23.51
MINIMUM_GRAPH_ADVANTAGE_DEG = 10.0

#: Preregistered clause-2 threshold (plan section 3.2): a >= 10% reduction in the objective's
#: rotation term counts as the objective rewarding relational conditioning.
OBJECTIVE_SEPARATION_THRESHOLD = 0.10

#: A signature must appear at least this often in train to be a usable conditional key, the
#: same rule the existing graph-conditional floor applies.
MINIMUM_SAMPLES = 2


def _scene_records(
    corpus_dir: str | Path, split: str, builder: WholeOrganBatchBuilder
) -> list[tuple[str, dict[int, torch.Tensor]]]:
    """Per scene: its relation signature, and each present entity's frame as the model sees it."""
    out: list[tuple[str, dict[int, torch.Tensor]]] = []
    for scene in load_step8_split(corpus_dir, split):
        whole = builder.build([scene])
        frames = whole.batch.entity_frames.double()[0]
        present = whole.batch.entity_present.bool()[0]
        entities = {
            slot: frames[slot].clone() for slot in range(frames.shape[0]) if bool(present[slot])
        }
        out.append((relation_signature(scene.edges), entities))
    return out


def _as_frame(rotation: torch.Tensor) -> torch.Tensor:
    """A 12-wide frame carrying only a rotation, which is all these predictors predict."""
    blank = torch.zeros(12, dtype=torch.float64)
    blank[6:12] = rotation
    return blank


def _variance_decomposition(
    records: Sequence[tuple[str, dict[int, torch.Tensor]]],
) -> dict[str, Any]:
    """How much of each slot's rotation variation is explained by its scene's graph.

    Measured in the objective's own units. For one slot, the *total* spread is the mean chordal
    distance of its rotations from their own overall mean; the *within-graph* spread is the mean
    distance from the mean of the slot's rotations that share a signature. What the graph
    explains is the difference.

    ``_mean_rotation`` reads channels 6:12 of a full frame, so whole frames are carried
    throughout rather than the rotation slice alone.
    """
    by_slot: dict[int, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for signature, entities in records:
        for slot, frame in entities.items():
            by_slot[slot].append((signature, frame))

    def spread(frames: torch.Tensor) -> float:
        """Mean chordal distance of a set of frames from their own chordal mean."""
        centre = _as_frame(_mean_rotation(frames)).expand(frames.shape[0], 12)
        return float(rotation_chordal(centre, frames).mean())

    total_spread: list[float] = []
    within_spread: list[float] = []
    per_slot: dict[str, dict[str, float]] = {}
    for slot, entries in by_slot.items():
        frames = torch.stack([f for _, f in entries])
        total = spread(frames)
        groups: dict[str, list[torch.Tensor]] = defaultdict(list)
        for signature, frame in entries:
            groups[signature].append(frame)
        within_values = [
            spread(torch.stack(members))
            for members in groups.values()
            if len(members) >= MINIMUM_SAMPLES
        ]
        if not within_values:
            continue
        within = statistics.fmean(within_values)
        total_spread.append(total)
        within_spread.append(within)
        per_slot[str(slot)] = {
            "total_spread": total,
            "within_graph_spread": within,
            "explained_by_graph": total - within,
            "share_explained": (total - within) / total if total > 0 else 0.0,
            "distinct_graphs": len(groups),
        }

    total_mean = statistics.fmean(total_spread)
    within_mean = statistics.fmean(within_spread)
    return {
        "metric": "mean chordal distance from the conditional mean, the objective's own units",
        "slots_measured": len(per_slot),
        "total_spread": total_mean,
        "within_graph_spread": within_mean,
        "explained_by_graph": total_mean - within_mean,
        "share_explained_by_graph": (total_mean - within_mean) / total_mean,
        "per_slot": per_slot,
    }


def _fit_tables(
    records: Sequence[tuple[str, dict[int, torch.Tensor]]],
) -> tuple[dict[int, torch.Tensor], dict[tuple[str, int], torch.Tensor], dict[str, Any]]:
    """Identity-only and graph-conditioned rotation tables, fitted on train."""
    by_entity: dict[int, list[torch.Tensor]] = defaultdict(list)
    by_pair: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    for signature, entities in records:
        for slot, frame in entities.items():
            by_entity[slot].append(frame)
            by_pair[(signature, slot)].append(frame)
    entity_table = {slot: _mean_rotation(torch.stack(v)) for slot, v in by_entity.items()}
    pair_table = {
        key: _mean_rotation(torch.stack(v))
        for key, v in by_pair.items()
        if len(v) >= MINIMUM_SAMPLES
    }
    return entity_table, pair_table, {"samples": {k: len(v) for k, v in by_pair.items()}}


def analyse(*, corpus_dir: Path) -> dict[str, Any]:
    """Every clause-1 and clause-2 measurement, plus the preregistered subset."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)

    train = _scene_records(corpus_dir, "train", builder)
    validation = _scene_records(corpus_dir, "validation", builder)
    entity_table, pair_table, _ = _fit_tables(train)

    print("[step14] variance decomposition on train", flush=True)
    clause_one = _variance_decomposition(train)

    # ---- clause 2: score both predictors with the objective's own rotation term ----
    print("[step14] objective identifiability on validation", flush=True)
    rows: list[dict[str, Any]] = []
    for signature, entities in validation:
        for slot, frame in entities.items():
            key = (signature, slot)
            covered = key in pair_table
            identity = _as_frame(entity_table[slot])
            conditioned = _as_frame(pair_table[key]) if covered else identity
            rows.append(
                {
                    "slot": slot,
                    "graph_seen_in_training": covered,
                    "identity_loss": float(rotation_chordal(identity, frame)),
                    "conditioned_loss": float(rotation_chordal(conditioned, frame)),
                    "identity_deg": float(rotation_angle(identity, frame)) * DEGREES,
                    "conditioned_deg": float(rotation_angle(conditioned, frame)) * DEGREES,
                }
            )

    covered_rows = [r for r in rows if r["graph_seen_in_training"]]
    identity_loss = float(np.mean([r["identity_loss"] for r in covered_rows]))
    conditioned_loss = float(np.mean([r["conditioned_loss"] for r in covered_rows]))
    reduction = (identity_loss - conditioned_loss) / identity_loss
    clause_two = {
        "scored_with": "rotation_chordal, the objective's own rotation term",
        "entities_scored": len(covered_rows),
        "coverage": len(covered_rows) / len(rows),
        "identity_only_loss": identity_loss,
        "graph_conditioned_loss": conditioned_loss,
        "absolute_reduction": identity_loss - conditioned_loss,
        "relative_reduction": reduction,
        "threshold": OBJECTIVE_SEPARATION_THRESHOLD,
        "objective_rewards_relational_conditioning": bool(
            reduction >= OBJECTIVE_SEPARATION_THRESHOLD
        ),
        "identity_only_deg": float(np.mean([r["identity_deg"] for r in covered_rows])),
        "graph_conditioned_deg": float(np.mean([r["conditioned_deg"] for r in covered_rows])),
    }

    # ---- the preregistered relationally-required subset ----
    required = [
        r
        for r in covered_rows
        if r["identity_deg"] >= IDENTITY_ONLY_VALIDATION_DEG
        and (r["identity_deg"] - r["conditioned_deg"]) >= MINIMUM_GRAPH_ADVANTAGE_DEG
    ]
    subset = {
        "rule": (
            f"identity-only error >= {IDENTITY_ONLY_VALIDATION_DEG} deg AND graph-aware error "
            f"at least {MINIMUM_GRAPH_ADVANTAGE_DEG} deg lower; graph signature seen in training"
        ),
        "selected_from": "validation",
        "entities": len(required),
        "share_of_validation": len(required) / len(rows),
        "identity_only_deg": float(np.mean([r["identity_deg"] for r in required])),
        "graph_conditioned_deg": float(np.mean([r["conditioned_deg"] for r in required])),
        "slots_represented": len({r["slot"] for r in required}),
        "slot_counts": {
            str(slot): sum(1 for r in required if r["slot"] == slot)
            for slot in sorted({r["slot"] for r in required})
        },
    }

    return {
        "experiment_id": "step14-relational-identifiability",
        "status": "diagnostic; no model was loaded for clause 1 or clause 2, nothing was trained",
        "preregistration": "docs/STEP_14_OBJECTIVE_DIAGNOSTIC_PLAN.md",
        "fitted_on": "train",
        "measured_on": "validation",
        "test_splits_read": [],
        "clause_1_target_variation": clause_one,
        "clause_2_objective_supervision": clause_two,
        "relationally_required_validation": subset,
        "notes": [
            "Clause 2 is model-independent: it compares two lookup predictors under the real "
            "loss, so it measures the objective rather than A3's behaviour.",
            "Entities whose graph is unseen in training are excluded from clause 2 and the "
            "subset, because the conditioned predictor is defined to equal the identity one.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 14 relational identifiability.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step14/relational_identifiability.json"),
    )
    args = parser.parse_args(argv)
    report = analyse(corpus_dir=args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    one = report["clause_1_target_variation"]
    print("\n=== clause 1: is relational variation present in the target? ===")
    print(f"  total spread            {one['total_spread']:.5f}")
    print(f"  within-graph spread     {one['within_graph_spread']:.5f}")
    print(
        f"  explained by the graph  {one['explained_by_graph']:.5f} "
        f"({one['share_explained_by_graph'] * 100:.1f}%)"
    )
    two = report["clause_2_objective_supervision"]
    print("\n=== clause 2: does the objective supervise it? ===")
    identity_deg = two["identity_only_deg"]
    print(f"  identity-only loss      {two['identity_only_loss']:.5f}  ({identity_deg:.2f} deg)")
    print(
        f"  graph-conditioned loss  {two['graph_conditioned_loss']:.5f}  "
        f"({two['graph_conditioned_deg']:.2f} deg)"
    )
    print(
        f"  reduction               {two['relative_reduction'] * 100:.1f}%  "
        f"(threshold {two['threshold'] * 100:.0f}%)  "
        f"rewards relational: {two['objective_rewards_relational_conditioning']}"
    )
    sub = report["relationally_required_validation"]
    print("\n=== relationally-required subset ===")
    print(
        f"  {sub['entities']} entities ({sub['share_of_validation'] * 100:.1f}% of validation), "
        f"{sub['slots_represented']} slots"
    )
    print(
        f"  identity {sub['identity_only_deg']:.2f} deg -> graph-aware "
        f"{sub['graph_conditioned_deg']:.2f} deg"
    )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
