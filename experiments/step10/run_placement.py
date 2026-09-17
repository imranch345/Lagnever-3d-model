"""Step 10 Change 1: parent-relative placement against the global control.

Each cell differs from its control in exactly one thing — what the frame head is asked to
predict — and in nothing else. The model composes local frames into scene coordinates
before anything downstream sees them, so the loss, the metrics and the evaluation are Step
9's unchanged, and the numbers are comparable with Step 9 rather than merely similar.

The three cells
---------------

======================  =====================================================
``global``              Step 9's target, reproduced. The control.
``spatial``             Parent-relative over the generator's construction tree.
``taxonomic``           Parent-relative over AWR's own hierarchy.
======================  =====================================================

``taxonomic`` is expected to be **bit-identical** to ``global``, because AWR nests every
whole-organ entity under a geometry-free category node and so leaves all twenty as roots.
That is not a wasted cell: it is the measurement of what "just use AWR's hierarchy" would
have delivered, and it separates the hierarchy from the rest of Change 1. The identity is
asserted rather than hoped for.

The floor
---------

Every arm is scored on global frames with the Step 9 metric, so every arm is read against
the same placement-blind floor: the best identity-only predictor, 0.1605 on ``test_seen``.
The parent-relative floor (0.1636) is reported alongside as context for what the new
formulation costs a blind predictor, but it is not the bar — the bar is a property of the
task and the ruler, not of the arm.

    python -m experiments.step10.run_placement \
        --corpus datasets/processed/step8_continuous --steps 1200 --seeds 0 \
        --out experiments/runs/step10-placement
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.continuous_corpus import (
    TEST_SPLITS,
    load_step8_manifest,
    load_step8_split,
)
from experiments.step8.evaluation import evaluate_step8
from experiments.step9.frame_report import decompose, floor_gap
from experiments.step10.placement_floor import placement_floor
from training.manifest import environment_report, git_commit
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = ["CELLS", "run_placement", "main"]

#: name -> (placement target, hierarchy, parent convention, what it tests)
CELLS: Mapping[str, tuple[str, str, str, str]] = {
    "T0_global": (
        "global", "spatial", "isotropic",
        "Step 9's target, reproduced as the control",
    ),
    "T1_spatial": (
        "parent_relative", "spatial", "isotropic",
        "Change 1: placement relative to the construction parent",
    ),
    "T2_taxonomic": (
        "parent_relative", "taxonomic", "isotropic",
        "what AWR's own hierarchy would have delivered; expected identical to T0",
    ),
    "T3_alternative": (
        "parent_relative", "spatial_alternative", "isotropic",
        "the one arbitrary choice in the table: AV annuli parented to the atrium instead",
    ),
    "T4_rigid": (
        "parent_relative", "spatial", "rigid",
        "does a parent's size need to reach its children, or only its orientation",
    ),
}

REPORTED: tuple[str, ...] = (
    "translation_error",
    "scale_error",
    "rotation_error",
    "composite_frame_error",
    "placement_error",
    "placement_floor_gap",
    "entity_iou_mean",
    "part_control_success",
    "spatial_relation_accuracy",
    "scene_iou",
)


def run_placement(
    *,
    corpus_dir: Path,
    arms: Sequence[str],
    cells: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    batch_size: int,
    device: str,
    splits: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Train each cell of each arm at each seed and score it against the floor."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")
    floors = placement_floor(corpus_dir, splits=splits)

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            for name in cells:
                target, hierarchy, convention, question = CELLS[name]
                config = with_variant(
                    default_config(arm),
                    placement_target=target,  # type: ignore[arg-type]
                    hierarchy=hierarchy,
                    seed=seed,
                )
                config = replace(
                    config,
                    parent_convention=convention,
                    steps=steps,
                    batch_size=batch_size,
                    device=device,
                    eval_every=max(1, steps // 3),
                )
                print(f"[step10] {arm} {name} seed {seed} ...", flush=True)
                trainer = Step10Trainer(
                    config,
                    ontology,
                    domain,
                    train,
                    validation,
                    dataset_info={
                        "corpus_id": manifest.corpus_id,
                        "corpus_dir": str(corpus_dir),
                        "train_used": len(train),
                        "data_label": manifest.data_label,
                    },
                )
                run = trainer.fit(checkpoint_dir=output_dir / "checkpoints")
                record = run.to_dict()
                record.update(
                    {
                        "cell": name,
                        "arm": arm,
                        "question": question,
                        "placement_target": target,
                        "hierarchy": hierarchy,
                        "parent_convention": convention,
                        "parented_entities": sum(
                            1 for slot in trainer.config.placement_parents if slot >= 0
                        ),
                    }
                )
                record["splits"] = {}
                for split in splits:
                    scenes = load_step8_split(corpus_dir, split)
                    conditions: dict[str, dict[str, float]] = {}
                    for condition, predicted in (("inferred", True), ("supplied", False)):
                        values = evaluate_step8(
                            trainer.model,
                            scenes,
                            trainer.builder,
                            trainer.slot_of,
                            device=trainer.device,
                            batch_size=batch_size,
                            use_predicted_frames=predicted,
                        )
                        conditions[condition] = decompose(values)
                    floor = float(floors["splits"][split]["global"]["position_error"])
                    inferred = conditions["inferred"]
                    inferred["placement_floor"] = floor
                    inferred["placement_floor_gap"] = floor_gap(
                        float(inferred["translation_error"]), floor
                    )
                    inferred["parent_relative_floor"] = float(
                        floors["splits"][split]["parent_relative"]["position_error"]
                    )
                    inferred["oracle_parent_floor"] = float(
                        floors["splits"][split]["parent_relative_oracle"]["position_error"]
                    )
                    inferred["oracle_gap_entity_iou"] = float(
                        conditions["supplied"]["entity_iou_mean"]
                    ) - float(inferred["entity_iou_mean"])
                    record["splits"][split] = conditions
                runs.append(record)
                head = record["splits"][splits[0]]["inferred"]
                print(
                    f"[step10]   {arm} {name} seed {seed} {splits[0]}: "
                    f"translation={head['translation_error']:.4f} "
                    f"floor={head['placement_floor']:.4f} "
                    f"gap={head['placement_floor_gap']:+.4f} "
                    f"iou={head['entity_iou_mean']:.4f}",
                    flush=True,
                )

    report = {
        "experiment_id": "step10-placement",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "status": "exploratory" if len(seeds) < 3 else "confirmatory",
        "arms": list(arms),
        "cells": {name: CELLS[name][3] for name in cells},
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "splits": list(splits),
            "data_label": manifest.data_label,
        },
        "placement_floor": floors,
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "device": device,
            "seeds": list(seeds),
        },
        "runs": runs,
        "aggregates": _aggregate(runs, splits),
        "control_identity": _control_identity(runs, splits),
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "Headline condition is inferred placement.",
            "Every cell is scored on GLOBAL frames with the Step 9 metric, unchanged.",
            "The bar is the identity-only floor, which does not move with the target.",
            "T2_taxonomic must match T0_global; a difference means the control is not one.",
            "rotation_error is structurally zero: the corpus rotation target is constant.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "placement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _aggregate(runs: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> dict[str, Any]:
    """Mean, spread and seed count per arm, cell and split, inferred placement."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["arm"]), str(run["cell"])), []).append(run)
    out: dict[str, Any] = {}
    for (arm, cell), cell_runs in grouped.items():
        entry: dict[str, Any] = {"seeds": len(cell_runs)}
        for split in splits:
            per_split: dict[str, Any] = {}
            for metric in REPORTED:
                values = [
                    float(run["splits"][split]["inferred"][metric])
                    for run in cell_runs
                    if metric in run["splits"][split]["inferred"]
                ]
                if not values:
                    continue
                per_split[metric] = {
                    "mean": statistics.fmean(values),
                    "spread": (statistics.stdev(values) if len(values) > 1 else 0.0),
                    "n": len(values),
                }
            entry[split] = per_split
        out[f"{arm}/{cell}"] = entry
    return out


def _control_identity(
    runs: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, Any]:
    """Whether the taxonomic cell really reproduced the global one.

    AWR's hierarchy leaves every whole-organ entity a root, so composing over it is the
    identity and the two cells must agree exactly. Checked rather than assumed, because a
    control that has quietly become a third treatment invalidates the comparison.
    """
    out: dict[str, Any] = {}
    for run in runs:
        if run["cell"] != "T2_taxonomic":
            continue
        match = next(
            (
                other
                for other in runs
                if other["cell"] == "T0_global"
                and other["arm"] == run["arm"]
                and other["seed"] == run["seed"]
            ),
            None,
        )
        if match is None:
            continue
        key = f"{run['arm']}/seed{run['seed']}"
        out[key] = {
            split: abs(
                float(run["splits"][split]["inferred"]["translation_error"])
                - float(match["splits"][split]["inferred"]["translation_error"])
            )
            for split in splits
        }
        out[key]["identical"] = all(
            value < 1e-9 for key_, value in out[key].items() if key_ != "identical"
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 parent-relative placement.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step8_continuous"))
    parser.add_argument("--arms", nargs="+", default=["A3Lite"])
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step10-placement"))
    args = parser.parse_args(argv)

    report = run_placement(
        corpus_dir=args.corpus,
        arms=args.arms,
        cells=args.cells,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        splits=args.splits,
        output_dir=args.out,
    )
    print(json.dumps(report["aggregates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
