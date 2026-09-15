"""Step 9 exploratory suite, asking which switch moves placement off the floor.

Four variants of one arm, differing only in the two things the audit identified. Every
other setting is the Step 8 configuration, so a difference between cells is attributable.

======  ==============  ==================  ==================================
name    scene context   frame objective     hypothesis it tests
======  ==============  ==================  ==================================
S0      no              L1                  the Step 8 control, reproduced
S1      **yes**         L1                  P4: placement needs scene context
S2      no              **euclidean**       the objective is misaligned with the metric
S3      **yes**         **euclidean**       both, to see whether they interact
======  ==============  ==================  ==================================

Run exploratory first at one seed. Only the cell that moves is worth three seeds, and the
distinction between an exploratory and a confirmatory run is recorded in the report rather
than left to the reader to infer.

    python -m experiments.step9.run_variants \
        --corpus datasets/processed/step8_continuous --steps 1200 --seeds 0 \
        --out experiments/runs/step9-variants
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

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
from experiments.step9.placement_floor import placement_floor
from training.manifest import environment_report, git_commit
from training.step9 import FrameObjective, Step9Trainer, default_config, with_variant

__all__ = ["VARIANTS", "run_variants", "main"]

#: name -> (scene context, frame objective, what it tests)
VARIANTS: Mapping[str, tuple[bool, str, str]] = {
    "S0_control": (False, "l1", "the Step 8 configuration, reproduced"),
    "S1_scene_context": (True, "l1", "P4: the frame head needs a summary of the scene"),
    "S2_euclidean": (False, "euclidean", "the objective is misaligned with the metric"),
    "S3_both": (True, "euclidean", "both, to see whether they interact"),
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


def run_variants(
    *,
    corpus_dir: Path,
    arm: str,
    variants: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    batch_size: int,
    device: str,
    splits: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Train each variant at each seed and score it against the placement-blind floor."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")
    floors = placement_floor(corpus_dir, splits=splits, batch_size=batch_size)

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for name in variants:
            context, objective, question = VARIANTS[name]
            config = with_variant(
                default_config(arm),
                scene_context=context,
                objective=cast(FrameObjective, objective),
                seed=seed,
            )
            config = replace(
                config,
                steps=steps,
                batch_size=batch_size,
                device=device,
                eval_every=max(1, steps // 3),
            )
            print(f"[step9] {name} seed {seed} ...", flush=True)
            trainer = Step9Trainer(
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
            record["variant"] = name
            record["question"] = question
            record["scene_context"] = context
            record["frame_objective"] = objective

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
                floor = float(floors["splits"][split]["placement_blind_floor"])
                conditions["inferred"]["placement_floor"] = floor
                conditions["inferred"]["placement_floor_gap"] = floor_gap(
                    float(conditions["inferred"]["translation_error"]), floor
                )
                conditions["inferred"]["oracle_gap_entity_iou"] = float(
                    conditions["supplied"]["entity_iou_mean"]
                ) - float(conditions["inferred"]["entity_iou_mean"])
                record["splits"][split] = conditions
            runs.append(record)
            head = record["splits"][splits[0]]["inferred"]
            print(
                f"[step9]   {name} seed {seed} {splits[0]}: "
                f"translation={head['translation_error']:.4f} "
                f"floor={head['placement_floor']:.4f} "
                f"gap={head['placement_floor_gap']:+.4f} "
                f"iou={head['entity_iou_mean']:.4f}",
                flush=True,
            )

    report = {
        "experiment_id": "step9-variants",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "status": "exploratory" if len(seeds) < 3 else "confirmatory",
        "arm": arm,
        "variants": {name: VARIANTS[name][2] for name in variants},
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
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "Headline condition is inferred placement.",
            "A translation error is read against the placement-blind floor, not zero.",
            "The Step 8 composite is preserved; placement_error is the corrected metric.",
            "rotation_error is structurally zero under the current frame target.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variants_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _aggregate(runs: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> dict[str, Any]:
    """Mean, spread and seed count per variant and split, inferred placement."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["variant"]), []).append(run)
    out: dict[str, Any] = {}
    for variant, variant_runs in grouped.items():
        out[variant] = {}
        for split in splits:
            metrics: dict[str, dict[str, float]] = {}
            for metric in REPORTED:
                values = [
                    float(run["splits"][split]["inferred"][metric])
                    for run in variant_runs
                    if metric in run["splits"][split]["inferred"]
                    and math.isfinite(float(run["splits"][split]["inferred"][metric]))
                ]
                if not values:
                    continue
                metrics[metric] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n": float(len(values)),
                }
            out[variant][split] = metrics
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 9 variant suite.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--arm", default="A3Lite")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step9-variants"))
    args = parser.parse_args(argv)

    run_variants(
        corpus_dir=args.corpus,
        arm=args.arm,
        variants=args.variants,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        splits=args.splits,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
