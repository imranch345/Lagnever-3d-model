"""Step 7 experiments 5 and 6: whole-organ generalisation and the graph ablation.

    python -m experiments.step7.run_whole_organ \
        --corpus datasets/processed/whole_organ_1600 --steps 900 \
        --seeds 0 1 2 --arms A3 A1M A0 --single-seed-arms A1 A2 A3L
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.corpus import load_manifest, load_split
from training.manifest import environment_report, git_commit
from training.whole_organ import WholeOrganConfig, WholeOrganTrainer

__all__ = ["run_arm", "run_suite", "main"]

REPORTED: tuple[str, ...] = (
    "part_control_success",
    "entity_iou_mean",
    "spatial_relation_accuracy",
    "spatial_relation_accuracy_strict",
    "spatial_relation_coverage",
    "placement_error",
    "scene_iou",
    "occupancy_chamfer",
    "lod1_iou",
    "lod2_iou",
    "lod3_iou",
    "lod_detail_gain",
    "lod_token_delta",
    "counterfactual_relation_sensitivity",
    "counterfactual_correctness",
    "invariant_preservation",
    "counterfactual_entities",
)


def run_arm(
    arm: str,
    seed: int,
    *,
    corpus_dir: Path,
    steps: int,
    batch_size: int,
    scene_points: int,
    entity_points: int,
    device: str,
    train_limit: int | None,
    eval_limit: int | None,
    checkpoint_dir: Path | None,
) -> Mapping[str, Any]:
    """Train and evaluate one arm at one seed on the whole-organ corpus."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_manifest(corpus_dir)
    train = load_split(corpus_dir, "train", limit=train_limit)
    evaluation = load_split(corpus_dir, "test", limit=eval_limit)
    config = WholeOrganConfig(
        arm=arm,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        scene_points=scene_points,
        entity_points=entity_points,
        device=device,
        eval_every=max(1, steps // 3),
    )
    trainer = WholeOrganTrainer(
        config,
        ontology,
        domain,
        train,
        evaluation,
        dataset_info={
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes": manifest.scenes,
            "families": manifest.families,
            "variants": dict(manifest.variants),
            "distinct_relation_graphs": manifest.distinct_relation_graphs,
            "train_used": len(train),
            "test_used": len(evaluation),
            "data_label": manifest.data_label,
        },
    )
    run = trainer.fit(checkpoint_dir=checkpoint_dir)
    return run.to_dict()


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean, spread and per-seed values per arm."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["arm"]), []).append(run)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, arm_runs in grouped.items():
        out[arm] = {}
        for metric in REPORTED:
            values = [
                float(run["results"][metric]) for run in arm_runs if metric in run["results"]
            ]
            if not values:
                continue
            entry = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "n": float(len(values)),
            }
            for run, value in zip(arm_runs, values, strict=False):
                entry[f"seed_{run['seed']}"] = value
            out[arm][metric] = entry
    return out


def run_suite(
    *,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    arms: Sequence[str],
    single_seed_arms: Sequence[str],
    batch_size: int,
    scene_points: int,
    entity_points: int,
    device: str,
    train_limit: int | None,
    eval_limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run every arm and write a machine-readable report."""
    started = time.time()
    manifest = load_manifest(corpus_dir)
    plan: list[tuple[str, int]] = [(arm, seed) for seed in seeds for arm in arms]
    plan.extend((arm, seeds[0]) for arm in single_seed_arms)

    runs: list[Mapping[str, Any]] = []
    for arm, seed in plan:
        print(f"[step7] {arm} seed {seed} ...", flush=True)
        run = run_arm(
            arm,
            seed,
            corpus_dir=corpus_dir,
            steps=steps,
            batch_size=batch_size,
            scene_points=scene_points,
            entity_points=entity_points,
            device=device,
            train_limit=train_limit,
            eval_limit=eval_limit,
            checkpoint_dir=output_dir / "checkpoints",
        )
        summary = {
            key: round(float(value), 4)
            for key, value in run["results"].items()
            if key in REPORTED
        }
        print(f"[step7]   {arm} seed {seed}: {summary}", flush=True)
        runs.append(run)

    report = {
        "experiment_id": "step7-whole-organ",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "scenes": manifest.scenes,
            "families": manifest.families,
            "variants": dict(manifest.variants),
            "split_counts": dict(manifest.split_counts),
            "distinct_relation_graphs": manifest.distinct_relation_graphs,
            "entity_count": manifest.entity_count,
            "data_label": manifest.data_label,
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "scene_points": scene_points,
            "entity_points": entity_points,
            "device": device,
            "seeds": list(seeds),
            "arms": list(arms),
            "single_seed_arms": list(single_seed_arms),
            "runtime_seconds": round(time.time() - started, 1),
            "checkpoint_selection": "final checkpoint",
            "parameter_counts": {str(run["arm"]): int(run["parameters"]["total"]) for run in runs},
        },
        "runs": [dict(run) for run in runs],
        "aggregates": aggregate(runs),
        "notes": [
            "Whole-organ corpus: the organ is generated first and segmented afterwards.",
            "Relationship graphs are measured per scene, so the graph carries information.",
            "Spatial relation metrics are never trained on.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "whole_organ_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 7 whole-organ experiments.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=["A3", "A1M", "A0"])
    parser.add_argument("--single-seed-arms", nargs="*", default=["A1", "A2", "A3L"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scene-points", type=int, default=256)
    parser.add_argument("--entity-points", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step7-whole-organ"))
    args = parser.parse_args(argv)

    report = run_suite(
        corpus_dir=args.corpus,
        steps=args.steps,
        seeds=args.seeds,
        arms=args.arms,
        single_seed_arms=args.single_seed_arms,
        batch_size=args.batch_size,
        scene_points=args.scene_points,
        entity_points=args.entity_points,
        device=args.device,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        output_dir=args.out,
    )
    print(json.dumps({"aggregates": report["aggregates"]}, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
