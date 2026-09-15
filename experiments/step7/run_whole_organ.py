"""Step 7 experiments 5 and 6: whole-organ generalisation and the graph ablation.

python -m experiments.step7.run_whole_organ \
        --corpus datasets/processed/whole_organ_1600 --steps 900 \
        --seeds 0 1 2 --arms A3 A1M A0 --single-seed-arms A1 A2 A3L
"""

from __future__ import annotations

import argparse
import json
import math
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

__all__ = ["run_arm", "run_suite", "load_runs", "build_report", "aggregate", "main"]

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
    "eval_scenes",
)

#: Per-arrangement breakdown of the headline metrics. The pre-registered headline set
#: above is unchanged; these keys split the same quantities by anatomical arrangement,
#: which is what shows whether a model treats the arrangements differently at all.
VARIANT_METRICS: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "spatial_relation_accuracy",
    "spatial_relation_accuracy_strict",
    "placement_error",
    "scene_iou",
)
VARIANTS: tuple[str, ...] = ("normal", "mirrored", "transposed", "rotated")
BY_VARIANT: tuple[str, ...] = tuple(
    f"variant_{variant}_{metric}" for variant in VARIANTS for metric in VARIANT_METRICS
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
    """Mean, spread and per-seed values per arm.

    A metric can be legitimately **undefined** for a run: ``placement_error`` has no
    value when an arm places no entity at all, which is exactly what the appearance
    baseline does. Undefined values are recorded as NaN per seed, excluded from the
    summary statistics, and counted in ``defined_n`` so that a summary computed from
    two of three seeds is never mistaken for one computed from three. Dropping the run
    instead would quietly flatter the arm that failed.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["arm"]), []).append(run)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, arm_runs in grouped.items():
        out[arm] = {}
        for metric in (*REPORTED, *BY_VARIANT):
            present = [
                (run, float(run["results"][metric]))
                for run in arm_runs
                if metric in run["results"]
            ]
            if not present:
                continue
            defined = [value for _, value in present if math.isfinite(value)]
            entry: dict[str, float] = {
                "mean": statistics.fmean(defined) if defined else float("nan"),
                "std": statistics.pstdev(defined) if len(defined) > 1 else 0.0,
                "min": min(defined) if defined else float("nan"),
                "max": max(defined) if defined else float("nan"),
                "n": float(len(present)),
                "defined_n": float(len(defined)),
            }
            for run, value in present:
                entry[f"seed_{run['seed']}"] = value
            out[arm][metric] = entry
    return out


def load_runs(checkpoint_dir: Path) -> list[Mapping[str, Any]]:
    """Recover completed runs from the per-run manifests written during training.

    Each run is persisted the moment it finishes, so a later failure in reporting
    never costs the training time again.
    """
    runs: list[Mapping[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("results"):
            runs.append(payload)
    return runs


def build_report(
    runs: Sequence[Mapping[str, Any]],
    *,
    manifest: Any,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    arms: Sequence[str],
    single_seed_arms: Sequence[str],
    batch_size: int,
    scene_points: int,
    entity_points: int,
    device: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Assemble the machine-readable report from completed runs."""
    return {
        "experiment_id": "step7-whole-organ",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes": manifest.scenes,
            "families": manifest.families,
            "variants": dict(manifest.variants),
            "split_counts": dict(manifest.split_counts),
            "split_variants": {
                split: dict(counts) for split, counts in manifest.split_variants.items()
            },
            "variants_per_family_min": (
                min(manifest.variants_per_family.values())
                if manifest.variants_per_family
                else 0
            ),
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
            "runtime_seconds": round(runtime_seconds, 1),
            "checkpoint_selection": "final checkpoint",
            "parameter_counts": {
                str(run["arm"]): int(run["parameters"]["total"]) for run in runs
            },
        },
        "runs": [dict(run) for run in runs],
        "aggregates": aggregate(runs),
        "notes": [
            "Whole-organ corpus: the organ is generated first and segmented afterwards.",
            "Relationship graphs are measured per scene, so the graph carries information.",
            "Family and variant are crossed, so every family is seen in every arrangement.",
            "Spatial relation metrics are never trained on.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


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

    report = build_report(
        runs,
        manifest=manifest,
        corpus_dir=corpus_dir,
        steps=steps,
        seeds=seeds,
        arms=arms,
        single_seed_arms=single_seed_arms,
        batch_size=batch_size,
        scene_points=scene_points,
        entity_points=entity_points,
        device=device,
        runtime_seconds=time.time() - started,
    )
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
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Re-assemble the report from finished run manifests without training.",
    )
    args = parser.parse_args(argv)

    if args.rebuild_only:
        runs = load_runs(args.out / "checkpoints")
        if not runs:
            parser.error(f"No finished runs under {args.out / 'checkpoints'}.")
        report = build_report(
            runs,
            manifest=load_manifest(args.corpus),
            corpus_dir=args.corpus,
            steps=args.steps,
            seeds=args.seeds,
            arms=args.arms,
            single_seed_arms=args.single_seed_arms,
            batch_size=args.batch_size,
            scene_points=args.scene_points,
            entity_points=args.entity_points,
            device=args.device,
            runtime_seconds=float(sum(float(r.get("runtime_seconds", 0.0)) for r in runs)),
        )
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "whole_organ_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps({"aggregates": report["aggregates"]}, indent=2)[:3000])
        return 0

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
