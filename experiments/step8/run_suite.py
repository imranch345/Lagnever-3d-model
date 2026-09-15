"""Step 8 principal experiment: five arms, three seeds, predicted placement.

Every headline number here is produced with the model predicting entity placement. The
supplied-placement condition is computed too and reported as an oracle upper bound, never
as a result, because supplying frames hands the model the arrangement a relationship graph
would otherwise have to supply.

    python -m experiments.step8.run_suite \
        --corpus datasets/processed/step8_continuous --steps 1200 \
        --seeds 0 1 2 --arms A1 A1M A3Lite A3L A3
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
from datasets.whole_organ.continuous_corpus import (
    TEST_SPLITS,
    load_step8_manifest,
    load_step8_split,
)
from experiments.step8.evaluation import evaluate_step8
from training.manifest import environment_report, git_commit
from training.step8 import Step8Config, Step8Trainer

__all__ = ["REPORTED", "run_arm", "run_suite", "main"]

#: Headline metrics. Fixed before the runs.
REPORTED: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "scene_iou",
    "occupancy_chamfer",
    "spatial_relation_accuracy",
    "spatial_relation_accuracy_strict",
    "spatial_relation_coverage",
    "placement_error",
    "position_error",
    "scale_error",
    "rotation_error",
    "composite_frame_error",
    "lod1_iou",
    "lod2_iou",
    "lod3_iou",
    "lod_detail_gain",
    "lod_containment_mean",
    "lod_preservation_mean",
    "eval_scenes",
)

#: Per-group frame errors, reported alongside the pooled ones.
GROUP_METRICS: tuple[str, ...] = tuple(
    f"{group}_{metric}"
    for group in ("chambers", "valves", "vessels", "septa", "walls")
    for metric in ("position_error", "composite_frame_error")
)


def run_arm(
    arm: str,
    seed: int,
    *,
    corpus_dir: Path,
    steps: int,
    batch_size: int,
    device: str,
    train_limit: int | None,
    eval_limit: int | None,
    checkpoint_dir: Path | None,
) -> Mapping[str, Any]:
    """Train one arm at one seed and evaluate it on every held-out split."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    train = load_step8_split(corpus_dir, "train", limit=train_limit)
    validation = load_step8_split(corpus_dir, "validation", limit=eval_limit)

    config = Step8Config(
        arm=arm,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        device=device,
        eval_every=max(1, steps // 3),
    )
    trainer = Step8Trainer(
        config,
        ontology,
        domain,
        train,
        validation,
        dataset_info={
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes": manifest.scenes,
            "families": manifest.families,
            "split_counts": dict(manifest.split_counts),
            "distinct_relation_graphs": manifest.distinct_relation_graphs,
            "train_used": len(train),
            "data_label": manifest.data_label,
        },
    )
    run = trainer.fit(checkpoint_dir=checkpoint_dir)
    record = run.to_dict()

    # Every held-out split, in both placement conditions. The inferred one is the result.
    splits: dict[str, dict[str, dict[str, float]]] = {}
    for split in TEST_SPLITS:
        scenes = load_step8_split(corpus_dir, split, limit=eval_limit)
        splits[split] = {
            "inferred": evaluate_step8(
                trainer.model,
                scenes,
                trainer.builder,
                trainer.slot_of,
                device=trainer.device,
                batch_size=batch_size,
                use_predicted_frames=True,
            ),
            "supplied": evaluate_step8(
                trainer.model,
                scenes,
                trainer.builder,
                trainer.slot_of,
                device=trainer.device,
                batch_size=batch_size,
                use_predicted_frames=False,
            ),
        }
    record["splits"] = splits
    return record


def aggregate(
    runs: Sequence[Mapping[str, Any]], split: str, condition: str
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean, spread and per-seed values per arm, for one split and condition."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["arm"]), []).append(run)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, arm_runs in grouped.items():
        out[arm] = {}
        for metric in (*REPORTED, *GROUP_METRICS):
            present = [
                (run, float(run["splits"][split][condition][metric]))
                for run in arm_runs
                if metric in run["splits"][split][condition]
            ]
            if not present:
                continue
            defined = [value for _, value in present if math.isfinite(value)]
            entry: dict[str, float] = {
                "mean": statistics.fmean(defined) if defined else float("nan"),
                "std": statistics.pstdev(defined) if len(defined) > 1 else 0.0,
                "n": float(len(present)),
                "defined_n": float(len(defined)),
            }
            for run, value in present:
                entry[f"seed_{run['seed']}"] = value
            out[arm][metric] = entry
    return out


def run_suite(
    *,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    arms: Sequence[str],
    batch_size: int,
    device: str,
    train_limit: int | None,
    eval_limit: int | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run every arm at every seed and write a machine-readable report."""
    started = time.time()
    manifest = load_step8_manifest(corpus_dir)
    runs: list[Mapping[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            print(f"[step8] {arm} seed {seed} ...", flush=True)
            record = run_arm(
                arm,
                seed,
                corpus_dir=corpus_dir,
                steps=steps,
                batch_size=batch_size,
                device=device,
                train_limit=train_limit,
                eval_limit=eval_limit,
                checkpoint_dir=output_dir / "checkpoints",
            )
            runs.append(record)
            summary = record["splits"]["test_seen"]["inferred"]
            print(
                f"[step8]   {arm} seed {seed} test_seen inferred: "
                f"iou={summary['entity_iou_mean']:.4f} "
                f"ctrl={summary['part_control_success']:.4f} "
                f"pos={summary['position_error']:.4f} "
                f"rel={summary['spatial_relation_accuracy']:.4f}",
                flush=True,
            )

    report = build_report(
        runs,
        manifest=manifest,
        corpus_dir=corpus_dir,
        steps=steps,
        seeds=seeds,
        arms=arms,
        batch_size=batch_size,
        device=device,
        runtime_seconds=time.time() - started,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step8_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_report(
    runs: Sequence[Mapping[str, Any]],
    *,
    manifest: Any,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    arms: Sequence[str],
    batch_size: int,
    device: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Assemble the report from completed runs."""
    aggregates = {
        split: {
            condition: aggregate(runs, split, condition) for condition in ("inferred", "supplied")
        }
        for split in TEST_SPLITS
    }
    return {
        "experiment_id": "step8-principal-suite",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes": manifest.scenes,
            "families": manifest.families,
            "split_counts": dict(manifest.split_counts),
            "split_families": {k: list(v) for k, v in manifest.split_families.items()},
            "distinct_relation_graphs": manifest.distinct_relation_graphs,
            "arrangement_fields": list(manifest.arrangement_fields),
            "data_label": manifest.data_label,
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "device": device,
            "seeds": list(seeds),
            "arms": list(arms),
            "runtime_seconds": round(runtime_seconds, 1),
            "checkpoint_selection": "final checkpoint",
            "headline_condition": "inferred placement",
            "parameter_counts": {str(run["arm"]): dict(run["parameters"]) for run in runs},
        },
        "runs": [dict(run) for run in runs],
        "aggregates": aggregates,
        "notes": [
            "Every headline number is produced with the model predicting placement.",
            "The supplied-placement condition is an oracle upper bound, not a result.",
            "Arrangements are continuous, so arrangement classification is not a shortcut.",
            "Relation accuracy must be read against the relation-blind floor, which is "
            "recomputed for this corpus.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def load_runs(checkpoint_dir: Path) -> list[Mapping[str, Any]]:
    """Recover completed runs from the manifests written during training."""
    runs: list[Mapping[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("results"):
            runs.append(payload)
    return runs


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 principal suite.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=["A1", "A1M", "A3Lite", "A3L", "A3"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step8-suite"))
    args = parser.parse_args(argv)

    report = run_suite(
        corpus_dir=args.corpus,
        steps=args.steps,
        seeds=args.seeds,
        arms=args.arms,
        batch_size=args.batch_size,
        device=args.device,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        output_dir=args.out,
    )
    print(json.dumps(report["aggregates"]["test_seen"]["inferred"], indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
