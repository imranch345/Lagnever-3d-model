"""Run experiment heart-001: structured anatomical representation versus appearance.

    python -m experiments.heart.run_experiment --smoke
    python -m experiments.heart.run_experiment --corpus datasets/processed/heart_tier0_2000 \
        --steps 500 --seeds 0 1 2 --ablations A1 A4 A5

The pre-registered success criteria from Step 5 are evaluated here exactly as written.
They are not parameters of this script: changing a threshold means editing the
pre-registration, which is deliberately a separate, visible act.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.synthetic.store import generate_corpus, load_manifest, load_split
from training.loop import ArmName, Trainer, TrainingConfig
from training.manifest import environment_report, git_commit

__all__ = ["CriterionOutcome", "ExperimentReport", "run_arm", "run_experiment", "smoke_run", "main"]

PRIMARY_METRIC = "part_control_success"
RELATIONSHIP_METRIC = "relationship_accuracy_strict"
"""The coverage-strict form: an entity the model cannot place fails its relations.

The pre-registration said "held-out relationship accuracy" without fixing which of the
two defensible definitions to use. They diverge sharply between the arms, so both are
reported and the strict one decides the verdict. This choice is recorded here, in the
report and in the Step 6 write-up, because it was made after seeing that the arms
differ in coverage.
"""
DRIFT_METRIC = "untouched_drift"
GEOMETRY_METRIC = "occupancy_chamfer"


@dataclass(slots=True)
class CriterionOutcome:
    """One pre-registered criterion, evaluated."""

    hypothesis: str
    metric: str
    threshold: str
    verdict: str
    detail: str
    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "hypothesis": self.hypothesis,
            "metric": self.metric,
            "threshold": self.threshold,
            "verdict": self.verdict,
            "detail": self.detail,
            "values": dict(self.values),
        }


@dataclass(slots=True)
class ExperimentReport:
    """Everything the run produced."""

    experiment_id: str
    commit: str
    environment: Mapping[str, Any]
    dataset: Mapping[str, Any]
    training: Mapping[str, Any]
    runs: list[Mapping[str, Any]] = field(default_factory=list)
    aggregates: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    criteria: list[CriterionOutcome] = field(default_factory=list)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "experiment_id": self.experiment_id,
            "commit": self.commit,
            "environment": dict(self.environment),
            "dataset": dict(self.dataset),
            "training": dict(self.training),
            "runs": [dict(run) for run in self.runs],
            "aggregates": self.aggregates,
            "criteria": [outcome.to_dict() for outcome in self.criteria],
            "notes": list(self.notes),
            "data_label": "SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED",
        }

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


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
    """Train and evaluate one arm at one seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_manifest(corpus_dir)
    train = load_split(corpus_dir, "train", limit=train_limit)
    evaluation = load_split(corpus_dir, "test", limit=eval_limit)
    config = TrainingConfig(
        arm=cast(ArmName, arm),
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        scene_points=scene_points,
        entity_points=entity_points,
        device=device,
        eval_every=max(1, steps // 2),
    )
    trainer = Trainer(
        config,
        ontology,
        domain,
        train,
        evaluation,
        dataset_info={
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes_total": manifest.scenes,
            "train_used": len(train),
            "test_used": len(evaluation),
            "ontology": f"{manifest.ontology_id}@{manifest.ontology_version}",
            "data_label": manifest.data_label,
        },
    )
    run = trainer.fit(checkpoint_dir=checkpoint_dir)
    return run.to_dict()


def _aggregate(
    runs: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean, standard deviation, range and per-seed values, per arm and metric."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["arm"]), []).append(run)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, arm_runs in grouped.items():
        out[arm] = {}
        for metric in metrics:
            values = [
                float(run["results"][metric])
                for run in arm_runs
                if metric in run["results"]
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


def evaluate_criteria(
    aggregates: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    structured: str = "A3",
    baseline: str = "A0",
) -> list[CriterionOutcome]:
    """Evaluate the Step 5 pre-registered criteria, unchanged."""
    outcomes: list[CriterionOutcome] = []

    def entry(arm: str, metric: str) -> Mapping[str, float] | None:
        return aggregates.get(arm, {}).get(metric)

    control_structured = entry(structured, PRIMARY_METRIC)
    control_baseline = entry(baseline, PRIMARY_METRIC)
    if control_structured and control_baseline:
        gap = (control_structured["mean"] - control_baseline["mean"]) * 100.0
        separated = control_structured["min"] > control_baseline["max"]
        verdict = "PASS" if gap >= 20.0 and separated else "FAIL"
        outcomes.append(
            CriterionOutcome(
                hypothesis="H1a",
                metric=PRIMARY_METRIC,
                threshold=(
                    "structured exceeds baseline by >= 20 points, non-overlapping seed ranges"
                ),
                verdict=verdict,
                detail=(
                    f"gap {gap:+.1f} points; structured range "
                    f"[{control_structured['min']:.3f}, {control_structured['max']:.3f}] "
                    "vs baseline "
                    f"[{control_baseline['min']:.3f}, {control_baseline['max']:.3f}]; "
                    f"ranges {'do not overlap' if separated else 'overlap'}"
                ),
                values={
                    "structured_mean": control_structured["mean"],
                    "baseline_mean": control_baseline["mean"],
                    "gap_points": gap,
                },
            )
        )
    else:
        outcomes.append(
            CriterionOutcome("H1a", PRIMARY_METRIC, ">= 20 points", "NOT MEASURED", "metric absent")
        )

    relationship_structured = entry(structured, RELATIONSHIP_METRIC)
    relationship_baseline = entry(baseline, RELATIONSHIP_METRIC)
    if relationship_structured and relationship_baseline:
        gap = (relationship_structured["mean"] - relationship_baseline["mean"]) * 100.0
        outcomes.append(
            CriterionOutcome(
                hypothesis="H1b",
                metric=RELATIONSHIP_METRIC,
                threshold="structured exceeds baseline by >= 15 points on held-out relations",
                verdict="PASS" if gap >= 15.0 else "FAIL",
                detail=f"gap {gap:+.1f} points (never trained on)",
                values={
                    "structured_mean": relationship_structured["mean"],
                    "baseline_mean": relationship_baseline["mean"],
                    "gap_points": gap,
                },
            )
        )
    else:
        outcomes.append(
            CriterionOutcome(
                "H1b", RELATIONSHIP_METRIC, ">= 15 points", "NOT MEASURED", "metric absent"
            )
        )

    drift_structured = entry(structured, DRIFT_METRIC)
    drift_baseline = entry(baseline, DRIFT_METRIC)
    if drift_structured and drift_baseline:
        near_zero = drift_structured["max"] <= 1.0e-6
        baseline_drifts = drift_baseline["mean"] > 1.0e-4
        outcomes.append(
            CriterionOutcome(
                hypothesis="H1c",
                metric=DRIFT_METRIC,
                threshold=(
                    "structured drift is essentially zero where the baseline drifts measurably"
                ),
                verdict="PASS" if near_zero and baseline_drifts else "FAIL",
                detail=(
                    f"structured max {drift_structured['max']:.3e}, baseline mean "
                    f"{drift_baseline['mean']:.3e}"
                ),
                values={
                    "structured_max": drift_structured["max"],
                    "baseline_mean": drift_baseline["mean"],
                },
            )
        )
    else:
        outcomes.append(
            CriterionOutcome("H1c", DRIFT_METRIC, "near zero", "NOT MEASURED", "metric absent")
        )

    geometry_structured = entry(structured, GEOMETRY_METRIC)
    geometry_baseline = entry(baseline, GEOMETRY_METRIC)
    if geometry_structured and geometry_baseline and geometry_baseline["mean"] > 0:
        regression = (
            (geometry_structured["mean"] - geometry_baseline["mean"]) / geometry_baseline["mean"]
        ) * 100.0
        outcomes.append(
            CriterionOutcome(
                hypothesis="guard",
                metric=GEOMETRY_METRIC,
                threshold="structured Chamfer no more than 20 percent worse than baseline",
                verdict="PASS" if regression <= 20.0 else "FAIL",
                detail=(
                    f"regression {regression:+.1f} percent; lower is better and negative is better"
                ),
                values={
                    "structured_mean": geometry_structured["mean"],
                    "baseline_mean": geometry_baseline["mean"],
                    "regression_percent": regression,
                },
            )
        )
    else:
        outcomes.append(
            CriterionOutcome(
                "guard",
                GEOMETRY_METRIC,
                "<= 20 percent regression",
                "NOT MEASURED",
                "metric absent",
            )
        )

    outcomes.append(
        CriterionOutcome(
            hypothesis="H1d",
            metric="sample_efficiency",
            threshold="advantage grows as the training set shrinks",
            verdict="NOT MEASURED",
            detail="requires runs at several corpus sizes; out of scope for this Step 6 budget",
        )
    )
    return outcomes


REPORTED_METRICS: tuple[str, ...] = (
    PRIMARY_METRIC,
    "part_control_success_addressable",
    "part_iou_mean",
    RELATIONSHIP_METRIC,
    "relationship_accuracy",
    "relationship_coverage",
    "relationship_checked",
    "relationship_possible",
    DRIFT_METRIC,
    "touched_change",
    "geometry_tokens_identical",
    GEOMETRY_METRIC,
    "scene_iou",
    "occupancy_accuracy",
    "lod_part_agreement_mean",
    "lod_centroid_shift_mean",
    "lod_iou_monotone",
    "lod_iou_coarsest",
    "lod_iou_finest",
)


def run_experiment(
    *,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    ablations: Sequence[str],
    batch_size: int,
    scene_points: int,
    entity_points: int,
    device: str,
    train_limit: int | None,
    eval_limit: int | None,
    output_dir: Path,
) -> ExperimentReport:
    """Run every arm at every seed, then evaluate the pre-registered criteria."""
    started = time.time()
    manifest = load_manifest(corpus_dir)
    runs: list[Mapping[str, Any]] = []
    plan: list[tuple[str, int]] = []
    for seed in seeds:
        plan.append(("A3", seed))
        plan.append(("A0", seed))
    for arm in ablations:
        plan.append((arm, seeds[0]))

    for arm, seed in plan:
        print(f"[experiment] running {arm} seed {seed} ...", flush=True)
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
            if key in REPORTED_METRICS
        }
        print(f"[experiment]   {arm} seed {seed}: {summary}", flush=True)
        runs.append(run)

    aggregates = _aggregate(runs, REPORTED_METRICS)
    report = ExperimentReport(
        experiment_id="heart-001-structured-vs-appearance",
        commit=git_commit(repo_root()),
        environment=environment_report(),
        dataset={
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "scenes": manifest.scenes,
            "families": manifest.families,
            "split_counts": dict(manifest.split_counts),
            "lod_distribution": {str(k): v for k, v in manifest.lod_distribution.items()},
            "data_label": manifest.data_label,
        },
        training={
            "steps": steps,
            "batch_size": batch_size,
            "scene_points": scene_points,
            "entity_points": entity_points,
            "device": device,
            "seeds": list(seeds),
            "ablations": list(ablations),
            "runtime_seconds": round(time.time() - started, 1),
            "parameter_counts": {
                str(run["arm"]): int(run["parameters"]["total"]) for run in runs
            },
        },
        runs=runs,
        aggregates=aggregates,
        criteria=evaluate_criteria(aggregates),
        notes=(
            "Tier-0 synthetic corpus. Not anatomy, not validated, not clinical.",
            "Relationship metrics were never trained on.",
            "The structured arm additionally trains entity_identity and entity_frame, which "
            "have no counterpart in the baseline; recorded as a threat to validity.",
            "n=3 seeds; ranges are reported per seed rather than as confidence intervals.",
        ),
    )
    report.save(output_dir / "experiment_report.json")
    return report


def smoke_run(
    output_dir: Path | None = None, *, scenes: int = 48, steps: int = 4
) -> dict[str, Any]:
    """Tiny end-to-end run: data, AWR, tensors, model, loss, backward, checkpoint, eval.

    Deliberately small enough for a test. It proves the pipeline executes, not that it
    learns anything.
    """
    target = output_dir or Path("experiments/runs/smoke")
    corpus = target / "corpus"
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = generate_corpus(
        ontology, corpus, scenes=scenes, families=12, lod_choices=(1, 2), seed=0,
        corpus_id="tier0-smoke",
    )
    train = load_split(corpus, "train")
    evaluation = load_split(corpus, "test") or load_split(corpus, "validation")
    outcomes: dict[str, Any] = {"corpus": manifest.to_dict(), "arms": {}}
    arms: tuple[ArmName, ...] = ("A3", "A0")
    for arm in arms:
        config = TrainingConfig(
            arm=arm,
            seed=0,
            steps=steps,
            batch_size=4,
            scene_points=96,
            entity_points=12,
            eval_every=0,
            log_every=1,
            eval_batches=1,
            device="cpu",
        )
        trainer = Trainer(config, ontology, domain, train, evaluation)
        run = trainer.fit(checkpoint_dir=target / "checkpoints")
        checkpoint = target / "checkpoints" / f"{run.run_id}.pt"
        trainer.load_checkpoint(checkpoint)
        outcomes["arms"][arm] = {
            "parameters": run.parameters["total"],
            "steps": run.steps_completed,
            "final_loss": run.train_history[-1]["total"],
            "results": {
                key: round(float(value), 4)
                for key, value in run.results.items()
                if key in REPORTED_METRICS
            },
            "checkpoint_exists": checkpoint.is_file(),
            "runtime_seconds": run.runtime_seconds,
        }
    (target / "smoke_report.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run heart experiment 001.")
    parser.add_argument("--smoke", action="store_true", help="run the tiny end-to-end smoke test")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ablations", nargs="*", default=["A1", "A4", "A5"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scene-points", type=int, default=256)
    parser.add_argument("--entity-points", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/heart-001"))
    args = parser.parse_args(argv)

    if args.smoke:
        print(json.dumps(smoke_run(args.out / "smoke"), indent=2)[:4000])
        return 0
    if args.corpus is None:
        parser.error("--corpus is required unless --smoke is given")
    report = run_experiment(
        corpus_dir=args.corpus,
        steps=args.steps,
        seeds=args.seeds,
        ablations=args.ablations,
        batch_size=args.batch_size,
        scene_points=args.scene_points,
        entity_points=args.entity_points,
        device=args.device,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        output_dir=args.out,
    )
    print(json.dumps({"criteria": [c.to_dict() for c in report.criteria]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
