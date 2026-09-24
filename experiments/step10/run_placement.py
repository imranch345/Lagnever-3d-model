"""Step 10 Change 1: parent-relative placement against the global control.

Each cell differs from its control in exactly one thing — what the frame head is asked to
predict — and in nothing else. The model composes local frames into scene coordinates
before anything downstream sees them, so the loss, the metrics and the evaluation are Step
9's unchanged, and the numbers are comparable with Step 9 rather than merely similar.

The pre-registered cells
------------------------

======================  =====================================================
``T0_global``           Step 9's target, reproduced. The control.
``T1_spatial``          Parent-relative over the generator's construction tree.
``T2_taxonomic``        Parent-relative over AWR's own hierarchy. The taxonomic control.
======================  =====================================================

``T2_taxonomic`` is expected to be **bit-identical** to ``T0_global``, because AWR nests
every whole-organ entity under a geometry-free category node and so leaves all twenty as
roots. That is not a wasted cell: it measures what "just impose AWR's hierarchy" would have
delivered, and separates having *a* hierarchy from having the right one. The identity is
checked per arm and seed rather than assumed.

``T3_alternative`` and ``T4_rigid`` stay defined for later work but are not part of the
pre-registered matrix: running them now would change the parent convention mid-experiment.

What every run records
----------------------

Written to ``<out>/runs/<arm>__<cell>__seed<n>.json`` the moment the run finishes, so a
sweep that dies loses one run rather than all of them, and a restarted sweep resumes
instead of retraining:

* global-frame placement on every split, inferred and supplied, against the floor;
* the same error resolved by depth in the spatial tree, reconciled against the headline;
* the §27 integrity gate's report, on the training data and on every evaluated split;
* parameter count, training steps, torch threads, and train, evaluation and depth time.

The floor
---------

Every arm is scored on global frames with the Step 9 metric, so every arm is read against
the same placement-blind floor: the best identity-only predictor, 0.1605 on ``test_seen``.

    python -m experiments.step10.run_placement --seeds 0 1 2 --out experiments/runs/step10-placement
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

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
from experiments.step10.depth_analysis import depth_analysis
from experiments.step10.placement_floor import placement_floor
from training.manifest import environment_report, git_commit
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = [
    "ARMS",
    "CELLS",
    "PREREGISTERED",
    "assemble",
    "main",
    "paired_comparison",
    "run_placement",
]

ARMS: tuple[str, ...] = ("A1", "A1M", "A3Lite", "A3L", "A3")

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
        "taxonomic control: AWR's own hierarchy imposed; expected identical to T0",
    ),
    "T3_alternative": (
        "parent_relative", "spatial_alternative", "isotropic",
        "not pre-registered: AV annuli parented to the atrium instead",
    ),
    "T4_rigid": (
        "parent_relative", "spatial", "rigid",
        "not pre-registered: a parent's orientation propagates but not its size",
    ),
}

PREREGISTERED: tuple[str, ...] = ("T0_global", "T1_spatial", "T2_taxonomic")

#: Two-sided 5% critical value of Student's t with 2 degrees of freedom (three seeds).
#: Fixed before any confirmatory result existed; see the README's decision rule.
T_CRITICAL_DF2 = 4.303

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

STEP9_BASELINE = Path("experiments/runs/step9-baseline/baseline_report.json")


def _protocol(
    *, steps: int, batch_size: int, device: str, splits: Sequence[str], corpus_id: str
) -> dict[str, Any]:
    """What must match for a saved run to be reused rather than retrained."""
    return {
        "steps": steps,
        "batch_size": batch_size,
        "device": device,
        "splits": list(splits),
        "corpus_id": corpus_id,
        "torch_threads": torch.get_num_threads(),
    }


def _free_gib(path: Path) -> float:
    probe = path if path.exists() else path.parent
    return shutil.disk_usage(probe).free / 2**30


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write-then-rename, so a file is either complete or absent — never truncated.

    The temporary name carries the process id: parallel workers assemble the same report,
    and a shared temporary file would let one worker rename the other's half-written one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


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
    min_free_gib: float = 2.0,
) -> list[Path]:
    """Train each cell of each arm at each seed; write one run file per finished run."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    protocol = _protocol(
        steps=steps,
        batch_size=batch_size,
        device=device,
        splits=splits,
        corpus_id=manifest.corpus_id,
    )
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")
    evaluated = {split: load_step8_split(corpus_dir, split) for split in splits}
    floors = placement_floor(corpus_dir, splits=splits)
    runs_dir = output_dir / "runs"
    written: list[Path] = []

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
                label = f"{arm} {name} seed {seed}"
                path = runs_dir / f"{arm}__{name}__seed{seed}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved.get("protocol") == protocol:
                        print(f"[step10] {label}: already complete, reusing", flush=True)
                        written.append(path)
                        continue
                    raise SystemExit(
                        f"{path} was produced under a different protocol "
                        f"({saved.get('protocol')} != {protocol}); move it aside rather than "
                        "mixing protocols in one report"
                    )
                free = _free_gib(output_dir)
                if free < min_free_gib:
                    raise SystemExit(
                        f"only {free:.2f} GiB free, below the {min_free_gib} GiB floor; "
                        f"stopping before {label} rather than risk a truncated write"
                    )

                print(f"[step10] {label} ...", flush=True)
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

                started = time.time()
                record_splits: dict[str, Any] = {}
                split_integrity: dict[str, Any] = {}
                for split in splits:
                    split_integrity[split] = trainer.validate(evaluated[split])
                    conditions: dict[str, dict[str, float]] = {}
                    for condition, predicted in (("inferred", True), ("supplied", False)):
                        values = evaluate_step8(
                            trainer.model,
                            evaluated[split],
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
                    inferred["placement_floor_margin"] = floor - float(
                        inferred["translation_error"]
                    )
                    inferred["placement_floor_gap"] = floor_gap(
                        float(inferred["translation_error"]), floor
                    )
                    inferred["oracle_gap_entity_iou"] = float(
                        conditions["supplied"]["entity_iou_mean"]
                    ) - float(inferred["entity_iou_mean"])
                    record_splits[split] = conditions
                eval_seconds = time.time() - started

                started = time.time()
                depth = depth_analysis(
                    trainer.model,
                    corpus_dir,
                    trainer.builder,
                    splits=splits,
                    device=trainer.device,
                    batch_size=batch_size,
                    hierarchy="spatial",
                )
                depth_seconds = time.time() - started
                reconciliation = {
                    split: abs(
                        float(depth["splits"][split]["headline_translation_error"])
                        - float(record_splits[split]["inferred"]["translation_error"])
                    )
                    for split in splits
                }
                if max(reconciliation.values()) > 1e-6:
                    raise SystemExit(
                        f"{label}: the depth buckets do not reconcile with the headline "
                        f"({reconciliation}); they are not scoring the same frames"
                    )

                record: dict[str, Any] = {
                    "run_id": run.run_id,
                    "arm": arm,
                    "cell": name,
                    "seed": seed,
                    "question": question,
                    "placement_target": target,
                    "hierarchy": hierarchy,
                    "parent_convention": convention,
                    "parented_entities": sum(
                        1 for slot in trainer.config.placement_parents if slot >= 0
                    ),
                    "protocol": protocol,
                    "parameters": dict(trainer.parameter_groups),
                    "timing": {
                        "train_seconds": run.runtime_seconds,
                        "eval_seconds": round(eval_seconds, 2),
                        "depth_seconds": round(depth_seconds, 2),
                    },
                    "integrity": {"train": trainer.integrity, "splits": split_integrity},
                    "splits": record_splits,
                    "depth": depth,
                    "depth_reconciliation": reconciliation,
                    "manifest": run.to_dict(),
                }
                _write_atomic(path, record)
                written.append(path)
                head = record_splits[splits[0]]["inferred"]
                print(
                    f"[step10]   {label} {splits[0]}: "
                    f"translation={head['translation_error']:.4f} "
                    f"floor={head['placement_floor']:.4f} "
                    f"margin={head['placement_floor_margin']:+.4f} "
                    f"iou={head['entity_iou_mean']:.4f}",
                    flush=True,
                )
    return written


def assemble(output_dir: Path, *, corpus_dir: Path) -> dict[str, Any]:
    """Build ``placement_report.json`` from every saved run file in ``output_dir``."""
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*.json"))
    ]
    if not runs:
        raise SystemExit(f"no run files under {output_dir / 'runs'}")
    protocols = {json.dumps(run["protocol"], sort_keys=True) for run in runs}
    if len(protocols) != 1:
        raise SystemExit(f"run files were produced under {len(protocols)} different protocols")
    protocol = runs[0]["protocol"]
    splits = list(protocol["splits"])
    floors = placement_floor(corpus_dir, splits=splits)
    seeds = sorted({int(run["seed"]) for run in runs})
    manifest = load_step8_manifest(corpus_dir)

    report = {
        "experiment_id": "step10-placement",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "status": "confirmatory" if len(seeds) >= 3 else "exploratory",
        "arms": [arm for arm in ARMS if any(run["arm"] == arm for run in runs)],
        "cells": {
            name: CELLS[name][3] for name in CELLS if any(run["cell"] == name for run in runs)
        },
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "splits": splits,
            "data_label": manifest.data_label,
        },
        "protocol": protocol,
        "seeds": seeds,
        "decision_rule": {
            "exceeds_seed_variability": (
                "all paired seed differences share a sign AND |t| > "
                f"{T_CRITICAL_DF2} (two-sided 5%, df=2)"
            ),
            "fixed_before_results": True,
        },
        "placement_floor": floors,
        "runs": [{key: value for key, value in run.items() if key != "manifest"} for run in runs],
        "aggregates": _aggregate(runs, splits),
        "depth_aggregates": _depth_aggregate(runs, splits),
        "paired": {
            split: {
                f"{arm}/{treatment}-vs-T0_global": paired_comparison(runs, arm, treatment, split)
                for arm in ARMS
                for treatment in ("T1_spatial", "T2_taxonomic")
                if any(run["arm"] == arm and run["cell"] == treatment for run in runs)
            }
            for split in splits
        },
        "control_identity": _control_identity(runs, splits),
        "step9_reproduction": _step9_reproduction(runs),
        "notes": [
            "Headline condition is inferred placement.",
            "Every cell is scored on GLOBAL frames with the Step 9 metric, unchanged.",
            "The bar is the identity-only floor, which does not move with the target.",
            "placement_floor_margin = floor - translation_error; positive beats the floor.",
            "Depth is read from the spatial tree for every cell, so buckets are comparable.",
            "T2_taxonomic must match T0_global; a difference means the control is not one.",
            "rotation_error is structurally zero: the corpus rotation target is constant.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    _write_atomic(output_dir / "placement_report.json", report)
    return report


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": float(len(values)),
        "min": min(values),
        "max": max(values),
    }


def _aggregate(runs: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> dict[str, Any]:
    """Mean, sample standard deviation and range over seeds, per arm, cell and split."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["arm"]), str(run["cell"])), []).append(run)
    out: dict[str, Any] = {}
    for (arm, cell), cell_runs in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "seeds": sorted(int(run["seed"]) for run in cell_runs),
            "placement_target": cell_runs[0]["placement_target"],
            "hierarchy": cell_runs[0]["hierarchy"],
            "parameters": int(cell_runs[0]["parameters"]["total"]),
            "training_steps": int(cell_runs[0]["protocol"]["steps"]),
            "train_seconds": _stats([float(run["timing"]["train_seconds"]) for run in cell_runs]),
        }
        for split in splits:
            entry[split] = {
                metric: _stats(
                    [float(run["splits"][split]["inferred"][metric]) for run in cell_runs]
                )
                for metric in (*REPORTED, "placement_floor_margin")
                if all(metric in run["splits"][split]["inferred"] for run in cell_runs)
            }
        out[f"{arm}/{cell}"] = entry
    return out


def _depth_aggregate(
    runs: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, Any]:
    """Per-depth errors over seeds, per arm, cell and split."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["arm"]), str(run["cell"])), []).append(run)
    out: dict[str, Any] = {}
    for (arm, cell), cell_runs in sorted(grouped.items()):
        entry: dict[str, Any] = {}
        for split in splits:
            depths = sorted(
                {d for run in cell_runs for d in run["depth"]["splits"][split]["depths"]},
                key=int,
            )
            entry[split] = {
                depth: {
                    "entities_scored": int(
                        cell_runs[0]["depth"]["splits"][split]["depths"][depth]["entities_scored"]
                    ),
                    **{
                        metric: _stats(
                            [
                                float(run["depth"]["splits"][split]["depths"][depth][metric])
                                for run in cell_runs
                            ]
                        )
                        for metric in ("position_error", "scale_error", "rotation_error")
                    },
                }
                for depth in depths
            }
        out[f"{arm}/{cell}"] = entry
    return out


def paired_comparison(
    runs: Sequence[Mapping[str, Any]], arm: str, treatment: str, split: str
) -> dict[str, Any]:
    """Treatment minus control, paired by seed, overall and per depth.

    Negative means the treatment placed better. Paired by seed because the seed fixes the
    initialisation and the data order, so the paired difference removes the variation the
    two cells share and leaves the variation the target change causes.
    """
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        if run["arm"] == arm and run["cell"] in (treatment, "T0_global"):
            by_seed.setdefault(int(run["seed"]), {})[str(run["cell"])] = run
    seeds = sorted(seed for seed, cells in by_seed.items() if len(cells) == 2)

    def summary(differences: list[float]) -> dict[str, Any]:
        n = len(differences)
        mean = statistics.fmean(differences) if n else float("nan")
        std = statistics.stdev(differences) if n > 1 else float("nan")
        t_stat = mean / (std / n**0.5) if n > 1 and std > 0 else float("nan")
        same_sign = n > 0 and (all(d < 0 for d in differences) or all(d > 0 for d in differences))
        return {
            "per_seed": dict(zip((str(seed) for seed in seeds), differences, strict=True)),
            "mean_difference": mean,
            "std_difference": std,
            "t_statistic": t_stat,
            "all_seeds_same_sign": same_sign,
            "exceeds_seed_variability": bool(
                n >= 3 and same_sign and abs(t_stat) > T_CRITICAL_DF2
            ),
        }

    overall = [
        float(by_seed[seed][treatment]["splits"][split]["inferred"]["translation_error"])
        - float(by_seed[seed]["T0_global"]["splits"][split]["inferred"]["translation_error"])
        for seed in seeds
    ]
    result: dict[str, Any] = {"seeds": seeds, "translation_error": summary(overall)}
    depth_keys = sorted(
        {
            depth
            for seed in seeds
            for depth in by_seed[seed]["T0_global"]["depth"]["splits"][split]["depths"]
        },
        key=int,
    )
    result["by_depth"] = {
        depth: summary(
            [
                float(
                    by_seed[seed][treatment]["depth"]["splits"][split]["depths"][depth][
                        "position_error"
                    ]
                )
                - float(
                    by_seed[seed]["T0_global"]["depth"]["splits"][split]["depths"][depth][
                        "position_error"
                    ]
                )
                for seed in seeds
            ]
        )
        for depth in depth_keys
    }
    return result


def _control_identity(
    runs: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, Any]:
    """Whether the taxonomic cell reproduced the global one exactly, per arm and seed.

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
        differences = {
            split: abs(
                float(run["splits"][split]["inferred"]["translation_error"])
                - float(match["splits"][split]["inferred"]["translation_error"])
            )
            for split in splits
        }
        out[f"{run['arm']}/seed{run['seed']}"] = {
            "max_abs_difference": max(differences.values()),
            "identical": all(value == 0.0 for value in differences.values()),
        }
    return out


def _step9_reproduction(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether each T0_global run reproduces the Step 9 baseline for its arm and seed."""
    if not STEP9_BASELINE.exists():
        return {"available": False}
    baseline = json.loads(STEP9_BASELINE.read_text(encoding="utf-8"))
    reference = {
        (str(run["arm"]), int(run["seed"])): float(
            run["splits"]["test_seen"]["inferred"]["translation_error"]
        )
        for run in baseline["runs"]
    }
    out: dict[str, Any] = {"available": True}
    for run in runs:
        if run["cell"] != "T0_global":
            continue
        key = (str(run["arm"]), int(run["seed"]))
        if key not in reference:
            continue
        ours = float(run["splits"]["test_seen"]["inferred"]["translation_error"])
        out[f"{key[0]}/seed{key[1]}"] = {
            "step10": ours,
            "step9": reference[key],
            "abs_difference": abs(ours - reference[key]),
            "identical": abs(ours - reference[key]) < 1e-12,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 parent-relative placement.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step8_continuous"))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--cells", nargs="+", default=list(PREREGISTERED))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step10-placement"))
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="build the report from saved run files without training anything",
    )
    args = parser.parse_args(argv)

    unknown = [cell for cell in args.cells if cell not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; expected some of {list(CELLS)}")
    if not args.assemble_only:
        run_placement(
            corpus_dir=args.corpus,
            arms=args.arms,
            cells=args.cells,
            seeds=args.seeds,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            splits=args.splits,
            output_dir=args.out,
            min_free_gib=args.min_free_gib,
        )
    report = assemble(args.out, corpus_dir=args.corpus)
    print(json.dumps(report["control_identity"], indent=2))
    print(json.dumps(report["step9_reproduction"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
