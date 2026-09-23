"""Step 10 Change 2: whether the model learns position, rotation and scale together.

Change 1 ran on a corpus whose every rotation was the identity, so a quarter of the frame
objective was spent on a constant and no rotation result was obtainable. This runs the same
five arms on the rotated corpus, with the rotation term replaced by a chordal distance on
rotation matrices, and scores position, rotation and scale **separately** against this
corpus's own floors.

The cells
---------

======================  =====================================================
``T0_global``           the primary cell: global frames, the Change 2 question
``T4_rigid``            pre-registered control: a parent passes orientation but not size
``T1_spatial``          secondary: Change 1's parent-relative arm, replicated here
======================  =====================================================

``T1_spatial`` is **a replication under a different corpus, not a fresh hypothesis**. Change 1
established that recursive parent-relative placement worsened global placement in all five
arms on the identity-rotation corpus; running it again on rotated data says whether that
survives once a parent's orientation carries information, and it is labelled as such
everywhere it is reported.

``T4_rigid`` earns its place for the first time here. Under ``rigid`` a parent contributes its
rotation but not its scale, so it cannot propagate a scale error into its descendants while
still orienting them. On the Change 1 corpus that was nearly a no-op, because a parent had no
orientation to give.

What is frozen before any run
-----------------------------

* the corpus, and its checksums;
* the transform convention: parent rotation plus isotropic scale, child rotation plus full
  anisotropic scale;
* the rotation objective (chordal) and its weight, from the train-only calibration;
* the decision rule: a difference exceeds seed variability only if all three seeds agree in
  sign and |t| > 4.303;
* the floors, which are this corpus's, per split, per component.

    python -m experiments.step10.run_change2 --seeds 0
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
from experiments.step10.rotation_metrics import rotation_report
from training.manifest import environment_report, git_commit
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = [
    "ARMS",
    "CELLS",
    "ROTATION_LOSS_WEIGHT",
    "SCREENED_ROTATION_WEIGHT",
    "assemble",
    "main",
    "run_change2",
]

ARMS: tuple[str, ...] = ("A1", "A1M", "A3Lite", "A3L", "A3")

#: Change 2's weight, from the train-only calibration in ``calibrate_rotation_weight.py``.
#: The default, so the confirmatory runs already on record reproduce exactly.
ROTATION_LOSS_WEIGHT = 0.33

#: Chosen by the validation-only screening in ``weight_study.py`` after Change 2 found the
#: rotation term was the binding constraint at 0.33. Pass it explicitly; it writes to its own
#: output directory, because a run at a different weight is a different experiment.
SCREENED_ROTATION_WEIGHT = 3.0
ROTATION_OBJECTIVE = "chordal"

#: name -> (placement target, hierarchy, parent convention, role, what it tests)
CELLS: Mapping[str, tuple[str, str, str, str, str]] = {
    "T0_global": (
        "global", "spatial", "isotropic", "primary",
        "global frames with a real rotation target: the Change 2 question",
    ),
    "T4_rigid": (
        "parent_relative", "spatial", "rigid", "pre-registered control",
        "a parent passes its orientation but not its size",
    ),
    "T1_spatial": (
        "parent_relative", "spatial", "isotropic", "secondary replication",
        "Change 1's parent-relative arm, replicated on rotated data; not a fresh hypothesis",
    ),
}

#: Two-sided 5% critical value of Student's t with 2 degrees of freedom. Change 1's rule.
T_CRITICAL_DF2 = 4.303

REPORTED: tuple[str, ...] = (
    "translation_error",
    "scale_error",
    "rotation_error",
    "composite_frame_error",
    "placement_error",
    "entity_iou_mean",
    "part_control_success",
    "spatial_relation_accuracy",
    "scene_iou",
)

CORPUS = Path("datasets/processed/step10_rotated")
CHECKSUMS = Path("experiments/runs/step10-change2/CORPUS_FROZEN.sha256")


def _protocol(
    *,
    steps: int,
    batch_size: int,
    device: str,
    splits: Sequence[str],
    corpus_id: str,
    rotation_weight: float,
    relation_values: bool,
) -> dict[str, Any]:
    """What must match for a saved run to be reused rather than retrained."""
    return {
        "steps": steps,
        "batch_size": batch_size,
        "device": device,
        "splits": list(splits),
        "corpus_id": corpus_id,
        "torch_threads": torch.get_num_threads(),
        "rotation_objective": ROTATION_OBJECTIVE,
        "rotation_loss_weight": rotation_weight,
        "relation_values": relation_values,
    }


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write-then-rename, with a per-process temporary name so parallel workers cannot race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _corpus_unchanged() -> dict[str, Any]:
    """The corpus must be the one that was validated, byte for byte (§25)."""
    if not CHECKSUMS.exists():
        raise SystemExit(f"{CHECKSUMS} is missing; the corpus is not frozen")
    import hashlib

    checked = 0
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", 1)
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise SystemExit(f"the corpus changed: {path} no longer matches its checksum")
        checked += 1
    return {"files_verified": checked, "manifest": str(CHECKSUMS)}


def run_change2(
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
    rotation_weight: float = ROTATION_LOSS_WEIGHT,
    relation_values: bool = False,
) -> list[Path]:
    """Train each cell of each arm at each seed; write one run file per finished run."""
    frozen = _corpus_unchanged()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    if not manifest.rotation_enabled:
        raise SystemExit(f"{corpus_dir} carries no rotations; this is not the Change 2 corpus")
    protocol = _protocol(
        steps=steps,
        batch_size=batch_size,
        device=device,
        splits=splits,
        corpus_id=manifest.corpus_id,
        rotation_weight=rotation_weight,
        relation_values=relation_values,
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
                target, hierarchy, convention, role, question = CELLS[name]
                config = with_variant(
                    default_config(arm),
                    placement_target=target,  # type: ignore[arg-type]
                    hierarchy=hierarchy,
                    seed=seed,
                )
                config = replace(
                    config,
                    parent_convention=convention,
                    rotation_objective=ROTATION_OBJECTIVE,  # type: ignore[arg-type]
                    rotation_loss_weight=rotation_weight,
                    relation_values=relation_values,
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
                        print(f"[change2] {label}: already complete, reusing", flush=True)
                        written.append(path)
                        continue
                    raise SystemExit(
                        f"{path} was produced under a different protocol; move it aside "
                        "rather than mixing protocols in one report"
                    )
                free = shutil.disk_usage(
                    output_dir if output_dir.exists() else output_dir.parent
                ).free / 2**30
                if free < min_free_gib:
                    raise SystemExit(
                        f"only {free:.2f} GiB free, below the {min_free_gib} GiB floor; "
                        f"stopping before {label}"
                    )

                print(f"[change2] {label} ...", flush=True)
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
                if trainer.config.rotation_loss_weight != rotation_weight:
                    raise SystemExit(
                        "the run's rotation weight is not the one this experiment declared"
                    )
                run = trainer.fit(checkpoint_dir=output_dir / "checkpoints")

                started = time.time()
                record_splits: dict[str, Any] = {}
                split_integrity: dict[str, Any] = {}
                rotation: dict[str, Any] = {}
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
                    entry = floors["splits"][split]["global"]
                    inferred = conditions["inferred"]
                    inferred["placement_floor"] = float(entry["position_error"])
                    inferred["placement_floor_margin"] = float(entry["position_error"]) - float(
                        inferred["translation_error"]
                    )
                    inferred["placement_floor_gap"] = floor_gap(
                        float(inferred["translation_error"]), float(entry["position_error"])
                    )
                    inferred["scale_floor"] = float(entry["scale_error"])
                    inferred["scale_floor_margin"] = float(entry["scale_error"]) - float(
                        inferred["scale_error"]
                    )
                    record_splits[split] = conditions
                    rotation[split] = rotation_report(
                        trainer.model,
                        evaluated[split],
                        trainer.builder,
                        device=trainer.device,
                        batch_size=batch_size,
                        floor_radians=float(entry["rotation_error"]),
                    )
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
                    raise SystemExit(f"{label}: depth buckets do not reconcile {reconciliation}")

                record: dict[str, Any] = {
                    "run_id": run.run_id,
                    "arm": arm,
                    "cell": name,
                    "role": role,
                    "seed": seed,
                    "question": question,
                    "placement_target": target,
                    "hierarchy": hierarchy,
                    "parent_convention": convention,
                    "rotation_objective": ROTATION_OBJECTIVE,
                    "rotation_loss_weight": rotation_weight,
                    "relation_values": relation_values,
                    "parented_entities": sum(
                        1 for slot in trainer.config.placement_parents if slot >= 0
                    ),
                    "protocol": protocol,
                    "corpus_frozen": frozen,
                    "parameters": dict(trainer.parameter_groups),
                    "timing": {
                        "train_seconds": run.runtime_seconds,
                        "eval_seconds": round(eval_seconds, 2),
                        "depth_seconds": round(depth_seconds, 2),
                    },
                    "integrity": {"train": trainer.integrity, "splits": split_integrity},
                    "splits": record_splits,
                    "rotation": rotation,
                    "depth": depth,
                    "depth_reconciliation": reconciliation,
                    "manifest": run.to_dict(),
                }
                _write_atomic(path, record)
                written.append(path)
                head = record_splits[splits[0]]["inferred"]
                turn = rotation[splits[0]]
                print(
                    f"[change2]   {label} {splits[0]}: "
                    f"position={head['translation_error']:.4f} "
                    f"(floor {head['placement_floor']:.4f}) "
                    f"rotation={turn['mean_deg']:.2f}deg "
                    f"(floor {turn['floor_deg']:.2f}) "
                    f"scale={head['scale_error']:.4f}",
                    flush=True,
                )
    return written


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": float(len(values)),
        "min": min(values),
        "max": max(values),
    }


def paired_comparison(
    runs: Sequence[Mapping[str, Any]], arm: str, treatment: str, split: str, metric: str
) -> dict[str, Any]:
    """Treatment minus control, paired by seed, for one metric. Negative means better."""
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        if run["arm"] == arm and run["cell"] in (treatment, "T0_global"):
            by_seed.setdefault(int(run["seed"]), {})[str(run["cell"])] = run
    seeds = sorted(seed for seed, cells in by_seed.items() if len(cells) == 2)

    def read(run: Mapping[str, Any]) -> float:
        if metric == "rotation_deg":
            return float(run["rotation"][split]["mean_deg"])
        return float(run["splits"][split]["inferred"][metric])

    differences = [
        read(by_seed[seed][treatment]) - read(by_seed[seed]["T0_global"]) for seed in seeds
    ]
    count = len(differences)
    mean = statistics.fmean(differences) if count else float("nan")
    std = statistics.stdev(differences) if count > 1 else float("nan")
    t_stat = mean / (std / count**0.5) if count > 1 and std > 0 else float("nan")
    same_sign = count > 0 and (
        all(value < 0 for value in differences) or all(value > 0 for value in differences)
    )
    return {
        "metric": metric,
        "per_seed": dict(zip((str(seed) for seed in seeds), differences, strict=True)),
        "mean_difference": mean,
        "std_difference": std,
        "t_statistic": t_stat,
        "all_seeds_same_sign": same_sign,
        "exceeds_seed_variability": bool(
            count >= 3 and same_sign and abs(t_stat) > T_CRITICAL_DF2
        ),
    }


def _aggregate(runs: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> dict[str, Any]:
    """Mean, spread and range over seeds, per arm, cell and split."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["arm"]), str(run["cell"])), []).append(run)
    out: dict[str, Any] = {}
    for (arm, cell), cell_runs in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "seeds": sorted(int(run["seed"]) for run in cell_runs),
            "role": cell_runs[0]["role"],
            "placement_target": cell_runs[0]["placement_target"],
            "parent_convention": cell_runs[0]["parent_convention"],
            "parameters": int(cell_runs[0]["parameters"]["total"]),
            "training_steps": int(cell_runs[0]["protocol"]["steps"]),
            "train_seconds": _stats([float(run["timing"]["train_seconds"]) for run in cell_runs]),
        }
        for split in splits:
            per_split = {
                metric: _stats(
                    [float(run["splits"][split]["inferred"][metric]) for run in cell_runs]
                )
                for metric in (*REPORTED, "placement_floor_margin", "scale_floor_margin")
                if all(metric in run["splits"][split]["inferred"] for run in cell_runs)
            }
            for field in (
                "mean_deg",
                "median_deg",
                "std_deg",
                "within_5_deg",
                "within_10_deg",
                "within_20_deg",
                "below_floor",
                "floor_margin_deg",
            ):
                per_split[f"rotation_{field}"] = _stats(
                    [float(run["rotation"][split][field]) for run in cell_runs]
                )
            entry[split] = per_split
        out[f"{arm}/{cell}"] = entry
    return out


def _depth_aggregate(runs: Sequence[Mapping[str, Any]], splits: Sequence[str]) -> dict[str, Any]:
    """Per-depth position, rotation and scale over seeds."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["arm"]), str(run["cell"])), []).append(run)
    out: dict[str, Any] = {}
    for (arm, cell), cell_runs in sorted(grouped.items()):
        entry: dict[str, Any] = {}
        for split in splits:
            depths = sorted(
                {d for run in cell_runs for d in run["depth"]["splits"][split]["depths"]}, key=int
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


def assemble(output_dir: Path, *, corpus_dir: Path) -> dict[str, Any]:
    """Build ``change2_report.json`` from every saved run file."""
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*.json"))
    ]
    if not runs:
        raise SystemExit(f"no run files under {output_dir / 'runs'}")
    protocols = {json.dumps(run["protocol"], sort_keys=True) for run in runs}
    if len(protocols) != 1:
        raise SystemExit(f"run files span {len(protocols)} protocols")
    protocol = runs[0]["protocol"]
    splits = list(protocol["splits"])
    floors = placement_floor(corpus_dir, splits=splits)
    seeds = sorted({int(run["seed"]) for run in runs})
    manifest = load_step8_manifest(corpus_dir)
    calibration_path = output_dir / "rotation_weight_calibration.json"
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.exists()
        else {}
    )

    report = {
        "experiment_id": "step10-change2-training",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "status": "confirmatory" if len(seeds) >= 3 else "exploratory",
        "arms": [arm for arm in ARMS if any(run["arm"] == arm for run in runs)],
        "cells": {
            name: {"role": CELLS[name][3], "question": CELLS[name][4]}
            for name in CELLS
            if any(run["cell"] == name for run in runs)
        },
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "parent_corpus_id": manifest.parent_corpus_id,
            "corpus_dir": str(corpus_dir),
            "rotation_enabled": manifest.rotation_enabled,
            "splits": splits,
            "data_label": manifest.data_label,
        },
        "protocol": protocol,
        "objective": {
            "rotation_objective": ROTATION_OBJECTIVE,
            "rotation_loss_weight": protocol["rotation_loss_weight"],
            "scale_weight": 0.5,
            "translation_weight": 1.0,
            "frame_weight": 2.0,
            "calibration": calibration.get("rule"),
            "calibration_shares": calibration.get("resulting_shares_of_frame_loss"),
        },
        "seeds": seeds,
        "decision_rule": {
            "exceeds_seed_variability": (
                "all paired seed differences share a sign AND |t| > "
                f"{T_CRITICAL_DF2} (two-sided 5%, df=2)"
            ),
            "fixed_before_results": True,
            "inherited_from": "Change 1",
        },
        "floors": floors,
        "runs": [{k: v for k, v in run.items() if k != "manifest"} for run in runs],
        "aggregates": _aggregate(runs, splits),
        "depth_aggregates": _depth_aggregate(runs, splits),
        "paired": {
            split: {
                f"{arm}/{treatment}-vs-T0_global/{metric}": paired_comparison(
                    runs, arm, treatment, split, metric
                )
                for arm in ARMS
                for treatment in ("T4_rigid", "T1_spatial")
                for metric in ("translation_error", "rotation_deg", "scale_error")
                if any(run["arm"] == arm and run["cell"] == treatment for run in runs)
            }
            for split in splits
        },
        "notes": [
            "Position, rotation and scale are reported separately and never collapsed.",
            "Every component is read against this corpus's own floor on the same split.",
            "T1_spatial is a replication of Change 1 under a different corpus.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    _write_atomic(output_dir / "change2_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 Change 2 training.")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step10-change2"))
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=ROTATION_LOSS_WEIGHT,
        help=(
            "the rotation term's coefficient. Defaults to Change 2's calibrated 0.33; the "
            f"validation screening later chose {SCREENED_ROTATION_WEIGHT}. A different "
            "weight is a different experiment and needs its own --out."
        ),
    )
    parser.add_argument(
        "--relation-values",
        action="store_true",
        help=(
            "give each relation its own message as well as an attention weight. Step 11 "
            "measured the weight-only channel as worth 0.0003 degrees; this is the change "
            "that tests it. A different encoder is a different experiment and needs its own "
            "--out."
        ),
    )
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args(argv)

    unknown = [cell for cell in args.cells if cell not in CELLS]
    if unknown:
        raise SystemExit(f"unknown cells {unknown}; expected some of {list(CELLS)}")
    if not args.assemble_only:
        run_change2(
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
            rotation_weight=args.rotation_weight,
            relation_values=args.relation_values,
        )
    report = assemble(args.out, corpus_dir=args.corpus)
    print(json.dumps(report["objective"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
