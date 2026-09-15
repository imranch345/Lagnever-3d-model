"""Step 9 baseline: the Step 8 models, re-scored against the placement-blind floor.

No retraining and no code change to what is being measured. This re-evaluates the Step 8
checkpoints so that Step 9 starts from a reproducible baseline in its own run directory,
with three things Step 8 did not report:

* the **placement-blind floor** beside every frame number, so a frame error is read against
  what identity alone achieves rather than against zero;
* the **frame decomposition**, so the structurally-zero rotation term is visible instead of
  folded into a composite;
* the **oracle gap** per split, which is what says whether placement is still the binding
  constraint.

    python -m experiments.step9.run_baseline \
        --corpus datasets/processed/step8_continuous \
        --checkpoints experiments/runs/step8-suite/checkpoints \
        --out experiments/runs/step9-baseline
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

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
from experiments.step9.frame_report import decompose, describe_rotation, floor_gap
from experiments.step9.placement_floor import placement_floor
from generation.neural.nn.device import resolve_device
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.manifest import environment_report, git_commit

__all__ = ["ARMS", "REPORTED", "run_baseline", "main"]

ARMS: tuple[str, ...] = ("A1", "A1M", "A3Lite", "A3L", "A3")

REPORTED: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "scene_iou",
    "spatial_relation_accuracy",
    "translation_error",
    "scale_error",
    "rotation_error",
    "composite_frame_error",
    "placement_error",
    "eval_scenes",
)


def _seeds_for(checkpoint_dir: Path, arm: str) -> list[int]:
    """Which seeds have a checkpoint for this arm."""
    seeds: list[int] = []
    for path in sorted(checkpoint_dir.glob(f"s8-{arm}-seed*.pt")):
        seeds.append(int(path.stem.rsplit("seed", 1)[1]))
    return seeds


@torch.no_grad()
def run_baseline(
    *,
    corpus_dir: Path,
    checkpoint_dir: Path,
    arms: Sequence[str],
    splits: Sequence[str],
    device: str,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Score every Step 8 checkpoint on every split, in both placement conditions."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_step8_manifest(corpus_dir)
    slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    choice = resolve_device(device)

    floors = placement_floor(corpus_dir, splits=splits, batch_size=batch_size)
    probe = builder.build(load_step8_split(corpus_dir, "train", limit=1))
    text_features = int(probe.batch.text_features.shape[1])

    runs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in _seeds_for(checkpoint_dir, arm):
            path = checkpoint_dir / f"s8-{arm}-seed{seed}.pt"
            state = torch.load(path, map_location="cpu", weights_only=False)
            model, parameters = build_model(
                cast(Any, arm), cast(Any, builder), text_features=text_features
            )
            model.load_state_dict(state["model"])
            model.to(choice.device)
            model.eval()

            record: dict[str, Any] = {
                "arm": arm,
                "seed": seed,
                "checkpoint": str(path),
                "parameters": dict(parameters),
                "splits": {},
            }
            for split in splits:
                scenes = load_step8_split(corpus_dir, split)
                conditions: dict[str, dict[str, float]] = {}
                for name, predicted in (("inferred", True), ("supplied", False)):
                    values = evaluate_step8(
                        model,
                        scenes,
                        builder,
                        slot_of,
                        device=choice.device,
                        batch_size=batch_size,
                        use_predicted_frames=predicted,
                    )
                    conditions[name] = decompose(values)
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
            summary = record["splits"][splits[0]]["inferred"]
            print(
                f"[step9-baseline] {arm} seed {seed} {splits[0]}: "
                f"translation={summary['translation_error']:.4f} "
                f"floor={summary['placement_floor']:.4f} "
                f"gap={summary['placement_floor_gap']:+.4f} "
                f"iou={summary['entity_iou_mean']:.4f}",
                flush=True,
            )

    report = {
        "experiment_id": "step9-baseline",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "what_this_is": (
            "The Step 8 checkpoints re-scored, with no retraining and no change to what is "
            "measured. It exists so Step 9 has a reproducible starting point in its own "
            "run directory."
        ),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "splits": list(splits),
            "data_label": manifest.data_label,
        },
        "placement_floor": floors,
        "rotation_note": describe_rotation(
            float(runs[0]["splits"][splits[0]]["inferred"]["rotation_error"]) if runs else 0.0
        ),
        "runs": runs,
        "aggregates": _aggregate(runs, splits),
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "Headline condition is inferred placement. Supplied is an oracle bound.",
            "A frame error is read against the placement-blind floor, not against zero.",
            "The Step 8 composite is preserved unchanged; placement_error is the "
            "corrected position-oriented metric reported beside it.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _aggregate(
    runs: Sequence[Mapping[str, Any]], splits: Sequence[str]
) -> dict[str, Any]:
    """Mean, spread and seed count per arm, split and condition."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["arm"]), []).append(run)
    out: dict[str, Any] = {}
    for arm, arm_runs in grouped.items():
        out[arm] = {}
        for split in splits:
            out[arm][split] = {}
            for condition in ("inferred", "supplied"):
                metrics: dict[str, dict[str, float]] = {}
                keys = set()
                for run in arm_runs:
                    keys |= set(run["splits"][split][condition])
                for metric in sorted(keys):
                    values = [
                        float(run["splits"][split][condition][metric])
                        for run in arm_runs
                        if metric in run["splits"][split][condition]
                        and math.isfinite(float(run["splits"][split][condition][metric]))
                    ]
                    if not values:
                        continue
                    metrics[metric] = {
                        "mean": statistics.fmean(values),
                        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                        "n": float(len(values)),
                    }
                out[arm][split][condition] = metrics
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 9 baseline.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step9-baseline"))
    args = parser.parse_args(argv)

    run_baseline(
        corpus_dir=args.corpus,
        checkpoint_dir=args.checkpoints,
        arms=args.arms,
        splits=args.splits,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
