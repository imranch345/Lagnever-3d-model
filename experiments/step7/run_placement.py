"""Step 7: re-evaluate trained checkpoints under both placement conditions.

The whole-organ suite evaluated with the scene's own entity frames supplied, so the
model was asked only what shape to put at a given place. That condition cannot decide
whether the relationship graph is necessary, because the arrangement the graph would
supply is already given through another input (amendment A8).

This script scores every saved checkpoint twice, on the full held-out split:

``given``
    entity frames supplied, the original condition, unchanged.
``inferred``
    the model must predict placement from structure, relations and text.

No retraining. The same weights are judged in both conditions, so a difference between
the two is a property of what the model was asked, not of how it was trained.

    python -m experiments.step7.run_placement \
        --corpus datasets/processed/whole_organ_1600_v2 \
        --checkpoints experiments/runs/step7-whole-organ/checkpoints
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
from datasets.whole_organ.corpus import WholeOrganScene, load_manifest, load_split
from experiments.step7.metrics import evaluate_whole_organ
from experiments.step7.relation_baseline import relation_baseline
from generation.neural.nn.device import resolve_device
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.manifest import environment_report, git_commit
from training.whole_organ import WholeOrganLoader

__all__ = ["CONDITIONS", "REPORTED", "evaluate_checkpoint", "run_placement", "main"]

#: The two placement conditions, and whether the model must predict frames.
CONDITIONS: Mapping[str, bool] = {"given": False, "inferred": True}

REPORTED: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "spatial_relation_accuracy",
    "spatial_relation_accuracy_strict",
    "spatial_relation_coverage",
    "placement_error",
    "scene_iou",
    "occupancy_chamfer",
    "counterfactual_relation_sensitivity",
    "counterfactual_correctness",
    "invariant_preservation",
    "eval_scenes",
)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    *,
    scenes: Sequence[WholeOrganScene],
    builder: WholeOrganBatchBuilder,
    device: str,
    batch_size: int,
    full: bool,
) -> dict[str, Any]:
    """Score one checkpoint under every placement condition."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = str(state["config"]["arm"])
    seed = int(state["config"]["seed"])
    choice = resolve_device(device)

    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    probe = builder.build(
        [scene for scene in scenes if scene.active_lod == scenes[0].active_lod][:1]
    )
    model, parameters = build_model(
        cast(Any, arm), cast(Any, builder), text_features=int(probe.batch.text_features.shape[1])
    )
    model.load_state_dict(state["model"])
    model.to(choice.device)
    model.eval()

    slot_of = {entity_id: index for index, entity_id in enumerate(builder.ontology.ids())}
    ids = list(builder.ontology.ids())

    out: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "parameters": int(parameters["total"]),
        "conditions": {},
    }
    for name, predicted in CONDITIONS.items():
        totals: dict[str, list[float]] = {}
        by_variant: dict[str, dict[str, list[float]]] = {}
        scenes_seen = 0
        for whole in loader.epoch(0):
            values = evaluate_whole_organ(
                model,
                whole,
                slot_of,
                ids,
                device=choice.device,
                full=full,
                builder=builder,
                use_predicted_frames=predicted,
            )
            scenes_seen += len(whole.scenes)
            for key, value in values.items():
                totals.setdefault(key, []).append(value)
            variants = {str(scene.variant) for scene in whole.scenes}
            if len(variants) == 1:
                bucket = by_variant.setdefault(next(iter(variants)), {})
                for key, value in values.items():
                    bucket.setdefault(key, []).append(value)

        summary: dict[str, float] = {}
        for key, collected in totals.items():
            finite = [value for value in collected if math.isfinite(value)]
            summary[key] = statistics.fmean(finite) if finite else float("nan")
        summary["eval_scenes"] = float(scenes_seen)
        for variant, bucket in sorted(by_variant.items()):
            for key, collected in bucket.items():
                finite = [value for value in collected if math.isfinite(value)]
                summary[f"variant_{variant}_{key}"] = (
                    statistics.fmean(finite) if finite else float("nan")
                )
        out["conditions"][name] = summary
    return out


def run_placement(
    *,
    corpus_dir: Path,
    checkpoint_dir: Path,
    split: str,
    device: str,
    batch_size: int,
    limit: int | None,
    full: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Score every checkpoint in a directory under both placement conditions."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_manifest(corpus_dir)
    scenes = load_split(corpus_dir, split, limit=limit)

    runs: list[dict[str, Any]] = []
    for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
        print(f"[step7-placement] {checkpoint.name}", flush=True)
        record = evaluate_checkpoint(
            checkpoint,
            scenes=scenes,
            builder=builder,
            device=device,
            batch_size=batch_size,
            full=full,
        )
        runs.append(record)
        for condition, values in record["conditions"].items():
            shown = {
                key: round(float(values[key]), 4)
                for key in ("entity_iou_mean", "spatial_relation_accuracy", "placement_error")
                if key in values
            }
            print(
                f"[step7-placement]   {record['arm']} seed {record['seed']} {condition}: {shown}",
                flush=True,
            )

    gathered: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in runs:
        arm = str(record["arm"])
        for condition, summary in record["conditions"].items():
            slot = gathered.setdefault(arm, {}).setdefault(condition, {})
            for metric in REPORTED:
                if metric not in summary:
                    continue
                slot.setdefault(metric, []).append(float(summary[metric]))

    aggregates: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for arm, conditions in gathered.items():
        for condition, metrics in conditions.items():
            for metric, collected in metrics.items():
                finite = [value for value in collected if math.isfinite(value)]
                aggregates.setdefault(arm, {}).setdefault(condition, {})[metric] = {
                    "mean": statistics.fmean(finite) if finite else float("nan"),
                    "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
                    "n": float(len(finite)),
                }

    report = {
        "experiment_id": "step7-placement-conditions",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "split": split,
            "scenes": len(scenes),
            "distinct_relation_graphs": manifest.distinct_relation_graphs,
            "data_label": manifest.data_label,
        },
        "relation_baseline": relation_baseline(corpus_dir, split),
        "conditions": {
            "given": "entity frames supplied from the scene; the original condition",
            "inferred": "the model predicts placement; the condition the graph decision rests on",
        },
        "runs": runs,
        "aggregates": aggregates,
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "No retraining. The same weights are scored in both conditions.",
            "Spatial relation accuracy must be read against the relation-blind floor in "
            "`relation_baseline`, not against zero.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "placement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 7 placement-condition evaluation.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-full", action="store_true", help="Skip the expensive probes.")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step7-placement"))
    args = parser.parse_args(argv)

    report = run_placement(
        corpus_dir=args.corpus,
        checkpoint_dir=args.checkpoints,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        full=not args.no_full,
        output_dir=args.out,
    )
    print(json.dumps(report["aggregates"], indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
