"""Step 7 Experiments 1, 2 and 3: which parts of the representation are being read.

Loads trained checkpoints, damages one channel of the Anatomical World Representation at
a time, and measures what stops working. A channel that can be deleted with no effect is
not being used, whatever the architecture says it is for.

Scored under both placement conditions, because supplying ground-truth entity frames
hands the model the arrangement a relationship graph would otherwise have to supply
(amendment A8). The inferred condition is the one a KEEP or REMOVE decision rests on.

    python -m experiments.step7.run_perturbation \
        --corpus datasets/processed/whole_organ_1600_v2 \
        --checkpoints experiments/runs/step7-whole-organ/checkpoints
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.corpus import load_manifest, load_split
from experiments.step7.metrics import entity_ownership, predicted_centroids, relation_accuracy
from experiments.step7.perturbations import (
    AWR_CASES,
    RELATION_PERTURBATIONS,
    describe,
    perturb_relations,
    restrict_graphs,
)
from experiments.step7.relation_baseline import relation_baseline
from generation.neural.nn.device import resolve_device
from generation.neural.nn.tensors import PrototypeBatch
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.manifest import environment_report, git_commit
from training.whole_organ import WholeOrganLoader

__all__ = ["METRICS", "run_perturbation", "main"]

#: Metrics reported per perturbation. A subset of the pre-registered set: the ones that
#: can be computed without re-deriving ground truth for a damaged graph.
METRICS: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "spatial_relation_accuracy",
    "spatial_relation_accuracy_strict",
    "placement_error",
)


@torch.no_grad()
def _score(
    model: torch.nn.Module,
    loader: WholeOrganLoader,
    slot_of: Mapping[str, int],
    *,
    device: torch.device,
    transform: Callable[[PrototypeBatch], PrototypeBatch],
    use_predicted_frames: bool,
) -> dict[str, float]:
    """Evaluate one transformed view of the held-out split."""
    totals: dict[str, list[float]] = {}
    for whole in loader.epoch(0):
        batch = transform(whole.batch).to(device)
        output = model(batch, use_predicted_frames=use_predicted_frames)
        slots = [int(slot) for slot in batch.structure.visible_slots]
        values = dict(entity_ownership(output.part_logits, batch.part_owner, slots))
        centroids = predicted_centroids(output.part_logits, batch.scene_points, slots)
        values.update(relation_accuracy(centroids, whole.scenes, slot_of))
        from experiments.step7.metrics import placement_error

        values["placement_error"] = placement_error(centroids, whole.scenes, slot_of)
        for key, value in values.items():
            totals.setdefault(key, []).append(value)
    out: dict[str, float] = {}
    for key, collected in totals.items():
        finite = [value for value in collected if math.isfinite(value)]
        out[key] = statistics.fmean(finite) if finite else float("nan")
    return out


def run_perturbation(
    *,
    corpus_dir: Path,
    checkpoint_dir: Path,
    split: str,
    arms: Sequence[str],
    seed: int,
    device: str,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run every perturbation and every partial-representation case for each arm."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_manifest(corpus_dir)
    scenes = load_split(corpus_dir, split)
    loader = WholeOrganLoader(
        scenes, builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    inverse = builder.inverse_relation_table()
    choice = resolve_device(device)
    probe = builder.build(scenes[:1])
    text_features = int(probe.batch.text_features.shape[1])

    results: dict[str, Any] = {}
    for arm in arms:
        checkpoint = checkpoint_dir / f"wo-{arm}-seed{seed}.pt"
        if not checkpoint.exists():
            print(f"[step7-perturb] {arm}: no checkpoint, skipped", flush=True)
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model, _ = build_model(cast(Any, arm), cast(Any, builder), text_features=text_features)
        model.load_state_dict(state["model"])
        model.to(choice.device)
        model.eval()

        arm_results: dict[str, Any] = {"perturbations": {}, "awr_cases": {}}
        for condition, predicted in (("given", False), ("inferred", True)):
            for kind in RELATION_PERTURBATIONS:
                generator = torch.Generator().manual_seed(1234)

                def damage(
                    batch: PrototypeBatch,
                    kind: str = kind,
                    generator: torch.Generator = generator,
                ) -> PrototypeBatch:
                    return perturb_relations(
                        batch, kind, inverse_relations=inverse, generator=generator
                    )

                transform: Callable[[PrototypeBatch], PrototypeBatch] = damage
                values = _score(
                    model,
                    loader,
                    slot_of,
                    device=choice.device,
                    transform=transform,
                    use_predicted_frames=predicted,
                )
                arm_results["perturbations"].setdefault(kind, {})[condition] = values
            for case, keep in AWR_CASES.items():

                def restrict(
                    batch: PrototypeBatch, keep: Sequence[str] = keep
                ) -> PrototypeBatch:
                    return restrict_graphs(batch, keep)

                restricted: Callable[[PrototypeBatch], PrototypeBatch] = restrict
                values = _score(
                    model,
                    loader,
                    slot_of,
                    device=choice.device,
                    transform=restricted,
                    use_predicted_frames=predicted,
                )
                arm_results["awr_cases"].setdefault(case, {})[condition] = values
            print(f"[step7-perturb] {arm} {condition} done", flush=True)

        # Degradation against the intact control, which is what the reading rule uses.
        for group, control in (("perturbations", "intact"), ("awr_cases", "A_full")):
            reference = arm_results[group][control]
            for name, conditions in arm_results[group].items():
                for condition, values in conditions.items():
                    base = reference[condition]
                    values["delta_entity_iou_mean"] = (
                        values["entity_iou_mean"] - base["entity_iou_mean"]
                    )
                    values["delta_spatial_relation_accuracy"] = (
                        values["spatial_relation_accuracy"] - base["spatial_relation_accuracy"]
                    )
                del name
        results[arm] = arm_results
        for kind, conditions in arm_results["perturbations"].items():
            shown = conditions["inferred"]
            print(
                f"[step7-perturb]   {arm} {kind:30} inferred "
                f"iou={shown['entity_iou_mean']:.4f} "
                f"(delta {shown['delta_entity_iou_mean']:+.4f})",
                flush=True,
            )

    report = {
        "experiment_id": "step7-perturbation",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "split": split,
            "scenes": len(scenes),
            "data_label": manifest.data_label,
        },
        "relation_baseline": relation_baseline(corpus_dir, split),
        "perturbations": {name: describe(name) for name in RELATION_PERTURBATIONS},
        "awr_cases": {name: describe(name) for name in AWR_CASES},
        "reading_rule": (
            "A channel the model depends on must degrade the output when corrupted. A "
            "channel that can be deleted with no measurable effect is not being used."
        ),
        "arms": results,
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "Models were trained on intact input; only evaluation is perturbed.",
            "Entities, presence, text features and points are identical across "
            "perturbations, so any change came through the relationship graph.",
            "Both placement conditions are reported; the inferred one decides.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perturbation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 7 perturbation experiments.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--arms", nargs="+", default=["A3", "A3L", "A2", "A1", "A1M"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step7-perturbation"))
    args = parser.parse_args(argv)

    run_perturbation(
        corpus_dir=args.corpus,
        checkpoint_dir=args.checkpoints,
        split=args.split,
        arms=args.arms,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
