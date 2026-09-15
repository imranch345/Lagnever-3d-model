"""Step 8 ablations R5 to R11: isolating what causes what.

The principal suite compares five arms. These ablations act on trained checkpoints and on
the evaluation itself, so each isolates one cause rather than confounding several.

======  ==========================================================================
R5      graph with shuffled endpoints: connectivity destroyed, edge count kept
R6      spatial edges removed
R7      functional edges removed
R8      predicted placement (the headline condition, repeated here as the control)
R9      ground-truth placement oracle: the upper bound better placement could buy
R10     nested level of detail, as trained
R11     non-nested level-of-detail baseline: each level decoded from its own block
======  ==========================================================================

R5, R6 and R7 damage the graph at evaluation time on a model trained on intact input. A
channel the model depends on must degrade the output when corrupted; a channel that can be
deleted with no measurable effect is not being read, whatever role the architecture
assigns it.

R11 is the one that needs care. A genuinely non-nested baseline would need its own trained
model, which would confound the comparison with a different training run. Instead it is
measured on the trained model by decoding each level from a **disjoint** token slice
rather than a prefix, which is the closest thing to "three unrelated representations" the
same weights can express, and is reported as an approximation rather than a trained
control.
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
from datasets.whole_organ.continuous_corpus import load_step8_manifest, load_step8_split
from experiments.step7.perturbations import perturb_relations, restrict_graphs
from experiments.step8.evaluation import evaluate_step8
from generation.neural.nn.tensors import PrototypeBatch
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.manifest import environment_report, git_commit
from training.whole_organ import WholeOrganLoader

__all__ = ["ABLATIONS", "run_ablations", "main"]

#: Each ablation, with the sentence that says what it isolates.
ABLATIONS: Mapping[str, str] = {
    "R5_shuffled_endpoints": "spatial edges rewired to random pairs; connectivity destroyed",
    "R6_no_spatial": "every spatial edge removed",
    "R7_no_functional": "every functional edge removed",
    "R8_predicted_placement": "the headline condition: the model places entities itself",
    "R9_oracle_placement": "true frames supplied; the upper bound better placement could buy",
    "R10_nested_lod": "levels decoded from nested prefixes, as trained",
    "R11_non_nested_lod": "levels decoded from disjoint token slices; an approximation",
}

METRICS: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "scene_iou",
    "spatial_relation_accuracy",
    "position_error",
    "composite_frame_error",
    "lod1_iou",
    "lod2_iou",
    "lod3_iou",
    "lod_containment_mean",
    "lod_preservation_mean",
)


@torch.no_grad()
def _non_nested_lod(
    model: torch.nn.Module,
    scenes: Sequence[Any],
    builder: WholeOrganBatchBuilder,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Decode each level from a disjoint token slice instead of a nested prefix.

    The decoder reads a prefix, so a disjoint slice is produced by zeroing the tokens
    outside it. Each level then sees a different, non-overlapping part of the block, which
    is the closest approximation to three unrelated representations these weights allow.
    Reported as an approximation: a properly non-nested control would need its own run.
    """
    from generation.neural.nn.nested_lod import nested_lod_metrics

    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    bounds = {1: (0, 8), 2: (8, 16), 3: (16, 24)}
    totals: dict[str, list[float]] = {}
    for whole in loader.epoch(0):
        batch = whole.batch.to(device)
        staged = model.stage(batch)  # type: ignore[operator]
        if staged.tokens is None:
            continue
        frames = model.frame_head(staged.entity_latent)  # type: ignore[operator]
        decodes: dict[int, torch.Tensor] = {}
        targets: dict[int, torch.Tensor] = {}
        for level, (low, high) in bounds.items():
            masked = torch.zeros_like(staged.tokens)
            masked[:, :, low:high] = staged.tokens[:, :, low:high]
            scene_logits, _, _, _ = model.decode_tokens(  # type: ignore[operator]
                batch,
                masked,
                frames=frames,
                present=batch.entity_present,
                active_tokens=high,
            )
            decodes[level] = scene_logits
            targets[level] = whole.lod_targets[level].to(device)
        values = nested_lod_metrics(decodes, targets)
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value))
    return {
        key: statistics.fmean([v for v in values if math.isfinite(v)])
        for key, values in totals.items()
        if any(math.isfinite(v) for v in values)
    }


def run_ablations(
    *,
    corpus_dir: Path,
    checkpoint_dir: Path,
    arms: Sequence[str],
    seed: int,
    split: str,
    device: str,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run every ablation on every arm that has a checkpoint."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_step8_manifest(corpus_dir)
    scenes = load_step8_split(corpus_dir, split)
    slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    inverse = builder.inverse_relation_table()
    from generation.neural.nn.device import resolve_device

    choice = resolve_device(device)
    text_features = int(builder.build(scenes[:1]).batch.text_features.shape[1])

    results: dict[str, Any] = {}
    for arm in arms:
        path = checkpoint_dir / f"s8-{arm}-seed{seed}.pt"
        if not path.exists():
            print(f"[step8-ablate] {arm}: no checkpoint, skipped", flush=True)
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model, _ = build_model(cast(Any, arm), cast(Any, builder), text_features=text_features)
        model.load_state_dict(state["model"])
        model.to(choice.device)
        model.eval()

        def score(
            transform: Callable[[PrototypeBatch], PrototypeBatch] | None,
            *,
            predicted: bool,
            model: torch.nn.Module = model,
        ) -> dict[str, float]:
            if transform is None:
                return evaluate_step8(
                    model,
                    scenes,
                    builder,
                    slot_of,
                    device=choice.device,
                    batch_size=batch_size,
                    use_predicted_frames=predicted,
                )
            wrapped = _TransformedBuilder(builder, transform)
            return evaluate_step8(
                model,
                scenes,
                cast(Any, wrapped),
                slot_of,
                device=choice.device,
                batch_size=batch_size,
                use_predicted_frames=predicted,
            )

        generator = torch.Generator().manual_seed(4242)
        arm_results: dict[str, Any] = {}
        arm_results["R8_predicted_placement"] = score(None, predicted=True)
        arm_results["R9_oracle_placement"] = score(None, predicted=False)

        def rewire(batch: PrototypeBatch, gen: torch.Generator = generator) -> PrototypeBatch:
            return perturb_relations(
                batch, "randomise_spatial_endpoints", inverse_relations=inverse, generator=gen
            )

        arm_results["R5_shuffled_endpoints"] = score(rewire, predicted=True)

        def drop_spatial(batch: PrototypeBatch) -> PrototypeBatch:
            return restrict_graphs(batch, ("structure", "functional"))

        def drop_functional(batch: PrototypeBatch) -> PrototypeBatch:
            return restrict_graphs(batch, ("structure", "spatial"))

        arm_results["R6_no_spatial"] = score(drop_spatial, predicted=True)
        arm_results["R7_no_functional"] = score(drop_functional, predicted=True)
        arm_results["R10_nested_lod"] = {
            key: value
            for key, value in arm_results["R8_predicted_placement"].items()
            if key.startswith("lod")
        }
        arm_results["R11_non_nested_lod"] = _non_nested_lod(
            model, scenes, builder, device=choice.device, batch_size=batch_size
        )

        control = arm_results["R8_predicted_placement"]
        for name, values in arm_results.items():
            if name in ("R10_nested_lod", "R11_non_nested_lod"):
                continue
            for metric in ("entity_iou_mean", "spatial_relation_accuracy"):
                if metric in values and metric in control:
                    values[f"delta_{metric}"] = values[metric] - control[metric]
        results[arm] = arm_results
        spatial_delta = arm_results["R6_no_spatial"].get("delta_entity_iou_mean", float("nan"))
        oracle_iou = arm_results["R9_oracle_placement"]["entity_iou_mean"]
        print(
            f"[step8-ablate] {arm}: R6 delta_iou={spatial_delta:+.4f} "
            f"R9 oracle_iou={oracle_iou:.4f}",
            flush=True,
        )

    report = {
        "experiment_id": "step8-ablations",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "split": split,
            "scenes": len(scenes),
            "data_label": manifest.data_label,
        },
        "ablations": dict(ABLATIONS),
        "seed": seed,
        "arms": results,
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "R5, R6 and R7 damage the graph at evaluation time on models trained intact.",
            "R11 approximates a non-nested baseline on trained weights; a proper control "
            "would need its own run and is named as missing rather than implied.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class _TransformedBuilder:
    """Wraps a batch builder so every batch is damaged the same way before use."""

    def __init__(
        self,
        builder: WholeOrganBatchBuilder,
        transform: Callable[[PrototypeBatch], PrototypeBatch],
    ) -> None:
        self._builder = builder
        self._transform = transform

    def __getattr__(self, name: str) -> Any:
        return getattr(self._builder, name)

    def build(self, scenes: Sequence[Any], **kwargs: Any) -> Any:
        from dataclasses import replace as dataclass_replace

        whole = self._builder.build(scenes, **kwargs)
        return dataclass_replace(whole, batch=self._transform(whole.batch))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 ablations.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["A3Lite", "A3L", "A3", "A1M", "A1"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step8-ablations"))
    args = parser.parse_args(argv)

    run_ablations(
        corpus_dir=args.corpus,
        checkpoint_dir=args.checkpoints,
        arms=args.arms,
        seed=args.seed,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
