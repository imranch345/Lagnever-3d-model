"""Measure whether a model's output changes when only the arrangement changes.

This is the most direct form of the Step 7 question. Take one organ family, render it in
all four arrangements, and ask how much the four **predictions** differ from one another
compared with how much the four **truths** differ.

Presence, entity identity and text features are identical across the four by
construction. The only input that differs is the relationship graph. So:

* if the model reads the graph, its four predictions differ roughly as much as the four
  truths do, and the ratio approaches 1;
* if it does not, the predictions collapse toward a single average arrangement while the
  truths stay apart, and the ratio approaches 0.

The ratio is reported per placement condition, because supplying ground-truth entity
frames hands the model the arrangement through a different channel and the ratio then
says nothing about the graph.

**Disclosure:** this diagnostic was written *after* the placement-condition finding, not
before the runs. It measures the same thing the pre-registered
`counterfactual_relation_sensitivity` measures, on the composed scene rather than on
entity centroids, and it is reported alongside that metric rather than in place of it.
See amendment A14.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.parameters import Variant
from generation.neural.nn.device import resolve_device
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model

__all__ = ["arrangement_response", "main"]


def _disagreement(labels: Mapping[str, np.ndarray]) -> float:
    """Mean pairwise share of points that two arrangements label differently."""
    values = [
        float((labels[first] != labels[second]).mean())
        for first, second in itertools.combinations(sorted(labels), 2)
    ]
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def arrangement_response(
    checkpoint: Path,
    *,
    families: Sequence[int] = (0, 1, 2, 3),
    points: int = 1024,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure how much a model's output moves when only the arrangement changes."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    choice = resolve_device(device)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = str(state["config"]["arm"])
    probe = builder.build(
        [build_scene(scene_index=0, family_id=0, variant=Variant.NORMAL, lod=3)]
    )
    model, _ = build_model(
        cast(Any, arm), cast(Any, builder), text_features=int(probe.batch.text_features.shape[1])
    )
    model.load_state_dict(state["model"])
    model.to(choice.device)
    model.eval()

    rng = np.random.default_rng(20250915)
    truth_scores: list[float] = []
    scores: dict[str, list[float]] = {"given": [], "inferred": []}
    for family in families:
        sample = rng.uniform(-1.0, 1.0, size=(points, 3))
        probe_points = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(choice.device)
        truths: dict[str, np.ndarray] = {}
        predictions: dict[str, dict[str, np.ndarray]] = {"given": {}, "inferred": {}}
        for variant in Variant:
            scene = build_scene(scene_index=0, family_id=family, variant=variant, lod=3)
            truths[str(variant)] = scene.field().ownership(sample)
            batch = builder.build([scene]).batch.to(choice.device)
            shared = type(batch)(
                **{
                    **{name: getattr(batch, name) for name in batch.__dataclass_fields__},
                    "scene_points": probe_points,
                }
            )
            for condition, predicted in (("given", False), ("inferred", True)):
                output = model(shared, use_predicted_frames=predicted)
                predictions[condition][str(variant)] = (
                    output.part_logits.argmax(dim=-1)[0].cpu().numpy()
                )
        truth_scores.append(_disagreement(truths))
        for condition in scores:
            scores[condition].append(_disagreement(predictions[condition]))

    truth = float(np.mean(truth_scores))
    out: dict[str, Any] = {
        "arm": arm,
        "checkpoint": str(checkpoint),
        "families": list(families),
        "points_per_family": points,
        "truth_arrangement_disagreement": truth,
        "conditions": {},
    }
    for condition, values in scores.items():
        moved = float(np.mean(values))
        out["conditions"][condition] = {
            "prediction_arrangement_disagreement": moved,
            "response_ratio": moved / truth if truth > 1.0e-9 else float("nan"),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Arrangement response of trained models.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["A3", "A3L", "A2", "A1", "A1M", "A0"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = []
    for arm in args.arms:
        path = args.checkpoints / f"wo-{arm}-seed{args.seed}.pt"
        if not path.exists():
            print(f"[arrangement] {arm}: no checkpoint, skipped")
            continue
        record = arrangement_response(path, points=args.points)
        results.append(record)
        given = record["conditions"]["given"]["response_ratio"]
        inferred = record["conditions"]["inferred"]["response_ratio"]
        print(
            f"[arrangement] {arm:4} response ratio  given {given:.4f}   inferred {inferred:.4f}"
        )
    report = {
        "experiment_id": "step7-arrangement-response",
        "description": (
            "Share of the ground-truth variation between arrangements that the model "
            "reproduces. 1.0 means the model distinguishes arrangements as well as the "
            "truth does; 0.0 means it produces one average organ regardless."
        ),
        "disclosure": (
            "Written after the placement-condition finding, not before the runs. "
            "Reported alongside the pre-registered counterfactual metrics, not instead "
            "of them. See amendment A14."
        ),
        "runs": results,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
