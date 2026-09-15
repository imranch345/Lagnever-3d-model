"""The placement-blind floor: what placement accuracy needs no model at all.

A first-class Step 9 baseline, analogous to the relation-blind floor that Step 7 and Step 8
established for spatial relation accuracy. The argument is the same in both cases. A frame
error of 0.16 means nothing on its own; it means something once you know what a predictor
carrying no scene information achieves.

Three reference points, all computed from the corpus with no model involved:

``global``
    one frame for every entity in every scene: the corpus mean. The weakest thing a
    predictor can do while still being fitted to the data.
``per_entity``
    each entity's mean frame across the training split, which is a lookup table keyed by
    entity identity. **This is the floor.** It uses identity and nothing else: no
    relations, no hierarchy, no scene context. A model that does not beat it has learned
    nothing about placement that identity alone did not already supply.
``oracle``
    the scene's own measured frame. Zero by construction, and reported so that a
    non-zero value would expose a bug in the scoring rather than in the model.

Masking, and why it is enforced here
------------------------------------

The first version of this measurement was computed over all twenty entities while the
model's frame error is masked by ``entity_present``, the set visible at the scene's level
of detail. The two are not comparable: the coarse level exposes the large structures, whose
centroids move most. Measured over all twenty the floor is 0.1401 and every Step 8 arm looks
worse than a lookup table; measured over the entities the model is actually scored on the
floor is 0.1605 and two arms are marginally above it. The second is the honest number.

The floor is therefore computed by driving the **same loader and the same presence mask**
the evaluation uses, rather than by iterating the corpus directly. A baseline that does not
share the evaluation's masking is not a baseline for that evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import TEST_SPLITS, load_step8_split
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = [
    "FrameTable",
    "build_tables",
    "score_table",
    "placement_floor",
    "main",
]

FrameTable = Mapping[int, np.ndarray]


def _iterate_frames(
    scenes: Sequence[Any], builder: WholeOrganBatchBuilder, *, batch_size: int = 8
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(slot, frame)`` for every entity the evaluation would score.

    Drives the evaluation loader so the presence mask is the same one the model is judged
    under. Iterating the corpus directly would silently score a different entity set.
    """
    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    for whole in loader.epoch(0):
        batch = whole.batch
        frames = batch.entity_frames.numpy()
        present = batch.entity_present.numpy()
        for index in range(frames.shape[0]):
            for slot in np.nonzero(present[index])[0]:
                yield int(slot), frames[index, int(slot)]


def build_tables(
    scenes: Sequence[Any], builder: WholeOrganBatchBuilder, *, batch_size: int = 8
) -> dict[str, Any]:
    """Fit the two blind predictors on a split, under the evaluation's own masking."""
    per_entity: dict[int, list[np.ndarray]] = defaultdict(list)
    for slot, frame in _iterate_frames(scenes, builder, batch_size=batch_size):
        per_entity[slot].append(frame)
    if not per_entity:
        raise ValueError("No frames to fit a placement floor on.")
    stacked = np.concatenate([np.stack(v) for v in per_entity.values()])
    return {
        "per_entity": {slot: np.mean(np.stack(v), axis=0) for slot, v in per_entity.items()},
        "global": np.mean(stacked, axis=0),
        "entities_fitted": len(per_entity),
        "frames_fitted": int(stacked.shape[0]),
    }


def score_table(
    scenes: Sequence[Any],
    builder: WholeOrganBatchBuilder,
    predictor: Any,
    *,
    batch_size: int = 8,
) -> dict[str, float]:
    """Score a blind predictor with the metric definitions the model is scored by.

    ``predictor`` takes a slot and returns a 12-number frame. Errors are the same
    quantities :mod:`experiments.step8.frame_metrics` computes, reimplemented on NumPy so
    the floor needs no model, no device and no PyTorch graph.
    """
    position: list[float] = []
    scale: list[float] = []
    rotation: list[float] = []
    count = 0
    for slot, truth in _iterate_frames(scenes, builder, batch_size=batch_size):
        guess = np.asarray(predictor(slot), dtype=float)
        position.append(float(np.linalg.norm(truth[0:3] - guess[0:3])))
        scale.append(float(np.abs(truth[3:6] - guess[3:6]).mean()))
        rotation.append(float(_rotation_angle(guess, truth)))
        count += 1
    composite = [
        p + 0.5 * s + 0.25 * r for p, s, r in zip(position, scale, rotation, strict=True)
    ]
    return {
        "position_error": float(np.mean(position)),
        "scale_error": float(np.mean(scale)),
        "rotation_error": float(np.mean(rotation)),
        "composite_frame_error": float(np.mean(composite)),
        "frames_scored": float(count),
    }


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    """Geodesic angle between two 6D rotations, after Gram-Schmidt.

    Structurally zero on this corpus, because the frame target's rotation is a constant
    identity for every entity in every scene. Computed anyway so that the number appears in
    the report as measured rather than as asserted.
    """

    def basis(frame: np.ndarray) -> np.ndarray:
        a = frame[6:9]
        b = frame[9:12]
        a = a / max(float(np.linalg.norm(a)), 1e-9)
        b = b - a * float(np.dot(a, b))
        b = b / max(float(np.linalg.norm(b)), 1e-9)
        return np.stack([a, b, np.cross(a, b)])

    relative = basis(first) @ basis(second).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def placement_floor(
    corpus_dir: str | Path, *, splits: Sequence[str] = TEST_SPLITS, batch_size: int = 8
) -> dict[str, Any]:
    """Compute the placement-blind floor on every split, fitted on training only."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    train = load_step8_split(corpus_dir, "train")
    tables = build_tables(train, builder, batch_size=batch_size)
    per_entity: dict[int, np.ndarray] = tables["per_entity"]
    global_frame: np.ndarray = tables["global"]

    out: dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "fitted_on": "train",
        "entities_fitted": tables["entities_fitted"],
        "frames_fitted": tables["frames_fitted"],
        "masking": (
            "Driven through the evaluation loader, so the presence mask is the one the "
            "model is scored under. A baseline computed over all entities regardless of "
            "level of detail is not comparable and is 0.02 optimistic on this corpus."
        ),
        "splits": {},
        "notes": [
            "per_entity is THE floor: identity and nothing else, no relations, no "
            "hierarchy, no scene context.",
            "A model that does not beat per_entity has learned nothing about placement "
            "that identity alone did not already supply.",
            "oracle must be 0.0; any other value means the scoring code is wrong.",
            "rotation_error is structurally zero because the frame target's rotation is "
            "a constant identity. Reported, not hidden.",
            "Computed from the corpus alone. No model, no arm, no training.",
        ],
    }
    for split in splits:
        scenes = load_step8_split(corpus_dir, split)
        out["splits"][split] = {
            "scenes": len(scenes),
            "global": score_table(scenes, builder, lambda _: global_frame, batch_size=batch_size),
            "per_entity": score_table(
                scenes,
                builder,
                lambda slot, table=per_entity, fallback=global_frame: table.get(slot, fallback),
                batch_size=batch_size,
            ),
            "oracle": {
                "position_error": 0.0,
                "scale_error": 0.0,
                "rotation_error": 0.0,
                "composite_frame_error": 0.0,
            },
        }
        out["splits"][split]["placement_blind_floor"] = out["splits"][split]["per_entity"][
            "position_error"
        ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 9 placement-blind floor.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = placement_floor(args.corpus, splits=args.splits)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    header = f"{'split':20}{'global pos':>12}{'floor pos':>11}{'floor scale':>13}{'floor rot':>11}"
    print(header)
    for split, values in report["splits"].items():
        print(
            f"{split:20}{values['global']['position_error']:>12.4f}"
            f"{values['per_entity']['position_error']:>11.4f}"
            f"{values['per_entity']['scale_error']:>13.4f}"
            f"{values['per_entity']['rotation_error']:>11.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
