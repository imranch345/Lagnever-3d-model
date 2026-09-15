"""The relation-blind reference point for the spatial-relation metric.

``spatial_relation_accuracy`` counts the share of a scene's measured spatial relations
that a prediction reproduces. On its own the number is hard to read, because many
relations survive every arrangement: the apex is inferior to the base whichever way the
organ is mirrored. A model that ignores relations completely therefore does not score
zero, it scores whatever those invariant relations are worth.

This module measures that floor directly. The **blind predictor** emits the `NORMAL`
arrangement of the scene's own family no matter which variant the scene actually is,
which is precisely the behaviour of a model that cannot see the relationship graph. The
**oracle** emits the scene's own measured centroids and must score 1.0; it exists to
prove the scoring code is not the thing under test.

The baseline depends only on the corpus, never on a model or an arm, so it is computed
once per corpus and referenced by every run. It does not alter the metric: it says what
value of the unchanged metric corresponds to using no relational information at all.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np

from datasets.whole_organ.arrangement import ARRANGEMENT_FIELDS, Arrangement
from datasets.whole_organ.corpus import WholeOrganScene, load_split
from datasets.whole_organ.field import WholeOrganField

__all__ = [
    "AXIS_OF",
    "blind_centroids",
    "oracle_centroids",
    "score_against_measured_relations",
    "relation_baseline",
    "main",
]

#: The origin of the arrangement space: no mirroring, no transposition, every
#: continuous coordinate zero. What a model that ignores relations can emit.
CANONICAL_ARRANGEMENT = Arrangement(
    mirror=False,
    transpose=False,
    **{name: 0.0 for name in ARRANGEMENT_FIELDS},
)

AXIS_OF: Mapping[str, tuple[int, float]] = {
    "left_of": (0, 1.0),
    "right_of": (0, -1.0),
    "superior_to": (1, 1.0),
    "inferior_to": (1, -1.0),
    "anterior_to": (2, 1.0),
    "posterior_to": (2, -1.0),
}


def oracle_centroids(scene: WholeOrganScene) -> dict[str, np.ndarray]:
    """The scene's own measured centroids. Scores 1.0 by construction."""
    return {name: np.asarray(value, dtype=float) for name, value in scene.centroids.items()}


def blind_centroids(
    scene: WholeOrganScene, cache: dict[tuple[int, int], dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """The canonical upright arrangement of this scene's family, whatever the scene is.

    Step 7 defined this as the `NORMAL` variant, one of four. Step 8's arrangements are
    continuous, so "normal" is no longer a point anyone draws; the canonical arrangement
    is the origin of the space, with no mirroring, no transposition and every continuous
    coordinate at zero. It is still exactly what a relation-blind model can do: commit to
    one fixed arrangement and ignore what the relations say.

    The floor this produces is not the Step 7 floor and must be recomputed rather than
    carried over.
    """
    key = (int(scene.family_id), int(scene.seed))
    if key not in cache:
        parameters = dataclass_replace(
            scene.parameters, arrangement=CANONICAL_ARRANGEMENT
        )
        organ = WholeOrganField(parameters)
        statistics = organ.entity_statistics(np.random.default_rng(scene.seed + 101))
        cache[key] = {
            name: np.asarray(value["centroid"], dtype=float)
            for name, value in statistics.items()
        }
    return cache[key]


def score_against_measured_relations(
    scenes: Sequence[WholeOrganScene],
    centroids_for: Any,
) -> dict[str, Any]:
    """Share of measured spatial relations satisfied, overall and broken down."""
    per_relation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_variant: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = [0, 0]
    for scene in scenes:
        centroids = centroids_for(scene)
        for edge in scene.edges:
            if edge.relation not in AXIS_OF:
                continue
            first = centroids.get(edge.subject)
            second = centroids.get(edge.object)
            if first is None or second is None:
                continue
            axis, sign = AXIS_OF[edge.relation]
            satisfied = int(sign * float(first[axis] - second[axis]) > 0.0)
            for bucket in (
                per_relation[edge.relation],
                per_variant[str(scene.variant)],
                total,
            ):
                bucket[0] += satisfied
                bucket[1] += 1
    def ratio(bucket: Sequence[int]) -> float:
        return float(bucket[0] / bucket[1]) if bucket[1] else 0.0

    return {
        "accuracy": ratio(total),
        "edges_scored": total[1],
        "by_relation": {
            name: {"accuracy": ratio(bucket), "n": bucket[1]}
            for name, bucket in sorted(per_relation.items())
        },
        "by_variant": {
            name: {"accuracy": ratio(bucket), "n": bucket[1]}
            for name, bucket in sorted(per_variant.items())
        },
    }


def relation_baseline(corpus_dir: str | Path, split: str = "test") -> dict[str, Any]:
    """Compute the oracle and the relation-blind floor for one split of a corpus."""
    scenes = load_split(corpus_dir, split)
    cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    oracle = score_against_measured_relations(scenes, oracle_centroids)
    blind = score_against_measured_relations(scenes, lambda s: blind_centroids(s, cache))
    headroom = 1.0 - blind["accuracy"]
    return {
        "corpus_dir": str(corpus_dir),
        "split": split,
        "scenes": len(scenes),
        "oracle": oracle,
        "blind_normal_arrangement": blind,
        "discriminative_headroom": headroom,
        "interpretation": (
            "A model that ignores relations entirely scores "
            f"{blind['accuracy']:.4f} on spatial_relation_accuracy. Only the "
            f"{headroom:.4f} above that is evidence of relational competence. Report "
            "the gap, not the raw accuracy."
        ),
        "notes": [
            "Oracle must be 1.0; any other value means the scoring code is wrong.",
            "The blind predictor is the NORMAL arrangement of the scene's own family.",
            "Computed from the corpus alone. No model, no arm, no training.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Relation-blind baseline for a corpus.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test", "validation"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {split: relation_baseline(args.corpus, split) for split in args.splits}
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for split, values in report.items():
        print(
            f"[{split}] oracle={values['oracle']['accuracy']:.4f} "
            f"blind={values['blind_normal_arrangement']['accuracy']:.4f} "
            f"headroom={values['discriminative_headroom']:.4f} "
            f"edges={values['oracle']['edges_scored']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
