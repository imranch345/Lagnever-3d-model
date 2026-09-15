"""The relation-blind floor for the Step 8 corpus, recomputed rather than carried over.

Step 7 established that spatial relation accuracy must be read against what a model with
no relational information can achieve, because many relations survive every arrangement.
Its floor was 0.8878. That number belongs to the Step 7 corpus and does not transfer:
Step 8's arrangements are continuous and range further, so a fixed guess should do worse.

Three reference points, all computed from the corpus alone with no model involved:

``oracle``
    the scene's own measured centroids. Must be 1.0, or the scoring code is what is
    broken rather than the model.
``blind_canonical``
    the canonical upright arrangement of the scene's own family, whatever the scene
    actually is. This is what a relation-blind model can do: commit to one arrangement.
``blind_mean``
    the mean centroid of each entity across the training split, which is the other thing
    a relation-blind model can do: predict the average. Reported because it is the
    stronger of the two blind strategies on some corpora and the honest floor is the
    higher of them.

The floor used in the report is the **higher** of the two blind predictors. Taking the
lower one would flatter every model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np

from datasets.whole_organ.continuous_corpus import TEST_SPLITS, load_step8_split
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.field import WholeOrganField
from experiments.step7.relation_baseline import (
    CANONICAL_ARRANGEMENT,
    oracle_centroids,
    score_against_measured_relations,
)

__all__ = ["step8_relation_floor", "main"]


def _canonical_centroids(
    scene: WholeOrganScene, cache: dict[tuple[int, int], dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """The canonical upright arrangement of this scene's family."""
    key = (int(scene.family_id), int(scene.seed))
    if key not in cache:
        parameters = dataclass_replace(scene.parameters, arrangement=CANONICAL_ARRANGEMENT)
        organ = WholeOrganField(parameters)
        statistics = organ.entity_statistics(np.random.default_rng(scene.seed + 101))
        cache[key] = {
            name: np.asarray(value["centroid"], dtype=float) for name, value in statistics.items()
        }
    return cache[key]


def _training_mean_centroids(scenes: Sequence[WholeOrganScene]) -> dict[str, np.ndarray]:
    """Mean centroid of each entity over a split: the average-organ predictor."""
    gathered: dict[str, list[np.ndarray]] = defaultdict(list)
    for scene in scenes:
        for name, centroid in scene.centroids.items():
            gathered[name].append(np.asarray(centroid, dtype=float))
    return {name: np.mean(np.stack(values), axis=0) for name, values in gathered.items()}


def step8_relation_floor(corpus_dir: str | Path) -> dict[str, Any]:
    """Compute the oracle and both relation-blind predictors on every split."""
    train = load_step8_split(corpus_dir, "train")
    mean_centroids = _training_mean_centroids(train)

    out: dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "training_scenes_used_for_mean": len(train),
        "splits": {},
        "notes": [
            "The floor reported for a split is the higher of the two blind predictors.",
            "Oracle must be 1.0; any other value means the scoring code is wrong.",
            "Computed from the corpus alone. No model, no arm, no training.",
        ],
    }
    for split in ("validation", *TEST_SPLITS):
        scenes = load_step8_split(corpus_dir, split)
        cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        oracle = score_against_measured_relations(scenes, oracle_centroids)
        canonical = score_against_measured_relations(
            scenes, lambda scene, cache=cache: _canonical_centroids(scene, cache)
        )
        average = score_against_measured_relations(scenes, lambda _, table=mean_centroids: table)
        floor = max(canonical["accuracy"], average["accuracy"])
        out["splits"][split] = {
            "scenes": len(scenes),
            "oracle": oracle,
            "blind_canonical": canonical,
            "blind_mean": average,
            "relation_blind_floor": floor,
            "discriminative_headroom": 1.0 - floor,
            "floor_source": (
                "blind_canonical" if canonical["accuracy"] >= average["accuracy"] else "blind_mean"
            ),
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 relation-blind floor.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = step8_relation_floor(args.corpus)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'split':20}{'oracle':>9}{'canonical':>11}{'mean':>9}{'floor':>9}{'headroom':>10}")
    for split, values in report["splits"].items():
        print(
            f"{split:20}{values['oracle']['accuracy']:>9.4f}"
            f"{values['blind_canonical']['accuracy']:>11.4f}"
            f"{values['blind_mean']['accuracy']:>9.4f}"
            f"{values['relation_blind_floor']:>9.4f}"
            f"{values['discriminative_headroom']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
