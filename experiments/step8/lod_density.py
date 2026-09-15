"""Per-level occupancy target density, the number that makes level IoU comparable.

A finer level of detail exposes more entities, so its occupancy target covers more of the
space. A model that simply calls every point occupied scores an IoU equal to the target
density, which rises with the level and produces a large positive "detail gain" while
adding nothing at all.

This measures the densities from the corpus so the corrected quantity can be computed:

    iou_over_trivial = (iou - density) / (1 - density)

which is 0.0 for the trivial predictor and 1.0 for a perfect decode, at every level.

Computed from the corpus alone. No model, no arm, no training.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import STEP8_SPLITS, load_step8_split
from datasets.whole_organ.sampling import sample_scene

__all__ = ["level_densities", "corrected_iou", "main"]


def corrected_iou(iou: float, density: float) -> float:
    """Share of the available headroom a decode takes at one level."""
    headroom = 1.0 - density
    return float((iou - density) / headroom) if headroom > 1e-9 else float("nan")


def level_densities(corpus_dir: str | Path, split: str, *, limit: int = 60) -> dict[str, Any]:
    """Mean share of sampled points each level's target calls occupied."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    scenes = load_step8_split(corpus_dir, split, limit=limit)
    shares: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for scene in scenes:
        sampled = sample_scene(scene, slot_of)
        for level in (1, 2, 3):
            shares[level].append(float(sampled.lod_occupancy[level].mean()))
    return {
        "split": split,
        "scenes_measured": len(scenes),
        "density": {str(level): float(np.mean(values)) for level, values in shares.items()},
        "trivial_predictor_iou": {
            str(level): float(np.mean(values)) for level, values in shares.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 level-of-detail target densities.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(STEP8_SPLITS))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    splits: dict[str, dict[str, Any]] = {
        split: level_densities(args.corpus, split, limit=args.limit) for split in args.splits
    }
    report: dict[str, Any] = {
        "corpus_dir": str(args.corpus),
        "splits": splits,
        "note": (
            "A trivial predictor that calls every point occupied scores an IoU equal to "
            "the density. Use (iou - density) / (1 - density) to compare levels."
        ),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{'split':20}{'level 1':>10}{'level 2':>10}{'level 3':>10}")
    for split, values in splits.items():
        row = "".join(f"{values['density'][str(level)]:>10.4f}" for level in (1, 2, 3))
        print(f"{split:20}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
