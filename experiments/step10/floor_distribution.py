"""Step 10 Change 2: the placement-blind floor's own error distribution, not only its mean.

A model can sit above the floor on the mean while beating it on most entities, because a
long tail drags a mean and not a median. Deciding whether that is a real advantage needs the
floor's distribution too, measured the same way on the same entities — otherwise a model's
median is being compared with a lookup table's mean, which is not a comparison.

Fitted on ``train`` only, scored through the evaluation loader, exactly as the floor is.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import TEST_SPLITS
from experiments.step10.placement_floor import _batches, _fit
from experiments.step10.rotation_metrics import DEGREES
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

__all__ = ["floor_distribution", "main"]


def floor_distribution(
    corpus_dir: str | Path, *, splits: Sequence[str] = TEST_SPLITS
) -> dict[str, Any]:
    """Per-entity rotation, position and scale errors of the identity-only lookup."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    frames, _ = next(_batches(corpus_dir, "train", builder))
    placement = HierarchicalPlacement(dict(builder.slot_of), slots=frames.shape[1])
    table = _fit(corpus_dir, builder, placement, parent_relative=False)

    out: dict[str, Any] = {"corpus_dir": str(corpus_dir), "splits": {}}
    for split in splits:
        rotation: list[np.ndarray] = []
        position: list[np.ndarray] = []
        for batch_frames, present in _batches(corpus_dir, split, builder):
            guess = torch.zeros_like(batch_frames)
            for slot, value in table.items():
                guess[:, slot] = value
            rotation.append(rotation_angle(guess, batch_frames)[present].numpy())
            position.append(
                (guess[..., 0:3] - batch_frames[..., 0:3]).norm(dim=-1)[present].numpy()
            )
        angles = np.concatenate(rotation) * DEGREES
        distances = np.concatenate(position)
        out["splits"][split] = {
            "entities_scored": int(angles.size),
            "rotation": {
                "mean_deg": float(angles.mean()),
                "median_deg": float(np.median(angles)),
                "std_deg": float(angles.std(ddof=1)),
                "max_deg": float(angles.max()),
                "within_5_deg": float((angles <= 5.0).mean()),
                "within_10_deg": float((angles <= 10.0).mean()),
                "within_20_deg": float((angles <= 20.0).mean()),
            },
            "position": {
                "mean": float(distances.mean()),
                "median": float(np.median(distances)),
            },
        }
    out["notes"] = [
        "The floor is a lookup keyed on entity identity: no relations, hierarchy or context.",
        "Reported so a model's median can be read against the floor's median, not its mean.",
    ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="The floor's error distribution.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step10-change2/floor_distribution.json"),
    )
    args = parser.parse_args(argv)
    report = floor_distribution(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, entry in report["splits"].items():
        turn = entry["rotation"]
        print(
            f"{split:<18} rotation mean {turn['mean_deg']:6.2f}  median "
            f"{turn['median_deg']:6.2f}  within10 {turn['within_10_deg']:.2f}  "
            f"within20 {turn['within_20_deg']:.2f}"
        )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
