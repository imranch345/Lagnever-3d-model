"""Step 10 §20: placement error resolved by depth in the hierarchy.

Change 1 can only touch entities that have a parent. On this corpus that is ten of twenty,
and the other ten are roots whose target is identical in both cells. So an aggregate that
moves tells you almost nothing on its own: it is consistent with the hierarchy working, and
equally consistent with the run having drifted for an unrelated reason.

Resolving the error by depth separates those. Depth 0 is the control built into every run —
if it moves, the difference is not the hierarchy, because those entities were predicted the
same way in both cells. Depth 1 and below are the only places Change 1 can act, and the
cascading-error prediction says the deeper an entity sits the more of its parent's error it
inherits.

This reports both, per split, for one trained model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import TEST_SPLITS, load_step8_split
from datasets.whole_organ.hierarchy import ROOT, depth_of, parent_slots
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = ["depth_table", "depth_analysis", "main"]


def depth_table(slot_of: dict[str, int], *, width: int, hierarchy: str = "spatial") -> list[int]:
    """Depth per batch slot, padded slots counted as roots."""
    parents = list(parent_slots(hierarchy, slot_of))
    parents.extend([ROOT] * (width - len(parents)))
    return list(depth_of(tuple(parents)))


@torch.no_grad()
def depth_analysis(
    model: Any,
    corpus_dir: str | Path,
    builder: WholeOrganBatchBuilder,
    *,
    splits: Sequence[str] = TEST_SPLITS,
    device: torch.device | None = None,
    batch_size: int = 8,
    hierarchy: str = "spatial",
) -> dict[str, Any]:
    """Translation, scale and rotation error bucketed by hierarchy depth.

    Placement is always inferred here. Supplying true frames would make the deeper buckets
    look like the shallow ones, because the cascade is exactly what the true frames remove.
    """
    model.eval()
    target = device or torch.device("cpu")
    out: dict[str, Any] = {"hierarchy": hierarchy, "splits": {}}
    depths: list[int] | None = None

    for split in splits:
        scenes = load_step8_split(corpus_dir, split)
        loader = WholeOrganLoader(
            scenes, builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
        )
        buckets: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: {"position": [], "scale": [], "rotation": []}
        )
        for whole in loader.epoch(0):
            batch = whole.batch.to(target)
            present = batch.entity_present.bool()
            if depths is None:
                depths = depth_table(
                    dict(builder.slot_of), width=int(present.shape[1]), hierarchy=hierarchy
                )
            # The model's own forward, with placement inferred, so the frames scored here
            # are exactly the frames the metric scores. Reaching into the frame head
            # directly would skip the composition and measure a different quantity.
            output = model(batch, use_predicted_frames=True)
            predicted = output.frames
            if predicted is None:
                raise ValueError(
                    "the model produced no frames; this arm has no frame head and cannot "
                    "be depth-analysed"
                )
            truth = batch.entity_frames
            position = (predicted[..., 0:3] - truth[..., 0:3]).norm(dim=-1)
            scale = (predicted[..., 3:6] - truth[..., 3:6]).abs().mean(dim=-1)
            rotation = rotation_angle(predicted, truth)
            for slot in range(present.shape[1]):
                mask = present[:, slot]
                if not bool(mask.any()):
                    continue
                bucket = buckets[depths[slot]]
                bucket["position"].extend(position[mask, slot].tolist())
                bucket["scale"].extend(scale[mask, slot].tolist())
                bucket["rotation"].extend(rotation[mask, slot].tolist())
        out["splits"][split] = {
            str(depth): {
                "entities_scored": len(values["position"]),
                "position_error": float(np.mean(values["position"])),
                "scale_error": float(np.mean(values["scale"])),
                "rotation_error": float(np.mean(values["rotation"])),
            }
            for depth, values in sorted(buckets.items())
        }
    out["notes"] = [
        "Depth 0 is the built-in control: those entities are roots and are predicted "
        "identically in the global and parent-relative cells.",
        "A difference at depth 0 between two cells means the run drifted, not that the "
        "hierarchy worked.",
        "Placement is inferred, never supplied; supplying frames removes the cascade "
        "the deeper buckets exist to measure.",
    ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: report the depth composition of the hierarchy itself."""
    parser = argparse.ArgumentParser(description="Step 10 depth analysis.")
    parser.add_argument("--hierarchy", default="spatial")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    depths = depth_table(
        dict(builder.slot_of), width=len(builder.slot_of), hierarchy=args.hierarchy
    )
    histogram: dict[str, int] = {}
    for depth in depths:
        histogram[str(depth)] = histogram.get(str(depth), 0) + 1
    report = {"hierarchy": args.hierarchy, "depth_histogram": histogram}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
