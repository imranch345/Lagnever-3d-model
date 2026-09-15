"""Render what Step 8 measures: placement inferred against placement supplied.

Two pictures are worth more than the tables they summarise.

The first shows the same organ family at several points along one continuous arrangement
axis, so a reader can see that the arrangement really is continuous rather than one of a
few labels.

The second shows a trained model's predicted ownership beside the truth, in **both**
placement conditions. Supplying the frames hands the model the arrangement; predicting them
does not. If the two predicted images look alike while the two truths do not, the model is
producing one average organ, which is what the relation-blind floor is there to detect.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from datasets.whole_organ.arrangement import ARRANGEMENT_FIELDS, Arrangement
from datasets.whole_organ.corpus import build_scene
from visualization.export import colour_for_slot, slice_points, write_png
from visualization.whole_organ_views import render_ownership

__all__ = ["render_arrangement_sweep", "render_placement_comparison", "main"]


def render_arrangement_sweep(
    output_dir: Path,
    *,
    axis: str = "yaw",
    family_id: int = 0,
    steps: int = 5,
    low: float = -0.5,
    high: float = 0.5,
    resolution: int = 160,
) -> dict[str, Path]:
    """One image per point along a continuous arrangement axis.

    The point of the picture: these are not five labels, they are five samples from an
    interval, and the corpus draws anywhere in it.
    """
    if axis not in ARRANGEMENT_FIELDS:
        raise ValueError(
            f"{axis!r} is not an arrangement axis; expected one of {ARRANGEMENT_FIELDS}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    zero = {name: 0.0 for name in ARRANGEMENT_FIELDS}
    for index, value in enumerate(np.linspace(low, high, steps)):
        arrangement = Arrangement(mirror=False, transpose=False, **{**zero, axis: float(value)})
        scene = build_scene(scene_index=0, family_id=family_id, lod=3, arrangement=arrangement)
        pixels = render_ownership(scene.field(), resolution=resolution)
        name = f"{axis}-{index:02d}-{value:+.3f}"
        written[name] = write_png(output_dir / f"{name}.png", pixels)
    return written


def render_placement_comparison(
    model: Any,
    builder: Any,
    output_dir: Path,
    *,
    family_id: int = 0,
    arrangements: Sequence[Arrangement] | None = None,
    resolution: int = 128,
) -> dict[str, Path]:
    """Truth, supplied-placement prediction and inferred-placement prediction, side by side.

    Imported lazily and typed loosely so this module stays usable without PyTorch.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    zero = {name: 0.0 for name in ARRANGEMENT_FIELDS}
    if arrangements is None:
        arrangements = [
            Arrangement(mirror=False, transpose=False, **{**zero, "yaw": -0.4}),
            Arrangement(mirror=False, transpose=False, **{**zero, "yaw": 0.4}),
            Arrangement(mirror=True, transpose=False, **zero),
        ]

    points = slice_points(resolution, 2, 0.0)
    probe = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    written: dict[str, Path] = {}
    for index, arrangement in enumerate(arrangements):
        scene = build_scene(scene_index=0, family_id=family_id, lod=3, arrangement=arrangement)
        batch = builder.build([scene]).batch
        shared = dataclass_replace(batch, scene_points=probe)
        written[f"{index}-truth"] = write_png(
            output_dir / f"{index}-truth.png",
            render_ownership(scene.field(), resolution=resolution),
        )
        for condition, predicted in (("supplied", False), ("inferred", True)):
            with torch.no_grad():
                output = model(shared, use_predicted_frames=predicted)
            labels = output.part_logits.argmax(dim=-1)[0].numpy()
            background = int(output.part_logits.shape[-1]) - 1
            pixels = np.zeros((resolution, resolution, 3), dtype=np.uint8)
            flat = pixels.reshape(-1, 3)
            for position, label in enumerate(labels):
                if int(label) == background:
                    continue
                flat[position] = colour_for_slot(int(label))
            written[f"{index}-{condition}"] = write_png(
                output_dir / f"{index}-{condition}.png", pixels
            )
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Render Step 8 views.")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step8-views"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--axis", default="yaw")
    parser.add_argument("--resolution", type=int, default=144)
    args = parser.parse_args(argv)

    sweep = render_arrangement_sweep(args.out / "sweep", axis=args.axis, resolution=args.resolution)
    print(f"[views] {len(sweep)} images along the {args.axis} axis")

    if args.checkpoint is not None:
        import torch

        from awr.config import load_domain_config
        from awr.ontology import load_ontology
        from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
        from training.loop import build_model

        domain = load_domain_config()
        ontology = load_ontology(
            domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
        )
        builder = WholeOrganBatchBuilder(ontology, domain)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        arm = str(state["config"]["arm"])
        model, _ = build_model(cast(Any, arm), cast(Any, builder), text_features=65)
        model.load_state_dict(state["model"])
        model.eval()
        comparison = render_placement_comparison(
            model, builder, args.out / "placement", resolution=args.resolution
        )
        print(f"[views] {len(comparison)} placement-comparison images for {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
