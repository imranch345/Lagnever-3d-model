"""Render whole-organ scenes, arrangements and edits as ownership slices.

Three things are worth looking at directly rather than through a metric:

* that the four **arrangements** really do differ, and differ in the way their names
  claim, so the variant is not a label attached to identical geometry;
* that every entity, wall layers included, **owns a region**, which is what Step 6's
  corpus failed at;
* that an **edit** changes the entity it names and leaves the entities the generator
  says it did not touch.

Each image is a slice through the organ, coloured by which entity owns each point. The
colours come from the shared slot palette, so the same entity is the same colour in every
image and two images can be compared by eye.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from datasets.whole_organ.corpus import WholeOrganScene, build_scene
from datasets.whole_organ.field import WholeOrganField
from datasets.whole_organ.parameters import Variant
from datasets.whole_organ.sampling import EDIT_SPECIFICATIONS, EditOperation, edit_pair
from visualization.export import colour_for_slot, slice_points, write_png

__all__ = ["render_ownership", "render_variants", "render_edit", "main"]


def render_ownership(
    field: WholeOrganField, *, resolution: int = 192, axis: int = 2, position: float = 0.0
) -> np.ndarray:
    """One slice through an organ, coloured by owning entity."""
    points = slice_points(resolution, axis, position)
    labels = field.ownership(points)
    pixels = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    flat = pixels.reshape(-1, 3)
    for index, label in enumerate(labels):
        if label < 0:
            continue
        flat[index] = colour_for_slot(int(label))
    return pixels


def render_variants(
    output_dir: Path, *, family_id: int = 0, resolution: int = 192
) -> dict[str, Path]:
    """One image per arrangement, from one family, so the four can be compared.

    Presence and text are identical across these four by construction. Anything that
    differs between the images differs only because the arrangement does.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for variant in Variant:
        scene = build_scene(scene_index=0, family_id=family_id, variant=variant, lod=3)
        pixels = render_ownership(scene.field(), resolution=resolution)
        written[str(variant)] = write_png(output_dir / f"variant-{variant}.png", pixels)
    return written


def render_edit(
    scene: WholeOrganScene, operation: EditOperation, output_dir: Path, *, resolution: int = 192
) -> dict[str, Path]:
    """Before, after, and the points whose owner changed.

    The difference image is the honest one: it shows that a targeted edit has genuine
    non-local consequences, because valves sit on the surfaces that move and septa are
    defined by the tissue between two cavities.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    before, after, _ = edit_pair(scene, operation)
    points = slice_points(resolution, 2, 0.0)
    before_labels = before.ownership(points)
    after_labels = after.ownership(points)

    target = EDIT_SPECIFICATIONS[operation][2]
    target_slot = before.slot_of.get(target, -1)

    difference = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    flat = difference.reshape(-1, 3)
    for index, (first, second) in enumerate(zip(before_labels, after_labels, strict=True)):
        if first == second:
            continue
        # Green where the edited entity gained ground, red where it lost it, grey where
        # some other pair of entities swapped, which is the coupling made visible.
        if second == target_slot:
            flat[index] = (60, 200, 90)
        elif first == target_slot:
            flat[index] = (210, 70, 70)
        else:
            flat[index] = (120, 120, 120)

    name = str(operation)
    return {
        "before": write_png(
            output_dir / f"{name}-before.png", render_ownership(before, resolution=resolution)
        ),
        "after": write_png(
            output_dir / f"{name}-after.png", render_ownership(after, resolution=resolution)
        ),
        "difference": write_png(output_dir / f"{name}-difference.png", difference),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Render whole-organ views.")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step7-views"))
    parser.add_argument("--family", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=192)
    parser.add_argument(
        "--operations",
        nargs="*",
        default=["enlarge_ventricle", "thicken_valve", "thicken_lining"],
    )
    args = parser.parse_args(argv)

    variants = render_variants(
        args.out / "variants", family_id=args.family, resolution=args.resolution
    )
    print(f"[views] {len(variants)} arrangement images")

    scene = build_scene(scene_index=0, family_id=args.family, variant=Variant.NORMAL, lod=3)
    total = 0
    for name in args.operations:
        written = render_edit(
            scene, EditOperation(name), args.out / "edits", resolution=args.resolution
        )
        total += len(written)
        print(f"[views] {name}: {len(written)} images")
    print(f"[views] wrote {len(variants) + total} images under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def render_prediction_comparison(
    model: object,
    builder: object,
    output_dir: Path,
    *,
    family_id: int = 0,
    resolution: int = 128,
    use_predicted_frames: bool = False,
) -> dict[str, Path]:
    """Predicted ownership beside the truth, one pair per arrangement.

    This is the picture behind the per-arrangement table. If a model cannot read the
    relationship graph, its four predictions should look alike while the four truths do
    not, and that is visible here in a way a column of numbers is not.

    Imported lazily and typed loosely so this module stays usable without PyTorch.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for variant in Variant:
        scene = build_scene(scene_index=0, family_id=family_id, variant=variant, lod=3)
        whole = builder.build([scene])  # type: ignore[attr-defined]
        batch = whole.batch
        points = slice_points(resolution, 2, 0.0)
        probe = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(  # type: ignore[operator]
                type(batch)(
                    **{
                        **{
                            field: getattr(batch, field)
                            for field in batch.__dataclass_fields__
                        },
                        "scene_points": probe,
                    }
                ),
                use_predicted_frames=use_predicted_frames,
            )
        predicted = output.part_logits.argmax(dim=-1)[0].numpy()
        background = int(output.part_logits.shape[-1]) - 1

        pixels = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        flat = pixels.reshape(-1, 3)
        for index, label in enumerate(predicted):
            if int(label) == background:
                continue
            flat[index] = colour_for_slot(int(label))
        written[f"{variant}-predicted"] = write_png(
            output_dir / f"{variant}-predicted.png", pixels
        )
        written[f"{variant}-truth"] = write_png(
            output_dir / f"{variant}-truth.png",
            render_ownership(scene.field(), resolution=resolution),
        )
    return written
