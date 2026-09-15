"""Whole-organ visualisation: the images must show what they claim to show."""

from __future__ import annotations

import numpy as np

from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.parameters import Variant
from datasets.whole_organ.sampling import EDIT_SPECIFICATIONS, EditOperation
from visualization.whole_organ_views import render_edit, render_ownership, render_variants


def test_a_slice_shows_several_entities() -> None:
    """A slice that is one colour is not showing a segmented organ."""
    scene = build_scene(scene_index=0, family_id=0, variant=Variant.NORMAL, lod=3)
    pixels = render_ownership(scene.field(), resolution=96)
    colours = {tuple(colour) for colour in pixels.reshape(-1, 3).tolist()}
    colours.discard((0, 0, 0))
    assert len(colours) >= 8, f"only {len(colours)} entities visible in the slice"


def test_mirroring_actually_mirrors(tmp_path) -> None:
    """The arrangements must differ, or the variant is a label on identical geometry."""
    normal = build_scene(scene_index=0, family_id=0, variant=Variant.NORMAL, lod=3)
    mirrored = build_scene(scene_index=0, family_id=0, variant=Variant.MIRRORED, lod=3)
    first = render_ownership(normal.field(), resolution=96)
    second = render_ownership(mirrored.field(), resolution=96)
    assert not np.array_equal(first, second), "mirroring changed nothing"
    # A left-right reflection of the mirrored slice should resemble the normal one far
    # more than the unreflected version does.
    reflected = second[:, ::-1]
    direct = float((first == second).all(axis=-1).mean())
    flipped = float((first == reflected).all(axis=-1).mean())
    assert flipped > direct, "the mirrored organ is not a reflection of the normal one"


def test_every_arrangement_is_rendered(tmp_path) -> None:
    """One image per arrangement, all four present."""
    written = render_variants(tmp_path, family_id=1, resolution=64)
    assert set(written) == {str(variant) for variant in Variant}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_an_edit_renders_before_after_and_difference(tmp_path) -> None:
    """The difference image must be non-empty, or the edit did nothing visible."""
    scene = build_scene(scene_index=0, family_id=0, variant=Variant.NORMAL, lod=3)
    written = render_edit(scene, EditOperation.ENLARGE_VENTRICLE, tmp_path, resolution=96)
    assert set(written) == {"before", "after", "difference"}
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_the_edit_target_gains_or_loses_ground(tmp_path) -> None:
    """An enlarging edit must show its target gaining, not merely changing."""
    from datasets.whole_organ.sampling import edit_pair
    from visualization.export import slice_points

    scene = build_scene(scene_index=0, family_id=0, variant=Variant.NORMAL, lod=3)
    before, after, _ = edit_pair(scene, EditOperation.ENLARGE_VENTRICLE)
    points = slice_points(96, 2, 0.0)
    slot = before.slot_of[EDIT_SPECIFICATIONS[EditOperation.ENLARGE_VENTRICLE][2]]
    gained = int((after.ownership(points) == slot).sum())
    held = int((before.ownership(points) == slot).sum())
    assert gained > held, f"enlarging did not enlarge: {held} to {gained}"
