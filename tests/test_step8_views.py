"""Step 8 views: the arrangement sweep must show a continuum, not a set of labels."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.whole_organ.arrangement import ARRANGEMENT_FIELDS
from visualization.step8_views import render_arrangement_sweep


def test_a_sweep_along_one_axis_produces_distinct_organs(tmp_path) -> None:
    """Neighbouring samples must differ, or the axis is not doing anything."""
    written = render_arrangement_sweep(tmp_path, axis="yaw", steps=4, resolution=64)
    assert len(written) == 4
    sizes = {path.stat().st_size for path in written.values()}
    assert len(sizes) > 1, "every image is identical, so the sweep changes nothing"


def test_every_sample_along_an_axis_is_a_different_organ(tmp_path) -> None:
    """Each point on the axis must give a distinct organ.

    Deliberately *not* asserting that the ends of the axis are the most different. Pixel
    difference is not monotone in rotation angle: a rotation of -0.5 and one of +0.5 are
    each far from the middle without being furthest from each other. What matters for the
    experiment is that no two samples coincide, so no finite set of labels covers the axis.
    """
    from datasets.whole_organ.arrangement import Arrangement
    from datasets.whole_organ.corpus import build_scene
    from visualization.whole_organ_views import render_ownership

    zero = {name: 0.0 for name in ARRANGEMENT_FIELDS}
    images = []
    for value in (-0.5, 0.0, 0.5):
        arrangement = Arrangement(mirror=False, transpose=False, **{**zero, "yaw": value})
        scene = build_scene(scene_index=0, family_id=0, lod=3, arrangement=arrangement)
        images.append(render_ownership(scene.field(), resolution=64))

    def difference(first: np.ndarray, second: np.ndarray) -> float:
        return float((first != second).any(axis=-1).mean())

    pairs = {
        "-0.5 vs 0.0": difference(images[0], images[1]),
        "0.0 vs +0.5": difference(images[1], images[2]),
        "-0.5 vs +0.5": difference(images[0], images[2]),
    }
    for name, value in pairs.items():
        assert value > 0.03, f"{name} are nearly identical ({value:.4f})"


def test_an_unknown_axis_is_rejected(tmp_path) -> None:
    """A typo must fail loudly rather than silently sweeping nothing."""
    with pytest.raises(ValueError, match="not an arrangement axis"):
        render_arrangement_sweep(tmp_path, axis="elevation", steps=2, resolution=32)
