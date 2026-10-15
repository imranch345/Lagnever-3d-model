"""Level 1: choosing a usable subject, checked on phantoms with known defects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("trimesh")
pytest.importorskip("skimage")
pytest.importorskip("fast_simplification")

from assets.meshes import build_subject  # noqa: E402
from assets.phantom import LAYOUT, ellipsoid, write_phantom  # noqa: E402
from assets.quality import assess, choose  # noqa: E402


def _build(root: Path, name: str, **kwargs: Any) -> dict[str, Any]:
    subject = write_phantom(root / "raw", name=name, **kwargs)
    return build_subject(subject, root / "built" / name)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, dict[str, Any]]:
    """A clean phantom and three with one defect each."""
    root = tmp_path_factory.mktemp("quality")
    truncated = dict(LAYOUT)
    truncated["lung_upper_lobe_left"] = ellipsoid((-60, 0, 55), (25, 35, 40))
    swapped = dict(LAYOUT)
    swapped["lung_upper_lobe_left"], swapped["lung_upper_lobe_right"] = (
        LAYOUT["lung_upper_lobe_right"], LAYOUT["lung_upper_lobe_left"],
    )
    return {
        "clean": _build(root, "clean"),
        "truncated": _build(root, "truncated", layout=truncated),
        "swapped": _build(root, "swapped", layout=swapped),
        "no_trachea": _build(root, "no_trachea", empty=("trachea",)),
    }


def test_a_clean_subject_is_usable(built) -> None:
    """No hard findings, no soft ones."""
    quality = assess(built["clean"])
    assert quality.usable, quality.hard
    assert quality.soft == []


def test_a_lung_cut_by_the_field_of_view_is_not_usable(built) -> None:
    """The left upper lobe reaches the top slice of the scan."""
    quality = assess(built["truncated"])
    assert not quality.usable
    assert any("cut by the field of view (superior)" in h for h in quality.hard)


def test_swapped_sides_are_not_usable(built) -> None:
    """A left lobe built from a right-sided shape is caught by the side check."""
    quality = assess(built["swapped"])
    assert not quality.usable
    assert any("wrong side" in h for h in quality.hard)


def test_a_missing_structure_is_not_usable(built) -> None:
    """An empty trachea label leaves the trachea without a mesh."""
    quality = assess(built["no_trachea"])
    assert not quality.usable
    assert any("missing" in h for h in quality.hard)


def test_the_trachea_may_leave_the_scan(built) -> None:
    """It runs into the neck by nature; touching the top face is not a defect."""
    manifest = dict(built["clean"])
    entities = {k: dict(v) for k, v in manifest["entities"].items()}
    trachea = dict(entities["respiratory.trachea"])
    trachea["quality"] = {**trachea["quality"], "touches_boundary": ["superior"]}
    entities["respiratory.trachea"] = trachea
    manifest["entities"] = entities
    assert assess(manifest).usable


def test_choose_picks_the_clean_subject(built) -> None:
    """Among four, only one is usable."""
    chosen = choose([assess(m) for m in built.values()])
    assert chosen is not None and chosen.subject == "clean"


def test_choose_returns_none_when_nothing_is_usable(built) -> None:
    """Better no subject than a broken one."""
    assert choose([assess(built["truncated"]), assess(built["swapped"])]) is None


def test_build_all_records_a_broken_subject_and_builds_the_rest(tmp_path) -> None:
    """One unreadable scan must not stop the batch; it is reported with its error."""
    from assets.quality import build_all

    raw = tmp_path / "raw"
    write_phantom(raw, name="s0001")
    broken = write_phantom(raw, name="s0002")
    (broken / "segmentations" / "trachea.nii.gz").write_bytes(b"not a nifti")
    failures = build_all(raw, tmp_path / "scenes")
    assert list(failures) == ["s0002"]
    assert (tmp_path / "scenes" / "s0001" / "manifest.json").is_file()
