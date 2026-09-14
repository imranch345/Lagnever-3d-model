"""Exporters and research views."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec
from visualization.export import (
    PALETTE,
    colour_for_slot,
    render_difference_slice,
    render_ownership_slice,
    render_slice,
    slice_points,
    write_png,
    write_point_cloud_ply,
    write_voxel_obj,
)
from visualization.render import render_ground_truth_views


def test_png_is_well_formed(tmp_path: Path) -> None:
    """The dependency-free encoder writes a valid PNG with correct dimensions."""
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)
    pixels[2:5, 3:7] = np.array([200, 40, 40], dtype=np.uint8)
    path = write_png(tmp_path / "image.png", pixels)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, depth, colour = struct.unpack(">2I2B", data[16:26])
    assert (width, height, depth, colour) == (12, 8, 8, 2)
    start = data.index(b"IDAT") + 4
    end = data.index(b"IEND") - 8
    raw = zlib.decompress(data[start:end])
    assert len(raw) == height * (width * 3 + 1)


def test_png_rejects_a_bad_shape(tmp_path: Path) -> None:
    """A greyscale array is refused rather than silently reinterpreted."""
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        write_png(tmp_path / "bad.png", np.zeros((4, 4), dtype=np.uint8))


def test_slices_render(tmp_path: Path) -> None:
    """Occupancy, ownership and difference slices all produce images."""
    points = slice_points(32, 2, 0.0)
    occupancy = (np.linalg.norm(points[:, :2], axis=1) < 0.5).astype(float)
    other = (np.linalg.norm(points[:, :2], axis=1) < 0.7).astype(float)
    owners = np.where(occupancy > 0.5, 3, -1).astype(np.int64)
    assert render_slice(occupancy, 32).shape == (32, 32, 3)
    assert render_ownership_slice(owners, 32).shape == (32, 32, 3)
    difference = render_difference_slice(occupancy, other, 32)
    assert difference.shape == (32, 32, 3)
    assert (difference == np.array([228, 60, 60], dtype=np.uint8)).all(axis=-1).any()


def test_colours_are_stable() -> None:
    """Entity colours are deterministic and background is distinct."""
    assert colour_for_slot(0) == PALETTE[0]
    assert colour_for_slot(len(PALETTE)) == PALETTE[0]
    assert colour_for_slot(-1) not in PALETTE


def test_voxel_obj_has_the_expected_face_count(tmp_path: Path) -> None:
    """A single voxel exports six quadrilateral faces."""
    grid = np.zeros((4, 4, 4), dtype=bool)
    grid[1, 1, 1] = True
    path = write_voxel_obj(tmp_path / "voxel.obj", grid)
    text = path.read_text(encoding="utf-8")
    assert sum(1 for line in text.splitlines() if line.startswith("f ")) == 6
    assert sum(1 for line in text.splitlines() if line.startswith("v ")) == 8


def test_interior_faces_are_not_written(tmp_path: Path) -> None:
    """Two adjacent voxels share a face, which the exporter drops."""
    grid = np.zeros((4, 4, 4), dtype=bool)
    grid[1, 1, 1] = True
    grid[2, 1, 1] = True
    text = write_voxel_obj(tmp_path / "pair.obj", grid).read_text(encoding="utf-8")
    assert sum(1 for line in text.splitlines() if line.startswith("f ")) == 10


def test_point_cloud_ply(tmp_path: Path) -> None:
    """The PLY writer produces a readable header and the right vertex count."""
    points = np.random.default_rng(0).uniform(-1.0, 1.0, size=(16, 3))
    text = write_point_cloud_ply(tmp_path / "cloud.ply", points).read_text(encoding="utf-8")
    assert text.startswith("ply")
    assert "element vertex 16" in text
    assert len(text.strip().splitlines()) == 16 + 7  # seven header lines


def test_ground_truth_views_cover_the_required_set(
    scene_specs: list[SceneSpec], ontology: AnatomyOntology, tmp_path: Path
) -> None:
    """All eight views the brief asks for are produced."""
    views = render_ground_truth_views(
        scene_specs[0], ontology, tmp_path / "views", resolution=48, grid_resolution=16
    )
    names = {path.name for path in views.files}
    for prefix in ("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_"):
        assert any(name.startswith(prefix) for name in names), prefix
    assert "whole_heart.obj" in names
    assert "surface_points.ply" in names
    assert all(path.stat().st_size > 0 for path in views.files)
