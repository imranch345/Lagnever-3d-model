"""Minimal exporters for inspecting the representation.

No renderer, no plotting dependency: a PNG encoder built on ``zlib`` from the
standard library, plus OBJ and PLY writers. The purpose is to look at whether the
representation behaves as expected, not to produce pictures for anybody.

Mesh extraction lives here, in the export layer, and deliberately not in the model:
a mesh is an output of the representation and never the representation itself.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "PALETTE",
    "write_png",
    "render_slice",
    "render_ownership_slice",
    "write_point_cloud_ply",
    "write_voxel_obj",
    "colour_for_slot",
]

FloatArray = npt.NDArray[np.float64]
ByteArray = npt.NDArray[np.uint8]

PALETTE: tuple[tuple[int, int, int], ...] = (
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 210, 40), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (26, 188, 156), (52, 73, 94), (230, 126, 34),
    (149, 165, 166), (192, 57, 43), (41, 128, 185), (39, 174, 96),
)
"""A fixed qualitative palette. Slot index modulo the palette length."""


def colour_for_slot(slot: int) -> tuple[int, int, int]:
    """Deterministic colour for an entity slot."""
    if slot < 0:
        return (24, 24, 28)
    return PALETTE[slot % len(PALETTE)]


def write_png(path: str | Path, pixels: ByteArray) -> Path:
    """Write an RGB image as a PNG.

    Args:
        path: Output path.
        pixels: ``[height, width, 3]`` array of bytes.

    """
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Expected an [H, W, 3] image, got {pixels.shape}.")
    height, width, _ = pixels.shape
    raw = b"".join(
        b"\x00" + pixels[row].astype(np.uint8).tobytes() for row in range(height)
    )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def _slice_grid(resolution: int, axis: int, position: float) -> FloatArray:
    line = np.linspace(-1.0, 1.0, resolution)
    first, second = np.meshgrid(line, line, indexing="xy")
    points = np.zeros((resolution * resolution, 3), dtype=np.float64)
    axes = [index for index in range(3) if index != axis]
    points[:, axes[0]] = first.reshape(-1)
    points[:, axes[1]] = second.reshape(-1)
    points[:, axis] = position
    return points


def render_slice(
    occupancy: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | FloatArray,
    resolution: int,
    *,
    background: tuple[int, int, int] = (18, 18, 22),
    foreground: tuple[int, int, int] = (220, 80, 80),
) -> ByteArray:
    """Turn a flat slice of occupancy values into an image."""
    values = np.asarray(occupancy, dtype=np.float64).reshape(resolution, resolution)
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    image[:] = np.array(background, dtype=np.uint8)
    mask = values > 0.5
    image[mask] = np.array(foreground, dtype=np.uint8)
    return np.flipud(image)


def render_ownership_slice(
    owners: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    resolution: int,
    *,
    background: tuple[int, int, int] = (18, 18, 22),
) -> ByteArray:
    """Colour a slice by which entity owns each point."""
    values = np.asarray(owners).reshape(resolution, resolution)
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    image[:] = np.array(background, dtype=np.uint8)
    for slot in np.unique(values):
        if slot < 0:
            continue
        image[values == slot] = np.array(colour_for_slot(int(slot)), dtype=np.uint8)
    return np.flipud(image)


def render_difference_slice(
    before: FloatArray, after: FloatArray, resolution: int
) -> ByteArray:
    """Visualise where two occupancy fields disagree.

    Grey where both agree on occupied, dark where both agree on empty, red where the
    edit added, blue where it removed. Used for the untouched-entity drift picture.
    """
    first = np.asarray(before).reshape(resolution, resolution)
    second = np.asarray(after).reshape(resolution, resolution)
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    image[:] = np.array((18, 18, 22), dtype=np.uint8)
    both = (first > 0.5) & (second > 0.5)
    added = (first <= 0.5) & (second > 0.5)
    removed = (first > 0.5) & (second <= 0.5)
    image[both] = np.array((120, 120, 128), dtype=np.uint8)
    image[added] = np.array((228, 60, 60), dtype=np.uint8)
    image[removed] = np.array((60, 120, 228), dtype=np.uint8)
    return np.flipud(image)


def slice_points(resolution: int, axis: int, position: float) -> FloatArray:
    """Query points for one orthographic slice through the unit cube."""
    return _slice_grid(resolution, axis, position)


def write_point_cloud_ply(
    path: str | Path, points: FloatArray, colours: ByteArray | None = None
) -> Path:
    """Write a point cloud as an ASCII PLY file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colours is not None:
        lines.extend(["property uchar red", "property uchar green", "property uchar blue"])
    lines.append("end_header")
    for index in range(points.shape[0]):
        x, y, z = points[index]
        if colours is not None:
            r, g, b = colours[index]
            lines.append(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}")
        else:
            lines.append(f"{x:.5f} {y:.5f} {z:.5f}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


_CUBE_FACES: tuple[tuple[tuple[int, int, int], tuple[int, ...]], ...] = (
    ((-1, 0, 0), (0, 4, 6, 2)),
    ((1, 0, 0), (1, 3, 7, 5)),
    ((0, -1, 0), (0, 1, 5, 4)),
    ((0, 1, 0), (2, 6, 7, 3)),
    ((0, 0, -1), (0, 2, 3, 1)),
    ((0, 0, 1), (4, 5, 7, 6)),
)


def write_voxel_obj(
    path: str | Path,
    occupancy: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> Path:
    """Write the surface of an occupancy grid as an OBJ mesh.

    A voxel-boundary mesh, not marching cubes: blocky but exact with respect to the
    grid, dependency-free, and adequate for inspection. It is an export artefact and
    has no path back into the model.
    """
    grid = np.asarray(occupancy, dtype=bool)
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3D occupancy grid, got {grid.shape}.")
    resolution = grid.shape[0]
    low, high = bounds
    step = (high - low) / resolution
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    index_of: dict[tuple[int, int, int], int] = {}

    def vertex(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in index_of:
            vertices.append((low + ix * step, low + iy * step, low + iz * step))
            index_of[key] = len(vertices)
        return index_of[key]

    occupied = np.argwhere(grid)
    for x, y, z in occupied:
        corners = [
            (x, y, z), (x + 1, y, z), (x, y + 1, z), (x + 1, y + 1, z),
            (x, y, z + 1), (x + 1, y, z + 1), (x, y + 1, z + 1), (x + 1, y + 1, z + 1),
        ]
        for (dx, dy, dz), corner_indices in _CUBE_FACES:
            nx, ny, nz = x + dx, y + dy, z + dz
            inside = (
                0 <= nx < resolution and 0 <= ny < resolution and 0 <= nz < resolution
            )
            if inside and grid[nx, ny, nz]:
                continue
            face = tuple(vertex(*corners[corner]) for corner in corner_indices)
            faces.append(face)  # type: ignore[arg-type]

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Lagnav voxel-boundary export, resolution {resolution}"]
    lines.extend(f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices)
    lines.extend("f " + " ".join(str(index) for index in face) for face in faces)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def summarise_exports(paths: Mapping[str, Sequence[Path]]) -> dict[str, int]:
    """Count exported artefacts by category, for the report."""
    return {name: len(items) for name, items in paths.items()}
