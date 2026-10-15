"""A synthetic phantom subject: every Level 1 label as a simple shape at a chosen position.

**Not anatomy.** Spheres, ellipsoids and cylinders placed roughly where the structures sit,
so the mesh pipeline can be tested against known answers and the viewer can be developed
before any real data is downloaded. Written in the TotalSegmentator v1 file layout.

Positions are RAS+ millimetres with the heart at the origin; the subject's left is negative x.
Lobes of one lung touch, and the heart's parts touch, as in real anatomy, so each organ
is one connected piece. Lungs span z from about -50 to 62, and the aorta and IVC run below
them so the thorax crop has something to remove. Needs the ``assets`` extra.

    python -m assets.phantom --out datasets/raw/phantom
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.typing as npt

__all__ = ["EXTENT", "LAYOUT", "SPACING", "phantom_affine", "phantom_grid", "write_phantom"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Shape = Callable[[FloatArray, FloatArray, FloatArray], BoolArray]

SPACING = 2.0
EXTENT = {"x": (-90.0, 90.0), "y": (-60.0, 60.0), "z": (-80.0, 80.0)}


def sphere(c: tuple[float, float, float], r: float) -> Shape:
    """A ball of radius ``r`` millimetres."""
    return lambda x, y, z: np.asarray(
        (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r**2
    )


def ellipsoid(c: tuple[float, float, float], r: tuple[float, float, float]) -> Shape:
    """An axis-aligned ellipsoid with semi-axes ``r``."""
    return lambda x, y, z: np.asarray(
        ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0
    )


def cylinder(x0: float, y0: float, r: float, z0: float, z1: float) -> Shape:
    """A vertical (superior-inferior) cylinder."""
    return lambda x, y, z: np.asarray(
        ((x - x0) ** 2 + (y - y0) ** 2 <= r**2) & (z >= z0) & (z <= z1)
    )


def _shell(c: tuple[float, float, float], outer: float, inner: float) -> Shape:
    return lambda x, y, z: sphere(c, outer)(x, y, z) & ~sphere(c, inner)(x, y, z)


LAYOUT: Mapping[str, Shape] = {
    "heart_ventricle_left": sphere((-15, 0, -10), 12),
    "heart_ventricle_right": sphere((10, 10, -10), 12),
    "heart_atrium_left": sphere((-12, -15, 10), 10),
    "heart_atrium_right": sphere((12, -3, 6), 12),
    "heart_myocardium": _shell((-15, 0, -10), 16, 12),
    "pulmonary_artery": sphere((0, 15, 22), 8),
    "aorta": cylinder(-10, -22, 8, -75, 60),
    "inferior_vena_cava": cylinder(20, -15, 7, -78, -5),
    "lung_upper_lobe_left": ellipsoid((-60, 0, 30), (25, 35, 30)),
    "lung_lower_lobe_left": ellipsoid((-60, 0, -22), (25, 35, 25)),
    "lung_upper_lobe_right": ellipsoid((60, 0, 40), (25, 35, 22)),
    "lung_middle_lobe_right": ellipsoid((60, 20, 8), (20, 15, 12)),
    "lung_lower_lobe_right": ellipsoid((60, 5, -22), (25, 30, 22)),
    "trachea": cylinder(0, -10, 8, 50, 75),
}


def phantom_affine(orientation: str = "RAS") -> FloatArray:
    """A RAS+-stored grid, or an LPS-stored one -- the common DICOM-derived orientation."""
    lo = {k: v[0] for k, v in EXTENT.items()}
    hi = {k: v[1] for k, v in EXTENT.items()}
    if orientation == "RAS":
        diagonal, offset = (SPACING, SPACING, SPACING), (lo["x"], lo["y"], lo["z"])
    elif orientation == "LPS":
        diagonal, offset = (-SPACING, -SPACING, SPACING), (hi["x"], hi["y"], lo["z"])
    else:
        raise ValueError(f"unknown orientation {orientation!r}")
    affine = np.eye(4)
    affine[:3, :3] = np.diag(diagonal)
    affine[:3, 3] = offset
    return affine


def phantom_grid(
    affine: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[int, int, int]]:
    """World coordinates of every voxel of a grid covering ``EXTENT`` under ``affine``."""
    nx, ny, nz = (int((hi - lo) / SPACING) for lo, hi in EXTENT.values())
    ijk = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"), -1)
    world = nib.affines.apply_affine(affine, ijk.reshape(-1, 3)).reshape(nx, ny, nz, 3)
    return world[..., 0], world[..., 1], world[..., 2], (nx, ny, nz)


def write_phantom(
    root: Path,
    orientation: str = "RAS",
    *,
    empty: Sequence[str] = (),
    layout: Mapping[str, Shape] | None = None,
    name: str | None = None,
) -> Path:
    """Write one phantom subject in TotalSegmentator v1 layout; return its folder."""
    subject = root / (name or f"phantom_{orientation}")
    (subject / "segmentations").mkdir(parents=True, exist_ok=True)
    affine = phantom_affine(orientation)
    x, y, z, _ = phantom_grid(affine)
    for label, shape in (layout or LAYOUT).items():
        mask = np.zeros(x.shape, dtype=bool) if label in empty else shape(x, y, z)
        image = nib.Nifti1Image(mask.astype(np.uint8), affine)
        nib.save(image, str(subject / "segmentations" / f"{label}.nii.gz"))
    return subject


def main(argv: Sequence[str] | None = None) -> int:
    """Write a phantom subject."""
    parser = argparse.ArgumentParser(description="Write a synthetic phantom subject.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orientation", choices=("RAS", "LPS"), default="RAS")
    args = parser.parse_args(argv)
    print(write_phantom(args.out, args.orientation, name="phantom"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
