"""Rotation statistics for a corpus, measured the way the model will read them.

A frame stores its rotation as two basis rows, and the model recovers the matrix by
Gram-Schmidt in :func:`generation.neural.nn.geometry.frame_rotation`. Any audit that
reconstructed the matrix differently would describe a rotation the model never sees, so the
reconstruction here is that same procedure in NumPy, and a test pins the two together.

The angle reported throughout is the geodesic angle from the identity, in degrees: the
amount of rotation the frame carries. ``arccos`` is used because this is measurement rather
than optimisation — nothing here is differentiated — and it is clamped first so a matrix a
floating-point hair outside the valid range returns 0 instead of a NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "ANGLE_THRESHOLDS",
    "scene_angles",
    "basis_to_matrix",
    "geodesic_angle_degrees",
    "relative_angle_degrees",
    "summarise_angles",
]

#: Reported fractions above each of these angles, in degrees, plus the near-identity share.
ANGLE_THRESHOLDS: tuple[float, ...] = (1.0, 10.0, 30.0, 60.0)


def basis_to_matrix(rows: Sequence[float]) -> FloatArray:
    """The 3x3 rotation from a frame's six stored numbers, by the model's own Gram-Schmidt."""
    first = np.asarray(rows[0:3], dtype=np.float64)
    second = np.asarray(rows[3:6], dtype=np.float64)
    b1 = first / max(float(np.linalg.norm(first)), 1e-6)
    residual = second - float(b1 @ second) * b1
    b2 = residual / max(float(np.linalg.norm(residual)), 1e-6)
    return np.stack([b1, b2, np.cross(b1, b2)])


def geodesic_angle_degrees(rotation: FloatArray) -> float:
    """Geodesic angle between ``rotation`` and the identity, in degrees."""
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def relative_angle_degrees(first: FloatArray, second: FloatArray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    return geodesic_angle_degrees(np.asarray(first @ second.T, dtype=np.float64))


def summarise_angles(angles: Sequence[float]) -> dict[str, Any]:
    """Mean, median, spread, range and the threshold fractions of a set of angles."""
    if not angles:
        return {"count": 0}
    values = np.asarray(angles, dtype=np.float64)
    out: dict[str, Any] = {
        "count": int(values.size),
        "mean_deg": float(values.mean()),
        "median_deg": float(np.median(values)),
        "std_deg": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min_deg": float(values.min()),
        "max_deg": float(values.max()),
        "fraction_below_1_deg": float((values < 1.0).mean()),
    }
    for threshold in ANGLE_THRESHOLDS:
        out[f"fraction_above_{int(threshold)}_deg"] = float((values > threshold).mean())
    return out


def scene_angles(frames: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Geodesic angle from the identity for every entity frame in one scene."""
    return {
        entity_id: geodesic_angle_degrees(basis_to_matrix(list(frame)[6:12]))
        for entity_id, frame in frames.items()
    }
