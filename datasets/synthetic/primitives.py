"""Parametric primitives for the tier-0 synthetic heart corpus.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED.

These are the geometric building blocks of the controlled testbed: each anatomical
entity is represented by one analytic primitive whose occupancy, surface samples and
canonical frame are all exact. That exactness is the point. The first experiment
measures whether a structured representation gives better part control and better
relationship consistency, and both need ground truth that is known rather than
annotated.

Coordinate convention, used everywhere downstream:

* ``+x`` is the anatomical left of the subject,
* ``+y`` is superior,
* ``+z`` is anterior,
* the whole organ fits inside the unit cube ``[-1, 1]^3``, matching the
  ``query_points`` contract from Step 5.

These shapes are not anatomy. An ellipsoid is not a ventricle. They are a
controlled stand-in with known structure, and no result obtained on them may be
described as anatomical accuracy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "PrimitiveKind",
    "Primitive",
    "Ellipsoid",
    "Capsule",
    "Shell",
    "rotation_from_euler",
    "rotation_to_6d",
    "build_primitive",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class PrimitiveKind(StrEnum):
    """Which analytic shape family a primitive belongs to."""

    ELLIPSOID = "ellipsoid"
    CAPSULE = "capsule"
    SHELL = "shell"


def rotation_from_euler(yaw: float, pitch: float, roll: float) -> FloatArray:
    """Rotation matrix from intrinsic yaw, pitch and roll in radians."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def rotation_to_6d(rotation: FloatArray) -> FloatArray:
    """First two rows of a rotation matrix, the 6D parameterisation used by the model."""
    return np.asarray(rotation[:2].reshape(6), dtype=np.float64)


class Primitive(ABC):
    """One analytic shape with exact occupancy, surface samples and a canonical frame."""

    kind: PrimitiveKind

    @abstractmethod
    def occupancy(self, points: FloatArray) -> BoolArray:
        """Return a boolean mask of which query points lie inside the shape."""

    @abstractmethod
    def sample_surface(self, count: int, rng: np.random.Generator) -> FloatArray:
        """Sample points on the shape's surface."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise the parameters to plain data."""

    @property
    @abstractmethod
    def centre(self) -> FloatArray:
        """Centroid of the shape, used for spatial relationship checks."""

    @property
    @abstractmethod
    def extent(self) -> FloatArray:
        """Half-extents along the shape's own axes, used for the frame target."""

    @property
    @abstractmethod
    def rotation(self) -> FloatArray:
        """Rotation of the shape's local frame into scene space."""

    def frame(self) -> FloatArray:
        """Canonical frame as 12 channels: translation, log-scale, 6D rotation."""
        scale = np.maximum(self.extent, 1e-4)
        return np.concatenate(
            [self.centre, np.log(scale), rotation_to_6d(self.rotation)]
        ).astype(np.float64)

    def bounding_radius(self) -> float:
        """Radius of a sphere around the shape, for cheap distance rejection."""
        return float(np.linalg.norm(self.extent))

    def distance_to(self, other: Primitive, rng: np.random.Generator, samples: int = 64) -> float:
        """Approximate minimum surface-to-surface distance, by sampling.

        Approximate on purpose: the adjacency checks it feeds are themselves
        thresholded, and an exact computation for every primitive pair would
        dominate corpus generation time for no measurable benefit.
        """
        a = self.sample_surface(samples, rng)
        b = other.sample_surface(samples, rng)
        deltas = a[:, None, :] - b[None, :, :]
        return float(np.sqrt((deltas**2).sum(axis=-1)).min())


@dataclass(slots=True)
class Ellipsoid(Primitive):
    """A solid ellipsoid. Used for chambers, valve discs, septa and small bodies."""

    centre_point: FloatArray
    radii: FloatArray
    rotation_matrix: FloatArray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    kind: PrimitiveKind = PrimitiveKind.ELLIPSOID

    def _local(self, points: FloatArray) -> FloatArray:
        return np.asarray((points - self.centre_point) @ self.rotation_matrix, dtype=np.float64)

    def occupancy(self, points: FloatArray) -> BoolArray:
        """Inside test: the scaled local radius is at most one."""
        local = self._local(points) / np.maximum(self.radii, 1e-6)
        return np.asarray((local**2).sum(axis=-1) <= 1.0, dtype=np.bool_)

    def sample_surface(self, count: int, rng: np.random.Generator) -> FloatArray:
        """Uniform-ish surface samples via a normalised Gaussian shell."""
        directions = rng.normal(size=(count, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        return np.asarray(
            directions * self.radii @ self.rotation_matrix.T + self.centre_point,
            dtype=np.float64,
        )

    @property
    def centre(self) -> FloatArray:
        """Centroid."""
        return self.centre_point

    @property
    def extent(self) -> FloatArray:
        """Half-extents along local axes."""
        return self.radii

    @property
    def rotation(self) -> FloatArray:
        """Local frame rotation."""
        return self.rotation_matrix

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "kind": str(self.kind),
            "centre": self.centre_point.tolist(),
            "radii": self.radii.tolist(),
            "rotation": self.rotation_matrix.reshape(9).tolist(),
        }


@dataclass(slots=True)
class Capsule(Primitive):
    """A capped cylinder between two points. Used for vessels and cords."""

    start: FloatArray
    end: FloatArray
    radius: float
    kind: PrimitiveKind = PrimitiveKind.CAPSULE

    def occupancy(self, points: FloatArray) -> BoolArray:
        """Inside test: distance to the axis segment is at most the radius."""
        axis = self.end - self.start
        length_squared = float(axis @ axis)
        if length_squared < 1e-12:
            deltas = points - self.start
            return np.asarray((deltas**2).sum(axis=-1) <= self.radius**2, dtype=np.bool_)
        t = np.clip(((points - self.start) @ axis) / length_squared, 0.0, 1.0)
        nearest = self.start + t[:, None] * axis
        deltas = points - nearest
        return np.asarray((deltas**2).sum(axis=-1) <= self.radius**2, dtype=np.bool_)

    def sample_surface(self, count: int, rng: np.random.Generator) -> FloatArray:
        """Samples on the lateral surface and the two caps."""
        axis = self.end - self.start
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            directions = rng.normal(size=(count, 3))
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
            return np.asarray(self.start + directions * self.radius, dtype=np.float64)
        unit = axis / length
        helper = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(unit, helper)
        u /= np.maximum(np.linalg.norm(u), 1e-9)
        v = np.cross(unit, u)
        angles = rng.uniform(0.0, 2.0 * math.pi, size=count)
        along = rng.uniform(0.0, 1.0, size=count)
        ring = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        return np.asarray(
            self.start + along[:, None] * axis + ring * self.radius, dtype=np.float64
        )

    @property
    def centre(self) -> FloatArray:
        """Midpoint of the axis."""
        return np.asarray((self.start + self.end) / 2.0, dtype=np.float64)

    @property
    def extent(self) -> FloatArray:
        """Half-length along the axis and the radius on the other two axes."""
        half_length = float(np.linalg.norm(self.end - self.start)) / 2.0
        return np.asarray([self.radius, half_length, self.radius], dtype=np.float64)

    @property
    def rotation(self) -> FloatArray:
        """Frame whose second axis follows the capsule axis."""
        axis = self.end - self.start
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            return np.eye(3, dtype=np.float64)
        unit = axis / length
        helper = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(unit, helper)
        u /= np.maximum(np.linalg.norm(u), 1e-9)
        v = np.cross(unit, u)
        return np.asarray(np.stack([u, unit, v], axis=1), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "kind": str(self.kind),
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "radius": self.radius,
        }


@dataclass(slots=True)
class Shell(Primitive):
    """The space between two concentric ellipsoids. Used for wall layers."""

    centre_point: FloatArray
    outer_radii: FloatArray
    thickness: float
    rotation_matrix: FloatArray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    kind: PrimitiveKind = PrimitiveKind.SHELL

    @property
    def inner_radii(self) -> FloatArray:
        """Radii of the inner boundary."""
        return np.maximum(self.outer_radii - self.thickness, 1e-3)

    def occupancy(self, points: FloatArray) -> BoolArray:
        """Inside the outer boundary and outside the inner one."""
        local = np.asarray((points - self.centre_point) @ self.rotation_matrix, dtype=np.float64)
        outer = ((local / np.maximum(self.outer_radii, 1e-6)) ** 2).sum(axis=-1) <= 1.0
        inner = ((local / np.maximum(self.inner_radii, 1e-6)) ** 2).sum(axis=-1) <= 1.0
        return np.asarray(outer & ~inner, dtype=np.bool_)

    def sample_surface(self, count: int, rng: np.random.Generator) -> FloatArray:
        """Samples split between the outer and inner boundaries."""
        directions = rng.normal(size=(count, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        radii = np.where(
            rng.uniform(size=(count, 1)) < 0.5, self.outer_radii, self.inner_radii
        )
        return np.asarray(
            directions * radii @ self.rotation_matrix.T + self.centre_point, dtype=np.float64
        )

    @property
    def centre(self) -> FloatArray:
        """Centroid."""
        return self.centre_point

    @property
    def extent(self) -> FloatArray:
        """Outer half-extents."""
        return self.outer_radii

    @property
    def rotation(self) -> FloatArray:
        """Local frame rotation."""
        return self.rotation_matrix

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "kind": str(self.kind),
            "centre": self.centre_point.tolist(),
            "outer_radii": self.outer_radii.tolist(),
            "thickness": self.thickness,
            "rotation": self.rotation_matrix.reshape(9).tolist(),
        }


def build_primitive(payload: dict[str, Any]) -> Primitive:
    """Rebuild a primitive from :meth:`Primitive.to_dict` output."""
    kind = PrimitiveKind(payload["kind"])
    if kind is PrimitiveKind.ELLIPSOID:
        return Ellipsoid(
            centre_point=np.asarray(payload["centre"], dtype=np.float64),
            radii=np.asarray(payload["radii"], dtype=np.float64),
            rotation_matrix=np.asarray(payload["rotation"], dtype=np.float64).reshape(3, 3),
        )
    if kind is PrimitiveKind.CAPSULE:
        return Capsule(
            start=np.asarray(payload["start"], dtype=np.float64),
            end=np.asarray(payload["end"], dtype=np.float64),
            radius=float(payload["radius"]),
        )
    return Shell(
        centre_point=np.asarray(payload["centre"], dtype=np.float64),
        outer_radii=np.asarray(payload["outer_radii"], dtype=np.float64),
        thickness=float(payload["thickness"]),
        rotation_matrix=np.asarray(payload["rotation"], dtype=np.float64).reshape(3, 3),
    )


def occupancy_of_union(
    primitives: Sequence[Primitive], points: FloatArray
) -> BoolArray:
    """Occupancy of the union of several primitives."""
    if not primitives:
        return np.zeros(points.shape[0], dtype=np.bool_)
    result = primitives[0].occupancy(points)
    for primitive in primitives[1:]:
        result = result | primitive.occupancy(points)
    return result
