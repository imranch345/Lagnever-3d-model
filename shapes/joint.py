"""One shape model over several entities at once, so samples stay mutually consistent.

Per-entity models vary independently: sampled separately, a large left ventricle can come
with a small myocardium that no longer contains it. Concatenating every modelled entity's
corresponded vertices per subject, and fitting one model, keeps the correlations the
population has -- a larger heart has larger chambers -- and keeps relative positions, because
every subject was aligned to the same reference before registration.

Only subjects with every entity registered within the fit threshold can contribute.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from shapes.ssm import ShapeModel, fit

__all__ = ["JointModel", "fit_joint"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class JointModel:
    """A shape model over concatenated entities, and where each entity's vertices sit."""

    model: ShapeModel
    segments: Mapping[str, tuple[int, int]]
    faces: Mapping[str, IntArray]

    @property
    def entities(self) -> tuple[str, ...]:
        """Modelled entities, in concatenation order."""
        return tuple(self.segments)

    def split(self, vertices: FloatArray) -> dict[str, FloatArray]:
        """Concatenated vertices back into per-entity arrays."""
        return {e: vertices[a:b] for e, (a, b) in self.segments.items()}

    def sample_entities(self, coefficients: npt.ArrayLike) -> dict[str, FloatArray]:
        """Per-entity vertices for weights in standard deviations."""
        return self.split(self.model.sample(coefficients))

    def save(self, path: Path) -> None:
        """The shared model, plus segments and faces in a second archive."""
        self.model.save(path)
        arrays: dict[str, Any] = {e.replace(".", "__"): f for e, f in self.faces.items()}
        np.savez_compressed(path.with_name(path.stem + "_faces.npz"), **arrays)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        meta["segments"] = {e: list(s) for e, s in self.segments.items()}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> JointModel:
        """Inverse of :meth:`save`."""
        model = ShapeModel.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        stored: Any = np.load(path.with_name(path.stem + "_faces.npz"))
        segments = {e: (int(a), int(b)) for e, (a, b) in meta["segments"].items()}
        faces = {e: stored[e.replace(".", "__")] for e in segments}
        return cls(model=model, segments=segments, faces=faces)


def fit_joint(
    shapes: Mapping[str, Mapping[str, FloatArray]],
    faces: Mapping[str, npt.ArrayLike],
    subjects: Sequence[str],
    *,
    variance: float = 0.95,
) -> JointModel:
    """Fit over ``subjects`` that have every entity in ``shapes``; others are left out."""
    entities = list(shapes)
    complete = [s for s in subjects if all(s in shapes[e] for e in entities)]
    if len(complete) < 2:
        raise ValueError("fewer than two subjects have every entity")
    segments: dict[str, tuple[int, int]] = {}
    start = 0
    for entity in entities:
        count = len(next(iter(shapes[entity].values())))
        segments[entity] = (start, start + count)
        start += count
    stacked = np.stack([np.concatenate([shapes[e][s] for e in entities]) for s in complete])
    model = fit(
        "joint", stacked, np.zeros((0, 3), dtype=np.int64), tuple(complete), variance=variance
    )
    return JointModel(
        model=model,
        segments=segments,
        faces={e: np.asarray(faces[e], dtype=np.int64) for e in entities},
    )
