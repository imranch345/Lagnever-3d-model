"""A principal-component shape model over corresponded meshes.

Every training shape is the same template mesh, deformed: vertex ``i`` is the same anatomical
place in every subject. Stacking them gives an ``n x 3V`` matrix; its principal components are
the model's modes, ordered by the variance they explain. A shape is then the mean plus a
weighted sum of modes, with each weight measured in standard deviations of the training
population, which is what makes "two standard deviations larger" a meaningful request.

NumPy only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["ShapeModel", "fit"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ShapeModel:
    """Mean shape, modes and their standard deviations, for one entity's template."""

    entity_id: str
    faces: IntArray
    mean: FloatArray
    """(V, 3)"""
    modes: FloatArray
    """(k, V, 3), orthonormal when flattened."""
    stdevs: FloatArray
    """(k,), the training population's standard deviation along each mode."""
    explained: FloatArray
    """(k,), fraction of total variance each kept mode explains."""
    subjects: tuple[str, ...]

    @property
    def n_modes(self) -> int:
        """Modes kept."""
        return int(self.modes.shape[0])

    def sample(self, coefficients: npt.ArrayLike) -> FloatArray:
        """Vertices for weights in standard deviations; missing trailing weights are zero."""
        b = np.zeros(self.n_modes)
        given = np.asarray(coefficients, dtype=np.float64).ravel()[: self.n_modes]
        b[: given.size] = given
        offset = np.tensordot(b * self.stdevs, self.modes, axes=1)
        result: FloatArray = self.mean + offset
        return result

    def project(self, vertices: npt.ArrayLike) -> FloatArray:
        """Weights, in standard deviations, of the closest shape the model can express."""
        delta = (np.asarray(vertices, dtype=np.float64) - self.mean).reshape(-1)
        raw = self.modes.reshape(self.n_modes, -1) @ delta
        with np.errstate(divide="ignore", invalid="ignore"):
            out: FloatArray = np.where(self.stdevs > 0, raw / self.stdevs, 0.0)
        return out

    def reconstruct(self, vertices: npt.ArrayLike, n_modes: int | None = None) -> FloatArray:
        """The vertices as the model sees them, using the first ``n_modes`` modes."""
        b = self.project(vertices)
        if n_modes is not None:
            b[n_modes:] = 0.0
        return self.sample(b)

    def save(self, path: Path) -> None:
        """Arrays to ``.npz``, metadata to a sidecar ``.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            faces=self.faces,
            mean=self.mean,
            modes=self.modes,
            stdevs=self.stdevs,
            explained=self.explained,
        )
        meta = {
            "entity_id": self.entity_id,
            "subjects": list(self.subjects),
            "n_modes": self.n_modes,
            "explained": self.explained.tolist(),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ShapeModel:
        """Inverse of :meth:`save`."""
        arrays: Any = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(
            entity_id=str(meta["entity_id"]),
            faces=arrays["faces"],
            mean=arrays["mean"],
            modes=arrays["modes"],
            stdevs=arrays["stdevs"],
            explained=arrays["explained"],
            subjects=tuple(meta["subjects"]),
        )


def fit(
    entity_id: str,
    shapes: npt.ArrayLike,
    faces: npt.ArrayLike,
    subjects: tuple[str, ...],
    *,
    variance: float = 0.95,
    max_modes: int | None = None,
) -> ShapeModel:
    """Fit a model to corresponded shapes ``(n, V, 3)``, keeping modes up to ``variance``.

    Uses the sample standard deviation (``n - 1``), so a mode's unit is a population spread,
    not a property of how many subjects happened to be used.
    """
    data = np.asarray(shapes, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"shapes must be (n, V, 3), got {data.shape}")
    n = data.shape[0]
    if n < 2:
        raise ValueError("at least two shapes are needed")
    if len(subjects) != n:
        raise ValueError("one subject id per shape")
    mean = data.mean(axis=0)
    centred = (data - mean).reshape(n, -1)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variances = singular**2 / (n - 1)
    total = variances.sum()
    explained = variances / total if total > 0 else np.zeros_like(variances)
    keep = int(np.searchsorted(np.cumsum(explained), variance) + 1)
    keep = min(keep, len(variances), max_modes or len(variances))
    return ShapeModel(
        entity_id=entity_id,
        faces=np.asarray(faces, dtype=np.int64),
        mean=mean,
        modes=vt[:keep].reshape(keep, -1, 3),
        stdevs=np.sqrt(variances[:keep]),
        explained=explained[:keep],
        subjects=tuple(subjects),
    )
