"""Resize one entity within its shape model, and only that entity (ADR-L2-001).

The request is a change of linear size. The current shape, in model correspondence, is scaled
about its centroid, then projected onto the entity's own shape model, with weights clipped at
``CLIP`` standard deviations. The model's shape is what appears: the closest shape the
population supports, not a magnified copy. The result says how much size and volume it
actually achieved, and how far it lies from plain scaling.

Before anything is written, the new shape goes through the validity checks
(``shapes.validity``) with the calibrated limits:

* folding against the model mean;
* volume within the training range;
* overlap deeper than 2 mm, both ways, with every other modelled entity.

All three are measured on correspondence meshes, the kind the limits were calibrated on.
They are judged against "no worse than real anatomy, nor than this subject already is". A
subject slightly beyond a training extreme can still be resized, so long as the change does
not make that worse.

If any check fails, nothing is written. The reply gives the reason and the largest change in
the asked direction that does pass, found by bisection to 1%. On a real heart that is often
small: a chamber is packed against its neighbours (Level 2a report, §7).

The base shape must be in correspondence. It is found in this order:

1. ``correspondence.npz`` in the scene folder, written by an earlier resize;
2. the coefficients of a generated sample;
3. the subject's cached registration.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh

from assets.meshes import side_report
from assets.quality import assess
from awr.ontology import AnatomyOntology, load_ontology
from language.program import MAX_RESIZE_PERCENT
from shapes.generate import entity_entry, model_mesh
from shapes.joint import JointModel
from shapes.register import load_scene
from shapes.ssm import ShapeModel
from shapes.validity import CLIP, Limits, face_normals, overlap_share

__all__ = ["MAX_PERCENT", "ResizeOutcome", "Resizer", "resize_shape"]

FloatArray = npt.NDArray[np.float64]

MAX_PERCENT = MAX_RESIZE_PERCENT
CORRESPONDENCE = "correspondence.npz"


@dataclass(frozen=True)
class ResizeOutcome:
    """What a resize did, or why it did not."""

    ok: bool
    message: str
    entity: str
    factor: float
    achieved: float | None = None
    volume_change: float | None = None
    from_plain_scaling_mm: float | None = None
    failures: tuple[str, ...] = ()
    feasible_percent: int | None = None
    manifest: dict[str, Any] | None = field(default=None, repr=False)

    def record(self) -> dict[str, Any]:
        """Plain data for the manifest and the reply."""
        return {
            "entity": self.entity,
            "requested_scale": round(self.factor, 4),
            "achieved_scale": None if self.achieved is None else round(self.achieved, 4),
            "volume_change": None if self.volume_change is None else round(self.volume_change, 4),
            "from_plain_scaling_mm": None
            if self.from_plain_scaling_mm is None
            else round(self.from_plain_scaling_mm, 3),
        }


def resize_shape(model: ShapeModel, current: FloatArray, factor: float) -> FloatArray:
    """The model's closest shape to ``current`` scaled by ``factor`` about its centroid."""
    centre = current.mean(axis=0)
    target = centre + factor * (current - centre)
    weights = np.clip(model.project(target), -CLIP, CLIP)
    return model.sample(weights)


def _volume(vertices: FloatArray, faces: npt.ArrayLike) -> float:
    return float(abs(model_mesh(vertices, faces).volume))


class Resizer:
    """Resize entities of scenes made from, or registered to, one Level 2a model folder."""

    def __init__(self, models_dir: Path, *, ontology: AnatomyOntology | None = None) -> None:
        self.models_dir = models_dir
        self.joint = JointModel.load(models_dir / "models" / "joint.npz")
        validity = json.loads((models_dir / "validity.json").read_text(encoding="utf-8"))
        self.limits = Limits.from_dict(validity["limits"])
        self.ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
        self._models: dict[str, ShapeModel] = {}

    @property
    def entities(self) -> tuple[str, ...]:
        """Entities that can be resized."""
        return self.joint.entities

    def model(self, entity: str) -> ShapeModel:
        """The entity's own shape model, loaded once."""
        if entity not in self._models:
            self._models[entity] = ShapeModel.load(self.models_dir / "models" / f"{entity}.npz")
        return self._models[entity]

    def base_shapes(
        self, scene_dir: Path, manifest: Mapping[str, Any]
    ) -> dict[str, FloatArray] | None:
        """The scene's modelled entities in correspondence, or None if it has none."""
        stored = scene_dir / CORRESPONDENCE
        if stored.is_file():
            arrays: Any = np.load(stored)
            return {e: np.asarray(arrays[e.replace(".", "__")]) for e in self.entities}
        generated = manifest.get("generated")
        if generated and "coefficients" in generated:
            return self.joint.sample_entities(generated["coefficients"])
        cache = self.models_dir / "registrations"
        subject = str(manifest.get("subject", ""))
        paths = {e: cache / f"{subject}__{e}.npz" for e in self.entities}
        if subject and all(p.is_file() for p in paths.values()):
            return {e: np.asarray(np.load(p)["vertices"]) for e, p in paths.items()}
        return None

    def resize(
        self,
        scene_dir: Path,
        manifest: Mapping[str, Any],
        entity: str,
        factor: float,
        out_dir: Path,
    ) -> ResizeOutcome:
        """Resize ``entity`` by ``factor``; write the new scene to ``out_dir`` only if valid."""

        def refuse(message: str, **found: Any) -> ResizeOutcome:
            return ResizeOutcome(False, message, entity, factor, **found)

        if entity not in self.entities:
            return refuse(f"{entity} has no shape model; resizable: {', '.join(self.entities)}.")
        if not 1 - MAX_PERCENT / 100 <= factor <= 1 + MAX_PERCENT / 100:
            return refuse(f"One resize can change size by at most {MAX_PERCENT:g}%.")
        shapes = self.base_shapes(scene_dir, manifest)
        if shapes is None:
            return refuse(
                f"This scene ({manifest.get('subject')}) has no registration to the shape "
                "model, so it cannot be resized. Generate a thorax first."
            )

        model = self.model(entity)
        current = shapes[entity]
        resized = resize_shape(model, current, factor)
        faces = self.joint.faces[entity]
        volume_ratio = _volume(resized, faces) / _volume(current, faces)
        centre = current.mean(axis=0)
        plain = centre + factor * (current - centre)
        found: dict[str, Any] = {
            "achieved": float(volume_ratio ** (1 / 3)),
            "volume_change": float(volume_ratio - 1),
            "from_plain_scaling_mm": float(np.linalg.norm(resized - plain, axis=1).mean()),
        }

        baseline = self._measure(entity, current, shapes)
        failures = self._check(entity, resized, shapes, baseline)
        if failures:
            feasible = self._largest(entity, model, current, shapes, baseline, factor)
            sign = "+" if factor > 1 else "-"
            offer = (
                f" The largest {('increase' if factor > 1 else 'decrease')} that passes for this "
                f"subject is {sign}{feasible}%."
                if feasible
                else " No change of even 1% in that direction passes for this subject."
            )
            return refuse(
                f"Not applied: {entity} {factor - 1:+.0%} in size is outside what the training "
                f"anatomy shows ({'; '.join(failures)}).{offer}",
                failures=tuple(failures),
                feasible_percent=feasible,
                **found,
            )
        written = self._write(scene_dir, manifest, entity, resized, shapes, out_dir, found, factor)
        achieved = found["achieved"]
        return ResizeOutcome(
            True,
            f"{entity}: asked {factor - 1:+.0%} in size; the shape model gives "
            f"{achieved - 1:+.1%} ({found['volume_change']:+.1%} volume), "
            f"{found['from_plain_scaling_mm']:.1f} mm on average from plain scaling.",
            entity,
            factor,
            failures=(),
            manifest=written,
            **found,
        )

    def _measure(
        self, entity: str, shape: FloatArray, shapes: Mapping[str, FloatArray]
    ) -> dict[str, float]:
        """Folding, volume and overlap with every other entity, on correspondence meshes."""
        faces = self.joint.faces[entity]
        mean = self.model(entity).mean
        measures = {
            "folding": float(
                np.mean(np.sum(face_normals(shape, faces) * face_normals(mean, faces), axis=1) < 0)
            ),
            "volume": _volume(shape, faces) / 1000.0,
        }
        mesh = model_mesh(shape, faces)
        for other in self.entities:
            if other == entity:
                continue
            other_mesh = model_mesh(shapes[other], self.joint.faces[other])
            measures[f"{entity}|{other}"] = overlap_share(mesh, other_mesh)
            measures[f"{other}|{entity}"] = overlap_share(other_mesh, mesh)
        return measures

    def _check(
        self,
        entity: str,
        shape: FloatArray,
        shapes: Mapping[str, FloatArray],
        baseline: Mapping[str, float],
    ) -> list[str]:
        """Failures of ``shape``: worse than real anatomy and worse than the subject was."""
        found = self._measure(entity, shape, shapes)
        failures: list[str] = []
        if found["folding"] > max(self.limits.folding[entity], baseline["folding"]):
            failures.append(f"{found['folding']:.1%} of its faces fold")
        low, high = self.limits.volume_ml[entity]
        low, high = min(low, baseline["volume"]), max(high, baseline["volume"])
        if not low <= found["volume"] <= high:
            failures.append(f"volume {found['volume']:.0f} mL outside {low:.0f}-{high:.0f} mL")
        for pair, share in found.items():
            limit = self.limits.overlap.get(pair)
            if limit is not None and share > max(limit, baseline[pair]):
                a, b = pair.split("|")
                failures.append(f"{share:.1%} of {a} inside {b}, real anatomy {limit:.1%}")
        return failures

    def _largest(
        self,
        entity: str,
        model: ShapeModel,
        current: FloatArray,
        shapes: Mapping[str, FloatArray],
        baseline: Mapping[str, float],
        factor: float,
    ) -> int | None:
        """The largest whole-percent change towards ``factor`` that passes, or None."""
        sign = 1 if factor > 1 else -1
        low, high = 0, round(abs(factor - 1) * 100)
        while high - low > 1:
            middle = (low + high) // 2
            shape = resize_shape(model, current, 1 + sign * middle / 100)
            if self._check(entity, shape, shapes, baseline):
                high = middle
            else:
                low = middle
        return low or None

    def _write(
        self,
        scene_dir: Path,
        manifest: Mapping[str, Any],
        entity: str,
        resized: FloatArray,
        shapes: Mapping[str, FloatArray],
        out_dir: Path,
        found: Mapping[str, Any],
        factor: float,
    ) -> dict[str, Any]:
        meshes = load_scene(scene_dir)
        meshes[entity] = model_mesh(resized, self.joint.faces[entity])
        scene = trimesh.Scene()
        for name, mesh in meshes.items():
            scene.add_geometry(mesh, node_name=name, geom_name=name)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scene.glb").write_bytes(scene.export(file_type="glb"))
        arrays: dict[str, Any] = {
            e.replace(".", "__"): (resized if e == entity else v) for e, v in shapes.items()
        }
        np.savez_compressed(out_dir / CORRESPONDENCE, **arrays)
        written = copy.deepcopy(dict(manifest))
        written["entities"][entity] = entity_entry(entity, meshes[entity])
        centroids = {e: entry["centroid_viewer_mm"] for e, entry in written["entities"].items()}
        written["sides_correct"] = side_report(centroids, self.ontology)
        record = ResizeOutcome(True, "", entity, factor, **found).record()
        written["resized"] = [*manifest.get("resized", []), record]
        written["checks"] = asdict(assess(written))
        (out_dir / "manifest.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
        return written
