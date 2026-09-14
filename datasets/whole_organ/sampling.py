"""Query points, exact labels, level-specific targets and edit pairs.

Two things here that Step 6 did not have:

* **Level-specific targets.** The occupancy target at each level of detail is the union
  of the entities that level shows. Step 6 trained every token prefix against the same
  full-detail target, which is why its later tokens had nothing to add.
* **Edit pairs.** An edit changes one global parameter and the whole organ is rebuilt
  from it. Which entities actually changed is then **measured**, not assumed, so edit
  locality can be scored against what the generator really did.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from datasets.whole_organ.corpus import LOD_ENTITIES, WholeOrganScene
from datasets.whole_organ.field import WholeOrganField

__all__ = [
    "EditOperation",
    "WholeOrganSampling",
    "SampledWholeOrgan",
    "sample_scene",
    "edit_pair",
    "EDIT_SPECIFICATIONS",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


class EditOperation(StrEnum):
    """Controlled geometric edits the generator can apply."""

    THICKEN_WALL = "thicken_wall"
    THIN_WALL = "thin_wall"
    ENLARGE_VENTRICLE = "enlarge_ventricle"
    SHRINK_VENTRICLE = "shrink_ventricle"
    ENLARGE_ATRIUM = "enlarge_atrium"
    WIDEN_VESSEL = "widen_vessel"
    THICKEN_VALVE = "thicken_valve"


EDIT_SPECIFICATIONS: Mapping[EditOperation, tuple[str, float, str]] = {
    EditOperation.THICKEN_WALL: ("wall_fraction", 1.45, "heart.myocardium"),
    EditOperation.THIN_WALL: ("wall_fraction", 0.68, "heart.myocardium"),
    EditOperation.ENLARGE_VENTRICLE: ("entity", 1.28, "heart.left_ventricle"),
    EditOperation.SHRINK_VENTRICLE: ("entity", 0.76, "heart.left_ventricle"),
    EditOperation.ENLARGE_ATRIUM: ("entity", 1.32, "heart.left_atrium"),
    EditOperation.WIDEN_VESSEL: ("entity", 1.60, "heart.aorta"),
    EditOperation.THICKEN_VALVE: ("entity", 1.75, "heart.mitral_valve"),
}
"""Each edit names the parameter it scales and the entity a user would name.

The entity is what the instruction talks about. Which entities actually move is measured
afterwards, because in a coherent organ an edit to one structure can legitimately change
its neighbours.
"""


@dataclass(frozen=True, slots=True)
class WholeOrganSampling:
    """How many points to draw."""

    scene_points: int = 256
    entity_points: int = 24
    entity_slots: int = 64
    surface_jitter: float = 0.03


@dataclass(frozen=True, slots=True)
class SampledWholeOrgan:
    """One scene's points and exact labels."""

    scene_id: str
    scene_points: FloatArray
    ownership: IntArray
    scene_occupancy: BoolArray
    lod_occupancy: Mapping[int, BoolArray]
    entity_points: FloatArray
    entity_occupancy: BoolArray
    entity_frames: FloatArray
    present: BoolArray
    visible: BoolArray

    def counts(self) -> dict[str, int]:
        """Shapes, for tests and logging."""
        return {
            "scene_points": int(self.scene_points.shape[0]),
            "owned": int((self.ownership >= 0).sum()),
            "entities": int(self.present.sum()),
            "visible": int(self.visible.sum()),
        }


def sample_scene(
    scene: WholeOrganScene,
    slot_of: Mapping[str, int],
    sampling: WholeOrganSampling | None = None,
    *,
    epoch: int = 0,
    organ: WholeOrganField | None = None,
) -> SampledWholeOrgan:
    """Sample one scene and label every point exactly."""
    settings = sampling or WholeOrganSampling()
    field = organ or scene.field()
    rng = np.random.default_rng(scene.seed * 6151 + epoch)

    points = field.sample_points(settings.scene_points, rng, jitter=settings.surface_jitter)
    labels = field.ownership(points)
    owner_names = {field.slot_of[name]: name for name in field.entity_ids}
    codebook_labels = np.full(points.shape[0], -1, dtype=np.int64)
    for local_slot, name in owner_names.items():
        codebook_labels[labels == local_slot] = slot_of[name]

    visible_names = set(scene.visible_entity_ids())
    lod_occupancy: dict[int, BoolArray] = {}
    for level in (1, 2, 3):
        allowed = {
            slot_of[name] for name in LOD_ENTITIES[level] if name in scene.centroids
        }
        lod_occupancy[level] = np.isin(codebook_labels, list(allowed))

    total = settings.entity_slots
    entity_points = np.zeros((total, settings.entity_points, 3), dtype=np.float64)
    entity_occupancy = np.zeros((total, settings.entity_points), dtype=np.bool_)
    frames = np.zeros((total, 12), dtype=np.float64)
    present = np.zeros(total, dtype=np.bool_)
    visible = np.zeros(total, dtype=np.bool_)

    scene_frames = scene.frames()
    for name in scene.entity_ids:
        slot = slot_of[name]
        present[slot] = True
        visible[slot] = name in visible_names
        frames[slot] = np.asarray(scene_frames[name], dtype=np.float64)
        if name not in visible_names:
            continue
        sampled, owned = field.entity_points(name, settings.entity_points, rng)
        entity_points[slot] = sampled
        entity_occupancy[slot] = owned

    return SampledWholeOrgan(
        scene_id=scene.scene_id,
        scene_points=points,
        ownership=codebook_labels,
        scene_occupancy=lod_occupancy[min(max(scene.active_lod, 1), 3)],
        lod_occupancy=lod_occupancy,
        entity_points=entity_points,
        entity_occupancy=entity_occupancy,
        entity_frames=frames,
        present=present,
        visible=visible,
    )


def edit_pair(
    scene: WholeOrganScene,
    operation: EditOperation,
    *,
    points: int = 1500,
    change_threshold: float = 0.08,
) -> tuple[WholeOrganField, WholeOrganField, dict[str, float]]:
    """Build a before and after organ for one controlled edit.

    Returns the two fields and, per entity, the fraction of its points whose ownership
    changed. The generator decides what an edit really does; nothing is assumed.
    """
    parameter, scale, target = EDIT_SPECIFICATIONS[operation]
    before_parameters = scene.parameters
    from dataclasses import replace

    if parameter == "entity":
        after_parameters = replace(
            before_parameters,
            entity_scale=(*before_parameters.entity_scale, (target, scale)),
        )
    else:
        after_parameters = replace(
            before_parameters, **{parameter: float(getattr(before_parameters, parameter)) * scale}
        )
    before = WholeOrganField(before_parameters)
    after = WholeOrganField(after_parameters)

    rng = np.random.default_rng(scene.seed * 131 + 7)
    probe = before.sample_points(points, rng)
    before_labels = before.ownership(probe)
    after_labels = after.ownership(probe)
    changed: dict[str, float] = {}
    for name, slot in before.slot_of.items():
        mask = (before_labels == slot) | (after_labels == slot)
        if not bool(mask.any()):
            changed[name] = 0.0
            continue
        disagreement = (before_labels[mask] == slot) != (after_labels[mask] == slot)
        changed[name] = float(disagreement.mean())
    del change_threshold
    return before, after, changed


def changed_entities(
    changes: Mapping[str, float], threshold: float = 0.08
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split entities into those the edit really changed and those it did not."""
    moved = tuple(name for name, value in changes.items() if value >= threshold)
    still = tuple(name for name, value in changes.items() if value < threshold)
    return moved, still
