"""Scene editing operations.

Every change to a scene is an operation object. Nothing edits
:class:`~awr.scene.AWRScene` state directly. The benefits are concrete:

* An edit touches only the entities it names. The scene is never regenerated to
  satisfy a command, so entity ids, geometry references and history survive.
* Each operation reports explicit before/after deltas, which the editor commits
  as one versioned event.
* Operations are pure with respect to identity: they change state, never
  identity, hierarchy or relationships.

Operations that depend on research not yet done are *declared* here and raise
:class:`~awr.errors.NotYetImplementedError`. That keeps the roadmap visible in
code without shipping a pretend implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from awr.errors import NotYetImplementedError, OperationError
from awr.scene import AWRScene, EntityDelta
from awr.schema import (
    AnimationState,
    EducationalLevel,
    GeometryReference,
    LodLevel,
    MaterialState,
    Transform,
    VisibilitySource,
)

__all__ = [
    "OperationResult",
    "SceneOperation",
    "SetVisibility",
    "ShowEntities",
    "HideEntities",
    "RevealEntities",
    "SetOpacity",
    "IsolateEntities",
    "SetLod",
    "SetEducationalLevel",
    "ResetToLodDefaults",
    "SetMaterial",
    "SetTransform",
    "BindAnimation",
    "ClearAnimation",
    "AttachGeometryReferences",
    "PLANNED_OPERATIONS",
]


@dataclass(frozen=True, slots=True)
class OperationResult:
    """What an operation changed."""

    operation: str
    summary: str
    deltas: tuple[EntityDelta, ...] = ()
    scene_fields_changed: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def affected(self) -> tuple[str, ...]:
        """Entity ids touched, in first-touched order."""
        seen: set[str] = set()
        out: list[str] = []
        for delta in self.deltas:
            if delta.entity_id not in seen:
                seen.add(delta.entity_id)
                out.append(delta.entity_id)
        return tuple(out)

    @property
    def changed(self) -> bool:
        """Whether anything actually changed (a no-op does not bump the version)."""
        return bool(self.deltas) or bool(self.scene_fields_changed)


class SceneOperation(ABC):
    """Base class for every scene edit."""

    name: ClassVar[str] = "operation"

    def validate(self, scene: AWRScene) -> None:
        """Check the operation can be applied. Raise :class:`OperationError` if not."""
        return None

    @abstractmethod
    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the change and return its deltas. Does not commit history."""

    def describe(self) -> str:
        """One-line human-readable description."""
        return self.name

    # -- helpers shared by concrete operations -------------------------
    @staticmethod
    def _expand(scene: AWRScene, entity_ids: Sequence[str]) -> tuple[str, ...]:
        """Expand group ids to renderable members and validate every id exists."""
        if not entity_ids:
            raise OperationError("No target entities were given.")
        for entity_id in entity_ids:
            scene.get(entity_id)  # raises EntityNotFoundError with a useful message
        expanded = scene.expand(entity_ids)
        if not expanded:
            raise OperationError(
                f"None of {list(entity_ids)} expands to a renderable entity; "
                "organisational groups must contain something that can be shown."
            )
        return expanded


@dataclass(slots=True)
class SetVisibility(SceneOperation):
    """Set visibility on the given entities (groups expand to their members)."""

    entity_ids: tuple[str, ...]
    visible: bool
    source: VisibilitySource = VisibilitySource.MANUAL
    include_descendants: bool = False

    name: ClassVar[str] = "set_visibility"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the visibility change."""
        targets = list(self._expand(scene, self.entity_ids))
        if self.include_descendants:
            for entity_id in tuple(targets):
                for descendant in scene.descendants(entity_id):
                    if scene.get(descendant).renderable and descendant not in targets:
                        targets.append(descendant)
        deltas = [
            delta
            for entity_id in targets
            if (delta := scene.set_entity_visibility(entity_id, self.visible, source=self.source))
        ]
        verb = "Showed" if self.visible else "Hid"
        state = "visible" if self.visible else "hidden"
        summary = (
            f"All {len(targets)} targeted entities were already {state}."
            if not deltas
            else f"{verb} {len(deltas)} of {len(targets)} targeted entities."
        )
        return OperationResult(
            operation=self.name,
            summary=summary,
            deltas=tuple(deltas),
            parameters={
                "requested": list(self.entity_ids),
                "targets": targets,
                "visible": self.visible,
            },
        )

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"{'show' if self.visible else 'hide'}({', '.join(self.entity_ids)})"


class ShowEntities(SetVisibility):
    """Make entities visible."""

    name: ClassVar[str] = "show"

    def __init__(self, entity_ids: Sequence[str], *, include_descendants: bool = False) -> None:
        super().__init__(
            entity_ids=tuple(entity_ids),
            visible=True,
            source=VisibilitySource.MANUAL,
            include_descendants=include_descendants,
        )


class HideEntities(SetVisibility):
    """Make entities invisible. They remain in the AWR."""

    name: ClassVar[str] = "hide"

    def __init__(self, entity_ids: Sequence[str], *, include_descendants: bool = False) -> None:
        super().__init__(
            entity_ids=tuple(entity_ids),
            visible=False,
            source=VisibilitySource.MANUAL,
            include_descendants=include_descendants,
        )


class RevealEntities(ShowEntities):
    """Show entities together with everything beneath them in the tree."""

    name: ClassVar[str] = "reveal"

    def __init__(self, entity_ids: Sequence[str]) -> None:
        super().__init__(entity_ids, include_descendants=True)


@dataclass(slots=True)
class SetOpacity(SceneOperation):
    """Set opacity on the given entities."""

    entity_ids: tuple[str, ...]
    opacity: float
    preset_name: str | None = None

    name: ClassVar[str] = "set_opacity"

    def validate(self, scene: AWRScene) -> None:
        """Reject opacity values outside ``[0, 1]`` before touching the scene."""
        if not 0.0 <= self.opacity <= 1.0:
            raise OperationError(
                f"Opacity must be within [0.0, 1.0], got {self.opacity!r}. "
                "Use a fraction (0.3), not a percentage (30)."
            )

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the opacity change."""
        targets = self._expand(scene, self.entity_ids)
        deltas = [
            delta
            for entity_id in targets
            if (delta := scene.set_entity_opacity(entity_id, self.opacity))
        ]
        label = f" ({self.preset_name})" if self.preset_name else ""
        return OperationResult(
            operation=self.name,
            summary=f"Set opacity {self.opacity:g}{label} on {len(deltas)} entities.",
            deltas=tuple(deltas),
            parameters={
                "targets": list(targets),
                "opacity": self.opacity,
                "preset": self.preset_name,
            },
        )

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"set_opacity({', '.join(self.entity_ids)}, {self.opacity:g})"


@dataclass(slots=True)
class IsolateEntities(SceneOperation):
    """Show the given entities and hide every other renderable entity."""

    entity_ids: tuple[str, ...]

    name: ClassVar[str] = "isolate"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the isolation."""
        keep = set(self._expand(scene, self.entity_ids))
        deltas: list[EntityDelta] = []
        for entity in scene.iter_entities():
            if not entity.renderable:
                continue
            wanted = entity.entity_id in keep
            delta = scene.set_entity_visibility(
                entity.entity_id, wanted, source=VisibilitySource.MANUAL
            )
            if delta:
                deltas.append(delta)
        return OperationResult(
            operation=self.name,
            summary=f"Isolated {len(keep)} entities; {len(deltas)} visibility changes.",
            deltas=tuple(deltas),
            parameters={"kept": sorted(keep), "requested": list(self.entity_ids)},
        )

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"isolate({', '.join(self.entity_ids)})"


@dataclass(slots=True)
class SetLod(SceneOperation):
    """Change the active LOD and recompute default visibility.

    Rule, deliberate and tested: an LOD change reapplies the LOD policy to every
    renderable entity and therefore clears manual visibility overrides, while
    material state such as opacity is left untouched. Without this, "show more
    detail" after a few manual edits would produce an unpredictable scene.
    """

    lod: LodLevel

    name: ClassVar[str] = "set_lod"

    def validate(self, scene: AWRScene) -> None:
        """Reject LOD values outside the configured ladder."""
        if self.lod not in scene.lod_ladder:
            valid = ", ".join(str(spec.level) for spec in scene.lod_ladder.levels)
            raise OperationError(f"LOD {self.lod} is not defined. Valid levels: {valid}.")

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the LOD change."""
        previous = scene.active_lod
        scene.set_active_lod(self.lod)
        deltas: list[EntityDelta] = []
        cleared = 0
        for entity in scene.iter_entities():
            if not entity.renderable:
                continue
            if entity.state.visibility_source is VisibilitySource.MANUAL:
                cleared += 1
            wanted = scene.default_visibility(entity.entity_id)
            delta = scene.set_entity_visibility(
                entity.entity_id, wanted, source=VisibilitySource.LOD_DEFAULT
            )
            if delta:
                deltas.append(delta)
        spec = scene.lod_spec()
        return OperationResult(
            operation=self.name,
            summary=(
                f"Active LOD {previous} -> {self.lod} ({spec.name}); "
                f"{len(deltas)} visibility changes, {cleared} manual overrides cleared."
            ),
            deltas=tuple(deltas),
            scene_fields_changed=("active_lod",) if previous != self.lod else (),
            parameters={"previous_lod": previous, "lod": self.lod, "cleared_overrides": cleared},
            data={"lod_name": spec.name, "lod_description": spec.description},
        )

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"set_lod({self.lod})"


@dataclass(slots=True)
class SetEducationalLevel(SceneOperation):
    """Switch audience level, which also switches LOD through the configured map."""

    level: EducationalLevel

    name: ClassVar[str] = "set_educational_level"

    def validate(self, scene: AWRScene) -> None:
        """Reject audience levels that are not configured."""
        if self.level not in scene.educational_levels.levels():
            valid = ", ".join(str(level) for level in scene.educational_levels.levels())
            raise OperationError(
                f"Educational level {self.level!r} is not configured. Valid levels: {valid}."
            )

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the audience-level change."""
        previous_level = scene.educational_level
        target_lod = scene.educational_levels.lod_for(self.level)
        scene.set_educational_level(self.level)
        lod_result = SetLod(target_lod).apply(scene)
        changed = ("educational_level",) if previous_level is not self.level else ()
        return OperationResult(
            operation=self.name,
            summary=(
                f"Educational level {previous_level} -> {self.level}; " + lod_result.summary
            ),
            deltas=lod_result.deltas,
            scene_fields_changed=changed + lod_result.scene_fields_changed,
            parameters={
                "previous_level": str(previous_level),
                "level": str(self.level),
                "lod": target_lod,
            },
            data=dict(lod_result.data),
        )

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"set_educational_level({self.level})"


@dataclass(slots=True)
class ResetToLodDefaults(SceneOperation):
    """Discard manual visibility and opacity edits, keeping the active LOD."""

    default_opacity: float = 1.0

    name: ClassVar[str] = "reset_view"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the reset."""
        deltas: list[EntityDelta] = []
        for entity in scene.iter_entities():
            if entity.renderable:
                wanted = scene.default_visibility(entity.entity_id)
                delta = scene.set_entity_visibility(
                    entity.entity_id, wanted, source=VisibilitySource.LOD_DEFAULT
                )
                if delta:
                    deltas.append(delta)
            opacity_delta = scene.set_entity_opacity(entity.entity_id, self.default_opacity)
            if opacity_delta:
                deltas.append(opacity_delta)
        return OperationResult(
            operation=self.name,
            summary=f"Reset view to LOD {scene.active_lod} defaults; {len(deltas)} changes.",
            deltas=tuple(deltas),
            parameters={"lod": scene.active_lod, "default_opacity": self.default_opacity},
        )


@dataclass(slots=True)
class SetMaterial(SceneOperation):
    """Set semantic material state on entities.

    Semantic only: this records what the material *is*, not how a renderer should
    shade it. The material decoder that turns this into appearance is future work.
    """

    entity_ids: tuple[str, ...]
    material: MaterialState

    name: ClassVar[str] = "set_material"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the material change."""
        targets = self._expand(scene, self.entity_ids)
        deltas = [
            delta
            for entity_id in targets
            if (delta := scene.set_entity_material(entity_id, self.material))
        ]
        return OperationResult(
            operation=self.name,
            summary=f"Set material on {len(deltas)} entities.",
            deltas=tuple(deltas),
            parameters={"targets": list(targets), "material": self.material.to_dict()},
        )


@dataclass(slots=True)
class SetTransform(SceneOperation):
    """Set the rigid transform of entities."""

    entity_ids: tuple[str, ...]
    transform: Transform

    name: ClassVar[str] = "set_transform"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the transform change."""
        targets = self._expand(scene, self.entity_ids)
        deltas = [
            delta
            for entity_id in targets
            if (delta := scene.set_entity_transform(entity_id, self.transform))
        ]
        return OperationResult(
            operation=self.name,
            summary=f"Set transform on {len(deltas)} entities.",
            deltas=tuple(deltas),
            parameters={"targets": list(targets), "transform": self.transform.to_dict()},
        )


@dataclass(slots=True)
class BindAnimation(SceneOperation):
    """Bind entities to an animation clip.

    Semantic binding only: which entities participate in which clip, in which
    role and phase. No geometry is deformed; the animation decoder is future work.
    """

    bindings: Mapping[str, AnimationState]
    clip_id: str
    summary_note: str | None = None

    name: ClassVar[str] = "bind_animation"

    def validate(self, scene: AWRScene) -> None:
        """Check every bound entity exists."""
        if not self.bindings:
            raise OperationError("An animation binding must name at least one entity.")
        for entity_id in self.bindings:
            scene.get(entity_id)

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the animation bindings."""
        deltas = [
            delta
            for entity_id, animation in self.bindings.items()
            if (delta := scene.set_entity_animation(entity_id, animation))
        ]
        note = f" {self.summary_note}" if self.summary_note else ""
        return OperationResult(
            operation=self.name,
            summary=f"Bound {len(deltas)} entities to clip {self.clip_id!r}.{note}",
            deltas=tuple(deltas),
            parameters={"clip_id": self.clip_id, "entities": list(self.bindings)},
        )


@dataclass(slots=True)
class ClearAnimation(SceneOperation):
    """Remove animation bindings from entities (or from the whole scene)."""

    entity_ids: tuple[str, ...] | None = None

    name: ClassVar[str] = "clear_animation"

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the clear."""
        targets = (
            tuple(self.entity_ids)
            if self.entity_ids
            else tuple(
                e.entity_id for e in scene.iter_entities() if e.animation_state.is_animated
            )
        )
        deltas = [
            delta
            for entity_id in targets
            if (delta := scene.set_entity_animation(entity_id, AnimationState()))
        ]
        return OperationResult(
            operation=self.name,
            summary=f"Cleared animation on {len(deltas)} entities.",
            deltas=tuple(deltas),
            parameters={"targets": list(targets)},
        )


@dataclass(slots=True)
class AttachGeometryReferences(SceneOperation):
    """Attach geometry component references to entities.

    Used when a decoder allocates or fills component slots. Attaching a reference
    never changes an entity id, which is what preserves part identity across
    regeneration.
    """

    references: Mapping[str, GeometryReference]

    name: ClassVar[str] = "attach_geometry"

    def validate(self, scene: AWRScene) -> None:
        """Check entities exist and no component is claimed twice."""
        claimed: dict[str, str] = {}
        for entity_id, reference in self.references.items():
            scene.get(entity_id)
            owner = claimed.get(reference.component_id)
            if owner is not None:
                raise OperationError(
                    f"Geometry component {reference.component_id!r} would be attached to both "
                    f"{owner!r} and {entity_id!r}."
                )
            claimed[reference.component_id] = entity_id

    def apply(self, scene: AWRScene) -> OperationResult:
        """Apply the geometry references."""
        deltas = [
            delta
            for entity_id, reference in self.references.items()
            if (delta := scene.set_entity_geometry_reference(entity_id, reference))
        ]
        return OperationResult(
            operation=self.name,
            summary=f"Attached geometry references to {len(deltas)} entities.",
            deltas=tuple(deltas),
            parameters={"entities": list(self.references)},
        )


# ---------------------------------------------------------------------------
# Declared but unimplemented operations.
#
# These are the operations the editing model must eventually support. They are
# declared so the roadmap lives in typed code rather than in a document, and they
# raise on construction so nothing can depend on a fake implementation.
# ---------------------------------------------------------------------------


class _PlannedOperation(SceneOperation):
    """Base for operations that are declared but not implemented yet."""

    planned_in: ClassVar[str] = "a later R&D step"
    rationale: ClassVar[str] = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            f"{type(self).__name__} ({self.rationale})" if self.rationale else type(self).__name__,
            planned_in=self.planned_in,
        )

    def apply(self, scene: AWRScene) -> OperationResult:  # pragma: no cover - construction raises
        """Unreachable: construction always raises."""
        raise NotYetImplementedError(type(self).__name__, planned_in=self.planned_in)


class SegmentEntity(_PlannedOperation):
    """Split one entity into finer sub-entities at runtime."""

    name: ClassVar[str] = "segment"
    planned_in: ClassVar[str] = "ontology v0.2 and Step 5"
    rationale: ClassVar[str] = "requires sub-entity ontology and geometry segmentation"


class CrossSection(_PlannedOperation):
    """Cut the scene with a plane and expose interior surfaces."""

    name: ClassVar[str] = "cross_section"
    planned_in: ClassVar[str] = "after the geometry decoder exists"
    rationale: ClassVar[str] = "needs real geometry to cut"


class ExplodeView(_PlannedOperation):
    """Spatially separate components to show how they fit together."""

    name: ClassVar[str] = "explode"
    planned_in: ClassVar[str] = "after the geometry decoder exists"
    rationale: ClassVar[str] = "needs geometry extents to compute offsets"


class AddLabel(_PlannedOperation):
    """Attach a displayed label or annotation to an entity."""

    name: ClassVar[str] = "add_label"
    planned_in: ClassVar[str] = "the presentation layer step"
    rationale: ClassVar[str] = "label placement is a rendering concern not yet designed"


class ReplaceComponent(_PlannedOperation):
    """Replace one entity's geometry with an alternative variant."""

    name: ClassVar[str] = "replace_component"
    planned_in: ClassVar[str] = "Step 5"
    rationale: ClassVar[str] = "requires a geometry decoder and variant library"


class RegenerateComponent(_PlannedOperation):
    """Regenerate a single component without rebuilding the scene."""

    name: ClassVar[str] = "regenerate_component"
    planned_in: ClassVar[str] = "Step 5"
    rationale: ClassVar[str] = (
        "the architectural requirement is already fixed: regeneration must be per entity"
    )


class EditGeometry(_PlannedOperation):
    """Apply a direct geometric edit to one component."""

    name: ClassVar[str] = "edit_geometry"
    planned_in: ClassVar[str] = "after the geometry decoder exists"
    rationale: ClassVar[str] = "no geometry exists to edit"


PLANNED_OPERATIONS: tuple[type[_PlannedOperation], ...] = (
    SegmentEntity,
    CrossSection,
    ExplodeView,
    AddLabel,
    ReplaceComponent,
    RegenerateComponent,
    EditGeometry,
)
"""Operations the editing model must eventually support. Declared, not implemented."""
