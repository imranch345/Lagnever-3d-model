"""Scene editor: the single place where scene state changes are committed.

The editor applies :class:`~editing.operations.SceneOperation` objects to an
:class:`~awr.scene.AWRScene` and records each one as a versioned history event.
Nothing else in the system is allowed to commit history, which is what makes
scene history a complete and trustworthy log of everything that happened.

A no-op command (hiding something already hidden) produces an
:class:`~editing.operations.OperationResult` with no deltas and does **not** bump
the scene version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from awr.scene import AWRScene, SceneEvent
from awr.schema import (
    AnimationState,
    EducationalLevel,
    GeometryReference,
    LodLevel,
    MaterialState,
)
from editing.operations import (
    AttachGeometryReferences,
    BindAnimation,
    ClearAnimation,
    HideEntities,
    IsolateEntities,
    OperationResult,
    ResetToLodDefaults,
    RevealEntities,
    SceneOperation,
    SetEducationalLevel,
    SetLod,
    SetOpacity,
    SetVisibility,
    ShowEntities,
)

__all__ = ["AppliedOperation", "SceneEditor"]


@dataclass(frozen=True, slots=True)
class AppliedOperation:
    """An operation result together with the history event it produced."""

    result: OperationResult
    event: SceneEvent | None

    @property
    def committed(self) -> bool:
        """Whether the operation produced a history event."""
        return self.event is not None

    @property
    def version(self) -> int | None:
        """Scene version after the operation, if it changed anything."""
        return self.event.version if self.event else None


class SceneEditor:
    """Applies operations to one scene and records history."""

    def __init__(self, scene: AWRScene) -> None:
        self.scene = scene

    # ------------------------------------------------------------------
    def apply(
        self, operation: SceneOperation, *, command_text: str | None = None
    ) -> AppliedOperation:
        """Validate, apply and commit one operation."""
        operation.validate(self.scene)
        result = operation.apply(self.scene)
        event: SceneEvent | None = None
        if result.changed:
            event = self.scene.commit(
                operation=result.operation,
                summary=result.summary,
                deltas=result.deltas,
                command_text=command_text,
                parameters=result.parameters,
            )
        return AppliedOperation(result=result, event=event)

    def apply_all(
        self, operations: Sequence[SceneOperation], *, command_text: str | None = None
    ) -> tuple[AppliedOperation, ...]:
        """Apply several operations in order."""
        return tuple(self.apply(op, command_text=command_text) for op in operations)

    # ------------------------------------------------------------------
    # Convenience wrappers. These exist so that programmatic callers and tests
    # do not have to build operation objects for the common cases.
    # ------------------------------------------------------------------
    def show(self, *entity_ids: str, command_text: str | None = None) -> AppliedOperation:
        """Make entities visible."""
        return self.apply(ShowEntities(entity_ids), command_text=command_text)

    def hide(self, *entity_ids: str, command_text: str | None = None) -> AppliedOperation:
        """Make entities invisible."""
        return self.apply(HideEntities(entity_ids), command_text=command_text)

    def reveal(self, *entity_ids: str, command_text: str | None = None) -> AppliedOperation:
        """Show entities together with their descendants."""
        return self.apply(RevealEntities(entity_ids), command_text=command_text)

    def set_visibility(
        self, entity_ids: Sequence[str], visible: bool, *, command_text: str | None = None
    ) -> AppliedOperation:
        """Set visibility explicitly."""
        return self.apply(SetVisibility(tuple(entity_ids), visible), command_text=command_text)

    def set_opacity(
        self,
        entity_ids: Sequence[str],
        opacity: float,
        *,
        preset_name: str | None = None,
        command_text: str | None = None,
    ) -> AppliedOperation:
        """Set opacity on entities."""
        return self.apply(
            SetOpacity(tuple(entity_ids), opacity, preset_name=preset_name),
            command_text=command_text,
        )

    def isolate(self, *entity_ids: str, command_text: str | None = None) -> AppliedOperation:
        """Show only the given entities."""
        return self.apply(IsolateEntities(entity_ids), command_text=command_text)

    def set_lod(self, lod: LodLevel, *, command_text: str | None = None) -> AppliedOperation:
        """Change the active level of detail."""
        return self.apply(SetLod(lod), command_text=command_text)

    def set_educational_level(
        self, level: EducationalLevel, *, command_text: str | None = None
    ) -> AppliedOperation:
        """Change the audience level (and the LOD it maps to)."""
        return self.apply(SetEducationalLevel(level), command_text=command_text)

    def reset_view(
        self, *, default_opacity: float = 1.0, command_text: str | None = None
    ) -> AppliedOperation:
        """Discard manual visibility and opacity edits."""
        return self.apply(
            ResetToLodDefaults(default_opacity=default_opacity), command_text=command_text
        )

    def set_material(
        self, entity_ids: Sequence[str], material: MaterialState, *, command_text: str | None = None
    ) -> AppliedOperation:
        """Set semantic material state."""
        from editing.operations import SetMaterial

        return self.apply(SetMaterial(tuple(entity_ids), material), command_text=command_text)

    def bind_animation(
        self,
        bindings: Mapping[str, AnimationState],
        clip_id: str,
        *,
        summary_note: str | None = None,
        command_text: str | None = None,
    ) -> AppliedOperation:
        """Bind entities to an animation clip."""
        return self.apply(
            BindAnimation(bindings=bindings, clip_id=clip_id, summary_note=summary_note),
            command_text=command_text,
        )

    def clear_animation(
        self, entity_ids: Sequence[str] | None = None, *, command_text: str | None = None
    ) -> AppliedOperation:
        """Remove animation bindings."""
        return self.apply(
            ClearAnimation(tuple(entity_ids) if entity_ids else None), command_text=command_text
        )

    def attach_geometry(
        self, references: Mapping[str, GeometryReference], *, command_text: str | None = None
    ) -> AppliedOperation:
        """Attach geometry component references to entities."""
        return self.apply(AttachGeometryReferences(references), command_text=command_text)

    # ------------------------------------------------------------------
    def history_summary(self) -> list[dict[str, Any]]:
        """Compact view of scene history, for reports and tests."""
        return [
            {
                "version": event.version,
                "operation": event.operation,
                "command": event.command_text,
                "entities_changed": len(event.touched_entities),
                "summary": event.summary,
            }
            for event in self.scene.history
        ]
