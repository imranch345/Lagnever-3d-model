"""Validation utilities for ontologies and scenes.

Two flavours of checking exist in the project and they are deliberately kept
apart:

* The loaders raise immediately on anything that would produce a corrupt object
  (missing entity, two parents, unregistered relation).
* The validators here *report*. They walk a fully built ontology or scene and
  return a :class:`ValidationReport` of issues with severities, so that quality
  problems which are not fatal (a synonym shared by two entities, a group with
  no children) can be surfaced, counted and tracked over time.

Scope for this milestone, as specified: ontology validity, relationship validity,
entity identity, scene persistence and edit consistency. Geometry, topology,
multi-view and educational-appropriateness checks are declared in
``evaluation/`` and intentionally not implemented yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from awr.errors import LagnavError
from awr.ontology import AnatomyOntology
from awr.scene import AWRScene

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "validate_ontology",
    "validate_scene",
    "validate_scene_roundtrip",
]


class Severity(StrEnum):
    """How serious a validation finding is."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One finding produced by a validator."""

    code: str
    severity: Severity
    message: str
    entity_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        where = f" [{', '.join(self.entity_ids)}]" if self.entity_ids else ""
        return f"{self.severity.upper()} {self.code}: {self.message}{where}"


@dataclass(slots=True)
class ValidationReport:
    """Collected findings for one validation run."""

    subject: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        entity_ids: Iterable[str] = (),
    ) -> None:
        """Append one finding."""
        self.issues.append(ValidationIssue(code, severity, message, tuple(entity_ids)))

    def extend(self, other: ValidationReport) -> None:
        """Merge another report into this one."""
        self.issues.extend(other.issues)

    def by_severity(self, severity: Severity) -> tuple[ValidationIssue, ...]:
        """All findings of one severity."""
        return tuple(issue for issue in self.issues if issue.severity is severity)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """All error-level findings."""
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """All warning-level findings."""
        return self.by_severity(Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        """True when no error-level findings were recorded."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Raise :class:`~awr.errors.LagnavError` if any error was recorded."""
        if self.is_valid:
            return
        details = "\n  ".join(str(issue) for issue in self.errors)
        raise LagnavError(f"Validation failed for {self.subject}:\n  {details}")

    def summary(self) -> dict[str, Any]:
        """Counts per severity plus the subject, for reports and tests."""
        return {
            "subject": self.subject,
            "valid": self.is_valid,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len(self.by_severity(Severity.INFO)),
            "total": len(self.issues),
        }

    def __str__(self) -> str:
        if not self.issues:
            return f"{self.subject}: no issues"
        lines = [f"{self.subject}: {len(self.issues)} issue(s)"]
        lines.extend(f"  {issue}" for issue in self.issues)
        return "\n".join(lines)


def validate_ontology(ontology: AnatomyOntology) -> ValidationReport:
    """Check identity, hierarchy, LOD policy and relationship integrity."""
    report = ValidationReport(subject=f"ontology {ontology.ontology_id}@{ontology.version}")

    # --- identity --------------------------------------------------------
    canonical_seen: dict[str, str] = {}
    surface_seen: dict[str, set[str]] = {}
    for entity in ontology.iter_entities():
        if not entity.entity_id.strip():
            report.add("EMPTY_ID", Severity.ERROR, "An entity has an empty id.")
        if entity.entity_id != ontology.root_id and not entity.entity_id.startswith(
            f"{ontology.root_id}."
        ):
            report.add(
                "ID_NAMESPACE",
                Severity.WARNING,
                f"Entity id does not use the {ontology.root_id!r} namespace prefix.",
                [entity.entity_id],
            )
        previous = canonical_seen.get(entity.canonical_name)
        if previous:
            report.add(
                "DUPLICATE_CANONICAL_NAME",
                Severity.ERROR,
                f"Canonical name {entity.canonical_name!r} is used by two entities.",
                [previous, entity.entity_id],
            )
        canonical_seen[entity.canonical_name] = entity.entity_id
        for form in entity.surface_forms():
            surface_seen.setdefault(form.strip().lower(), set()).add(entity.entity_id)

    for form, owners in surface_seen.items():
        if len(owners) > 1:
            report.add(
                "AMBIGUOUS_SURFACE_FORM",
                Severity.WARNING,
                f"The phrase {form!r} maps to several entities; the resolver will refuse to guess.",
                sorted(owners),
            )

    # --- hierarchy -------------------------------------------------------
    for entity in ontology.iter_entities():
        entity_id = entity.entity_id
        parent = ontology.parent_of.get(entity_id)
        if parent is None and entity_id != ontology.root_id:
            report.add(
                "ORPHAN_ENTITY", Severity.ERROR, "Entity has no parent and is not the root.",
                [entity_id],
            )
        if parent is not None and entity_id not in ontology.children_of.get(parent, ()):
            report.add(
                "BROKEN_PARENT_LINK",
                Severity.ERROR,
                f"Entity is not listed among the children of its parent {parent!r}.",
                [entity_id],
            )
        for child in ontology.children_of.get(entity_id, ()):
            if ontology.parent_of.get(child) != entity_id:
                report.add(
                    "BROKEN_CHILD_LINK",
                    Severity.ERROR,
                    f"Child {child!r} does not point back to this parent.",
                    [entity_id, child],
                )
        if entity.is_group:
            if entity.renderable:
                report.add(
                    "RENDERABLE_GROUP",
                    Severity.ERROR,
                    "Organisational groups must not be renderable.",
                    [entity_id],
                )
            if not ontology.children_of.get(entity_id):
                report.add(
                    "EMPTY_GROUP", Severity.ERROR, "Organisational group has no children.",
                    [entity_id],
                )
        if parent is not None:
            parent_entity = ontology.get(parent)
            if entity.lod_policy.min_lod < parent_entity.lod_policy.min_lod:
                report.add(
                    "LOD_INVERSION",
                    Severity.WARNING,
                    f"Entity appears at LOD {entity.lod_policy.min_lod} but its parent "
                    f"{parent!r} only appears at LOD {parent_entity.lod_policy.min_lod}.",
                    [entity_id, parent],
                )

    # --- relationships ---------------------------------------------------
    known = set(ontology.ids())
    for edge in ontology.relationships.all_edges():
        for role, value in (("subject", edge.subject), ("object", edge.object)):
            if value not in known:
                report.add(
                    "DANGLING_RELATION",
                    Severity.ERROR,
                    f"Relation {edge.relation!r} has an unknown {role} {value!r}.",
                    [edge.subject, edge.object],
                )
        if edge.relation not in ontology.registry:
            report.add(
                "UNREGISTERED_RELATION",
                Severity.ERROR,
                f"Relation type {edge.relation!r} is not registered.",
                [edge.subject, edge.object],
            )
            continue
        spec = ontology.registry.get(edge.relation)
        if spec.symmetric:
            mirrored = (edge.object, edge.relation, edge.subject)
            if any(other.key == mirrored for other in ontology.relationships.all_edges()):
                report.add(
                    "REDUNDANT_SYMMETRIC_EDGE",
                    Severity.WARNING,
                    f"Symmetric relation {edge.relation!r} is declared in both directions.",
                    [edge.subject, edge.object],
                )

    if not ontology.provenance.medically_validated:
        report.add(
            "NOT_MEDICALLY_VALIDATED",
            Severity.INFO,
            "This ontology is a draft and has not been clinically reviewed.",
        )
    return report


def validate_scene(scene: AWRScene) -> ValidationReport:
    """Check scene invariants: identity, tree, state ranges and geometry links."""
    report = ValidationReport(subject=f"scene {scene.scene_id}")

    component_owner: dict[str, str] = {}
    for key, entity in scene.entities.items():
        if key != entity.entity_id:
            report.add(
                "ID_KEY_MISMATCH",
                Severity.ERROR,
                f"Scene stores entity {entity.entity_id!r} under key {key!r}.",
                [entity.entity_id],
            )
        try:
            entity.state.validate()
        except LagnavError as exc:
            report.add("INVALID_STATE", Severity.ERROR, str(exc), [entity.entity_id])
        if entity.visibility and not entity.renderable:
            report.add(
                "VISIBLE_NON_RENDERABLE",
                Severity.ERROR,
                "A non-renderable entity is marked visible.",
                [entity.entity_id],
            )
        parent_id = entity.parent_id
        if parent_id is not None:
            if parent_id not in scene:
                report.add(
                    "MISSING_PARENT", Severity.ERROR,
                    f"Parent {parent_id!r} is not present in the scene.", [entity.entity_id],
                )
            elif entity.entity_id not in scene.get(parent_id).children:
                report.add(
                    "BROKEN_PARENT_LINK",
                    Severity.ERROR,
                    f"Entity is not listed among the children of {parent_id!r}.",
                    [entity.entity_id],
                )
        for child in entity.children:
            if child not in scene:
                report.add(
                    "MISSING_CHILD", Severity.ERROR,
                    f"Child {child!r} is not present in the scene.", [entity.entity_id],
                )
        reference = entity.geometry_reference
        if reference is not None:
            owner = component_owner.get(reference.component_id)
            if owner is not None:
                report.add(
                    "SHARED_GEOMETRY_COMPONENT",
                    Severity.ERROR,
                    f"Geometry component {reference.component_id!r} is claimed by two entities; "
                    "the entity to geometry correspondence must stay unambiguous.",
                    [owner, entity.entity_id],
                )
            component_owner[reference.component_id] = entity.entity_id

    if scene.active_lod not in scene.lod_ladder:
        report.add(
            "INVALID_ACTIVE_LOD",
            Severity.ERROR,
            f"Active LOD {scene.active_lod} is not part of the configured ladder.",
        )
    if scene.root_id not in scene:
        report.add("MISSING_ROOT", Severity.ERROR, f"Root {scene.root_id!r} is absent.")

    expected_versions = list(range(1, len(scene.history) + 1))
    actual_versions = [event.version for event in scene.history]
    if actual_versions != expected_versions:
        report.add(
            "HISTORY_NOT_MONOTONIC",
            Severity.ERROR,
            f"Scene history versions {actual_versions} are not a contiguous sequence.",
        )
    if scene.history and scene.version != scene.history[-1].version:
        report.add(
            "VERSION_MISMATCH",
            Severity.ERROR,
            f"Scene version {scene.version} does not match the last event version "
            f"{scene.history[-1].version}.",
        )

    known = set(scene.ids())
    for edge in scene.relationships.all_edges():
        if edge.subject not in known or edge.object not in known:
            report.add(
                "DANGLING_RELATION",
                Severity.ERROR,
                f"Relation {edge.relation!r} refers to an entity outside the scene.",
                [edge.subject, edge.object],
            )
    return report


def validate_scene_roundtrip(scene: AWRScene) -> ValidationReport:
    """Check that a scene survives serialisation without losing identity or state."""
    report = ValidationReport(subject=f"scene roundtrip {scene.scene_id}")
    restored = AWRScene.from_dict(scene.to_dict())

    if restored.scene_id != scene.scene_id:
        report.add("SCENE_ID_CHANGED", Severity.ERROR, "Scene id changed across serialisation.")
    if restored.ids() != scene.ids():
        missing = sorted(set(scene.ids()) - set(restored.ids()))
        report.add(
            "ENTITY_SET_CHANGED",
            Severity.ERROR,
            f"Entity ids changed across serialisation; missing: {missing or 'none'}.",
        )
    if restored.state_snapshot() != scene.state_snapshot():
        differing = [
            entity_id
            for entity_id, before in scene.state_snapshot().items()
            if restored.state_snapshot().get(entity_id) != before
        ]
        report.add(
            "STATE_CHANGED", Severity.ERROR, "Entity state changed across serialisation.", differing
        )
    if [e.to_dict() for e in restored.history] != [e.to_dict() for e in scene.history]:
        report.add("HISTORY_CHANGED", Severity.ERROR, "Scene history changed across serialisation.")
    if restored.version != scene.version:
        report.add("VERSION_CHANGED", Severity.ERROR, "Scene version changed across serialisation.")
    return report


def report_for_snapshots(
    subject: str,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    expected_changed: Sequence[str] = (),
) -> ValidationReport:
    """Compare two scene snapshots and flag unexpected changes.

    Used by edit-consistency checks: an edit must touch exactly the entities it
    claims to touch, and must never add or remove entities.
    """
    report = ValidationReport(subject=subject)
    if set(before) != set(after):
        report.add(
            "ENTITY_SET_CHANGED",
            Severity.ERROR,
            "An edit added or removed entities; edits must only change state.",
        )
    changed = sorted(
        entity_id
        for entity_id in before
        if entity_id in after and before[entity_id] != after[entity_id]
    )
    unexpected = sorted(set(changed) - set(expected_changed))
    if expected_changed and unexpected:
        report.add(
            "UNEXPECTED_CHANGE",
            Severity.ERROR,
            "Entities changed that the operation did not declare.",
            unexpected,
        )
    return report
