"""Consistency evaluation: identity, edits and persistence.

These are the properties the first milestone actually claims, so they are the
ones that are measured:

* **Entity identity** survives every edit.
* **Edit consistency**: a command changes what it said it would change and
  nothing else.
* **Scene persistence**: a scene saved and reloaded is the same scene, history
  included.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from awr.scene import AWRScene
from awr.validation import Severity, ValidationReport, validate_scene_roundtrip

__all__ = [
    "evaluate_identity_stability",
    "evaluate_edit_consistency",
    "evaluate_persistence",
    "evaluate_history_integrity",
]


def evaluate_identity_stability(
    before_ids: Sequence[str], after: AWRScene, *, context: str = "edit"
) -> ValidationReport:
    """Check that no entity id was added, removed or renamed by an edit."""
    report = ValidationReport(subject=f"identity stability ({context})")
    before = tuple(before_ids)
    now = after.ids()
    if before == now:
        return report
    missing = sorted(set(before) - set(now))
    added = sorted(set(now) - set(before))
    if missing:
        report.add(
            "ENTITY_ID_LOST",
            Severity.ERROR,
            "Entity ids disappeared; identity must survive every edit.",
            missing,
        )
    if added:
        report.add(
            "ENTITY_ID_ADDED",
            Severity.ERROR,
            "New entity ids appeared during an edit; edits change state, not identity.",
            added,
        )
    if not missing and not added:
        report.add(
            "ENTITY_ORDER_CHANGED",
            Severity.WARNING,
            "Entity order changed; identity is intact but iteration order is not stable.",
        )
    return report


def evaluate_edit_consistency(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    expected_changed: Iterable[str] = (),
    context: str = "edit",
) -> ValidationReport:
    """Compare two scene snapshots against the entities an edit claimed to touch."""
    report = ValidationReport(subject=f"edit consistency ({context})")
    expected = set(expected_changed)

    if set(before) != set(after):
        report.add(
            "ENTITY_SET_CHANGED",
            Severity.ERROR,
            "The entity set changed during an edit.",
        )

    changed = {
        entity_id
        for entity_id in before
        if entity_id in after and before[entity_id] != after[entity_id]
    }
    unexpected = sorted(changed - expected)
    if unexpected:
        report.add(
            "UNDECLARED_CHANGE",
            Severity.ERROR,
            "Entities changed that the operation did not report in its deltas.",
            unexpected,
        )
    unchanged = sorted(expected - changed)
    if unchanged:
        report.add(
            "DECLARED_BUT_UNCHANGED",
            Severity.WARNING,
            "The operation reported changes to entities whose state is identical.",
            unchanged,
        )
    return report


def evaluate_persistence(scene: AWRScene) -> ValidationReport:
    """Check that the scene round-trips through serialisation unchanged."""
    return validate_scene_roundtrip(scene)


def evaluate_history_integrity(scene: AWRScene) -> ValidationReport:
    """Check that history is a complete, contiguous log of the scene's changes."""
    report = ValidationReport(subject=f"history integrity {scene.scene_id}")
    versions = [event.version for event in scene.history]
    if versions != list(range(1, len(versions) + 1)):
        report.add(
            "NON_CONTIGUOUS_HISTORY",
            Severity.ERROR,
            f"History versions are not contiguous: {versions}.",
        )
    if scene.version != len(versions):
        report.add(
            "VERSION_DRIFT",
            Severity.ERROR,
            f"Scene version {scene.version} does not match {len(versions)} recorded events.",
        )
    for event in scene.history:
        for delta in event.deltas:
            if delta.entity_id not in scene:
                report.add(
                    "HISTORY_REFERENCES_MISSING_ENTITY",
                    Severity.ERROR,
                    f"Event {event.version} references an entity that no longer exists.",
                    [delta.entity_id],
                )
    return report
