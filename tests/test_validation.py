"""Validation framework: reporting, scene invariants and corruption detection."""

from __future__ import annotations

import pytest

from awr.errors import LagnavError
from awr.ontology import AnatomyOntology
from awr.relationships import Relationship
from awr.scene import AWRScene, SceneEvent
from awr.schema import GeometryReference
from awr.validation import (
    Severity,
    ValidationReport,
    validate_ontology,
    validate_scene,
    validate_scene_roundtrip,
)


def test_report_separates_severities() -> None:
    """A report counts errors, warnings and notes separately."""
    report = ValidationReport(subject="test")
    report.add("A", Severity.ERROR, "bad", ["heart"])
    report.add("B", Severity.WARNING, "odd")
    report.add("C", Severity.INFO, "note")
    assert report.is_valid is False
    assert report.summary() == {
        "subject": "test",
        "valid": False,
        "errors": 1,
        "warnings": 1,
        "info": 1,
        "total": 3,
    }
    with pytest.raises(LagnavError, match="Validation failed"):
        report.raise_if_invalid()


def test_clean_report_does_not_raise(ontology: AnatomyOntology) -> None:
    """A valid subject passes raise_if_invalid."""
    validate_ontology(ontology).raise_if_invalid()


def test_shared_geometry_component_is_detected(scene: AWRScene) -> None:
    """Two entities claiming one component is an error."""
    reference = GeometryReference(component_id="geometry_part_0000")
    scene.entities["heart.aorta"].state.geometry_reference = reference
    scene.entities["heart.left_ventricle"].state.geometry_reference = reference
    report = validate_scene(scene)
    assert any(issue.code == "SHARED_GEOMETRY_COMPONENT" for issue in report.errors)


def test_broken_tree_link_is_detected(scene: AWRScene) -> None:
    """A parent link that does not point back is an error."""
    entity = scene.entities["heart.aorta"]
    scene.entities["heart.aorta"] = entity.with_children(())
    scene.entities["heart.aorta"].parent_id = "heart.chambers"
    report = validate_scene(scene)
    assert any(issue.code == "BROKEN_PARENT_LINK" for issue in report.errors)


def test_visible_group_is_detected(scene: AWRScene) -> None:
    """A group marked visible is an error even if set directly."""
    scene.entities["heart.valves"].state.visibility = True
    assert any(issue.code == "VISIBLE_NON_RENDERABLE" for issue in validate_scene(scene).errors)


def test_history_gap_is_detected(scene: AWRScene) -> None:
    """Non-contiguous history versions are reported."""
    scene.history.append(SceneEvent(version=99, operation="fake", summary="injected"))
    report = validate_scene(scene)
    assert any(issue.code == "HISTORY_NOT_MONOTONIC" for issue in report.errors)


def test_dangling_relationship_in_a_scene_is_detected(scene: AWRScene) -> None:
    """A relationship pointing outside the scene is an error."""
    scene.relationships.add(Relationship("heart.aorta", "connects_to", "heart.trachea"))
    assert any(issue.code == "DANGLING_RELATION" for issue in validate_scene(scene).errors)


def test_roundtrip_validation_detects_state_loss(scene: AWRScene, monkeypatch) -> None:
    """A serialisation that loses state is caught."""
    original = AWRScene.from_dict

    def lossy(payload):
        restored = original(payload)
        restored.entities["heart.aorta"].state.opacity = 0.5
        return restored

    monkeypatch.setattr(AWRScene, "from_dict", staticmethod(lossy))
    report = validate_scene_roundtrip(scene)
    assert any(issue.code == "STATE_CHANGED" for issue in report.errors)


def test_ontology_validation_flags_ambiguous_surface_forms(ontology: AnatomyOntology) -> None:
    """Two entities sharing a phrase is reported as a resolver risk."""
    from dataclasses import replace

    entities = dict(ontology.entities)
    entities["heart.aorta"] = replace(entities["heart.aorta"], synonyms=("left ventricle",))
    mutated = AnatomyOntology(
        ontology_id=ontology.ontology_id,
        title=ontology.title,
        version=ontology.version,
        root_id=ontology.root_id,
        entities=entities,
        parent_of=ontology.parent_of,
        children_of=ontology.children_of,
        relationships=ontology.relationships,
        registry=ontology.registry,
        provenance=ontology.provenance,
    )
    report = validate_ontology(mutated)
    assert any(issue.code == "AMBIGUOUS_SURFACE_FORM" for issue in report.warnings)
