"""Ontology loading, identity, hierarchy and relationship integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from awr.errors import EntityNotFoundError, OntologyError
from awr.ontology import AnatomyOntology, load_ontology
from awr.relationships import GraphKind
from awr.schema import AnatomyType
from awr.validation import validate_ontology
from evaluation.anatomy import (
    MANDATORY_HEART_ENTITIES,
    REQUIRED_FUNCTIONAL_EDGES,
    evaluate_anatomy,
)


def test_ontology_loads(ontology: AnatomyOntology) -> None:
    """The heart ontology loads and reports itself as an unvalidated draft."""
    assert ontology.ontology_id == "lagnav.heart"
    assert ontology.version == "0.1.0"
    assert ontology.root_id == "heart"
    assert len(ontology) == 42
    assert ontology.provenance.medically_validated is False
    assert ontology.provenance.is_draft is True


def test_all_mandatory_entities_present(ontology: AnatomyOntology) -> None:
    """Every entity named in the specification exists."""
    missing = [entity_id for entity_id in MANDATORY_HEART_ENTITIES if entity_id not in ontology]
    assert missing == []


def test_entity_ids_are_unique_and_namespaced(ontology: AnatomyOntology) -> None:
    """Ids are unique and live under the root namespace."""
    ids = ontology.ids()
    assert len(ids) == len(set(ids))
    for entity_id in ids:
        assert entity_id == "heart" or entity_id.startswith("heart.")


def test_hierarchy_is_a_single_parent_tree(ontology: AnatomyOntology) -> None:
    """Every entity has exactly one parent and reaches the root."""
    for entity in ontology.iter_entities():
        entity_id = entity.entity_id
        parent = ontology.parent(entity_id)
        if entity_id == ontology.root_id:
            assert parent is None
            continue
        assert parent is not None
        assert entity_id in ontology.children(parent)
        assert ontology.root_id in ontology.ancestors(entity_id)


def test_every_entity_is_reachable_from_the_root(ontology: AnatomyOntology) -> None:
    """Walking down from the root reaches all 42 entities."""
    reachable = set(ontology.descendants(ontology.root_id, include_self=True))
    assert reachable == set(ontology.ids())


def test_relationship_targets_exist(ontology: AnatomyOntology) -> None:
    """No edge points at an entity that is not in the ontology."""
    known = set(ontology.ids())
    for edge in ontology.relationships.all_edges():
        assert edge.subject in known
        assert edge.object in known


def test_required_blood_flow_edges_are_declared(ontology: AnatomyOntology) -> None:
    """The blood-flow relationships the specification lists are present."""
    for subject, relation, obj in REQUIRED_FUNCTIONAL_EDGES:
        assert obj in ontology.relationships.related(subject, relation)


def test_valves_sit_between_the_right_structures(ontology: AnatomyOntology) -> None:
    """Each valve opens from the expected chamber into the expected target."""
    expected = {
        "heart.tricuspid_valve": ("heart.right_atrium", "heart.right_ventricle"),
        "heart.pulmonary_valve": ("heart.right_ventricle", "heart.pulmonary_trunk"),
        "heart.mitral_valve": ("heart.left_atrium", "heart.left_ventricle"),
        "heart.aortic_valve": ("heart.left_ventricle", "heart.aorta"),
    }
    functional = ontology.relationships.graph(GraphKind.FUNCTIONAL)
    for valve_id, (upstream, downstream) in expected.items():
        assert downstream in functional.related(valve_id, "opens_into")
        assert upstream in [edge.subject for edge in functional.incoming(valve_id, "opens_into")]


def test_groups_are_not_renderable_and_have_children(ontology: AnatomyOntology) -> None:
    """Organisational groups contain something and are never drawn themselves."""
    groups = ontology.groups()
    assert len(groups) == 6
    for group_id in groups:
        group = ontology.get(group_id)
        assert group.anatomy_type is AnatomyType.GROUP
        assert group.renderable is False
        assert ontology.children(group_id)


def test_ontology_validation_reports_no_errors(ontology: AnatomyOntology) -> None:
    """The reporting validator finds no errors in the shipped ontology."""
    report = validate_ontology(ontology)
    assert report.is_valid, str(report)
    assert report.warnings == ()


def test_anatomy_coverage_evaluation_passes(ontology: AnatomyOntology) -> None:
    """Coverage evaluation passes and states that it is not a medical check."""
    report = evaluate_anatomy(ontology)
    assert report.is_valid, str(report)
    assert any(issue.code == "COVERAGE_ONLY" for issue in report.issues)


def test_unknown_entity_lookup_raises(ontology: AnatomyOntology) -> None:
    """Looking up a missing id fails loudly."""
    with pytest.raises(EntityNotFoundError) as exc:
        ontology.get("heart.spleen")
    assert "heart.spleen" in str(exc.value)


def test_version_mismatch_is_rejected(config) -> None:
    """A configuration that expects another ontology version is an error."""
    with pytest.raises(OntologyError, match="version mismatch"):
        load_ontology(config.domain.ontology_dir, expected_version="9.9.9")


def _write_ontology(tmp_path: Path, mutate) -> Path:
    source = Path("ontology/heart")
    target = tmp_path / "heart"
    target.mkdir()
    payloads = {
        name: json.loads((source / name).read_text(encoding="utf-8"))
        for name in ("anatomy.json", "hierarchy.json", "relationships.json")
    }
    mutate(payloads)
    for name, payload in payloads.items():
        (target / name).write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_duplicate_entity_id_is_rejected(tmp_path: Path) -> None:
    """A duplicated id fails at load time rather than silently overwriting."""

    def mutate(payloads: dict) -> None:
        entities = payloads["anatomy.json"]["entities"]
        entities.append(dict(entities[3]))

    with pytest.raises(OntologyError, match="Duplicate entity id"):
        load_ontology(_write_ontology(tmp_path, mutate))


def test_two_parents_are_rejected(tmp_path: Path) -> None:
    """An entity listed under two parents fails at load time."""

    def mutate(payloads: dict) -> None:
        payloads["hierarchy.json"]["nodes"][1]["children"].append("heart.aorta")

    with pytest.raises(OntologyError, match="more than one parent"):
        load_ontology(_write_ontology(tmp_path, mutate))


def test_dangling_relationship_is_rejected(tmp_path: Path) -> None:
    """An edge pointing at a missing entity fails at load time."""

    def mutate(payloads: dict) -> None:
        payloads["relationships.json"]["edges"].append(
            {"subject": "heart.left_ventricle", "relation": "pumps_to", "object": "heart.trachea"}
        )

    with pytest.raises(OntologyError, match="unknown object"):
        load_ontology(_write_ontology(tmp_path, mutate))


def test_unregistered_relation_is_rejected(tmp_path: Path) -> None:
    """A relation name that is not in the registry fails at load time."""

    def mutate(payloads: dict) -> None:
        payloads["relationships.json"]["edges"].append(
            {"subject": "heart.aorta", "relation": "vibes_with", "object": "heart.left_ventricle"}
        )

    with pytest.raises(OntologyError, match="unregistered relation"):
        load_ontology(_write_ontology(tmp_path, mutate))


def test_version_disagreement_between_files_is_rejected(tmp_path: Path) -> None:
    """The three ontology files must agree on their version."""

    def mutate(payloads: dict) -> None:
        payloads["hierarchy.json"]["version"] = "0.2.0"

    with pytest.raises(OntologyError, match="disagree on version"):
        load_ontology(_write_ontology(tmp_path, mutate))


def test_unknown_anatomy_type_is_rejected(tmp_path: Path) -> None:
    """An unregistered anatomy type is a data error, not a new concept."""

    def mutate(payloads: dict) -> None:
        payloads["anatomy.json"]["entities"][7]["anatomy_type"] = "sparkle"

    with pytest.raises(OntologyError, match="anatomy_type"):
        load_ontology(_write_ontology(tmp_path, mutate))
