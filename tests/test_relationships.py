"""The three graphs: separation, typed queries, symmetry, inverses, extension."""

from __future__ import annotations

import pytest

from awr.errors import RelationshipError, UnknownRelationTypeError
from awr.ontology import AnatomyOntology
from awr.relationships import (
    FlowMedium,
    FlowSemantics,
    Granularity,
    GraphKind,
    Relationship,
    RelationshipStore,
    RelationTypeSpec,
    default_relation_registry,
)


def test_registry_covers_the_specified_vocabulary() -> None:
    """Every relation type named in the specification is registered."""
    registry = default_relation_registry()
    expected = {
        GraphKind.STRUCTURE: {"part_of", "contains", "connects_to", "continuous_with"},
        GraphKind.SPATIAL: {
            "adjacent_to", "inside", "surrounds", "anterior_to", "posterior_to",
            "superior_to", "inferior_to", "left_of", "right_of",
        },
        GraphKind.FUNCTIONAL: {
            "receives_from", "pumps_to", "opens_into", "exits_into", "conducts_to"
        },
    }
    for kind, names in expected.items():
        assert {spec.name for spec in registry.by_graph(kind)} == names


def test_inverses_are_mutual_and_share_a_graph() -> None:
    """Declared inverses are consistent in both directions."""
    default_relation_registry().validate_inverses()


def test_graphs_stay_separate(ontology: AnatomyOntology) -> None:
    """Each graph holds only its own relation types and none is empty."""
    registry = ontology.registry
    for kind in GraphKind:
        graph = ontology.relationships.graph(kind)
        assert len(graph) > 0
        for edge in graph.edges:
            assert registry.graph_of(edge.relation) is kind


def test_structure_spatial_functional_example_edges(ontology: AnatomyOntology) -> None:
    """The three example edges from the specification are each in the right graph."""
    store = ontology.relationships
    structure = store.graph(GraphKind.STRUCTURE)
    assert "heart.left_ventricle" in structure.related("heart.chambers", "contains")
    assert "heart.interventricular_septum" in store.graph(GraphKind.SPATIAL).related(
        "heart.left_ventricle", "adjacent_to"
    )
    assert "heart.aorta" in store.graph(GraphKind.FUNCTIONAL).related(
        "heart.left_ventricle", "pumps_to"
    )


def test_hierarchy_derived_partonomy_is_not_duplicated(ontology: AnatomyOntology) -> None:
    """Organisational part_of edges are derived from the tree, once each."""
    derived = [e for e in ontology.relationships.structure.edges if e.derived_from_hierarchy]
    assert len(derived) == len(ontology) - 1
    assert all(edge.relation == "part_of" for edge in derived)
    keys = [edge.key for edge in ontology.relationships.all_edges()]
    assert len(keys) == len(set(keys))


def test_symmetric_relations_answer_in_both_directions(ontology: AnatomyOntology) -> None:
    """connects_to is declared once and queried from either side."""
    store = ontology.relationships
    assert "heart.right_ventricle" in store.related("heart.right_atrium", "connects_to")
    assert "heart.right_atrium" in store.related("heart.right_ventricle", "connects_to")


def test_inverse_relations_are_resolved_without_duplication(ontology: AnatomyOntology) -> None:
    """An inverse query is answered from the single stored direction."""
    store = ontology.relationships
    assert "heart.right_ventricle" in store.related("heart.right_atrium", "superior_to")
    assert "heart.right_atrium" in store.related("heart.right_ventricle", "inferior_to")
    assert store.graph(GraphKind.SPATIAL).outgoing("heart.right_ventricle", "inferior_to") == ()


def test_flow_semantics_and_medium_are_declared(ontology: AnatomyOntology) -> None:
    """Blood and impulse relations are distinguishable without name matching."""
    registry = ontology.registry
    assert registry.get("receives_from").flow is FlowSemantics.REVERSE
    assert registry.get("pumps_to").flow is FlowSemantics.FORWARD
    assert registry.get("conducts_to").medium is FlowMedium.IMPULSE
    assert registry.get("opens_into").medium is FlowMedium.BLOOD


def test_granularity_filters_parallel_paths(ontology: AnatomyOntology) -> None:
    """Summary and detailed flow edges do not mix in one query."""
    store = ontology.relationships
    summary = store.related("heart.right_atrium", "opens_into", granularity=Granularity.SUMMARY)
    detailed = store.related("heart.right_atrium", "opens_into", granularity=Granularity.DETAILED)
    assert summary == ("heart.right_ventricle",)
    assert detailed == ("heart.tricuspid_valve",)


def test_unknown_relation_type_raises() -> None:
    """Using an unregistered relation name fails with the registered list."""
    store = RelationshipStore()
    with pytest.raises(UnknownRelationTypeError, match="not registered"):
        store.add(Relationship("a", "teleports_to", "b"))


def test_self_edges_are_rejected() -> None:
    """An entity cannot be related to itself."""
    store = RelationshipStore()
    with pytest.raises(RelationshipError, match="Self-referential"):
        store.add(Relationship("a", "connects_to", "a"))


def test_duplicate_edges_are_ignored() -> None:
    """Adding the same triple twice inserts it once."""
    store = RelationshipStore()
    assert store.add(Relationship("a", "connects_to", "b")) is True
    assert store.add(Relationship("a", "connects_to", "b")) is False
    assert len(store) == 1


def test_registry_can_be_extended_with_a_new_relation_type() -> None:
    """A fourth relation type slots in without touching query code."""
    registry = default_relation_registry()
    registry.register(
        RelationTypeSpec(
            "develops_from",
            GraphKind.STRUCTURE,
            "The subject develops from the object during embryogenesis.",
            inverse=None,
        )
    )
    store = RelationshipStore(registry)
    store.add(Relationship("heart.fossa_ovalis", "develops_from", "heart.interatrial_septum"))
    assert store.related("heart.fossa_ovalis", "develops_from") == ("heart.interatrial_septum",)
    assert "develops_from" not in default_relation_registry()


def test_relationship_store_round_trips(ontology: AnatomyOntology) -> None:
    """Edges survive serialisation with their graph assignment intact."""
    restored = RelationshipStore.from_list(ontology.relationships.to_list())
    assert restored.counts() == ontology.relationships.counts()
    assert len(restored) == len(ontology.relationships)
