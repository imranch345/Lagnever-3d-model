"""The AWR to tensor bridge: identity axis, vocabularies, adjacency and padding."""

from __future__ import annotations

import pytest

from awr.errors import ContractError
from awr.ontology import AnatomyOntology
from awr.relationships import GraphKind
from awr.scene import AWRScene
from editing.scene_editor import SceneEditor
from generation.neural.features import (
    AWR_BATCH_CONTRACT,
    ENTITY_STATE_LAYOUT,
    AWRFeatureExtractor,
    EntityCodebook,
    Vocabulary,
)


@pytest.fixture
def extractor(ontology: AnatomyOntology) -> AWRFeatureExtractor:
    """Feature extractor over the heart ontology."""
    return AWRFeatureExtractor(ontology)


def test_batch_satisfies_the_contract(extractor: AWRFeatureExtractor, scene: AWRScene) -> None:
    """A real heart scene produces a contract-valid batch."""
    payload = extractor.encode_batch([scene, scene])
    bindings = AWR_BATCH_CONTRACT.validate_payload(payload, bindings={"B": 2})
    assert bindings["B"] == 2
    assert bindings["N_ENT"] == 64
    assert set(payload) == set(AWR_BATCH_CONTRACT.names())


def test_entity_axis_is_the_identity_axis(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """Slot order follows the codebook, so slot 4 is always the same structure."""
    payload = extractor.encode_batch([scene])
    slot = list(scene.ids()).index("heart.left_ventricle")
    assert int(payload["entity_ids"].at(0, slot)) == extractor.codebook.index(
        "heart.left_ventricle"
    )
    assert extractor.codebook.entity(int(payload["entity_ids"].at(0, slot))) == (
        "heart.left_ventricle"
    )


def test_codebook_is_stable_across_construction(ontology: AnatomyOntology) -> None:
    """Two codebooks built from one ontology are identical."""
    first = EntityCodebook.from_ontology(ontology)
    second = EntityCodebook.from_ontology(ontology)
    assert first == second
    assert len(first) == 42


def test_codebook_mismatch_is_detected(extractor: AWRFeatureExtractor, scene: AWRScene) -> None:
    """A scene from a different ontology version cannot be encoded silently."""
    scene.ontology_version = "9.9.9"
    with pytest.raises(ContractError, match="codebook"):
        extractor.encode_scene(scene)


def test_padding_is_marked_and_never_looks_like_entity_zero(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """Padded slots are masked out and hold -1, not a valid index."""
    payload = extractor.encode_batch([scene])
    real = len(scene.ids())
    for slot in range(real, 64):
        assert payload["entity_mask"].at(0, slot) is False
        assert int(payload["entity_ids"].at(0, slot)) == -1
        assert int(payload["parent_index"].at(0, slot)) == -1


def test_state_channels_follow_the_declared_layout(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """Visibility and opacity land in their declared channels."""
    editor = SceneEditor(scene)
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    editor.hide("heart.aorta")
    payload = extractor.encode_batch([scene])
    ids = list(scene.ids())
    lv = ids.index("heart.left_ventricle")
    aorta = ids.index("heart.aorta")
    opacity = ENTITY_STATE_LAYOUT.index("opacity")
    visibility = ENTITY_STATE_LAYOUT.index("visibility")
    assert payload["entity_state"].at(0, lv, opacity) == pytest.approx(0.3)
    assert payload["entity_state"].at(0, lv, visibility) == pytest.approx(1.0)
    assert payload["entity_state"].at(0, aorta, visibility) == pytest.approx(0.0)


def test_state_channel_count_matches_the_contract() -> None:
    """The layout and the declared dimension are one decision."""
    assert ENTITY_STATE_LAYOUT.width == AWR_BATCH_CONTRACT.spec("entity_state").resolve(
        {"B": 1}
    )[-1]


def test_hierarchy_is_encoded_as_slot_indices(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """Parent pointers reference slots, and the root has none."""
    payload = extractor.encode_batch([scene])
    ids = list(scene.ids())
    lv = ids.index("heart.left_ventricle")
    chambers = ids.index("heart.chambers")
    root = ids.index("heart")
    assert int(payload["parent_index"].at(0, lv)) == chambers
    assert int(payload["parent_index"].at(0, root)) == -1
    assert int(payload["depth"].at(0, lv)) == 2


def test_adjacency_preserves_graph_identity(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """A functional edge does not appear in the spatial graph, and the reverse."""
    payload = extractor.encode_batch([scene])
    ids = list(scene.ids())
    lv, aorta = ids.index("heart.left_ventricle"), ids.index("heart.aorta")
    septum = ids.index("heart.interventricular_septum")

    functional = extractor.adjacency(payload, graph=GraphKind.FUNCTIONAL)
    spatial = extractor.adjacency(payload, graph=GraphKind.SPATIAL)

    assert extractor.relations.symbol(functional[lv][aorta]) == "pumps_to"
    assert functional[lv][septum] == -1
    assert extractor.relations.symbol(spatial[lv][septum]) == "adjacent_to"
    assert spatial[lv][aorta] == -1


def test_all_declared_edges_are_encoded(
    extractor: AWRFeatureExtractor, scene: AWRScene
) -> None:
    """Nothing is dropped between the AWR and the tensors."""
    payload = extractor.encode_batch([scene])
    live = sum(
        1 for position in range(256) if payload["edge_mask"].at(0, position)
    )
    assert live == len(scene.relationships.all_edges())


def test_vocabulary_sizes_are_reported_for_model_configuration(
    extractor: AWRFeatureExtractor,
) -> None:
    """Embedding tables can be sized without guessing."""
    sizes = extractor.vocabulary_sizes()
    assert sizes["entity"] == 42
    assert sizes["relation"] == 18
    assert sizes["graph"] == 3


def test_vocabulary_rejects_unknown_symbols() -> None:
    """Closed vocabularies fail loudly."""
    vocabulary = Vocabulary("demo", ("a", "b"))
    assert vocabulary.index("b") == 1
    with pytest.raises(ContractError, match="unknown symbol"):
        vocabulary.index("c")
    with pytest.raises(ContractError, match="duplicate symbols"):
        Vocabulary("bad", ("a", "a"))


def test_empty_batch_is_rejected(extractor: AWRFeatureExtractor) -> None:
    """An empty batch is a programming error, not an edge case to absorb."""
    with pytest.raises(ContractError, match="at least one scene"):
        extractor.encode_batch([])


def test_encoding_is_deterministic(extractor: AWRFeatureExtractor, scene: AWRScene) -> None:
    """The same scene encodes identically every time."""
    first = extractor.encode_batch([scene])
    second = extractor.encode_batch([scene])
    assert {k: v.data for k, v in first.items()} == {k: v.data for k, v in second.items()}


def test_too_many_entities_is_refused(ontology: AnatomyOntology, scene: AWRScene) -> None:
    """A scene larger than the declared slot count fails with a clear message."""
    narrow = AWRFeatureExtractor(ontology, max_entities=8)
    with pytest.raises(ContractError, match="entity slots"):
        narrow.encode_scene(scene)
