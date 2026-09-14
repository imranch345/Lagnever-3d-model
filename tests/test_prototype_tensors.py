"""AWR to tensor conversion for the prototype: shapes, masks and stable indexing."""

from __future__ import annotations

import pytest
import torch

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch


def test_batch_shapes_and_dtypes(batch: PrototypeBatch) -> None:
    """Every tensor has the declared shape and dtype."""
    structure = batch.structure
    entities = structure.entity_count
    assert batch.scene_points.shape == (4, 96, 3)
    assert batch.scene_points.dtype is torch.float32
    assert batch.scene_occupancy.shape == (4, 96)
    assert batch.part_owner.shape == (4, 96)
    assert batch.part_owner.dtype is torch.int64
    assert batch.entity_points.shape == (4, entities, 12, 3)
    assert batch.entity_occupancy.shape == (4, entities, 12)
    assert batch.entity_frames.shape == (4, entities, 12)
    assert batch.entity_present.dtype is torch.bool
    assert structure.graph_adjacency.shape == (3, entities, entities)


def test_entity_indexing_is_deterministic_and_matches_the_ontology(
    builder: BatchBuilder, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """Slot order is ontology order, so slot i is the same structure every time."""
    ids = list(ontology.ids())
    for name in ("heart.left_ventricle", "heart.aorta", "heart.mitral_valve"):
        slot = ids.index(name)
        assert builder.slot_of[name] == slot
        assert int(batch.structure.entity_ids[slot]) == builder.extractor.codebook.index(name)


def test_padding_is_masked(batch: PrototypeBatch) -> None:
    """Padded slots are marked and never look like entity zero."""
    mask = batch.structure.entity_mask
    assert bool(mask[:42].all())
    assert not bool(mask[42:].any())
    assert int(batch.structure.entity_ids[63]) == -1


def test_presence_matches_the_level_of_detail(batch: PrototypeBatch) -> None:
    """Every scene in a batch shares the level's visible entity set."""
    visible = batch.structure.visible_slots
    for index in range(batch.batch_size):
        present = batch.entity_present[index].nonzero().flatten()
        assert torch.equal(present, visible)


def test_mixed_levels_in_one_batch_are_refused(
    builder: BatchBuilder, mixed_lod_specs: list[SceneSpec]
) -> None:
    """Bucketing by level is a requirement, not a convention."""
    first = mixed_lod_specs[0]
    other = next(s for s in mixed_lod_specs if s.active_lod != first.active_lod)
    specs = [first, other]
    with pytest.raises(ValueError, match="one level of detail"):
        builder.build(specs)


def test_structure_is_cached_per_level(builder: BatchBuilder, scene_specs: list[SceneSpec]) -> None:
    """The AWR structure depends on the level, not on the geometry."""
    first = builder.build(list(scene_specs[:2])).structure
    second = builder.build(list(scene_specs[2:4])).structure
    assert first is second


def test_inverse_relation_table_is_direction_aware(builder: BatchBuilder) -> None:
    """Symmetric relations map to themselves; inverse pairs map to each other."""
    table = builder.inverse_relation_table()
    relations = builder.extractor.relations
    assert int(table[relations.index("adjacent_to")]) == relations.index("adjacent_to")
    assert int(table[relations.index("superior_to")]) == relations.index("inferior_to")
    assert int(table[relations.index("inferior_to")]) == relations.index("superior_to")
    assert int(table[relations.index("pumps_to")]) == -1


def test_graph_adjacency_separates_the_graphs(
    batch: PrototypeBatch, builder: BatchBuilder, ontology: AnatomyOntology
) -> None:
    """A functional edge does not appear in the spatial adjacency."""
    ids = list(ontology.ids())
    lv, aorta = ids.index("heart.left_ventricle"), ids.index("heart.aorta")
    septum = ids.index("heart.interventricular_septum")
    structure, spatial, functional = 0, 1, 2
    adjacency = batch.structure.graph_adjacency
    assert bool(adjacency[functional, lv, aorta])
    assert not bool(adjacency[functional, lv, septum])
    assert bool(adjacency[spatial, lv, septum])
    assert not bool(adjacency[spatial, lv, aorta])
    assert bool(adjacency[structure].any())


def test_text_features_describe_the_request(batch: PrototypeBatch) -> None:
    """The oracle text feature marks exactly the requested entities."""
    entities = batch.structure.entity_count
    for index in range(batch.batch_size):
        marked = batch.text_features[index, :entities].nonzero().flatten()
        assert torch.equal(marked, batch.structure.visible_slots)


def test_batch_moves_to_a_device(batch: PrototypeBatch) -> None:
    """Moving a batch keeps every tensor together."""
    moved = batch.to(torch.device("cpu"))
    assert moved.scene_points.device.type == "cpu"
    assert moved.structure.entity_ids.device.type == "cpu"
