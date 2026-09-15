"""Step 8: the position shortcuts a model must not be able to use.

Predicted placement is only a meaningful test if placement cannot be recovered from
something other than the representation. These tests close the routes named in the Step 8
brief: fixed entity index, dataset ordering, family identifier, arrangement identifier,
hard-coded coordinates, hidden frame tensors, and labels that encode position.

They are deliberately adversarial. Each one tries to obtain the answer by a route the
architecture is not supposed to offer, and fails.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import ARRANGEMENT_FIELDS, sample_arrangement
from datasets.whole_organ.continuous_corpus import load_step8_split
from datasets.whole_organ.corpus import build_scene
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model

CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def builder() -> WholeOrganBatchBuilder:
    """A batch builder on the real heart ontology."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return WholeOrganBatchBuilder(ontology, domain)


@pytest.fixture(scope="module")
def scenes():
    """Four scenes drawn from different points in the arrangement space."""
    rng = np.random.default_rng(11)
    return [
        build_scene(scene_index=i, family_id=i % 3, lod=3, arrangement=sample_arrangement(rng))
        for i in range(4)
    ]


def test_the_batch_carries_no_arrangement_identifier(builder, scenes) -> None:
    """An arrangement label in the input would let the model classify rather than infer."""
    batch = builder.build(scenes).batch
    fields = {name: getattr(batch, name) for name in batch.__dataclass_fields__}
    for name, value in fields.items():
        if not isinstance(value, torch.Tensor):
            continue
        assert "variant" not in name and "arrangement" not in name, name
    structure = batch.structure
    for name in structure.__dataclass_fields__:
        assert "variant" not in name and "arrangement" not in name, name


def test_text_features_do_not_vary_with_the_arrangement(builder, scenes) -> None:
    """Text is presence, not placement. If it moved, the graph would not be the only route."""
    rng = np.random.default_rng(5)
    base = build_scene(scene_index=0, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    other = build_scene(scene_index=0, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    first = builder.build([base]).batch.text_features
    second = builder.build([other]).batch.text_features
    assert torch.equal(first, second), "text features encode the arrangement"


def test_the_family_identifier_is_not_an_input(builder, scenes) -> None:
    """Family fixes the organ's proportions; handing it over would leak shape."""
    same_arrangement = scenes[0].parameters.arrangement
    first = build_scene(scene_index=0, family_id=0, lod=3, arrangement=same_arrangement)
    second = build_scene(scene_index=0, family_id=5, lod=3, arrangement=same_arrangement)
    one = builder.build([first]).batch
    two = builder.build([second]).batch
    assert torch.equal(one.text_features, two.text_features)
    assert torch.equal(one.structure.entity_ids, two.structure.entity_ids)


def test_scene_order_does_not_change_a_scene_s_prediction(builder, scenes) -> None:
    """Dataset ordering must not be usable as a positional cue."""
    domain = load_domain_config()
    del domain
    model, _ = build_model(
        "A3Lite", builder, text_features=int(builder.build(scenes).batch.text_features.shape[1])
    )
    model.eval()
    forward = builder.build(scenes).batch
    backward = builder.build(list(reversed(scenes))).batch
    with torch.no_grad():
        first = model(forward, use_predicted_frames=True).frames
        second = model(backward, use_predicted_frames=True).frames
    assert first is not None and second is not None
    assert torch.allclose(first, torch.flip(second, dims=[0]), atol=1e-5)


def test_predicted_frames_do_not_come_from_the_slot_index(builder, scenes) -> None:
    """Two different arrangements must receive two different placements.

    If the head predicted from the entity's slot alone, every scene would get the same
    frames and the arrangement would be irrelevant.
    """
    rng = np.random.default_rng(17)
    first = build_scene(scene_index=0, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    second = build_scene(scene_index=1, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    batch = builder.build([first, second]).batch
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    with torch.no_grad():
        for parameter in model.frame_head.parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.05)
        frames = model(batch, use_predicted_frames=True).frames
    assert frames is not None
    assert not torch.allclose(frames[0], frames[1], atol=1e-6), (
        "both scenes received identical placement, so the frames ignore the input"
    )


def test_the_true_frame_tensor_never_reaches_a_predicted_forward(builder, scenes) -> None:
    """The decisive guard: corrupt the true frames and the inferred output must not move."""
    batch = builder.build(scenes).batch
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    poisoned = dataclass_replace(batch, entity_frames=torch.randn_like(batch.entity_frames))
    with torch.no_grad():
        clean = model(batch, use_predicted_frames=True).scene_logits
        dirty = model(poisoned, use_predicted_frames=True).scene_logits
    assert torch.equal(clean, dirty), "the true frame tensor is reaching the inferred path"


def test_corrupting_the_true_frames_does_change_the_supplied_path(builder, scenes) -> None:
    """The counterpart: if it changed nothing there either, the test above proves nothing."""
    batch = builder.build(scenes).batch
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    poisoned = dataclass_replace(batch, entity_frames=torch.randn_like(batch.entity_frames))
    with torch.no_grad():
        clean = model(batch, use_predicted_frames=False).scene_logits
        dirty = model(poisoned, use_predicted_frames=False).scene_logits
    assert not torch.equal(clean, dirty)


def test_part_labels_do_not_encode_position(builder, scenes) -> None:
    """The supervision target names an owner, not a place.

    A label that encoded position would let the model recover placement from the loss
    rather than from the representation.
    """
    batch = builder.build(scenes).batch
    owners = batch.part_owner
    assert owners.dtype == torch.int64
    values = set(owners.unique().tolist())
    assert values <= set(range(-1, int(batch.entity_present.shape[1]))), values


def test_entity_ordering_is_the_same_in_every_scene(builder) -> None:
    """Slot order is fixed by the ontology, so it carries no scene-specific information."""
    rng = np.random.default_rng(23)
    orders = set()
    for index in range(4):
        scene = build_scene(
            scene_index=index, family_id=index % 2, lod=3, arrangement=sample_arrangement(rng)
        )
        orders.add(tuple(builder.build([scene]).batch.structure.entity_ids[0].tolist()))
    assert len(orders) == 1, "entity slot order varies between scenes"


def test_the_corpus_stores_the_arrangement_but_the_batch_does_not(builder) -> None:
    """The arrangement is recorded for analysis and withheld from the model."""
    loaded = load_step8_split(CORPUS, "test_seen")
    assert loaded, "corpus not generated"
    level = loaded[0].active_lod
    scenes = [scene for scene in loaded if scene.active_lod == level][:2]
    for scene in scenes:
        assert len(scene.parameters.arrangement.coordinates()) == len(ARRANGEMENT_FIELDS)
    batch = builder.build(scenes).batch
    flat = torch.cat(
        [
            value.flatten().float()
            for value in (batch.text_features, batch.structure.entity_state.flatten())
        ]
    )
    for scene in scenes:
        for value in scene.parameters.arrangement.coordinates():
            if abs(value) < 1e-3:
                continue
            assert not bool((flat - value).abs().min() < 1e-6), (
                f"arrangement coordinate {value} appears verbatim in the model input"
            )
