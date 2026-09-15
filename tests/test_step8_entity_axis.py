"""Step 8 entity factorisation, checked under predicted placement.

Step 7 established that the entity axis carries most of the architecture's value, but
measured it with ground-truth placement supplied. These tests check the properties that
make the axis useful still hold when the model has to place structures itself, because
that is the condition Step 8 judges everything in.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model


@pytest.fixture(scope="module")
def setup():
    """A batch of continuous-arrangement scenes and the builder that made it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    rng = np.random.default_rng(29)
    scenes = [
        build_scene(scene_index=i, family_id=i % 3, lod=3, arrangement=sample_arrangement(rng))
        for i in range(3)
    ]
    return builder.build(scenes).batch, builder


def test_each_entity_gets_its_own_geometry_token_block(setup) -> None:
    """Per-entity blocks are what make the scene addressable one structure at a time."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        staged = model.stage(batch)
    assert staged.tokens is not None
    assert staged.tokens.shape[0] == batch.batch_size
    assert staged.tokens.shape[1] == batch.structure.entity_count
    assert staged.tokens.dim() == 4


def test_each_entity_gets_its_own_predicted_frame(setup) -> None:
    """Placement is per entity, not one transform for the whole organ."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    with torch.no_grad():
        for parameter in model.frame_head.parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.05)
        frames = model(batch, use_predicted_frames=True).frames
    assert frames is not None
    assert frames.shape == (batch.batch_size, batch.structure.entity_count, 12)
    visible = batch.structure.visible_slots
    placed = frames[0, visible, 0:3]
    assert float(placed.std(dim=0).mean()) > 1e-4, "every entity received the same position"


def test_part_labels_name_the_owning_entity(setup) -> None:
    """Geometry correspondence is a tensor axis, not a learned association step."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        output = model(batch, use_predicted_frames=True)
    entities = batch.structure.entity_count
    assert output.part_logits.shape == (batch.batch_size, batch.scene_points.shape[1], entities + 1)


def test_absent_entities_cannot_own_a_point(setup) -> None:
    """An entity not in the scene must never win a point, at any placement condition."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        output = model(batch, use_predicted_frames=True)
    predicted = output.part_logits.argmax(dim=-1)
    entities = batch.structure.entity_count
    for index in range(batch.batch_size):
        owned = set(predicted[index].unique().tolist()) - {entities}
        present = set(batch.entity_present[index].nonzero().flatten().tolist())
        assert owned <= present, f"scene {index} assigned points to absent entities"


def test_changing_one_entity_s_latent_leaves_other_blocks_alone(setup) -> None:
    """The factorisation claim, stated as an intervention rather than an assertion."""
    batch, builder = setup
    model, _ = build_model("A1", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        staged = model.stage(batch)
        assert staged.tokens is not None
        latent = staged.entity_latent.clone()
        slot = int(batch.structure.visible_slots[2])
        latent[:, slot] = latent[:, slot] + 1.0
        moved = model.tokens(latent, staged.scene_latent, lod=batch.structure.lod)
    changed = [
        index
        for index in range(batch.structure.entity_count)
        if not torch.allclose(moved[:, index], staged.tokens[:, index], atol=1e-6)
    ]
    assert changed == [slot], f"changing one entity moved blocks {changed}"


def test_the_monolithic_ablation_really_is_monolithic(setup) -> None:
    """A4 must not quietly keep per-entity geometry, or the contrast measures nothing."""
    batch, builder = setup
    model, groups = build_model("A4", builder, text_features=int(batch.text_features.shape[1]))
    assert model.config.per_entity_geometry is False
    assert groups["shared_part_head"] > 0
    model.eval()
    with torch.no_grad():
        staged = model.stage(batch)
    assert staged.tokens is None, "A4 still produces per-entity token blocks"


def test_the_identity_subspace_is_write_protected_in_every_arm(setup) -> None:
    """Persistent identity is what makes an edited entity the same entity."""
    batch, builder = setup
    for arm in ("A1", "A1M", "A3Lite", "A3L", "A3"):
        model, _ = build_model(arm, builder, text_features=int(batch.text_features.shape[1]))
        model.eval()
        with torch.no_grad():
            staged = model.stage(batch)
        width = model.config.identity_width
        assert torch.equal(
            staged.entity_latent[..., :width], staged.entity_latent_in[..., :width]
        ), arm
