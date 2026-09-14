"""Persistent editing: identity, locality and the Step 4 session running end to end."""

from __future__ import annotations

import copy

import torch

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from generation.neural.editing import plan_edit
from generation.neural.language_encoder import ProgramTranscoder
from generation.neural.nn.metrics import edit_drift
from generation.neural.nn.model import LagnavPrototype
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch
from reasoning.command_engine import CommandEngine
from training.loop import build_model

SESSION = (
    "generate a human heart",
    "show the four chambers",
    "make the left ventricle transparent",
    "show the valves",
    "hide everything except the chambers",
)


def _model(builder: BatchBuilder, batch: PrototypeBatch, arm: str) -> torch.nn.Module:
    torch.manual_seed(0)
    model, _ = build_model(arm, builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    return model


def test_presentation_edit_leaves_geometry_bit_identical(
    builder: BatchBuilder, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """The central editing claim, measured on the real model rather than argued."""
    model = _model(builder, batch, "A3")
    assert isinstance(model, LagnavPrototype)
    assert model.geometry_is_state_invariant
    hidden = int(batch.structure.visible_slots[-1])
    edited = copy.copy(batch)
    edited.text_features = batch.text_features.clone()
    edited.text_features[:, hidden] = 0.0
    with torch.no_grad():
        before = model(batch)
        after = model(edited)
    assert torch.equal(before.geometry_tokens, after.geometry_tokens)
    assert torch.equal(before.scene_entity_logits, after.scene_entity_logits)  # type: ignore[arg-type]
    assert before.entity_latent is not None and after.entity_latent is not None
    assert torch.equal(before.entity_latent, after.entity_latent)


def test_drift_is_exactly_zero_for_the_structured_arm(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """Untouched entities do not move at all, which is stronger than 'barely moves'."""
    model = _model(builder, batch, "A3")
    hidden = int(batch.structure.visible_slots[-1])
    result = edit_drift(model, batch, hidden_slot=hidden)
    assert result.values["untouched_drift"] == 0.0
    assert result.values["geometry_tokens_identical"] == 1.0


def test_the_baseline_does_drift(builder: BatchBuilder, batch: PrototypeBatch) -> None:
    """The comparison is meaningful only because the baseline can drift, and does."""
    model = _model(builder, batch, "A0")
    hidden = int(batch.structure.visible_slots[-1])
    result = edit_drift(model, batch, hidden_slot=hidden)
    assert result.values["geometry_tokens_identical"] == 0.0
    assert result.values["untouched_drift"] > 0.0


def test_level_of_detail_edit_keeps_identity(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """Changing detail refines the field and leaves entity latents untouched."""
    model = _model(builder, batch, "A3")
    with torch.no_grad():
        coarse = model(batch, token_prefix=8)
        fine = model(batch, token_prefix=32)
    assert coarse.entity_latent is not None and fine.entity_latent is not None
    assert torch.equal(coarse.entity_latent, fine.entity_latent)
    assert not torch.equal(coarse.scene_logits, fine.scene_logits)


def test_full_session_preserves_scene_and_entity_identity(
    config: DomainConfig, ontology: AnatomyOntology
) -> None:
    """The Step 4 session still runs, and nothing about identity moves."""
    engine = CommandEngine(config=config, ontology=ontology)
    responses = engine.run(SESSION)
    assert all(response.ok for response in responses), [r.message for r in responses if not r.ok]
    scene = engine.scene
    assert scene is not None
    assert scene.get("heart.left_ventricle").entity_id == "heart.left_ventricle"
    assert scene.get("heart.left_ventricle").opacity == 0.3
    assert len(scene) == 42
    assert scene.ids() == ontology.ids()


def test_session_edits_are_planned_as_state_only(
    config: DomainConfig, ontology: AnatomyOntology
) -> None:
    """Every edit in the session recomputes no geometry, and the plan says which."""
    engine = CommandEngine(config=config, ontology=ontology)
    engine.execute("generate a human heart")
    scene = engine.scene
    assert scene is not None
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    for utterance in SESSION[1:]:
        plan = plan_edit(transcoder.transcode(utterance), scene)
        assert plan.recompute_entities == (), utterance
        assert len(plan.frozen_entities) == len(scene.renderable_ids())
        assert plan.touched_entities
        engine.execute(utterance)


def test_scene_and_entity_ids_survive_a_model_backed_session(
    builder: BatchBuilder, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """Running the model at each step of a session never changes an entity id."""
    model = _model(builder, batch, "A3")
    identifiers = batch.structure.entity_ids.clone()
    for hidden in [int(slot) for slot in batch.structure.visible_slots[:3]]:
        edited = copy.copy(batch)
        edited.text_features = batch.text_features.clone()
        edited.text_features[:, hidden] = 0.0
        with torch.no_grad():
            model(edited)
        assert torch.equal(edited.structure.entity_ids, identifiers)
