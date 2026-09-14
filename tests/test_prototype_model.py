"""The prototype model: forward pass, identity protection, graphs and ablations."""

from __future__ import annotations

import copy

import pytest
import torch

from awr.ontology import AnatomyOntology
from generation.neural.graph_encoder import HeadAllocation, HeadRole
from generation.neural.nn.baseline import AppearanceBaseline, matched_baseline_config
from generation.neural.nn.model import LagnavPrototype, PrototypeConfig
from generation.neural.nn.tensors import AWRStructure, BatchBuilder, PrototypeBatch
from training.loop import build_model


@pytest.fixture
def structured(builder: BatchBuilder, batch: PrototypeBatch) -> LagnavPrototype:
    """The full structured arm."""
    torch.manual_seed(0)
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    assert isinstance(model, LagnavPrototype)
    model.eval()
    return model


def test_forward_shapes(structured: LagnavPrototype, batch: PrototypeBatch) -> None:
    """Every output matches the Step 5 contract shapes."""
    with torch.no_grad():
        output = structured(batch)
    entities = batch.structure.entity_count
    assert output.scene_latent.shape == (4, 256)
    assert output.entity_latent is not None
    assert output.entity_latent.shape == (4, entities, 256)
    assert output.geometry_tokens.shape == (4, entities, 32, 64)
    assert output.entity_field_logits is not None
    assert output.entity_field_logits.shape == (4, entities, 12)
    assert output.scene_logits.shape == (4, 96)
    assert output.part_logits.shape == (4, 96, entities + 1)
    assert output.frames is not None
    assert output.frames.shape == (4, entities, 12)


def test_identity_subspace_is_untouched_by_the_encoder(
    structured: LagnavPrototype, batch: PrototypeBatch
) -> None:
    """The write-protected slice is bit-identical after contextualisation."""
    with torch.no_grad():
        output = structured(batch)
    assert output.entity_latent_in is not None and output.entity_latent is not None
    assert torch.equal(output.entity_latent_in[..., :96], output.entity_latent[..., :96])
    assert not torch.equal(output.entity_latent_in[..., 96:], output.entity_latent[..., 96:])


def test_gradients_reach_every_component(
    structured: LagnavPrototype, batch: PrototypeBatch
) -> None:
    """A backward pass updates the encoder, the tokeniser, the decoder and the frames."""
    structured.train()
    output = structured(batch)
    assert output.entity_field_logits is not None and output.frames is not None
    loss = (
        output.scene_logits.square().mean()
        + output.entity_field_logits.square().mean()
        + output.frames.square().mean()
    )
    loss.backward()
    named = dict(structured.named_parameters())
    for name in (
        "composer.identity.weight",
        "graph_encoder.layers.0.query.weight",
        "tokens.entity_projection.weight",
        "field.head.weight",
        "frame_head.projection.weight",
    ):
        gradient = named[name].grad
        assert gradient is not None, name
        assert float(gradient.abs().sum()) > 0.0, name


def test_frames_are_teacher_forced_during_decoding(
    structured: LagnavPrototype, batch: PrototypeBatch
) -> None:
    """The occupancy path uses ground-truth frames, so it sends the frame head no gradient.

    This is the documented training protocol, not an accident: early frame error would
    otherwise corrupt the occupancy signal and neither component would learn. The frame
    head is supervised by its own objective instead.
    """
    structured.train()
    output = structured(batch)
    assert output.entity_field_logits is not None
    output.entity_field_logits.square().mean().backward()
    assert structured.frame_head.projection.weight.grad is None or float(
        structured.frame_head.projection.weight.grad.abs().sum()
    ) == 0.0


def test_padding_never_participates_in_attention(
    structured: LagnavPrototype, batch: PrototypeBatch
) -> None:
    """Padded slots neither attend nor are attended to."""
    assert structured.graph_encoder is not None
    mask = structured.graph_encoder.build_mask(batch.structure, 1)[0]
    assert not bool(mask[:, 63].any())
    assert not bool(mask[:, :, 63].any())


def test_graph_heads_are_restricted_to_their_graph(
    structured: LagnavPrototype, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """A functional relationship is not visible to a spatial head, and the reverse."""
    assert structured.graph_encoder is not None
    roles = HeadAllocation.default().roles()
    mask = structured.graph_encoder.build_mask(batch.structure, 1)[0]
    ids = list(ontology.ids())
    lv, aorta = ids.index("heart.left_ventricle"), ids.index("heart.aorta")
    septum = ids.index("heart.interventricular_septum")
    functional = roles.index(HeadRole.FUNCTIONAL)
    spatial = roles.index(HeadRole.SPATIAL)
    assert bool(mask[functional, lv, aorta])
    assert not bool(mask[functional, lv, septum])
    assert bool(mask[spatial, lv, septum])
    assert not bool(mask[spatial, lv, aorta])


def test_relation_type_changes_the_bias(
    structured: LagnavPrototype, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """Typed bias makes pumps_to and adjacent_to different messages."""
    assert structured.graph_encoder is not None
    with torch.no_grad():
        structured.graph_encoder.relation_bias.weight.normal_()
    bias = structured.graph_encoder.build_bias(batch.structure, 1)[0].detach()
    roles = HeadAllocation.default().roles()
    ids = list(ontology.ids())
    lv, aorta = ids.index("heart.left_ventricle"), ids.index("heart.aorta")
    septum = ids.index("heart.interventricular_septum")
    functional = roles.index(HeadRole.FUNCTIONAL)
    spatial = roles.index(HeadRole.SPATIAL)
    assert float(bias[functional, lv, aorta]) != 0.0
    assert float(bias[functional, lv, septum]) == 0.0
    assert float(bias[spatial, lv, septum]) != 0.0


def test_inverse_direction_is_preserved(
    structured: LagnavPrototype, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """A directed relation biases the two directions differently."""
    assert structured.graph_encoder is not None
    with torch.no_grad():
        structured.graph_encoder.relation_bias.weight.normal_()
    bias = structured.graph_encoder.build_bias(batch.structure, 1)[0].detach()
    ids = list(ontology.ids())
    atrium = ids.index("heart.right_atrium")
    ventricle = ids.index("heart.right_ventricle")
    roles = HeadAllocation.default().roles()
    spatial = roles.index(HeadRole.SPATIAL)
    forward = float(bias[spatial, atrium, ventricle])
    reverse = float(bias[spatial, ventricle, atrium])
    assert forward != 0.0 and reverse != 0.0
    assert forward != reverse


def test_changing_a_relation_changes_the_latent(
    structured: LagnavPrototype, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """Adding one typed edge moves the two entities' context and nothing else's identity."""
    with torch.no_grad():
        structured.graph_encoder.relation_bias.weight.normal_()  # type: ignore[union-attr]
        before = structured(batch)
        structure = batch.structure
        free = (~structure.edge_mask).nonzero().flatten()
        position = int(free[0])
        ids = list(ontology.ids())
        subject, target = ids.index("heart.left_ventricle"), ids.index("heart.superior_vena_cava")
        edge_mask = structure.edge_mask.clone()
        edge_mask[position] = True
        edge_source = structure.edge_source.clone()
        edge_source[position] = subject
        edge_target = structure.edge_target.clone()
        edge_target[position] = target
        edge_relation = structure.edge_relation.clone()
        edge_relation[position] = 0
        edge_graph = structure.edge_graph.clone()
        edge_graph[position] = int(structure.edge_graph[structure.edge_mask][0])
        adjacency = structure.graph_adjacency.clone()
        adjacency[int(edge_graph[position]), subject, target] = True
        altered = copy.copy(batch)
        altered.structure = AWRStructure(
            lod=structure.lod,
            entity_ids=structure.entity_ids,
            entity_type=structure.entity_type,
            semantic_role=structure.semantic_role,
            laterality=structure.laterality,
            parent_index=structure.parent_index,
            depth=structure.depth,
            lod_min=structure.lod_min,
            lod_max=structure.lod_max,
            entity_state=structure.entity_state,
            entity_mask=structure.entity_mask,
            edge_source=edge_source,
            edge_target=edge_target,
            edge_relation=edge_relation,
            edge_graph=edge_graph,
            edge_mask=edge_mask,
            graph_adjacency=adjacency,
            visible_slots=structure.visible_slots,
        )
        after = structured(altered)
    assert before.entity_latent is not None and after.entity_latent is not None
    delta = (after.entity_latent - before.entity_latent).abs()
    assert float(delta[0, subject, 96:].max()) > 0.0
    assert float(delta[..., :96].max()) == 0.0


def test_ablation_switches_change_the_model(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """Each ablation produces the architecture it claims to."""
    features = int(batch.text_features.shape[1])
    a1, _ = build_model("A1", builder, text_features=features)
    assert isinstance(a1, LagnavPrototype)
    assert a1.graph_encoder is None
    a4, _ = build_model("A4", builder, text_features=features)
    assert isinstance(a4, LagnavPrototype)
    assert a4.shared_part_head is not None
    with torch.no_grad():
        output = a4(batch)
    assert output.entity_field_logits is None
    a5, _ = build_model("A5", builder, text_features=features)
    assert isinstance(a5, LagnavPrototype)
    assert a5.config.geometry.per_lod_blocks is True
    assert a5.tokens.queries.shape[0] == 5


def test_unknown_ablation_is_refused() -> None:
    """Only the declared ablation keys exist."""
    with pytest.raises(ValueError, match="Unknown ablation"):
        PrototypeConfig().with_ablation("A9")


def test_baseline_is_parameter_matched(builder: BatchBuilder, batch: PrototypeBatch) -> None:
    """The arms are within five percent, checked before training."""
    features = int(batch.text_features.shape[1])
    structured, structured_groups = build_model("A3", builder, text_features=features)
    baseline, baseline_groups = build_model("A0", builder, text_features=features)
    assert isinstance(baseline, AppearanceBaseline)
    gap = abs(baseline_groups["total"] - structured_groups["total"]) / structured_groups["total"]
    assert gap <= 0.05, f"{baseline_groups['total']} vs {structured_groups['total']}"
    assert baseline_groups["matched_target"] == structured_groups["total"]


def test_unmatchable_target_is_refused() -> None:
    """A budget that cannot be matched fails rather than running unmatched."""
    with pytest.raises(ValueError, match="Could not match"):
        matched_baseline_config(10)


def test_baseline_has_no_entity_axis(builder: BatchBuilder, batch: PrototypeBatch) -> None:
    """The baseline is genuinely appearance-driven: one shared latent, no frames."""
    baseline, _ = build_model("A0", builder, text_features=int(batch.text_features.shape[1]))
    with torch.no_grad():
        output = baseline(batch)
    assert output.entity_latent is None
    assert output.frames is None
    assert output.entity_field_logits is None
    assert output.geometry_tokens.shape[1] == 1
