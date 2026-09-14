"""Per-entity geometry: frames, independent querying, composition and level of detail."""

from __future__ import annotations

import numpy as np
import torch

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec
from datasets.synthetic.primitives import Ellipsoid, rotation_from_euler
from generation.neural.nn.geometry import (
    GeometryConfig,
    GeometryTokenGenerator,
    OccupancyFieldDecoder,
    compose_scene,
    frame_rotation,
    points_to_local,
)
from generation.neural.nn.model import LagnavPrototype
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE
from training.loop import build_model


def test_frame_transform_matches_the_corpus_convention() -> None:
    """The model's canonical frame is the same transform the generator used."""
    rotation = rotation_from_euler(0.3, -0.2, 0.1)
    primitive = Ellipsoid(
        np.array([0.1, -0.2, 0.05]), np.array([0.3, 0.2, 0.25]), rotation
    )
    frame = torch.tensor(primitive.frame(), dtype=torch.float64).view(1, 1, 12)
    points = np.random.default_rng(0).uniform(-1.0, 1.0, size=(32, 3))
    expected = (points - primitive.centre_point) @ primitive.rotation_matrix / primitive.radii
    actual = points_to_local(torch.tensor(points).view(1, 1, 32, 3), frame)[0, 0].numpy()
    assert np.allclose(expected, actual, atol=1e-6)
    assert np.allclose(frame_rotation(frame)[0, 0].numpy(), rotation, atol=1e-6)


def test_entity_geometry_is_independent(builder: BatchBuilder, batch: PrototypeBatch) -> None:
    """Entity N's tokens cannot be moved by entity M's latent."""
    torch.manual_seed(0)
    config = GeometryConfig()
    generator = GeometryTokenGenerator(config)
    entity_latent = torch.randn(2, 64, 256)
    scene_latent = torch.randn(2, 256)
    with torch.no_grad():
        tokens = generator(entity_latent, scene_latent, lod=2)
        perturbed = entity_latent.clone()
        perturbed[:, 5] += 1.0
        after = generator(perturbed, scene_latent, lod=2)
    assert torch.equal(tokens[:, 6], after[:, 6])
    assert not torch.equal(tokens[:, 5], after[:, 5])


def test_geometry_can_be_queried_per_entity(
    builder: BatchBuilder, batch: PrototypeBatch, ontology: AnatomyOntology
) -> None:
    """``geometry(entity, points)`` is a real operation, not an aspiration."""
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    assert isinstance(model, LagnavPrototype)
    model.eval()
    with torch.no_grad():
        output = model(batch)
        slot = list(ontology.ids()).index("heart.left_ventricle")
        tokens = output.geometry_tokens[:, slot : slot + 1]
        frames = batch.entity_frames[:, slot : slot + 1]
        points = batch.scene_points.unsqueeze(1)
        local = points_to_local(points, frames)
        logits = model.field(local, tokens, active_tokens=output.active_tokens)
    assert logits.shape == (batch.batch_size, 1, batch.scene_points.shape[1])
    assert torch.allclose(logits[:, 0], output.scene_entity_logits[:, slot], atol=1e-5)  # type: ignore[index]


def test_composition_attributes_every_point(builder: BatchBuilder, batch: PrototypeBatch) -> None:
    """Part logits cover every entity plus background, and absent entities are masked."""
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        output = model(batch)
    entities = batch.structure.entity_count
    assert output.part_logits.shape[-1] == entities + 1
    absent = ~batch.entity_present
    absent_logits = output.part_logits[..., :entities][absent.unsqueeze(1).expand(-1, 96, -1)]
    assert float(absent_logits.max()) < -1e8


def test_compose_scene_is_a_soft_union() -> None:
    """The scene field is at least as occupied as its most occupied entity."""
    logits = torch.tensor([[[2.0, -3.0], [-1.0, 4.0]]])
    present = torch.ones(1, 2, dtype=torch.bool)
    scene, parts = compose_scene(logits, present)
    assert scene.shape == (1, 2)
    assert float(scene[0, 0]) >= 2.0 - 1e-3
    assert float(scene[0, 1]) >= 4.0 - 1e-3
    assert parts.shape == (1, 2, 3)


def test_level_of_detail_reads_a_prefix_of_one_block(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """More detail means more tokens from the same block, not a different block."""
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    assert isinstance(model, LagnavPrototype)
    model.eval()
    with torch.no_grad():
        coarse = model(batch, token_prefix=4)
        fine = model(batch, token_prefix=32)
    assert coarse.active_tokens == 4
    assert fine.active_tokens == 32
    assert torch.equal(coarse.geometry_tokens, fine.geometry_tokens)
    assert torch.equal(coarse.geometry_tokens[:, :, :4], fine.geometry_tokens[:, :, :4])


def test_level_of_detail_never_changes_entity_identity(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """Entity latents and their identity slices are identical at every prefix."""
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        outputs = [model(batch, token_prefix=prefix) for prefix in LOD_TOKEN_SCHEDULE]
    reference = outputs[0].entity_latent
    assert reference is not None
    for output in outputs[1:]:
        assert output.entity_latent is not None
        assert torch.equal(output.entity_latent, reference)


def test_token_schedule_matches_the_step_five_design() -> None:
    """The prefix schedule is the one the design document declares."""
    assert LOD_TOKEN_SCHEDULE == (4, 8, 16, 24, 32)
    config = GeometryConfig()
    assert [config.tokens_at(level) for level in range(5)] == [4, 8, 16, 24, 32]


def test_field_features_are_conditioned(builder: BatchBuilder) -> None:
    """Every head reads token-conditioned features, not the bare point encoding."""
    torch.manual_seed(0)
    config = GeometryConfig()
    decoder = OccupancyFieldDecoder(config)
    points = torch.randn(1, 1, 8, 3) * 0.3
    first = torch.randn(1, 1, 32, 64)
    second = torch.randn(1, 1, 32, 64)
    with torch.no_grad():
        a = decoder.features(points, first, active_tokens=32)
        b = decoder.features(points, second, active_tokens=32)
    assert not torch.allclose(a, b)


def test_geometry_reconstructs_the_corpus_convention(
    scene_specs: list[SceneSpec], ontology: AnatomyOntology
) -> None:
    """Sanity: the ground-truth occupancy a model is trained against is the primitive's."""
    spec = scene_specs[0]
    primitive = spec.primitives["heart.left_ventricle"]
    points = primitive.sample_surface(64, np.random.default_rng(0))
    pulled_inward = primitive.centre + (points - primitive.centre) * 0.5
    assert bool(primitive.occupancy(pulled_inward).all())
    pushed_outward = primitive.centre + (points - primitive.centre) * 1.5
    assert not bool(primitive.occupancy(pushed_outward).any())
