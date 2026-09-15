"""Step 8 untyped graph attention: what it must read, and what it must not.

Step 7 measured that the partitioned typed encoder responded to which entities are
connected and almost not at all to what the relation says. This encoder keeps the first
and drops the second, and these tests check both halves of that claim rather than
trusting the design note.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from experiments.step7.perturbations import GRAPH_INDEX, perturb_relations
from generation.neural.nn.graph_lite import GraphLiteConfig, UntypedGraphEncoder
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model


@pytest.fixture(scope="module")
def setup():
    """A real batch built from continuous arrangements, plus the builder."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    rng = np.random.default_rng(0)
    scenes = [
        build_scene(scene_index=i, family_id=i % 3, lod=3, arrangement=sample_arrangement(rng))
        for i in range(4)
    ]
    return builder.build(scenes).batch, builder


def _encoder() -> UntypedGraphEncoder:
    return UntypedGraphEncoder(GraphLiteConfig())


def test_the_identity_subspace_is_never_written(setup) -> None:
    """Persistent entity identity is what makes an entity the same entity."""
    batch, _ = setup
    encoder = _encoder()
    latent = torch.randn(batch.batch_size, batch.structure.entity_count, 256)
    out = encoder(latent, batch.structure)
    width = encoder.config.identity_width
    assert torch.equal(out[..., :width], latent[..., :width])


def test_the_mask_is_the_union_of_the_three_adjacencies(setup) -> None:
    """Dropping the typed partitioning must not drop the connectivity."""
    batch, _ = setup
    encoder = _encoder()
    mask = encoder.build_mask(batch.structure, batch.batch_size)
    adjacency = batch.structure.graph_adjacency
    for name, index in GRAPH_INDEX.items():
        edges = adjacency[:, index].bool()
        reachable = edges & mask
        assert int(reachable.sum()) == int(edges.sum()), f"{name} edges are not all in the mask"


def test_every_entity_can_attend_to_itself(setup) -> None:
    """An isolated entity with nothing to attend to would produce NaN."""
    batch, _ = setup
    encoder = _encoder()
    mask = encoder.build_mask(batch.structure, batch.batch_size)
    valid = batch.structure.entity_mask
    entities = mask.shape[-1]
    diagonal = torch.eye(entities, dtype=torch.bool).unsqueeze(0)
    assert bool((mask & diagonal)[valid.unsqueeze(-1).expand_as(mask) & diagonal].all())


def test_padded_slots_are_excluded_from_both_sides(setup) -> None:
    """A padded entity must neither attend nor be attended to."""
    batch, _ = setup
    encoder = _encoder()
    mask = encoder.build_mask(batch.structure, batch.batch_size)
    valid = batch.structure.entity_mask
    padded = ~valid
    if not bool(padded.any()):
        pytest.skip("this batch has no padded slots")
    assert not bool(mask[padded.unsqueeze(-1).expand_as(mask)].any())
    assert not bool(mask[padded.unsqueeze(-2).expand_as(mask)].any())


def test_relation_types_are_unreadable(setup) -> None:
    """The typed machinery is gone, so corrupting relation labels must do nothing.

    Not an approximation: the output must be bit-identical, because the encoder never
    reads ``edge_relation`` at all.
    """
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    inverse = builder.inverse_relation_table()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        base = model(batch).part_logits
        for kind in ("invert_spatial_labels", "shuffle_spatial_types"):
            damaged = perturb_relations(
                batch, kind, inverse_relations=inverse, generator=generator
            )
            with torch.no_grad():
                other = model(damaged).part_logits
            assert torch.equal(base, other), f"{kind} changed the output"
    assert model.untyped_graph is not None
    assert model.untyped_graph.is_relation_type_blind()


def test_connectivity_is_readable(setup) -> None:
    """Dropping or rewiring edges must change the output, or the graph is decoration."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    inverse = builder.inverse_relation_table()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        base = model(batch).part_logits
    for kind in ("drop_spatial", "randomise_spatial_endpoints"):
        damaged = perturb_relations(batch, kind, inverse_relations=inverse, generator=generator)
        with torch.no_grad():
            other = model(damaged).part_logits
        assert not torch.equal(base, other), f"{kind} changed nothing"


def test_endpoint_identity_is_preserved(setup) -> None:
    """"A connects to B" must differ from "A connects to C", or the graph says nothing."""
    batch, _ = setup
    encoder = _encoder()
    latent = torch.randn(batch.batch_size, batch.structure.entity_count, 256)
    with torch.no_grad():
        base = encoder(latent, batch.structure)

    from dataclasses import replace as dataclass_replace

    adjacency = batch.structure.graph_adjacency.clone()
    live = adjacency.any(dim=1).nonzero()
    assert live.numel(), "no edges to move"
    scene, source, target = (int(v) for v in live[0])
    adjacency[scene, :, source, target] = False
    other_target = (target + 7) % adjacency.shape[-1]
    adjacency[scene, 1, source, other_target] = True
    moved = dataclass_replace(batch.structure, graph_adjacency=adjacency)
    with torch.no_grad():
        after = encoder(latent, moved)
    assert not torch.equal(base, after), "moving an edge's endpoint changed nothing"


def test_permuting_entities_permutes_the_output(setup) -> None:
    """The encoder must be equivariant: identity comes from the latent, not the slot."""
    batch, _ = setup
    encoder = _encoder()
    entities = batch.structure.entity_count
    order = torch.randperm(entities)
    latent = torch.randn(batch.batch_size, entities, 256)

    from dataclasses import replace as dataclass_replace

    structure = batch.structure
    adjacency = structure.graph_adjacency[..., order, :][..., :, order]
    permuted = dataclass_replace(
        structure,
        entity_mask=structure.entity_mask[..., order],
        graph_adjacency=adjacency,
    )
    with torch.no_grad():
        direct = encoder(latent[:, order], permuted)
        indirect = encoder(latent, structure)[:, order]
    assert torch.allclose(direct, indirect, atol=1e-5)


def test_the_budget_matches_the_reduced_typed_encoder() -> None:
    """A3Lite must be a capacity comparison, not a smaller model winning by being small."""
    encoder = _encoder()
    graph = encoder.parameter_groups()["graph"]
    a3l_graph_budget = 1_094_415 - 427_655
    assert 0.9 <= graph / a3l_graph_budget <= 1.15, (
        f"{graph:,} parameters against A3L's {a3l_graph_budget:,}"
    )


def test_a_mismatched_head_count_is_rejected() -> None:
    """Silently rounding the head width would corrupt every attention computation."""
    with pytest.raises(ValueError, match="must divide"):
        GraphLiteConfig(width=100, heads=8)


def test_an_identity_subspace_wider_than_the_latent_is_rejected() -> None:
    """The encoder would then have nothing it is allowed to write."""
    with pytest.raises(ValueError, match="narrower"):
        GraphLiteConfig(width=64, identity_width=64)
