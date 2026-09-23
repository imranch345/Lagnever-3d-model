"""Step 12: relation-conditioned values, and the isolation the comparison depends on.

Step 11 measured relation type as worth 0.0003 degrees and found the reason in the weights:
type entered only as a scalar attention bias carrying 0.55% of the logit scale. Step 12 gives
each relation a vector as well, so a relation can say something rather than only say it louder.

The first class is the one that matters most. The treatment adds a parameter tensor, and a new
tensor ordinarily consumes RNG at construction and shifts every weight initialised after it.
That would make the control differ from the treatment by seed *as well as* by the channel under
test — and A3's seed spread is about 1.0 deg, the same size as the effect being measured. The
confound would have been fatal and invisible, so it is pinned here rather than trusted.

The rest check that the term, once it exists, is actually a function of relation type and is
routed the same way the bias it joins is routed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from generation.neural.nn.graph import GraphEncoderConfig, PartitionedGraphEncoder
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model

#: The plan's stop condition: this is meant to be the *smallest* change that gives a relation
#: its own message, and a larger one is a different experiment.
MAXIMUM_PARAMETER_GROWTH = 0.01


@pytest.fixture(scope="module")
def setup():
    """A real batch and the builder that made it."""
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


def _a3(builder, *, relation_values: bool, seed: int = 0):
    """One A3 model, built the way the runner builds it."""
    torch.manual_seed(seed)
    model, _ = build_model(
        "A3",
        builder,
        text_features=64,
        overrides={"relation_values": relation_values},
    )
    return model


class TestTheControlIsUnchanged:
    """The treatment must differ from the control by one tensor and nothing else."""

    def test_the_two_models_share_every_pre_existing_weight(self, setup) -> None:
        """The isolation guarantee, at the level of raw weights.

        If a single shared parameter differed, the comparison would be measuring a different
        initialisation as well as a different architecture, and at this effect size the two
        are indistinguishable.
        """
        _, builder = setup
        off = _a3(builder, relation_values=False).state_dict()
        on = _a3(builder, relation_values=True).state_dict()
        added = set(on) - set(off)
        assert added == {"graph_encoder.relation_value.weight"}
        assert not set(off) - set(on)
        differing = [name for name in off if not torch.equal(off[name], on[name])]
        assert differing == [], f"initialisation shifted for {differing}"

    def test_the_new_tensor_starts_at_zero(self, setup) -> None:
        """Zero initialisation is what makes the treatment start as the control."""
        _, builder = setup
        model = _a3(builder, relation_values=True)
        weight = model.state_dict()["graph_encoder.relation_value.weight"]
        assert torch.equal(weight, torch.zeros_like(weight))

    def test_the_latent_is_identical_at_initialisation(self, setup) -> None:
        """The behavioural form of the same guarantee: same seed, same forward pass."""
        batch, builder = setup
        off = _a3(builder, relation_values=False)
        on = _a3(builder, relation_values=True)
        off.eval()
        on.eval()
        with torch.no_grad():
            assert torch.equal(off.stage(batch).entity_latent, on.stage(batch).entity_latent)

    def test_the_encoder_is_off_by_default(self, setup) -> None:
        """Every result up to Step 11 was measured without this term."""
        _, builder = setup
        assert GraphEncoderConfig().relation_values is False
        encoder = PartitionedGraphEncoder(GraphEncoderConfig(), builder.inverse_relation_table())
        assert encoder.relation_value is None

    def test_the_change_stays_small(self, setup) -> None:
        """A stop condition from the plan, not a style preference."""
        _, builder = setup
        off = sum(p.numel() for p in _a3(builder, relation_values=False).parameters())
        on = sum(p.numel() for p in _a3(builder, relation_values=True).parameters())
        assert (on - off) / off < MAXIMUM_PARAMETER_GROWTH


class TestTheTermDependsOnRelationType:
    """A term that ignores its relation would reproduce exactly the defect it is fixing."""

    @staticmethod
    def _trained_like(builder, *, seed: int = 0):
        """A model whose relation vectors are non-zero, as they are after any training step."""
        model = _a3(builder, relation_values=True, seed=seed)
        weight = model.graph_encoder.relation_value.weight
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape, generator=torch.Generator().manual_seed(12)))
        model.eval()
        return model

    def test_permuting_the_relation_vectors_changes_the_latent(self, setup) -> None:
        """The isolation Step 11's `shuffle_spatial_types` performs, done to the weights.

        Same endpoints, same attention, different vector per relation. If the latent does not
        move, the term is not reading relation type.
        """
        batch, builder = setup
        model = self._trained_like(builder)
        with torch.no_grad():
            before = model.stage(batch).entity_latent.clone()
            weight = model.graph_encoder.relation_value.weight
            weight.copy_(
                weight[torch.randperm(weight.shape[0], generator=torch.Generator().manual_seed(3))]
            )
            after = model.stage(batch).entity_latent
        assert not torch.allclose(before, after)

    def test_non_zero_vectors_change_the_latent_at_all(self, setup) -> None:
        """A guard against the term being computed and then discarded."""
        batch, builder = setup
        plain = _a3(builder, relation_values=True)
        plain.eval()
        trained = self._trained_like(builder)
        with torch.no_grad():
            assert not torch.allclose(
                plain.stage(batch).entity_latent, trained.stage(batch).entity_latent
            )

    def test_the_identity_slice_survives_the_new_term(self, setup) -> None:
        """Protected identity is protected regardless of what the encoder learns to add."""
        batch, builder = setup
        model = self._trained_like(builder)
        encoder = model.graph_encoder
        latent = torch.randn(batch.batch_size, batch.structure.entity_count, encoder.config.width)
        with torch.no_grad():
            out = encoder(latent, batch.structure)
        width = encoder.config.identity_width
        assert torch.equal(out[..., :width], latent[..., :width])


class TestTheRoutingMatchesTheBias:
    """The value term and the bias term must agree about what a relation is and who sees it."""

    def test_routing_only_places_edges_on_heads_that_may_see_them(self, setup) -> None:
        """A head scoped to one graph must not receive another graph's relations."""
        batch, builder = setup
        encoder = _a3(builder, relation_values=True).graph_encoder
        routing = encoder.build_routing(batch.structure, batch.batch_size)
        assert routing is not None
        for position in routing.head.unique().tolist():
            assert int(encoder.head_graph[position].item()) >= 0

    def test_routing_covers_both_directions_of_an_edge(self, setup) -> None:
        """Direction is carried by the inverse relation, as it is in the bias."""
        batch, builder = setup
        encoder = _a3(builder, relation_values=True).graph_encoder
        routing = encoder.build_routing(batch.structure, batch.batch_size)
        assert routing is not None
        pairs = {(int(q), int(k)) for q, k in zip(routing.query, routing.key, strict=True)}
        reversed_pairs = {(k, q) for q, k in pairs}
        assert pairs & reversed_pairs, "no edge is routed in both directions"

    def test_an_empty_graph_routes_nothing(self, setup) -> None:
        """With no live edges there is no mass to place, and the term must vanish cleanly."""
        batch, builder = setup
        from experiments.step7.perturbations import restrict_graphs

        encoder = _a3(builder, relation_values=True).graph_encoder
        bare = restrict_graphs(batch, keep=()).structure
        assert encoder.build_routing(bare, batch.batch_size) is None


class TestTheDecisionArithmetic:
    """The pre-registered rule, checked against cases whose answer is known in advance.

    These are the functions that turn six numbers into the step's conclusion, so each branch is
    exercised with a case constructed to land on it.
    """

    @staticmethod
    def _verdict(control: dict[str, float], treatment: dict[str, float]) -> dict:
        from experiments.step12.compare import _score_primary

        return _score_primary(control, treatment)

    def test_a_large_consistent_improvement_passes(self) -> None:
        """Every seed better, mean well over the minimum."""
        result = self._verdict(
            {"0": 20.96, "1": 23.43, "2": 21.01}, {"0": 19.0, "1": 21.0, "2": 19.5}
        )
        assert result["verdict"] == "improves"
        assert result["exceeds_minimum"]

    def test_a_small_improvement_does_not_pass(self) -> None:
        """Below A3's own seed spread is indistinguishable from it, however consistent."""
        result = self._verdict(
            {"0": 20.96, "1": 23.43, "2": 21.01}, {"0": 20.7, "1": 23.1, "2": 20.8}
        )
        assert result["verdict"] == "no meaningful change"

    def test_a_large_but_inconsistent_change_does_not_pass(self) -> None:
        """Sign disagreement disqualifies however big the mean."""
        result = self._verdict(
            {"0": 20.96, "1": 23.43, "2": 21.01}, {"0": 17.0, "1": 25.0, "2": 18.0}
        )
        assert not result["all_seeds_same_sign"]
        assert result["verdict"] == "no meaningful change"

    def test_a_consistent_regression_is_reported_as_worse(self) -> None:
        """A harmful result has its own verdict and is not folded into 'no change'."""
        result = self._verdict(
            {"0": 20.96, "1": 23.43, "2": 21.01}, {"0": 23.0, "1": 25.5, "2": 23.2}
        )
        assert result["verdict"] == "worse"
        assert result["mean_gain_deg"] < 0

    def test_p1_needs_the_shuffle_to_become_expensive(self) -> None:
        """Step 11 measured 0.0003 deg; P1 asks for at least 1.0."""
        from experiments.step12.compare import _score_p1

        cheap = {
            "conditions": {
                "shuffle_spatial_types": {
                    "delta_rotation_deg": 0.0003,
                    "per_seed_delta": {"0": 0.0003, "1": 0.0002, "2": 0.0002},
                    "all_seeds_same_sign": True,
                }
            }
        }
        assert not _score_p1(cheap)["holds"]
        expensive = {
            "conditions": {
                "shuffle_spatial_types": {
                    "delta_rotation_deg": 2.4,
                    "per_seed_delta": {"0": 2.5, "1": 2.2, "2": 2.5},
                    "all_seeds_same_sign": True,
                }
            }
        }
        assert _score_p1(expensive)["holds"]

    def test_p2_is_an_ordering_not_a_threshold(self) -> None:
        """P2 asks where the gain lands, and holds even when every gain is small."""
        from experiments.step12.compare import _score_p2

        assert _score_p2({"0": 0.3, "1": 0.05, "2": 0.25})["holds"]
        assert not _score_p2({"0": 0.3, "1": 2.0, "2": 0.25})["holds"]


class TestTheMagnitudeDiagnostic:
    """The 0.33% figure is only meaningful if the replicated pass is the model's own pass.

    ``term_magnitude`` recomputes each layer by hand in order to separate the ordinary attended
    value from the relation message, which the layer itself adds together and never exposes. A
    replication that drifted from the real forward pass would produce a plausible number for a
    computation the model never performed, so the drift is checked rather than assumed.
    """

    def test_the_replicated_pass_reproduces_the_encoder(self, setup) -> None:
        """Hand-rolled layers must agree with the encoder they are imitating."""
        import torch as _torch

        from experiments.step12.term_magnitude import _per_layer

        batch, builder = setup
        model = _a3(builder, relation_values=True)
        weight = model.graph_encoder.relation_value.weight
        with _torch.no_grad():
            weight.copy_(_torch.randn(weight.shape, generator=_torch.Generator().manual_seed(12)))
        model.eval()
        encoder = model.graph_encoder
        latent = _torch.randn(batch.batch_size, batch.structure.entity_count, encoder.config.width)
        with _torch.no_grad():
            expected = encoder(latent, batch.structure)
            rows = _per_layer(
                encoder, latent, batch.structure, batch.batch_size, _return_latent=True
            )
        assert _torch.allclose(rows["latent"], expected, atol=1e-5)

    def test_a_zero_weight_gives_a_zero_share(self, setup) -> None:
        """The scale's zero point: an untrained term contributes nothing measurable."""
        import torch as _torch

        from experiments.step12.term_magnitude import _per_layer

        batch, builder = setup
        model = _a3(builder, relation_values=True)
        model.eval()
        latent = _torch.randn(
            batch.batch_size, batch.structure.entity_count, model.graph_encoder.config.width
        )
        with _torch.no_grad():
            rows = _per_layer(model.graph_encoder, latent, batch.structure, batch.batch_size)
        assert all(row["share_of_attended"] == 0.0 for row in rows)
