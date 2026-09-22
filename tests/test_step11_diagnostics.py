"""Step 11: the diagnostics measure what they claim to.

The Step 11 conclusion rests on a negative result — destroying relation type and endpoint
identity changes A3's output by almost nothing. A negative result is only worth anything if
the destruction actually happened, so the first class here asserts that every perturbation
really does alter the tensors the model reads. A silent no-op would have produced the same
table for entirely the wrong reason.

The rest pin the summary arithmetic and the probe's scoring, both of which decide how the
numbers read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step7.perturbations import perturb_relations, restrict_graphs
from experiments.step11.graph_ablation import CONDITIONS, MINIMUM_MEANINGFUL_DEG
from experiments.step11.representation_probe import TAPS, _score_probe
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

CORPUS = "datasets/processed/step10_rotated"


@pytest.fixture(scope="module")
def batch():
    """One real validation batch, as the model receives it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = load_step8_split(CORPUS, "validation", limit=16)
    level = scenes[0].active_lod
    chosen = [scene for scene in scenes if scene.active_lod == level][:8]
    return builder.build(chosen).batch, builder.inverse_relation_table()


class TestThePerturbationsReallyPerturb:
    """The negative result depends entirely on these being real changes."""

    def test_the_batch_starts_with_live_edges(self, batch) -> None:
        """A control: without edges to destroy, every ablation below is vacuous."""
        structure = batch[0].structure
        assert int(structure.edge_mask.sum()) > 100
        assert int(structure.graph_adjacency.sum()) > 100

    @pytest.mark.parametrize("kind", ("drop_spatial", "drop_functional", "drop_structure"))
    def test_dropping_a_graph_removes_edges(self, batch, kind: str) -> None:
        """Each typed graph really leaves, from the edge list and the adjacency alike."""
        source, inverse = batch
        before = int(source.structure.edge_mask.sum())
        after = perturb_relations(source, kind, inverse_relations=inverse).structure
        assert int(after.edge_mask.sum()) < before
        assert not torch.equal(after.graph_adjacency, source.structure.graph_adjacency)

    def test_shuffling_types_changes_types_and_nothing_else(self, batch) -> None:
        """The isolation the relation-type claim needs: same endpoints, different relations."""
        source, inverse = batch
        generator = torch.Generator().manual_seed(11)
        after = perturb_relations(
            source, "shuffle_spatial_types", inverse_relations=inverse, generator=generator
        ).structure
        assert not torch.equal(after.edge_relation, source.structure.edge_relation)
        assert torch.equal(after.graph_adjacency, source.structure.graph_adjacency)
        assert torch.equal(after.edge_mask, source.structure.edge_mask)

    def test_randomising_endpoints_rewires_the_adjacency(self, batch) -> None:
        """The other isolation: same relation types, different entities."""
        source, inverse = batch
        generator = torch.Generator().manual_seed(11)
        after = perturb_relations(
            source, "randomise_spatial_endpoints", inverse_relations=inverse, generator=generator
        ).structure
        assert not torch.equal(after.graph_adjacency, source.structure.graph_adjacency)

    def test_entities_only_removes_every_relationship(self, batch) -> None:
        """The floor of the series: no edges at all."""
        after = restrict_graphs(batch[0], keep=()).structure
        assert int(after.edge_mask.sum()) == 0
        assert int(after.graph_adjacency.sum()) == 0

    def test_every_declared_condition_is_runnable(self, batch) -> None:
        """The plan names seven conditions; none may be a typo discovered mid-sweep."""
        source, inverse = batch
        for condition in CONDITIONS:
            generator = torch.Generator().manual_seed(11)
            damaged = (
                restrict_graphs(source, keep=())
                if condition == "entities_only"
                else perturb_relations(
                    source, condition, inverse_relations=inverse, generator=generator
                )
            )
            assert damaged.structure.edge_mask.shape == source.structure.edge_mask.shape


class TestTheProbeScoring:
    """A probe's number is only meaningful if the scoring is the reports' metric."""

    def test_a_perfect_probe_scores_zero(self) -> None:
        """The scale's zero point, to the precision the metric actually has.

        Not exactly zero: ``rotation_angle`` goes through ``arccos``, whose derivative is
        unbounded at 1, so it amplifies the square root of the floating-point epsilon. In
        float32 a perfect match reads as a few thousandths of a degree. That is the metric's
        own noise floor, three orders below the smallest effect this study calls meaningful,
        and it is pinned here so it cannot drift unnoticed.
        """
        target = torch.randn(64, 6, dtype=torch.float32)
        identity = nn.Identity()
        assert _score_probe(identity, target, target) < 0.05

    def test_a_probe_predicting_the_identity_rotation_scores_the_target_angle(self) -> None:
        """A probe that learned nothing scores the target's own distance from the identity."""
        angle = math.radians(30.0)
        target = torch.zeros(32, 6)
        target[:, 0] = math.cos(angle)
        target[:, 1] = math.sin(angle)
        target[:, 3] = -math.sin(angle)
        target[:, 4] = math.cos(angle)
        flat = torch.zeros(32, 6)
        flat[:, 0] = 1.0
        flat[:, 4] = 1.0

        class _Constant(nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return flat[: inputs.shape[0]]

        assert _score_probe(_Constant(), target, target) == pytest.approx(30.0, abs=0.5)

    def test_both_taps_are_named(self) -> None:
        """The control tap and the head's own input, in that order."""
        assert TAPS == ("entity_in", "entity_latent")


class TestTheDecisionArithmetic:
    """How a difference becomes 'meaningful' decides every conclusion in the report."""

    def test_the_threshold_is_the_seed_spread(self) -> None:
        """1.0 deg, because A3's own spread is 1.03 and anything smaller is indistinguishable."""
        assert MINIMUM_MEANINGFUL_DEG == 1.0

    @staticmethod
    def _verdict(deltas: list[float]) -> bool:
        same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
        return bool(abs(float(np.mean(deltas))) >= MINIMUM_MEANINGFUL_DEG and same_sign)

    def test_a_large_consistent_change_is_meaningful(self) -> None:
        """The entities_only case: +2.59, +0.38, +2.51."""
        assert self._verdict([2.5903, 0.3765, 2.5096])

    def test_a_tiny_consistent_change_is_not(self) -> None:
        """The relation-type case: three ten-thousandths of a degree, all positive."""
        assert not self._verdict([0.0003, 0.0002, 0.0002])

    def test_a_large_inconsistent_change_is_not(self) -> None:
        """Sign disagreement disqualifies however big the mean."""
        assert not self._verdict([3.0, -3.2, 3.1])
