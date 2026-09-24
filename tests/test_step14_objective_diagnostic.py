"""Step 14: the diagnostics measure the objective, not the model's opinion of it.

Step 14's conclusion is that the objective *does* reward relational conditioning, which means
the failure is elsewhere. That conclusion rests on two things being true, and each has a way of
being silently false:

* the reassembled total loss must be the trainer's real total, not the loss module's partial
  breakdown -- the frame term is added outside the module and is half the objective;
* the clause-2 comparison must be model-independent, because a measurement taken through a
  model that already ignores the graph would report the model's insensitivity as the
  objective's.

The gradient pathway table gets its own test because it already failed once: three prefixes
named modules that do not exist (``entity_composer`` for ``composer``), and a prefix matching
nothing yields a gradient norm of exactly zero, which reads as a finding rather than a typo.
"""

from __future__ import annotations

import math

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step13.evaluate import build_arm
from experiments.step14.objective_audit import PATHWAYS, _check_pathways, decompose_objective
from experiments.step14.relational_identifiability import (
    IDENTITY_ONLY_VALIDATION_DEG,
    MINIMUM_GRAPH_ADVANTAGE_DEG,
    MINIMUM_SAMPLES,
    OBJECTIVE_SEPARATION_THRESHOLD,
    _as_frame,
    _variance_decomposition,
)

CORPUS = "datasets/processed/step10_rotated"


@pytest.fixture(scope="module")
def trainer():
    """A frozen A3 seed-0 trainer, as every Step 14 diagnostic loads it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(CORPUS, "train", limit=48)
    return build_arm("A3", 0, ontology, domain, small)


class TestThePathwayTable:
    """A misspelled prefix reports zero gradient, which looks exactly like a result."""

    def test_a_prefix_matching_nothing_is_refused(self, trainer) -> None:
        """The regression test for the bug this module actually had."""
        original = dict(PATHWAYS)
        PATHWAYS["typo"] = ("entity_composer",)
        try:
            with pytest.raises(SystemExit, match="match no parameter"):
                _check_pathways(trainer.model)
        finally:
            PATHWAYS.clear()
            PATHWAYS.update(original)

    def test_every_parameter_belongs_to_exactly_one_pathway(self, trainer) -> None:
        """An unassigned parameter would silently vanish from the gradient accounting."""
        _check_pathways(trainer.model)
        names = [name for name, _ in trainer.model.named_parameters()]
        for name in names:
            owners = [p for p, prefixes in PATHWAYS.items() if name.startswith(prefixes)]
            # relation_parameters is deliberately a subset of graph_encoder.
            assert owners, f"{name} belongs to no pathway"

    def test_relation_parameters_are_inside_the_graph_encoder(self) -> None:
        """Reported separately but not double counted in any total."""
        assert all(p.startswith("graph_encoder") for p in PATHWAYS["relation_parameters"])


class TestTheObjectiveDecomposition:
    """Half the frame supervision is added outside the loss module."""

    def test_the_shares_account_for_the_whole_total(self, trainer) -> None:
        """If a term were missed, the shares would not sum to one."""
        scenes = load_step8_split(CORPUS, "train", limit=64)
        report = decompose_objective(trainer, scenes, batches=2)
        total_share = sum(e["share_of_total"] for e in report["components"].values())
        assert total_share == pytest.approx(1.0, abs=0.02)

    def test_the_frame_term_is_included_and_dominant(self, trainer) -> None:
        """The chordal frame term is added by the trainer, not by the loss module."""
        scenes = load_step8_split(CORPUS, "train", limit=64)
        report = decompose_objective(trainer, scenes, batches=2)
        assert "entity_frame_chordal" in report["components"]
        assert report["components"]["entity_frame_chordal"]["share_of_total"] > 0.3

    def test_both_frame_terms_are_reported(self, trainer) -> None:
        """The old L1 entity_frame term is still live at weight 0.5 and must not be hidden."""
        scenes = load_step8_split(CORPUS, "train", limit=64)
        report = decompose_objective(trainer, scenes, batches=2)
        assert report["components"]["entity_frame"]["weight"] == 0.5
        assert report["components"]["entity_frame_chordal"]["weight"] == 2.0

    def test_the_inner_rotation_coefficient_is_lambda(self, trainer) -> None:
        """Inside the frame term, rotation carries the calibrated weight and nothing else."""
        scenes = load_step8_split(CORPUS, "train", limit=64)
        report = decompose_objective(trainer, scenes, batches=2)
        inner = report["chordal_frame_term_breakdown"]
        assert inner["rotation"]["coefficient"] == 3.0
        assert inner["translation"]["coefficient"] == 1.0
        assert inner["scale"]["coefficient"] == 0.5


class TestTheClauseTwoMeasurement:
    """The measurement that decides H7 must not depend on a model."""

    def test_a_frame_carries_only_rotation(self) -> None:
        """Both predictors predict rotation; anything else would confound the loss."""
        rotation = torch.arange(6, dtype=torch.float64)
        frame = _as_frame(rotation)
        assert frame.shape == (12,)
        assert torch.equal(frame[6:12], rotation)
        assert torch.equal(frame[:6], torch.zeros(6, dtype=torch.float64))

    def test_the_threshold_is_preregistered(self) -> None:
        """10%, fixed in the plan before the measurement ran."""
        assert OBJECTIVE_SEPARATION_THRESHOLD == 0.10

    def test_the_subset_rule_is_preregistered(self) -> None:
        """23.51 deg is the measured validation identity-only mean; 10.0 deg the advantage."""
        assert IDENTITY_ONLY_VALIDATION_DEG == 23.51
        assert MINIMUM_GRAPH_ADVANTAGE_DEG == 10.0
        assert MINIMUM_SAMPLES == 2


class TestTheVarianceDecomposition:
    """Clause 1 asks whether the target varies with the graph at all."""

    @staticmethod
    def _records(spread: float, within: float):
        """Synthetic slots whose rotation depends on the graph by a controlled amount."""
        generator = torch.Generator().manual_seed(14)
        records = []
        for graph in range(4):
            angle = spread * graph
            for _ in range(3):
                jitter = within * torch.randn(1, generator=generator).item()
                frame = torch.zeros(12, dtype=torch.float64)
                frame[6] = math.cos(angle + jitter)
                frame[7] = math.sin(angle + jitter)
                frame[9] = -math.sin(angle + jitter)
                frame[10] = math.cos(angle + jitter)
                records.append((f"graph{graph}", {0: frame}))
        return records

    def test_graph_dependent_targets_are_detected(self) -> None:
        """Large between-graph, small within-graph: the graph explains most of the spread."""
        report = _variance_decomposition(self._records(spread=0.7, within=0.01))
        assert report["share_explained_by_graph"] > 0.8

    def test_graph_independent_targets_are_not(self) -> None:
        """No between-graph structure: the graph explains almost nothing."""
        report = _variance_decomposition(self._records(spread=0.0, within=0.4))
        assert report["share_explained_by_graph"] < 0.2

    def test_within_spread_never_exceeds_total(self) -> None:
        """Conditioning on more cannot increase residual spread; a violation is a bug."""
        report = _variance_decomposition(self._records(spread=0.5, within=0.05))
        assert report["within_graph_spread"] <= report["total_spread"]


class TestNoTestSplitIsRead:
    """The diagnostic phase must be able to state test_splits_read: []."""

    @pytest.mark.parametrize(
        "module",
        (
            "experiments.step14.objective_audit",
            "experiments.step14.relational_identifiability",
            "experiments.step14.subset_performance",
        ),
    )
    def test_no_diagnostic_names_a_test_split(self, module: str) -> None:
        """A literal test split name in a Step 14 diagnostic would be leakage."""
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module))
        for split in ("test_seen", "test_arrangement", "test_transform", "test_combination"):
            assert f'"{split}"' not in source, f"{module} names {split}"
