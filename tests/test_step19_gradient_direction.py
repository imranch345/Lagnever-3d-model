"""Step 19: direction measurement that must not disturb the run it measures.

This callback is heavier than Step 16's -- it runs extra backward passes and holds flattened
gradients -- so the neutrality claim needs retesting rather than inheriting. If it perturbed
training, the trajectory would not be seed 9's any more and nothing measured would bear on the
escape.

The other tests pin the two quantities' meanings, because they carry different licences: the
successive cosine uses only information available at step t and may be read as a candidate
precursor, while the escape-axis cosine is built from the end of the run and can only ever be
descriptive. Confusing them would turn a tautology into a finding.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step14.objective_audit import PATHWAYS
from experiments.step16.trajectory import DENSE_EVERY, TrajectoryRecorder
from experiments.step19.gradient_direction import DirectionRecorder, _cosine
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

CORPUS = Path("datasets/processed/step10_rotated")
SHORT_STEPS = 60
WINDOW = (20, 50)


@pytest.fixture(scope="module")
def setting():
    """Ontology, domain and a builder, shared across this module."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return ontology, domain, WholeOrganBatchBuilder(ontology, domain)


def _trainer(ontology, domain, *, seed: int = 9, steps: int = SHORT_STEPS):
    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
        ),
        rotation_objective="chordal",
        rotation_loss_weight=3.0,
        device="cpu",
        steps=steps,
    )
    return Step10Trainer(
        config,
        ontology,
        domain,
        load_step8_split(CORPUS, "train", limit=96),
        load_step8_split(CORPUS, "validation", limit=32),
    )


def _recorder(builder, *, window=WINDOW):
    fixed = WholeOrganLoader(
        load_step8_split(CORPUS, "train", limit=32), builder, batch_size=8, seed=0, shuffle=False
    )
    inner = TrajectoryRecorder(
        corpus_dir=CORPUS,
        builder=builder,
        required=set(),
        validation=load_step8_split(CORPUS, "validation", limit=32),
        train_batches=list(fixed.epoch(0)),
        dense_window=window,
    )
    return DirectionRecorder(inner, window=window)


class TestTheCallbackDoesNotChangeTraining:
    """Heavier than Step 16's callback, so the claim is retested rather than inherited."""

    def test_an_instrumented_run_matches_an_uninstrumented_one(self, setting) -> None:
        """Extra backward passes must leave the optimiser's trajectory untouched."""
        ontology, domain, builder = setting
        plain = _trainer(ontology, domain)
        plain.fit()
        baseline = {k: v.clone() for k, v in plain.model.state_dict().items()}

        watched = _trainer(ontology, domain)
        recorder = _recorder(builder)
        watched.fit(on_step=recorder)

        after = watched.model.state_dict()
        differing = [k for k in baseline if not torch.equal(baseline[k], after[k])]
        assert differing == [], f"instrumentation changed training for {differing}"
        assert recorder.rows, "the window never opened"

    def test_no_gradients_are_left_behind(self, setting) -> None:
        """The direction pass calls backward several times and must clean up."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        recorder(trainer, WINDOW[0])
        leftover = [
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        ]
        assert leftover == []

    def test_training_mode_is_restored(self, setting) -> None:
        """Measurement happens in train mode for gradients, eval for metrics."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        trainer.model.train()
        _recorder(builder)(trainer, WINDOW[0])
        assert trainer.model.training


class TestTheCosine:
    """The quantity the whole step turns on."""

    def test_identical_directions_score_one(self) -> None:
        """The upper bound of the scale."""
        vector = torch.randn(64)
        assert _cosine(vector, vector) == pytest.approx(1.0, abs=1e-6)

    def test_opposed_directions_score_minus_one(self) -> None:
        """The lower bound, which a reversal would produce."""
        vector = torch.randn(64)
        assert _cosine(vector, -vector) == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_directions_score_zero(self) -> None:
        """The midpoint: no shared direction at all."""
        first = torch.zeros(4)
        first[0] = 1.0
        second = torch.zeros(4)
        second[1] = 1.0
        assert _cosine(first, second) == pytest.approx(0.0, abs=1e-6)

    def test_scale_does_not_matter(self) -> None:
        """Direction only: a norm change must not move the cosine."""
        first = torch.randn(32)
        second = torch.randn(32)
        assert _cosine(first, second) == pytest.approx(_cosine(first * 1000.0, second), abs=1e-5)

    def test_a_zero_vector_is_not_silently_scored(self) -> None:
        """A degenerate pathway must read as missing, not as alignment."""
        assert math.isnan(_cosine(torch.zeros(8), torch.randn(8)))


class TestTheWindowAndPathways:
    """Where the measurement happens, and over which parameters."""

    def test_direction_is_measured_only_inside_the_window(self, setting) -> None:
        """Outside it, the inner recorder still runs but no direction row is added."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        for step in (0, 5, 15):
            recorder(trainer, step)
        assert recorder.rows == []
        recorder(trainer, WINDOW[0])
        assert [row["step"] for row in recorder.rows] == [WINDOW[0]]

    def test_the_dense_interval_is_shared_with_step_16(self) -> None:
        """One definition of "every fifth step", not two."""
        assert DENSE_EVERY == 5

    def test_the_first_window_checkpoint_has_no_successive_cosine(self, setting) -> None:
        """There is no previous gradient yet; reporting one would be inventing it."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        recorder(trainer, WINDOW[0])
        assert "cosine_successive_all_parameters" not in recorder.rows[0]
        recorder(trainer, WINDOW[0] + DENSE_EVERY)
        assert "cosine_successive_all_parameters" in recorder.rows[1]

    def test_every_pathway_is_covered(self, setting) -> None:
        """Norms are reported per pathway, using Step 14's partition."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        recorder(trainer, WINDOW[0])
        for pathway in PATHWAYS:
            assert f"norm_{pathway}" in recorder.rows[0]
        assert "norm_all_parameters" in recorder.rows[0]


class TestTheEscapeAxis:
    """Built from the end of the run, so its licence is narrower and must stay labelled."""

    def test_it_is_absent_until_the_run_finishes(self, setting) -> None:
        """Without the final weights there is no axis, and none is invented."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        recorder(trainer, WINDOW[0])
        assert recorder.finalise()["escape_axis"] is None

    def test_it_declares_that_it_uses_future_information(self, setting) -> None:
        """The field exists so no reader can mistake it for a predictive signal."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        trainer.fit(on_step=recorder)
        axis = recorder.finalise()["escape_axis"]
        assert axis is not None
        assert axis["uses_future_information"] is True
        assert "not a" in axis["interpretation"]

    def test_the_axis_cosine_is_attached_to_every_window_row(self, setting) -> None:
        """Computed retrospectively, but for all of them."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain)
        recorder = _recorder(builder)
        trainer.fit(on_step=recorder)
        recorder.finalise()
        for row in recorder.rows:
            assert "cosine_escape_axis_all_parameters" in row
