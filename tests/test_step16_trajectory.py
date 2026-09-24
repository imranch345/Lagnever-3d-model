"""Step 16: the instrumentation must not change the run it is measuring.

Step 16 trains three A3 runs with a diagnostic callback and reads the trajectory to find what
precedes escape. The whole study is worthless if the callback perturbs training, and it has
three ways to do so, each silent:

* building a validation batch samples points, advancing the RNG, so an instrumented run would
  see different data from its frozen twin;
* the gradient measurement calls ``backward``, leaving ``.grad`` populated for the next step;
* measuring in ``eval`` mode and forgetting to return to ``train`` would disable nothing visible
  but change dropout and normalisation behaviour.

The first test is the one that matters: same seed, same steps, with and without the callback,
weights compared exactly. The rest pin the pieces that decide how the trajectory reads.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step15.seed_modes import GRAPH_USING_DEG, classify
from experiments.step16.trajectory import (
    EVERY,
    SEED_ROLES,
    TrajectoryRecorder,
    _ordered_validation_batches,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

CORPUS = Path("datasets/processed/step10_rotated")
SHORT_STEPS = 60


@pytest.fixture(scope="module")
def setting():
    """Ontology, domain and a builder, shared by every test here."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return ontology, domain, WholeOrganBatchBuilder(ontology, domain)


def _trainer(ontology, domain, *, seed: int, steps: int):
    """A frozen-configuration trainer with a shortened budget, for cheap comparisons."""
    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
        ),
        rotation_objective="chordal",
        rotation_loss_weight=3.0,
        device="cpu",
        steps=steps,
    )
    train = load_step8_split(CORPUS, "train", limit=96)
    validation = load_step8_split(CORPUS, "validation", limit=32)
    return Step10Trainer(config, ontology, domain, train, validation)


class TestTheCallbackDoesNotChangeTraining:
    """The claim the whole step rests on."""

    def test_an_instrumented_run_matches_an_uninstrumented_one(self, setting) -> None:
        """Same seed, same budget, weights identical to the last bit.

        A callback that advanced the RNG, left gradients behind or stayed in eval mode would
        show up here as diverged weights, and nowhere else.
        """
        ontology, domain, builder = setting
        validation = load_step8_split(CORPUS, "validation", limit=32)
        fixed = WholeOrganLoader(
            load_step8_split(CORPUS, "train", limit=32),
            builder,
            batch_size=8,
            seed=0,
            shuffle=False,
        )

        plain = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        plain.fit()
        baseline = {k: v.clone() for k, v in plain.model.state_dict().items()}

        watched = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        recorder = TrajectoryRecorder(
            corpus_dir=CORPUS,
            builder=builder,
            required=set(),
            validation=validation,
            train_batches=list(fixed.epoch(0)),
        )
        watched.fit(on_step=recorder)
        after = watched.model.state_dict()

        differing = [k for k in baseline if not torch.equal(baseline[k], after[k])]
        assert differing == [], f"instrumentation changed training for {differing}"
        assert len(recorder.rows) >= 2

    def test_the_callback_leaves_the_model_in_training_mode(self, setting) -> None:
        """Measuring happens in eval; training must resume in train."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        trainer.model.train()
        recorder = TrajectoryRecorder(
            corpus_dir=CORPUS,
            builder=builder,
            required=set(),
            validation=load_step8_split(CORPUS, "validation", limit=16),
            train_batches=[],
        )
        recorder(trainer, 0)
        assert trainer.model.training

    def test_the_callback_leaves_no_gradients_behind(self, setting) -> None:
        """The gradient measurement calls backward; it must clean up after itself."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        fixed = WholeOrganLoader(
            load_step8_split(CORPUS, "train", limit=16),
            builder,
            batch_size=8,
            seed=0,
            shuffle=False,
        )
        recorder = TrajectoryRecorder(
            corpus_dir=CORPUS,
            builder=builder,
            required=set(),
            validation=load_step8_split(CORPUS, "validation", limit=16),
            train_batches=list(fixed.epoch(0)),
        )
        recorder(trainer, 0)
        leftover = [
            name
            for name, parameter in trainer.model.named_parameters()
            if parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        ]
        assert leftover == []


class TestTheCheckpointAccounting:
    """Every conclusion is read off a step index, so the indices must be right."""

    def test_measurements_land_on_the_declared_interval(self, setting) -> None:
        """Fifty steps, plus the final step, and nothing in between."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        recorder = TrajectoryRecorder(
            corpus_dir=CORPUS,
            builder=builder,
            required=set(),
            validation=load_step8_split(CORPUS, "validation", limit=16),
            train_batches=[],
        )
        for step in range(SHORT_STEPS):
            recorder(trainer, step)
        recorder(trainer, SHORT_STEPS)
        steps = [row["step"] for row in recorder.rows]
        assert steps == [0, 50, SHORT_STEPS]

    def test_the_interval_is_fifty(self) -> None:
        """Fixed by the brief, not chosen after seeing a trajectory."""
        assert EVERY == 50

    def test_checkpoints_are_ordered_and_unique(self, setting) -> None:
        """A repeated or out-of-order step would corrupt every ordering claim."""
        ontology, domain, builder = setting
        trainer = _trainer(ontology, domain, seed=9, steps=SHORT_STEPS)
        recorder = TrajectoryRecorder(
            corpus_dir=CORPUS,
            builder=builder,
            required=set(),
            validation=load_step8_split(CORPUS, "validation", limit=16),
            train_batches=[],
        )
        for step in (0, 50):
            recorder(trainer, step)
        steps = [row["step"] for row in recorder.rows]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))


class TestTheValidationAlignment:
    """Per-entity errors are joined to a preregistered subset; a mis-join is invisible."""

    def test_every_validation_scene_appears_exactly_once(self, setting) -> None:
        """Batching by level of detail must not drop or duplicate a scene."""
        _, _, builder = setting
        scenes = load_step8_split(CORPUS, "validation")
        batches = _ordered_validation_batches(scenes, builder)
        seen = [scene for _, group in batches for scene in group]
        assert len(seen) == len(scenes)
        assert {id(s) for s in seen} == {id(s) for s in scenes}

    def test_each_batch_shares_one_level_of_detail(self, setting) -> None:
        """The builder refuses a mixed batch; this is why the grouping exists."""
        _, _, builder = setting
        scenes = load_step8_split(CORPUS, "validation")
        for _, group in _ordered_validation_batches(scenes, builder):
            assert len({scene.active_lod for scene in group}) == 1

    def test_the_order_is_deterministic(self, setting) -> None:
        """Two calls must produce the same batches, or trajectories are incomparable."""
        _, _, builder = setting
        scenes = load_step8_split(CORPUS, "validation")
        first = [[id(s) for s in g] for _, g in _ordered_validation_batches(scenes, builder)]
        second = [[id(s) for s in g] for _, g in _ordered_validation_batches(scenes, builder)]
        assert first == second


class TestTheSeedSelection:
    """The three roles were fixed from the frozen Step 15 report, before instrumenting."""

    def test_the_three_roles_are_declared(self) -> None:
        """Late escaper, non-escaper, intermediate."""
        assert set(SEED_ROLES) == {9, 5, 3}

    def test_escape_uses_the_step_15_threshold(self) -> None:
        """No new rule: the crossing is Step 15's graph-using boundary."""
        assert GRAPH_USING_DEG == 1.5
        assert classify(GRAPH_USING_DEG) == "graph_using"
        assert classify(GRAPH_USING_DEG - 0.01) == "intermediate"


class TestTheDepartureCriterion:
    """The rule that decides when a series "first changes", and so the whole ordering."""

    @staticmethod
    def _flat_then_step(jump_at: int, jump: float) -> dict[int, float]:
        """A quiet series that shifts once, so the detected step is known in advance."""
        return {
            step: (0.0 if step < jump_at else jump) + (0.01 if step % 100 else -0.01)
            for step in range(0, 1250, 50)
        }

    def test_a_real_shift_is_detected_at_its_first_checkpoint(self) -> None:
        """A large, persistent departure lands on the checkpoint where it starts."""
        from experiments.step16.analysis import _first_departure

        result = _first_departure(self._flat_then_step(850, 5.0), (150, 800))
        assert result["detected"]
        assert result["step"] == 850

    def test_a_single_checkpoint_excursion_is_not_detected(self) -> None:
        """Persistence is required: one point at this resolution is indistinguishable from noise."""
        from experiments.step16.analysis import _first_departure

        values = {step: 0.0 for step in range(0, 1250, 50)}
        values[900] = 99.0
        assert not _first_departure(values, (150, 800))["detected"]

    def test_a_transient_in_the_baseline_hides_later_change(self) -> None:
        """Documents why the preregistered 0-400 window could not fire for rotation.

        The window contains the initial convergence, so its standard deviation dwarfs every
        later movement. This is the measured reason the plan's criterion was reported alongside
        a post-hoc one rather than quietly replaced.
        """
        from experiments.step16.analysis import _first_departure

        values = {0: 61.7, 50: 41.6, 100: 24.7, 150: 23.9, 200: 23.6, 250: 23.7}
        values.update({step: 23.5 for step in range(300, 850, 50)})
        values.update({step: 21.0 for step in range(850, 1250, 50)})
        assert not _first_departure(values, (0, 400))["detected"]
        assert _first_departure(values, (150, 800))["detected"]


class TestTheSeparationCriterion:
    """Between-group difference must beat within-group difference to count."""

    def test_a_clean_separation_is_found(self) -> None:
        """Two series together, one apart, gap wider than the pair's own spread."""
        from experiments.step16.analysis import _separation

        steps = range(0, 900, 50)
        a = {s: 10.0 for s in steps}
        b = {s: 10.1 for s in steps}
        other = {s: 5.0 for s in steps}
        result = _separation([a, b], other, until=850)
        assert result["separated"]
        assert result["earliest_sustained_step"] == 0

    def test_a_gap_smaller_than_the_within_spread_does_not_count(self) -> None:
        """Two escapers disagreeing more than they differ from the third proves nothing."""
        from experiments.step16.analysis import _separation

        steps = range(0, 900, 50)
        a = {s: 10.0 for s in steps}
        b = {s: 20.0 for s in steps}
        other = {s: 14.0 for s in steps}
        assert not _separation([a, b], other, until=850)["separated"]

    def test_straddling_the_third_series_does_not_count(self) -> None:
        """Same-side is required, or "separation" is just being on either flank."""
        from experiments.step16.analysis import _separation

        steps = range(0, 900, 50)
        a = {s: 4.0 for s in steps}
        b = {s: 6.0 for s in steps}
        other = {s: 5.0 for s in steps}
        assert not _separation([a, b], other, until=850)["separated"]

    def test_the_grouping_control_flags_an_uninformative_test(self) -> None:
        """If an arbitrary grouping separates as much, the real hit means nothing.

        This is the control that decided Step 16's verdict, so its logic is pinned: three
        mutually unrelated series should let every grouping find a "separation".
        """
        from experiments.step16.analysis import _grouping_control

        def rows(offset: float) -> list[dict]:
            return [
                {
                    "step": step,
                    "validation_rotation_deg": offset + step * 0.0,
                    "graph_usage_deg": offset,
                    "relational_subset_deg": offset,
                    "representation": {
                        "context_written_norm": offset,
                        "context_written_share": offset,
                        "entity_latent_std": offset,
                        "relation_bias_abs_max": offset,
                        "frame_rotation_output_std": offset,
                    },
                    "gradients": {
                        "graph_encoder": offset,
                        "relation_parameters": offset,
                        "frame_head": offset,
                    },
                    "objective": {
                        "frame_rotation": offset,
                        "frame_chordal": offset,
                        "total": offset,
                    },
                }
                for step in range(0, 900, 50)
            ]

        control = _grouping_control({9: rows(1.0), 5: rows(2.0), 3: rows(3.0)}, until=850)
        assert "no discriminating power" in control["verdict"]
