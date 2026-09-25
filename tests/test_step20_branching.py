"""Step 20: the branching machinery, which has to be exact before any branch means anything.

The experiment holds a mid-training state fixed and varies only the data ordering after it. Two
things must be true for that to be the experiment it claims to be, and neither is obvious:

* **resumption must be exact.** ``WholeOrganLoader.epoch`` draws two permutations up front to
  build a whole epoch's batch list, so a generator state saved mid-epoch has already spent the
  draws that produced the current epoch. Restoring it would rebuild a *different* epoch and the
  "reference" continuation would silently not be a continuation.
* **branches must start bit-identical.** If the branch seed reached the weights or the optimiser
  moments, the branches would differ in more than ordering and the result would be
  uninterpretable.

The reference control is what validates the whole setup, so the tests here are mostly about it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step20.branching import BRANCH_POINT, BRANCH_SEEDS, SEED, WINDOW
from experiments.step20.state import (
    ResumableStream,
    StreamState,
    fingerprint,
    load_state,
    save_state,
)
from training.step10 import Step10Trainer, default_config, with_variant

CORPUS = Path("datasets/processed/step10_rotated")


@pytest.fixture(scope="module")
def trainer():
    """A short-budget trainer at the frozen configuration."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=SEED
        ),
        rotation_objective="chordal",
        rotation_loss_weight=3.0,
        device="cpu",
        steps=40,
    )
    return Step10Trainer(
        config,
        ontology,
        domain,
        load_step8_split(CORPUS, "train", limit=96),
        load_step8_split(CORPUS, "validation", limit=16),
    )


def _groups(whole) -> list[int]:
    """A batch's identity, as the entity-frame contents which differ per scene group."""
    return [round(float(whole.batch.entity_frames.sum()), 6)]


class TestTheStreamResumesExactly:
    """The reference continuation depends entirely on this."""

    def test_restoring_mid_epoch_yields_the_same_batches(self, trainer) -> None:
        """Take 5, save, take 5 more; restore and the second 5 must match.

        This is the test that catches saving the generator mid-epoch instead of at the epoch
        boundary: a mid-epoch save rebuilds a different permutation and the batches diverge.
        """
        stream = ResumableStream(trainer.train_loader)
        iterator = stream.iterate()
        for _ in range(5):
            next(iterator)
        state = stream.state()
        expected = [_groups(next(iterator)) for _ in range(5)]

        resumed = ResumableStream(trainer.train_loader)
        again = resumed.restore(state)
        actual = [_groups(next(again)) for _ in range(5)]
        assert actual == expected

    def test_restoring_across_an_epoch_boundary_is_exact(self, trainer) -> None:
        """The stream must keep going correctly once the restored epoch is exhausted."""
        stream = ResumableStream(trainer.train_loader)
        iterator = stream.iterate()
        for _ in range(3):
            next(iterator)
        state = stream.state()
        expected = [_groups(next(iterator)) for _ in range(30)]

        resumed = ResumableStream(trainer.train_loader)
        again = resumed.restore(state)
        assert [_groups(next(again)) for _ in range(30)] == expected

    def test_a_different_seed_gives_a_different_order(self, trainer) -> None:
        """The intervention must actually intervene."""
        trainer.train_loader.generator.manual_seed(2000)
        first = [_groups(w) for w in [next(ResumableStream(trainer.train_loader).iterate())]]
        trainer.train_loader.generator.manual_seed(2001)
        second = [_groups(w) for w in [next(ResumableStream(trainer.train_loader).iterate())]]
        assert first != second

    def test_the_same_seed_gives_the_same_order(self, trainer) -> None:
        """Determinism: the replay control depends on this."""
        trainer.train_loader.generator.manual_seed(2000)
        first = [_groups(next(ResumableStream(trainer.train_loader).iterate())) for _ in range(1)]
        trainer.train_loader.generator.manual_seed(2000)
        second = [_groups(next(ResumableStream(trainer.train_loader).iterate())) for _ in range(1)]
        assert first == second

    def test_the_state_records_the_epoch_boundary_not_the_moment(self, trainer) -> None:
        """Documents the invariant the first test enforces."""
        stream = ResumableStream(trainer.train_loader)
        iterator = stream.iterate()
        next(iterator)
        state = stream.state()
        assert isinstance(state, StreamState)
        assert state.batches_consumed_in_epoch == 1
        assert state.epoch_index == 0


class TestTheStateRoundTrips:
    """Save and reload must restore weights and optimiser moments exactly."""

    def test_weights_and_optimiser_survive_a_round_trip(self, trainer, tmp_path) -> None:
        """A branch loads this; anything lost here is silently lost from every branch."""
        stream = ResumableStream(trainer.train_loader)
        iterator = stream.iterate()
        for step in range(3):
            trainer.train_step(next(iterator), step)
        before_model = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        before_moments = {
            f"{i}:{name}": value.clone()
            for i, entry in trainer.optimizer.state_dict()["state"].items()
            for name, value in entry.items()
            if isinstance(value, torch.Tensor)
        }
        path = tmp_path / "state.pt"
        save_state(trainer=trainer, stream=stream, step=3, path=path, label="test")

        for step in range(3, 6):
            trainer.train_step(next(iterator), step)
        assert any(
            not torch.equal(before_model[k], v) for k, v in trainer.model.state_dict().items()
        ), "training did not move the weights, so the test proves nothing"

        restored = load_state(trainer=trainer, path=path)
        assert restored["step"] == 3
        assert [
            k for k, v in trainer.model.state_dict().items() if not torch.equal(before_model[k], v)
        ] == []
        after_moments = {
            f"{i}:{name}": value
            for i, entry in trainer.optimizer.state_dict()["state"].items()
            for name, value in entry.items()
            if isinstance(value, torch.Tensor)
        }
        assert [
            k for k in before_moments if not torch.equal(before_moments[k], after_moments[k])
        ] == []

    def test_the_optimiser_state_is_not_empty(self, trainer, tmp_path) -> None:
        """Saving a fresh optimiser would look fine and lose all the momentum."""
        stream = ResumableStream(trainer.train_loader)
        trainer.train_step(next(stream.iterate()), 0)
        path = tmp_path / "state.pt"
        marks = save_state(trainer=trainer, stream=stream, step=1, path=path, label="test")
        assert marks["optimizer_entries"] > 100
        assert marks["optimizer_moments"] is not None


class TestTheFingerprint:
    """Used to demonstrate branches started identically, so it must detect when they did not."""

    def test_identical_states_share_a_fingerprint(self, trainer, tmp_path) -> None:
        """Two reads of one saved state must agree."""
        stream = ResumableStream(trainer.train_loader)
        trainer.train_step(next(stream.iterate()), 0)
        path = tmp_path / "state.pt"
        first = save_state(trainer=trainer, stream=stream, step=1, path=path, label="a")
        second = load_state(trainer=trainer, path=path)["fingerprint"]
        assert first == second

    def test_a_changed_weight_changes_the_fingerprint(self, trainer, tmp_path) -> None:
        """A silent divergence between branches has to be detectable."""
        stream = ResumableStream(trainer.train_loader)
        trainer.train_step(next(stream.iterate()), 0)
        path = tmp_path / "state.pt"
        before = save_state(trainer=trainer, stream=stream, step=1, path=path, label="a")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = next(iter(payload["model"]))
        payload["model"][key] = payload["model"][key] + 1e-6
        assert fingerprint(payload) != before


class TestTheDeclaredDesign:
    """Fixed before execution, per the brief."""

    def test_eight_branches_with_distinct_seeds(self) -> None:
        """Eight, named branch_00 to branch_07, never extended after seeing results."""
        assert len(BRANCH_SEEDS) == 8
        assert sorted(BRANCH_SEEDS) == [f"branch_{i:02d}" for i in range(8)]
        assert len(set(BRANCH_SEEDS.values())) == 8

    def test_branch_seeds_do_not_collide_with_programme_seeds(self) -> None:
        """A branch seed must not be mistakable for one of the ten run seeds."""
        assert all(value >= 2000 for value in BRANCH_SEEDS.values())

    def test_the_branch_point_is_exactly_805(self) -> None:
        """The brief forbids 800 or 810."""
        assert BRANCH_POINT == 805
        assert SEED == 9

    def test_the_measurement_window_covers_the_transition(self) -> None:
        """Escape timing must be resolvable at 5-step spacing from the branch point."""
        assert WINDOW[0] == BRANCH_POINT
        assert WINDOW[1] >= 900
