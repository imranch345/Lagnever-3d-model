"""Step 22: seed 5 from initialisation, the crossed swap, and the preregistered verdicts.

What must hold before any run means anything (plan section 8):

* **Steps 20 and 21 are unchanged** by the new seed parameter: its default is seed 9 everywhere.
* **Seed 5's capture starts from seed 5's weights**, not seed 9's.
* **A swap swaps only the batch order.** Loader seed *s* from step 0 gives run *s*'s first
  batches, whatever the weights; the weights stay those of the state continued.
* **The decision rules** (none / split / all, Stage 2, the identity controls) are pinned so
  they cannot be edited after a run.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step20 import branching
from experiments.step20.state import ResumableStream, load_state
from experiments.step21 import basin_entry
from experiments.step21.basin_entry import capture_states, state_path
from experiments.step22 import mirror
from training.step10 import Step10Trainer, default_config, with_variant

CORPUS = Path("datasets/processed/step10_rotated")


@pytest.fixture(scope="module")
def make_trainer():
    """Fresh short-budget trainers at the frozen configuration, for any seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train = load_step8_split(CORPUS, "train", limit=96)
    validation = load_step8_split(CORPUS, "validation", limit=16)

    def build(seed: int) -> Step10Trainer:
        config = replace(
            with_variant(
                default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
            ),
            rotation_objective="chordal",
            rotation_loss_weight=3.0,
            device="cpu",
            steps=40,
        )
        return Step10Trainer(config, ontology, domain, train, validation)

    return build


def _weights(trainer: Any) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}


def _differing(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    return [k for k in a if not torch.equal(a[k], b[k])]


def _first_batch(whole: Any) -> float:
    return round(float(whole.batch.entity_frames.sum()), 6)


class TestStepsTwentyAndTwentyOneAreUnchanged:
    """The seed parameter defaults to seed 9, so every earlier call means what it meant."""

    def test_the_defaults_are_seed_nine(self) -> None:
        """_build, continue_run and capture_states all default to Step 20's seed."""
        assert branching.SEED == 9
        assert inspect.signature(branching._build).parameters["seed"].default == 9
        assert inspect.signature(branching.continue_run).parameters["seed"].default == 9
        assert inspect.signature(capture_states).parameters["seed"].default == 9

    def test_state_file_names_are_unchanged(self, tmp_path: Path) -> None:
        """Step 21's states are found where Step 21 saved them."""
        assert state_path(tmp_path, 400).name == "state_seed9_step400.pt"
        assert state_path(tmp_path, 0, 5).name == "state_seed5_step0.pt"
        assert basin_entry.SEED == 9


class TestSeedFiveAndTheSwap:
    """The capture and the crossed swap do what the plan says, on short trainers."""

    def test_seed_five_is_captured_from_its_own_initialisation(
        self, make_trainer, tmp_path
    ) -> None:
        """Step 0 holds seed 5's initial weights, which differ from seed 9's."""
        capture_states(make_trainer(5), points=(0,), out_dir=tmp_path, seed=5)
        payload = torch.load(state_path(tmp_path, 0, 5), map_location="cpu", weights_only=False)
        assert _differing(_weights(make_trainer(5)), payload["model"]) == []
        assert _differing(_weights(make_trainer(9)), payload["model"]) != []
        assert payload["optimizer"]["state"] == {}

    @pytest.mark.parametrize(("weights", "order"), [(5, 9), (9, 5), (5, 5), (9, 9)])
    def test_a_loader_seed_from_step_zero_is_that_runs_batch_order(
        self, make_trainer, tmp_path, weights: int, order: int
    ) -> None:
        """Weights from one seed, first batch exactly as the other seed's original run."""
        capture_states(make_trainer(weights), points=(0,), out_dir=tmp_path, seed=weights)
        swapped = make_trainer(weights)
        load_state(trainer=swapped, path=state_path(tmp_path, 0, weights))
        swapped.train_loader.generator.manual_seed(order)
        first = next(ResumableStream(swapped.train_loader).iterate())

        original = next(ResumableStream(make_trainer(order).train_loader).iterate())
        assert _first_batch(first) == _first_batch(original)
        assert _differing(_weights(swapped), _weights(make_trainer(weights))) == []


class TestTheDeclaredDesign:
    """Fixed by the plan before any run; never edited after one."""

    def test_arm_one_and_two_at_step_zero(self) -> None:
        """Reference, 8 branches, a replay of branch_00, and the four swap runs."""
        runs = mirror.plan_runs((0,))
        names = [mirror._file_name(r) for r in runs]
        assert names == [
            "reference",
            *[f"branch_{i:02d}" for i in range(8)],
            "branch_00_replay",
            "cross_5w_9o",
            "cross_9w_5o",
            "ident_5w_5o",
            "ident_9w_9o",
        ]
        assert all(r["point"] == 0 for r in runs)
        assert {r["name"]: (r["weights"], r["loader"]) for r in runs[-4:]} == {
            "cross_5w_9o": (5, 9),
            "cross_9w_5o": (9, 5),
            "ident_5w_5o": (5, 5),
            "ident_9w_9o": (9, 9),
        }

    def test_branch_orderings_are_step_twenty_ones(self) -> None:
        """branch_k is the same batch order as from seed 9's step 0."""
        branches = [r for r in mirror.plan_runs((0,)) if r["name"].startswith("branch")]
        assert {r["name"]: r["loader"] for r in branches} == branching.BRANCH_SEEDS
        assert all(r["weights"] == 5 for r in branches)

    def test_stage_two_is_four_hundred_and_eight_hundred_without_swaps(self) -> None:
        """Reference and 8 branches at each; no replay; Arm 2 only at step 0."""
        runs = mirror.plan_runs(mirror.STAGE_TWO_POINTS, arm_two=False)
        assert len(runs) == 18 and {r["point"] for r in runs} == {400, 800}
        assert mirror.CAPTURE_POINTS == (0, 400, 800)

    def test_seed_nines_state_is_step_twenty_ones(self, tmp_path: Path) -> None:
        """Seed 9's weights come from Step 21's step-0 state, not a new capture."""
        run = next(r for r in mirror.plan_runs((0,)) if r["name"] == "cross_9w_5o")
        assert mirror._state_for(run, tmp_path) == state_path(mirror.STEP21_OUT, 0, 9)
        own = next(r for r in mirror.plan_runs((0,)) if r["name"] == "reference")
        assert mirror._state_for(own, tmp_path) == state_path(tmp_path, 0, 5)


class TestTheVerdicts:
    """Section 4 and section 6."""

    @pytest.mark.parametrize(
        ("escaped", "expected"), [(0, "none"), (1, "split"), (7, "split"), (8, "all")]
    )
    def test_none_split_all(self, escaped: int, expected: str) -> None:
        """One escape is enough for split; none means zero."""
        assert mirror.verdict(escaped, 8) == expected

    def test_impossible_counts_are_refused(self) -> None:
        """Counts outside 0..total are a bug, not a verdict."""
        for escaped, total in ((-1, 8), (9, 8), (0, 0)):
            with pytest.raises(ValueError):
                mirror.verdict(escaped, total)

    @pytest.mark.parametrize(
        ("arm_one", "points"), [("none", ()), ("split", (400, 800)), ("all", ())]
    )
    def test_stage_two_runs_only_on_a_split(self, arm_one: str, points: tuple) -> None:
        """A' and C' stop at Arm 1; B' brackets with 400 and 800."""
        assert mirror.stage_two_points(arm_one) == points


def _report(name: str, *, escaped: bool, usage: float, rows: list[dict[str, Any]]) -> dict:
    return {
        "branch": name,
        "escape": {"escaped": escaped, "permanent_crossing_step": 900 if escaped else None},
        "final": {"graph_usage_deg": usage, "validation_rotation_deg": 23.5},
        "starting_fingerprint": {"sum": 1.0},
        "first_batches_after_branch": [[sum(map(ord, name))]],
        "trajectory": rows,
    }


class TestIntegrity:
    """Any failed check makes the whole step invalid (section 6)."""

    rows = [{"step": 0, "graph_usage_deg": 0.0, "seconds": 1.0}]

    def _valid(self) -> dict[str, Any]:
        branches = [
            _report(f"branch_{i:02d}", escaped=False, usage=0.1, rows=self.rows) for i in range(8)
        ]
        reference = {**_report("reference", escaped=False, usage=0.1, rows=self.rows)}
        reference["reproduces_frozen_seed5"] = {"reproduces": True}
        replay = {**branches[0], "trajectory": [{**self.rows[0], "seconds": 9.0}]}
        return {
            "capture_report": {
                "reproduces_frozen_seed5": {"reproduces": True},
                "step0_differs_from_seed9": True,
            },
            "reference": reference,
            "branches": branches,
            "replay": replay,
            "swaps": {
                "ident_5w_5o": {"trajectory": self.rows},
                "ident_9w_9o": {"trajectory": [{"step": 0, "graph_usage_deg": 2.0}]},
            },
            "seed9_zero": {"trajectory": [{"step": 0, "graph_usage_deg": 2.0}]},
        }

    def test_a_valid_run_passes(self) -> None:
        """Every check true, and wall-clock time ignored in comparisons."""
        checks = mirror.integrity(**self._valid())
        assert checks["valid"], checks

    def test_an_identity_mismatch_invalidates(self) -> None:
        """A swap that is not exactly its reference cannot be read as a swap."""
        broken = self._valid()
        broken["swaps"]["ident_5w_5o"] = {"trajectory": [{"step": 0, "graph_usage_deg": 1e-9}]}
        checks = mirror.integrity(**broken)
        assert not checks["ident_5w_5o_equals_reference"] and not checks["valid"]

    def test_a_reference_that_does_not_reproduce_invalidates(self) -> None:
        """Section 7, check 2."""
        broken = self._valid()
        broken["reference"]["reproduces_frozen_seed5"] = {"reproduces": False}
        assert not mirror.integrity(**broken)["valid"]

    def test_arm_one_counts_and_classes(self) -> None:
        """Eight stuck branches: none, all stuck, no escape steps."""
        branches = [
            _report(f"branch_{i:02d}", escaped=False, usage=0.1, rows=self.rows) for i in range(8)
        ]
        outcome = mirror.arm_one(branches)
        assert outcome["verdict"] == "none" and outcome["escape_steps"] is None
        assert set(outcome["classes"].values()) == {"stuck"}


def test_every_declared_run_is_accepted_and_nothing_else() -> None:
    """The first launch refused all step-0 runs: point 800 shadowed point 0. Never again."""
    runs = mirror.plan_runs((0, *mirror.STAGE_TWO_POINTS))
    assert len(runs) == 14 + 18
    assert all(mirror.declared(r) for r in runs)
    for name, point in (("branch_05", 0), ("branch_05", 400), ("reference", 800)):
        match = next(r for r in runs if r["name"] == name and r["point"] == point)
        assert mirror.declared(match)
    wrong_point = {**dict(runs[0]), "point": 600}
    wrong_weights = {**dict(runs[1]), "weights": 9}
    assert not mirror.declared(wrong_point) and not mirror.declared(wrong_weights)
    swap_later = {"name": "cross_5w_9o", "point": 400, "weights": 5, "loader": 9, "replay": False}
    assert not mirror.declared(swap_later)


def test_the_cli_round_trips_a_declared_run(monkeypatch) -> None:
    """What the runner sends on the command line is exactly what the check accepts."""
    for run in mirror.plan_runs((0, *mirror.STAGE_TWO_POINTS)):
        assert mirror.declared(mirror.Run(json.loads(json.dumps(dict(run)))))


def test_the_cli_refuses_a_run_the_plan_does_not_declare(monkeypatch, tmp_path) -> None:
    """'continue' only runs what plan_runs declares, exactly."""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    bogus = {"name": "branch_00", "point": 0, "weights": 9, "loader": 2000, "replay": False}
    with pytest.raises(SystemExit, match="not a run the plan declares"):
        mirror.main(["continue", "--run", json.dumps(bogus), "--out", str(tmp_path)])
