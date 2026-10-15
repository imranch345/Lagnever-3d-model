"""Step 21: the multi-point capture and the preregistered verdicts.

Continuation itself is Step 20's ``continue_run``, already validated by its reference control,
so it is reused rather than retested. What is new, and has to be exact before any branch means
anything, is:

* **the capture.** One run saves states at many steps. A state saved mid-run must continue
  bit-identically to the run that never stopped, or every branch point but the last is a
  different experiment from the one the plan names.
* **step 0.** A branch there must keep seed 9's initial weights and change only the order --
  that is the whole reason the plan includes it.
* **the decision rules.** Verdicts, the stuck / in-transition split and the Stage 2 table were
  fixed in the plan before any run; these tests pin them so they cannot be edited after one.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step15.seed_modes import GRAPH_USING_DEG
from experiments.step20.branching import BRANCH_POINT as STEP20_BRANCH_POINT
from experiments.step20.branching import BRANCH_SEEDS as STEP20_BRANCH_SEEDS
from experiments.step20.branching import SEED
from experiments.step20.state import ResumableStream, fingerprint, load_state
from experiments.step21.basin_entry import (
    CAPTURE_POINTS,
    MAX_JOBS,
    STAGE_ONE,
    STAGE_TWO,
    STUCK_BELOW_DEG,
    THREADS,
    _jobs,
    capture_states,
    classify_branch,
    next_points,
    reproduction,
    stage_two_gate,
    state_path,
    summarise_point,
    verdict,
    verify_freezes,
)
from training.step10 import Step10Trainer, default_config, with_variant

CORPUS = Path("datasets/processed/step10_rotated")


@pytest.fixture(scope="module")
def make_trainer():
    """Fresh short-budget trainers at the frozen configuration, built on demand."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train = load_step8_split(CORPUS, "train", limit=96)
    validation = load_step8_split(CORPUS, "validation", limit=16)

    def build(seed: int = SEED) -> Step10Trainer:
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
    """A batch's identity, as its entity-frame contents, which differ per scene group."""
    return round(float(whole.batch.entity_frames.sum()), 6)


class TestTheCaptureIsExact:
    """Every branch point but the last comes from a state saved mid-run."""

    def test_a_mid_capture_state_continues_like_the_run_that_never_stopped(
        self, make_trainer, tmp_path
    ) -> None:
        """Save at 4 while running on to 9; resuming from 4 must land on the same weights."""
        uninterrupted = make_trainer()
        capture_states(uninterrupted, points=(0, 4, 9), out_dir=tmp_path)
        expected = _weights(uninterrupted)

        resumed = make_trainer()
        restored = load_state(trainer=resumed, path=state_path(tmp_path, 4))
        assert restored["step"] == 4
        iterator = ResumableStream(resumed.train_loader).restore(restored["stream"])
        for step in range(4, 9):
            resumed.train_step(next(iterator), step)

        assert _differing(expected, _weights(resumed)) == []
        saved_at_nine = torch.load(state_path(tmp_path, 9), map_location="cpu", weights_only=False)
        assert fingerprint(saved_at_nine) == load_state(
            trainer=resumed, path=state_path(tmp_path, 9)
        )["fingerprint"]

    def test_a_state_at_s_has_seen_exactly_s_training_steps(self, make_trainer, tmp_path) -> None:
        """Step 20's 805 state had seen 805 steps; every capture point must mean the same."""
        trainer = make_trainer()
        marks = capture_states(trainer, points=(3, 7), out_dir=tmp_path)
        for point in (3, 7):
            assert marks[point]["step"] == point
            payload = torch.load(
                state_path(tmp_path, point), map_location="cpu", weights_only=False
            )
            steps = {float(entry["step"]) for entry in payload["optimizer"]["state"].values()}
            assert steps == {float(point)}

    def test_the_capture_saves_only_the_requested_points(self, make_trainer, tmp_path) -> None:
        """No state is written that the plan did not name."""
        trainer = make_trainer()
        capture_states(trainer, points=(2, 5), out_dir=tmp_path)
        assert sorted(p.name for p in tmp_path.glob("*.pt")) == [
            state_path(tmp_path, 2).name,
            state_path(tmp_path, 5).name,
        ]

    def test_negative_points_are_refused(self, make_trainer, tmp_path) -> None:
        """A typo in the point list must fail, not silently capture nothing."""
        with pytest.raises(ValueError):
            capture_states(make_trainer(), points=(-1, 4), out_dir=tmp_path)


def _resume(make_trainer, path: Path, stop: int, *, scramble_rng: bool = False) -> Any:
    """Load a saved state and train on to ``stop`` with the original stream restored."""
    trainer = make_trainer()
    restored = load_state(trainer=trainer, path=path)
    if scramble_rng:
        torch.manual_seed(12345)
        random.seed(12345)
        np.random.seed(12345)
    iterator = ResumableStream(trainer.train_loader).restore(restored["stream"])
    for step in range(restored["step"], stop):
        trainer.train_step(next(iterator), step)
    return trainer


class TestGlobalRngInsideTheTeacherForcingRamp:
    """Amendment 1. At 40 steps the ramp is steps 6-23: P0 ends at 0.15, P2 starts at 0.60.

    The model draws its teacher-forcing mask with ``torch.rand`` while the ratio is strictly
    between 0 and 1, so a state saved inside the ramp continues exactly only with its RNG. The
    first Stage 1 failed on this, and the original capture test above did not catch it because
    its resume point, step 4, is in P0.
    """

    def test_a_state_inside_the_ramp_continues_exactly(self, make_trainer, tmp_path) -> None:
        """load_state restores global RNG, so a mid-ramp reference is a true continuation."""
        uninterrupted = make_trainer()
        capture_states(uninterrupted, points=(10, 14), out_dir=tmp_path)
        resumed = _resume(make_trainer, state_path(tmp_path, 10), 14)
        assert _differing(_weights(uninterrupted), _weights(resumed)) == []

    def test_the_ramp_does_consume_global_rng(self, make_trainer, tmp_path) -> None:
        """Without the saved RNG the same state diverges -- the failure Stage 1 hit."""
        uninterrupted = make_trainer()
        capture_states(uninterrupted, points=(10, 14), out_dir=tmp_path)
        scrambled = _resume(make_trainer, state_path(tmp_path, 10), 14, scramble_rng=True)
        assert _differing(_weights(uninterrupted), _weights(scrambled)) != []

    def test_after_the_ramp_global_rng_is_not_consumed(self, make_trainer, tmp_path) -> None:
        """In P2 the RNG cannot matter: why Step 20's 805 result is unaffected by the fix."""
        uninterrupted = make_trainer()
        capture_states(uninterrupted, points=(26, 30), out_dir=tmp_path)
        scrambled = _resume(make_trainer, state_path(tmp_path, 26), 30, scramble_rng=True)
        assert _differing(_weights(uninterrupted), _weights(scrambled)) == []


class TestStepZeroSeparatesInitialisationFromOrder:
    """The reason step 0 is in the plan at all (section 2)."""

    def test_the_state_at_step_zero_is_seed_9s_initialisation(
        self, make_trainer, tmp_path
    ) -> None:
        """Nothing has trained yet: the weights are the init and the optimiser is empty."""
        trainer = make_trainer()
        capture_states(trainer, points=(0,), out_dir=tmp_path)
        payload = torch.load(state_path(tmp_path, 0), map_location="cpu", weights_only=False)
        fresh = make_trainer()
        assert _differing(_weights(fresh), payload["model"]) == []
        assert payload["optimizer"]["state"] == {}

    def test_a_branch_at_step_zero_keeps_the_weights_and_changes_the_order(
        self, make_trainer, tmp_path
    ) -> None:
        """Same initial weights, different first batch: initialisation held, order varied."""
        capture_states(make_trainer(), points=(0,), out_dir=tmp_path)

        reference = make_trainer()
        restored = load_state(trainer=reference, path=state_path(tmp_path, 0))
        original = next(ResumableStream(reference.train_loader).restore(restored["stream"]))

        branch = make_trainer()
        load_state(trainer=branch, path=state_path(tmp_path, 0))
        branch.train_loader.generator.manual_seed(STEP20_BRANCH_SEEDS["branch_00"])
        changed = next(ResumableStream(branch.train_loader).iterate())

        assert _differing(_weights(reference), _weights(branch)) == []
        assert _first_batch(original) != _first_batch(changed)

    def test_the_run_seed_does_change_the_initialisation(self, make_trainer) -> None:
        """Section 2's measured fact: seeds 9 and 5 start from different weights."""
        assert _differing(_weights(make_trainer(9)), _weights(make_trainer(5))) != []


class TestTheDeclaredDesign:
    """Fixed by the plan before any run; never edited after one."""

    def test_capture_points_are_the_plans(self) -> None:
        """Nine states: every possible branch point, plus 805 for the Step 20 cross-check."""
        assert CAPTURE_POINTS == (0, 200, 400, 500, 600, 650, 700, 750, 805)
        assert STEP20_BRANCH_POINT == 805
        assert SEED == 9

    def test_stage_one_is_the_three_points_step20_named(self) -> None:
        """400, 600 and 700, from the Step 20 report's section 14."""
        assert STAGE_ONE == (400, 600, 700)

    def test_stage_two_only_uses_captured_points(self) -> None:
        """A Stage 2 point without a captured state could not be run."""
        for points in STAGE_TWO.values():
            assert set(points) <= set(CAPTURE_POINTS) - {STEP20_BRANCH_POINT}

    def test_branch_seeds_are_step20s(self) -> None:
        """branch_k means the same loader seed here as it did at 805."""
        assert sorted(STEP20_BRANCH_SEEDS) == [f"branch_{i:02d}" for i in range(8)]
        assert STEP20_BRANCH_SEEDS["branch_00"] == 2000

    def test_protocol_constants(self) -> None:
        """Thread count, concurrency and both usage thresholds."""
        assert THREADS == 4
        assert MAX_JOBS == 3
        assert STUCK_BELOW_DEG == 0.5
        assert GRAPH_USING_DEG == 1.5

    def test_each_point_gets_a_reference_and_eight_branches(self) -> None:
        """And exactly one replay, at the point it names."""
        jobs = _jobs([400, 600], replay_point=400)
        assert len(jobs) == 2 * 9 + 1
        assert jobs[-1] == (400, "branch_00", True)
        assert sum(1 for p, name, replay in jobs if p == 600 and name == "reference") == 1
        with pytest.raises(ValueError):
            _jobs([400], replay_point=600)


class TestTheVerdicts:
    """Section 4 and section 6 of the plan."""

    @pytest.mark.parametrize(
        ("escaped", "expected"),
        [(8, "unanimous"), (7, "split"), (1, "split"), (0, "none")],
    )
    def test_one_non_escape_makes_a_point_split(self, escaped: int, expected: str) -> None:
        """Unanimous means 8/8 and nothing less."""
        assert verdict(escaped, 8) == expected

    def test_impossible_counts_are_refused(self) -> None:
        """A bookkeeping error must not become a verdict."""
        for escaped, total in ((9, 8), (-1, 8), (0, 0)):
            with pytest.raises(ValueError):
                verdict(escaped, total)

    @pytest.mark.parametrize(
        ("escaped", "usage", "expected"),
        [
            (True, 2.55, "escaped"),
            (False, 0.30, "stuck"),
            (False, 0.49, "stuck"),
            (False, 0.50, "in_transition"),
            (False, 1.30, "in_transition"),
        ],
    )
    def test_non_escapes_are_split_at_half_a_degree(
        self, escaped: bool, usage: float, expected: str
    ) -> None:
        """Seed 3 at 1.30 is in transition; Step 15's non-using seeds at 0.09-0.38 are stuck."""
        report = {"escape": {"escaped": escaped}, "final": {"graph_usage_deg": usage}}
        assert classify_branch(report) == expected

    def test_a_non_escape_above_the_threshold_is_refused(self) -> None:
        """Contradicts the permanent-crossing criterion; better to stop than mislabel."""
        with pytest.raises(ValueError):
            classify_branch({"escape": {"escaped": False}, "final": {"graph_usage_deg": 1.6}})

    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            (("unanimous", "unanimous", "unanimous"), (200, 0)),
            (("split", "unanimous", "unanimous"), (500,)),
            (("none", "unanimous", "unanimous"), (500,)),
            (("split", "split", "unanimous"), (650,)),
            (("split", "split", "split"), (750,)),
            (("none", "split", "none"), (750,)),
            (("unanimous", "split", "unanimous"), ()),
            (("unanimous", "unanimous", "split"), ()),
            (("split", "unanimous", "split"), ()),
        ],
    )
    def test_stage_two_follows_the_plans_table(
        self, pattern: tuple[str, str, str], expected: tuple[int, ...]
    ) -> None:
        """Monotone patterns narrow the bracket; any other runs no Stage 2."""
        assert next_points(dict(zip(STAGE_ONE, pattern, strict=True))) == expected

    def test_stage_two_needs_all_of_stage_one(self) -> None:
        """Choosing from a partial Stage 1 would be choosing after seeing some results."""
        with pytest.raises(ValueError):
            next_points({400: "unanimous", 600: "unanimous"})


def _report(name: str, *, escaped: bool, usage: float, start: float = 1.0) -> dict[str, Any]:
    rows = [{"step": 1200, "graph_usage_deg": usage, "seconds": 7.0}]
    return {
        "branch": name,
        "escape": {"escaped": escaped, "permanent_crossing_step": 830 if escaped else None},
        "final": {"graph_usage_deg": usage, "validation_rotation_deg": 21.0},
        "starting_fingerprint": {"sum": start},
        "first_batches_after_branch": [[int(name[-2:])]],
        "trajectory": rows,
    }


class TestThePointSummary:
    """What the report's per-point table is built from."""

    def test_counts_classes_and_integrity_fields(self) -> None:
        """Seven escapes and one stuck branch is a split, with every branch accounted for."""
        branches = [_report(f"branch_{i:02d}", escaped=i != 3, usage=2.5) for i in range(8)]
        branches[3] = _report("branch_03", escaped=False, usage=0.2)
        summary = summarise_point(branches)
        assert summary["escaped"] == 7
        assert summary["verdict"] == "split"
        assert summary["classes"]["branch_03"] == "stuck"
        assert summary["distinct_starting_fingerprints"] == 1
        assert summary["escape_steps"]["mean"] == 830.0

    def test_a_divergent_start_is_visible(self) -> None:
        """Branches that did not start from one state must show up as more than one start."""
        branches = [_report(f"branch_{i:02d}", escaped=True, usage=2.5) for i in range(8)]
        branches[5] = _report("branch_05", escaped=True, usage=2.5, start=2.0)
        assert summarise_point(branches)["distinct_starting_fingerprints"] == 2

    def test_the_replay_comparison_ignores_wall_clock_time(self) -> None:
        """Two identical runs never take the same number of seconds."""
        branches = [_report(f"branch_{i:02d}", escaped=True, usage=2.5) for i in range(8)]
        replay = _report("branch_00", escaped=True, usage=2.5)
        replay["trajectory"] = [{**replay["trajectory"][0], "seconds": 99.0}]
        assert summarise_point(branches, replay=replay)["replay_identical"] is True
        replay["trajectory"] = [{**replay["trajectory"][0], "graph_usage_deg": 2.4}]
        assert summarise_point(branches, replay=replay)["replay_identical"] is False


def _valid_summary() -> dict[str, Any]:
    points = {
        str(p): {
            "verdict": "unanimous",
            "reference_reproduces": True,
            "distinct_starting_fingerprints": 1,
            "distinct_first_batches": 8,
        }
        for p in STAGE_ONE
    }
    points[str(STAGE_ONE[0])]["replay_identical"] = True
    return {"points": points, "freezes": {"a.sha256": {"ok": 3, "bad": 0}}, "stage_two": [200, 0]}


class TestTheStageTwoGate:
    """Stage 2 is chosen from Stage 1 only when Stage 1 may be read at all."""

    def test_a_valid_stage_one_passes(self) -> None:
        """Every check satisfied: nothing blocks."""
        assert stage_two_gate(_valid_summary()) == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("reference_reproduces", False),
            ("distinct_starting_fingerprints", 2),
            ("distinct_first_batches", 7),
        ],
    )
    def test_any_failed_integrity_check_blocks(self, field: str, value: Any) -> None:
        """A failed reference is what invalidated the first Stage 1."""
        summary = _valid_summary()
        summary["points"]["600"][field] = value
        assert any(problem.startswith("600:") for problem in stage_two_gate(summary))

    def test_an_incomplete_stage_one_blocks(self) -> None:
        """Choosing from a partial Stage 1 would be choosing after seeing part of it."""
        summary = _valid_summary()
        summary["points"]["700"] = {"incomplete": ["branch_00"]}
        assert "700: incomplete" in stage_two_gate(summary)

    def test_a_missing_replay_blocks(self) -> None:
        """Determinism has to be shown, not assumed."""
        summary = _valid_summary()
        del summary["points"]["400"]["replay_identical"]
        assert stage_two_gate(summary) != []

    def test_a_changed_freeze_manifest_blocks(self) -> None:
        """A prior step's artifacts changing is a stop condition."""
        summary = _valid_summary()
        summary["freezes"]["a.sha256"] = {"ok": 2, "bad": 1}
        assert stage_two_gate(summary) != []


class TestReproductionAndFreezes:
    """The two checks that decide whether anything is run or read at all."""

    def _frozen(self, tmp_path: Path) -> Path:
        rows = [
            {"step": s, "graph_usage_deg": 0.1 * s, "validation_rotation_deg": 20.0 + s}
            for s in (0, 50, 100)
        ]
        path = tmp_path / "frozen.json"
        path.write_text(json.dumps({"trajectory": rows}), encoding="utf-8")
        return path

    def test_exact_agreement_reproduces(self, tmp_path) -> None:
        """Every shared checkpoint equal, to the last bit."""
        path = self._frozen(tmp_path)
        rows = json.loads(path.read_text(encoding="utf-8"))["trajectory"]
        check = reproduction(rows, path)
        assert check["reproduces"] is True
        assert check["shared_checkpoints"] == 3

    def test_a_tiny_difference_does_not_reproduce(self, tmp_path) -> None:
        """Step 17's thread-count divergence first appeared as a difference this small."""
        path = self._frozen(tmp_path)
        rows = json.loads(path.read_text(encoding="utf-8"))["trajectory"]
        rows[1]["validation_rotation_deg"] += 1e-9
        check = reproduction(rows, path)
        assert check["reproduces"] is False
        assert [m["step"] for m in check["mismatches"]] == [50]

    def test_a_continuation_is_compared_from_its_branch_point(self, tmp_path) -> None:
        """A reference from step 50 has nothing to say about step 0."""
        path = self._frozen(tmp_path)
        rows = json.loads(path.read_text(encoding="utf-8"))["trajectory"]
        rows[0]["graph_usage_deg"] = 99.0
        assert reproduction(rows, path, start=50)["reproduces"] is True

    def test_no_shared_checkpoint_is_not_a_reproduction(self, tmp_path) -> None:
        """Vacuous agreement must not pass."""
        assert reproduction([], self._frozen(tmp_path))["reproduces"] is False

    def test_freeze_manifests_detect_a_changed_file(self, tmp_path, monkeypatch) -> None:
        """Manifest paths are relative to the repository root, as in experiments/runs."""
        monkeypatch.chdir(tmp_path)
        artifact = tmp_path / "runs" / "stepX" / "result.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}", encoding="utf-8")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        manifest = artifact.parent / "STEPX_FROZEN.sha256"
        manifest.write_text(f"{digest}  runs/stepX/result.json\n", encoding="utf-8")
        key = str(manifest.relative_to(tmp_path))
        assert verify_freezes(Path("runs")) == {key: {"ok": 1, "bad": 0}}
        artifact.write_text('{"changed": true}', encoding="utf-8")
        assert verify_freezes(Path("runs"))[key] == {"ok": 0, "bad": 1}
