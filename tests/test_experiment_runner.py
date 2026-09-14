"""The experiment runner: criteria arithmetic, smoke run and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.heart.experiment_001_structured_vs_appearance import EXPERIMENT_001
from experiments.heart.run_experiment import (
    DRIFT_METRIC,
    GEOMETRY_METRIC,
    PRIMARY_METRIC,
    RELATIONSHIP_METRIC,
    REPORTED_METRICS,
    ExperimentReport,
    evaluate_criteria,
    smoke_run,
)


def _aggregate(structured: dict[str, float], baseline: dict[str, float]) -> dict:
    def entry(value: float, spread: float = 0.0) -> dict[str, float]:
        return {
            "mean": value,
            "std": spread,
            "min": value - spread,
            "max": value + spread,
            "n": 3.0,
        }

    return {
        "A3": {metric: entry(value) for metric, value in structured.items()},
        "A0": {metric: entry(value) for metric, value in baseline.items()},
    }


def test_criteria_pass_when_the_thresholds_are_met() -> None:
    """A clear win on every axis passes every criterion."""
    aggregates = _aggregate(
        {
            PRIMARY_METRIC: 0.90,
            RELATIONSHIP_METRIC: 0.90,
            DRIFT_METRIC: 0.0,
            GEOMETRY_METRIC: 0.02,
        },
        {
            PRIMARY_METRIC: 0.40,
            RELATIONSHIP_METRIC: 0.60,
            DRIFT_METRIC: 0.001,
            GEOMETRY_METRIC: 0.03,
        },
    )
    verdicts = {outcome.hypothesis: outcome.verdict for outcome in evaluate_criteria(aggregates)}
    assert verdicts["H1a"] == "PASS"
    assert verdicts["H1b"] == "PASS"
    assert verdicts["H1c"] == "PASS"
    assert verdicts["guard"] == "PASS"
    assert verdicts["H1d"] == "NOT MEASURED"


def test_criteria_fail_on_a_small_gap() -> None:
    """A gap under the pre-registered threshold fails rather than being rounded up."""
    aggregates = _aggregate(
        {PRIMARY_METRIC: 0.50, RELATIONSHIP_METRIC: 0.70, DRIFT_METRIC: 0.0, GEOMETRY_METRIC: 0.02},
        {
            PRIMARY_METRIC: 0.45,
            RELATIONSHIP_METRIC: 0.65,
            DRIFT_METRIC: 0.001,
            GEOMETRY_METRIC: 0.02,
        },
    )
    verdicts = {outcome.hypothesis: outcome.verdict for outcome in evaluate_criteria(aggregates)}
    assert verdicts["H1a"] == "FAIL"
    assert verdicts["H1b"] == "FAIL"


def test_overlapping_seed_ranges_fail_the_first_criterion() -> None:
    """A large mean gap with overlapping ranges does not count as support."""
    aggregates = {
        "A3": {PRIMARY_METRIC: {"mean": 0.8, "std": 0.3, "min": 0.4, "max": 1.0, "n": 3.0}},
        "A0": {PRIMARY_METRIC: {"mean": 0.5, "std": 0.3, "min": 0.2, "max": 0.9, "n": 3.0}},
    }
    outcome = evaluate_criteria(aggregates)[0]
    assert outcome.verdict == "FAIL"
    assert "overlap" in outcome.detail


def test_geometry_guard_fails_a_large_regression() -> None:
    """Semantics bought with geometry does not count as a win."""
    aggregates = _aggregate(
        {PRIMARY_METRIC: 0.9, RELATIONSHIP_METRIC: 0.9, DRIFT_METRIC: 0.0, GEOMETRY_METRIC: 0.05},
        {PRIMARY_METRIC: 0.2, RELATIONSHIP_METRIC: 0.2, DRIFT_METRIC: 0.01, GEOMETRY_METRIC: 0.02},
    )
    guard = next(o for o in evaluate_criteria(aggregates) if o.hypothesis == "guard")
    assert guard.verdict == "FAIL"
    assert "regression" in guard.detail


def test_missing_metrics_report_not_measured() -> None:
    """An absent metric is never scored as a pass or a zero."""
    outcomes = evaluate_criteria({})
    assert all(outcome.verdict == "NOT MEASURED" for outcome in outcomes)


def test_criteria_match_the_pre_registration() -> None:
    """The thresholds evaluated here are the ones Step 5 registered."""
    registered = {criterion.hypothesis for criterion in EXPERIMENT_001.success_criteria}
    evaluated = {outcome.hypothesis for outcome in evaluate_criteria({})}
    assert registered <= evaluated


def test_reported_metrics_cover_every_required_axis() -> None:
    """Part control, relationships, drift, geometry and level of detail are all reported."""
    for metric in (
        "part_control_success",
        "relationship_accuracy_strict",
        "untouched_drift",
        "occupancy_chamfer",
        "scene_iou",
        "lod_part_agreement_mean",
    ):
        assert metric in REPORTED_METRICS


@pytest.mark.slow
def test_smoke_run_executes_the_whole_pipeline(tmp_path: Path) -> None:
    """Data, AWR, tensors, model, loss, backward, optimiser, checkpoint, evaluation."""
    result = smoke_run(tmp_path / "smoke", scenes=36, steps=2)
    assert set(result["arms"]) == {"A3", "A0"}
    for arm, payload in result["arms"].items():
        assert payload["steps"] == 2, arm
        assert payload["checkpoint_exists"], arm
        assert payload["parameters"] > 1_000_000, arm
        assert "scene_iou" in payload["results"], arm
    assert (tmp_path / "smoke" / "smoke_report.json").is_file()


def test_report_serialises(tmp_path: Path) -> None:
    """A report round-trips to JSON with its data label intact."""
    report = ExperimentReport(
        experiment_id="test",
        commit="abc",
        environment={"torch": "x"},
        dataset={"scenes": 1},
        training={"steps": 1},
    )
    path = report.save(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["data_label"].startswith("SYNTHETIC_RESEARCH_DATA")
