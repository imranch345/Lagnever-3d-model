"""Experiment 11: the convergence criterion must be mechanical, not a judgement call."""

from __future__ import annotations

from experiments.step7.convergence import FAILURE_FLOOR, TOLERANCE, classify


def _history(values):
    return [{"scene_iou": value, "step": 300.0 * (index + 1)} for index, value in enumerate(values)]


def test_a_plateaued_run_is_converged() -> None:
    """Final value within tolerance of the best, and the best is not the first point."""
    result = classify(_history([0.60, 0.80, 0.81]))
    assert result["verdict"] == "converged"


def test_a_still_improving_run_is_not_converged() -> None:
    """An equal-budget comparison at a moving point shows which arm learns faster.

    That is not the same as which arm ends up better, so the verdict must say so.
    """
    result = classify(_history([0.50, 0.70, 0.85]))
    assert result["verdict"] == "not_converged"
    assert "improving" in result["reason"]


def test_a_run_below_the_floor_has_failed() -> None:
    """A baseline that never trained is not evidence that structure helps."""
    result = classify(_history([0.0, 0.0, 0.0]))
    assert result["verdict"] == "failed"
    assert str(FAILURE_FLOOR) in result["reason"]


def test_a_run_that_regressed_is_not_converged() -> None:
    """Ending well below the best is not a plateau."""
    result = classify(_history([0.40, 0.85, 0.60]))
    assert result["verdict"] == "not_converged"
    assert f"{TOLERANCE}" in result["reason"]


def test_a_run_whose_best_was_its_first_measurement_is_not_converged() -> None:
    """If training never improved on its starting point, nothing converged."""
    result = classify(_history([0.82, 0.81, 0.815]))
    assert result["verdict"] == "not_converged"
    assert "first measurement" in result["reason"]


def test_an_empty_history_fails_rather_than_passing_quietly() -> None:
    """A run with no validation record must not be reported as fine."""
    result = classify([])
    assert result["verdict"] == "failed"


def test_the_trajectory_is_reported() -> None:
    """The verdict must be checkable against the numbers behind it."""
    result = classify(_history([0.10, 0.50, 0.80, 0.81]))
    assert result["trajectory"] == [0.1, 0.5, 0.8, 0.81]
    assert result["final"] == 0.81
    assert result["best"] == 0.81
