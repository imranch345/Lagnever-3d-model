"""Step 17: the definitions that decide whether a precursor was found.

At 5-step resolution almost every series has *some* checkpoint that moves, so the verdict is
decided by the definitions rather than the measurements. Two of them are easy to get wrong in a
way that manufactures a finding, and both were got wrong in the first draft of the analysis:

* **graph usage is ``rotation_without_graph - rotation_intact``**, so validation rotation is one
  of its two terms and the relationally-required subset is that term on a subset. A rotation
  series crossing its own detection threshold before usage crosses the 1.5 degree classification
  threshold is one transition seen through two variables. Reported naively it looks like
  "geometry precedes escape", which would have been a precursor claim built on an identity.
* **the standing-offset check must use the declared baseline**, not every step before the window.
  Including the initial convergence transient makes almost any later value look like a
  pre-existing offset, which suppresses real candidates instead of inventing them -- the
  opposite failure, equally wrong.

Both are pinned here, along with escape detection and the transient-crossing rule.
"""

from __future__ import annotations

import pytest

from experiments.step15.seed_modes import GRAPH_USING_DEG
from experiments.step17.analysis import (
    BASELINE,
    DERIVED_FROM_GRAPH_USAGE,
    PERSISTENCE,
    PRECURSOR_CLASSES,
    SIGMA,
    WINDOW,
    _dense,
    _escape,
    _first_change,
)


class TestTheWindowAndGrid:
    """The window was fixed from Step 16; the grid must match what was declared."""

    def test_the_window_is_the_preregistered_one(self) -> None:
        """750-900, bracketing Step 16's 800 < escape <= 850."""
        assert WINDOW == (750, 900)

    def test_the_baseline_precedes_the_window_and_excludes_the_transient(self) -> None:
        """400-750: after convergence, before the window."""
        assert BASELINE == (400, 750)
        assert BASELINE[1] == WINDOW[0]

    def test_the_dense_selection_is_every_fifth_step(self) -> None:
        """31 checkpoints, 750 and 900 inclusive."""
        values = {step: 0.0 for step in range(700, 1000)}
        dense = _dense(values, WINDOW)
        assert sorted(dense) == list(range(750, 901, 5))
        assert len(dense) == 31

    def test_persistence_spans_fifteen_steps(self) -> None:
        """Three dense checkpoints, because single points are noisy at this spacing."""
        assert PERSISTENCE == 3
        assert SIGMA == 3.0


class TestEscapeDetection:
    """Step 15's threshold, unmodified, with transients separated from permanent escape."""

    def test_a_clean_crossing_is_reported_as_an_interval(self) -> None:
        """Last checkpoint below, first at or above."""
        usage = {750: 0.4, 800: 0.6, 830: 1.2, 835: 1.6, 840: 2.0, 900: 2.5}
        result = _escape(usage)
        assert result["escaped"]
        assert result["interval"] == "830 < escape <= 835"
        assert result["transient_crossings"] == []

    def test_a_crossing_that_falls_back_is_transient_not_escape(self) -> None:
        """The plan requires these be reported separately, not counted as escape."""
        usage = {750: 0.4, 800: 1.7, 805: 0.9, 830: 1.2, 835: 1.6, 900: 2.5}
        result = _escape(usage)
        assert result["transient_crossings"] == [800]
        assert result["permanent_crossing_step"] == 835

    def test_never_reaching_the_threshold_is_not_an_escape(self) -> None:
        """A non-escaper must not be credited with one."""
        assert not _escape({s: 0.3 for s in range(750, 905, 5)})["escaped"]

    def test_the_threshold_is_step_15s(self) -> None:
        """No new rule was introduced for the dense run."""
        assert GRAPH_USING_DEG == 1.5


class TestTheDefinitionalExclusions:
    """A rotation series cannot be a precursor to a quantity computed from it."""

    def test_rotation_and_subset_are_excluded_with_graph_usage(self) -> None:
        """Usage is bare minus intact; both rotation series are terms of it."""
        assert DERIVED_FROM_GRAPH_USAGE == {
            "graph_usage_deg",
            "validation_rotation_deg",
            "relational_subset_deg",
        }

    def test_only_upstream_classes_are_eligible(self) -> None:
        """Representation, gradient and objective: things upstream of the prediction."""
        assert PRECURSOR_CLASSES == ("representation", "gradient", "objective")
        assert "geometry" not in PRECURSOR_CLASSES
        assert "graph_usage" not in PRECURSOR_CLASSES


class TestTheChangeCriterion:
    """When a series counts as having changed, and when it does not."""

    @staticmethod
    def _flat(value: float) -> dict[int, float]:
        return {step: value for step in list(range(400, 751, 50)) + list(range(750, 901, 5))}

    def test_a_sustained_departure_is_detected_at_its_first_checkpoint(self) -> None:
        """Three consecutive dense checkpoints outside the band."""
        values = self._flat(1.0)
        values[400] = 0.99
        values[450] = 1.01
        for step in range(815, 901, 5):
            values[step] = 5.0
        result = _first_change(values, BASELINE, WINDOW)
        assert result["detected"]
        assert result["step"] == 815
        assert result["direction"] == "up"

    def test_a_two_checkpoint_excursion_is_not_sustained(self) -> None:
        """Persistence is three; two is not enough at 5-step spacing."""
        values = self._flat(1.0)
        values[400] = 0.99
        values[450] = 1.01
        values[820] = 5.0
        values[825] = 5.0
        assert not _first_change(values, BASELINE, WINDOW)["detected"]

    def test_a_perfectly_flat_series_never_changes(self) -> None:
        """A zero-variance baseline must not make everything a detection."""
        result = _first_change(self._flat(2.0), BASELINE, WINDOW)
        assert not result["detected"]

    def test_the_baseline_uses_only_its_declared_window(self) -> None:
        """A wild value before step 400 must not widen the band and hide a real change.

        This is the second bug the module had: using every step before the window pulled the
        initial convergence transient into the reference and suppressed candidates.
        """
        values = self._flat(1.0)
        values[0] = 60.0
        values[100] = 40.0
        values[400] = 0.99
        values[450] = 1.01
        for step in range(815, 901, 5):
            values[step] = 5.0
        result = _first_change(values, BASELINE, WINDOW)
        assert result["detected"], "an out-of-baseline transient changed the verdict"
        assert result["baseline_mean"] == pytest.approx(1.0, abs=0.02)
