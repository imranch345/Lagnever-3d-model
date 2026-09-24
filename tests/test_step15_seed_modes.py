"""Step 15: the classification that decides whether "mode" is the right word.

Every interpretation since Step 11 has used the word "mode" on the strength of three seeds.
Step 15 measures ten, and the verdict comes from a rule fixed before the sweep. The rule is
small enough to get wrong quietly -- a band boundary off by a tenth, or a shape verdict that
ignores an empty outer band -- so each branch is exercised against a case whose answer is known
in advance.

The loader's assertions matter as much. Seeds 0 to 2 come from one runs directory and 3 to 9
from another; if the configuration drifted between them the survey would be comparing two
things at once and would still produce a tidy table.
"""

from __future__ import annotations

import pytest

from experiments.step15.seed_modes import (
    FROZEN_RUNS,
    GRAPH_USING_DEG,
    MAXIMUM_INTERMEDIATE,
    NON_USING_DEG,
    SEEDS,
    STEP15_RUNS,
    _runs_dir,
    classify,
)


class TestTheClassificationBands:
    """Boundaries are inclusive on the side the plan says they are."""

    def test_the_thresholds_are_preregistered(self) -> None:
        """1.5 and 0.8, with the gap left deliberately unlabelled."""
        assert GRAPH_USING_DEG == 1.5
        assert NON_USING_DEG == 0.8
        assert MAXIMUM_INTERMEDIATE == 2

    @pytest.mark.parametrize(
        ("usage", "band"),
        (
            (2.59, "graph_using"),
            (2.51, "graph_using"),
            (1.5, "graph_using"),
            (1.49, "intermediate"),
            (1.15, "intermediate"),
            (0.81, "intermediate"),
            (0.8, "non_using"),
            (0.38, "non_using"),
            (0.0, "non_using"),
            (-0.2, "non_using"),
        ),
    )
    def test_each_band_is_assigned_as_the_plan_states(self, usage: float, band: str) -> None:
        """Including the three observed values and Step 13's intermediate 1.15."""
        assert classify(usage) == band

    def test_the_step_13_depth_value_lands_in_the_middle(self) -> None:
        """1.15 was flagged in the plan as weak evidence against a clean bimodality."""
        assert classify(1.15) == "intermediate"


class TestTheShapeVerdict:
    """Whether ten seeds are two modes, one spread, or effectively one outcome."""

    @staticmethod
    def _shape(usages: list[float]) -> str:
        """The plan's section 4.1 rule, applied to a list of graph-usage values."""
        bands = {
            b: [u for u in usages if classify(u) == b]
            for b in ("graph_using", "intermediate", "non_using")
        }
        if not (bands["graph_using"] and bands["non_using"]):
            return "near-deterministic"
        if len(bands["intermediate"]) <= MAXIMUM_INTERMEDIATE:
            return "bimodal"
        return "continuum"

    def test_the_three_seed_picture_reads_bimodal(self) -> None:
        """What Steps 11 to 14 assumed, on the evidence they had."""
        assert self._shape([2.59, 0.38, 2.51]) == "bimodal"

    def test_a_populated_middle_is_a_continuum(self) -> None:
        """Three or more intermediates means the mode framing was wrong."""
        assert self._shape([2.6, 2.5, 1.4, 1.2, 1.0, 0.9, 0.4, 0.3, 2.4, 0.5]) == "continuum"

    def test_an_empty_outer_band_is_not_bimodal(self) -> None:
        """If no seed fails to use the graph, the three-seed picture was a small sample."""
        assert self._shape([2.6, 2.5, 2.4, 2.7, 2.3, 2.5, 2.6, 2.2, 2.4, 2.5]) == (
            "near-deterministic"
        )

    def test_two_intermediates_still_counts_as_bimodal(self) -> None:
        """The boundary case of the rule, in the direction that permits the verdict."""
        assert self._shape([2.6, 2.5, 2.4, 1.2, 1.1, 0.4, 0.3, 0.2, 2.5, 0.5]) == "bimodal"


class TestTheSeedSources:
    """Ten seeds drawn from two directories must still be one experiment."""

    def test_ten_seeds_are_surveyed(self) -> None:
        """Three cannot distinguish a two-mode distribution from a wide one."""
        assert SEEDS == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        assert len(SEEDS) == 10

    def test_the_first_three_come_from_the_frozen_runs(self) -> None:
        """Seeds 0-2 are the Step 10 Change 2 checkpoints, reused rather than retrained."""
        for seed in (0, 1, 2):
            assert _runs_dir(seed) == FROZEN_RUNS

    def test_the_new_seeds_come_from_step_15(self) -> None:
        """Seeds 3-9 are this step's, and must not be looked for in the frozen directory."""
        for seed in range(3, 10):
            assert _runs_dir(seed) == STEP15_RUNS
