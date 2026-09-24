"""Step 10 analysis: the paired comparison and the controls read the runs correctly.

These run on synthetic run records, so the sign convention and the decision rule are
pinned independently of any trained model — a flipped subtraction here would turn every
conclusion in the report upside down without changing a single number in the runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.step10.run_placement import (
    T_CRITICAL_DF2,
    _control_identity,
    paired_comparison,
)


def _run(
    arm: str, cell: str, seed: int, translation: float, depths: dict[str, float]
) -> dict[str, Any]:
    return {
        "arm": arm,
        "cell": cell,
        "seed": seed,
        "splits": {"test_seen": {"inferred": {"translation_error": translation}}},
        "depth": {
            "splits": {
                "test_seen": {
                    "depths": {d: {"position_error": value} for d, value in depths.items()}
                }
            }
        },
    }


def _matrix(control: list[float], treatment: list[float]) -> list[dict[str, Any]]:
    runs = []
    for seed, (base, treated) in enumerate(zip(control, treatment, strict=True)):
        runs.append(_run("A3", "T0_global", seed, base, {"0": 0.10, "1": base}))
        runs.append(_run("A3", "T1_spatial", seed, treated, {"0": 0.10, "1": treated}))
    return runs


class TestThePairedComparison:
    """Treatment minus control, per seed, with the pre-registered rule."""

    def test_negative_means_the_treatment_placed_better(self) -> None:
        """The sign convention every table and conclusion depends on."""
        result = paired_comparison(
            _matrix([0.160, 0.162, 0.158], [0.150, 0.151, 0.149]), "A3", "T1_spatial", "test_seen"
        )
        entry = result["translation_error"]
        assert entry["mean_difference"] < 0
        assert entry["per_seed"] == pytest.approx({"0": -0.010, "1": -0.011, "2": -0.009})

    def test_a_consistent_large_effect_exceeds_seed_variability(self) -> None:
        """Same sign on every seed and |t| far above the critical value."""
        result = paired_comparison(
            _matrix([0.160, 0.162, 0.158], [0.150, 0.151, 0.149]), "A3", "T1_spatial", "test_seen"
        )
        entry = result["translation_error"]
        assert entry["all_seeds_same_sign"]
        assert abs(entry["t_statistic"]) > T_CRITICAL_DF2
        assert entry["exceeds_seed_variability"]

    def test_mixed_signs_never_exceed_seed_variability(self) -> None:
        """One seed going the other way fails the rule, whatever the mean."""
        result = paired_comparison(
            _matrix([0.160, 0.160, 0.160], [0.140, 0.140, 0.161]), "A3", "T1_spatial", "test_seen"
        )
        entry = result["translation_error"]
        assert not entry["all_seeds_same_sign"]
        assert not entry["exceeds_seed_variability"]

    def test_same_sign_but_noisy_does_not_exceed_seed_variability(self) -> None:
        """Agreement in sign alone is not enough; the t threshold also applies."""
        result = paired_comparison(
            _matrix([0.160, 0.160, 0.160], [0.159, 0.150, 0.1599]), "A3", "T1_spatial", "test_seen"
        )
        entry = result["translation_error"]
        assert entry["all_seeds_same_sign"]
        assert abs(entry["t_statistic"]) < T_CRITICAL_DF2
        assert not entry["exceeds_seed_variability"]

    def test_depth_zero_is_reported_separately_from_depth_one(self) -> None:
        """An effect confined to depth 1 must not show at depth 0."""
        result = paired_comparison(
            _matrix([0.160, 0.162, 0.158], [0.150, 0.151, 0.149]), "A3", "T1_spatial", "test_seen"
        )
        assert result["by_depth"]["0"]["mean_difference"] == 0.0
        assert result["by_depth"]["1"]["mean_difference"] < 0

    def test_seeds_without_both_cells_are_left_out(self) -> None:
        """A missing run shrinks the pairing; it is never paired with the wrong seed."""
        runs = _matrix([0.160, 0.162, 0.158], [0.150, 0.151, 0.149])
        runs = [run for run in runs if not (run["seed"] == 2 and run["cell"] == "T1_spatial")]
        result = paired_comparison(runs, "A3", "T1_spatial", "test_seen")
        assert result["seeds"] == [0, 1]
        assert not result["translation_error"]["exceeds_seed_variability"]


class TestTheTaxonomicControl:
    """Identical means identical, not close."""

    def test_an_exact_match_is_identical(self) -> None:
        """Bit-equal numbers pass."""
        runs = [
            _run("A1", "T0_global", 0, 0.1690, {"0": 0.1}),
            _run("A1", "T2_taxonomic", 0, 0.1690, {"0": 0.1}),
        ]
        assert _control_identity(runs, ["test_seen"])["A1/seed0"]["identical"]

    def test_a_tiny_difference_is_not_identical(self) -> None:
        """A control that differs at all has become a third treatment."""
        runs = [
            _run("A1", "T0_global", 0, 0.1690, {"0": 0.1}),
            _run("A1", "T2_taxonomic", 0, 0.1690 + 1e-12, {"0": 0.1}),
        ]
        assert not _control_identity(runs, ["test_seen"])["A1/seed0"]["identical"]
