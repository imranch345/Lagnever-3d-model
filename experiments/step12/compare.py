"""Step 12: the paired comparison, and the two predictions recorded before it ran.

Step 11 found relation type worth 0.0003 degrees because its only channel was a scalar
attention bias holding 0.55% of the logit scale. Step 12 adds a per-relation vector to the
attended value — +4,608 parameters, zero-initialised — and asks whether that changes anything.

Both arms are scored **through the same evaluation path**, the Step 11 ablation harness, on the
same validation split with the same batching and the same seeded generator. Scoring the control
from its own training-time record instead would have compared two numbers produced by different
code, and a 1.0 degree decision rule cannot survive that.

Pre-registered in ``docs/STEP_12_RELATION_CONDITIONED_VALUES_PLAN.md``:

* **decision rule** — the encoding helps only if mean validation rotation error improves by at
  least 1.0 degrees *and* all three seeds move the same way;
* **P1** — ``shuffle_spatial_types`` should become expensive, costing at least 1.0 degrees
  against 0.0003 in Step 11;
* **P2** — the gain should appear first in seeds 0 and 2, which Step 11 measured as using the
  graph (2.59 and 2.51 degrees), before seed 1, which did not (0.38).

P1 and P2 are the mechanism's claims about itself and are reported whichever way they fall.
Neither is the decision rule.

No test split is read.

    python -m experiments.step12.compare
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.step11.graph_ablation import MINIMUM_MEANINGFUL_DEG, ablate

__all__ = ["GRAPH_USE_BY_SEED", "PREDICTION_CONDITION", "compare", "main"]

#: What the relationship graph was worth to each seed in Step 11 section 12.
GRAPH_USE_BY_SEED: dict[str, float] = {"0": 2.59, "1": 0.38, "2": 2.51}

#: The condition P1 speaks about, and what Step 11 measured it at.
PREDICTION_CONDITION = "shuffle_spatial_types"
STEP11_SHUFFLE_COST_DEG = 0.0003

#: ``entities_only`` is not part of the decision rule. It is carried over from Step 11 section
#: 12, where intact-minus-entities_only measured what the relationship graph was worth to each
#: seed, so the same quantity can be read for the new encoder.
SCORED_CONDITIONS: tuple[str, ...] = ("intact", PREDICTION_CONDITION, "entities_only")


def _intact(report: dict[str, Any]) -> dict[str, float]:
    """Per-seed rotation error with the graph untouched."""
    return {
        seed: float(value)
        for seed, value in report["conditions"]["intact"]["per_seed_rotation_deg"].items()
    }


def _score_primary(control: dict[str, float], treatment: dict[str, float]) -> dict[str, Any]:
    """The pre-registered decision rule, applied without adjustment."""
    seeds = sorted(control)
    gains = {seed: control[seed] - treatment[seed] for seed in seeds}
    mean_gain = statistics.fmean(gains.values())
    same_sign = all(g > 0 for g in gains.values()) or all(g < 0 for g in gains.values())
    meaningful = abs(mean_gain) >= MINIMUM_MEANINGFUL_DEG and same_sign
    if meaningful and mean_gain > 0:
        verdict = "improves"
    elif meaningful:
        verdict = "worse"
    else:
        verdict = "no meaningful change"
    return {
        "control_per_seed_deg": control,
        "treatment_per_seed_deg": treatment,
        "control_mean_deg": statistics.fmean(control.values()),
        "treatment_mean_deg": statistics.fmean(treatment.values()),
        "per_seed_gain_deg": gains,
        "mean_gain_deg": mean_gain,
        "all_seeds_same_sign": same_sign,
        "minimum_meaningful_deg": MINIMUM_MEANINGFUL_DEG,
        "exceeds_minimum": bool(meaningful),
        "verdict": verdict,
        "rule": (
            "the relation-conditioned value encoding helps only if mean validation rotation "
            f"error improves by at least {MINIMUM_MEANINGFUL_DEG} deg and all three seeds move "
            "in the same direction"
        ),
    }


def _score_p1(treatment_report: dict[str, Any]) -> dict[str, Any]:
    """Whether destroying relation type became expensive under the new encoding."""
    entry = treatment_report["conditions"][PREDICTION_CONDITION]
    cost = float(entry["delta_rotation_deg"])
    holds = bool(cost >= MINIMUM_MEANINGFUL_DEG and entry["all_seeds_same_sign"])
    return {
        "prediction": (
            "if H3 is the binding constraint, shuffle_spatial_types should become expensive "
            "under the new encoding"
        ),
        "condition": PREDICTION_CONDITION,
        "step11_cost_deg": STEP11_SHUFFLE_COST_DEG,
        "step12_cost_deg": cost,
        "per_seed_cost_deg": entry["per_seed_delta"],
        "all_seeds_same_sign": entry["all_seeds_same_sign"],
        "threshold_deg": MINIMUM_MEANINGFUL_DEG,
        "holds": holds,
    }


def _score_p2(gains: dict[str, float]) -> dict[str, Any]:
    """Whether the gain arrived first in the seeds that already used the graph."""
    users = [gains[s] for s in ("0", "2") if s in gains]
    laggard = gains.get("1")
    mean_users = statistics.fmean(users) if users else 0.0
    holds = bool(laggard is not None and users and laggard < mean_users)
    return {
        "prediction": (
            "the gain should appear in the two seeds that already use connectivity before it "
            "appears in seed 1"
        ),
        "graph_use_by_seed_deg": GRAPH_USE_BY_SEED,
        "gain_seeds_0_and_2_deg": mean_users,
        "gain_seed_1_deg": laggard,
        "holds": holds,
        "note": (
            "scored as an ordering, not a threshold; it says nothing about whether the "
            "encoding helped overall"
        ),
    }


def compare(
    *,
    corpus_dir: Path,
    control_runs: Path,
    treatment_runs: Path,
    conditions: Sequence[str] = SCORED_CONDITIONS,
) -> dict[str, Any]:
    """Score both arms through the Step 11 harness and apply the pre-registered rules."""
    print("[step12] scoring the control (relation values off)", flush=True)
    control = ablate(
        corpus_dir=corpus_dir,
        runs_dir=control_runs,
        conditions=conditions,
        relation_values=False,
    )
    print("[step12] scoring the treatment (relation values on)", flush=True)
    treatment = ablate(
        corpus_dir=corpus_dir,
        runs_dir=treatment_runs,
        conditions=conditions,
        relation_values=True,
    )
    primary = _score_primary(_intact(control), _intact(treatment))
    graph_worth = {
        name: {
            seed: report["conditions"]["entities_only"]["per_seed_rotation_deg"][seed]
            - report["conditions"]["intact"]["per_seed_rotation_deg"][seed]
            for seed in sorted(report["conditions"]["intact"]["per_seed_rotation_deg"])
        }
        for name, report in (("control", control), ("treatment", treatment))
    }
    return {
        "experiment_id": "step12-relation-conditioned-values",
        "preregistration": "docs/STEP_12_RELATION_CONDITIONED_VALUES_PLAN.md",
        "split": "validation",
        "test_splits_read": [],
        "arm": "A3",
        "rotation_loss_weight": 3.0,
        "parameter_delta": {"off": 3226647, "on": 3231255, "added": 4608, "fraction": 0.0014},
        "primary": primary,
        "predictions": {"P1": _score_p1(treatment), "P2": _score_p2(primary["per_seed_gain_deg"])},
        "graph_worth_deg": graph_worth,
        "control_ablation": control["conditions"],
        "treatment_ablation": treatment["conditions"],
        "notes": [
            "Both arms scored through one evaluation path so the comparison is paired.",
            "Control and treatment are bit-identical at initialisation but for the new tensor.",
            "P1 and P2 were recorded before the run and are not the decision rule.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 12 paired comparison.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--control", type=Path, default=Path("experiments/runs/step10-change2-w3"))
    parser.add_argument(
        "--treatment", type=Path, default=Path("experiments/runs/step12-relation-values")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step12-relation-values/comparison.json")
    )
    args = parser.parse_args(argv)

    report = compare(
        corpus_dir=args.corpus, control_runs=args.control, treatment_runs=args.treatment
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    primary = report["primary"]
    print("\nseed   control   treatment      gain")
    for seed in sorted(primary["control_per_seed_deg"]):
        gain = primary["per_seed_gain_deg"][seed]
        print(
            f"{seed:<6} {primary['control_per_seed_deg'][seed]:7.2f} "
            f"{primary['treatment_per_seed_deg'][seed]:11.2f} {gain:+9.2f}"
        )
    print(
        f"mean   {primary['control_mean_deg']:7.2f} {primary['treatment_mean_deg']:11.2f} "
        f"{primary['mean_gain_deg']:+9.2f}"
    )
    print(f"\nverdict: {primary['verdict']}  (rule: >= {MINIMUM_MEANINGFUL_DEG} deg, seeds agree)")
    for name, entry in report["predictions"].items():
        print(f"{name}: {'holds' if entry['holds'] else 'does not hold'}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
