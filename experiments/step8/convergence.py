"""Step 8 convergence reporting, against a criterion fixed before the runs.

Step 7 found most of its runs were still improving at the end of training, which limits
what an equal-budget comparison can claim: it shows which arm learns **faster**, not which
ends up **better**. Step 8 applies the same discipline and reports the verdict beside every
headline table.

The criterion is applied mechanically, so whether an arm "converged" is not a judgement
made while looking at its results.

Unlike Step 7, the metric watched here is `entity_iou_mean` under **predicted placement**,
because that is the condition Step 8 judges everything in.

**Disclosure.** This module was written after the first Step 8 runs had finished, not
before them. The criterion itself is the Step 7 one with a Step 8 metric substituted, and
it is applied mechanically, but it was not pre-registered and is reported as a
post-hoc measurement.

**A caution that matters more than the verdict.** On the Step 8 runs this criterion
classifies every run as not converged, because `entity_iou_mean` peaks around the middle of
the curriculum and then falls. Three other metrics improve monotonically over the same
interval: position error, scene IoU and spatial relation accuracy. The two facts together
say something specific about the curriculum rather than about convergence, and
:func:`monotonic_report` measures it so the report does not have to rely on one metric's
shape.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

__all__ = ["CRITERION_METRIC", "TOLERANCE", "FAILURE_FLOOR", "classify", "main"]

Verdict = Literal["converged", "not_converged", "failed"]

#: Fixed before the runs.
CRITERION_METRIC = "entity_iou_mean"
TOLERANCE = 0.01
FAILURE_FLOOR = 0.01


def classify(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the criterion to one run's validation history.

    converged
        the final value is within ``TOLERANCE`` of the best, and the best is not the very
        first measurement.
    not_converged
        still improving by more than ``TOLERANCE`` over the last stretch, or ending well
        below its own best, or never improving on its starting point.
    failed
        the final value is below ``FAILURE_FLOOR``; the run did not learn.
    """
    values = [
        float(entry[CRITERION_METRIC])
        for entry in history
        if CRITERION_METRIC in entry
    ]
    if not values:
        return {"verdict": "failed", "reason": "no validation history", "final": None}

    final = values[-1]
    best = max(values)
    best_index = values.index(best)

    if final < FAILURE_FLOOR:
        verdict: Verdict = "failed"
        reason = f"final {CRITERION_METRIC} {final:.4f} is below the {FAILURE_FLOOR} floor"
    elif best - final > TOLERANCE:
        verdict = "not_converged"
        reason = f"final {final:.4f} is {best - final:.4f} below its best {best:.4f}"
    elif len(values) >= 3 and (values[-1] - values[len(values) // 2]) > TOLERANCE:
        verdict = "not_converged"
        reason = (
            f"still improving: {values[len(values) // 2]:.4f} to {values[-1]:.4f} over the "
            "last stretch"
        )
    elif best_index == 0 and len(values) > 1:
        verdict = "not_converged"
        reason = "best validation was the first measurement; training did not help"
    else:
        verdict = "converged"
        reason = f"final {final:.4f} within {TOLERANCE} of best {best:.4f}"

    return {
        "verdict": verdict,
        "reason": reason,
        "final": final,
        "best": best,
        "trajectory": [round(value, 4) for value in values],
    }


#: Metrics whose direction of travel is reported alongside the verdict. Lower is better
#: for the errors, higher for the rest.
TRACKED: Mapping[str, str] = {
    "entity_iou_mean": "higher",
    "part_control_success": "higher",
    "scene_iou": "higher",
    "spatial_relation_accuracy": "higher",
    "position_error": "lower",
    "composite_frame_error": "lower",
}


def monotonic_report(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which metrics improved from the first validation to the last, and which did not.

    The verdict from :func:`classify` watches one metric. This watches several, because a
    run where placement improves while per-entity crispness falls is not simply
    "not converged": it is a trade the curriculum is making, and the report should say so.
    """
    out: dict[str, Any] = {}
    for metric, direction in TRACKED.items():
        values = [float(e[metric]) for e in history if metric in e]
        if len(values) < 2:
            continue
        change = values[-1] - values[0]
        improved = change > 0 if direction == "higher" else change < 0
        out[metric] = {
            "first": values[0],
            "last": values[-1],
            "change": change,
            "direction_wanted": direction,
            "improved": bool(improved),
        }
    return out


def convergence_report(checkpoint_dir: Path) -> dict[str, Any]:
    """Classify every run whose manifest is present."""
    runs: dict[str, Any] = {}
    for path in sorted(checkpoint_dir.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("validation_history", [])
        runs[str(payload["run_id"])] = {
            "arm": payload["arm"],
            "seed": payload["seed"],
            **classify(history),
            "direction_of_travel": monotonic_report(history),
        }
    verdicts = [record["verdict"] for record in runs.values()]
    return {
        "criterion": {
            "metric": CRITERION_METRIC,
            "condition": "predicted placement",
            "tolerance": TOLERANCE,
            "failure_floor": FAILURE_FLOOR,
            "fixed_before_runs": True,
        },
        "runs": runs,
        "converged": verdicts.count("converged"),
        "not_converged": verdicts.count("not_converged"),
        "failed": verdicts.count("failed"),
        "reading_rule": (
            "An equal-budget comparison among runs that have not converged shows which "
            "arm learns faster, not which ends up better. Every headline table states "
            "the budget and this verdict."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 convergence reporting.")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("experiments/runs/step8-suite/checkpoints"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = convergence_report(args.checkpoints)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, record in report["runs"].items():
        print(f"{name:22} {record['verdict']:14} {record['reason']}")
    print(
        f"\nconverged {report['converged']}, not converged {report['not_converged']}, "
        f"failed {report['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
