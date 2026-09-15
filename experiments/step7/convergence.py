"""Experiment 11: apply the pre-registered convergence criterion to a run.

The criterion is fixed in `docs/step7/08_baseline_convergence_method.md` and applied
here mechanically, so that whether an arm "converged" is not a judgement made while
looking at the arm's results.

An arm reported as **failed** invalidates comparisons drawn against it. That matters
most for the appearance baseline: a baseline that never trained is not evidence that
structure helps.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

__all__ = ["Verdict", "classify", "convergence_report", "main"]

Verdict = Literal["converged", "not_converged", "failed"]

#: Validation metric the criterion is applied to. Fixed in advance.
CRITERION_METRIC = "scene_iou"
#: Tolerance for "within reach of its best" and "still improving". Fixed in advance.
TOLERANCE = 0.02
#: Below this final value an arm is reported as failed. Fixed in advance.
FAILURE_FLOOR = 0.1


def classify(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the criterion to one run's validation history."""
    values = [float(entry[CRITERION_METRIC]) for entry in history if CRITERION_METRIC in entry]
    steps = [float(entry.get("step", index)) for index, entry in enumerate(history)]
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
        reason = (
            f"final {final:.4f} is {best - final:.4f} below the best {best:.4f}, "
            f"more than the {TOLERANCE} tolerance"
        )
    elif len(values) >= 3 and (values[-1] - values[len(values) // 2]) > TOLERANCE:
        verdict = "not_converged"
        reason = (
            f"still improving over the last stretch: {values[len(values) // 2]:.4f} "
            f"to {values[-1]:.4f}"
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
        "best_step": steps[best_index] if steps else None,
        "trajectory": [round(value, 4) for value in values],
    }


def convergence_report(checkpoint_dir: Path) -> dict[str, Any]:
    """Classify every run whose manifest is present."""
    runs: dict[str, Any] = {}
    for path in sorted(checkpoint_dir.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs[str(payload["run_id"])] = {
            "arm": payload["arm"],
            "seed": payload["seed"],
            **classify(payload.get("validation_history", [])),
        }
    verdicts = {name: record["verdict"] for name, record in runs.items()}
    failed = sorted(name for name, verdict in verdicts.items() if verdict == "failed")
    return {
        "criterion": {
            "metric": CRITERION_METRIC,
            "tolerance": TOLERANCE,
            "failure_floor": FAILURE_FLOOR,
            "fixed_before_runs": True,
        },
        "runs": runs,
        "failed_runs": failed,
        "comparisons_invalidated": [
            f"Any comparison drawn against {name} is unavailable, not favourable."
            for name in failed
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 7 convergence reporting.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = convergence_report(args.checkpoints)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, record in report["runs"].items():
        print(f"{name:20} {record['verdict']:14} {record['reason']}")
    if report["failed_runs"]:
        print()
        for line in report["comparisons_invalidated"]:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
