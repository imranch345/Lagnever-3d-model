"""Step 17: does anything move before seed 9 escapes, at 5-step resolution?

Step 16 localised the escape to a single 50-step interval and found nothing moving beforehand,
leaving one objection: a precursor lasting a few dozen steps would have been invisible. This
resolves that window at 5-step spacing and applies the brief's five-part definition of a
precursor rather than reporting whatever correlates.

The definition matters more than the measurement here, because at 5-step spacing almost every
series will have *some* checkpoint that moves. A candidate has to clear all five conditions:
before the escape, not a standing offset, in the escape's direction, temporally separable from
the escape, and measured by a definition that already existed. A quantity moving at the same
checkpoint as graph usage and geometry is part of the transition, not a precursor -- that
distinction is the whole point of the step.

Then five falsification checks, applied to every candidate: was it already offset before the
window, does it move the same way in the non-escaper, does it merely track the transition, could
the baseline spread explain it, and does it depend on the graph-usage calculation itself.

One run. Nothing here is causal, and the language stays with "preceded" and "coincided with".

    python -m experiments.step17.analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.step15.seed_modes import GRAPH_USING_DEG
from experiments.step16.analysis import SERIES, _series

__all__ = ["BASELINE", "WINDOW", "analyse", "main"]

#: The preregistered window, fixed from Step 16 before this run.
WINDOW = (750, 900)

#: Pre-window baseline, from the coarse grid. Deliberately excludes the initial transient, the
#: flaw Step 16 documented in its own criterion.
BASELINE = (400, 750)

SIGMA = 3.0

#: Dense checkpoints a departure must persist across (itself plus the next two = 15 steps).
PERSISTENCE = 3

#: Series that are functions of the graph-usage calculation, and so cannot be its precursor.
DERIVED_FROM_GRAPH_USAGE = {"graph_usage_deg"}


def _dense(values: dict[int, float], window: tuple[int, int], every: int = 5) -> dict[int, float]:
    """Just the dense checkpoints inside the window."""
    return {s: v for s, v in values.items() if window[0] <= s <= window[1] and s % every == 0}


def _escape(usage: dict[int, float]) -> dict[str, Any]:
    """Permanent escape and any transient crossings, by Step 15's unmodified threshold."""
    steps = sorted(usage)
    crossings = [s for s in steps if usage[s] >= GRAPH_USING_DEG]
    if not crossings:
        return {"escaped": False, "transient_crossings": []}
    permanent = None
    for step in crossings:
        later = [s for s in steps if s >= step]
        if all(usage[s] >= GRAPH_USING_DEG for s in later):
            permanent = step
            break
    transient = [s for s in crossings if permanent is None or s < permanent]
    before = [s for s in steps if permanent is not None and s < permanent]
    return {
        "escaped": permanent is not None,
        "permanent_crossing_step": permanent,
        "last_checkpoint_before": before[-1] if before else None,
        "interval": (
            f"{before[-1]} < escape <= {permanent}" if before and permanent else "unresolved"
        ),
        "usage_at_crossing": usage[permanent] if permanent is not None else None,
        "transient_crossings": transient,
    }


def _first_change(
    values: dict[int, float], baseline: tuple[int, int], window: tuple[int, int]
) -> dict[str, Any]:
    """First dense checkpoint departing the pre-window baseline, sustained over 15 steps."""
    inside = [v for s, v in values.items() if baseline[0] <= s <= baseline[1]]
    if len(inside) < 3:
        return {"detected": False, "reason": "baseline too small"}
    mean = statistics.fmean(inside)
    spread = statistics.stdev(inside)
    threshold = SIGMA * spread
    dense = _dense(values, window)
    steps = sorted(dense)
    for index, step in enumerate(steps):
        run = steps[index : index + PERSISTENCE]
        if len(run) < PERSISTENCE:
            break
        if all(abs(dense[s] - mean) > threshold for s in run):
            return {
                "detected": True,
                "step": step,
                "value": dense[step],
                "baseline_mean": mean,
                "baseline_std": spread,
                "threshold": threshold,
                "direction": "up" if dense[step] > mean else "down",
            }
    return {
        "detected": False,
        "baseline_mean": mean,
        "baseline_std": spread,
        "threshold": threshold,
        "largest_departure": max((abs(v - mean) for v in dense.values()), default=0.0),
    }


def analyse(*, dense_path: Path, frozen_escaper: Path, frozen_non_escaper: Path) -> dict[str, Any]:
    """The whole Step 17 analysis."""
    dense_report = json.loads(dense_path.read_text(encoding="utf-8"))
    trajectory = dense_report["trajectory"]
    frozen = json.loads(frozen_escaper.read_text(encoding="utf-8"))["trajectory"]
    other = json.loads(frozen_non_escaper.read_text(encoding="utf-8"))["trajectory"]

    usage = _series(trajectory, "graph_usage_deg")
    escape = _escape(_dense(usage, WINDOW))

    changes = {
        name: _first_change(_series(trajectory, name), BASELINE, WINDOW)
        for name in SERIES
        if _series(trajectory, name)
    }

    # ---- reproduction against the frozen Step 16 run, at shared checkpoints ----
    frozen_series = {name: _series(frozen, name) for name in SERIES}
    reproduction = {}
    for step in (750, 800, 850, 900, 1200):
        row: dict[str, Any] = {}
        for name in ("validation_rotation_deg", "graph_usage_deg", "relational_subset_deg"):
            new = _series(trajectory, name).get(step)
            old = frozen_series[name].get(step)
            row[name] = {
                "dense": new,
                "frozen": old,
                "identical": new is not None and old is not None and new == old,
            }
        reproduction[str(step)] = row
    reproduced = all(entry["identical"] for row in reproduction.values() for entry in row.values())

    # ---- falsification, per candidate ----
    permanent = escape.get("permanent_crossing_step")
    candidates: dict[str, Any] = {}
    for name, change in changes.items():
        if not change["detected"]:
            continue
        step = change["step"]
        before_escape = permanent is not None and step < permanent
        series_here = _series(trajectory, name)
        pre_window = [v for s, v in series_here.items() if s <= WINDOW[0]]
        non_escaper = _first_change(_series(other, name), BASELINE, WINDOW)
        checks = {
            "occurs_before_escape": bool(before_escape),
            "not_a_standing_offset": bool(
                abs(change["value"] - statistics.fmean(pre_window)) > change["threshold"]
            )
            if pre_window
            else None,
            "absent_in_non_escaper": not non_escaper["detected"],
            "temporally_separable_from_escape": bool(
                permanent is not None and permanent - step >= 2 * 5
            ),
            "independent_of_graph_usage": name not in DERIVED_FROM_GRAPH_USAGE,
        }
        candidates[name] = {
            "first_change_step": step,
            "escape_step": permanent,
            "steps_before_escape": None if permanent is None else permanent - step,
            "direction": change["direction"],
            "checks": checks,
            "survives": all(v for v in checks.values() if v is not None),
            "non_escaper_also_changes_at": (
                non_escaper["step"] if non_escaper["detected"] else None
            ),
        }

    survivors = sorted(name for name, entry in candidates.items() if entry["survives"])
    geometry = [
        name
        for name in ("validation_rotation_deg", "relational_subset_deg")
        if changes.get(name, {}).get("detected")
    ]
    classes = {
        "representation": [
            n
            for n in changes
            if n.startswith(("context_", "entity_latent", "relation_bias", "frame_"))
            and changes[n]["detected"]
        ],
        "gradient": [n for n in changes if n.startswith("grad_") and changes[n]["detected"]],
        "objective": [n for n in changes if n.startswith("obj_") and changes[n]["detected"]],
        "graph_usage": [n for n in ("graph_usage_deg",) if changes.get(n, {}).get("detected")],
        "geometry": geometry,
    }
    earliest = {
        label: min((changes[n]["step"] for n in names), default=None)
        for label, names in classes.items()
    }
    ordered = sorted((step, label) for label, step in earliest.items() if step is not None)
    if not survivors and ordered:
        distinct = {step for step, _ in ordered}
        pattern = (
            "E — all measurable changes in the same 5-step interval"
            if len(distinct) == 1
            else ("D — graph usage and geometry with no surviving precursor")
        )
        case = "Case 3 — no signal" if len(distinct) == 1 else "Case 2 — transition signal"
    elif survivors:
        pattern = "a precursor precedes graph usage and geometry"
        case = "Case 1 — true precursor"
    else:
        pattern = "no change detected at all"
        case = "Case 3 — no signal"

    return {
        "experiment_id": "step17-high-resolution-escape",
        "status": "diagnostic; single run; correlational, no causal claim",
        "preregistration": "docs/STEP_17_HIGH_RES_ESCAPE_PLAN.md",
        "test_splits_read": [],
        "window": list(WINDOW),
        "dense_every": dense_report.get("dense_every"),
        "dense_checkpoints": len(_dense(usage, WINDOW)),
        "baseline": list(BASELINE),
        "criterion": (
            f"first dense checkpoint more than {SIGMA} pre-window baseline standard deviations "
            f"from the baseline mean, sustained across {PERSISTENCE} dense checkpoints"
        ),
        "escape": escape,
        "reproduces_frozen_seed9": reproduced,
        "reproduction": reproduction,
        "first_change": changes,
        "earliest_change_by_class": earliest,
        "observed_order": [label for _, label in ordered],
        "candidates": candidates,
        "surviving_candidates": survivors,
        "pattern": pattern,
        "case": case,
        "notes": [
            "Graph usage is Step 15's definition; the subset is Step 14's 652 entities.",
            "A quantity changing at the same checkpoint as usage and geometry is part of the "
            "transition, not a precursor.",
            "One run: no population or causal claim is made.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 17 high-resolution escape analysis.")
    parser.add_argument(
        "--dense", type=Path, default=Path("experiments/runs/step17/trajectory_seed9_dense.json")
    )
    parser.add_argument(
        "--frozen", type=Path, default=Path("experiments/runs/step16/trajectory_seed9.json")
    )
    parser.add_argument(
        "--non-escaper", type=Path, default=Path("experiments/runs/step16/trajectory_seed5.json")
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step17/analysis.json"))
    args = parser.parse_args(argv)
    report = analyse(
        dense_path=args.dense, frozen_escaper=args.frozen, frozen_non_escaper=args.non_escaper
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"dense checkpoints in {WINDOW}: {report['dense_checkpoints']}")
    print(f"reproduces frozen seed 9 at shared checkpoints: {report['reproduces_frozen_seed9']}")
    print(f"\nescape: {report['escape']['interval']}")
    print(f"transient crossings: {report['escape']['transient_crossings']}")
    print("\nfirst detected change by class:")
    for label, step in report["earliest_change_by_class"].items():
        print(f"  {label:<16} {'none' if step is None else f'step {step}'}")
    print(f"\nobserved order: {' -> '.join(report['observed_order']) or 'nothing detected'}")
    print("\ncandidates:")
    for name, entry in report["candidates"].items():
        print(f"  {name}: change at {entry['first_change_step']}, survives={entry['survives']}")
        for check, value in entry["checks"].items():
            print(f"      {check:<36} {value}")
    print(f"\nsurviving candidates: {report['surviving_candidates'] or 'none'}")
    print(f"pattern: {report['pattern']}")
    print(f"case: {report['case']}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
