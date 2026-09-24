"""Step 16: locate the earliest change, and test it against the non-escaper.

The trajectories are recorded; this decides what they say. Three questions, in order of how much
they can support.

**When does each series first change?** The plan fixed a criterion before any trajectory was
seen: the first checkpoint at which a series departs from its own step-0-to-400 baseline by more
than three baseline standard deviations, and stays departed at the next checkpoint.

That criterion turns out to be badly suited to these runs, and the reason is visible rather than
arguable: the 0-400 window contains the initial convergence transient, where validation rotation
falls from 61.7 to 23.6. Its standard deviation is therefore about 12 degrees, and three of those
is larger than any subsequent movement, so the criterion cannot fire for the very series whose
change matters most. It is computed and reported anyway, because it was preregistered, and a
second criterion using the **plateau** (steps 150-800) as the baseline is computed beside it and
labelled **post hoc** throughout. The post-hoc one is not used to claim a mechanism.

**Does anything separate the escapers from the non-escaper before escape?** Here the trap is that
three runs differ from initialisation, so a persistent offset is not a precursor. The criterion
used is scale-free and stated in advance of looking: a series separates the groups at a
checkpoint when the gap between the escapers' mean and the non-escaper exceeds the spread
*within* the escaper group at that same checkpoint, and does so from there through the first
escape. Between-group difference has to beat within-group difference to count.

**H9 to H12.** Evaluated from the above, with H12 -- no detectable precursor -- treated as a real
answer rather than a fallback.

    python -m experiments.step16.analysis
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.step15.seed_modes import GRAPH_USING_DEG

__all__ = ["ESCAPERS", "NON_ESCAPER", "SERIES", "analyse", "main"]

#: Determined from the recorded trajectories: seed 3 crosses no threshold by 1200 but its usage
#: climbs 0.06 -> 1.30 over the final 400 steps, so it is a slow escaper caught mid-transition
#: rather than a third mode. Recorded here because the grouping affects every separation test.
ESCAPERS: tuple[int, ...] = (9, 3)
NON_ESCAPER = 5

#: Preregistered baseline window, and the post-hoc plateau window.
BASELINE = (0, 400)
PLATEAU = (150, 800)
SIGMA = 3.0

#: series name -> (where it lives, the key)
SERIES: dict[str, tuple[str, str]] = {
    "validation_rotation_deg": ("row", "validation_rotation_deg"),
    "graph_usage_deg": ("row", "graph_usage_deg"),
    "relational_subset_deg": ("row", "relational_subset_deg"),
    "context_written_norm": ("representation", "context_written_norm"),
    "context_written_share": ("representation", "context_written_share"),
    "entity_latent_std": ("representation", "entity_latent_std"),
    "relation_bias_abs_max": ("representation", "relation_bias_abs_max"),
    "frame_rotation_output_std": ("representation", "frame_rotation_output_std"),
    "grad_graph_encoder": ("gradients", "graph_encoder"),
    "grad_relation_parameters": ("gradients", "relation_parameters"),
    "grad_frame_head": ("gradients", "frame_head"),
    "obj_frame_rotation": ("objective", "frame_rotation"),
    "obj_frame_chordal": ("objective", "frame_chordal"),
    "obj_total": ("objective", "total"),
}


def _series(trajectory: Sequence[dict[str, Any]], name: str) -> dict[int, float]:
    """One named series as ``step -> value``."""
    where, key = SERIES[name]
    out: dict[int, float] = {}
    for row in trajectory:
        value = row[key] if where == "row" else row[where][key]
        if value is not None:
            out[int(row["step"])] = float(value)
    return out


def _first_departure(values: dict[int, float], window: tuple[int, int]) -> dict[str, Any]:
    """First checkpoint outside ``sigma`` baseline deviations that stays outside."""
    steps = sorted(values)
    inside = [values[s] for s in steps if window[0] <= s <= window[1]]
    if len(inside) < 2:
        return {"detected": False, "reason": "baseline window too small"}
    mean = statistics.fmean(inside)
    deviation = statistics.stdev(inside)
    threshold = SIGMA * deviation
    after = [s for s in steps if s > window[1]]
    for index, step in enumerate(after[:-1]):
        if (
            abs(values[step] - mean) > threshold
            and abs(values[after[index + 1]] - mean) > threshold
        ):
            return {
                "detected": True,
                "step": step,
                "baseline_mean": mean,
                "baseline_std": deviation,
                "threshold": threshold,
                "value": values[step],
            }
    return {
        "detected": False,
        "baseline_mean": mean,
        "baseline_std": deviation,
        "threshold": threshold,
        "largest_later_deviation": max((abs(values[s] - mean) for s in after), default=0.0),
    }


def _separation(
    escapers: Sequence[dict[int, float]], other: dict[int, float], until: int
) -> dict[str, Any]:
    """Earliest checkpoint from which between-group gap beats within-group spread."""
    steps = [s for s in sorted(other) if s <= until]
    qualifying: list[int] = []
    detail: list[dict[str, float]] = []
    for step in steps:
        if not all(step in e for e in escapers):
            continue
        values = [e[step] for e in escapers]
        gap = abs(statistics.fmean(values) - other[step])
        within = max(values) - min(values)
        same_side = all(v > other[step] for v in values) or all(v < other[step] for v in values)
        detail.append({"step": step, "gap": gap, "within_group_spread": within})
        if same_side and gap > within:
            qualifying.append(step)
        else:
            qualifying.clear()
    sustained = None
    if qualifying:
        # Sustained means: qualifying from this step through the last measured step here.
        candidate = qualifying[0]
        if qualifying[-1] == detail[-1]["step"]:
            sustained = candidate
    return {
        "separated": sustained is not None,
        "earliest_sustained_step": sustained,
        "checkpoints_examined": len(detail),
        "per_step": detail,
    }


def _grouping_control(trajectories: dict[int, list[dict[str, Any]]], until: int) -> dict[str, Any]:
    """How many series "separate" under each possible odd-one-out grouping.

    With two escapers and one non-escaper, "both escapers on the same side of the non-escaper"
    happens by chance two thirds of the time, so the separation test needs a control before any
    of its hits can be believed. Every seed is tried as the odd one out; if an arbitrary
    grouping yields as many separations as the real one, the test has no discriminating power at
    this sample size and its hits cannot support a precursor claim.
    """
    out: dict[str, Any] = {}
    seeds = sorted(trajectories)
    for odd in seeds:
        pair = [s for s in seeds if s != odd]
        hits: dict[str, int] = {}
        for name in SERIES:
            group = [_series(trajectories[s], name) for s in pair]
            other = _series(trajectories[odd], name)
            if not other or not all(group):
                continue
            result = _separation(group, other, until=until)
            step = result["earliest_sustained_step"]
            if result["separated"] and step is not None and step < until:
                hits[name] = step
        out[str(odd)] = {
            "is_the_real_grouping": odd == NON_ESCAPER,
            "series_separating": hits,
            "count": len(hits),
        }
    counts = {key: entry["count"] for key, entry in out.items()}
    real = counts[str(NON_ESCAPER)]
    out["verdict"] = (
        "the separation test has no discriminating power at this sample size: an arbitrary "
        "grouping separates as many series as the real one"
        if max(counts.values()) >= real
        and any(k != str(NON_ESCAPER) and v >= real for k, v in counts.items())
        else "the real grouping separates more series than any arbitrary one"
    )
    return out


def analyse(*, runs_dir: Path) -> dict[str, Any]:
    """Every Step 16 analysis, over the three recorded trajectories."""
    loaded = {
        seed: json.loads((runs_dir / f"trajectory_seed{seed}.json").read_text(encoding="utf-8"))
        for seed in (*ESCAPERS, NON_ESCAPER)
    }
    trajectories = {seed: report["trajectory"] for seed, report in loaded.items()}
    escape = {seed: report["escape"] for seed, report in loaded.items()}
    first_escape = min(
        (e["first_crossing_step"] for e in escape.values() if e.get("escaped")), default=1200
    )

    departures: dict[str, Any] = {}
    for seed, trajectory in trajectories.items():
        for name in SERIES:
            values = _series(trajectory, name)
            if not values:
                continue
            departures.setdefault(str(seed), {})[name] = {
                "preregistered_baseline_0_400": _first_departure(values, BASELINE),
                "post_hoc_baseline_150_800": _first_departure(values, PLATEAU),
            }

    separations: dict[str, Any] = {}
    for name in SERIES:
        escaper_series = [_series(trajectories[s], name) for s in ESCAPERS]
        other = _series(trajectories[NON_ESCAPER], name)
        if not other or not all(escaper_series):
            continue
        separations[name] = _separation(escaper_series, other, until=first_escape)

    separating_before = sorted(
        name
        for name, entry in separations.items()
        if entry["separated"] and entry["earliest_sustained_step"] < first_escape
    )
    ordering: dict[str, Any] = {
        "first_escape_step": first_escape,
        "escape_intervals": {str(s): escape[s].get("interval", "never crossed") for s in escape},
        "series_separating_before_first_escape": separating_before,
    }
    earliest = {name: separations[name]["earliest_sustained_step"] for name in separating_before}
    ordering["earliest_separating_step"] = earliest
    ordering["earliest_series"] = (
        sorted(earliest, key=lambda n: earliest[n])[0] if earliest else None
    )
    return {
        "experiment_id": "step16-escape-analysis",
        "status": "diagnostic; correlational, no causal claim",
        "preregistration": "docs/STEP_16_ESCAPE_TRIGGER_PLAN.md",
        "test_splits_read": [],
        "escapers": list(ESCAPERS),
        "non_escaper": NON_ESCAPER,
        "escape_threshold_deg": GRAPH_USING_DEG,
        "criterion": {
            "preregistered": (
                f"first checkpoint more than {SIGMA} baseline standard deviations from the "
                f"step {BASELINE[0]}-{BASELINE[1]} mean, persisting one further checkpoint"
            ),
            "post_hoc": (
                f"the same, with the plateau (steps {PLATEAU[0]}-{PLATEAU[1]}) as the baseline; "
                "computed because the preregistered window contains the initial transient"
            ),
            "separation": (
                "between-group gap exceeds within-escaper spread, on the same side, sustained "
                "through the first escape"
            ),
        },
        "departures": departures,
        "separations": separations,
        "grouping_control": _grouping_control(trajectories, until=first_escape),
        "ordering": ordering,
        "notes": [
            "Seed 3 is grouped with the escapers: its usage climbs 0.06 to 1.30 over the final "
            "400 steps and is still rising, so it is mid-transition rather than a third mode.",
            "Three runs cannot establish reproducibility; separations here are suggestive only.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 16 trajectory analysis.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs/step16"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step16/analysis.json"))
    args = parser.parse_args(argv)
    report = analyse(runs_dir=args.runs)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("escape intervals:")
    for seed, interval in report["ordering"]["escape_intervals"].items():
        print(f"  seed {seed}: {interval}")
    print(f"\nfirst escape at step {report['ordering']['first_escape_step']}")
    print("\npreregistered criterion (baseline 0-400), seed 9:")
    for name, entry in report["departures"]["9"].items():
        pre = entry["preregistered_baseline_0_400"]
        state = f"step {pre['step']}" if pre["detected"] else "no departure"
        print(f"  {name:<30} {state}")
    print("\npost-hoc criterion (baseline 150-800, LABELLED POST HOC), seed 9:")
    for name, entry in report["departures"]["9"].items():
        post = entry["post_hoc_baseline_150_800"]
        state = f"step {post['step']}" if post["detected"] else "no departure"
        print(f"  {name:<30} {state}")
    print("\nseries separating escapers from the non-escaper before first escape:")
    earliest = report["ordering"]["earliest_separating_step"]
    for name in report["ordering"]["series_separating_before_first_escape"]:
        print(f"  {name:<30} sustained from step {earliest[name]}")
    print(f"\nearliest separating series: {report['ordering']['earliest_series']}")
    print("\ngrouping control (odd one out -> series separating):")
    for seed, entry in report["grouping_control"].items():
        if seed == "verdict":
            continue
        label = "REAL" if entry["is_the_real_grouping"] else "control"
        print(f"  seed {seed:>2} ({label:<7}): {entry['count']} series")
    print(f"  -> {report['grouping_control']['verdict']}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
