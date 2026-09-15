"""Render Step 7 result tables straight from the run reports.

Every number in the Step 7 documents is produced here rather than typed by hand, so a
table and the JSON it claims to summarise cannot drift apart. Missing files are reported
as missing rather than silently omitted, because a table that quietly loses an arm is
worse than one that says the arm has no data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["load", "arm_table", "variant_table", "perturbation_table", "editing_table", "main"]

ARM_ORDER: tuple[str, ...] = ("A3", "A3L", "A2", "A1", "A1M", "A0")
VARIANTS: tuple[str, ...] = ("normal", "mirrored", "transposed", "rotated")


def load(path: Path) -> Mapping[str, Any] | None:
    """Read a run report, or return None when it has not been produced."""
    if not path.exists():
        return None
    payload: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _cell(value: float | None, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _row(name: str, cells: Sequence[str]) -> str:
    return "| " + " | ".join([name, *cells]) + " |"


def _header(first: str, columns: Sequence[str]) -> list[str]:
    return [
        _row(first, list(columns)),
        "| " + " | ".join(["---"] * (len(columns) + 1)) + " |",
    ]


def arm_table(report: Mapping[str, Any], metrics: Sequence[str]) -> str:
    """One row per arm, one column per metric, with the seed spread."""
    aggregates = report["aggregates"]
    counts = report["training"]["parameter_counts"]
    lines = _header("arm", ["parameters", *metrics])
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        cells = [f"{int(counts.get(arm, 0)):,}"]
        for metric in metrics:
            entry = aggregates[arm].get(metric)
            if entry is None:
                cells.append("n/a")
                continue
            mean = _cell(entry["mean"])
            # A spread beside an undefined mean reads as though the value were measured
            # and merely uncertain. It was not measured at all.
            defined = entry.get("defined_n", entry.get("n", 1))
            spread = f" ± {entry['std']:.3f}" if mean != "n/a" and defined > 1 else ""
            note = "" if mean != "n/a" else f" (0 of {int(entry.get('n', 0))} seeds defined)"
            cells.append(f"{mean}{spread}{note}")
        lines.append(_row(arm, cells))
    return "\n".join(lines)


def variant_table(report: Mapping[str, Any], metric: str) -> str:
    """One row per arm, one column per anatomical arrangement."""
    aggregates = report["aggregates"]
    lines = _header("arm", list(VARIANTS) + ["spread"])
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        values: list[float] = []
        cells: list[str] = []
        for variant in VARIANTS:
            entry = aggregates[arm].get(f"variant_{variant}_{metric}")
            if entry is None:
                cells.append("n/a")
                continue
            cells.append(_cell(entry["mean"]))
            if math.isfinite(entry["mean"]):
                values.append(entry["mean"])
        cells.append(f"{max(values) - min(values):.4f}" if len(values) > 1 else "n/a")
        lines.append(_row(arm, cells))
    return "\n".join(lines)


def placement_table(report: Mapping[str, Any], metric: str) -> str:
    """One row per arm, one column per placement condition, plus the gap."""
    aggregates = report["aggregates"]
    lines = _header("arm", ["placement given", "placement inferred", "cost of inferring"])
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        given = aggregates[arm].get("given", {}).get(metric, {}).get("mean")
        inferred = aggregates[arm].get("inferred", {}).get(metric, {}).get("mean")
        gap = (
            given - inferred
            if isinstance(given, float)
            and isinstance(inferred, float)
            and math.isfinite(given)
            and math.isfinite(inferred)
            else None
        )
        lines.append(_row(arm, [_cell(given), _cell(inferred), _cell(gap)]))
    return "\n".join(lines)


def perturbation_table(
    report: Mapping[str, Any], arm: str, group: str, condition: str, metric: str
) -> str:
    """One row per perturbation, showing the value and its change from the control."""
    arms = report["arms"]
    if arm not in arms:
        return f"_No perturbation data for {arm}._"
    entries = arms[arm][group]
    labels = report[group if group == "awr_cases" else "perturbations"]
    lines = _header("perturbation", [metric, "change from control", "what it does"])
    for name, conditions in entries.items():
        values = conditions.get(condition, {})
        delta = values.get(f"delta_{metric}")
        lines.append(
            _row(
                f"`{name}`",
                [
                    _cell(values.get(metric)),
                    ("n/a" if delta is None else f"{delta:+.4f}"),
                    str(labels.get(name, "")),
                ],
            )
        )
    return "\n".join(lines)


def editing_table(report: Mapping[str, Any], metrics: Sequence[str]) -> str:
    """One row per edit mechanism."""
    lines = _header("arm", ["head parameters", *metrics])
    for arm, payload in report["arms"].items():
        values = payload["metrics"]
        cells = [f"{int(payload['head_parameters'].get('total', 0)):,}"]
        cells.extend(_cell(values.get(metric)) for metric in metrics)
        lines.append(_row(arm, cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print every table that has data behind it."""
    parser = argparse.ArgumentParser(description="Render Step 7 result tables.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args(argv)

    whole = load(args.runs / "step7-whole-organ" / "whole_organ_report.json")
    placement = load(args.runs / "step7-placement" / "placement_report.json")
    perturbation = load(args.runs / "step7-perturbation" / "perturbation_report.json")
    editing = load(args.runs / "step7-editing" / "editing_report.json")

    if whole is not None:
        print("## Whole-organ benchmark, placement given\n")
        print(
            arm_table(
                whole,
                (
                    "entity_iou_mean",
                    "part_control_success",
                    "spatial_relation_accuracy",
                    "placement_error",
                    "scene_iou",
                ),
            )
        )
        print("\n## Entity ownership IoU by arrangement\n")
        print(variant_table(whole, "entity_iou_mean"))
        print("\n## Counterfactual response\n")
        print(
            arm_table(
                whole,
                (
                    "counterfactual_relation_sensitivity",
                    "counterfactual_correctness",
                    "invariant_preservation",
                ),
            )
        )
        print("\n## Level of detail\n")
        print(arm_table(whole, ("lod1_iou", "lod2_iou", "lod3_iou", "lod_detail_gain")))
    else:
        print("_Whole-organ report not produced._")

    if placement is not None:
        print("\n## Placement conditions, entity ownership IoU\n")
        print(placement_table(placement, "entity_iou_mean"))
        print("\n## Placement conditions, spatial relation accuracy\n")
        print(placement_table(placement, "spatial_relation_accuracy"))
        blind = placement["relation_baseline"]["blind_normal_arrangement"]["accuracy"]
        print(f"\nRelation-blind floor: {blind:.4f}. Oracle: 1.0000.")

    if perturbation is not None:
        for arm in ("A3", "A1M"):
            if arm not in perturbation["arms"]:
                continue
            print(f"\n## Perturbations, {arm}, placement inferred\n")
            print(
                perturbation_table(
                    perturbation, arm, "perturbations", "inferred", "entity_iou_mean"
                )
            )
            print(f"\n## Partial representation, {arm}, placement inferred\n")
            print(perturbation_table(perturbation, arm, "awr_cases", "inferred", "entity_iou_mean"))

    if editing is not None:
        print("\n## Persistent local editing\n")
        print(
            editing_table(
                editing,
                (
                    "edit_target_accuracy",
                    "local_edit_locality",
                    "unrelated_entity_drift",
                    "identity_preservation",
                    "edit_persistence",
                    "sequence_consistency",
                ),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
