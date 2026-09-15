"""Render Step 8 result tables straight from the run artefacts.

Every number in the Step 8 documents is produced here rather than typed by hand, so a
table and the JSON it claims to summarise cannot drift apart. A missing file is reported
as missing rather than silently omitted.

Every table names its placement condition. A relational number without one is not
interpretable, which is the lesson Step 7 paid for.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["load", "principal_table", "split_table", "lod_table", "ablation_table", "main"]

ARM_ORDER: tuple[str, ...] = ("A0", "A1", "A1M", "A3Lite", "A3L", "A3", "A4")

GRAPH_KIND: Mapping[str, str] = {
    "A0": "none",
    "A1": "none",
    "A1M": "none",
    "A3Lite": "untyped",
    "A3L": "reduced typed",
    "A3": "full typed",
    "A4": "untyped",
}


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
    return [_row(first, list(columns)), "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"]


def _spread(entry: Mapping[str, Any]) -> str:
    mean = _cell(entry.get("mean"))
    if mean == "n/a":
        return "n/a"
    if float(entry.get("defined_n", 1)) > 1:
        return f"{mean} ± {entry['std']:.3f}"
    return mean


def _graph_parameters(counts: Mapping[str, Any], arm: str) -> int:
    groups = counts.get(arm, {})
    if not isinstance(groups, Mapping):
        return 0
    return int(groups.get("graph_encoder", 0)) + int(groups.get("untyped_graph", 0))


def principal_table(
    report: Mapping[str, Any],
    split: str,
    condition: str,
    metrics: Sequence[str],
    *,
    floor: float | None = None,
) -> str:
    """One row per arm for one split and placement condition."""
    aggregates = report["aggregates"][split][condition]
    counts = report["training"]["parameter_counts"]
    columns = ["params", "graph params", "graph", *metrics]
    if floor is not None:
        columns.append("gap above floor")
    lines = _header("arm", columns)
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        total = int(counts.get(arm, {}).get("total", 0))
        graph = _graph_parameters(counts, arm)
        cells = [f"{total:,}", f"{graph:,}", GRAPH_KIND.get(arm, "?")]
        cells.extend(
            _spread(aggregates[arm][m]) if m in aggregates[arm] else "n/a" for m in metrics
        )
        if floor is not None:
            entry = aggregates[arm].get("spatial_relation_accuracy")
            if entry and math.isfinite(float(entry["mean"])):
                headroom = 1.0 - floor
                gap = (float(entry["mean"]) - floor) / headroom if headroom > 1e-9 else float("nan")
                cells.append(f"{gap:+.3f}")
            else:
                cells.append("n/a")
        lines.append(_row(arm, cells))
    return "\n".join(lines)


def split_table(report: Mapping[str, Any], metric: str, condition: str) -> str:
    """One row per arm, one column per held-out split."""
    splits = list(report["aggregates"])
    lines = _header("arm", splits)
    for arm in ARM_ORDER:
        cells: list[str] = []
        present = False
        for split in splits:
            aggregates = report["aggregates"][split][condition]
            if arm in aggregates and metric in aggregates[arm]:
                present = True
                cells.append(_cell(aggregates[arm][metric]["mean"]))
            else:
                cells.append("n/a")
        if present:
            lines.append(_row(arm, cells))
    return "\n".join(lines)


def condition_table(report: Mapping[str, Any], split: str, metric: str) -> str:
    """Inferred against supplied placement, and the cost of inferring."""
    lines = _header("arm", ["placement inferred", "placement supplied (oracle)", "cost"])
    for arm in ARM_ORDER:
        inferred = report["aggregates"][split]["inferred"].get(arm, {}).get(metric)
        supplied = report["aggregates"][split]["supplied"].get(arm, {}).get(metric)
        if inferred is None and supplied is None:
            continue
        first = float(inferred["mean"]) if inferred else float("nan")
        second = float(supplied["mean"]) if supplied else float("nan")
        cost = second - first if math.isfinite(first) and math.isfinite(second) else float("nan")
        lines.append(_row(arm, [_cell(first), _cell(second), _cell(cost)]))
    return "\n".join(lines)


def lod_table(report: Mapping[str, Any], split: str, condition: str) -> str:
    """Level-of-detail reconstruction and the nesting properties."""
    metrics = (
        "lod1_iou",
        "lod2_iou",
        "lod3_iou",
        "lod_detail_gain",
        "lod_containment_mean",
        "lod_preservation_mean",
    )
    aggregates = report["aggregates"][split][condition]
    lines = _header("arm", list(metrics))
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        lines.append(
            _row(
                arm,
                [
                    _cell(aggregates[arm][m]["mean"]) if m in aggregates[arm] else "n/a"
                    for m in metrics
                ],
            )
        )
    return "\n".join(lines)


def frame_group_table(report: Mapping[str, Any], split: str, condition: str) -> str:
    """Position error broken down by entity group."""
    groups = ("chambers", "valves", "vessels", "septa", "walls")
    aggregates = report["aggregates"][split][condition]
    lines = _header("arm", ["pooled", *groups])
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        cells = [_cell(aggregates[arm].get("position_error", {}).get("mean"))]
        cells.extend(
            _cell(aggregates[arm].get(f"{group}_position_error", {}).get("mean"))
            for group in groups
        )
        lines.append(_row(arm, cells))
    return "\n".join(lines)


def ablation_table(report: Mapping[str, Any], arm: str, metric: str) -> str:
    """One row per ablation, with its change from the predicted-placement control."""
    arms = report["arms"]
    if arm not in arms:
        return f"_No ablation data for {arm}._"
    labels = report["ablations"]
    lines = _header("ablation", [metric, "change from control", "what it isolates"])
    for name, values in arms[arm].items():
        delta = values.get(f"delta_{metric}")
        lines.append(
            _row(
                f"`{name}`",
                [
                    _cell(values.get(metric)),
                    "n/a" if delta is None else f"{delta:+.4f}",
                    str(labels.get(name, "")),
                ],
            )
        )
    return "\n".join(lines)


def entity_axis_table(report: Mapping[str, Any], split: str, condition: str) -> str:
    """The entity-axis ladder under one placement condition."""
    metrics = ("entity_iou_mean", "part_control_success", "scene_iou", "position_error")
    lines = _header("arm", ["what it has", *metrics])
    for run in report["runs"]:
        values = run["splits"][split][condition]
        lines.append(
            _row(
                str(run["arm"]),
                [str(run.get("ladder_note", "")), *[_cell(values.get(m)) for m in metrics]],
            )
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print every table that has data behind it."""
    parser = argparse.ArgumentParser(description="Render Step 8 result tables.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args(argv)

    suite = load(args.runs / "step8-suite" / "step8_report.json")
    ablations = load(args.runs / "step8-ablations" / "ablation_report.json")
    entity = load(args.runs / "step8-entity-axis" / "entity_axis_report.json")
    floors = load(args.runs / "step8-relation-floor.json")

    if suite is None:
        print("_Step 8 suite report not produced._")
        return 0

    headline = (
        "entity_iou_mean",
        "part_control_success",
        "position_error",
        "composite_frame_error",
        "spatial_relation_accuracy",
        "scene_iou",
    )
    for split in suite["aggregates"]:
        floor = None
        if floors is not None and split in floors["splits"]:
            floor = float(floors["splits"][split]["relation_blind_floor"])
        print(f"\n## {split}, placement inferred\n")
        if floor is not None:
            print(f"Relation-blind floor for this split: {floor:.4f}. Oracle: 1.0000.\n")
        print(principal_table(suite, split, "inferred", headline, floor=floor))

    print("\n## Entity ownership IoU across splits, placement inferred\n")
    print(split_table(suite, "entity_iou_mean", "inferred"))
    print("\n## Spatial relation accuracy across splits, placement inferred\n")
    print(split_table(suite, "spatial_relation_accuracy", "inferred"))
    print("\n## Cost of inferring placement, test_seen\n")
    print(condition_table(suite, "test_seen", "entity_iou_mean"))
    print("\n## Level of detail, test_seen, placement inferred\n")
    print(lod_table(suite, "test_seen", "inferred"))
    print("\n## Position error by entity group, test_seen, placement inferred\n")
    print(frame_group_table(suite, "test_seen", "inferred"))

    if ablations is not None:
        for arm in ("A3Lite", "A3"):
            if arm in ablations["arms"]:
                print(f"\n## Ablations, {arm}, entity ownership IoU\n")
                print(ablation_table(ablations, arm, "entity_iou_mean"))

    if entity is not None:
        print("\n## Entity-axis ladder, test_seen, placement inferred\n")
        print(entity_axis_table(entity, "test_seen", "inferred"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
