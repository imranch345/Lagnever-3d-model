"""Generate the Step 8 results sections from the run artefacts.

Section 31 of the Step 8 brief requires that headline numbers be generated rather than
typed. This module emits the results, ablation, level-of-detail, entity-axis, convergence
and decision sections as markdown, reading only from `experiments/runs/`.

The prose that surrounds a table is written by hand; the numbers inside it are not.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.step8.report_tables import (
    ablation_table,
    condition_table,
    entity_axis_table,
    frame_group_table,
    load,
    lod_table,
    principal_table,
    split_table,
)

__all__ = ["build_results_sections", "main"]

HEADLINE: tuple[str, ...] = (
    "entity_iou_mean",
    "part_control_success",
    "position_error",
    "composite_frame_error",
    "spatial_relation_accuracy",
    "scene_iou",
)


def _floor_for(floors: Mapping[str, Any] | None, split: str) -> float | None:
    if floors is None or split not in floors["splits"]:
        return None
    return float(floors["splits"][split]["relation_blind_floor"])


def _gap(value: float, floor: float) -> float:
    headroom = 1.0 - floor
    return (value - floor) / headroom if headroom > 1e-9 else float("nan")


def _relation_verdict(
    suite: Mapping[str, Any], floors: Mapping[str, Any] | None
) -> list[str]:
    """One line per arm per split: above the floor, or below it."""
    lines = [
        "| arm | " + " | ".join(suite["aggregates"]) + " |",
        "| --- | " + " | ".join("---" for _ in suite["aggregates"]) + " |",
    ]
    arms = ["A1", "A1M", "A3Lite", "A3L", "A3"]
    for arm in arms:
        cells: list[str] = []
        present = False
        for split in suite["aggregates"]:
            entry = suite["aggregates"][split]["inferred"].get(arm, {}).get(
                "spatial_relation_accuracy"
            )
            floor = _floor_for(floors, split)
            if entry is None or floor is None or not math.isfinite(float(entry["mean"])):
                cells.append("n/a")
                continue
            present = True
            gap = _gap(float(entry["mean"]), floor)
            mark = "above" if gap > 0 else "**below**"
            cells.append(f"{float(entry['mean']):.4f} ({gap:+.3f}, {mark})")
        if present:
            lines.append("| " + " | ".join([arm, *cells]) + " |")
    return lines


def build_results_sections(runs_dir: Path) -> str:
    """Emit every results section as markdown."""
    suite = load(runs_dir / "step8-suite" / "step8_report.json")
    floors = load(runs_dir / "step8-relation-floor.json")
    ablations = load(runs_dir / "step8-ablations" / "ablation_report.json")
    entity = load(runs_dir / "step8-entity-axis" / "entity_axis_report.json")
    convergence = load(runs_dir / "step8-convergence.json")

    out: list[str] = []
    if suite is None:
        return "_The Step 8 suite report has not been produced, so no results section exists._"

    training = suite["training"]
    out.append("## 17. Results\n")
    out.append(
        f"Generated from `experiments/runs/step8-suite/step8_report.json`. "
        f"{len(training['seeds'])} seeds, {training['steps']} steps, "
        f"{training['device']}. Headline condition: **{training['headline_condition']}**.\n"
    )

    for split in suite["aggregates"]:
        floor = _floor_for(floors, split)
        out.append(f"\n### {split}, placement inferred\n")
        if floor is not None:
            out.append(
                f"Relation-blind floor for this split: **{floor:.4f}**. Oracle: 1.0000. "
                "A model below the floor has no relational competence on that metric.\n"
            )
        out.append(principal_table(suite, split, "inferred", HEADLINE, floor=floor))
        out.append("")

    out.append("\n### The required summary table\n")
    out.append(
        "One row per model on `test_seen`, the split with held-out families and "
        "in-distribution arrangements. The oracle row is the same weights with true "
        "frames supplied, and is an upper bound rather than a result.\n"
    )
    out.append(_summary_table(suite, _floor_for(floors, "test_seen")))

    out.append("\n### Relation accuracy against the floor, every split\n")
    out.append(
        "Each cell is the accuracy, then the share of the available headroom taken. "
        "Negative means a fixed arrangement-blind guess would have done better.\n"
    )
    out.extend(_relation_verdict(suite, floors))

    out.append("\n### Entity ownership IoU across splits, placement inferred\n")
    out.append(split_table(suite, "entity_iou_mean", "inferred"))

    out.append("\n### The cost of inferring placement (test_seen)\n")
    out.append(
        "The supplied column is an oracle upper bound, not a result. It says what better "
        "placement could buy.\n"
    )
    out.append(condition_table(suite, "test_seen", "entity_iou_mean"))

    out.append("\n### Position error by entity group, test_seen, placement inferred\n")
    out.append(frame_group_table(suite, "test_seen", "inferred"))

    out.append("\n## 13. Level-of-detail experiments\n")
    out.append("Placement inferred, test_seen.\n")
    out.append(lod_table(suite, "test_seen", "inferred"))

    density = load(runs_dir / "step8-lod-density.json")
    if density is not None:
        levels = density["splits"]["test_seen"]["density"]
        out.append(
            "\nRaw IoU is not comparable across levels. A finer level exposes more "
            "entities, so its target covers more of the space, and a model that calls "
            "every point occupied scores an IoU equal to the density:\n"
        )
        out.append("| level | target density, which is the trivial predictor's IoU |")
        out.append("| --- | ---: |")
        for level in ("1", "2", "3"):
            out.append(f"| {level} | {float(levels[level]):.4f} |")
        out.append("\nCorrected for density, `(iou - density) / (1 - density)`:\n")
        out.append(_corrected_lod_table(suite, levels))

    if ablations is not None:
        out.append("\n### R10 nested against R11 non-nested\n")
        out.append(
            "`R11` decodes each level from a **disjoint** token slice rather than a "
            "prefix, which is the closest approximation to three unrelated "
            "representations these weights can express. It is an approximation, not a "
            "trained control.\n"
        )
        out.append(_nested_comparison_table(ablations))

    if ablations is not None:
        out.append("\n## 21. Ablations\n")
        out.append(
            f"Seed {ablations['seed']}, split `{ablations['dataset']['split']}`, "
            f"{ablations['dataset']['scenes']} scenes.\n"
        )
        for arm in ("A3Lite", "A3L", "A3", "A1M", "A1"):
            if arm in ablations["arms"]:
                out.append(f"\n### {arm}\n")
                out.append(ablation_table(ablations, arm, "entity_iou_mean"))
    else:
        out.append("\n## 21. Ablations\n\n_Not produced._")

    if entity is not None:
        out.append("\n## 14. Entity-axis experiments\n")
        out.append("Placement inferred, test_seen.\n")
        out.append(entity_axis_table(entity, "test_seen", "inferred"))
        out.append("\nThe same ladder with placement supplied, as an oracle bound:\n")
        out.append(entity_axis_table(entity, "test_seen", "supplied"))
    else:
        out.append("\n## 14. Entity-axis experiments\n\n_Not produced._")

    if convergence is not None:
        out.append("\n## 20. Convergence analysis\n")
        criterion = convergence["criterion"]
        out.append(
            f"Criterion: `{criterion['metric']}` under {criterion['condition']}, "
            f"tolerance {criterion['tolerance']}, failure floor {criterion['failure_floor']}.\n"
        )
        out.append(
            f"**converged {convergence['converged']}, not converged "
            f"{convergence['not_converged']}, failed {convergence['failed']}** "
            f"of {len(convergence['runs'])} runs.\n"
        )
        out.append("| run | verdict | reason |")
        out.append("| --- | --- | --- |")
        for name, record in convergence["runs"].items():
            out.append(f"| `{name}` | {record['verdict']} | {record['reason']} |")
        out.append("\n### Direction of travel, first validation to last\n")
        out.append(_direction_table(convergence))
    return "\n".join(out)


def _summary_table(suite: Mapping[str, Any], floor: float | None) -> str:
    """The table the Step 8 brief asks for, on test_seen."""
    aggregates = suite["aggregates"]["test_seen"]
    counts = suite["training"]["parameter_counts"]
    graph_kind = {
        "A0": "none",
        "A1": "none",
        "A1M": "none",
        "A3Lite": "untyped",
        "A3L": "reduced typed",
        "A3": "full typed",
    }
    lines = [
        "| Model | Params | Placement | Graph | Entity IoU | Part Control | Frame Error "
        "| Relation Accuracy | Relation-Blind Gap | Scene IoU |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def row(arm: str, condition: str, label: str) -> str | None:
        values = aggregates[condition].get(arm)
        if not values:
            return None
        total = int(counts.get(arm, {}).get("total", 0))
        gap = "n/a"
        if floor is not None and "spatial_relation_accuracy" in values:
            accuracy = float(values["spatial_relation_accuracy"]["mean"])
            if math.isfinite(accuracy):
                gap = f"{_gap(accuracy, floor):+.3f}"
        cells = [
            label,
            f"{total:,}",
            "inferred" if condition == "inferred" else "supplied",
            graph_kind.get(arm, "?"),
        ]
        for metric in (
            "entity_iou_mean",
            "part_control_success",
            "composite_frame_error",
            "spatial_relation_accuracy",
        ):
            entry = values.get(metric)
            cells.append("n/a" if entry is None else f"{float(entry['mean']):.4f}")
        cells.append(gap)
        entry = values.get("scene_iou")
        cells.append("n/a" if entry is None else f"{float(entry['mean']):.4f}")
        return "| " + " | ".join(cells) + " |"

    for arm in ("A0", "A1", "A1M", "A3Lite", "A3L", "A3"):
        line = row(arm, "inferred", arm)
        if line:
            lines.append(line)
    best = row("A3", "supplied", "Oracle frame control (A3)")
    if best:
        lines.append(best)
    return "\n".join(lines)


def _corrected_lod_table(suite: Mapping[str, Any], density: Mapping[str, Any]) -> str:
    """Per-level IoU expressed as the share of available headroom taken."""
    aggregates = suite["aggregates"]["test_seen"]["inferred"]
    lines = [
        "| arm | level 1 | level 2 | level 3 | corrected detail gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("A1", "A1M", "A3Lite", "A3L", "A3"):
        if arm not in aggregates:
            continue
        corrected: list[float] = []
        for level in (1, 2, 3):
            key = f"lod{level}_iou"
            if key not in aggregates[arm]:
                corrected.append(float("nan"))
                continue
            value = float(aggregates[arm][key]["mean"])
            floor = float(density[str(level)])
            headroom = 1.0 - floor
            corrected.append((value - floor) / headroom if headroom > 1e-9 else float("nan"))
        gain = corrected[-1] - corrected[0]
        cells = " | ".join(f"{value:.4f}" for value in corrected)
        lines.append(f"| {arm} | {cells} | {gain:+.4f} |")
    return "\n".join(lines)


def _nested_comparison_table(ablations: Mapping[str, Any]) -> str:
    """Nested prefixes against disjoint slices, on the same trained weights."""
    lines = [
        "| arm | nested gain | non-nested gain | nested preservation "
        "| non-nested preservation | nested containment | non-nested containment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, values in ablations["arms"].items():
        nested = values.get("R10_nested_lod", {})
        other = values.get("R11_non_nested_lod", {})
        if not nested or not other:
            continue

        def cell(source: Mapping[str, Any], key: str) -> str:
            value = source.get(key)
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    cell(nested, "lod_detail_gain_over_trivial"),
                    cell(other, "lod_detail_gain_over_trivial"),
                    cell(nested, "lod_preservation_mean"),
                    cell(other, "lod_preservation_mean"),
                    cell(nested, "lod_containment_mean"),
                    cell(other, "lod_containment_mean"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _lod_densities(suite: Mapping[str, Any]) -> dict[int, float]:
    """Per-level target density, if the runs recorded it."""
    out: dict[int, float] = {}
    for split_values in suite["aggregates"].values():
        for arm_values in split_values["inferred"].values():
            for level in (1, 2, 3):
                key = f"lod{level}_target_density"
                if key in arm_values and math.isfinite(float(arm_values[key]["mean"])):
                    out[level] = float(arm_values[key]["mean"])
    return out


def _direction_table(convergence: Mapping[str, Any]) -> str:
    """Which metrics improved over training, pooled across runs."""
    metrics: dict[str, list[bool]] = {}
    changes: dict[str, list[float]] = {}
    for record in convergence["runs"].values():
        for metric, values in record.get("direction_of_travel", {}).items():
            metrics.setdefault(metric, []).append(bool(values["improved"]))
            changes.setdefault(metric, []).append(float(values["change"]))
    lines = ["| metric | runs that improved | mean change |", "| --- | ---: | ---: |"]
    for metric in sorted(metrics):
        improved = sum(metrics[metric])
        total = len(metrics[metric])
        mean = sum(changes[metric]) / max(total, 1)
        lines.append(f"| `{metric}` | {improved} of {total} | {mean:+.4f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate Step 8 results sections.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    text = build_results_sections(args.runs)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
