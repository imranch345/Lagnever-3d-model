"""Generate the Step 9 results sections from the run artefacts.

Every headline number in the Step 9 documents is produced here, from JSON, rather than
typed. Missing artefacts are reported as missing rather than omitted.

Two reporting rules are enforced structurally rather than remembered:

* a translation error is always printed beside its **placement-blind floor** and the gap
  above it, because a frame error read against zero means nothing;
* the frame decomposition always shows the rotation term, so its being structurally zero
  is visible instead of folded into a composite.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["load", "baseline_table", "variant_table", "floor_table", "main"]

ARM_ORDER: tuple[str, ...] = ("A1", "A1M", "A3Lite", "A3L", "A3")
VARIANT_ORDER: tuple[str, ...] = ("S0_control", "S1_scene_context", "S2_euclidean", "S3_both")


def load(path: Path) -> Mapping[str, Any] | None:
    """Read a run report, or return None when it has not been produced."""
    if not path.exists():
        return None
    payload: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _cell(value: float | None, places: int = 4, sign: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.{places}f}" if sign else f"{value:.{places}f}"


def _spread(entry: Mapping[str, Any] | None, sign: bool = False) -> str:
    if entry is None:
        return "n/a"
    mean = _cell(entry.get("mean"), sign=sign)
    if mean == "n/a" or float(entry.get("n", 1)) <= 1:
        return mean
    return f"{mean} ± {entry['std']:.4f}"


def _rows(header: Sequence[str]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]


def floor_table(report: Mapping[str, Any]) -> str:
    """The placement-blind floor on every split, with the weaker blind predictor beside it."""
    floors = report["placement_floor"]["splits"]
    lines = _rows(["split", "global mean", "**floor** (per-entity)", "scale at floor", "oracle"])
    for split, values in floors.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{split}`",
                    _cell(values["global"]["position_error"]),
                    f"**{values['placement_blind_floor']:.4f}**",
                    _cell(values["per_entity"]["scale_error"]),
                    "0.0000",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def baseline_table(report: Mapping[str, Any], split: str) -> str:
    """The Step 8 arms re-scored against the floor, inferred placement."""
    aggregates = report["aggregates"]
    lines = _rows(
        [
            "arm",
            "translation",
            "gap above floor",
            "scale",
            "rotation",
            "Step 8 composite",
            "placement (corrected)",
            "entity IoU",
            "oracle gap",
        ]
    )
    for arm in ARM_ORDER:
        if arm not in aggregates:
            continue
        values = aggregates[arm][split]["inferred"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    _spread(values.get("translation_error")),
                    _spread(values.get("placement_floor_gap"), sign=True),
                    _spread(values.get("scale_error")),
                    _cell(values.get("rotation_error", {}).get("mean"), places=6),
                    _spread(values.get("composite_frame_error")),
                    _spread(values.get("placement_error")),
                    _spread(values.get("entity_iou_mean")),
                    _spread(values.get("oracle_gap_entity_iou")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def variant_table(report: Mapping[str, Any], split: str) -> str:
    """The Step 9 variants, inferred placement, with their seed spreads."""
    aggregates = report["aggregates"]
    lines = _rows(
        [
            "variant",
            "seeds",
            "translation",
            "gap above floor",
            "entity IoU",
            "part control",
            "scene IoU",
        ]
    )
    for variant in VARIANT_ORDER:
        if variant not in aggregates:
            continue
        values = aggregates[variant][split]
        seeds = int(float(values.get("translation_error", {}).get("n", 0)))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{variant}`",
                    str(seeds),
                    _spread(values.get("translation_error")),
                    _spread(values.get("placement_floor_gap"), sign=True),
                    _spread(values.get("entity_iou_mean")),
                    _spread(values.get("part_control_success")),
                    _spread(values.get("scene_iou")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def variant_split_table(report: Mapping[str, Any], metric: str) -> str:
    """One row per variant, one column per held-out split."""
    aggregates = report["aggregates"]
    splits = list(next(iter(aggregates.values())))
    lines = _rows(["variant", *splits])
    for variant in VARIANT_ORDER:
        if variant not in aggregates:
            continue
        cells = [
            _spread(aggregates[variant][split].get(metric)) for split in splits
        ]
        lines.append("| " + " | ".join([f"`{variant}`", *cells]) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print every table that has data behind it."""
    parser = argparse.ArgumentParser(description="Render Step 9 result tables.")
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args(argv)

    baseline = load(args.runs / "step9-baseline" / "baseline_report.json")
    exploratory = load(args.runs / "step9-variants-exploratory" / "variants_report.json")
    confirmatory = load(args.runs / "step9-confirmatory" / "variants_report.json")

    if baseline is not None:
        print("## The placement-blind floor\n")
        print(floor_table(baseline))
        print("\n## Baseline: the Step 8 models against the floor, `test_seen`, inferred\n")
        print(baseline_table(baseline, "test_seen"))
        print(f"\nRotation: {baseline['rotation_note']}\n")
    else:
        print("_Step 9 baseline not produced._")

    for name, report in (("Exploratory, 1 seed", exploratory), ("Confirmatory", confirmatory)):
        if report is None:
            print(f"\n_{name} suite not produced._")
            continue
        print(f"\n## {name}: `test_seen`, inferred placement\n")
        print(variant_table(report, "test_seen"))
        print(f"\n### {name}: translation error across splits\n")
        print(variant_split_table(report, "translation_error"))
        print(f"\n### {name}: entity ownership IoU across splits\n")
        print(variant_split_table(report, "entity_iou_mean"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
