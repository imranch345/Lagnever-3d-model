"""Step 10 §29: the required result table, built from the run JSON rather than by hand.

Transcribing numbers into a report is where they get rounded, mixed between conditions, or
quietly taken from the wrong split. Everything here reads `placement_report.json` and
`placement_floor.json` and emits Markdown, so the report and the artefact cannot disagree.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["placement_table", "depth_table", "floor_table", "main"]


def _cell(
    aggregates: dict[str, Any], key: str, split: str, metric: str
) -> tuple[float, float] | None:
    entry = aggregates.get(key, {}).get(split, {}).get(metric)
    if entry is None:
        return None
    return float(entry["mean"]), float(entry["spread"])


def placement_table(report: dict[str, Any], *, split: str = "test_seen") -> str:
    """Translation error per arm and cell, against the floor, with seed spread."""
    aggregates = report["aggregates"]
    arms = list(dict.fromkeys(str(run["arm"]) for run in report["runs"]))
    cells = list(dict.fromkeys(str(run["cell"]) for run in report["runs"]))
    floor = float(report["placement_floor"]["splits"][split]["global"]["position_error"])

    lines = [
        f"Split `{split}`. Placement inferred. Floor = {floor:.4f} (identity-only lookup).",
        "",
        "| arm | " + " | ".join(cells) + " | best cell | beats floor |",
        "| --- | " + " | ".join("---" for _ in cells) + " | --- | --- |",
    ]
    for arm in arms:
        values: list[str] = []
        scored: dict[str, float] = {}
        for cell in cells:
            found = _cell(aggregates, f"{arm}/{cell}", split, "translation_error")
            if found is None:
                values.append("—")
                continue
            mean, spread = found
            scored[cell] = mean
            values.append(f"{mean:.4f} ± {spread:.4f}" if spread else f"{mean:.4f}")
        if not scored:
            continue
        best = min(scored, key=lambda name: scored[name])
        beats = "yes" if scored[best] < floor else "no"
        lines.append(f"| {arm} | " + " | ".join(values) + f" | {best} | {beats} |")

    lines += [
        "",
        "Lower is better. A value above the floor means the arm places worse than a lookup "
        "table keyed on identity alone.",
    ]
    return "\n".join(lines)


def floor_table(floor: dict[str, Any]) -> str:
    """What each target formulation costs a blind predictor."""
    lines = [
        "| split | global | parent-relative | with true parent | reproduces Step 9 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split, entry in floor["splits"].items():
        base = entry["global"]["position_error"]
        blind = entry["parent_relative"]["position_error"]
        oracle = entry["parent_relative_oracle"]["position_error"]
        lines.append(
            f"| {split} | {base:.4f} | {blind:.4f} ({entry['target_change_blind']:+.1%}) "
            f"| {oracle:.4f} ({entry['target_change_with_true_parent']:+.1%}) "
            f"| {'yes' if entry.get('reproduces_step9') else 'NO'} |"
        )
    return "\n".join(lines)


def depth_table(depths: dict[str, Any], *, split: str = "test_seen") -> str:
    """Error by hierarchy depth; depth 0 is the built-in control."""
    entry = depths["splits"][split]
    lines = [
        f"Split `{split}`, hierarchy `{depths['hierarchy']}`.",
        "",
        "| depth | entities scored | position | scale | rotation (rad) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for depth, values in entry.items():
        lines.append(
            f"| {depth} | {values['entities_scored']} | {values['position_error']:.4f} "
            f"| {values['scale_error']:.4f} | {values['rotation_error']:.6f} |"
        )
    lines += [
        "",
        "Depth 0 entities are roots and are predicted identically in every cell, so a "
        "difference there between two cells is drift, not hierarchy.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 report tables.")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--floor", type=Path, default=Path("experiments/step10/placement_floor.json")
    )
    parser.add_argument("--depths", type=Path, default=None)
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args(argv)

    if args.floor.exists():
        print("## Placement-blind floor\n")
        print(floor_table(json.loads(args.floor.read_text(encoding="utf-8"))))
        print()
    if args.report is not None and args.report.exists():
        print("## Placement by arm and cell\n")
        payload = json.loads(args.report.read_text(encoding="utf-8"))
        print(placement_table(payload, split=args.split))
        print()
    if args.depths is not None and args.depths.exists():
        print("## Error by hierarchy depth\n")
        print(depth_table(json.loads(args.depths.read_text(encoding="utf-8")), split=args.split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
