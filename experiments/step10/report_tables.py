"""Step 10 §29: every result table, built from the run JSON rather than by hand.

Transcribing numbers into a report is where they get rounded, mixed between conditions, or
quietly taken from the wrong split. Everything here reads ``placement_report.json`` (and the
floor it embeds) and emits Markdown, so the report and the artefact cannot disagree. The
completion report pastes this output; it does not retype it.

    python -m experiments.step10.report_tables --all-splits
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "compute_table",
    "inheritance_table",
    "control_table",
    "depth_table",
    "floor_table",
    "integrity_table",
    "main",
    "main_table",
    "paired_table",
    "reproduction_table",
]

DEPTHS: tuple[str, ...] = ("0", "1", "2", "3")
BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"
TARGET_LABEL = {"global": "global", "parent_relative": "parent-relative"}


def _f(value: float, digits: int = 4) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else (
        f"{value:.{digits}f}"
    )


def _signed(value: float, digits: int = 4) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:+.{digits}f}"


def _cells(report: Mapping[str, Any]) -> list[str]:
    return list(report["cells"])


def floor_table(report: Mapping[str, Any]) -> str:
    """What each target formulation costs a blind predictor."""
    lines = [
        "| split | floor (global) | blind parent-relative | with true parent (diagnostic) "
        "| reproduces Step 9 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split, entry in report["placement_floor"]["splits"].items():
        base = entry["global"]["position_error"]
        blind = entry["parent_relative"]["position_error"]
        oracle = entry["parent_relative_oracle"]["position_error"]
        lines.append(
            f"| {split} | {base:.4f} | {blind:.4f} ({entry['target_change_blind']:+.1%}) "
            f"| {oracle:.4f} ({entry['target_change_with_true_parent']:+.1%}) "
            f"| {'yes' if entry.get('reproduces_step9') else 'NO'} |"
        )
    return "\n".join(lines)


def main_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """The §13 table: one row per arm and cell, seed mean and spread, floor, depth."""
    floor = float(report["placement_floor"]["splits"][split]["global"]["position_error"])
    aggregates = report["aggregates"]
    depths = report["depth_aggregates"]
    lines = [
        f"Split `{split}`, inferred placement, global-frame translation error (lower is "
        f"better). Floor = {floor:.4f}. Margin = floor − mean; positive beats the floor. "
        "Depth columns are position error by depth in the spatial tree, seed mean.",
        "",
        "| arm | cell | target | hierarchy | seeds | mean translation | seed std | floor "
        "| floor margin | depth 0 | depth 1 | depth 2 | depth 3 | parameters | steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
        "| --- |",
    ]
    for arm in report["arms"]:
        for cell in _cells(report):
            key = f"{arm}/{cell}"
            if key not in aggregates:
                continue
            entry = aggregates[key]
            translation = entry[split]["translation_error"]
            by_depth = depths[key][split]
            depth_values = [
                _f(by_depth[d]["position_error"]["mean"]) if d in by_depth else "—"
                for d in DEPTHS
            ]
            lines.append(
                f"| {arm} | {cell} | {TARGET_LABEL.get(entry['placement_target'], '?')} "
                f"| {entry['hierarchy'] if entry['placement_target'] != 'global' else '—'} "
                f"| {len(entry['seeds'])} | {translation['mean']:.4f} | {translation['std']:.4f} "
                f"| {floor:.4f} | {_signed(floor - translation['mean'])} "
                f"| {' | '.join(depth_values)} "
                f"| {entry['parameters']:,} | {entry['training_steps']} |"
            )
    return "\n".join(lines)


def paired_table(
    report: Mapping[str, Any], *, split: str = "test_seen", treatment: str = "T1_spatial"
) -> str:
    """Treatment minus control, paired by seed; negative means the treatment placed better."""
    paired = report["paired"][split]
    seeds = [str(seed) for seed in report["seeds"]]
    lines = [
        f"Split `{split}`: `{treatment}` − `T0_global`, paired by seed. Negative = "
        "parent-relative placed better. Rule fixed before results: exceeds seed variability "
        "only if every seed agrees in sign AND |t| > 4.303 (df = 2).",
        "",
        "| arm | scope | "
        + " | ".join(f"seed {seed}" for seed in seeds)
        + " | mean | std | t | same sign | exceeds seed variability |",
        "| --- | --- | " + " | ".join("---" for _ in seeds) + " | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        key = f"{arm}/{treatment}-vs-T0_global"
        if key not in paired:
            continue
        rows = [("all entities", paired[key]["translation_error"])]
        by_depth = paired[key]["by_depth"]
        rows += [(f"depth {d}", by_depth[d]) for d in DEPTHS if d in by_depth]
        for scope, entry in rows:
            per_seed = [_signed(entry["per_seed"].get(seed, float("nan"))) for seed in seeds]
            lines.append(
                f"| {arm} | {scope} | {' | '.join(per_seed)} | {_signed(entry['mean_difference'])} "
                f"| {_f(entry['std_difference'])} | {_signed(entry['t_statistic'], 2)} "
                f"| {'yes' if entry['all_seeds_same_sign'] else 'no'} "
                f"| {'YES' if entry['exceeds_seed_variability'] else 'no'} |"
            )
    return "\n".join(lines)


def depth_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """Position and scale error by depth for global and parent-relative, seed mean ± std."""
    depths = report["depth_aggregates"]
    lines = [
        f"Split `{split}`. Depth from the spatial tree for every cell. Position error, then "
        "scale error, each seed mean ± std.",
        "",
        "| arm | cell | depth | entities | position | scale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in ("T0_global", "T1_spatial"):
            key = f"{arm}/{cell}"
            if key not in depths:
                continue
            for depth, entry in depths[key][split].items():
                lines.append(
                    f"| {arm} | {cell} | {depth} | {entry['entities_scored']} "
                    f"| {entry['position_error']['mean']:.4f} "
                    f"± {entry['position_error']['std']:.4f} "
                    f"| {entry['scale_error']['mean']:.4f} ± {entry['scale_error']['std']:.4f} |"
                )
    return "\n".join(lines)


def control_table(report: Mapping[str, Any]) -> str:
    """The taxonomic control against the global control, per arm and seed."""
    lines = [
        "| arm / seed | max abs difference, all splits | bit-identical |",
        "| --- | --- | --- |",
    ]
    for key, entry in sorted(report["control_identity"].items()):
        lines.append(
            f"| {key} | {entry['max_abs_difference']:.3e} "
            f"| {'yes' if entry['identical'] else 'NO'} |"
        )
    return "\n".join(lines)


def reproduction_table(report: Mapping[str, Any]) -> str:
    """Each T0_global run against the Step 9 baseline for the same arm and seed."""
    entries = {k: v for k, v in report["step9_reproduction"].items() if k != "available"}
    lines = [
        "| arm / seed | Step 10 T0_global | Step 9 baseline | abs difference | identical |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, entry in sorted(entries.items()):
        lines.append(
            f"| {key} | {entry['step10']:.6f} | {entry['step9']:.6f} "
            f"| {entry['abs_difference']:.3e} | {'yes' if entry['identical'] else 'NO'} |"
        )
    return "\n".join(lines)


def compute_table(report: Mapping[str, Any]) -> str:
    """Capacity and compute per arm and cell: the same budget for every comparison."""
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for run in report["runs"]:
        by_key.setdefault(f"{run['arm']}/{run['cell']}", []).append(run)
    lines = [
        "| arm | cell | parameters | steps | batch | torch threads | seeds "
        "| train s (mean) | eval s (mean) | depth s (mean) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in _cells(report):
            runs = by_key.get(f"{arm}/{cell}")
            if not runs:
                continue

            def mean(field: str, chosen: Sequence[Mapping[str, Any]] = runs) -> float:
                return sum(float(run["timing"][field]) for run in chosen) / len(chosen)

            protocol = runs[0]["protocol"]
            lines.append(
                f"| {arm} | {cell} | {int(runs[0]['parameters']['total']):,} "
                f"| {protocol['steps']} | {protocol['batch_size']} | {protocol['torch_threads']} "
                f"| {','.join(str(run['seed']) for run in sorted(runs, key=lambda r: r['seed']))} "
                f"| {mean('train_seconds'):.0f} | {mean('eval_seconds'):.0f} "
                f"| {mean('depth_seconds'):.0f} |"
            )
    return "\n".join(lines)


def integrity_table(report: Mapping[str, Any]) -> str:
    """The §27 gate's outcome across every run and every evaluated split."""
    runs = report["runs"]
    composed = [run for run in runs if run["placement_target"] == "parent_relative"]
    checks = [run["integrity"]["train"] for run in runs] + [
        entry for run in runs for entry in run["integrity"]["splits"].values()
    ]
    round_trip = [
        max(entry.get("round_trip_error", 0.0), entry.get("probe_round_trip_error", 0.0))
        for entry in checks
    ]
    recon = [max(run["depth_reconciliation"].values()) for run in runs]
    rotation = max(entry["max_rotation_deviation_from_identity"] for entry in checks)
    lines = [
        "| check | result |",
        "| --- | --- |",
        f"| runs that passed the gate before training | {len(runs)} of {len(runs)} |",
        f"| split-level gate passes | {sum(len(r['integrity']['splits']) for r in runs)} |",
        f"| runs composing a hierarchy | {len(composed)} |",
        f"| worst round-trip error, stored and probe-rotated | {max(round_trip):.2e} |",
        f"| worst depth-vs-headline reconciliation | {max(recon):.2e} |",
        f"| largest stored rotation deviation from identity | {rotation:.2e} |",
    ]
    return "\n".join(lines)


def inheritance_table(diagnostic: Mapping[str, Any]) -> str:
    """Post hoc: parented children recomposed with the true parent, or its true scale."""
    lines = [
        f"**Post hoc, exploratory** — added after the confirmatory results were known. Split "
        f"`{diagnostic['split']}`, translation error of parented children, seed mean. "
        "`true parent`: the child's predicted local frame composed onto its parent's true "
        "frame. `true parent scale`: onto its predicted parent with the scale replaced by "
        "the truth.",
        "",
        "| arm | depth | T0_global | T1 as trained | T1, true parent | T1, true parent scale |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, depths in diagnostic["summary"].items():
        for depth, entry in depths.items():
            lines.append(
                f"| {arm} | {depth} | {entry['T0_global']:.4f} | {entry['as_trained']:.4f} "
                f"| {entry['true_parent']:.4f} | {entry['true_parent_scale']:.4f} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: print every table for the completion report."""
    parser = argparse.ArgumentParser(description="Step 10 report tables.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments/runs/step10-placement/placement_report.json"),
    )
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument(
        "--insert-into",
        type=Path,
        default=None,
        help="replace the text between the generated-table markers in this Markdown file",
    )
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    splits = list(report["dataset"]["splits"]) if args.all_splits else [args.split]
    sections: list[tuple[str, str]] = [("Placement-blind floor", floor_table(report))]
    for split in splits:
        sections.append((f"§29 results — {split}", main_table(report, split=split)))
    for split in splits:
        sections.append((f"Paired: spatial vs global — {split}", paired_table(report, split=split)))
    sections += [
        ("Depth analysis", depth_table(report, split=splits[0])),
        ("Taxonomic control vs global control", control_table(report)),
        ("Step 9 reproduction by the global control", reproduction_table(report)),
        ("Compute and capacity", compute_table(report)),
        ("Integrity", integrity_table(report)),
    ]
    diagnostic_path = args.report.parent / "inheritance_diagnostic.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        sections.append(("Inheritance diagnostic (post hoc)", inheritance_table(diagnostic)))
    rendered = "\n".join(f"### {title}\n\n{body}\n" for title, body in sections)
    if args.insert_into is None:
        print(rendered)
        return 0
    document = args.insert_into.read_text(encoding="utf-8")
    if BEGIN not in document or END not in document:
        raise SystemExit(f"{args.insert_into} has no {BEGIN!r} ... {END!r} markers")
    head, rest = document.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    source = f"<!-- generated from {args.report} by experiments.step10.report_tables -->"
    args.insert_into.write_text(
        f"{head}{BEGIN}\n{source}\n\n{rendered}\n{END}{tail}", encoding="utf-8"
    )
    print(f"tables written into {args.insert_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
