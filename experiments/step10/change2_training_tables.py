"""Step 10 Change 2 §20: every results table, generated from the run artifacts.

Position, rotation and scale are kept in separate columns everywhere and never summed into
one score: a model that traded position away for rotation would look better under a composite
and worse under the thing anyone cares about. Each component is read against **this corpus's**
floor on **that split**, because the four splits' floors are not interchangeable.

    python -m experiments.step10.change2_training_tables --insert-into <report.md>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "aggregate_table",
    "compute_table",
    "depth_table",
    "main",
    "oracle_table",
    "paired_table",
    "per_seed_table",
    "propagation_table",
    "rotation_detail_table",
]

BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"
DEGREES = 180.0 / math.pi
DEPTHS: tuple[str, ...] = ("0", "1", "2", "3")


def _signed(value: float, digits: int = 4) -> str:
    return "—" if value is None or math.isnan(value) else f"{value:+.{digits}f}"


def _floors(report: Mapping[str, Any], split: str) -> dict[str, float]:
    entry = report["floors"]["splits"][split]["global"]
    return {
        "position": float(entry["position_error"]),
        "rotation_deg": float(entry["rotation_error"]) * DEGREES,
        "scale": float(entry["scale_error"]),
    }


def per_seed_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """§20: one row per arm, cell and seed, with every floor margin."""
    floors = _floors(report, split)
    lines = [
        f"Split `{split}`, inferred placement. Margin = floor − value: **positive beats the "
        f"floor**. Floors: position {floors['position']:.4f}, rotation "
        f"{floors['rotation_deg']:.2f}°, scale {floors['scale']:.4f}.",
        "",
        "| arm | cell | seed | position | rotation° | scale | position margin "
        "| rotation margin° | scale margin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in sorted(
        report["runs"], key=lambda r: (r["arm"], r["cell"], r["seed"])
    ):
        inferred = run["splits"][split]["inferred"]
        rotation = run["rotation"][split]
        position = float(inferred["translation_error"])
        scale = float(inferred["scale_error"])
        lines.append(
            f"| {run['arm']} | {run['cell']} | {run['seed']} | {position:.4f} "
            f"| {rotation['mean_deg']:.2f} | {scale:.4f} "
            f"| {_signed(floors['position'] - position)} "
            f"| {_signed(rotation['floor_margin_deg'], 2)} "
            f"| {_signed(floors['scale'] - scale)} |"
        )
    return "\n".join(lines)


def aggregate_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """§20: seed mean and spread per arm and cell."""
    floors = _floors(report, split)
    aggregates = report["aggregates"]
    lines = [
        f"Split `{split}`, three seeds each. Floors: position {floors['position']:.4f}, "
        f"rotation {floors['rotation_deg']:.2f}°, scale {floors['scale']:.4f}.",
        "",
        "| arm | cell | role | mean position | position std | mean rotation° | rotation std "
        "| mean scale | scale std | beats position floor | beats rotation floor |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in report["cells"]:
            key = f"{arm}/{cell}"
            if key not in aggregates:
                continue
            entry = aggregates[key]
            position = entry[split]["translation_error"]
            scale = entry[split]["scale_error"]
            rotation = entry[split]["rotation_mean_deg"]
            lines.append(
                f"| {arm} | {cell} | {entry['role']} | {position['mean']:.4f} "
                f"| {position['std']:.4f} | {rotation['mean']:.2f} | {rotation['std']:.2f} "
                f"| {scale['mean']:.4f} | {scale['std']:.4f} "
                f"| {'yes' if position['mean'] < floors['position'] else 'no'} "
                f"| {'yes' if rotation['mean'] < floors['rotation_deg'] else 'no'} |"
            )
    return "\n".join(lines)


def rotation_detail_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """§8: the rotation error distribution, not only its mean."""
    floors = _floors(report, split)
    aggregates = report["aggregates"]
    lines = [
        f"Split `{split}`, seed mean of each statistic. `below floor` is the share of "
        f"individual entities under the {floors['rotation_deg']:.2f}° floor — a model can sit "
        "on the floor on average while being better on most entities and far worse on a few.",
        "",
        "| arm | cell | mean° | median° | std° | within 5° | within 10° | within 20° "
        "| below floor | floor margin° |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in report["cells"]:
            key = f"{arm}/{cell}"
            if key not in aggregates:
                continue
            entry = aggregates[key][split]
            lines.append(
                f"| {arm} | {cell} | {entry['rotation_mean_deg']['mean']:.2f} "
                f"| {entry['rotation_median_deg']['mean']:.2f} "
                f"| {entry['rotation_std_deg']['mean']:.2f} "
                f"| {entry['rotation_within_5_deg']['mean']:.2f} "
                f"| {entry['rotation_within_10_deg']['mean']:.2f} "
                f"| {entry['rotation_within_20_deg']['mean']:.2f} "
                f"| {entry['rotation_below_floor']['mean']:.2f} "
                f"| {_signed(entry['rotation_floor_margin_deg']['mean'], 2)} |"
            )
    return "\n".join(lines)


def depth_table(report: Mapping[str, Any], *, split: str = "test_seen") -> str:
    """§14: position, rotation and scale by depth in the spatial tree."""
    depths = report["depth_aggregates"]
    lines = [
        f"Split `{split}`, seed mean. Depth is read from the spatial tree for every cell, so "
        "depth 0 names the same ten root entities everywhere. Rotation is in degrees.",
        "",
        "| arm | cell | depth | entities | position | rotation° | scale |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in report["cells"]:
            key = f"{arm}/{cell}"
            if key not in depths:
                continue
            for depth, entry in depths[key][split].items():
                lines.append(
                    f"| {arm} | {cell} | {depth} | {entry['entities_scored']} "
                    f"| {entry['position_error']['mean']:.4f} "
                    f"| {entry['rotation_error']['mean'] * DEGREES:.2f} "
                    f"| {entry['scale_error']['mean']:.4f} |"
                )
    return "\n".join(lines)


def paired_table(
    report: Mapping[str, Any], *, split: str = "test_seen", treatment: str = "T4_rigid"
) -> str:
    """Treatment minus the global cell, paired by seed, per component."""
    paired = report["paired"][split]
    seeds = [str(seed) for seed in report["seeds"]]
    lines = [
        f"Split `{split}`: `{treatment}` − `T0_global`, paired by seed. **Negative means the "
        "treatment was better.** Rule fixed before results: every seed agrees in sign AND "
        "|t| > 4.303.",
        "",
        "| arm | component | " + " | ".join(f"seed {s}" for s in seeds)
        + " | mean | t | same sign | exceeds seed variability |",
        "| --- | --- | " + " | ".join("---" for _ in seeds) + " | --- | --- | --- | --- |",
    ]
    labels = {
        "translation_error": "position",
        "rotation_deg": "rotation°",
        "scale_error": "scale",
    }
    for arm in report["arms"]:
        for metric, label in labels.items():
            key = f"{arm}/{treatment}-vs-T0_global/{metric}"
            if key not in paired:
                continue
            entry = paired[key]
            digits = 2 if metric == "rotation_deg" else 4
            per_seed = [
                _signed(entry["per_seed"].get(seed, float("nan")), digits) for seed in seeds
            ]
            lines.append(
                f"| {arm} | {label} | {' | '.join(per_seed)} "
                f"| {_signed(entry['mean_difference'], digits)} "
                f"| {_signed(entry['t_statistic'], 2)} "
                f"| {'yes' if entry['all_seeds_same_sign'] else 'no'} "
                f"| {'YES' if entry['exceeds_seed_variability'] else 'no'} |"
            )
    return "\n".join(lines)


def propagation_table(diagnostics: Mapping[str, Any]) -> str:
    """§15: whether a parent's rotation error reaches its children."""
    lines = [
        f"Split `{diagnostics['split']}`, seed mean, degrees. `T0_global` composes nothing, "
        "so its correlation is the shared-representation baseline; anything above it in the "
        "parent-relative cells is what composition added.",
        "",
        "| cell | arm | depth | parent rotation° | child rotation° | correlation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell, arms in diagnostics["summary"].items():
        for arm, depths in arms.items():
            for depth, entry in depths.items():
                lines.append(
                    f"| {cell} | {arm} | {depth} | {entry['parent_rotation_deg']:.2f} "
                    f"| {entry['child_rotation_deg']:.2f} "
                    f"| {entry['parent_child_correlation']:+.2f} |"
                )
    return "\n".join(lines)


def oracle_table(diagnostics: Mapping[str, Any]) -> str:
    """§16: the child recomposed onto its parent's true frame. Diagnostic, never a benchmark."""
    lines = [
        f"Split `{diagnostics['split']}`, seed mean. Parent-relative cells only — the global "
        "cell composes nothing, so there is no parent to replace. `lost to predicted parent` "
        "is as-trained minus oracle: what composing onto a predicted parent costs.",
        "",
        "| cell | arm | depth | rotation, as trained° | rotation, true parent° "
        "| rotation lost° | position, as trained | position, true parent | position lost |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell, arms in diagnostics["summary"].items():
        for arm, depths in arms.items():
            for depth, entry in depths.items():
                if "oracle_rotation_deg" not in entry:
                    continue
                lines.append(
                    f"| {cell} | {arm} | {depth} | {entry['child_rotation_deg']:.2f} "
                    f"| {entry['oracle_rotation_deg']:.2f} "
                    f"| {_signed(entry['rotation_lost_to_predicted_parent_deg'], 2)} "
                    f"| {entry['child_position']:.4f} | {entry['oracle_position']:.4f} "
                    f"| {_signed(entry['position_lost_to_predicted_parent'])} |"
                )
    return "\n".join(lines)


def compute_table(report: Mapping[str, Any]) -> str:
    """§19: the same budget for every cell, and the loss weights every run used."""
    by_key: dict[str, list[Mapping[str, Any]]] = {}
    for run in report["runs"]:
        by_key.setdefault(f"{run['arm']}/{run['cell']}", []).append(run)
    objective = report["objective"]
    lines = [
        f"Objective: translation {objective['translation_weight']}, scale "
        f"{objective['scale_weight']}, rotation {objective['rotation_loss_weight']} "
        f"({objective['rotation_objective']}), all inside a frame term weighted "
        f"{objective['frame_weight']}. Every run used these; the runner refuses to start "
        "otherwise.",
        "",
        "| arm | cell | parameters | steps | batch | threads | seeds | train s | eval s "
        "| depth s | rotation weight |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        for cell in report["cells"]:
            runs = by_key.get(f"{arm}/{cell}")
            if not runs:
                continue
            protocol = runs[0]["protocol"]

            def mean(field: str, chosen: Sequence[Mapping[str, Any]] = runs) -> float:
                return sum(float(run["timing"][field]) for run in chosen) / len(chosen)

            weights = {run["rotation_loss_weight"] for run in runs}
            lines.append(
                f"| {arm} | {cell} | {int(runs[0]['parameters']['total']):,} "
                f"| {protocol['steps']} | {protocol['batch_size']} "
                f"| {protocol['torch_threads']} "
                f"| {','.join(str(r['seed']) for r in sorted(runs, key=lambda x: x['seed']))} "
                f"| {mean('train_seconds'):.0f} | {mean('eval_seconds'):.0f} "
                f"| {mean('depth_seconds'):.0f} "
                f"| {'/'.join(str(w) for w in sorted(weights))} |"
            )
    return "\n".join(lines)


def integrity_table(report: Mapping[str, Any]) -> str:
    """§18: the gate's outcome across every run."""
    runs = report["runs"]
    checks = [run["integrity"]["train"] for run in runs] + [
        entry for run in runs for entry in run["integrity"]["splits"].values()
    ]
    round_trip = [
        max(entry.get("round_trip_error", 0.0), entry.get("probe_round_trip_error", 0.0))
        for entry in checks
    ]
    verified = {int(run["corpus_frozen"]["files_verified"]) for run in runs}
    return "\n".join(
        [
            "| check | result |",
            "| --- | --- |",
            f"| runs that passed the gate before training | {len(runs)} of {len(runs)} |",
            f"| corpus files verified against the frozen checksums, per run | {verified} |",
            f"| split-level gate passes | {sum(len(r['integrity']['splits']) for r in runs)} |",
            f"| worst round-trip error | {max(round_trip):.2e} |",
            f"| worst depth-vs-headline reconciliation | "
            f"{max(max(r['depth_reconciliation'].values()) for r in runs):.2e} |",
        ]
    )


def _sections(report: Mapping[str, Any], diagnostics: Mapping[str, Any] | None) -> list[
    tuple[str, str]
]:
    splits = list(report["dataset"]["splits"])
    sections = [
        ("Per seed — test_seen", per_seed_table(report)),
        ("Aggregated by arm and cell", aggregate_table(report)),
        ("Rotation in detail", rotation_detail_table(report)),
    ]
    for split in splits[1:]:
        sections.append((f"Aggregated — {split}", aggregate_table(report, split=split)))
    sections += [
        ("Depth analysis", depth_table(report)),
        ("T4_rigid against the global cell", paired_table(report, treatment="T4_rigid")),
        (
            "T1_spatial against the global cell (replication of Change 1)",
            paired_table(report, treatment="T1_spatial"),
        ),
    ]
    if diagnostics is not None:
        sections += [
            ("Parent-rotation error propagation", propagation_table(diagnostics)),
            ("True-parent oracle", oracle_table(diagnostics)),
        ]
    sections += [
        ("Compute and capacity", compute_table(report)),
        ("Integrity", integrity_table(report)),
    ]
    return sections


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 Change 2 result tables.")
    base = Path("experiments/runs/step10-change2")
    parser.add_argument("--report", type=Path, default=base / "change2_report.json")
    parser.add_argument("--diagnostics", type=Path, default=base / "change2_diagnostics.json")
    parser.add_argument("--insert-into", type=Path, default=None)
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    diagnostics = (
        json.loads(args.diagnostics.read_text(encoding="utf-8"))
        if args.diagnostics.exists()
        else None
    )
    rendered = "\n".join(
        f"### {title}\n\n{body}\n" for title, body in _sections(report, diagnostics)
    )
    if args.insert_into is None:
        print(rendered)
        return 0
    document = args.insert_into.read_text(encoding="utf-8")
    if BEGIN not in document or END not in document:
        raise SystemExit(f"{args.insert_into} has no {BEGIN!r} ... {END!r} markers")
    head, rest = document.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    source = "<!-- generated by experiments.step10.change2_training_tables -->"
    args.insert_into.write_text(
        f"{head}{BEGIN}\n{source}\n\n{rendered}\n{END}{tail}", encoding="utf-8"
    )
    print(f"tables written into {args.insert_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
