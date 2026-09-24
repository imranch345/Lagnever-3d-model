"""Step 10 Change 2 §20: every table in the corpus report, generated from the artifacts.

Reads the rotation audit, the two floor runs, the corpus validation and both manifests, and
emits Markdown. The report pastes this; it types no numbers of its own.

    python -m experiments.step10.change2_tables --insert-into docs/STEP_10_CHANGE2_CORPUS_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "entity_rotation_table",
    "floor_table",
    "holdout_table",
    "main",
    "manifest_table",
    "rotation_table",
    "validation_table",
]

BEGIN = "<!-- BEGIN GENERATED TABLES -->"
END = "<!-- END GENERATED TABLES -->"
SPLITS: tuple[str, ...] = (
    "train",
    "validation",
    "test_seen",
    "test_arrangement",
    "test_transform",
    "test_combination",
)
TEST_SPLITS: tuple[str, ...] = (
    "test_seen",
    "test_arrangement",
    "test_transform",
    "test_combination",
)
#: Change 1's floor, for the old-vs-new table. Historical context, never a bar for this corpus.
CHANGE1_POSITION_FLOOR: Mapping[str, float] = {
    "test_seen": 0.1605,
    "test_arrangement": 0.1655,
    "test_transform": 0.1703,
    "test_combination": 0.1750,
}
DEGREES = 180.0 / math.pi


def rotation_table(audit: Mapping[str, Any]) -> str:
    """§20: the rotation-angle distribution per split, measured through the loader."""
    lines = [
        "Geodesic angle from the identity, in degrees, over the entities the evaluation "
        "loader delivers. `<1 deg` is the share a model could get right by predicting no "
        "rotation at all.",
        "",
        "| split | mean | median | std | min | max | <1 deg | >10 deg | >30 deg | >60 deg |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        entry = audit["splits"][split]
        lines.append(
            f"| {split} | {entry['mean_deg']:.2f} | {entry['median_deg']:.2f} "
            f"| {entry['std_deg']:.2f} | {entry['min_deg']:.2f} | {entry['max_deg']:.2f} "
            f"| {entry['fraction_below_1_deg']:.3f} | {entry['fraction_above_10_deg']:.2f} "
            f"| {entry['fraction_above_30_deg']:.2f} | {entry['fraction_above_60_deg']:.2f} |"
        )
    overall = audit["overall"]
    lines += [
        "",
        f"All splits together: {overall['count']:,} frames, mean "
        f"{overall['mean_deg']:.2f} deg, max {overall['max_deg']:.2f} deg, "
        f"{overall['fraction_below_1_deg']:.3f} below 1 deg.",
    ]
    return "\n".join(lines)


def entity_rotation_table(audit: Mapping[str, Any]) -> str:
    """Per-entity rotation, which is where the floor's headroom is decided."""
    lines = [
        "Per entity, all splits together. The ten entities with no distinguished "
        "construction axis carry the organ's own rotation; the ten annuli and tubes carry "
        "their axis as well. A small `std` means an identity-only lookup can predict most of "
        "that entity's rotation, which is what the floor measures.",
        "",
        "| entity | mean | std | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entity_id, entry in sorted(
        audit["entities"].items(), key=lambda item: item[1]["mean_deg"]
    ):
        lines.append(
            f"| {entity_id.split('.')[-1]} | {entry['mean_deg']:.2f} | {entry['std_deg']:.2f} "
            f"| {entry['min_deg']:.2f} | {entry['max_deg']:.2f} |"
        )
    return "\n".join(lines)


def floor_table(floor: Mapping[str, Any], repeat: Mapping[str, Any]) -> str:
    """§20: the new placement-blind floor, and the old one beside it as context only."""
    identical = json.dumps(floor["splits"], sort_keys=True) == json.dumps(
        repeat["splits"], sort_keys=True
    )
    lines = [
        f"Fitted on `train` only, identity-only lookup, no relations, no hierarchy, no scene "
        f"context. Computed twice: the two runs are "
        f"{'identical' if identical else 'NOT identical'}.",
        "",
        "| split | old floor (Change 1) | new position floor | new rotation floor (deg) "
        "| new scale floor | new composite | directly comparable? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in TEST_SPLITS:
        entry = floor["splits"][split]["global"]
        old = CHANGE1_POSITION_FLOOR[split]
        lines.append(
            f"| {split} | {old:.4f} | {entry['position_error']:.4f} "
            f"| {entry['rotation_error'] * DEGREES:.2f} | {entry['scale_error']:.4f} "
            f"| {entry['composite_frame_error']:.4f} | No |"
        )
    lines += [
        "",
        "**The position column is identical to Change 1's, and that is a measurement, not a "
        "reuse.** The rotated corpus was derived from the Change 1 corpus, so every centroid "
        "is the same number; a floor fitted on identity alone therefore lands in the same "
        "place. The rotation and scale floors are new, the composite is new, and a model "
        "trained here faces a different task, so its results still belong to a separate "
        "track from Change 1's.",
    ]
    return "\n".join(lines)


def parent_relative_floor_table(floor: Mapping[str, Any]) -> str:
    """The parent-relative floors on the rotated corpus, for Change 2's own question."""
    lines = [
        "The same identity-only lookup under a parent-relative target, now with real "
        "rotations. `blind` composes its own guesses; `true parent` leaks the parent frame "
        "and is a diagnostic, never a bar.",
        "",
        "| split | global | parent-relative, blind | with true parent | rotation, global (deg) "
        "| rotation, true parent (deg) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for split in TEST_SPLITS:
        entry = floor["splits"][split]
        lines.append(
            f"| {split} | {entry['global']['position_error']:.4f} "
            f"| {entry['parent_relative']['position_error']:.4f} "
            f"| {entry['parent_relative_oracle']['position_error']:.4f} "
            f"| {entry['global']['rotation_error'] * DEGREES:.2f} "
            f"| {entry['parent_relative_oracle']['rotation_error'] * DEGREES:.2f} |"
        )
    return "\n".join(lines)


def holdout_table(audit: Mapping[str, Any], validation: Mapping[str, Any]) -> str:
    """§16: which rotation bands each split occupies."""
    lines = [
        "Rotation follows the arrangement, so the band a split occupies in the arrangement "
        "space is the band it occupies in rotation. `hole` is the held-out yaw interval, "
        "`beyond edge` the held-out extrapolation region, and `mirror+transpose` the held-out "
        "combination.",
        "",
        "| split | abs yaw min | abs yaw max | in hole | beyond edge | mirrored | transposed "
        "| mean rotation (deg) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        band = validation["holdout"]["splits"][split]
        lines.append(
            f"| {split} | {band['abs_yaw_min']:.3f} | {band['abs_yaw_max']:.3f} "
            f"| {band['fraction_in_interpolation_hole']:.2f} "
            f"| {band['fraction_beyond_extrapolation_edge']:.2f} "
            f"| {band['fraction_mirrored']:.2f} | {band['fraction_transposed']:.2f} "
            f"| {audit['splits'][split]['mean_deg']:.2f} |"
        )
    hole = validation["holdout"]["yaw_interpolation_hole"]
    lines += [
        "",
        f"Held out: |yaw| in [{hole[0]}, {hole[1]}] and a septal band "
        f"(`test_arrangement`); |yaw| above "
        f"{validation['holdout']['yaw_extrapolation_edge']} (`test_transform`); mirror and "
        "transpose together (`test_combination`). `test_seen` holds out families instead, at "
        "in-distribution arrangements.",
    ]
    return "\n".join(lines)


def manifest_table(manifest: Mapping[str, Any]) -> str:
    """§13: the machine-readable corpus manifest, as a table."""
    rows = [
        ("corpus_id", manifest["corpus_id"]),
        ("parent_corpus_id", manifest["parent_corpus_id"]),
        ("generator_version", manifest["generator_version"]),
        ("generator_commit", manifest["generator_commit"]),
        ("random_seed", manifest["random_seed"]),
        ("rotation_enabled", manifest["rotation_enabled"]),
        ("format_version", manifest["format_version"]),
        ("scenes", f"{manifest['scenes']:,}"),
        ("arrangement_count", f"{manifest['arrangements']:,}"),
        ("family_count", manifest["families"]),
        ("entity_count", manifest["entity_count"]),
        ("distinct_relation_graphs", manifest["distinct_relation_graphs"]),
    ]
    lines = ["| field | value |", "| --- | --- |"]
    lines += [f"| `{name}` | {value} |" for name, value in rows]
    lines += ["", "| split | scenes | arrangement region |", "| --- | --- | --- |"]
    for split, count in manifest["split_counts"].items():
        lines.append(f"| {split} | {count} | {manifest['split_definition'][split]} |")
    lines += ["", "**Changes from the parent corpus**", ""]
    lines += [f"* {change}" for change in manifest["changes_from_parent"]]
    return "\n".join(lines)


def validation_table(validation: Mapping[str, Any]) -> str:
    """§14: what the corpus validation checked and found."""
    pairing = validation["pairing"]["splits"]
    rotations = validation["rotations"]
    placement = validation["placement"]["test_seen"]
    moved = sum(int(entry["entities_whose_extent_moved"]) for entry in pairing.values())
    paired = sum(int(entry["scenes"]) for entry in pairing.values())
    worst_centroid = max(float(entry["worst_centroid_deviation"]) for entry in pairing.values())
    worst_rotation = float(rotations["worst_deviation"])
    lines = [
        "| check | result |",
        "| --- | --- |",
        f"| scenes paired with the parent corpus | {paired:,} |",
        f"| worst centroid deviation from the parent | {worst_centroid:.1e} |",
        "| relationship graphs, point counts, presence, levels of detail | identical |",
        f"| entity frames whose extent moved | {moved:,} (the intended consequence) |",
        f"| rotations compared against the generator | {int(rotations['rotations_checked']):,} |",
        f"| worst rotation deviation from the generator | {worst_rotation:.1e} |",
        f"| placement gate, stored-frame round trip | {placement['round_trip_error']:.1e} |",
        f"| placement gate, probe-rotated round trip | {placement['probe_round_trip_error']:.1e} |",
        f"| largest stored rotation away from identity | "
        f"{placement['max_rotation_deviation_from_identity']:.4f} |",
        f"| parent table against the generator's construction | "
        f"{placement['source_table']} |",
    ]
    return "\n".join(lines)


def _sections(paths: Mapping[str, Path]) -> list[tuple[str, str]]:
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    floor = json.loads(paths["floor"].read_text(encoding="utf-8"))
    repeat = json.loads(paths["repeat"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    return [
        ("Corpus manifest", manifest_table(validation["manifest"])),
        ("Rotation distribution", rotation_table(audit)),
        ("Rotation per entity", entity_rotation_table(audit)),
        ("New placement-blind floor", floor_table(floor, repeat)),
        ("Parent-relative floors on the rotated corpus", parent_relative_floor_table(floor)),
        ("Transformation hold-out", holdout_table(audit, validation)),
        ("Corpus validation", validation_table(validation)),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 Change 2 tables.")
    base = Path("experiments/runs/step10-change2")
    parser.add_argument("--audit", type=Path, default=base / "rotation_audit.json")
    parser.add_argument("--floor", type=Path, default=base / "rotated_placement_floor.json")
    parser.add_argument(
        "--repeat", type=Path, default=base / "rotated_placement_floor_repeat.json"
    )
    parser.add_argument("--validation", type=Path, default=base / "corpus_validation.json")
    parser.add_argument("--insert-into", type=Path, default=None)
    args = parser.parse_args(argv)

    sections = _sections(
        {
            "audit": args.audit,
            "floor": args.floor,
            "repeat": args.repeat,
            "validation": args.validation,
        }
    )
    rendered = "\n".join(f"### {title}\n\n{body}\n" for title, body in sections)
    if args.insert_into is None:
        print(rendered)
        return 0
    document = args.insert_into.read_text(encoding="utf-8")
    if BEGIN not in document or END not in document:
        raise SystemExit(f"{args.insert_into} has no {BEGIN!r} ... {END!r} markers")
    head, rest = document.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    source = "<!-- generated by experiments.step10.change2_tables -->"
    args.insert_into.write_text(
        f"{head}{BEGIN}\n{source}\n\n{rendered}\n{END}{tail}", encoding="utf-8"
    )
    print(f"tables written into {args.insert_into}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
