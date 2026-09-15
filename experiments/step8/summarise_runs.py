"""Summarise Step 8 runs from the per-run manifests, before the suite finishes.

Each run writes its manifest the moment it completes, so results can be read while later
arms are still training. Useful during a long suite, and it is also the recovery path if
the suite fails after training: the expensive part is already on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["load_manifests", "summarise", "main"]

COLUMNS: tuple[tuple[str, str], ...] = (
    ("entity_iou_mean", "iou"),
    ("part_control_success", "ctrl"),
    ("position_error", "pos_err"),
    ("composite_frame_error", "frame"),
    ("spatial_relation_accuracy", "relation"),
    ("scene_iou", "scene"),
    ("lod_containment_mean", "contain"),
    ("lod_preservation_mean", "preserve"),
    ("lod_detail_gain", "gain_raw"),
)


def load_manifests(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Every finished run's manifest, newest ordering irrelevant."""
    out: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("results"):
            out.append(payload)
    return out


def summarise(manifests: Sequence[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and spread per arm over whatever seeds have finished."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        grouped[str(manifest["arm"])].append(manifest)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, runs in grouped.items():
        out[arm] = {}
        for metric, _ in COLUMNS:
            values = [
                float(run["results"][metric])
                for run in runs
                if metric in run["results"] and math.isfinite(float(run["results"][metric]))
            ]
            out[arm][metric] = {
                "mean": statistics.fmean(values) if values else float("nan"),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "seeds": float(len(values)),
            }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Summarise Step 8 runs so far.")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("experiments/runs/step8-suite/checkpoints"),
    )
    parser.add_argument("--floor", type=float, default=None, help="Relation-blind floor.")
    args = parser.parse_args(argv)

    manifests = load_manifests(args.checkpoints)
    if not manifests:
        print("No finished runs yet.")
        return 0
    summary = summarise(manifests)
    order = ["A1", "A1M", "A3Lite", "A3L", "A3", "A4", "A0"]
    header = f"{'arm':8}{'seeds':>6}" + "".join(f"{label:>10}" for _, label in COLUMNS)
    print(header)
    for arm in order:
        if arm not in summary:
            continue
        seeds = summary[arm][COLUMNS[0][0]]["seeds"]
        row = f"{arm:8}{int(seeds):>6}"
        for metric, _ in COLUMNS:
            value = summary[arm][metric]["mean"]
            row += "       n/a" if not math.isfinite(value) else f"{value:>10.4f}"
        print(row)
    if args.floor is not None:
        print(f"\nRelation-blind floor: {args.floor:.4f}. Below it means no relational competence.")
        for arm in order:
            if arm not in summary:
                continue
            value = summary[arm]["spatial_relation_accuracy"]["mean"]
            if math.isfinite(value):
                gap = (value - args.floor) / max(1.0 - args.floor, 1e-9)
                print(f"  {arm:8} {value:.4f}  gap {gap:+.3f}")
    print(
        "\nValidation split, predicted placement. Synthetic research data; not anatomy, "
        "not validated, not clinical."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
