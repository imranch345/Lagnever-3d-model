"""Decoded ÷ true volume per structure on held-out subjects (Level 2b).

A field can put a structure in the right place and still make it too thick. Surface
distance barely shows that for a thin tube; its volume does. For each held-out subject:
1. fit a code to its points, as evaluation does;
2. decode on the 2 mm grid and cut with the data's crops (``apply_crops``);
3. count each structure's grid points against the CT labels at the same points.

A ratio of 1 means the right volume. The chest model's trachea was 1.49 before calibration
(``fields.calibrate``).

    python -m fields.volumes --data datasets/processed/level2b-chest-v2
        --raw datasets/raw/totalsegmentator_v1 --subjects 6
        runs/gpu/chest-a-b8/checkpoint.pt runs/gpu/chest-a-b8/checkpoint-calibrated.pt
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fields.data import labels_on_grid
from fields.evaluate import Grid, apply_crops, decode_margins, fit_code, load_checkpoint
from fields.model import Field, pick_device

__all__ = ["volume_ratios"]


def volume_ratios(
    field: Field,
    classes: Sequence[str],
    grid: Grid,
    subjects: Sequence[str],
    data_dir: Path,
    raw_root: Path,
    stats: dict[str, Any],
    *,
    steps: int = 300,
) -> np.ndarray:
    """Decoded ÷ true grid-point count, (subjects, classes); NaN where the truth is empty."""
    everywhere = grid.points()
    ratios = []
    for subject in subjects:
        stored = np.load(data_dir / "points" / f"{subject}.npz")
        code = fit_code(field, stored["points"], stored["labels"], steps=steps)
        decoded = apply_crops(decode_margins(field, code, grid).argmax(0).astype(np.uint8), classes)
        truth = labels_on_grid(raw_root / subject, stats[subject]["frame"], everywhere, classes)
        n = len(classes) + 1
        found = np.bincount(decoded.ravel(), minlength=n)[1:].astype(np.float64)
        true = np.bincount(truth, minlength=n)[1:].astype(np.float64)
        ratios.append(np.where(true > 0, found / np.maximum(true, 1), np.nan))
        print(f"fitted {subject}", flush=True)
    return np.array(ratios)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the median ratio per structure for each checkpoint, side by side."""
    parser = argparse.ArgumentParser(description="Decoded ÷ true volume on held-out subjects.")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--subjects", type=int, default=6, help="first N held-out subjects")
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON of the per-subject ratios")
    args = parser.parse_args(argv)

    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    held_out = list(manifest["split"]["held_out"][: args.subjects])
    medians: dict[str, np.ndarray] = {}
    per_subject: dict[str, dict[str, list[float]]] = {}
    classes: list[str] = []
    for path in args.checkpoints:
        field, checkpoint = load_checkpoint(path, pick_device(args.device))
        classes = list(checkpoint["classes"][1:])
        grid = Grid.around(args.data, checkpoint["subjects"], args.spacing)
        ratios = volume_ratios(
            field, classes, grid, held_out, args.data, args.raw, manifest["stats"]
        )
        medians[str(path)] = np.nanmedian(ratios, axis=0)
        per_subject[str(path)] = {
            c: [round(float(r), 4) for r in ratios[:, k]] for k, c in enumerate(classes)
        }
    print(f"{'structure':36s}" + "".join(f"{Path(p).name[:24]:>26s}" for p in medians))
    for k, c in enumerate(classes):
        print(f"{c:36s}" + "".join(f"{m[k]:26.2f}" for m in medians.values()))
    if args.out is not None:
        report = {"subjects": held_out, "ratios": per_subject}
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
