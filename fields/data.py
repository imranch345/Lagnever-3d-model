"""Level 2b training data: labelled 3D points from the CT label masks, in Level 2a's frame.

For each subject, the label masks of the modelled structures become one label volume:
- 0 is "nothing";
- 1..K are the structures in ``CLASSES`` order;
- a lung is the union of its lobes, as in Level 1.

Points are drawn from that volume:

* **near boundaries** (``boundary_fraction`` of them): voxels whose 3×3×3 neighbourhood holds
  more than one label, jittered within the voxel and by ``jitter_mm``, so the network learns
  where one structure ends and the next begins;
* **uniformly** in the region holding any structure, plus ``margin_mm``.

Each point takes the label of its nearest voxel. Coordinates are mapped as Level 1 maps mesh
vertices:
1. RAS+ millimetres by the volume's affine;
2. minus the subject's heart-shell centroid (``origin_ras_mm`` in its manifest);
3. ``RAS_TO_VIEWER``;
4. Level 2a's rigid alignment by the heart shell onto the reference subject (s0029).

Points, Level 1 meshes and Level 2a models therefore share one frame, and every comparison is
direct.

    python -m fields.data --raw datasets/raw/totalsegmentator_v1
        --scenes datasets/processed/level1/real --out datasets/processed/level2b
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]

from assets.labels import DEFAULT_LABEL_MAP, LabelMap, load_label_map
from assets.meshes import RAS_TO_VIEWER, read_mask
from shapes.build import DEFAULT_ENTITIES
from shapes.register import load_scene, rigid_to

__all__ = [
    "CLASSES",
    "Frame",
    "build",
    "frame_for",
    "label_volume",
    "sample",
    "to_aligned",
]

FloatArray = npt.NDArray[np.float64]
#: The structures, in label order 1..K; 0 is "nothing". Level 2a's seven, for a direct comparison.
CLASSES: tuple[str, ...] = DEFAULT_ENTITIES
REFERENCE = "s0029"


@dataclass(frozen=True)
class Frame:
    """How a subject's RAS+ millimetres reach the shared, aligned viewer frame."""

    origin_ras_mm: FloatArray
    rigid: FloatArray

    def to_dict(self) -> dict[str, Any]:
        """Plain data for the manifest."""
        return {"origin_ras_mm": self.origin_ras_mm.tolist(), "rigid": self.rigid.tolist()}


def label_volume(
    subject_dir: Path,
    classes: Sequence[str] = CLASSES,
    label_map: LabelMap | None = None,
) -> tuple[npt.NDArray[np.uint8], FloatArray, int]:
    """One label volume (0 = nothing), its affine, and the count of voxels claimed twice.

    TotalSegmentator's labels do not overlap. A voxel claimed by two structures would be
    counted, and keeps the later class, so nothing is hidden.
    """
    label_map = label_map or load_label_map(DEFAULT_LABEL_MAP)
    labels: npt.NDArray[np.uint8] | None = None
    affine: FloatArray | None = None
    overlaps = 0
    for index, entity in enumerate(classes, start=1):
        entry = label_map.for_entity(entity)
        if entry.crop is not None:
            raise ValueError(f"{entity} needs a crop; Level 2b's classes do not support crops yet")
        for name in entry.labels:
            mask, this_affine = read_mask(subject_dir / "segmentations" / f"{name}.nii.gz")
            if labels is None:
                labels = np.zeros(mask.shape, dtype=np.uint8)
                affine = this_affine
            elif affine is None or mask.shape != labels.shape:
                raise ValueError(f"{name} is on a different grid from the other labels")
            elif not np.allclose(this_affine, affine, atol=1e-4):
                raise ValueError(f"{name} is on a different grid from the other labels")
            overlaps += int(np.count_nonzero(mask & (labels > 0) & (labels != index)))
            labels[mask] = index
    assert labels is not None and affine is not None
    return labels, affine, overlaps


def frame_for(scene_dir: Path, reference_heart: Any, seed: int = 0) -> Frame:
    """The subject's origin from its manifest, and Level 2a's rigid alignment onto the reference."""
    manifest = json.loads((scene_dir / "manifest.json").read_text(encoding="utf-8"))
    origin = np.asarray(manifest["origin_ras_mm"], dtype=np.float64)
    heart = load_scene(scene_dir)["heart"]
    rigid, _ = rigid_to(reference_heart, heart, seed=seed)
    return Frame(origin_ras_mm=origin, rigid=np.asarray(rigid, dtype=np.float64))


def to_aligned(points_ras: npt.ArrayLike, frame: Frame) -> FloatArray:
    """RAS+ millimetres to the aligned viewer frame (Level 1's mapping, then Level 2a's)."""
    viewer = (np.asarray(points_ras, dtype=np.float64) - frame.origin_ras_mm) @ RAS_TO_VIEWER.T
    return np.asarray(viewer @ frame.rigid[:3, :3].T + frame.rigid[:3, 3], dtype=np.float64)


def sample(
    labels: npt.NDArray[np.uint8],
    affine: FloatArray,
    frame: Frame,
    *,
    n_points: int,
    rng: np.random.Generator,
    boundary_fraction: float = 0.6,
    jitter_mm: float = 1.5,
    margin_mm: float = 10.0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Labelled points in the aligned frame: near boundaries, then uniform around the structures."""
    occupied = np.argwhere(labels > 0)
    if occupied.size == 0:
        raise ValueError("no structure in this volume")
    spacing = float(np.mean(np.abs(np.diag(affine)[:3])))
    margin = int(np.ceil(margin_mm / spacing))
    low = np.maximum(occupied.min(axis=0) - margin, 0)
    high = np.minimum(occupied.max(axis=0) + margin, np.array(labels.shape) - 1)
    box = labels[low[0] : high[0] + 1, low[1] : high[1] + 1, low[2] : high[2] + 1]
    boundary = ndimage.maximum_filter(box, size=3) != ndimage.minimum_filter(box, size=3)
    edges = np.argwhere(boundary) + low

    n_near = int(round(n_points * boundary_fraction))
    near = edges[rng.integers(len(edges), size=n_near)].astype(np.float64)
    near += rng.uniform(-0.5, 0.5, size=near.shape)
    near += rng.normal(0.0, jitter_mm / spacing, size=near.shape)
    uniform = rng.uniform(low - 0.5, high + 0.5, size=(n_points - n_near, 3))
    ijk = np.concatenate([near, uniform])

    index = np.clip(np.rint(ijk).astype(np.int64), 0, np.array(labels.shape) - 1)
    point_labels = labels[index[:, 0], index[:, 1], index[:, 2]]
    points = to_aligned(nib.affines.apply_affine(affine, ijk), frame)
    order = rng.permutation(len(points))
    return points[order].astype(np.float32), point_labels[order].astype(np.uint8)


def _usable(scenes_root: Path) -> list[str]:
    quality = json.loads((scenes_root / "quality_report.json").read_text(encoding="utf-8"))
    return sorted(q["subject"] for q in quality["subjects"] if q["usable"])


def build(
    raw_root: Path,
    scenes_root: Path,
    out_dir: Path,
    *,
    subjects: Sequence[str] | None = None,
    n_points: int = 200_000,
    seed: int = 0,
    split_from: Path | None = Path("datasets/processed/level2a/report.json"),
) -> dict[str, Any]:
    """Points for every subject, resumable; returns and writes ``manifest.json``."""
    subjects = list(subjects) if subjects is not None else _usable(scenes_root)
    reference_heart = load_scene(scenes_root / REFERENCE)["heart"]
    points_dir = out_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    label_map = load_label_map(DEFAULT_LABEL_MAP)
    stats: dict[str, Any] = {}
    for position, subject in enumerate(subjects):
        target = points_dir / f"{subject}.npz"
        if target.exists():
            stored = np.load(target)
            stats[subject] = json.loads(str(stored["stats"]))
            continue
        labels, affine, overlaps = label_volume(raw_root / subject, CLASSES, label_map)
        frame = frame_for(scenes_root / subject, reference_heart, seed=seed)
        rng = np.random.default_rng([seed, position])
        points, point_labels = sample(labels, affine, frame, n_points=n_points, rng=rng)
        voxel_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
        counts = np.bincount(labels.ravel(), minlength=len(CLASSES) + 1)
        stats[subject] = {
            "overlapping_voxels": overlaps,
            "volume_ml": {
                c: round(float(counts[i + 1]) * voxel_ml, 1) for i, c in enumerate(CLASSES)
            },
            "point_label_counts": np.bincount(point_labels, minlength=len(CLASSES) + 1).tolist(),
            "frame": frame.to_dict(),
        }
        np.savez_compressed(
            target, points=points, labels=point_labels, stats=json.dumps(stats[subject])
        )
        print(f"points {subject}: {n_points} ({position + 1}/{len(subjects)})", flush=True)

    split = None
    if split_from is not None and split_from.is_file():
        report = json.loads(split_from.read_text(encoding="utf-8"))
        split = {"train": report["train"], "held_out": report["held_out"], "from": str(split_from)}
    manifest = {
        "dataset": "lagnav level 2b labelled points",
        "classes": ["nothing", *CLASSES],
        "n_points": n_points,
        "sampling": {"boundary_fraction": 0.6, "jitter_mm": 1.5, "margin_mm": 10.0, "seed": seed},
        "frame": "Level 1 viewer frame (heart-shell centroid origin), aligned rigidly onto "
        f"{REFERENCE} by the heart shell as in Level 2a",
        "reference": REFERENCE,
        "subjects": subjects,
        "split": split,
        "source": asdict(label_map.source),
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Build the labelled points."""
    parser = argparse.ArgumentParser(description="Level 2b labelled points.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args(argv)
    manifest = build(args.raw, args.scenes, args.out, subjects=args.subjects, n_points=args.points)
    print(f"{len(manifest['subjects'])} subjects, classes {manifest['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
