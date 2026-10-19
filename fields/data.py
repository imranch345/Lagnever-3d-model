"""Level 2b training data: labelled 3D points from the CT label masks, in Level 2a's frame.

For each subject, the label masks of the modelled structures become one label volume:
- 0 is "nothing";
- 1..K are the structures in class order (``PRESETS``: ``heart-lungs``, the first seven,
  or ``chest``, fourteen with the great vessels, the lobes and the trachea);
- a lung is the union of its lobes, as in Level 1;
- a structure with a crop is cropped as Level 1 crops it. The aorta and the IVC keep only
  the slices within the lungs' extent, so a vessel's length does not depend on how far the
  scan happened to reach. Level 2b adds the trachea (``EXTRA_CROPS``): its upper end is
  often where the scan stops, not where the trachea does, so it too is "the thoracic
  part".

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
from assets.meshes import RAS_TO_VIEWER, _crop_to_extent, read_mask
from shapes.build import DEFAULT_ENTITIES
from shapes.register import load_scene, rigid_to

__all__ = [
    "CHEST",
    "CLASSES",
    "EXTRA_CROPS",
    "PRESETS",
    "crop_of",
    "labels_on_grid",
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
#: Stage 2 of the roadmap: the whole chest Level 1 builds, lungs by lobe.
CHEST: tuple[str, ...] = (
    "heart.left_atrium",
    "heart.left_ventricle",
    "heart.right_atrium",
    "heart.right_ventricle",
    "heart.myocardium",
    "heart.aorta",
    "heart.pulmonary_trunk",
    "heart.inferior_vena_cava",
    "respiratory.lung_upper_lobe_left",
    "respiratory.lung_lower_lobe_left",
    "respiratory.lung_upper_lobe_right",
    "respiratory.lung_middle_lobe_right",
    "respiratory.lung_lower_lobe_right",
    "respiratory.trachea",
)
PRESETS: dict[str, tuple[str, ...]] = {"heart-lungs": CLASSES, "chest": CHEST}
#: Crops Level 2b adds to the label map's, for structures a scan's edge often cuts.
EXTRA_CROPS: dict[str, str] = {"respiratory.trachea": "thorax"}


def crop_of(entity: str, label_map: LabelMap) -> str | None:
    """The crop a structure gets in Level 2b: the label map's, else ``EXTRA_CROPS``."""
    return label_map.for_entity(entity).crop or EXTRA_CROPS.get(entity)


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
    counted, and keeps the later class, so nothing is hidden. Crops are applied as Level 1
    applies them (``assets.meshes``).
    """
    label_map = label_map or load_label_map(DEFAULT_LABEL_MAP)
    labels: npt.NDArray[np.uint8] | None = None
    affine: FloatArray | None = None
    overlaps = 0
    references: dict[str, npt.NDArray[np.bool_]] = {}

    def read(name: str) -> npt.NDArray[np.bool_]:
        nonlocal labels, affine
        mask, this_affine = read_mask(subject_dir / "segmentations" / f"{name}.nii.gz")
        if labels is None:
            labels = np.zeros(mask.shape, dtype=np.uint8)
            affine = this_affine
        elif affine is None or mask.shape != labels.shape:
            raise ValueError(f"{name} is on a different grid from the other labels")
        elif not np.allclose(this_affine, affine, atol=1e-4):
            raise ValueError(f"{name} is on a different grid from the other labels")
        return np.asarray(mask, dtype=bool)

    for index, entity in enumerate(classes, start=1):
        entry = label_map.for_entity(entity)
        union = np.logical_or.reduce([read(name) for name in entry.labels])
        crop = crop_of(entity, label_map)
        if crop is not None:
            if crop not in references:
                names = label_map.crops[crop].reference_labels
                references[crop] = np.logical_or.reduce([read(n) for n in names])
            union = _crop_to_extent(union, references[crop])
        assert labels is not None
        overlaps += int(np.count_nonzero(union & (labels > 0) & (labels != index)))
        labels[union] = index
    assert labels is not None and affine is not None
    return labels, affine, overlaps


def labels_on_grid(
    subject_dir: Path,
    frame: Any,
    points_aligned: npt.ArrayLike,
    classes: Sequence[str] = CLASSES,
    label_map: LabelMap | None = None,
) -> npt.NDArray[np.uint8]:
    """The subject's true label at each aligned-frame point: its nearest CT voxel, 0 outside.

    ``frame`` is a ``Frame`` or its manifest form (``origin_ras_mm`` and ``rigid``). This is
    ``to_aligned`` run backwards, so it is what a perfect field would decode.
    """
    if not isinstance(frame, Frame):
        frame = Frame(
            origin_ras_mm=np.asarray(frame["origin_ras_mm"], dtype=np.float64),
            rigid=np.asarray(frame["rigid"], dtype=np.float64),
        )
    labels, affine, _ = label_volume(subject_dir, classes, label_map)
    aligned = np.asarray(points_aligned, dtype=np.float64)
    viewer = (aligned - frame.rigid[:3, 3]) @ frame.rigid[:3, :3]
    ras = viewer @ RAS_TO_VIEWER + frame.origin_ras_mm
    ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), ras)).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(labels.shape)), axis=1)
    truth = np.zeros(len(ijk), dtype=np.uint8)
    truth[inside] = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
    return truth


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
    classes: Sequence[str] = CLASSES,
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
        labels, affine, overlaps = label_volume(raw_root / subject, classes, label_map)
        frame = frame_for(scenes_root / subject, reference_heart, seed=seed)
        rng = np.random.default_rng([seed, position])
        points, point_labels = sample(labels, affine, frame, n_points=n_points, rng=rng)
        voxel_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
        counts = np.bincount(labels.ravel(), minlength=len(classes) + 1)
        stats[subject] = {
            "overlapping_voxels": overlaps,
            "volume_ml": {
                c: round(float(counts[i + 1]) * voxel_ml, 1) for i, c in enumerate(classes)
            },
            "point_label_counts": np.bincount(point_labels, minlength=len(classes) + 1).tolist(),
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
        "classes": ["nothing", *classes],
        "n_points": n_points,
        "sampling": {"boundary_fraction": 0.6, "jitter_mm": 1.5, "margin_mm": 10.0, "seed": seed},
        "frame": "Level 1 viewer frame (heart-shell centroid origin), aligned rigidly onto "
        f"{REFERENCE} by the heart shell as in Level 2a",
        "reference": REFERENCE,
        "crops": {c: crop for c in classes if (crop := crop_of(c, label_map)) is not None},
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
    parser.add_argument("--preset", choices=sorted(PRESETS), default="heart-lungs")
    args = parser.parse_args(argv)
    manifest = build(
        args.raw,
        args.scenes,
        args.out,
        subjects=args.subjects,
        n_points=args.points,
        classes=PRESETS[args.preset],
    )
    print(f"{len(manifest['subjects'])} subjects, classes {manifest['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
