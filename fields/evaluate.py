"""Evaluate a Level 2b label field on held-out subjects, and Level 2a on the same yardstick.

For each held-out subject:

1. **Fit.** A fresh code is optimised on the subject's labelled points, with the network
   frozen (auto-decoder test-time fitting, ``fit_steps`` Adam steps). With ``fit_weighted``
   the fit uses training's class weights, from the training subjects' point counts.
   Otherwise every point counts the same, and the lungs and "nothing" (over 90% of points)
   dominate the fit.
2. **Decode.** The field is evaluated on a grid (``spacing_mm``) over the training points'
   extent. Each structure is meshed three ways:
   - **binary:** marching cubes on its label mask, the M1 method;
   - **clean:** the same after ``keep_largest``, which returns every piece but a structure's
     largest to "nothing". Each modelled structure is one connected piece in a real thorax,
     and a fitted field can leave small stray islands in parts of the grid with no data;
     generation cleans them the same way;
   - **smooth:** marching cubes at 0 on its margin, its logit minus the best other logit.
     The margin is positive exactly where the structure is the most likely label, so the
     smooth surfaces bound the same regions as the labels, without the voxel staircase.
3. **Compare** with the subject's real Level 1 meshes, moved by the same rigid alignment:
   - symmetric mean surface distance in millimetres (both directions, averaged), for each
     meshing, summarised by mean and median over subjects;
   - the number of pieces each structure decoded into;
   - Dice on the subject's labelled points.

Level 2a is measured the same way: each held-out subject's registered shape is reconstructed by
its entity model and compared with the same real mesh. The two models are fitted to the
held-out person by their own mechanisms, then scored by one rule.

    python -m fields.evaluate --checkpoint runs/level2b/m1/checkpoint.pt
        --data datasets/processed/level2b --scenes datasets/processed/level1/real
        --level2a datasets/processed/level2a --out runs/level2b/m1/eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import trimesh
from scipy import ndimage  # type: ignore[import-untyped]
from skimage import measure
from torch.nn import functional

from fields.data import EXTRA_CROPS
from fields.model import Field, FieldConfig, make_field, pick_device
from fields.train import weights_from_counts
from shapes.register import load_scene

__all__ = [
    "Grid",
    "decode",
    "decode_margins",
    "apply_crops",
    "fit_code",
    "keep_largest",
    "label_meshes",
    "load_checkpoint",
    "lung_slab",
    "surface_distance",
]

FloatArray = npt.NDArray[np.float64]


class Grid:
    """An axis-aligned grid in the aligned frame: origin (mm), spacing (mm) and shape."""

    def __init__(self, low: npt.ArrayLike, high: npt.ArrayLike, spacing_mm: float) -> None:
        self.low = np.asarray(low, dtype=np.float64)
        self.spacing = float(spacing_mm)
        extent = np.asarray(high, dtype=np.float64) - self.low
        self.shape = tuple(int(n) for n in np.ceil(extent / self.spacing).astype(int) + 1)

    def points(self) -> npt.NDArray[np.float32]:
        """Every grid point, (X·Y·Z, 3), x slowest."""
        axes = [self.low[i] + self.spacing * np.arange(self.shape[i]) for i in range(3)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return mesh.reshape(-1, 3).astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> tuple[Field, dict[str, Any]]:
    """The trained field on ``device``, and the checkpoint's other contents."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["field_config"])
    config["part_of"] = tuple(config.get("part_of", ()))
    field = make_field(FieldConfig(**config))
    field.load_state_dict(checkpoint["field"])
    field.to(device).eval()
    for parameter in field.parameters():
        parameter.requires_grad_(False)
    return field, checkpoint


def fit_code(
    field: Field,
    points: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    steps: int = 300,
    batch: int = 16384,
    lr: float = 1e-2,
    prior: float = 1e-4,
    weights: torch.Tensor | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """A code for a subject the network never saw, the network frozen."""
    device = next(field.parameters()).device
    generator = np.random.default_rng(seed)
    points_t = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.int64)
    code = torch.zeros(1, field.config.code_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([code], lr=lr)
    for _ in range(steps):
        index = torch.from_numpy(generator.integers(len(points_t), size=batch))
        logits = field(points_t[index].unsqueeze(0).to(device), code)
        loss = (
            functional.cross_entropy(logits[0], labels_t[index].to(device), weight=weights)
            + prior * code.pow(2).sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return code.detach()


@torch.no_grad()
def decode(field: Field, code: torch.Tensor, grid: Grid, chunk: int = 65536) -> np.ndarray:
    """The most likely label at every grid point, as a (X, Y, Z) uint8 volume."""
    device = next(field.parameters()).device
    points = torch.from_numpy(grid.points())
    labels = torch.empty(len(points), dtype=torch.uint8)
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk].unsqueeze(0).to(device)
        labels[start : start + chunk] = field(part, code.to(device))[0].argmax(-1).to("cpu")
    return labels.numpy().reshape(grid.shape)


@torch.no_grad()
def decode_margins(field: Field, code: torch.Tensor, grid: Grid, chunk: int = 65536) -> np.ndarray:
    """Per class k, logit k minus the best other logit, as a (classes, X, Y, Z) float32 array.

    Positive exactly where k is the most likely label (``decode``'s argmax), zero where it ties
    with the runner-up. Classes therefore never claim the same point here either.
    """
    device = next(field.parameters()).device
    points = torch.from_numpy(grid.points())
    n_classes = field.config.n_classes
    margins = torch.empty(n_classes, len(points), dtype=torch.float32)
    classes = torch.arange(n_classes, device=device)
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk].unsqueeze(0).to(device)
        logits = field(part, code.to(device))[0]
        top = logits.topk(2, dim=-1)
        is_best = top.indices[:, :1] == classes
        other = torch.where(is_best, top.values[:, 1:2], top.values[:, :1])
        margins[:, start : start + chunk] = (logits - other).T.to("cpu")
    return margins.numpy().reshape(n_classes, *grid.shape)


def keep_largest(volume: np.ndarray, n_classes: int) -> tuple[np.ndarray, list[int]]:
    """Each class 1..n−1 cut to its largest connected piece; the rest become 0 ("nothing").

    Returns the cleaned volume and, per class 1..n−1, how many pieces it had.
    """
    clean = volume.copy()
    pieces: list[int] = []
    for k in range(1, n_classes):
        components, count = ndimage.label(volume == k)
        pieces.append(int(count))
        if count > 1:
            sizes = np.bincount(components.ravel())[1:]
            clean[(components > 0) & (components != 1 + int(np.argmax(sizes)))] = 0
    return clean, pieces


def apply_crops(volume: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Cut classes that the label map crops to their reference's extent, as the data was cut.

    The aorta, the IVC and the trachea are labelled only within the lungs'
    superior–inferior extent (``fields.data.crop_of``). Where no training point ever was,
    outside that extent, a decoded field can continue them. Decoded volumes are therefore
    cut the same way, along the aligned frame's superior axis (y). Classes without a crop
    are unchanged.
    """
    from assets.labels import DEFAULT_LABEL_MAP, load_label_map
    from fields.data import crop_of

    label_map = load_label_map(DEFAULT_LABEL_MAP)
    known = set(label_map.entity_ids())
    out = volume.copy()
    for index, entity in enumerate(classes, start=1):
        if entity not in known:
            continue
        crop = crop_of(entity, label_map)
        if crop is None:
            continue
        wanted = set(label_map.crops[crop].reference_labels)
        reference = [
            k
            for k, other in enumerate(classes, start=1)
            if other in known and wanted & set(label_map.for_entity(other).labels)
        ]
        rows = np.flatnonzero(np.isin(volume, reference).any(axis=(0, 2)))
        if len(rows) == 0:
            continue
        outside = np.ones(volume.shape[1], dtype=bool)
        outside[rows.min() : rows.max() + 1] = False
        mask = (out == index) & outside[None, :, None]
        out[mask] = 0
    return out


def label_meshes(
    volume: np.ndarray,
    grid: Grid,
    classes: Sequence[str],
    margins: np.ndarray | None = None,
) -> dict[str, trimesh.Trimesh]:
    """One mesh per structure present (class 1..K), in the aligned frame.

    With ``margins`` (from ``decode_margins``) the surface is the margin's zero level, found to
    sub-voxel precision; without, it is the label mask's boundary.
    """
    meshes: dict[str, trimesh.Trimesh] = {}
    for index, entity in enumerate(classes, start=1):
        mask = np.pad(volume == index, 1)
        if mask.sum() < 8:
            continue
        if margins is None:
            values, level = mask.astype(np.float32), 0.5
        else:
            values, level = np.pad(margins[index], 1, constant_values=-1.0), 0.0
        vertices, faces, _, _ = measure.marching_cubes(values, level)
        vertices = (vertices - 1.0) * grid.spacing + grid.low
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        # Marching cubes on an inside-high mask winds faces inward; Level 1 fixes them the same way.
        trimesh.repair.fix_normals(mesh)
        meshes[entity] = mesh
    return meshes


def surface_distance(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    samples: int = 4000,
    slab: tuple[float, float] | None = None,
    margin_mm: float = 2.0,
) -> float:
    """Symmetric mean surface distance in mm: the average of both directions.

    With ``slab`` (low, high along y), only surface points strictly inside it (by
    ``margin_mm``) are scored, on both meshes. A structure cropped to the lungs (the trachea,
    whose reference Level 1 does not crop) is then compared over the same stretch, and the
    flat faces where one mesh was cut never count against the other.
    """
    pa, _ = trimesh.sample.sample_surface(a, samples, seed=0)
    pb, _ = trimesh.sample.sample_surface(b, samples, seed=1)
    if slab is not None:
        low, high = slab[0] + margin_mm, slab[1] - margin_mm
        pa = pa[(pa[:, 1] >= low) & (pa[:, 1] <= high)]
        pb = pb[(pb[:, 1] >= low) & (pb[:, 1] <= high)]
        if len(pa) == 0 or len(pb) == 0:
            return float("nan")
    _, d_ab, _ = trimesh.proximity.closest_point(b, pa)
    _, d_ba, _ = trimesh.proximity.closest_point(a, pb)
    return float((np.mean(d_ab) + np.mean(d_ba)) / 2.0)


def _real_meshes(scenes_root: Path, subject: str, rigid: FloatArray) -> dict[str, trimesh.Trimesh]:
    meshes = {}
    for entity, mesh in load_scene(scenes_root / subject).items():
        moved = mesh.copy()
        moved.apply_transform(rigid)
        meshes[entity] = moved
    return meshes


def lung_slab(meshes: Mapping[str, trimesh.Trimesh]) -> tuple[float, float] | None:
    """The lowest and highest point of a thorax's lungs along the superior axis (y)."""
    lungs = [m for e, m in meshes.items() if "lobe" in e or e.endswith("_lung")]
    if not lungs:
        return None
    return (
        min(float(m.vertices[:, 1].min()) for m in lungs),
        max(float(m.vertices[:, 1].max()) for m in lungs),
    )


def _dice(predicted: np.ndarray, truth: np.ndarray, k: int) -> float | None:
    both = int(np.sum((predicted == k) & (truth == k)))
    total = int(np.sum(predicted == k) + np.sum(truth == k))
    return None if total == 0 else 2.0 * both / total


def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    scenes_root: Path,
    *,
    subjects: Sequence[str] | None = None,
    level2a_dir: Path | None = None,
    fit_steps: int = 300,
    fit_weighted: bool = False,
    spacing_mm: float = 2.0,
    device: str | None = None,
    raw_root: Path | None = None,
    smooth_meshes: bool = True,
) -> dict[str, Any]:
    """Per held-out subject and structure: surface distance and point Dice, for 2b and 2a.

    ``smooth_meshes=False`` skips the margin meshing, which never beat the binary one
    (notebook, 2026-10-18), and saves a third of the time with many structures.

    With ``raw_root`` (the CT label volumes), each structure also gets ``surface_mm_floor``:
    the subject's true labels decoded on the same grid and meshed the same way, scored by
    the same rule. No field can do better, and a structure with no Level 2a model is read
    against this floor.
    """
    run_device = pick_device(device)
    field, checkpoint = load_checkpoint(checkpoint_path, run_device)
    classes = list(checkpoint["classes"][1:])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if subjects is None:
        subjects = list((manifest.get("split") or {}).get("held_out", []))
    training = np.concatenate(
        [
            np.load(data_dir / "points" / f"{s}.npz")["points"][::20]
            for s in checkpoint["subjects"][:20]
        ]
    )
    grid = Grid(
        np.percentile(training, 0.5, axis=0) - 10,
        np.percentile(training, 99.5, axis=0) + 10,
        spacing_mm,
    )
    models_2a = _level2a(level2a_dir) if level2a_dir is not None else None
    weights = None
    if fit_weighted:
        counts = np.sum(
            [manifest["stats"][s]["point_label_counts"] for s in checkpoint["subjects"]], axis=0
        )
        weights = weights_from_counts(torch.as_tensor(counts)).to(run_device)
    results: dict[str, Any] = {}
    for subject in subjects:
        started = time.time()
        stored = np.load(data_dir / "points" / f"{subject}.npz")
        points, labels = stored["points"], stored["labels"]
        rigid = np.asarray(manifest["stats"][subject]["frame"]["rigid"], dtype=np.float64)
        code = fit_code(field, points, labels, steps=fit_steps, weights=weights)
        margins = decode_margins(field, code, grid)
        volume = apply_crops(margins.argmax(axis=0).astype(np.uint8), classes)
        meshes = label_meshes(volume, grid, classes)
        smooth = label_meshes(volume, grid, classes, margins) if smooth_meshes else {}
        cleaned, pieces = keep_largest(volume, len(classes) + 1)
        clean = label_meshes(cleaned, grid, classes)
        real = _real_meshes(scenes_root, subject, rigid)
        slab = lung_slab(real)

        def score(
            mesh: trimesh.Trimesh,
            entity: str,
            real: dict[str, trimesh.Trimesh] = real,
            slab: tuple[float, float] | None = slab,
        ) -> float:
            # Level 2b crops the trachea; its Level 1 reference is not cropped.
            within = slab if entity in EXTRA_CROPS else None
            return round(surface_distance(mesh, real[entity], slab=within), 3)

        predicted = (
            field(
                torch.from_numpy(points[:50000]).unsqueeze(0).to(run_device), code.to(run_device)
            )[0]
            .argmax(-1)
            .cpu()
            .numpy()
        )
        entry: dict[str, Any] = {}
        for index, entity in enumerate(classes, start=1):
            row: dict[str, Any] = {
                "dice_points": _dice(predicted, labels[:50000], index),
                "pieces": pieces[index - 1],
            }
            if entity in meshes and entity in real:
                row["surface_mm_2b"] = score(meshes[entity], entity)
                if entity in smooth:
                    row["surface_mm_2b_smooth"] = score(smooth[entity], entity)
                row["surface_mm_2b_clean"] = score(clean[entity], entity)
            if models_2a is not None and entity in models_2a and entity in real:
                rebuilt = models_2a[entity](subject)
                if rebuilt is not None:
                    row["surface_mm_2a"] = score(rebuilt, entity)
            entry[entity] = row
        if raw_root is not None:
            from fields.data import labels_on_grid

            truth = labels_on_grid(
                raw_root / subject, manifest["stats"][subject]["frame"], grid.points(), classes
            )
            perfect = label_meshes(truth.reshape(grid.shape), grid, classes)
            for entity in classes:
                if entity in perfect and entity in real:
                    entry[entity]["surface_mm_floor"] = score(perfect[entity], entity)
        entry["seconds"] = round(time.time() - started, 1)
        results[subject] = entry
        print(f"evaluated {subject} in {entry['seconds']} s", flush=True)
    return {
        "checkpoint": str(checkpoint_path),
        "classes": classes,
        "grid": {"low": grid.low.tolist(), "spacing_mm": grid.spacing, "shape": list(grid.shape)},
        "fit_steps": fit_steps,
        "fit_weighted": fit_weighted,
        "subjects": results,
        "summary": _summary(results, classes),
    }


def _level2a(level2a_dir: Path) -> dict[str, Any]:
    """Per entity, a function from subject to its Level 2a reconstruction mesh (or None)."""
    from shapes.ssm import ShapeModel

    rebuilders: dict[str, Any] = {}
    for model_path in sorted((level2a_dir / "models").glob("*.npz")):
        entity = model_path.stem
        if entity == "joint" or entity.endswith("_faces"):
            continue
        model = ShapeModel.load(model_path)

        def rebuild(subject: str, model: Any = model, entity: str = entity) -> Any:
            path = level2a_dir / "registrations" / f"{subject}__{entity}.npz"
            if not path.exists():
                return None
            vertices = model.reconstruct(np.load(path)["vertices"])
            return trimesh.Trimesh(vertices, model.faces, process=False)

        rebuilders[entity] = rebuild
    return rebuilders


SUMMARISED = (
    "surface_mm_floor",
    "surface_mm_2b",
    "surface_mm_2b_smooth",
    "surface_mm_2b_clean",
    "surface_mm_2a",
    "dice_points",
    "pieces",
)


def _summary(results: Mapping[str, Any], classes: Sequence[str]) -> dict[str, Any]:
    """Per structure, the mean of each measure (its plain name) and the median (``_median``)."""
    summary: dict[str, Any] = {}
    for entity in classes:
        rows = [r[entity] for r in results.values() if entity in r]
        entry: dict[str, Any] = {}
        for key in SUMMARISED:
            values = [row[key] for row in rows if row.get(key) is not None]
            entry[key] = round(float(np.mean(values)), 3) if values else None
            entry[f"{key}_median"] = round(float(np.median(values)), 3) if values else None
        summary[entity] = entry
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate a Level 2b label field.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--level2a", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--fit-weighted", action="store_true", help="training's class weights")
    parser.add_argument("--floor", type=Path, default=None, help="CT labels: score the floor")
    parser.add_argument("--no-smooth", action="store_true", help="skip the margin meshing")
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(
        args.checkpoint,
        args.data,
        args.scenes,
        subjects=args.subjects,
        level2a_dir=args.level2a,
        fit_steps=args.fit_steps,
        fit_weighted=args.fit_weighted,
        raw_root=args.floor,
        smooth_meshes=not args.no_smooth,
        spacing_mm=args.spacing,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entity, row in report["summary"].items():
        print(f"{entity:26s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
