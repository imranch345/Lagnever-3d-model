"""New thoraxes from a Level 2b field, judged as Level 2a's samples are judged.

**Sampling.** A Gaussian is fitted to the training codes (mean and principal axes). A draw at
``spread`` SD is the mean plus ``spread`` times N(0, 1) weights on the axes, clipped at
``CLIP``, as in Level 2a. Each draw is decoded on the evaluation grid.

**Checks.** Each can fail, and limits come from real subjects as "no worse than the worst of
them":
* **volume:** every structure within the range the training subjects' CT volumes span, with no
  margin;
* **sides:** lungs either side of the midline, left chambers left of right ones (Level 1's
  rule);
* **stray:** the share of a structure's decoded volume outside its largest piece. The largest
  piece is then kept, as in evaluation. The limit is the worst share among real subjects,
  plus ``MARGIN``, and comes from one of two sources:
  - **model** (the first version): the training subjects' own decoded codes. This ties the
    limit to the model, so a cleaner model gets a stricter limit;
  - **real** (``real_stray``): the training subjects' CT label volumes, resampled on the same
    grid. This is Level 2a's rule, calibrated on real anatomy and the same for every model.
    It is the default when the raw labels are given.
* **closed:** every meshed structure is watertight;
* **overlap:** zero by construction (one label per point), and verified on every sample from
  the margins.

**Novelty.** For each structure, the symmetric mean distance between surface samples of the
thorax and of a training subject's real Level 1 mesh. These are averaged over structures,
and the nearest training subject's value is the novelty. The same number for each held-out
*real* subject gives the baseline: how far a new real person is from the training set. Samples
should be about as novel as that. Much less would mean copies of training subjects.

**Scenes.** ``SampleWriter`` turns a valid draw into a Level 1 scene, so the viewer, AWR and
language layer work unchanged. It writes a GLB with one mesh per structure, made from the
cleaned labels by Level 1's own ``mask_to_mesh`` (smoothing, decimation), and a manifest with
the "generated" record and the verdict. As in Level 2a, a seed tries up to ``MAX_TRIES``
draws and keeps the first that passes; the limits come from the report this module writes.

    python -m fields.generate --checkpoint runs/gpu/m2-h512-focus-prior/checkpoint.pt
        --data datasets/processed/level2b --scenes datasets/processed/level1/real
        --samples 50 --out runs/gpu/m2-h512-focus-prior/samples.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import trimesh
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from assets.meshes import VIEWER_FRAME, mask_to_mesh, side_report
from assets.quality import assess
from awr.ontology import AnatomyOntology, load_ontology
from fields.evaluate import (
    Grid,
    _real_meshes,
    decode_margins,
    keep_largest,
    label_meshes,
    load_checkpoint,
)
from fields.model import Field, pick_device

__all__ = [
    "CLIP",
    "MARGIN",
    "MAX_TRIES",
    "SampleWriter",
    "CodeGaussian",
    "Reference",
    "Thorax",
    "decode_thorax",
    "generate",
    "judge",
    "real_stray",
    "rejudge_stray",
    "write_scene",
]

CLIP = 3.0
MARGIN = 0.01
#: Draws a seed tries before giving up (Level 2a uses the same number).
MAX_TRIES = 6
MEDICAL_STATUS = (
    "generated anatomy from a neural label field; not a measurement of anyone; "
    "not validated; not for clinical use"
)
SURFACE_SAMPLES = 3000

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CodeGaussian:
    """A Gaussian over codes: the mean and the principal axes scaled by their SD."""

    mean: FloatArray
    axes: FloatArray

    @classmethod
    def fit(cls, codes: npt.ArrayLike) -> CodeGaussian:
        """From training codes (S, D). Axes with no spread are dropped (S − 1 < D is fine)."""
        codes = np.asarray(codes, dtype=np.float64)
        centred = codes - codes.mean(axis=0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        sd = singular / np.sqrt(max(len(codes) - 1, 1))
        keep = sd > 1e-6 * sd.max()
        return cls(mean=codes.mean(axis=0), axes=sd[keep, None] * vt[keep])

    def draw(self, seed: int, spread: float = 1.0, clip: float = CLIP) -> FloatArray:
        """One code, reproducible from ``seed``."""
        weights = np.random.default_rng(seed).standard_normal(len(self.axes))
        return np.asarray(self.mean + spread * np.clip(weights, -clip, clip) @ self.axes)


@dataclass
class Thorax:
    """One decoded thorax: its cleaned label volume, meshes and measurements."""

    volume: np.ndarray
    meshes: dict[str, trimesh.Trimesh]
    measures: dict[str, Any]


def decode_thorax(field: Field, code: torch.Tensor, grid: Grid, classes: Sequence[str]) -> Thorax:
    """Decode, measure stray pieces, keep each structure's largest piece, mesh and measure."""
    margins = decode_margins(field, code, grid)
    labels = margins.argmax(axis=0).astype(np.uint8)
    claims = int(((margins > 0).sum(axis=0) > 1).sum())
    cleaned, pieces = keep_largest(labels, len(classes) + 1)
    voxel_ml = grid.spacing**3 / 1000.0
    meshes = label_meshes(cleaned, grid, classes)
    coords = grid.points().reshape(*grid.shape, 3)
    measures: dict[str, Any] = {
        "points_claimed_twice": claims,
        "pieces": dict(zip(classes, pieces, strict=True)),
        "stray": {},
        "volume_ml": {},
        "centroids": {},
        "closed": {e: bool(m.is_watertight) for e, m in meshes.items()},
    }
    for index, entity in enumerate(classes, start=1):
        before = int((labels == index).sum())
        after = int((cleaned == index).sum())
        measures["stray"][entity] = 0.0 if before == 0 else 1.0 - after / before
        measures["volume_ml"][entity] = after * voxel_ml
        if after:
            measures["centroids"][entity] = coords[cleaned == index].mean(axis=0).tolist()
    return Thorax(volume=cleaned, meshes=meshes, measures=measures)


def real_stray(
    raw_root: Path,
    stats: Mapping[str, Any],
    subjects: Sequence[str],
    grid: Grid,
    classes: Sequence[str],
) -> dict[str, list[float]]:
    """Per structure, each real subject's stray share: its CT labels on ``grid``, then cleaned."""
    import nibabel as nib

    from assets.meshes import RAS_TO_VIEWER
    from fields.data import label_volume

    shares: dict[str, list[float]] = {e: [] for e in classes}
    aligned = grid.points().astype(np.float64)
    for subject in subjects:
        labels, affine, _ = label_volume(raw_root / subject, classes)
        frame = stats[subject]["frame"]
        rigid = np.asarray(frame["rigid"], dtype=np.float64)
        viewer = (aligned - rigid[:3, 3]) @ rigid[:3, :3]
        ras = viewer @ RAS_TO_VIEWER + np.asarray(frame["origin_ras_mm"], dtype=np.float64)
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), ras)).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(labels.shape)), axis=1)
        truth = np.zeros(len(ijk), dtype=np.uint8)
        truth[inside] = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
        volume = truth.reshape(grid.shape)
        cleaned, _ = keep_largest(volume, len(classes) + 1)
        for index, entity in enumerate(classes, start=1):
            before = int((volume == index).sum())
            shares[entity].append(
                0.0 if before == 0 else 1 - int((cleaned == index).sum()) / before
            )
    return shares


def rejudge_stray(report: Mapping[str, Any], stray: Mapping[str, float]) -> dict[str, Any]:
    """The report's samples judged again with other stray limits, from their stored shares."""
    rows = []
    for row in report["samples"]:
        failures = [f for f in row["failures"] if not f.startswith("stray:")]
        failures += [
            f"stray: {e} {share:.1%} outside its largest piece"
            for e, share in row["stray"].items()
            if share > stray[e]
        ]
        rows.append({**row, "failures": failures, "passes": not failures})
    kinds = sorted({f.split(":")[0] for r in rows for f in r["failures"]})
    return {
        "pass_rate": sum(r["passes"] for r in rows) / len(rows),
        "failing_by_kind": {
            k: sum(any(f.startswith(k) for f in r["failures"]) for r in rows) / len(rows)
            for k in kinds
        },
        "samples": rows,
    }


def judge(
    measured: Mapping[str, Any],
    volume_ml: Mapping[str, tuple[float, float]],
    stray: Mapping[str, float],
    ontology: AnatomyOntology,
) -> list[tuple[str, str]]:
    """Every failed check as (kind, message); empty means plausible by these checks."""
    failures: list[tuple[str, str]] = []
    if measured["points_claimed_twice"]:
        failures.append(("overlap", f"{measured['points_claimed_twice']} points claimed twice"))
    for entity, (low, high) in volume_ml.items():
        volume = measured["volume_ml"].get(entity, 0.0)
        if not low <= volume <= high:
            failures.append(("volume", f"{entity} {volume:.0f} mL outside {low:.0f}-{high:.0f}"))
    for entity, share in measured["stray"].items():
        if share > stray[entity]:
            failures.append(("stray", f"{entity} {share:.1%} outside its largest piece"))
    for entity, closed in measured["closed"].items():
        if not closed:
            failures.append(("closed", f"{entity} not closed"))
    missing = [e for e in volume_ml if e not in measured["centroids"]]
    if missing:
        failures.append(("missing", f"no {missing}"))
    wrong = [e for e, ok in side_report(measured["centroids"], ontology).items() if not ok]
    if wrong:
        failures.append(("sides", f"wrong side {wrong}"))
    return failures


class Reference:
    """Surface samples of each training subject's real meshes, for nearest-subject distances."""

    def __init__(
        self,
        scenes_root: Path,
        subjects: Sequence[str],
        frames: Mapping[str, Any],
        classes: Sequence[str],
    ) -> None:
        self.subjects = list(subjects)
        self.classes = list(classes)
        self.trees: dict[str, dict[str, Any]] = {}
        self.points: dict[str, dict[str, FloatArray]] = {}
        for subject in self.subjects:
            rigid = np.asarray(frames[subject]["frame"]["rigid"], dtype=np.float64)
            real = _real_meshes(scenes_root, subject, rigid)
            self.points[subject] = {e: _surface(real[e]) for e in self.classes if e in real}
            self.trees[subject] = {e: cKDTree(p) for e, p in self.points[subject].items()}

    def novelty(self, meshes: Mapping[str, trimesh.Trimesh]) -> tuple[float, str]:
        """The mean over structures of the symmetric distance to the nearest training subject."""
        mine = {e: _surface(meshes[e]) for e in self.classes if e in meshes}
        my_trees = {e: cKDTree(p) for e, p in mine.items()}
        best, nearest = np.inf, ""
        for subject in self.subjects:
            shared = [e for e in mine if e in self.trees[subject]]
            distances = [
                (
                    self.trees[subject][e].query(mine[e])[0].mean()
                    + my_trees[e].query(self.points[subject][e])[0].mean()
                )
                / 2.0
                for e in shared
            ]
            value = float(np.mean(distances))
            if value < best:
                best, nearest = value, subject
        return best, nearest


def _surface(mesh: trimesh.Trimesh) -> FloatArray:
    points, _ = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)
    return np.asarray(points, dtype=np.float64)


def generate(
    checkpoint_path: Path,
    data_dir: Path,
    scenes_root: Path,
    *,
    samples: int = 50,
    spread: float = 1.0,
    first_seed: int = 0,
    calibration: int | None = None,
    spacing_mm: float = 2.0,
    device: str | None = None,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    """Draw, decode, judge and measure novelty; with the limits and the real baseline.

    With ``raw_root`` (the CT label volumes), stray limits come from real anatomy
    (``real_stray``); the model-based limits are still recorded beside them.
    """
    started = time.time()
    run_device = pick_device(device)
    field, checkpoint = load_checkpoint(checkpoint_path, run_device)
    classes = list(checkpoint["classes"][1:])
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["stats"]
    train = list(checkpoint["subjects"])
    held_out = list((manifest.get("split") or {}).get("held_out", []))
    ontology = load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    codes = checkpoint["codes"].numpy().astype(np.float64)
    gaussian = CodeGaussian.fit(codes)
    training = np.concatenate(
        [np.load(data_dir / "points" / f"{s}.npz")["points"][::20] for s in train[:20]]
    )
    grid = Grid(
        np.percentile(training, 0.5, axis=0) - 10,
        np.percentile(training, 99.5, axis=0) + 10,
        spacing_mm,
    )

    def as_code(vector: npt.ArrayLike) -> torch.Tensor:
        return torch.as_tensor(np.asarray(vector), dtype=torch.float32).unsqueeze(0)

    volume_limits = {
        e: (
            min(stats[s]["volume_ml"][e] for s in train),
            max(stats[s]["volume_ml"][e] for s in train),
        )
        for e in classes
    }
    calibrated = train if calibration is None else train[:calibration]
    own = [decode_thorax(field, as_code(codes[train.index(s)]), grid, classes) for s in calibrated]
    model_stray = {e: max(t.measures["stray"][e] for t in own) + MARGIN for e in classes}
    stray_limits = model_stray
    stray_from = f"{len(own)} training subjects' own codes, plus {MARGIN:.0%}"
    if raw_root is not None:
        shares = real_stray(raw_root, stats, train, grid, classes)
        stray_limits = {e: max(v) + MARGIN for e, v in shares.items()}
        stray_from = f"{len(train)} training subjects' CT labels on this grid, plus {MARGIN:.0%}"
    print(f"limits from {len(own)} training decodes in {time.time() - started:.0f} s", flush=True)

    reference = Reference(scenes_root, train, stats, classes)
    baseline = {}
    for subject in held_out:
        rigid = np.asarray(stats[subject]["frame"]["rigid"], dtype=np.float64)
        value, nearest = reference.novelty(_real_meshes(scenes_root, subject, rigid))
        baseline[subject] = {"novelty_mm": round(value, 3), "nearest": nearest}
    control = [judge(t.measures, volume_limits, stray_limits, ontology) for t in own]

    rows = []
    for seed in range(first_seed, first_seed + samples):
        thorax = decode_thorax(field, as_code(gaussian.draw(seed, spread)), grid, classes)
        failures = judge(thorax.measures, volume_limits, stray_limits, ontology)
        value, nearest = reference.novelty(thorax.meshes)
        rows.append(
            {
                "seed": seed,
                "passes": not failures,
                "failures": [f"{kind}: {message}" for kind, message in failures],
                "novelty_mm": round(value, 3),
                "nearest": nearest,
                "volume_ml": {e: round(v, 1) for e, v in thorax.measures["volume_ml"].items()},
                "stray": {e: round(v, 4) for e, v in thorax.measures["stray"].items()},
                "points_claimed_twice": thorax.measures["points_claimed_twice"],
            }
        )
        print(
            f"seed {seed}: {'valid' if not failures else failures[0][1]}; novelty {value:.2f} mm",
            flush=True,
        )
    kinds = sorted({f.split(":")[0] for r in rows for f in r["failures"]})
    return {
        "checkpoint": str(checkpoint_path),
        "spread": spread,
        "clip": CLIP,
        "code_axes": len(gaussian.axes),
        "grid": {"low": grid.low.tolist(), "spacing_mm": grid.spacing, "shape": list(grid.shape)},
        "limits": {
            "volume_ml": volume_limits,
            "stray": stray_limits,
            "stray_from": stray_from,
            "stray_model": model_stray,
        },
        "control": {
            "training_decodes_passing": sum(not c for c in control) / len(control),
            "failures": [[f"{k}: {m}" for k, m in c] for c in control if c],
        },
        "baseline": {
            "novelty_mm_median": round(
                float(np.median([b["novelty_mm"] for b in baseline.values()])), 3
            ),
            "subjects": baseline,
        },
        "summary": {
            "samples": len(rows),
            "pass_rate": sum(r["passes"] for r in rows) / len(rows),
            "failing_by_kind": {
                kind: sum(any(f.startswith(kind) for f in r["failures"]) for r in rows) / len(rows)
                for kind in kinds
            },
            "novelty_mm_median": round(float(np.median([r["novelty_mm"] for r in rows])), 3),
            "novelty_mm_median_valid": (
                round(float(np.median([r["novelty_mm"] for r in rows if r["passes"]])), 3)
                if any(r["passes"] for r in rows)
                else None
            ),
            "points_claimed_twice": sum(r["points_claimed_twice"] for r in rows),
        },
        "samples": rows,
        "seconds": round(time.time() - started, 1),
    }


class SampleWriter:
    """Valid Level 2b thoraxes written as Level 1 scenes, for the viewer.

    ``report`` is this module's JSON report for the same checkpoint; its limits are reused, so
    a scene is judged exactly as the measured samples were.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        data_dir: Path,
        report: Path,
        *,
        device: str | None = None,
        ontology: AnatomyOntology | None = None,
    ) -> None:
        self.field, checkpoint = load_checkpoint(checkpoint_path, pick_device(device))
        self.checkpoint = checkpoint_path
        self.classes = list(checkpoint["classes"][1:])
        self.gaussian = CodeGaussian.fit(checkpoint["codes"].numpy())
        self.training_subjects = len(checkpoint["subjects"])
        measured = json.loads(report.read_text(encoding="utf-8"))
        if Path(measured["checkpoint"]).name != checkpoint_path.name:
            raise ValueError(f"{report} was measured for {measured['checkpoint']}")
        grid = measured["grid"]
        high = np.asarray(grid["low"]) + grid["spacing_mm"] * (np.asarray(grid["shape"]) - 1)
        self.grid = Grid(grid["low"], high, grid["spacing_mm"])
        self.volume_ml = {e: tuple(v) for e, v in measured["limits"]["volume_ml"].items()}
        self.stray = dict(measured["limits"]["stray"])
        self.ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
        source = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))["source"]
        self.source = source

    def write(self, seed: int, out_dir: Path, spread: float = 1.0) -> dict[str, Any]:
        """The first valid draw of ``seed``'s ``MAX_TRIES``, written; or an error if none is."""
        rejected: list[list[str]] = []
        for attempt in range(MAX_TRIES):
            code = self.gaussian.draw(seed * 1000 + attempt, spread)
            thorax = decode_thorax(
                self.field,
                torch.as_tensor(code, dtype=torch.float32).unsqueeze(0),
                self.grid,
                self.classes,
            )
            failures = judge(thorax.measures, self.volume_ml, self.stray, self.ontology)
            if not failures:
                generated = {
                    "model": "Level 2b neural label field",
                    "checkpoint": self.checkpoint.name,
                    "seed": seed,
                    "spread": spread,
                    "code": [round(float(c), 6) for c in code],
                    "training_subjects": self.training_subjects,
                    "sampler": {
                        "kind": f"gaussian over codes at {spread:g} SD, kept only if valid",
                        "tries": attempt + 1,
                        "rejected": rejected,
                    },
                }
                return write_scene(
                    thorax.volume,
                    self.grid,
                    self.classes,
                    out_dir,
                    subject=f"sample-2b-{seed}",
                    generated=generated,
                    source=self.source,
                    ontology=self.ontology,
                )
            rejected.append([message for _, message in failures][:3])
        raise RuntimeError(f"no valid thorax in {MAX_TRIES} draws for seed {seed}: {rejected}")


def write_scene(
    volume: np.ndarray,
    grid: Grid,
    classes: Sequence[str],
    out_dir: Path,
    *,
    subject: str,
    generated: Mapping[str, Any],
    source: Mapping[str, Any],
    ontology: AnatomyOntology,
) -> dict[str, Any]:
    """A cleaned label volume as a Level 1 scene: ``scene.glb`` and ``manifest.json``."""
    affine = np.diag([grid.spacing, grid.spacing, grid.spacing, 1.0])
    affine[:3, 3] = grid.low
    scene = trimesh.Scene()
    entities: dict[str, Any] = {}
    centroids: dict[str, list[float]] = {}
    for index, entity_id in enumerate(classes, start=1):
        mask = volume == index
        if not mask.any():
            continue
        mesh, quality = mask_to_mesh(mask, affine)
        quality.voxel_components = 1  # each structure was cut to its largest piece
        entities[entity_id] = {
            "node": entity_id,
            "centroid_viewer_mm": quality.centroid_mm,
            "quality": asdict(quality),
        }
        centroids[entity_id] = quality.centroid_mm
        scene.add_geometry(mesh, node_name=entity_id, geom_name=entity_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.glb").write_bytes(scene.export(file_type="glb"))
    renderable = [e for e in ontology.ids() if ontology.get(e).renderable]
    manifest: dict[str, Any] = {
        "manifest_version": "0.1.0",
        "subject": subject,
        "scene": "scene.glb",
        "frame": VIEWER_FRAME,
        "generated": dict(generated),
        "ontology": {"ontology_id": ontology.ontology_id, "version": ontology.version},
        "source": {
            "dataset": source.get("dataset", ""),
            "version": source.get("version", ""),
            "license": source.get("license", ""),
            "attribution": source.get("attribution", ""),
            "url": source.get("url", ""),
            "derived": "sampled from a model trained on the listed dataset",
        },
        "entities": entities,
        "missing": [],
        "no_geometry": [e for e in renderable if e not in entities],
        "sides_correct": side_report(centroids, ontology),
        "medical_status": MEDICAL_STATUS,
        "validity": {"passes": True, "failures": []},
    }
    manifest["checks"] = asdict(assess(manifest))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and judge from the command line."""
    parser = argparse.ArgumentParser(description="Generate and judge Level 2b thoraxes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--calibration", type=int, default=None, help="training decodes")
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw", type=Path, default=None, help="CT labels: real stray limits")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = generate(
        args.checkpoint,
        args.data,
        args.scenes,
        samples=args.samples,
        spread=args.spread,
        first_seed=args.first_seed,
        calibration=args.calibration,
        spacing_mm=args.spacing,
        device=args.device,
        raw_root=args.raw,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"real-person novelty, median: {report['baseline']['novelty_mm_median']} mm")
    print(f"training decodes passing: {report['control']['training_decodes_passing']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
