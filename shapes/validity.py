"""How far from the mean generated thoraxes stay plausible, by checks that can fail.

A deformed template is closed and in one piece by construction, so the Level 1 hard checks
(closed, one piece, correct side) pass almost every sample and say little on their own. Three
checks that can fail are added. Each is calibrated on the real training subjects as "no worse
than the worst of them":

* **folding**: the share of an entity's faces whose normal points against the mean shape's,
  meaning part of the surface has turned inside out;
* **overlap**: for every ordered pair of entities, the share of one's vertices that lie more
  than ``DEPTH_MM`` inside the other, meaning structures run into each other. Neighbours touch:
  the left ventricle lies against the myocardium and the heart against the lungs. A vertex on
  a shared wall falls on either side, and decimation moves surfaces by about a millimetre. On
  the reference subject, 58% of the ventricle's vertices counted as inside the myocardium
  before the depth was required. The depth matches the registration fit threshold;
* **volume**: each entity's volume within the range the training subjects span.

Folding and overlap may exceed the worst training subject by ``MARGIN`` (one percentage
point). Volume has no margin.

Held-out real subjects go through the same checks as a control. If they fail, the checks are
too strict; the samples are not to blame.

For each spread (the multiplier on the N(0, 1) weights, clipped at ``CLIP``), the sweep
reports the share of samples passing every check and the share failing each kind. The same
seeds are used at every spread, so the samples differ only in how far they go. The usable
range is the largest spread up to which every pass rate is at least ``TARGET``.

    python -m shapes.validity --models datasets/processed/level2a
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh

from assets.meshes import side_report
from awr.ontology import AnatomyOntology, load_ontology
from shapes.generate import draw_coefficients
from shapes.joint import JointModel

__all__ = [
    "CLIP",
    "DEPTH_MM",
    "MARGIN",
    "TARGET",
    "Limits",
    "calibrate",
    "judge",
    "measure",
    "sweep",
]

FloatArray = npt.NDArray[np.float64]

MARGIN = 0.01
DEPTH_MM = 2.0
TARGET = 0.95
CLIP = 3.0
DEFAULT_SPREADS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
KINDS: tuple[str, ...] = ("level1", "folding", "overlap", "volume")


def face_normals(vertices: FloatArray, faces: npt.ArrayLike) -> FloatArray:
    """Unit normal per face; zero-area faces get a zero vector."""
    triangles = np.asarray(vertices)[np.asarray(faces)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.asarray(normals / np.maximum(lengths, 1e-12), dtype=np.float64)


def measure(
    shapes: Mapping[str, FloatArray],
    faces: Mapping[str, npt.ArrayLike],
    mean_normals: Mapping[str, FloatArray],
) -> dict[str, Any]:
    """Everything the checks read, for one thorax given as vertices per entity."""
    meshes = {e: trimesh.Trimesh(v, np.asarray(faces[e]), process=False) for e, v in shapes.items()}
    folding = {
        e: float(np.mean(np.sum(face_normals(v, faces[e]) * mean_normals[e], axis=1) < 0))
        for e, v in shapes.items()
    }
    overlap: dict[str, float] = {}
    for a, mesh_a in meshes.items():
        low_a, high_a = mesh_a.bounds
        for b, mesh_b in meshes.items():
            if a == b:
                continue
            low_b, high_b = mesh_b.bounds
            if np.any(high_a < low_b) or np.any(high_b < low_a):
                overlap[f"{a}|{b}"] = 0.0
            else:
                inside = np.asarray(mesh_b.contains(mesh_a.vertices), dtype=bool)
                deep = 0
                if inside.any():
                    _, depth, _ = trimesh.proximity.closest_point(mesh_b, mesh_a.vertices[inside])
                    deep = int(np.sum(np.asarray(depth) > DEPTH_MM))
                overlap[f"{a}|{b}"] = deep / len(mesh_a.vertices)
    return {
        "folding": folding,
        "overlap": overlap,
        "volume_ml": {e: float(abs(m.volume)) / 1000.0 for e, m in meshes.items()},
        "closed": {e: bool(m.is_watertight) for e, m in meshes.items()},
        "pieces": {e: len(m.split(only_watertight=False)) for e, m in meshes.items()},
        "centroids": {e: [float(x) for x in m.centroid] for e, m in meshes.items()},
    }


@dataclass(frozen=True)
class Limits:
    """What the training subjects allow: the worst of them, plus the margin where stated."""

    folding: dict[str, float]
    overlap: dict[str, float]
    volume_ml: dict[str, tuple[float, float]]


def calibrate(measures: Sequence[Mapping[str, Any]], margin: float = MARGIN) -> Limits:
    """Limits from the training subjects' own measurements."""
    if not measures:
        raise ValueError("calibration needs at least one training subject")
    first = measures[0]
    return Limits(
        folding={e: max(m["folding"][e] for m in measures) + margin for e in first["folding"]},
        overlap={p: max(m["overlap"][p] for m in measures) + margin for p in first["overlap"]},
        volume_ml={
            e: (min(m["volume_ml"][e] for m in measures), max(m["volume_ml"][e] for m in measures))
            for e in first["volume_ml"]
        },
    )


def judge(
    measured: Mapping[str, Any], limits: Limits, ontology: AnatomyOntology
) -> list[tuple[str, str]]:
    """Every failed check as (kind, message); empty means plausible by these checks."""
    failures: list[tuple[str, str]] = []
    for entity, closed in measured["closed"].items():
        if not closed:
            failures.append(("level1", f"{entity} not closed"))
        if measured["pieces"][entity] != 1:
            failures.append(("level1", f"{entity} in {measured['pieces'][entity]} pieces"))
    wrong = [e for e, ok in side_report(measured["centroids"], ontology).items() if not ok]
    if wrong:
        failures.append(("level1", f"wrong side {wrong}"))
    for entity, share in measured["folding"].items():
        if share > limits.folding[entity]:
            failures.append(("folding", f"{entity} {share:.1%} of faces folded"))
    for pair, share in measured["overlap"].items():
        if share > limits.overlap[pair]:
            a, b = pair.split("|")
            failures.append(("overlap", f"{share:.1%} of {a} inside {b}"))
    for entity, volume in measured["volume_ml"].items():
        low, high = limits.volume_ml[entity]
        if not low <= volume <= high:
            failures.append(("volume", f"{entity} {volume:.0f} mL outside {low:.0f}-{high:.0f}"))
    return failures


def sweep(
    joint: JointModel,
    train: Mapping[str, Mapping[str, FloatArray]],
    held_out: Mapping[str, Mapping[str, FloatArray]],
    *,
    spreads: Sequence[float] = DEFAULT_SPREADS,
    samples: int = 50,
    seed: int = 0,
    ontology: AnatomyOntology | None = None,
) -> dict[str, Any]:
    """Calibrate on ``train``, control on ``held_out``, then sample at every spread."""
    ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    mean = joint.split(joint.model.mean)
    mean_normals = {e: face_normals(mean[e], joint.faces[e]) for e in joint.entities}

    def check(shapes: Mapping[str, FloatArray]) -> list[tuple[str, str]]:
        return judge(measure(shapes, joint.faces, mean_normals), limits, ontology)

    limits = calibrate([measure(s, joint.faces, mean_normals) for s in train.values()])
    control = {subject: check(shapes) for subject, shapes in held_out.items()}
    rows: list[dict[str, Any]] = []
    for spread in spreads:
        results = [
            check(
                joint.sample_entities(
                    draw_coefficients(joint.model.n_modes, seed + i, spread, clip=CLIP)
                )
            )
            for i in range(samples)
        ]
        failed = [r for r in results if r]
        rows.append(
            {
                "spread": spread,
                "samples": samples,
                "pass_rate": round(1 - len(failed) / samples, 3),
                "failing": {
                    kind: round(sum(any(k == kind for k, _ in r) for r in results) / samples, 3)
                    for kind in KINDS
                },
                "examples": [message for r in failed[:3] for _, message in r[:2]],
            }
        )
    usable = None
    for row in rows:
        if row["pass_rate"] < TARGET:
            break
        usable = row["spread"]
    return {
        "method": {
            "margin": MARGIN,
            "depth_mm": DEPTH_MM,
            "target": TARGET,
            "clip": CLIP,
            "seed": seed,
            "calibrated_on": len(train),
        },
        "limits": asdict(limits),
        "control": {
            "subjects": len(control),
            "pass_rate": round(sum(not f for f in control.values()) / len(control), 3)
            if control
            else None,
            "failures": {s: [m for _, m in f] for s, f in control.items() if f},
        },
        "spreads": rows,
        "usable_spread": usable,
    }


def _load_shapes(cache: Path, subject: str, entities: Sequence[str]) -> dict[str, FloatArray]:
    return {
        e: np.asarray(np.load(cache / f"{subject}__{e}.npz")["vertices"], dtype=np.float64)
        for e in entities
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep the joint model built by ``shapes.build`` and write ``validity.json``."""
    parser = argparse.ArgumentParser(description="Validity of generated thoraxes by spread.")
    parser.add_argument("--models", type=Path, required=True, help="shapes.build output")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    joint = JointModel.load(args.models / "models" / "joint.npz")
    report = json.loads((args.models / "report.json").read_text(encoding="utf-8"))
    excluded = {s for e in joint.entities for s in report["entities"][e]["excluded"]}
    cache = args.models / "registrations"
    train = {s: _load_shapes(cache, s, joint.entities) for s in joint.model.subjects}
    held = {
        s: _load_shapes(cache, s, joint.entities) for s in report["held_out"] if s not in excluded
    }
    result = sweep(joint, train, held, samples=args.samples, seed=args.seed)
    (args.models / "validity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    control = result["control"]
    print(f"control: {control['pass_rate']} of {control['subjects']} held-out real subjects pass")
    for row in result["spreads"]:
        print(f"spread {row['spread']}: pass {row['pass_rate']}, failing {row['failing']}")
    print(f"usable spread (pass rate >= {TARGET}): {result['usable_spread']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
