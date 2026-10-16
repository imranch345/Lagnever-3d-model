"""Write a generated thorax as a Level 1 scene: one GLB node per entity, and a manifest.

A sample is drawn from the joint model with a seed: weights from N(0, 1), scaled, and clipped
so it stays inside the range the model was validated for. The meshes are already in the
reference subject's viewer frame, because registration happened there. The manifest has the
same shape as a built subject's, so the viewer, the state export and the quality report work
unchanged -- and it says plainly that this is generated anatomy, from which model, seed and
weights, trained on which data under which license.

    python -m shapes.generate --model datasets/processed/level2a/models/joint.npz
        --out datasets/processed/level2a/samples/seed-7 --seed 7
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import trimesh

from assets.meshes import VIEWER_FRAME, MeshQuality, side_report
from assets.quality import assess
from awr.ontology import AnatomyOntology, load_ontology
from shapes.joint import JointModel

__all__ = [
    "draw_coefficients",
    "entity_entry",
    "model_mesh",
    "write_sample_scene",
    "write_valid_sample_scene",
]

MEDICAL_STATUS = (
    "generated anatomy from a statistical shape model; not a measurement of anyone; "
    "not validated; not for clinical use"
)


def draw_coefficients(
    n_modes: int, seed: int, spread: float = 1.0, clip: float = 2.5
) -> list[float]:
    """Weights in standard deviations, reproducible from the seed."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in np.clip(rng.standard_normal(n_modes) * spread, -clip, clip)]


def model_mesh(vertices: npt.ArrayLike, faces: npt.ArrayLike) -> trimesh.Trimesh:
    """A mesh from model vertices and the template's faces, normals made consistent."""
    mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(faces), process=False)
    trimesh.repair.fix_normals(mesh)
    return mesh


def entity_entry(entity_id: str, mesh: trimesh.Trimesh) -> dict[str, Any]:
    """A manifest entry for a model-made mesh, measured like a built subject's."""
    watertight = bool(mesh.is_watertight)
    centroid = [float(v) for v in mesh.centroid]
    pieces = len(mesh.split(only_watertight=False))
    quality = MeshQuality(
        faces=len(mesh.faces),
        vertices=len(mesh.vertices),
        watertight=watertight,
        components=pieces,
        # No voxel mask exists for a model surface; a deformed template is one closed
        # surface, so its connectedness is its count of surface pieces.
        voxel_components=pieces,
        volume_ml=float(abs(mesh.volume)) / 1000.0 if watertight else None,
        centroid_mm=centroid,
        bounds_mm=[[float(v) for v in r] for r in mesh.bounds],
    )
    return {"node": entity_id, "centroid_viewer_mm": centroid, "quality": asdict(quality)}


#: Tries before a validity-filtered draw falls back to the always-valid spread.
MAX_TRIES = 6
#: The spread at which every sample passed the validity sweep (Level 2a report, section 6).
FALLBACK_SPREAD = 0.5


def write_valid_sample_scene(
    joint: JointModel,
    out_dir: Path,
    *,
    seed: int,
    limits: Any,
    spread: float = 1.0,
    max_tries: int = MAX_TRIES,
    provenance: Mapping[str, Any] | None = None,
    ontology: AnatomyOntology | None = None,
) -> dict[str, Any]:
    """A sample at ``spread`` that passes the validity checks, or the fallback, written.

    Draws are made from the seed's own sequence (``seed * 1000 + try``), so a seed always gives
    the same thorax. A draw is kept only if it passes every check.
    - Kept samples at 1 SD are as far from their nearest training subject as a new real person
      is (8.8 mm against 9.1 mm median; report, section 6).
    - Unfiltered, one draw in five fails a check.
    - If no draw in ``max_tries`` passes, the thorax is drawn at ``FALLBACK_SPREAD``, where all
      50 swept samples passed, and the manifest says so.
    """
    from shapes.validity import face_normals, judge, measure

    ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    mean = joint.split(joint.model.mean)
    normals = {e: face_normals(mean[e], joint.faces[e]) for e in joint.entities}
    rejected: list[list[str]] = []
    for attempt in range(max_tries):
        drawn = draw_coefficients(joint.model.n_modes, seed * 1000 + attempt, spread, clip=3.0)
        failures = judge(
            measure(joint.sample_entities(drawn), joint.faces, normals), limits, ontology
        )
        if not failures:
            manifest = write_sample_scene(
                joint,
                out_dir,
                seed=seed,
                spread=spread,
                provenance=provenance,
                ontology=ontology,
                coefficients=drawn,
                sampler={
                    "kind": f"gaussian at {spread:g} SD, kept only if valid",
                    "tries": attempt + 1,
                    "rejected": rejected,
                },
            )
            # Already checked above; recording the verdict saves checking it twice.
            manifest["validity"] = {"passes": True, "failures": []}
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return manifest
        rejected.append([message for _, message in failures][:3])
    return write_sample_scene(
        joint,
        out_dir,
        seed=seed,
        spread=FALLBACK_SPREAD,
        provenance=provenance,
        ontology=ontology,
        limits=limits,
        sampler={
            "kind": f"fallback to {FALLBACK_SPREAD:g} SD after {max_tries} rejected draws at "
            f"{spread:g} SD",
            "tries": max_tries,
            "rejected": rejected,
        },
    )


def write_sample_scene(
    joint: JointModel,
    out_dir: Path,
    *,
    seed: int,
    spread: float = 1.0,
    provenance: Mapping[str, Any] | None = None,
    ontology: AnatomyOntology | None = None,
    limits: Any = None,
    coefficients: Sequence[float] | None = None,
    sampler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw, mesh, measure and write one sample; return its manifest.

    With ``limits`` (``shapes.validity.Limits``), the sample also goes through the validity
    checks, and the manifest records the verdict under ``validity``. ``coefficients`` are used
    as given instead of being drawn from ``seed``; ``sampler`` describes how they were chosen.
    """
    ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    if coefficients is None:
        coefficients = draw_coefficients(joint.model.n_modes, seed, spread)
    coefficients = [float(c) for c in coefficients]
    vertices = joint.sample_entities(coefficients)

    scene = trimesh.Scene()
    entities: dict[str, Any] = {}
    centroids: dict[str, list[float]] = {}
    for entity_id, points in vertices.items():
        mesh = model_mesh(points, joint.faces[entity_id])
        entities[entity_id] = entity_entry(entity_id, mesh)
        centroids[entity_id] = entities[entity_id]["centroid_viewer_mm"]
        scene.add_geometry(mesh, node_name=entity_id, geom_name=entity_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.glb").write_bytes(scene.export(file_type="glb"))
    renderable = [e for e in ontology.ids() if ontology.get(e).renderable]
    source = dict(provenance or {})
    manifest: dict[str, Any] = {
        "manifest_version": "0.1.0",
        "subject": f"sample-{seed}",
        "scene": "scene.glb",
        "frame": VIEWER_FRAME,
        "generated": {
            "model": "Level 2a joint statistical shape model",
            "seed": seed,
            "spread": spread,
            "coefficients": coefficients,
            "modes": joint.model.n_modes,
            "training_subjects": len(joint.model.subjects),
            **({"sampler": dict(sampler)} if sampler else {}),
        },
        "ontology": {"ontology_id": ontology.ontology_id, "version": ontology.version},
        "source": {
            "dataset": source.get("dataset", "statistical shape model"),
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
    }
    manifest["checks"] = asdict(assess(manifest))
    if limits is not None:
        # Imported here: shapes.validity draws its samples with this module.
        from shapes.validity import face_normals, judge, measure

        mean = joint.split(joint.model.mean)
        normals = {e: face_normals(mean[e], joint.faces[e]) for e in joint.entities}
        failures = judge(measure(vertices, joint.faces, normals), limits, ontology)
        manifest["validity"] = {"passes": not failures, "failures": [m for _, m in failures]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Write one sample scene."""
    parser = argparse.ArgumentParser(description="Write a generated thorax scene.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="JSON with dataset, version, license, attribution, url",
    )
    args = parser.parse_args(argv)
    provenance = (
        json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance else None
    )
    manifest = write_sample_scene(
        JointModel.load(args.model),
        args.out,
        seed=args.seed,
        spread=args.spread,
        provenance=provenance,
    )
    print(
        f"sample {args.seed}: usable={manifest['checks']['usable']} "
        f"hard={manifest['checks']['hard']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
