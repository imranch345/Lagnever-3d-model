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

__all__ = ["draw_coefficients", "entity_entry", "model_mesh", "write_sample_scene"]

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


def write_sample_scene(
    joint: JointModel,
    out_dir: Path,
    *,
    seed: int,
    spread: float = 1.0,
    provenance: Mapping[str, Any] | None = None,
    ontology: AnatomyOntology | None = None,
) -> dict[str, Any]:
    """Draw, mesh, measure and write one sample; return its manifest."""
    ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    coefficients = draw_coefficients(joint.model.n_modes, seed, spread)
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
