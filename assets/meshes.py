"""Segmentation volumes to meshes, one per AWR entity, with a manifest that carries the source.

Per entity, in order:

1. read each label's NIfTI, reoriented to RAS+ (``nibabel.as_closest_canonical``) so every
   subject and every scanner orientation lands in one convention;
2. take the union of the entity's labels and apply its crop, if any;
3. marching cubes at 0.5 on the padded mask, vertices mapped to millimetres by the affine;
4. drop components smaller than a fraction of the largest, Taubin-smooth (which does not
   shrink the way Laplacian smoothing does), and decimate to a face budget;
5. measure: faces, watertightness, volume, components, bounds, centroid.

Every mesh is then moved into one viewer frame: origin at the heart shell's centroid, axes
turned from RAS+ (x right, y anterior, z superior; z-up) into glTF's y-up convention as seen
from the front of the subject. In that frame +X is the subject's **left**, +Y superior and
+Z anterior. The rotation is proper, so nothing is mirrored; a test pins the sides.

The meshes are reference geometry (ADR-L1-002): a view attached to entity ids, never the
source of truth. Every output names its dataset, license and parameters (ADR-L1-004).

Needs the ``assets`` extra (ADR-L1-003).

    python -m assets.meshes --subject datasets/raw/totalsegmentator_v1/s0011
        --out datasets/processed/level1/s0011
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import trimesh
from skimage import measure

from assets.labels import DEFAULT_LABEL_MAP, LabelMap, load_label_map, validate_label_map
from awr.ontology import AnatomyOntology, load_ontology

__all__ = [
    "RAS_TO_VIEWER",
    "SIDE_PAIRS",
    "boundary_faces",
    "significant_pieces",
    "MeshParameters",
    "MeshQuality",
    "build_subject",
    "mask_to_mesh",
    "read_mask",
    "side_report",
]

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

#: RAS+ millimetres to the viewer frame: x' = -x (subject's left), y' = z (superior),
#: z' = y (anterior). Determinant +1, so a rotation and not a mirror.
RAS_TO_VIEWER: FloatArray = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

VIEWER_FRAME = (
    "glTF y-up, millimetres, origin at the heart shell centroid; +X is the subject's left, "
    "+Y superior, +Z anterior (seen from the front, the subject's right is on screen left)."
)


@dataclass(frozen=True, slots=True)
class MeshParameters:
    """Everything that changes a mesh; recorded in the manifest."""

    level: float = 0.5
    min_component_fraction: float = 0.05
    smoothing_iterations: int = 10
    max_faces: int = 20_000


@dataclass(slots=True)
class MeshQuality:
    """Measured, not eyeballed."""

    faces: int
    """Faces of the display mesh, after decimation."""
    vertices: int
    watertight: bool
    """Of the full-resolution surface, before decimation."""
    components: int
    volume_ml: float | None
    centroid_mm: list[float]
    bounds_mm: list[list[float]]
    notes: list[str] = field(default_factory=list)
    touches_boundary: list[str] = field(default_factory=list)
    """Faces of the scanned volume the mask reaches (RAS+ names). A structure that touches the
    superior or inferior face was probably cut off by the scan's field of view."""
    display_watertight: bool = True
    """Whether the decimated display mesh is still closed; informational only."""
    voxel_components: int = 0
    """Connected pieces of the mask (6-connectivity) at least ``min_component_fraction`` of the
    largest. Unlike ``components``, which counts surfaces, a hollow structure such as the
    myocardium is one piece here."""


_FACES = (("right", 0, -1), ("left", 0, 0), ("anterior", 1, -1), ("posterior", 1, 0),
          ("superior", 2, -1), ("inferior", 2, 0))


def significant_pieces(mask: BoolArray, fraction: float) -> int:
    """Connected voxel pieces at least ``fraction`` of the largest one."""
    labelled = measure.label(mask, connectivity=1)
    sizes = np.bincount(labelled.ravel())[1:]
    if sizes.size == 0:
        return 0
    return int((sizes >= fraction * sizes.max()).sum())


def boundary_faces(mask: BoolArray) -> list[str]:
    """Which faces of the (RAS+, canonical) volume a mask touches.

    In a canonical RAS+ grid, voxel index grows towards right, anterior and superior, so the
    last slice along an axis is that direction's face and the first is the opposite one.
    """
    touched = []
    for name, axis, index in _FACES:
        if np.take(mask, index, axis=axis).any():
            touched.append(name)
    return touched


def read_mask(path: Path) -> tuple[BoolArray, FloatArray]:
    """A label volume as a boolean RAS+ mask and its voxel-to-millimetre affine."""
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj)
    return data > 0.5, np.asarray(image.affine, dtype=np.float64)


def _crop_to_extent(mask: BoolArray, reference: BoolArray) -> BoolArray:
    """Zero every axial slice outside the reference's superior-inferior extent (axis 2)."""
    slices = np.flatnonzero(reference.any(axis=(0, 1)))
    if slices.size == 0:
        return np.zeros_like(mask)
    cropped = mask.copy()
    cropped[:, :, : slices[0]] = False
    cropped[:, :, slices[-1] + 1 :] = False
    return cropped


def mask_to_mesh(
    mask: BoolArray, affine: FloatArray, parameters: MeshParameters | None = None
) -> tuple[trimesh.Trimesh, MeshQuality]:
    """One mask to one smoothed, decimated, measured mesh in RAS+ millimetres."""
    parameters = parameters or MeshParameters()
    if not mask.any():
        raise ValueError("empty mask")
    padded = np.pad(mask, 1).astype(np.float32)
    vertices, faces, _normals, _values = measure.marching_cubes(padded, level=parameters.level)
    world = nib.affines.apply_affine(affine, vertices - 1.0)
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=True)

    notes: list[str] = []
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        largest = max(len(p.faces) for p in parts)
        kept = [p for p in parts if len(p.faces) >= parameters.min_component_fraction * largest]
        if len(kept) < len(parts):
            notes.append(f"dropped {len(parts) - len(kept)} small components")
        mesh = trimesh.util.concatenate(kept)

    trimesh.smoothing.filter_taubin(mesh, iterations=parameters.smoothing_iterations)
    trimesh.repair.fix_normals(mesh)
    # Measured on the full-resolution surface. Decimation is for display and can open small
    # holes (real lung lobes do: watertight before, not after), which says nothing about the
    # segmentation. The display mesh's own watertightness is recorded separately.
    watertight = bool(mesh.is_watertight)
    components = len(mesh.split(only_watertight=False))
    volume_ml = float(abs(mesh.volume)) / 1000.0 if watertight else None
    centroid = [float(v) for v in mesh.centroid]
    bounds = [[float(v) for v in row] for row in mesh.bounds]
    if len(mesh.faces) > parameters.max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=parameters.max_faces)
        trimesh.repair.fix_normals(mesh)
        notes.append("decimated for display")

    return mesh, MeshQuality(
        faces=len(mesh.faces),
        vertices=len(mesh.vertices),
        watertight=watertight,
        components=components,
        volume_ml=volume_ml,
        centroid_mm=centroid,
        bounds_mm=bounds,
        notes=notes,
        display_watertight=bool(mesh.is_watertight),
    )


def _label_path(subject_dir: Path, label: str) -> Path:
    """TotalSegmentator v1 layout: ``<subject>/segmentations/<label>.nii.gz``."""
    return subject_dir / "segmentations" / f"{label}.nii.gz"


#: Pairs whose names mean "left heart" and "right heart": the circulations, not positions in
#: the body. The heart is rotated -- the right ventricle is anterior and reaches left of the
#: midline, the left atrium is posterior and near it -- so each chamber is judged against its
#: partner, never against the heart's centre or the midline.
SIDE_PAIRS = (
    ("heart.left_ventricle", "heart.right_ventricle"),
    ("heart.left_atrium", "heart.right_atrium"),
)
_MIDLINE_SIDED_ROLES = frozenset({"lung", "lung_lobe"})


def side_report(
    centroids: Mapping[str, Sequence[float]], ontology: AnatomyOntology
) -> dict[str, bool]:
    """Whether sided structures are on their sides, in the viewer frame (+X subject's left).

    * Lungs and lobes are judged against the body midline: the trachea's x if present, else
      the midpoint of the two lungs.
    * Paired heart chambers are judged against each other: the left one must lie further left.
    * Other sided entities (the IVC, for instance) carry laterality that is not a reliable
      position test and are not judged.
    """
    report: dict[str, bool] = {}
    if "respiratory.trachea" in centroids:
        midline: float | None = float(centroids["respiratory.trachea"][0])
    elif "respiratory.left_lung" in centroids and "respiratory.right_lung" in centroids:
        midline = (float(centroids["respiratory.left_lung"][0])
                   + float(centroids["respiratory.right_lung"][0])) / 2.0
    else:
        midline = None
    if midline is not None:
        for entity_id, centroid in centroids.items():
            entity = ontology.get(entity_id)
            if entity.semantic_role not in _MIDLINE_SIDED_ROLES:
                continue
            side = str(entity.laterality)
            if side == "left":
                report[entity_id] = float(centroid[0]) > midline
            elif side == "right":
                report[entity_id] = float(centroid[0]) < midline
    for left, right in SIDE_PAIRS:
        if left in centroids and right in centroids:
            ok = float(centroids[left][0]) > float(centroids[right][0])
            report[left] = ok
            report[right] = ok
    return report


def build_subject(
    subject_dir: Path,
    out_dir: Path,
    *,
    label_map: LabelMap | None = None,
    ontology: AnatomyOntology | None = None,
    parameters: MeshParameters | None = None,
    label_path: Callable[[Path, str], Path] = _label_path,
) -> dict[str, Any]:
    """Build every mapped entity's mesh for one subject; write ``scene.glb`` and a manifest.

    Returns the manifest. Entities whose labels are empty for this subject are listed under
    ``missing`` rather than failing the subject; the quality report decides whether the
    subject is usable.
    """
    parameters = parameters or MeshParameters()
    label_map = label_map or load_label_map(DEFAULT_LABEL_MAP)
    ontology = ontology or load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    validate_label_map(label_map, ontology)

    masks: dict[str, BoolArray] = {}
    affine: FloatArray | None = None
    for label in label_map.labels_needed():
        mask, this_affine = read_mask(label_path(subject_dir, label))
        if affine is None:
            affine = this_affine
        elif mask.shape != next(iter(masks.values())).shape or not np.allclose(
            this_affine, affine, atol=1e-4
        ):
            raise ValueError(f"{label} is on a different grid from the other labels")
        masks[label] = mask
    assert affine is not None

    meshes: dict[str, trimesh.Trimesh] = {}
    qualities: dict[str, MeshQuality] = {}
    missing: list[str] = []
    for entry in label_map.entities:
        mask = np.logical_or.reduce([masks[label] for label in entry.labels])
        if entry.crop is not None:
            references = label_map.crops[entry.crop].reference_labels
            mask = _crop_to_extent(mask, np.logical_or.reduce([masks[r] for r in references]))
        if not mask.any():
            missing.append(entry.entity_id)
            continue
        meshes[entry.entity_id], qualities[entry.entity_id] = mask_to_mesh(
            mask, affine, parameters
        )
        qualities[entry.entity_id].touches_boundary = boundary_faces(mask)
        qualities[entry.entity_id].voxel_components = significant_pieces(
            mask, parameters.min_component_fraction
        )

    if "heart" not in meshes:
        raise ValueError("no heart shell: the viewer frame is centred on it")
    origin = np.asarray(meshes["heart"].centroid, dtype=np.float64)
    scene = trimesh.Scene()
    centroids: dict[str, list[float]] = {}
    for entity_id, mesh in meshes.items():
        moved = mesh.copy()
        moved.vertices = (np.asarray(moved.vertices) - origin) @ RAS_TO_VIEWER.T
        trimesh.repair.fix_normals(moved)
        centroids[entity_id] = [float(v) for v in moved.centroid]
        scene.add_geometry(moved, node_name=entity_id, geom_name=entity_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "scene.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    renderable = [e for e in ontology.ids() if ontology.get(e).renderable]
    source = label_map.source
    manifest: dict[str, Any] = {
        "manifest_version": "0.1.0",
        "subject": subject_dir.name,
        "scene": glb_path.name,
        "frame": VIEWER_FRAME,
        "origin_ras_mm": [float(v) for v in origin],
        "ontology": {"ontology_id": ontology.ontology_id, "version": ontology.version},
        "label_map": {"id": label_map.label_map_id, "version": label_map.version},
        "source": {
            "dataset": source.dataset,
            "version": source.version,
            "doi": source.doi,
            "url": source.url,
            "license": source.license,
            "attribution": source.attribution,
            "derived": "marching cubes, component filtering, Taubin smoothing, quadric decimation",
        },
        "parameters": asdict(parameters),
        "entities": {
            entity_id: {
                "node": entity_id,
                "labels": list(label_map.for_entity(entity_id).labels),
                "crop": label_map.for_entity(entity_id).crop,
                "centroid_viewer_mm": centroids[entity_id],
                "quality": asdict(qualities[entity_id]),
            }
            for entity_id in meshes
        },
        "missing": missing,
        "no_geometry": [e for e in renderable if e not in meshes and e not in missing],
        "sides_correct": side_report(centroids, ontology),
        "medical_status": "research CT, not validated; not for clinical use",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Build one subject's scene."""
    parser = argparse.ArgumentParser(description="Level 1: segmentations to an entity scene.")
    parser.add_argument("--subject", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=MeshParameters().max_faces)
    args = parser.parse_args(argv)
    manifest = build_subject(
        args.subject, args.out, parameters=MeshParameters(max_faces=args.max_faces)
    )
    wrong = [e for e, ok in manifest["sides_correct"].items() if not ok]
    print(f"{len(manifest['entities'])} meshes, missing {manifest['missing']}, wrong side {wrong}")
    return 1 if wrong else 0


if __name__ == "__main__":
    raise SystemExit(main())
