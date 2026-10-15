"""Level 1: the mesh pipeline, on the synthetic phantom (assets.phantom) with known shapes.

Real data waits for permission to download, and would not have known answers anyway. Here
every label is a sphere, ellipsoid or cylinder placed at a chosen RAS+ position, so volumes,
sides, crops and orientation can be checked exactly.

The layout is written in RAS+ millimetres with the heart at the origin: the subject's left is
negative x, so the left lobes sit at x = -60.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")
pytest.importorskip("fast_simplification")

from assets.labels import load_label_map  # noqa: E402
from assets.meshes import (  # noqa: E402
    RAS_TO_VIEWER,
    MeshParameters,
    build_subject,
    mask_to_mesh,
)
from assets.phantom import (  # noqa: E402
    SPACING,
    phantom_affine,
    phantom_grid,
    sphere,
    write_phantom,
)


def _grid(affine):
    return phantom_grid(affine)


def _affine(orientation: str):
    return phantom_affine(orientation)


def _sphere(c, r):
    return sphere(c, r)


def _write_subject(root: Path, orientation: str = "RAS", *, empty: tuple[str, ...] = ()) -> Path:
    return write_phantom(root, orientation, empty=empty)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[dict[str, Any], Path]:
    """One synthetic subject in RAS orientation, built."""
    root = tmp_path_factory.mktemp("subject")
    out = root / "out"
    return build_subject(_write_subject(root), out), out


class TestOneMask:
    """mask_to_mesh on its own."""

    def test_a_sphere_keeps_its_volume(self) -> None:
        """Marching cubes plus Taubin smoothing stays within 5% of the true volume."""
        affine = _affine("RAS")
        x, y, z, _ = _grid(affine)
        mesh, quality = mask_to_mesh(_sphere((0, 0, 0), 30)(x, y, z), affine)
        expected = 4.0 / 3.0 * math.pi * 30.0**3 / 1000.0
        assert quality.watertight
        assert quality.volume_ml == pytest.approx(expected, rel=0.05)
        assert quality.centroid_mm == pytest.approx([0.0, 0.0, 0.0], abs=1.0)

    def test_the_face_budget_is_respected(self) -> None:
        """Decimation brings a large surface down to the requested face count."""
        affine = _affine("RAS")
        x, y, z, _ = _grid(affine)
        mesh, quality = mask_to_mesh(
            _sphere((0, 0, 0), 40)(x, y, z), affine, MeshParameters(max_faces=800)
        )
        assert quality.faces <= 800 * 1.05

    def test_specks_are_dropped(self) -> None:
        """A component far smaller than the main one is noise, and is reported as dropped."""
        affine = _affine("RAS")
        x, y, z, _ = _grid(affine)
        mask = _sphere((0, 0, 0), 30)(x, y, z) | _sphere((70, 40, 60), 3)(x, y, z)
        _mesh, quality = mask_to_mesh(mask, affine)
        assert quality.components == 1
        assert quality.notes == ["dropped 1 small components"]

    def test_an_empty_mask_is_refused(self) -> None:
        """There is no mesh of nothing."""
        affine = _affine("RAS")
        x, _, _, _ = _grid(affine)
        with pytest.raises(ValueError):
            mask_to_mesh(np.zeros(x.shape, dtype=bool), affine)


class TestTheViewerFrame:
    """RAS+ to glTF y-up, as seen from the front."""

    def test_it_is_a_rotation_not_a_mirror(self) -> None:
        """Orthonormal with determinant +1: no shape is ever flipped."""
        assert np.allclose(RAS_TO_VIEWER @ RAS_TO_VIEWER.T, np.eye(3))
        assert np.linalg.det(RAS_TO_VIEWER) == pytest.approx(1.0)

    def test_left_is_plus_x_and_superior_is_plus_y(self, built) -> None:
        """Left lobes at +X, right lobes at -X, trachea above the heart."""
        manifest, _out = built
        c = {e: v["centroid_viewer_mm"] for e, v in manifest["entities"].items()}
        assert c["heart"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        assert c["respiratory.lung_upper_lobe_left"][0] > 40
        assert c["respiratory.lung_lower_lobe_right"][0] < -40
        assert c["respiratory.trachea"][1] > 40
        assert c["respiratory.lung_upper_lobe_left"][1] > c["respiratory.lung_lower_lobe_left"][1]

    def test_every_sided_entity_is_on_its_side(self, built) -> None:
        """The manifest's own side check passes for every left and right entity."""
        manifest, _out = built
        assert manifest["sides_correct"]
        assert all(manifest["sides_correct"].values())

    def test_scanner_orientation_does_not_matter(self, built, tmp_path) -> None:
        """An LPS-stored subject builds to the same positions as the RAS-stored one."""
        manifest, _out = built
        lps = build_subject(_write_subject(tmp_path, "LPS"), tmp_path / "out")
        for entity_id, entry in manifest["entities"].items():
            assert lps["entities"][entity_id]["centroid_viewer_mm"] == pytest.approx(
                entry["centroid_viewer_mm"], abs=0.5
            ), entity_id


class TestTheSubjectBuild:
    """build_subject end to end."""

    def test_every_mapped_entity_gets_a_mesh(self, built) -> None:
        """17 entities, none missing, all watertight."""
        manifest, _out = built
        assert set(manifest["entities"]) == set(load_label_map().entity_ids())
        assert manifest["missing"] == []
        assert all(e["quality"]["watertight"] for e in manifest["entities"].values())

    def test_shells_contain_their_parts(self, built) -> None:
        """A union is at least as large as any of its parts."""
        entities = built[0]["entities"]
        for shell, parts in (
            ("heart", ("heart.left_ventricle", "heart.right_atrium")),
            ("respiratory.right_lung", ("respiratory.lung_middle_lobe_right",)),
        ):
            for part in parts:
                assert entities[shell]["quality"]["volume_ml"] >= entities[part]["quality"][
                    "volume_ml"
                ]

    def test_the_abdominal_vessels_are_cropped_to_the_lungs(self, built) -> None:
        """The aorta and IVC do not reach below the lowest lung slice (y is superior)."""
        entities = built[0]["entities"]
        lowest_lung = min(
            entities[e]["quality"]["bounds_mm"][0][2]
            for e in ("respiratory.lung_lower_lobe_left", "respiratory.lung_lower_lobe_right")
        )
        for vessel in ("heart.aorta", "heart.inferior_vena_cava"):
            assert entities[vessel]["quality"]["bounds_mm"][0][2] >= lowest_lung - SPACING

    def test_the_glb_has_one_node_per_entity_named_by_id(self, built) -> None:
        """The viewer finds meshes by AWR entity id, so node names must be exactly those."""
        manifest, out = built
        scene = trimesh.load(out / "scene.glb", force="scene")
        assert set(scene.graph.nodes_geometry) == set(manifest["entities"])

    def test_the_manifest_carries_source_and_limits(self, built) -> None:
        """License, attribution, parameters and the symbolic entities (ADR-L1-002, -004)."""
        manifest, out = built
        on_disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(manifest))
        assert manifest["source"]["license"] == "CC BY 4.0"
        assert "TotalSegmentator" in manifest["source"]["attribution"]
        assert manifest["parameters"]["max_faces"] == MeshParameters().max_faces
        assert "heart.mitral_valve" in manifest["no_geometry"]
        assert "not validated" in manifest["medical_status"]

    def test_an_empty_label_is_reported_not_fatal(self, tmp_path) -> None:
        """A subject missing a structure still builds; the gap is listed."""
        manifest = build_subject(
            _write_subject(tmp_path, empty=("trachea",)), tmp_path / "out"
        )
        assert manifest["missing"] == ["respiratory.trachea"]
        assert "respiratory.trachea" not in manifest["entities"]

    def test_labels_on_different_grids_are_refused(self, tmp_path) -> None:
        """Mixing two scans' labels would silently misplace a structure."""
        subject = _write_subject(tmp_path)
        other = _affine("RAS").copy()
        other[0, 3] += 10.0
        x, _, _, _ = _grid(other)
        nib.save(
            nib.Nifti1Image(np.ones(x.shape, dtype=np.uint8), other),
            str(subject / "segmentations" / "trachea.nii.gz"),
        )
        with pytest.raises(ValueError, match="different grid"):
            build_subject(subject, tmp_path / "out")


class TestRealAnatomyLessons:
    """Two checks the first real subjects proved wrong, now fixed and pinned."""

    @pytest.fixture()
    def thorax(self):
        """The thorax ontology, for laterality and roles."""
        from awr.ontology import load_ontology

        return load_ontology(Path("ontology/thorax"), expected_version="0.1.0")

    def test_chambers_are_judged_against_their_partner_not_the_heart(self, thorax) -> None:
        """A real left atrium lies right of the heart's centre yet left of the right atrium."""
        from assets.meshes import side_report

        centroids = {
            "heart": [0.0, 0.0, 0.0],
            "heart.left_atrium": [-5.0, 0.0, -30.0],   # right of the heart centre ...
            "heart.right_atrium": [-35.0, 0.0, 0.0],   # ... but left of its partner
            "heart.left_ventricle": [30.0, 0.0, 10.0],
            "heart.right_ventricle": [5.0, 0.0, 30.0],  # anterior and left of the heart centre
        }
        report = side_report(centroids, thorax)
        assert report == {e: True for e in centroids if e != "heart"}

    def test_lungs_are_judged_against_the_midline(self, thorax) -> None:
        """The trachea marks the midline; a lobe on the wrong side of it is wrong."""
        from assets.meshes import side_report

        centroids = {
            "respiratory.trachea": [10.0, 80.0, 0.0],
            "respiratory.lung_upper_lobe_left": [70.0, 40.0, 0.0],
            "respiratory.lung_upper_lobe_right": [40.0, 40.0, 0.0],  # left of the midline
        }
        report = side_report(centroids, thorax)
        assert report["respiratory.lung_upper_lobe_left"] is True
        assert report["respiratory.lung_upper_lobe_right"] is False

    def test_quality_is_measured_before_decimation(self) -> None:
        """Heavy decimation must not turn a closed segmentation into a 'not watertight' one."""
        affine = _affine("RAS")
        x, y, z, _ = _grid(affine)
        _mesh, quality = mask_to_mesh(
            _sphere((0, 0, 0), 40)(x, y, z), affine, MeshParameters(max_faces=300)
        )
        assert quality.watertight
        assert quality.volume_ml == pytest.approx(4 / 3 * math.pi * 40**3 / 1000, rel=0.05)
        assert "decimated for display" in quality.notes
