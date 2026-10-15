"""Level 2a: shape models, alignment, correspondence, and the build on a known population.

The model maths is checked on synthetic shapes with known modes. The pipeline is checked on a
phantom population whose only variation is the left ventricle's radius, so the first mode has
to be that variation and the model has to beat the mean-shape baseline on held-out subjects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from shapes.ssm import ShapeModel, fit

RNG = np.random.default_rng(0)


def _population(n: int = 12, vertices: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shapes = mean + a * mode1 + b * mode2, with mode1 four times the spread of mode2."""
    mean = RNG.standard_normal((vertices, 3))
    basis, _ = np.linalg.qr(RNG.standard_normal((vertices * 3, 2)))
    a = RNG.standard_normal(n) * 4.0
    b = RNG.standard_normal(n) * 1.0
    shapes = mean + (np.outer(a, basis[:, 0]) + np.outer(b, basis[:, 1])).reshape(n, vertices, 3)
    return shapes, basis, np.zeros((1, 3), dtype=np.int64)


class TestTheShapeModel:
    """PCA over corresponded shapes."""

    def test_it_recovers_the_population_modes(self) -> None:
        """Two true modes, so two kept, spanning the same space, the larger one first."""
        shapes, basis, faces = _population()
        model = fit("e", shapes, faces, tuple(f"s{i}" for i in range(len(shapes))), variance=0.999)
        assert model.n_modes == 2
        flat = model.modes.reshape(2, -1)
        assert np.allclose(flat @ flat.T, np.eye(2), atol=1e-9)
        overlap = np.abs(flat @ basis)
        assert overlap[0].max() > 0.9
        assert model.explained[0] > model.explained[1]

    def test_sample_and_project_are_inverse(self) -> None:
        """Weights in standard deviations survive a round trip; zero weights give the mean."""
        shapes, _, faces = _population()
        model = fit("e", shapes, faces, tuple(str(i) for i in range(len(shapes))), variance=0.999)
        assert np.allclose(model.sample([0, 0]), model.mean)
        b = np.array([1.5, -0.5])
        assert np.allclose(model.project(model.sample(b)), b, atol=1e-9)

    def test_a_training_shape_reconstructs_exactly_with_all_modes(self) -> None:
        """No information is lost inside the span of the data."""
        shapes, _, faces = _population()
        model = fit("e", shapes, faces, tuple(str(i) for i in range(len(shapes))), variance=1.0)
        assert np.allclose(model.reconstruct(shapes[3]), shapes[3], atol=1e-8)

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Arrays and metadata round-trip."""
        shapes, _, faces = _population()
        model = fit("heart.left_ventricle", shapes, faces, tuple(str(i) for i in range(12)))
        model.save(tmp_path / "m.npz")
        again = ShapeModel.load(tmp_path / "m.npz")
        assert again.entity_id == "heart.left_ventricle" and again.subjects == model.subjects
        assert np.allclose(again.modes, model.modes) and np.allclose(again.stdevs, model.stdevs)

    @pytest.mark.parametrize(
        ("shapes", "subjects"),
        [
            (np.zeros((1, 4, 3)), ("a",)),
            (np.zeros((3, 4, 2)), ("a", "b", "c")),
            (np.zeros((3, 4, 3)), ("a", "b")),
        ],
        ids=["one-shape", "not-3d", "subject-count"],
    )
    def test_bad_input_is_refused(self, shapes: np.ndarray, subjects: tuple[str, ...]) -> None:
        """No model from one shape, flat points, or unlabelled shapes."""
        with pytest.raises(ValueError):
            fit("e", shapes, np.zeros((1, 3), dtype=np.int64), subjects)


trimesh = pytest.importorskip("trimesh")
pytest.importorskip("rtree")
pytest.importorskip("skimage")


class TestRegistration:
    """Rigid alignment and template correspondence on simple shapes."""

    def test_rigid_alignment_undoes_a_rotation(self) -> None:
        """An ellipsoid rotated by 10 degrees is brought back to within half a millimetre."""
        from shapes.register import rigid_to

        reference = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        reference.apply_scale([40.0, 25.0, 15.0])
        moved = reference.copy()
        moved.apply_transform(trimesh.transformations.rotation_matrix(math.radians(10), [0, 0, 1]))
        matrix, _ = rigid_to(reference, moved)
        moved.apply_transform(matrix)
        assert np.abs(moved.vertices - reference.vertices).max() < 0.5

    def test_a_template_fits_a_larger_shape(self) -> None:
        """A sphere template deformed onto a bigger ellipsoid fits within a millimetre."""
        from shapes.register import fit_error, make_template, register

        template = make_template(trimesh.creation.icosphere(subdivisions=3, radius=20.0), 400)
        target = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        target.apply_scale([26.0, 22.0, 20.0])
        vertices = register(template, target)
        error = fit_error(vertices, template.faces, target)
        assert len(vertices) == len(template.vertices)
        assert error["mean_from_target"] < 1.0 and error["mean_to_target"] < 1.0


@pytest.mark.slow
def test_the_build_finds_the_population_variation(tmp_path: Path) -> None:
    """Six phantoms differing only in left ventricle radius: one dominant mode, beats the mean."""
    pytest.importorskip("fast_simplification")
    from assets.meshes import build_subject
    from assets.phantom import LAYOUT, sphere, write_phantom
    from shapes.build import build_models

    scenes = tmp_path / "scenes"
    for index, radius in enumerate((10.0, 11.0, 12.0, 13.0, 14.0, 15.0)):
        layout = dict(LAYOUT)
        centre = (-15, 0, -10)
        layout["heart_ventricle_left"] = sphere(centre, radius)
        layout["heart_myocardium"] = lambda x, y, z, r=radius, c=centre: (
            sphere(c, r + 4)(x, y, z) & ~sphere(c, r)(x, y, z)
        )
        name = f"p{index}"
        build_subject(write_phantom(tmp_path / "raw", layout=layout, name=name), scenes / name)
    report = {
        "chosen": "p2",
        "subjects": [{"subject": f"p{i}", "usable": True} for i in range(6)],
    }
    (scenes / "quality_report.json").write_text(json.dumps(report))

    built = build_models(
        scenes,
        tmp_path / "out",
        entities=["heart.left_ventricle"],
        template_faces=400,
        holdout_fraction=0.34,
        max_fit_mm=2.0,
    )
    entry = built["entities"]["heart.left_ventricle"]
    assert entry["excluded"] == {}
    assert entry["explained"][0] > 0.8
    assert entry["generality_mm"] < entry["mean_baseline_mm"]
    assert (tmp_path / "out" / "models" / "heart.left_ventricle.npz").is_file()

    again = build_models(
        scenes,
        tmp_path / "out",
        entities=["heart.left_ventricle"],
        template_faces=400,
        holdout_fraction=0.34,
        max_fit_mm=2.0,
    )
    assert again["entities"] == built["entities"], "cached registrations must give the same models"


def _sphere_population(n: int = 8) -> tuple[dict, dict, list[str]]:
    """Four entities as icospheres at anatomical sides (viewer frame: +X is the left)."""
    rng = np.random.default_rng(1)
    centres = {
        "heart.left_ventricle": (15.0, -5.0, 0.0),
        "heart.right_ventricle": (-10.0, -5.0, 10.0),
        "respiratory.left_lung": (60.0, 10.0, 0.0),
        "respiratory.right_lung": (-60.0, 10.0, 0.0),
    }
    radii = {
        "heart.left_ventricle": 12.0,
        "heart.right_ventricle": 12.0,
        "respiratory.left_lung": 35.0,
        "respiratory.right_lung": 38.0,
    }
    base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    subjects = [f"s{i}" for i in range(n)]
    shapes: dict[str, dict[str, np.ndarray]] = {e: {} for e in centres}
    for s in subjects:
        size = 1.0 + 0.1 * rng.standard_normal()
        for e, c in centres.items():
            shapes[e][s] = base.vertices * radii[e] * size + np.array(c)
    faces = {e: base.faces for e in centres}
    return shapes, faces, subjects


class TestJointModelAndSamples:
    """One model over several entities, and generated scenes from it."""

    def test_the_joint_model_keeps_entities_apart(self, tmp_path: Path) -> None:
        """Segments split back exactly; incomplete subjects are left out; save/load works."""
        from shapes.joint import JointModel, fit_joint

        shapes, faces, subjects = _sphere_population()
        del shapes["heart.right_ventricle"]["s7"]
        joint = fit_joint(shapes, faces, subjects)
        assert "s7" not in joint.model.subjects and len(joint.model.subjects) == 7
        mean = joint.sample_entities([])
        assert set(mean) == set(shapes)
        assert all(len(v) == len(faces[e]) * 0 + len(shapes[e]["s0"]) for e, v in mean.items())
        joint.save(tmp_path / "joint.npz")
        again = JointModel.load(tmp_path / "joint.npz")
        assert again.segments == joint.segments
        assert np.array_equal(again.faces["heart.left_ventricle"], faces["heart.left_ventricle"])

    def test_a_sample_scene_is_honest_and_checked(self, tmp_path: Path) -> None:
        """Same seed, same sample; the manifest says it is generated and passes the checks."""
        from shapes.generate import write_sample_scene
        from shapes.joint import fit_joint

        shapes, faces, subjects = _sphere_population()
        joint = fit_joint(shapes, faces, subjects)
        first = write_sample_scene(
            joint, tmp_path / "a", seed=3, provenance={"license": "CC BY 4.0"}
        )
        second = write_sample_scene(joint, tmp_path / "b", seed=3)
        assert first["generated"]["coefficients"] == second["generated"]["coefficients"]
        assert first["generated"]["seed"] == 3 and "not a measurement" in first["medical_status"]
        assert first["source"]["license"] == "CC BY 4.0"
        assert first["checks"]["usable"], first["checks"]["hard"]
        assert all(first["sides_correct"].values())
        loaded = trimesh.load(tmp_path / "a" / "scene.glb", force="scene")
        assert set(loaded.graph.nodes_geometry) == set(shapes)

    def test_the_viewer_swaps_geometry_and_keeps_scene_state(self, tmp_path: Path) -> None:
        """A sample changes the meshes served, not what the engine says is hidden."""
        from shapes.generate import write_sample_scene
        from shapes.joint import fit_joint
        from viewer.server import ViewerApp

        shapes, faces, subjects = _sphere_population()
        joint = fit_joint(shapes, faces, subjects)
        real = tmp_path / "real"
        write_sample_scene(joint, real, seed=0)
        app = ViewerApp(
            real,
            sampler=lambda seed, out: write_sample_scene(joint, out, seed=seed),
            samples_dir=tmp_path / "samples",
        )
        app.command("hide the trachea")
        reply = app.sample(5)
        assert reply["ok"] and app.scene_dir == tmp_path / "samples" / "seed-5"
        assert reply["state"]["generated"]["seed"] == 5
        trachea = next(e for e in reply["state"]["entities"] if e["id"] == "respiratory.trachea")
        assert trachea["visible"] is False
        assert app.original()["ok"] and app.scene_dir == real

    def test_without_a_model_sampling_says_how_to_enable_it(self, tmp_path: Path) -> None:
        """No sampler: a clear answer, nothing changes."""
        from shapes.generate import write_sample_scene
        from shapes.joint import fit_joint
        from viewer.server import ViewerApp

        shapes, faces, subjects = _sphere_population()
        write_sample_scene(fit_joint(shapes, faces, subjects), tmp_path / "real", seed=0)
        app = ViewerApp(tmp_path / "real")
        reply = app.sample()
        assert not reply["ok"] and "--model" in reply["message"]
