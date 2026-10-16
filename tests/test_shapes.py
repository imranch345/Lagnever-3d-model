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

    def test_a_template_keeps_only_its_largest_piece(self) -> None:
        """A two-face sliver beside the surface is dropped, so samples can be one closed piece."""
        from shapes.register import make_template, topology

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        sliver = trimesh.Trimesh(
            [[40.0, 0, 0], [41.0, 0, 0], [40.0, 1.0, 0]], [[0, 1, 2], [0, 2, 1]], process=False
        )
        template = make_template(trimesh.util.concatenate([sphere, sliver]), 400)
        assert len(template.split(only_watertight=False)) == 1
        assert topology(template) == {"faces": 400, "watertight": True, "euler": 2}

    def test_placement_reaches_a_target_beyond_the_threshold(self) -> None:
        """A 20 mm sphere onto a 60 mm-tall ellipsoid: 40 mm out of reach without placement."""
        from shapes.register import fit_error, make_template, placement, register

        template = make_template(trimesh.creation.icosphere(subdivisions=3, radius=20.0), 400)
        target = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        target.apply_scale([22.0, 60.0, 20.0])
        target.apply_translation([5.0, 10.0, 0.0])
        placed = template.copy()
        placed.apply_transform(placement(template, target))
        assert np.allclose(placed.extents, target.extents, rtol=0.05)
        unplaced = fit_error(register(template, target, place=False), template.faces, target)
        fitted = fit_error(register(template, target), template.faces, target)
        assert unplaced["mean_from_target"] > 3.0
        assert fitted["mean_from_target"] < 1.0


def test_a_failing_subject_is_excluded_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subject whose alignment or registration raises is recorded, and the build goes on."""
    import shapes.build as build

    # Sphere scenes. s2's heart (31 mm) makes alignment fail; s3's ventricle (13 mm) makes
    # registration fail. Both are injected failures around the real functions.
    scenes = tmp_path / "scenes"
    for index in range(5):
        scene = trimesh.Scene()
        heart = trimesh.creation.icosphere(subdivisions=2, radius=31.0 if index == 2 else 30.0)
        ventricle = trimesh.creation.icosphere(subdivisions=2, radius=10.0 + index)
        scene.add_geometry(heart, node_name="heart", geom_name="heart")
        scene.add_geometry(
            ventricle, node_name="heart.left_ventricle", geom_name="heart.left_ventricle"
        )
        (scenes / f"s{index}").mkdir(parents=True)
        (scenes / f"s{index}" / "scene.glb").write_bytes(scene.export(file_type="glb"))
    (scenes / "s0" / "manifest.json").write_text(json.dumps({"source": {"dataset": "spheres"}}))
    subjects = [{"subject": f"s{i}", "usable": True} for i in range(5)]
    (scenes / "quality_report.json").write_text(json.dumps({"chosen": "s0", "subjects": subjects}))

    def radius(mesh: trimesh.Trimesh) -> float:
        return float(np.linalg.norm(mesh.vertices - mesh.vertices.mean(axis=0), axis=1).max())

    real_rigid, real_register = build.rigid_to, build.register

    def rigid(reference: trimesh.Trimesh, moving: trimesh.Trimesh, **kw: int) -> tuple:
        if abs(radius(moving) - 31.0) < 0.5:
            raise RuntimeError("alignment diverged")
        return real_rigid(reference, moving, **kw)

    def register(template: trimesh.Trimesh, target: trimesh.Trimesh, **kw: object) -> np.ndarray:
        if abs(radius(target) - 13.0) < 0.5:
            raise ValueError("degenerate target")
        return real_register(template, target, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(build, "rigid_to", rigid)
    monkeypatch.setattr(build, "register", register)
    out = tmp_path / "out"
    built = build.build_models(
        scenes, out, entities=["heart.left_ventricle"], template_faces=200, holdout_fraction=0.2
    )
    entry = built["entities"]["heart.left_ventricle"]
    assert entry["excluded"] == {
        "s2": "failed: RuntimeError: alignment diverged",
        "s3": "failed: ValueError: degenerate target",
    }
    assert entry["registered"] == 3
    cache = out / "registrations"
    assert sorted(p.name.split("__")[0] for p in cache.iterdir()) == ["s0", "s1", "s4"]
    stored = np.load(cache / "s1__heart.left_ventricle.npz")
    assert "rigid_cost" in json.loads(str(stored["error"]))


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

    def test_a_sample_can_carry_its_validity_verdict(self, tmp_path: Path) -> None:
        """With limits, the manifest says whether the sample passes the validity checks."""
        from shapes.generate import write_sample_scene
        from shapes.joint import fit_joint
        from shapes.validity import calibrate, face_normals, measure

        shapes, faces, subjects = _sphere_population()
        joint = fit_joint(shapes, faces, subjects)
        mean = joint.split(joint.model.mean)
        normals = {e: face_normals(mean[e], joint.faces[e]) for e in joint.entities}
        limits = calibrate(
            [measure({e: shapes[e][s] for e in shapes}, joint.faces, normals) for s in subjects]
        )
        near = write_sample_scene(joint, tmp_path / "a", seed=3, spread=0.0, limits=limits)
        assert near["validity"] == {"passes": True, "failures": []}
        far = write_sample_scene(joint, tmp_path / "b", seed=3, spread=3.0, limits=limits)
        assert far["validity"]["passes"] is False and far["validity"]["failures"]
        assert "validity" not in write_sample_scene(joint, tmp_path / "c", seed=3)

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


class TestValidity:
    """Checks that can fail: folding, overlap and volume, calibrated on training subjects."""

    def _setup(self) -> tuple:
        from shapes.joint import fit_joint
        from shapes.validity import calibrate, face_normals, measure

        shapes, faces, subjects = _sphere_population()
        joint = fit_joint(shapes, faces, subjects)
        mean = joint.split(joint.model.mean)
        normals = {e: face_normals(mean[e], joint.faces[e]) for e in joint.entities}
        train = [measure({e: shapes[e][s] for e in shapes}, joint.faces, normals) for s in subjects]
        return joint, mean, normals, calibrate(train)

    def test_the_mean_passes_and_each_defect_is_caught(self) -> None:
        """A dent folded inward, a neighbour run into, a shrunken lung: each fails its check."""
        from awr.ontology import load_ontology
        from shapes.validity import judge, measure

        ontology = load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
        joint, mean, normals, limits = self._setup()

        def kinds(shapes: dict) -> set[str]:
            return {k for k, _ in judge(measure(shapes, joint.faces, normals), limits, ontology)}

        assert kinds(mean) == set()
        lv = mean["heart.left_ventricle"]
        centre = lv.mean(axis=0)
        # A cap pushed through a plane into the inside: a local fold, as a sample can make.
        cut = centre[0] + 0.4 * np.ptp(lv[:, 0]) / 2
        dented = lv.copy()
        dented[lv[:, 0] > cut, 0] = 2 * cut - lv[lv[:, 0] > cut, 0]
        assert "folding" in kinds({**mean, "heart.left_ventricle": dented})
        rv = mean["heart.right_ventricle"]
        inside = centre + 0.5 * (rv - rv.mean(axis=0))
        assert "overlap" in kinds({**mean, "heart.right_ventricle": inside})
        lung = mean["respiratory.left_lung"]
        shrunk = lung.mean(axis=0) + 0.6 * (lung - lung.mean(axis=0))
        assert kinds({**mean, "respiratory.left_lung": shrunk}) == {"volume"}

    def test_the_sweep_reports_rates_and_a_control(self) -> None:
        """Wider spreads fail no less often; held-out real shapes are the control."""
        from shapes.validity import KINDS, sweep

        shapes, faces, subjects = _sphere_population(10)
        from shapes.joint import fit_joint

        joint = fit_joint(shapes, faces, subjects[:8])
        train = {s: {e: shapes[e][s] for e in shapes} for s in subjects[:8]}
        held = {s: {e: shapes[e][s] for e in shapes} for s in subjects[8:]}
        result = sweep(joint, train, held, spreads=(0.5, 3.0), samples=12)
        assert result["control"]["subjects"] == 2
        narrow, wide = result["spreads"]
        assert set(narrow["failing"]) == set(KINDS)
        assert wide["pass_rate"] <= narrow["pass_rate"]
        assert wide["failing"]["volume"] > 0
        assert narrow["failing"]["overlap"] == 0 and narrow["failing"]["folding"] == 0
        assert result["usable_spread"] in (None, 0.5, 3.0)
        assert result["method"]["calibrated_on"] == 8


class TestResize:
    """One entity resized within its model; nothing else moves; implausible results refused."""

    def _resizable(self, tmp_path: Path) -> tuple:
        from shapes.generate import write_sample_scene
        from shapes.joint import fit_joint
        from shapes.resize import Resizer
        from shapes.validity import sweep

        shapes, faces, subjects = _sphere_population(10)
        joint = fit_joint(shapes, faces, subjects)
        models = tmp_path / "level2a"
        joint.save(models / "models" / "joint.npz")
        for e in shapes:
            stacked = np.stack([shapes[e][s] for s in subjects])
            fit(e, stacked, faces[e], tuple(subjects)).save(models / "models" / f"{e}.npz")
        train = {s: {e: shapes[e][s] for e in shapes} for s in subjects}
        validity = sweep(joint, train, {}, spreads=(1.0,), samples=2)
        (models / "validity.json").write_text(json.dumps(validity))
        scene = tmp_path / "mean"
        manifest = write_sample_scene(joint, scene, seed=0, spread=0.0)
        return Resizer(models), scene, manifest

    def test_one_entity_grows_and_nothing_else_moves(self, tmp_path: Path) -> None:
        """+5% is reached; other meshes are identical; a second resize builds on the first."""
        from shapes.register import load_scene

        resizer, scene, manifest = self._resizable(tmp_path)
        lv = "heart.left_ventricle"
        outcome = resizer.resize(scene, manifest, lv, 1.05, tmp_path / "r1")
        assert outcome.ok, outcome.message
        assert abs(outcome.achieved - 1.05) < 0.01
        assert outcome.from_plain_scaling_mm < 0.5
        before, after = load_scene(scene), load_scene(tmp_path / "r1")
        for entity in before:
            same = np.allclose(before[entity].vertices, after[entity].vertices)
            assert same == (entity != lv), entity
        assert outcome.manifest["resized"][0]["entity"] == lv
        assert outcome.manifest["checks"]["usable"], outcome.manifest["checks"]
        again = resizer.resize(tmp_path / "r1", outcome.manifest, lv, 0.95, tmp_path / "r2")
        assert again.ok, again.message
        grown = resizer.base_shapes(tmp_path / "r1", outcome.manifest)[lv]
        assert np.allclose(grown, after[lv].vertices)
        assert len(again.manifest["resized"]) == 2

    def test_beyond_the_training_range_is_refused(self, tmp_path: Path) -> None:
        """+30% from the mean exceeds every training volume: refused, nothing written."""
        resizer, scene, manifest = self._resizable(tmp_path)
        outcome = resizer.resize(scene, manifest, "heart.left_ventricle", 1.30, tmp_path / "r")
        assert not outcome.ok and outcome.manifest is None
        assert any("volume" in f for f in outcome.failures)
        assert not (tmp_path / "r").exists()

    def test_running_into_a_neighbour_is_refused(self, tmp_path: Path) -> None:
        """Checked on correspondence: an RV already touching the LV blocks the LV's growth."""
        resizer, scene, manifest = self._resizable(tmp_path)
        shapes = resizer.base_shapes(scene, manifest)
        lv, rv = shapes["heart.left_ventricle"], shapes["heart.right_ventricle"]
        direction = rv.mean(axis=0) - lv.mean(axis=0)
        direction /= np.linalg.norm(direction)
        # 21 mm apart: radii of 12 overlap by 3 mm already; growth only deepens it.
        shapes["heart.right_ventricle"] = rv - rv.mean(axis=0) + lv.mean(axis=0) + 21 * direction
        np.savez_compressed(
            scene / "correspondence.npz", **{e.replace(".", "__"): v for e, v in shapes.items()}
        )
        outcome = resizer.resize(scene, manifest, "heart.left_ventricle", 1.05, tmp_path / "r")
        assert not outcome.ok
        assert any("inside" in f for f in outcome.failures), outcome.failures
        shrink = resizer.resize(scene, manifest, "heart.left_ventricle", 0.95, tmp_path / "s")
        assert shrink.ok, "an overlap the subject already had may not block a change that eases it"

    def test_a_refusal_offers_the_largest_change_that_passes(self, tmp_path: Path) -> None:
        """+30% fails on volume; the offered percentage then succeeds, and one more does not."""
        resizer, scene, manifest = self._resizable(tmp_path)
        lv = "heart.left_ventricle"
        refused = resizer.resize(scene, manifest, lv, 1.30, tmp_path / "a")
        assert not refused.ok and refused.feasible_percent is not None
        best = refused.feasible_percent
        assert 0 < best < 30 and f"+{best}%" in refused.message
        assert resizer.resize(scene, manifest, lv, 1 + best / 100, tmp_path / "b").ok
        assert not resizer.resize(scene, manifest, lv, 1 + (best + 1) / 100, tmp_path / "c").ok

    def test_what_cannot_be_resized_says_why(self, tmp_path: Path) -> None:
        """No model for the entity, or no correspondence for the scene: refused with a reason."""
        resizer, scene, manifest = self._resizable(tmp_path)
        no_model = resizer.resize(scene, manifest, "respiratory.trachea", 1.1, tmp_path / "a")
        assert not no_model.ok and "no shape model" in no_model.message
        too_far = resizer.resize(scene, manifest, "heart.left_ventricle", 1.5, tmp_path / "b")
        assert not too_far.ok and "at most 30%" in too_far.message
        real = {k: v for k, v in manifest.items() if k != "generated"} | {"subject": "s9999"}
        unregistered = resizer.resize(
            tmp_path / "nowhere", real, "heart.left_ventricle", 1.1, tmp_path / "c"
        )
        assert not unregistered.ok and "Generate a thorax first" in unregistered.message
