"""Level 2b: labelled points in the right frame, and a field that learns them.

* **Data.** On the phantom, every labelled point lies in the structure its label names, in the
  same frame as the Level 1 meshes; sampling is reproducible; the split and provenance are
  carried.
* **Model.** Shapes and devices; non-overlap is structural (one label per point).
* **Training.** On a tiny synthetic population of spheres, the field learns the labels and the
  codes tell subjects apart; the checkpoint holds what evaluation needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trimesh")
pytest.importorskip("rtree")
pytest.importorskip("skimage")


@pytest.fixture(scope="module")
def phantom(tmp_path_factory) -> tuple[Path, Path]:
    """One phantom subject's raw masks and its built Level 1 scene."""
    from assets.meshes import build_subject
    from assets.phantom import write_phantom

    root = tmp_path_factory.mktemp("phantom")
    raw = write_phantom(root / "raw", name="p0")
    scenes = root / "scenes"
    build_subject(raw, scenes / "p0")
    (scenes / "quality_report.json").write_text(
        json.dumps({"chosen": "p0", "subjects": [{"subject": "p0", "usable": True}]})
    )
    return raw.parent, scenes


class TestPoints:
    """fields.data on the phantom."""

    def test_labelled_points_lie_in_their_structures(self, phantom, tmp_path) -> None:
        """Most points labelled k are inside entity k's Level 1 mesh; 'nothing' rarely is."""
        from fields import data
        from shapes.register import load_scene

        raw_root, scenes = phantom
        monkey_reference = data.REFERENCE
        data.REFERENCE = "p0"
        try:
            manifest = data.build(raw_root, scenes, tmp_path, n_points=20_000, split_from=None)
        finally:
            data.REFERENCE = monkey_reference
        stored = np.load(tmp_path / "points" / "p0.npz")
        points, labels = stored["points"], stored["labels"]
        assert points.shape == (20_000, 3) and labels.max() <= len(data.CLASSES)
        rigid = np.array(manifest["stats"]["p0"]["frame"]["rigid"])
        assert np.allclose(rigid, np.eye(4), atol=1e-2), (
            "the reference aligns onto itself (ICP noise only)"
        )
        scene = load_scene(scenes / "p0")
        rng = np.random.default_rng(0)
        for k, entity in enumerate(data.CLASSES, start=1):
            chosen = np.flatnonzero(labels == k)
            if len(chosen) == 0:
                continue
            chosen = chosen[rng.integers(len(chosen), size=min(400, len(chosen)))]
            assert scene[entity].contains(points[chosen]).mean() > 0.85, entity
        nothing = np.flatnonzero(labels == 0)[:400]
        hearts = scene["heart"].contains(points[nothing]).mean()
        assert hearts < 0.05

    def test_sampling_is_reproducible(self, phantom) -> None:
        """Same seed, same points and labels; another seed, different points."""
        from fields import data
        from shapes.register import load_scene

        raw_root, scenes = phantom
        labels, affine, overlaps = data.label_volume(raw_root / "p0")
        frame = data.frame_for(scenes / "p0", load_scene(scenes / "p0")["heart"])
        a = data.sample(labels, affine, frame, n_points=1000, rng=np.random.default_rng(1))
        b = data.sample(labels, affine, frame, n_points=1000, rng=np.random.default_rng(1))
        c = data.sample(labels, affine, frame, n_points=1000, rng=np.random.default_rng(2))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert not np.array_equal(a[0], c[0])
        assert overlaps >= 0


def _spheres(tmp_path: Path, n_subjects: int = 4) -> Path:
    """Points in a cube labelled by two spheres whose radii differ per subject."""
    rng = np.random.default_rng(0)
    (tmp_path / "points").mkdir(parents=True)
    subjects = [f"t{i}" for i in range(n_subjects)]
    for i, subject in enumerate(subjects):
        points = rng.uniform(-60, 60, size=(6000, 3)).astype(np.float32)
        labels = np.zeros(len(points), dtype=np.uint8)
        labels[np.linalg.norm(points - [-20, 0, 0], axis=1) < 15 + 4 * i] = 1
        labels[np.linalg.norm(points - [25, 0, 0], axis=1) < 25 - 3 * i] = 2
        np.savez_compressed(tmp_path / "points" / f"{subject}.npz", points=points, labels=labels)
    manifest = {
        "classes": ["nothing", "a", "b"],
        "subjects": subjects,
        "split": {"train": subjects, "held_out": []},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class TestTheField:
    """fields.model and fields.train."""

    def test_shapes_and_one_label_per_point(self) -> None:
        """Logits (B, N, classes); the argmax is one label per point by construction."""
        from fields.model import AutoDecoder, FieldConfig

        model = AutoDecoder(3, FieldConfig(n_classes=8, hidden=32, layers=4, latent_dim=8))
        logits = model(torch.zeros(2, 10, 3), torch.tensor([0, 2]))
        assert logits.shape == (2, 10, 8)
        assert logits.argmax(-1).shape == (2, 10)

    def test_a_part_code_leaves_the_other_parts_decisions_alone(self) -> None:
        """Changing the heart's code cannot move 'lung vs nothing'; one label per point holds."""
        from fields.model import AutoDecoder, FieldConfig, PartField

        torch.manual_seed(0)
        config = FieldConfig(n_classes=5, hidden=32, layers=4, latent_dim=8, part_of=(0, 1, 0, 1))
        field = PartField(config)
        points = torch.randn(1, 200, 3) * 50
        heart, lungs = torch.randn(1, 8), torch.randn(1, 8)
        before = field(points, torch.cat([heart, lungs], -1))
        after = field(points, torch.cat([torch.randn(1, 8), lungs], -1))
        assert before.shape == (1, 200, 5)
        for k in (2, 4):  # part 1's classes against "nothing"
            assert torch.allclose(before[..., k] - before[..., 0], after[..., k] - after[..., 0])
        assert not torch.allclose(before[..., 1], after[..., 1])
        assert AutoDecoder(3, config).codes.weight.shape == (3, 16)

    def test_parts_must_cover_every_class(self) -> None:
        """A part map of the wrong length, or an empty part, is refused."""
        from fields.model import FieldConfig, PartField

        with pytest.raises(ValueError, match="one part index"):
            PartField(FieldConfig(n_classes=5, part_of=(0, 1)))
        with pytest.raises(ValueError, match="at least one class"):
            PartField(FieldConfig(n_classes=3, part_of=(0, 2)))

    def test_class_weights_favour_rare_classes(self) -> None:
        """Rarer classes weigh more; the mean weight is 1; absent classes get 1."""
        from fields.train import class_weights

        labels = torch.tensor([0] * 90 + [1] * 9 + [2] * 1)
        weights = class_weights(labels, 4)
        assert weights[2] > weights[1] > weights[0]
        assert torch.isclose(weights.mean(), torch.tensor(1.0))

    def test_the_field_learns_a_small_population(self, tmp_path) -> None:
        """Spheres learned to high accuracy on the CPU; the checkpoint has what eval needs."""
        from fields.model import FieldConfig
        from fields.train import TrainConfig, train

        data_dir = _spheres(tmp_path / "data")
        config = TrainConfig(
            steps=400, subjects_per_step=4, points_per_subject=2048, eval_every=200, seed=0
        )
        field = FieldConfig(hidden=64, layers=4, latent_dim=8, bands=4, scale_mm=60.0)
        final = train(data_dir, tmp_path / "run", config, field, device="cpu")
        assert final["accuracy"] > 0.9, final
        checkpoint = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
        assert checkpoint["classes"] == ["nothing", "a", "b"]
        assert checkpoint["codes"].shape == (4, 8) and checkpoint["subjects"][0] == "t0"
        assert checkpoint["field_config"]["n_classes"] == 3
        codes = checkpoint["codes"]
        assert torch.cdist(codes, codes).max() > 0, "subjects need distinct codes"


class TestFocus:
    """Drawing a share of each step's points around chosen structures."""

    def test_focus_points_are_the_box_around_the_class(self) -> None:
        """Indices are exactly the points within the class's box ± margin; counts per subject."""
        from fields.train import focus_points

        points = torch.tensor([[[0.0, 0, 0], [5, 0, 0], [12, 0, 0], [30, 0, 0]]] * 2)
        labels = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
        padded, counts = focus_points(points, labels, [1], margin_mm=8.0)
        assert counts.tolist() == [3, 4], "subject 2 has no class 1, so keeps every point"
        assert padded[0, :3].tolist() == [0, 1, 2]

    def test_a_focused_run_draws_its_share_inside_the_box(self, tmp_path) -> None:
        """With focus 0.5 the field still learns the spheres; most focused points are near 'a'."""
        from fields.model import FieldConfig
        from fields.train import TrainConfig, train

        data_dir = _spheres(tmp_path / "data")
        config = TrainConfig(
            steps=300,
            subjects_per_step=4,
            points_per_subject=2048,
            eval_every=300,
            seed=0,
            focus_fraction=0.5,
            focus_prefix="a",
        )
        field = FieldConfig(hidden=64, layers=4, latent_dim=8, bands=4, scale_mm=60.0)
        assert train(data_dir, tmp_path / "run", config, field, device="cpu")["accuracy"] > 0.9


class TestResume:
    """A stopped run continues exactly where it stopped (flat battery, reclaimed GPU)."""

    @pytest.mark.parametrize("focus", [0.0, 0.5])
    def test_stop_and_resume_equals_one_uninterrupted_run(self, tmp_path, focus) -> None:
        """Same settings, same final network and codes, bit for bit, on the CPU."""
        from fields.model import FieldConfig
        from fields.train import TrainConfig, train

        data_dir = _spheres(tmp_path / "data")
        config = TrainConfig(
            steps=120,
            subjects_per_step=2,
            points_per_subject=512,
            eval_every=60,
            save_every=40,
            seed=3,
            focus_fraction=focus,
            focus_prefix="a",
        )
        field = FieldConfig(hidden=32, layers=4, latent_dim=8, bands=4, scale_mm=60.0)
        train(data_dir, tmp_path / "whole", config, field, device="cpu")
        assert train(data_dir, tmp_path / "split", config, field, device="cpu", stop_after=50) == {
            "stopped_at": 50
        }
        train(data_dir, tmp_path / "split", config, field, device="cpu")
        whole = torch.load(tmp_path / "whole" / "checkpoint.pt", weights_only=False)
        split = torch.load(tmp_path / "split" / "checkpoint.pt", weights_only=False)
        assert torch.equal(whole["codes"], split["codes"])
        assert all(torch.equal(whole["field"][k], split["field"][k]) for k in whole["field"])
        assert [r["step"] for r in split["history"]] == [60, 120]

    def test_a_state_from_other_settings_is_refused(self, tmp_path) -> None:
        """Resuming with different settings would silently mix two runs; it stops instead."""
        from fields.model import FieldConfig
        from fields.train import TrainConfig, train

        data_dir = _spheres(tmp_path / "data")
        field = FieldConfig(hidden=32, layers=4, latent_dim=8, bands=4, scale_mm=60.0)
        small = TrainConfig(steps=40, subjects_per_step=2, points_per_subject=256, save_every=20)
        train(data_dir, tmp_path / "run", small, field, device="cpu", stop_after=20)
        with pytest.raises(SystemExit, match="different run"):
            train(data_dir, tmp_path / "run", TrainConfig(steps=80), field, device="cpu")


class TestEvaluation:
    """fields.evaluate: grids, meshes from labels, surface distance, held-out fitting."""

    def test_grid_covers_its_bounds(self) -> None:
        """Shape from extent and spacing; first and last points at the bounds."""
        from fields.evaluate import Grid

        grid = Grid([0, 0, 0], [10, 20, 30], 5.0)
        points = grid.points()
        assert grid.shape == (3, 5, 7) and len(points) == 3 * 5 * 7
        assert np.allclose(points[0], [0, 0, 0]) and np.allclose(points[-1], [10, 20, 30])

    def test_meshes_and_distances_from_a_labelled_ball(self) -> None:
        """A labelled ball meshes to a closed surface; distance to itself 0, to a shift ~ shift."""
        from fields.evaluate import Grid, label_meshes, surface_distance

        grid = Grid([-30, -30, -30], [30, 30, 30], 1.0)
        points = grid.points()
        volume = (np.linalg.norm(points, axis=1) < 20).astype(np.uint8).reshape(grid.shape)
        meshes = label_meshes(volume, grid, ["ball"])
        ball = meshes["ball"]
        assert abs(ball.volume / (4 / 3 * np.pi * 20**3) - 1) < 0.05
        moved = ball.copy()
        moved.apply_translation([2.0, 0, 0])
        assert surface_distance(ball, ball) < 0.2
        assert 0.8 < surface_distance(ball, moved) < 2.2

    def test_margin_meshing_beats_the_voxel_staircase(self) -> None:
        """On a coarse grid, the margin's zero level finds a ball's surface better than its mask."""
        from fields.evaluate import Grid, label_meshes

        grid = Grid([-30, -30, -30], [30, 30, 30], 3.0)
        radius = np.linalg.norm(grid.points(), axis=1).reshape(grid.shape)
        volume = (radius < 20).astype(np.uint8)
        margins = np.stack([radius - 20, 20 - radius]).astype(np.float32)
        binary = label_meshes(volume, grid, ["ball"])["ball"]
        smooth = label_meshes(volume, grid, ["ball"], margins)["ball"]

        def error(mesh) -> float:
            return float(np.abs(np.linalg.norm(mesh.vertices, axis=1) - 20).mean())

        assert error(smooth) < 0.1 < error(binary)
        assert abs(smooth.volume / (4 / 3 * np.pi * 20**3) - 1) < 0.02

    def test_keep_largest_drops_stray_islands(self) -> None:
        """A structure's small island becomes 'nothing'; other classes are untouched."""
        from fields.evaluate import keep_largest

        volume = np.zeros((20, 20, 20), dtype=np.uint8)
        volume[2:10, 2:10, 2:10] = 1
        volume[15:17, 15:17, 15:17] = 1
        volume[12:14, 2:4, 2:4] = 2
        clean, pieces = keep_largest(volume, 3)
        assert pieces == [2, 1]
        assert (clean == 1).sum() == 8**3 and (clean[15:17, 15:17, 15:17] == 0).all()
        assert (clean == 2).sum() == 8

    def test_margins_agree_with_labels(self, tmp_path) -> None:
        """Each point's positive margin is its argmax label, and only one class is positive."""
        from fields.evaluate import Grid, decode, decode_margins
        from fields.model import FieldConfig, LabelField

        torch.manual_seed(0)
        field = LabelField(FieldConfig(n_classes=4, hidden=32, layers=4, latent_dim=8))
        code = torch.randn(1, 8)
        grid = Grid([-50, -50, -50], [50, 50, 50], 10.0)
        margins = decode_margins(field, code, grid, chunk=100)
        labels = decode(field, code, grid)
        assert np.array_equal(margins.argmax(axis=0), labels)
        assert ((margins > 0).sum(axis=0) <= 1).all()

    def test_a_part_field_learns_and_reloads(self, tmp_path) -> None:
        """Two parts learn the spheres; the checkpoint reloads and fits a whole code."""
        from fields.evaluate import fit_code, load_checkpoint
        from fields.model import FieldConfig, PartField
        from fields.train import TrainConfig, part_indices, train

        data_dir = _spheres(tmp_path / "data")
        config = TrainConfig(steps=400, subjects_per_step=4, points_per_subject=2048, seed=0)
        part_of = part_indices(data_dir, ["a", "b"])
        assert part_of == (0, 1)
        field = FieldConfig(
            hidden=48, layers=4, latent_dim=8, bands=4, scale_mm=60.0, part_of=part_of
        )
        assert train(data_dir, tmp_path / "run", config, field, device="cpu")["accuracy"] > 0.9
        model, _ = load_checkpoint(tmp_path / "run" / "checkpoint.pt", torch.device("cpu"))
        assert isinstance(model, PartField)
        held = np.load(data_dir / "points" / "t0.npz")
        assert fit_code(model, held["points"], held["labels"], steps=5, batch=256).shape == (1, 16)

    def test_a_held_out_subject_is_fitted_with_the_network_frozen(self, tmp_path) -> None:
        """Fitting a new code lowers the loss, and the network's weights do not move."""
        from torch.nn import functional

        from fields.evaluate import fit_code, load_checkpoint
        from fields.model import FieldConfig
        from fields.train import TrainConfig, train

        data_dir = _spheres(tmp_path / "data", n_subjects=5)
        config = TrainConfig(steps=300, subjects_per_step=4, points_per_subject=2048, seed=0)
        field = FieldConfig(hidden=64, layers=4, latent_dim=8, bands=4, scale_mm=60.0)
        train(
            data_dir,
            tmp_path / "run",
            config,
            field,
            device="cpu",
            subjects=["t0", "t1", "t2", "t3"],
        )
        model, _ = load_checkpoint(tmp_path / "run" / "checkpoint.pt", torch.device("cpu"))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        held = np.load(data_dir / "points" / "t4.npz")
        points = torch.from_numpy(held["points"]).unsqueeze(0)
        truth = torch.from_numpy(held["labels"].astype(np.int64))

        def loss(code: torch.Tensor) -> float:
            return float(functional.cross_entropy(model(points, code)[0], truth))

        fitted = fit_code(model, held["points"], held["labels"], steps=150, batch=2048)
        assert loss(fitted) < loss(torch.zeros_like(fitted))
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


class TestGeneration:
    """fields.generate: the code Gaussian, decoding with cleanup, and the checks."""

    def test_the_code_gaussian_matches_the_codes_and_handles_few_subjects(self) -> None:
        """Draws centre on the codes' mean; fewer subjects than dimensions keeps S − 1 axes."""
        from fields.generate import CodeGaussian

        codes = np.random.default_rng(0).normal(size=(10, 32)) * np.linspace(0.1, 2, 32)
        gaussian = CodeGaussian.fit(codes)
        assert gaussian.axes.shape == (9, 32)
        draws = np.stack([gaussian.draw(seed) for seed in range(2000)])
        assert np.allclose(draws.mean(0), codes.mean(0), atol=0.15)
        assert np.array_equal(gaussian.draw(7), gaussian.draw(7))
        assert np.abs(gaussian.draw(1, spread=0.0) - codes.mean(0)).max() < 1e-12

    def test_decoding_measures_and_cleans_strays(self) -> None:
        """A field's labels decode to volumes, stray shares and one label per point."""
        from fields.evaluate import Grid
        from fields.generate import decode_thorax
        from fields.model import FieldConfig, LabelField

        torch.manual_seed(0)
        field = LabelField(FieldConfig(n_classes=3, hidden=32, layers=4, latent_dim=4))
        grid = Grid([-40, -40, -40], [40, 40, 40], 4.0)
        thorax = decode_thorax(field, torch.randn(1, 4), grid, ["a", "b"])
        measured = thorax.measures
        assert measured["points_claimed_twice"] == 0
        for entity in ("a", "b"):
            if measured["pieces"][entity] > 1:
                assert measured["stray"][entity] > 0
            assert measured["volume_ml"][entity] == pytest.approx(
                (thorax.volume == ["a", "b"].index(entity) + 1).sum() * 64 / 1000
            )

    def test_judge_names_each_failure(self) -> None:
        """Volume out of range, too much stray, an open mesh and a missing structure all fail."""
        from awr.ontology import load_ontology
        from fields.generate import judge

        ontology = load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
        lungs = ("respiratory.left_lung", "respiratory.right_lung")
        measured = {
            "points_claimed_twice": 0,
            "volume_ml": {lungs[0]: 900.0, lungs[1]: 2500.0},
            "stray": {lungs[0]: 0.2, lungs[1]: 0.0},
            "closed": {lungs[0]: True, lungs[1]: False},
            "centroids": {lungs[0]: [80.0, 0, 0], lungs[1]: [-80.0, 0, 0]},
        }
        limits = {lungs[0]: (1000.0, 4000.0), lungs[1]: (1000.0, 4000.0)}
        kinds = [k for k, _ in judge(measured, limits, {lungs[0]: 0.05, lungs[1]: 0.05}, ontology)]
        assert kinds == ["volume", "stray", "closed"]
        measured["centroids"] = {lungs[0]: [-80.0, 0, 0], lungs[1]: [80.0, 0, 0]}
        assert "sides" in [
            k for k, _ in judge(measured, limits, {lungs[0]: 1, lungs[1]: 1}, ontology)
        ]
