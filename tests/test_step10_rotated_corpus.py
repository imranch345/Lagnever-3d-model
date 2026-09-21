"""Step 10 Change 2 §14: the rotated corpus's integrity, and the corruptions it refuses.

The four corruptions the brief names are applied deliberately to a real generated scene, and
each must raise. Two of them — a rotation replaced by the identity, and a rotation stored
transposed — are well formed in every local sense: orthonormal, right-handed, exactly
composable. Nothing but a comparison against the generator's own construction can see them,
which is why `check_corpus_rotations` exists and why these tests assert that it, specifically,
is what catches them.

The correct corpus passing the same checks is asserted first, so a gate that rejected
everything would fail here too.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.field import WholeOrganField, _basis_from_axis, _normalise
from datasets.whole_organ.parameters import sample_parameters
from datasets.whole_organ.rotation_stats import (
    basis_to_matrix,
    geodesic_angle_degrees,
    summarise_angles,
)
from generation.neural.nn.geometry import frame_rotation
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.placement_integrity import (
    PlacementIntegrityError,
    check_corpus_rotations,
    check_frames,
    check_order,
)


@pytest.fixture(scope="module")
def rotated_scene():
    """One generated scene carrying measured rotations."""
    arrangement = sample_arrangement(np.random.default_rng(5), region="in_distribution")
    return build_scene(
        scene_index=11, family_id=4, lod=3, arrangement=arrangement, rotations=True
    )


@pytest.fixture(scope="module")
def plain_scene():
    """The same scene measured without rotations: its unrotated twin."""
    arrangement = sample_arrangement(np.random.default_rng(5), region="in_distribution")
    return build_scene(scene_index=11, family_id=4, lod=3, arrangement=arrangement)


def _corrupt(scene, entity_id: str, rows: list[float]):
    """A copy of ``scene`` with one entity's stored rotation replaced."""
    rotations = dict(scene.rotations)
    rotations[entity_id] = rows
    return replace(scene, rotations=rotations)


def _angles(scene) -> dict[str, float]:
    return {
        entity_id: geodesic_angle_degrees(basis_to_matrix(rows))
        for entity_id, rows in scene.rotations.items()
    }


def _victim(scene) -> str:
    """An entity whose rotation is far from the identity, so a corruption is meaningful."""
    angles = _angles(scene)
    return max(angles, key=lambda key: angles[key])


def _most_asymmetric(scene) -> str:
    """The entity whose rotation is furthest from symmetric, which is nearest 90 degrees.

    A rotation of exactly 180 degrees equals its own transpose, and one near 180 degrees is
    nearly symmetric, so transposing the *largest* rotation in a scene is close to a no-op.
    Transposition bites hardest near 90 degrees, and that is where this corruption is tested.
    """
    angles = _angles(scene)
    return min(angles, key=lambda key: abs(angles[key] - 90.0))


class TestTheCorrectCorpusPasses:
    """Controls: the real thing passes every check the corruptions must fail."""

    def test_the_scene_carries_rotations(self, rotated_scene) -> None:
        """A rotated corpus must actually be rotated."""
        assert rotated_scene.has_rotations
        assert len(rotated_scene.rotations) == len(rotated_scene.centroids)

    def test_the_rotations_match_the_generator(self, rotated_scene) -> None:
        """The check that will be run over every scene of the corpus."""
        report = check_corpus_rotations([rotated_scene])
        assert report["rotations_checked"] == len(rotated_scene.centroids)
        assert report["worst_deviation"] < 1e-6

    def test_the_rotations_are_well_formed(self, rotated_scene) -> None:
        """Orthonormal, right-handed, and genuinely not the identity."""
        frames = torch.tensor(
            [rotated_scene.frames()[k] for k in rotated_scene.centroids], dtype=torch.float64
        ).unsqueeze(0)
        present = torch.ones(frames.shape[:2], dtype=torch.bool)
        report = check_frames(frames, present)
        assert report["max_rotation_deviation_from_identity"] > 0.5
        rotations = frame_rotation(frames)[0]
        for rotation in rotations:
            identity = torch.eye(3, dtype=torch.float64)
            assert torch.allclose(rotation @ rotation.T, identity, atol=1e-9)
            assert float(torch.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-9)

    def test_rotation_is_present_at_a_meaningful_magnitude(self, rotated_scene) -> None:
        """Predicting the identity must not be nearly free."""
        angles = [
            geodesic_angle_degrees(basis_to_matrix(rows))
            for rows in rotated_scene.rotations.values()
        ]
        stats = summarise_angles(angles)
        assert stats["max_deg"] > 60.0
        assert stats["fraction_below_1_deg"] == 0.0
        assert stats["mean_deg"] > 10.0

    def test_the_rotated_scene_is_its_plain_twin_apart_from_the_frame(
        self, rotated_scene, plain_scene
    ) -> None:
        """§13: nothing but the frame measurement changed."""
        assert rotated_scene.scene_id == plain_scene.scene_id
        assert rotated_scene.edges == plain_scene.edges
        assert rotated_scene.counts == plain_scene.counts
        assert rotated_scene.entity_ids == plain_scene.entity_ids
        assert rotated_scene.active_lod == plain_scene.active_lod
        for entity_id, centroid in plain_scene.centroids.items():
            assert rotated_scene.centroids[entity_id] == pytest.approx(centroid, abs=0.0)
        assert not plain_scene.has_rotations

    def test_the_extent_basis_is_the_one_intended_change(
        self, rotated_scene, plain_scene
    ) -> None:
        """Extents move because an oriented frame measures them on its own axes."""
        moved = [
            entity_id
            for entity_id, extent in plain_scene.extents.items()
            if not np.allclose(extent, rotated_scene.extents[entity_id], atol=1e-6)
        ]
        assert moved, "extents are unchanged, so they were not measured in the new frame"


class TestCorruptionAIdentityInsteadOfARotation:
    """§14 A. Well formed, exactly composable, and wrong."""

    IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    def test_an_identity_rotation_is_refused(self, rotated_scene) -> None:
        """The generator comparison catches it."""
        corrupt = _corrupt(rotated_scene, _victim(rotated_scene), self.IDENTITY)
        with pytest.raises(PlacementIntegrityError, match="not the generator's"):
            check_corpus_rotations([corrupt])

    def test_it_passes_every_local_check(self, rotated_scene) -> None:
        """Which is why the generator comparison is the only thing that can catch it."""
        corrupt = _corrupt(rotated_scene, _victim(rotated_scene), self.IDENTITY)
        frames = torch.tensor(
            [corrupt.frames()[k] for k in corrupt.centroids], dtype=torch.float64
        ).unsqueeze(0)
        present = torch.ones(frames.shape[:2], dtype=torch.bool)
        check_frames(frames, present)


class TestCorruptionBATransposedRotation:
    """§14 B. The corruption Change 1's corpus could not express."""

    @staticmethod
    def _transposed(rows: list[float]) -> list[float]:
        matrix = basis_to_matrix(rows).T
        return [float(x) for x in matrix[:2].reshape(-1)]

    def test_a_transposed_rotation_is_refused(self, rotated_scene) -> None:
        """Caught, and only by the generator comparison."""
        victim = _most_asymmetric(rotated_scene)
        rows = self._transposed(rotated_scene.rotations[victim])
        corrupt = _corrupt(rotated_scene, victim, rows)
        with pytest.raises(PlacementIntegrityError, match="not the generator's"):
            check_corpus_rotations([corrupt])

    def test_the_transpose_is_a_real_change_on_this_corpus(self, rotated_scene) -> None:
        """On the identity-rotation corpus this corruption was a no-op; here it is not."""
        victim = _most_asymmetric(rotated_scene)
        original = basis_to_matrix(rotated_scene.rotations[victim])
        transposed = basis_to_matrix(self._transposed(rotated_scene.rotations[victim]))
        assert float(np.abs(original - transposed).max()) > 0.1

    def test_a_rotation_near_180_degrees_is_nearly_its_own_transpose(self, rotated_scene) -> None:
        """Why the victim is chosen near 90 degrees and not at the maximum angle."""
        largest = _victim(rotated_scene)
        assert _angles(rotated_scene)[largest] > 150.0
        original = basis_to_matrix(rotated_scene.rotations[largest])
        transposed = basis_to_matrix(self._transposed(rotated_scene.rotations[largest]))
        assert float(np.abs(original - transposed).max()) < 0.1

    def test_it_is_still_a_valid_rotation(self, rotated_scene) -> None:
        """Orthonormal and right-handed, so well-formedness cannot see it."""
        victim = _most_asymmetric(rotated_scene)
        matrix = basis_to_matrix(self._transposed(rotated_scene.rotations[victim]))
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-9)


class TestCorruptionCANonOrthonormalRotation:
    """§14 C. This one the well-formedness check can see."""

    SKEWED = [1.0, 0.0, 0.0, 0.6, 0.8, 0.0]

    def test_a_non_orthonormal_basis_is_refused_by_the_frame_check(self, rotated_scene) -> None:
        """Read before Gram-Schmidt, which would otherwise repair it silently."""
        corrupt = _corrupt(rotated_scene, _victim(rotated_scene), self.SKEWED)
        frames = torch.tensor(
            [corrupt.frames()[k] for k in corrupt.centroids], dtype=torch.float64
        ).unsqueeze(0)
        present = torch.ones(frames.shape[:2], dtype=torch.bool)
        with pytest.raises(PlacementIntegrityError, match="not orthonormal"):
            check_frames(frames, present)

    def test_it_is_also_refused_by_the_generator_comparison(self, rotated_scene) -> None:
        """Two independent checks catch it, which is the belt-and-braces case."""
        corrupt = _corrupt(rotated_scene, _victim(rotated_scene), self.SKEWED)
        with pytest.raises(PlacementIntegrityError, match="not the generator's"):
            check_corpus_rotations([corrupt])

    def test_a_scene_with_no_rotations_at_all_is_refused(self, plain_scene) -> None:
        """A corpus generated without rotations must not pass as a rotated one."""
        with pytest.raises(PlacementIntegrityError, match="not a rotated corpus"):
            check_corpus_rotations([plain_scene])


class TestCorruptionDABrokenCompositionOrder:
    """§14 D. Unchanged from Change 1, and it must stay caught under real rotations."""

    def test_a_reversed_order_is_refused(self) -> None:
        """Children composed before parents."""
        from awr.config import load_domain_config
        from awr.ontology import load_ontology
        from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

        domain = load_domain_config()
        ontology = load_ontology(
            domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
        )
        builder = WholeOrganBatchBuilder(ontology, domain)
        placement = HierarchicalPlacement(dict(builder.slot_of), slots=len(builder.slot_of))
        with pytest.raises(PlacementIntegrityError, match="before its parent"):
            check_order(placement.table, tuple(reversed(placement.order)))


class TestTheFrameConventionIsStable:
    """The declared basis must not jump between scenes, or the target is unlearnable."""

    def test_no_entity_sits_near_the_reference_switch(self) -> None:
        """``_basis_from_axis`` swaps its reference at |x| = 0.9; nothing may hover there.

        An entity whose construction axis crossed that boundary between scenes would have a
        frame that jumps discontinuously for no geometric reason, and no model could learn
        it. Measured across every region, not argued.
        """
        rng = np.random.default_rng(0)
        regions = ("in_distribution", "test_arrangement", "test_transform", "test_combination")
        closest = 1.0
        for index in range(40):
            arrangement = sample_arrangement(rng, region=regions[index % len(regions)])
            field = WholeOrganField(
                sample_parameters(
                    np.random.default_rng(index * 97), index % 40, arrangement=arrangement
                )
            )
            axes = [annulus.axis for annulus in field.annuli]
            axes += [_normalise(tube.end - tube.start) for tube in field.tubes]
            for axis in axes:
                closest = min(closest, abs(abs(float(_normalise(axis)[0])) - 0.9))
        assert closest > 0.05, f"an axis comes within {closest:.3f} of the reference switch"

    def test_the_basis_is_right_handed_and_carries_the_axis(self) -> None:
        """b3 is the construction axis and b1 x b2 == b3."""
        for axis in ([0.0, 0.0, 1.0], [0.9, 0.1, 0.4], [-0.2, 0.95, 0.24], [0.958, 0.26, 0.11]):
            basis = _basis_from_axis(np.asarray(axis, dtype=np.float64))
            assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
            assert np.allclose(np.cross(basis[0], basis[1]), basis[2], atol=1e-12)
            assert np.allclose(basis[2], _normalise(np.asarray(axis, dtype=np.float64)), atol=1e-12)


class TestTheAuditReadsWhatTheModelReads:
    """The NumPy reconstruction must match the model's own Gram-Schmidt exactly."""

    def test_basis_to_matrix_matches_frame_rotation(self, rotated_scene) -> None:
        """Otherwise the audit would describe rotations the model never sees."""
        for rows in rotated_scene.rotations.values():
            mine = basis_to_matrix(rows)
            theirs = frame_rotation(
                torch.tensor([0.0] * 6 + list(rows), dtype=torch.float64)
            ).numpy()
            assert np.allclose(mine, theirs, atol=1e-12)
