"""Step 10: the transform algebra parent-relative placement is built on.

Change 1 defines the placement target by composition, so a composition that is subtly wrong
does not raise: it moves the target, and every arm is then measured against something other
than the truth. These tests pin the properties the target depends on.

The central one is closure. Twelve-number frames are not closed under composition when a
rotated parent carries an anisotropic scale, and that defect is invisible on the Step 8/9
corpus only because every rotation in it is exactly the identity. Change 2 replaces that
constant with a real rotation, so the tests below inject rotations deliberately.
"""

from __future__ import annotations

import math

import pytest
import torch

from generation.neural.nn.geometry import frame_rotation, points_to_local
from generation.neural.nn.transforms import (
    DEFAULT_PARENT_CONVENTION,
    PARENT_CONVENTIONS,
    as_parent,
    compose,
    compose_hierarchy,
    frame_to_matrix,
    identity_frame,
    invert,
    matrix_to_frame,
    relative,
    resolve_parents,
    rotation_angle,
    rotation_chordal,
    similarity_residual,
)

CLOSED = ("isotropic", "rigid")


def random_frames(count: int, *, seed: int = 0, scale: float = 0.5) -> torch.Tensor:
    """Frames with strongly anisotropic scales and unconstrained rotations."""
    generator = torch.Generator().manual_seed(seed)
    frame = torch.randn(count, 12, generator=generator, dtype=torch.float64)
    frame[:, 3:6] = torch.randn(count, 3, generator=generator, dtype=torch.float64) * scale
    return frame


def denotes_same(first: torch.Tensor, second: torch.Tensor, tolerance: float = 1e-10) -> bool:
    """Compare the transforms two frames denote, not their digits.

    The 6D rotation is redundant: Gram-Schmidt maps many 6-vectors to one rotation, so
    comparing frames elementwise fails on frames that mean exactly the same thing.
    """
    first_linear, first_translation = frame_to_matrix(first)
    second_linear, second_translation = frame_to_matrix(second)
    return bool(
        torch.allclose(first_linear, second_linear, atol=tolerance)
        and torch.allclose(first_translation, second_translation, atol=tolerance)
    )


class TestFrameMatrixCorrespondence:
    """The matrix form has to mean what the decoder means, or nothing below is grounded."""

    def test_forward_inverts_points_to_local(self) -> None:
        """The matrix form must be the decoder's own map, not a near relative of it."""
        frames = random_frames(16).float()
        points = torch.randn(16, 32, 3)
        local = points_to_local(points, frames)
        rebuilt_linear, rebuilt_translation = frame_to_matrix(frames)
        recovered = (
            torch.einsum("bij,bnj->bni", rebuilt_linear, local) + rebuilt_translation.unsqueeze(1)
        )
        assert torch.allclose(recovered, points, atol=1e-4)

    def test_linear_part_scales_the_columns_not_the_rows(self) -> None:
        """A transposed rotation here would invert every composition, silently."""
        frames = random_frames(8)
        linear, _ = frame_to_matrix(frames)
        rotation = frame_rotation(frames)
        scale = frames[:, 3:6].exp()
        assert torch.allclose(linear, rotation * scale.unsqueeze(-2))
        assert not torch.allclose(linear, rotation.transpose(-1, -2) * scale.unsqueeze(-2))

    def test_matrix_to_frame_round_trips(self) -> None:
        """Recovering a frame from its matrix must return the same transform."""
        frames = random_frames(32)
        assert denotes_same(matrix_to_frame(*frame_to_matrix(frames)), frames)


class TestClosure:
    """Why the parent contributes only an isotropic scale."""

    @pytest.mark.parametrize("convention", CLOSED)
    def test_round_trip_is_exact(self, convention: str) -> None:
        """Under a closed convention the target composes back to the truth exactly."""
        parent, child = random_frames(64, seed=1), random_frames(64, seed=2)
        recovered = compose(parent, relative(parent, child, convention), convention)
        assert denotes_same(recovered, child)

    def test_full_convention_is_not_closed(self) -> None:
        """The defect is real and measured, not assumed away."""
        parent, child = random_frames(64, seed=1), random_frames(64, seed=2)
        recovered = compose(parent, relative(parent, child, "full"), "full")
        assert not denotes_same(recovered, child, tolerance=1e-6)

    def test_closed_conventions_produce_no_shear(self) -> None:
        """Shear is what the frame cannot represent, so there must be none."""
        parent, child = random_frames(64, seed=1), random_frames(64, seed=2)
        for convention in CLOSED:
            linear, _ = frame_to_matrix(as_parent(parent, convention))
            product = linear @ frame_to_matrix(child)[0]
            assert float(similarity_residual(product).max()) < 1e-9

    def test_identity_rotation_hides_the_defect(self) -> None:
        """Documents why Step 8 and Step 9 never saw this."""
        parent, child = random_frames(64, seed=1), random_frames(64, seed=2)
        for frame in (parent, child):
            frame[:, 6:12] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=frame.dtype)
        recovered = compose(parent, relative(parent, child, "full"), "full")
        assert denotes_same(recovered, child)

    def test_default_convention_is_closed(self) -> None:
        """The default must not be the one that loses information."""
        assert DEFAULT_PARENT_CONVENTION in CLOSED
        assert set(CLOSED).issubset(PARENT_CONVENTIONS)


class TestGroupProperties:
    """Identity, inversion and the rotation distance."""

    def test_identity_is_neutral_on_the_left(self) -> None:
        """Composing onto the identity parent leaves a child where it was."""
        child = random_frames(16)
        identity = identity_frame(16, dtype=torch.float64)
        assert denotes_same(compose(identity, child), child)

    def test_identity_is_not_neutral_on_the_right(self) -> None:
        """By design: ``compose`` reduces the parent, so it is not a group operation.

        Pinned because it looks like a bug and removing the asymmetry would silently
        reintroduce the shear that the closed conventions exist to avoid.
        """
        child = random_frames(16)
        identity = identity_frame(16, dtype=torch.float64)
        assert denotes_same(compose(child, identity), as_parent(child))
        assert not denotes_same(compose(child, identity), child, tolerance=1e-6)

    def test_invert_is_a_two_sided_inverse(self) -> None:
        """Inversion must undo composition from either side."""
        frames = as_parent(random_frames(32), "isotropic")
        identity = identity_frame(32, dtype=torch.float64)
        assert denotes_same(compose(frames, invert(frames)), identity)
        assert denotes_same(compose(invert(frames), frames), identity)

    def test_rotation_angle_is_zero_for_equal_rotations_and_pi_for_opposed(self) -> None:
        """The angle is a metric: zero on equality, large on reversal."""
        frames = random_frames(16)
        # arccos amplifies the square root of the epsilon near zero; see rotation_chordal.
        assert float(rotation_angle(frames, frames).abs().max()) < 1e-7
        flipped = frames.clone()
        flipped[:, 6:9] = -frames[:, 6:9]
        flipped[:, 9:12] = -frames[:, 9:12]
        assert float(rotation_angle(frames, flipped).min()) > 0.5

    def test_chordal_distance_orders_the_same_as_the_angle(self) -> None:
        """Change 2 descends the chordal distance and reports the angle, so they must agree."""
        first, second = random_frames(256, seed=8), random_frames(256, seed=9)
        angle = rotation_angle(first, second)
        chordal = rotation_chordal(first, second)
        assert torch.allclose(chordal, (1.0 - torch.cos(angle)) / 2.0, atol=1e-9)
        order = torch.argsort(angle)
        assert bool((chordal[order].diff() >= -1e-12).all())

    def test_chordal_distance_is_zero_without_amplifying_epsilon(self) -> None:
        """The loss form is exact at zero where the angle is not."""
        frames = random_frames(16)
        assert float(rotation_chordal(frames, frames).abs().max()) < 1e-15

    def test_rotation_angle_never_exceeds_pi(self) -> None:
        """A geodesic angle outside [0, pi] means the trace was not clamped."""
        first, second = random_frames(128, seed=3), random_frames(128, seed=4)
        angle = rotation_angle(first, second)
        assert bool(((angle >= 0.0) & (angle <= math.pi + 1e-9)).all())


class TestHierarchy:
    """Composition down a tree, including the parts of it a level of detail hides."""

    @staticmethod
    def _chain() -> tuple[torch.Tensor, torch.Tensor]:
        parents = torch.tensor([-1, 0, 1, 2])
        return parents, torch.tensor([0, 1, 2, 3])

    def test_depth_three_chain_reconstructs_exactly(self) -> None:
        """Three composed levels must be exact, not approximately exact."""
        parents, order = self._chain()
        globals_ = random_frames(4 * 6, seed=5).reshape(6, 4, 12)
        local = globals_.clone()
        for slot in range(1, 4):
            local[:, slot] = relative(globals_[:, slot - 1], globals_[:, slot])
        assert denotes_same(compose_hierarchy(local, parents, order), globals_)

    def test_roots_pass_through_unchanged(self) -> None:
        """A root's frame is already in scene coordinates."""
        parents = torch.tensor([-1, -1, -1])
        local = random_frames(3 * 5, seed=6).reshape(5, 3, 12)
        assert denotes_same(compose_hierarchy(local, parents, torch.tensor([0, 1, 2])), local)

    def test_absent_parent_is_replaced_by_the_nearest_present_ancestor(self) -> None:
        """Skipping a hidden parent must reattach, not detach."""
        parents, order = self._chain()
        present = torch.tensor([[True, False, True, True]])
        resolved = resolve_parents(parents, present, order)
        # slot 1 is absent, so slot 2 must attach to slot 0 rather than to nothing.
        assert int(resolved[0, 2]) == 0
        assert int(resolved[0, 3]) == 2
        assert int(resolved[0, 0]) == -1

    def test_absent_entities_are_marked_rootless(self) -> None:
        """An entity not in the scene cannot be a parent or a child."""
        parents, order = self._chain()
        present = torch.tensor([[True, False, True, True]])
        assert int(resolve_parents(parents, present, order)[0, 1]) == -1

    def test_reparenting_still_reconstructs_the_present_entities(self) -> None:
        """Reparenting must not move the entities that are actually scored."""
        parents, order = self._chain()
        globals_ = random_frames(4 * 8, seed=7).reshape(8, 4, 12)
        present = torch.ones(8, 4, dtype=torch.bool)
        present[:, 1] = False
        resolved = resolve_parents(parents, present, order)
        local = globals_.clone()
        for slot in order.tolist():
            has = resolved[:, slot] >= 0
            if not bool(has.any()):
                continue
            rows = torch.arange(globals_.shape[0])
            candidate = relative(globals_[rows, resolved[:, slot].clamp_min(0)], globals_[:, slot])
            local[:, slot] = torch.where(has.unsqueeze(-1), candidate, globals_[:, slot])
        rebuilt = compose_hierarchy(local, resolved, order)
        assert denotes_same(rebuilt[present], globals_[present])

    def test_unknown_convention_is_rejected(self) -> None:
        """A typo in a convention name must fail loudly, not pick a default."""
        with pytest.raises(ValueError, match="Unknown parent convention"):
            as_parent(random_frames(2), "scaled")
