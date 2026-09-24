"""Step 10 Change 2 §7: the transform convention under genuinely non-identity rotations.

Change 1 ran on a corpus whose every rotation was the identity, which is its own transpose
and its own inverse. Almost any convention survives that. These tests run the convention at
the angles the brief names — 0, 30, 90 and 180 degrees — with anisotropic child scales and
non-unit parent scales, before a rotated corpus is generated.

Two claims are checked, and the second is the stronger one:

*round trip*
    ``compose(parent, relative(parent, child)) == child`` to machine precision.

*no hidden shear or projection loss*
    the frame the composition returns represents **exactly** the affine map
    ``parent_as_parent @ child``, verified by mapping points through both and comparing. A
    representation that silently projected a sheared matrix back onto rotation-times-diagonal
    would pass a round trip built from the same projection but fail this.

``full`` is included only to demonstrate that it is the convention that loses information,
which is why the closed ones are used.
"""

from __future__ import annotations

import math

import pytest
import torch

from generation.neural.nn.transforms import (
    as_parent,
    compose,
    frame_to_matrix,
    relative,
    similarity_residual,
)

CLOSED: tuple[str, ...] = ("isotropic", "rigid")
ANGLES_DEG: tuple[float, ...] = (0.0, 30.0, 90.0, 180.0)
AXES: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.36, 0.48, 0.80),
)
TOLERANCE = 1e-12


def _rotation(axis: tuple[float, float, float], degrees: float) -> torch.Tensor:
    """Rodrigues rotation, in float64, as a 3x3 matrix."""
    vector = torch.tensor(axis, dtype=torch.float64)
    vector = vector / vector.norm()
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    cross = torch.tensor(
        [
            [0.0, -float(vector[2]), float(vector[1])],
            [float(vector[2]), 0.0, -float(vector[0])],
            [-float(vector[1]), float(vector[0]), 0.0],
        ],
        dtype=torch.float64,
    )
    return (
        torch.eye(3, dtype=torch.float64)
        + sin * cross
        + (1.0 - cos) * (cross @ cross)
    )


def _frame(
    translation: tuple[float, float, float],
    log_scale: tuple[float, float, float],
    rotation: torch.Tensor,
) -> torch.Tensor:
    """A frame from a translation, a log scale and a rotation whose rows are its axes."""
    return torch.cat(
        [
            torch.tensor(translation, dtype=torch.float64),
            torch.tensor(log_scale, dtype=torch.float64),
            rotation[0],
            rotation[1],
        ]
    )


def _map_points(frame: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Take local points through a frame into its parent's space."""
    linear, translation = frame_to_matrix(frame)
    return points @ linear.transpose(-1, -2) + translation


PROBES = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, -0.3, 0.45],
        [-0.9, 0.6, -0.2],
    ],
    dtype=torch.float64,
)


@pytest.mark.parametrize("convention", CLOSED)
@pytest.mark.parametrize("parent_degrees", ANGLES_DEG)
@pytest.mark.parametrize("child_degrees", ANGLES_DEG)
class TestTheRoundTripUnderRealRotations:
    """A parent-relative target must compose back to the frame it came from."""

    def test_round_trip_with_an_anisotropic_child(
        self, convention: str, parent_degrees: float, child_degrees: float
    ) -> None:
        """The child keeps a full anisotropic scale; only the parent is reduced."""
        parent = _frame((0.3, -0.2, 0.55), (0.2, 0.2, 0.2), _rotation(AXES[0], parent_degrees))
        child = _frame((-0.4, 0.8, 0.1), (0.5, -0.3, 0.15), _rotation(AXES[2], child_degrees))
        local = relative(parent, child, convention)
        rebuilt = compose(parent, local, convention)
        assert float((rebuilt - child).abs().max()) < TOLERANCE

    def test_round_trip_with_a_scaled_rotated_parent(
        self, convention: str, parent_degrees: float, child_degrees: float
    ) -> None:
        """A parent with a non-unit scale and a real rotation, which is the hard case."""
        parent = _frame((1.1, 0.4, -0.7), (0.6, 0.6, 0.6), _rotation(AXES[2], parent_degrees))
        child = _frame((0.25, -0.15, 0.6), (-0.2, 0.4, 0.05), _rotation(AXES[1], child_degrees))
        local = relative(parent, child, convention)
        rebuilt = compose(parent, local, convention)
        assert float((rebuilt - child).abs().max()) < TOLERANCE


@pytest.mark.parametrize("convention", CLOSED)
class TestNoShearAndNoProjectionLoss:
    """The composed frame must *be* the composed map, not a projection of it."""

    @pytest.mark.parametrize("parent_degrees", ANGLES_DEG)
    @pytest.mark.parametrize("child_degrees", ANGLES_DEG)
    def test_the_frame_represents_the_composed_affine_map_exactly(
        self, convention: str, parent_degrees: float, child_degrees: float
    ) -> None:
        """Points mapped through the composition equal points mapped through both frames."""
        parent = _frame((0.3, -0.2, 0.55), (0.35, 0.35, 0.35), _rotation(AXES[2], parent_degrees))
        child = _frame((-0.4, 0.8, 0.1), (0.5, -0.3, 0.15), _rotation(AXES[1], child_degrees))
        composed = compose(parent, child, convention)
        stepwise = _map_points(as_parent(parent, convention), _map_points(child, PROBES))
        assert float((_map_points(composed, PROBES) - stepwise).abs().max()) < TOLERANCE

    @pytest.mark.parametrize("parent_degrees", ANGLES_DEG)
    def test_the_composed_linear_part_carries_no_shear(
        self, convention: str, parent_degrees: float
    ) -> None:
        """The product of parent and child linear parts stays rotation-times-diagonal."""
        parent = _frame((0.0, 0.0, 0.0), (0.4, 0.4, 0.4), _rotation(AXES[2], parent_degrees))
        child = _frame((0.2, 0.1, -0.3), (0.6, -0.25, 0.1), _rotation(AXES[0], 30.0))
        parent_linear, _ = frame_to_matrix(as_parent(parent, convention))
        child_linear, _ = frame_to_matrix(child)
        assert float(similarity_residual(parent_linear @ child_linear).abs().max()) < 1e-10

    def test_a_deep_chain_stays_exact(self, convention: str) -> None:
        """Depth 3, every level rotated and scaled: the corpus's deepest chain."""
        frames = [
            _frame((0.1 * i, -0.2 * i, 0.3 * i), (0.2, 0.2, 0.2), _rotation(AXES[i % 3], 30.0 * i))
            for i in range(4)
        ]
        frames[3] = _frame((0.4, 0.1, -0.2), (0.5, -0.3, 0.15), _rotation(AXES[2], 180.0))
        locals_ = [frames[0]]
        for depth in range(1, 4):
            locals_.append(relative(frames[depth - 1], frames[depth], convention))
        rebuilt = locals_[0]
        for depth in range(1, 4):
            rebuilt = compose(rebuilt, locals_[depth], convention)
        assert float((rebuilt - frames[3]).abs().max()) < 1e-11


class TestTheUnclosedConventionIsWhyTheseAreUsed:
    """`full` loses information once rotations are real; the closed conventions do not."""

    def test_full_shears_and_loses_it(self) -> None:
        """A rotated parent with an anisotropic scale produces shear that is discarded."""
        parent = _frame((0.0, 0.0, 0.0), (0.7, -0.2, 0.3), _rotation(AXES[2], 45.0))
        child = _frame((0.3, 0.2, -0.1), (0.4, -0.35, 0.2), _rotation(AXES[1], 60.0))
        parent_linear, _ = frame_to_matrix(as_parent(parent, "full"))
        child_linear, _ = frame_to_matrix(child)
        assert float(similarity_residual(parent_linear @ child_linear)) > 1e-3
        local = relative(parent, child, "full")
        rebuilt = compose(parent, local, "full")
        assert float((rebuilt - child).abs().max()) > 1e-6

    def test_identity_rotations_hide_the_defect(self) -> None:
        """Why Change 1's corpus could not have caught it: no rotation, no shear."""
        identity = torch.eye(3, dtype=torch.float64)
        parent = _frame((0.0, 0.0, 0.0), (0.7, -0.2, 0.3), identity)
        child = _frame((0.3, 0.2, -0.1), (0.4, -0.35, 0.2), identity)
        parent_linear, _ = frame_to_matrix(as_parent(parent, "full"))
        child_linear, _ = frame_to_matrix(child)
        assert float(similarity_residual(parent_linear @ child_linear)) < 1e-12
        rebuilt = compose(parent, relative(parent, child, "full"), "full")
        assert float((rebuilt - child).abs().max()) < 1e-12


class TestTheRotationMetricAtAndNearIdentity:
    """§8: the reported metric must be defined at identity and just off it."""

    @staticmethod
    def _angle(first: torch.Tensor, second: torch.Tensor) -> float:
        from generation.neural.nn.transforms import rotation_angle

        return float(rotation_angle(first, second))

    def test_identity_against_itself_is_finite_and_tiny(self) -> None:
        """Arccos at 1.0 is the singular point; it must return a number, not a NaN."""
        frame = _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), torch.eye(3, dtype=torch.float64))
        angle = self._angle(frame, frame)
        assert math.isfinite(angle)
        assert angle < 1e-6

    @pytest.mark.parametrize("degrees", (1e-4, 1e-2, 0.5, 30.0, 179.9))
    def test_small_and_large_angles_are_recovered(self, degrees: float) -> None:
        """The measured angle is the angle that was applied."""
        identity = _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), torch.eye(3, dtype=torch.float64))
        rotated = _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _rotation(AXES[2], degrees))
        assert self._angle(identity, rotated) == pytest.approx(math.radians(degrees), abs=1e-4)

    def test_the_chordal_form_is_monotone_where_the_geodesic_is_steep(self) -> None:
        """Why optimisation uses the chordal form: it is smooth where arccos is not."""
        from generation.neural.nn.transforms import rotation_chordal

        identity = _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), torch.eye(3, dtype=torch.float64))
        previous = -1.0
        for degrees in (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0, 60.0, 120.0, 180.0):
            rotated = _frame((0.0,) * 3, (0.0,) * 3, _rotation(AXES[2], degrees))
            value = float(rotation_chordal(identity, rotated))
            assert value >= previous - 1e-15
            previous = value

    def test_the_geodesic_gradient_blows_up_at_zero_and_the_chordal_does_not(self) -> None:
        """The concrete reason the geodesic form is not the training loss."""
        from generation.neural.nn.transforms import rotation_angle, rotation_chordal

        identity = _frame((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), torch.eye(3, dtype=torch.float64))
        rotated = _frame(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _rotation(AXES[2], 1e-3)
        ).requires_grad_(True)
        rotation_angle(identity, rotated).backward()
        geodesic = float(rotated.grad.abs().max())  # type: ignore[union-attr]
        rotated2 = _frame(
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), _rotation(AXES[2], 1e-3)
        ).requires_grad_(True)
        rotation_chordal(identity, rotated2).backward()
        chordal = float(rotated2.grad.abs().max())  # type: ignore[union-attr]
        assert geodesic > 100.0 * max(chordal, 1e-12)
