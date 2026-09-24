"""Step 8: a continuous arrangement space, replacing four discrete variants.

Step 7's corpus offered four arrangements. That was enough to show the relationship
graph carried scene-specific information, and not enough to show a model was *inferring*
from it: four labels can be classified. A model that learns "this is arrangement 2" has
learned nothing about relations.

Step 8 makes the arrangement continuous. Each scene draws a point from a space of
transformations, so no finite set of labels covers it and two scenes essentially never
share an arrangement. Relational inference becomes the only route to placement, because
there is nothing to memorise.

The space has two discrete axes, which are genuinely binary facts about a heart, and six
continuous ones:

======================  ==========  =====================================================
axis                    kind        what it moves
======================  ==========  =====================================================
``mirror``              binary      reflects the organ across the midline
``transpose``           binary      swaps which ventricle each great artery leaves
``yaw``                 continuous  rotation about the vertical axis
``pitch``, ``roll``     continuous  the remaining two rotations
``septal_shift``        continuous  lateral position of the septum between the ventricles
``av_shift``            continuous  height of the atrioventricular plane
``apex_swing``          continuous  anterior-posterior lean of the apex
``chamber_asymmetry``   continuous  relative size of the left and right ventricles
======================  ==========  =====================================================

Every one of these changes where structures sit **relative to one another**, so it
changes the relations measured from the finished organ. None of them changes which
entities are present, and none is visible in the text features. The relationship graph
remains the only channel carrying the arrangement.

Held-out regions are defined on this space rather than on a label, which is what makes
the generalisation tests meaningful. See :data:`SPLIT_RULES`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "Arrangement",
    "ARRANGEMENT_FIELDS",
    "axis_ranges",
    "SplitName",
    "SPLIT_RULES",
    "sample_arrangement",
    "classify_arrangement",
    "arrangement_distance",
]

#: The continuous coordinates, in a fixed order. Used for reporting and for the
#: nearest-neighbour leakage check.
ARRANGEMENT_FIELDS: tuple[str, ...] = (
    "yaw",
    "pitch",
    "roll",
    "septal_shift",
    "av_shift",
    "apex_swing",
    "chamber_asymmetry",
)

SplitName = Literal[
    "train",
    "validation",
    "test_seen",
    "test_arrangement",
    "test_transform",
    "test_combination",
]

#: Full sampling range of each continuous axis, before any split rule applies.
_RANGES: Mapping[str, tuple[float, float]] = {
    "yaw": (-0.60, 0.60),
    "pitch": (-0.14, 0.14),
    "roll": (-0.12, 0.12),
    "septal_shift": (-0.16, 0.16),
    "av_shift": (-0.10, 0.10),
    "apex_swing": (-0.18, 0.18),
    "chamber_asymmetry": (-0.22, 0.22),
}

def axis_ranges() -> dict[str, tuple[float, float]]:
    """Full sampling range of each continuous axis, before any split rule applies.

    A copy, so a caller recording it in a corpus manifest cannot edit the sampler.
    """
    return dict(_RANGES)


#: Boundaries of the held-out regions. Fixed here, before any Step 8 run.
YAW_INTERPOLATION_HOLE: tuple[float, float] = (0.15, 0.25)
"""A band of |yaw| removed from training: interpolation into a hole."""

YAW_EXTRAPOLATION_EDGE: float = 0.45
"""|yaw| above this is removed from training: extrapolation past the edge."""

SEPTAL_HOLE: tuple[float, float] = (0.07, 0.12)
"""A band of |septal_shift| removed from training, so two axes have holes, not one."""


@dataclass(frozen=True, slots=True)
class Arrangement:
    """One point in the arrangement space."""

    mirror: bool
    transpose: bool
    yaw: float
    pitch: float
    roll: float
    septal_shift: float
    av_shift: float
    apex_swing: float
    chamber_asymmetry: float

    @property
    def combination(self) -> tuple[bool, bool]:
        """The discrete part, which the combination hold-out is defined on."""
        return (self.mirror, self.transpose)

    def coordinates(self) -> tuple[float, ...]:
        """The continuous part, in :data:`ARRANGEMENT_FIELDS` order."""
        return tuple(float(getattr(self, name)) for name in ARRANGEMENT_FIELDS)

    def normalised(self) -> tuple[float, ...]:
        """Continuous coordinates scaled to roughly [-1, 1] by their own ranges."""
        out: list[float] = []
        for name in ARRANGEMENT_FIELDS:
            low, high = _RANGES[name]
            span = max(high - low, 1e-9)
            out.append(float((2.0 * (getattr(self, name) - low) / span) - 1.0))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        payload: dict[str, Any] = {"mirror": self.mirror, "transpose": self.transpose}
        payload.update({name: float(getattr(self, name)) for name in ARRANGEMENT_FIELDS})
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Arrangement:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            mirror=bool(payload["mirror"]),
            transpose=bool(payload["transpose"]),
            **{name: float(payload[name]) for name in ARRANGEMENT_FIELDS},
        )

    @property
    def label(self) -> str:
        """A coarse descriptive name. **Never** an input to any model.

        Kept only so that reports can group scenes for a reader. Two scenes sharing a
        label do not share an arrangement.
        """
        parts = []
        if self.mirror:
            parts.append("mirrored")
        if self.transpose:
            parts.append("transposed")
        if abs(self.yaw) > 0.3:
            parts.append("rotated")
        return "+".join(parts) if parts else "upright"


def _outside(value: float, band: tuple[float, float]) -> bool:
    return not (band[0] <= abs(value) <= band[1])


#: How a sampled arrangement is assigned to a split. Checked in order; the first rule
#: that matches wins. Written before any Step 8 corpus was generated.
SPLIT_RULES: tuple[tuple[str, str], ...] = (
    (
        "test_combination",
        "mirror and transpose are both set; that pair never appears in training",
    ),
    (
        "test_transform",
        "|yaw| exceeds the training edge, so the rotation is larger than any seen",
    ),
    (
        "test_arrangement",
        "|yaw| or |septal_shift| falls in a band removed from training",
    ),
    ("train/validation/test_seen", "everything else, split by family and scene index"),
)


def classify_arrangement(arrangement: Arrangement) -> str:
    """Which held-out region an arrangement belongs to, or ``in_distribution``.

    This is the single definition of the split boundaries. The generator uses it, and
    the leakage tests use it, so the two cannot disagree.
    """
    if arrangement.mirror and arrangement.transpose:
        return "test_combination"
    if abs(arrangement.yaw) > YAW_EXTRAPOLATION_EDGE:
        return "test_transform"
    if not _outside(arrangement.yaw, YAW_INTERPOLATION_HOLE):
        return "test_arrangement"
    if not _outside(arrangement.septal_shift, SEPTAL_HOLE):
        return "test_arrangement"
    return "in_distribution"


def sample_arrangement(
    rng: np.random.Generator, *, region: str = "in_distribution"
) -> Arrangement:
    """Draw an arrangement from the space, restricted to one region.

    Rejection sampling against :func:`classify_arrangement`, so the region a scene ends
    up in is decided by the same function the tests use. There is no second definition
    of the boundaries to drift out of step.
    """
    for _ in range(2048):
        if region == "test_combination":
            mirror, transpose = True, True
        else:
            mirror = bool(rng.random() < 0.5)
            transpose = bool(rng.random() < 0.5)
            if mirror and transpose:
                # Reserved for the combination hold-out; resample the pair.
                transpose = False

        values = {
            name: float(rng.uniform(*_RANGES[name])) for name in ARRANGEMENT_FIELDS
        }
        candidate = Arrangement(mirror=mirror, transpose=transpose, **values)
        if region == "test_combination":
            # Isolate the discrete pair. Letting the continuous coordinates wander into
            # the other held-out regions would confound "this combination is new" with
            # "this rotation is new", and the experiment could not say which mattered.
            probe = Arrangement(mirror=False, transpose=False, **values)
            if classify_arrangement(probe) == "in_distribution":
                return candidate
            continue
        if classify_arrangement(candidate) == region:
            return candidate
    raise RuntimeError(f"Could not sample an arrangement in region {region!r}.")


def arrangement_distance(first: Arrangement, second: Arrangement) -> float:
    """Distance between two arrangements, for the nearest-neighbour leakage check.

    Infinite when the discrete part differs, because a mirrored organ is not a small
    perturbation of an unmirrored one. Otherwise the Euclidean distance between the
    normalised continuous coordinates.
    """
    if first.combination != second.combination:
        return math.inf
    left = np.asarray(first.normalised())
    right = np.asarray(second.normalised())
    return float(np.linalg.norm(left - right))


def nearest_neighbour_distances(
    queries: Sequence[Arrangement], reference: Sequence[Arrangement]
) -> list[float]:
    """For each query, the distance to its closest arrangement in ``reference``.

    The leakage check: if a held-out scene's nearest training arrangement is at distance
    zero, the hold-out is not held out.
    """
    out: list[float] = []
    for query in queries:
        best = math.inf
        for other in reference:
            best = min(best, arrangement_distance(query, other))
        out.append(best)
    return out
