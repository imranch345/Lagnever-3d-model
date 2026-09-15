"""The Step 9 frame decomposition, reported so nothing is hidden inside a composite.

Step 8's composite frame error folded three components together at fixed weights. An audit
of the frame target found that one of them measures nothing:

    rotation 6D across the corpus : exactly one unique value
    measured rotation_error       : 0.000000 for every arm

The frame target stores the entity's measured centroid, the log of its measured extent, and
a rotation that is the identity for every entity in every scene. The organ's own yaw, pitch
and roll reach the target only indirectly, by moving centroids. So the placement task is six
numbers, not twelve, and a quarter of the composite's weight is spent on a constant.

**The Step 8 composite is preserved unchanged**, because changing a metric after seeing
results is the practice these reports exist to avoid, and because comparability with Step 8
matters more than tidiness. Alongside it this module reports:

``translation_error``
    Euclidean distance between predicted and true centroid, in scene units.
``scale_error``
    mean absolute log-scale error, a symmetric ratio error.
``rotation_error``
    the geodesic angle. **Structurally zero under the current target.** Reported at full
    precision so a future change to the target would show up here rather than pass silently.
``composite_frame_error``
    the Step 8 quantity, `position + 0.5*scale + 0.25*rotation`, unchanged.
``placement_error``
    the corrected, position-oriented metric introduced in Step 9:
    `position + 0.5*scale`, with the vacuous term removed. Numerically identical to the
    composite on the current target, which is itself the point: the difference between them
    is exactly the amount of information the rotation term carries, and it is zero.

Both are reported for every arm on every split. Neither replaces the other.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "ROTATION_IS_STRUCTURALLY_ZERO",
    "decompose",
    "floor_gap",
    "describe_rotation",
]

#: What the audit established about the frame target, asserted here so a test can check it.
ROTATION_IS_STRUCTURALLY_ZERO: bool = True

#: Tolerance below which a measured rotation error counts as confirming the above.
ROTATION_ZERO_TOLERANCE: float = 1.0e-5


def describe_rotation(measured: float) -> str:
    """One sentence on what a measured rotation error means under the current target."""
    if abs(measured) <= ROTATION_ZERO_TOLERANCE:
        return (
            "structurally zero: the frame target's rotation is a constant identity for "
            "every entity in every scene, so this component carries no information"
        )
    return (
        f"non-zero at {measured:.6f}, which means the frame target's rotation is no longer "
        "constant; the Step 8 composite's weighting should be revisited before it is read"
    )


def decompose(values: Mapping[str, float]) -> dict[str, float]:
    """Add the corrected placement metric beside the Step 8 composite.

    The Step 8 keys are passed through untouched. ``placement_error`` is added, and
    ``rotation_information`` records the difference the rotation term makes, which is the
    honest way to show that it makes none.
    """
    position = float(values.get("position_error", float("nan")))
    scale = float(values.get("scale_error", float("nan")))
    rotation = float(values.get("rotation_error", float("nan")))
    placement = position + 0.5 * scale
    return {
        **{key: float(value) for key, value in values.items()},
        "translation_error": position,
        "placement_error": placement,
        # What the rotation term contributes to the Step 8 composite. Zero under the
        # current target, and the only honest way to show that is to report it.
        "rotation_information": 0.25 * rotation,
    }


def floor_gap(error: float, floor: float) -> float:
    """Improvement over the placement-blind floor, as a share of the floor.

    Positive means the model placed entities better than a lookup table keyed on entity
    identity alone. Negative means it did worse, and the model has learned nothing about
    placement that identity did not already supply.
    """
    if floor <= 1.0e-9:
        return float("nan")
    return float((floor - error) / floor)
