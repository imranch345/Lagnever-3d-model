"""Relationships measured from the finished organ, not asserted from the ontology.

This is the change that makes Step 7's central question answerable. In Step 6 every
scene carried the same ontology graph, so the graph encoder read a constant and could
not have been shown to matter. Here the graph is **measured from each generated organ**,
so a mirrored heart really does report ``right_of`` where a normal one reports
``left_of``, and a transposed heart really does report the aorta leaving the right
ventricle.

The pair list is fixed; the relation **types** and the functional **endpoints** are what
vary. A model that ignores relation types therefore cannot tell the variants apart,
which is exactly the control the experiment needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from datasets.whole_organ.field import WholeOrganField

__all__ = [
    "MeasuredEdge",
    "SPATIAL_PAIRS",
    "ADJACENCY_PAIRS",
    "measure_relations",
    "relation_signature",
]

FloatArray = npt.NDArray[np.float64]

SPATIAL_PAIRS: tuple[tuple[str, str, int], ...] = (
    ("heart.left_ventricle", "heart.right_ventricle", 0),
    ("heart.left_atrium", "heart.right_atrium", 0),
    ("heart.left_atrium", "heart.left_ventricle", 1),
    ("heart.right_atrium", "heart.right_ventricle", 1),
    ("heart.right_ventricle", "heart.left_ventricle", 2),
    ("heart.left_atrium", "heart.right_atrium", 2),
    ("heart.aorta", "heart.pulmonary_trunk", 0),
    ("heart.pulmonary_trunk", "heart.aorta", 2),
    ("heart.superior_vena_cava", "heart.right_atrium", 1),
    ("heart.inferior_vena_cava", "heart.right_atrium", 1),
    ("heart.pulmonary_veins", "heart.left_atrium", 2),
    ("heart.mitral_valve", "heart.left_ventricle", 1),
    ("heart.tricuspid_valve", "heart.right_ventricle", 1),
    ("heart.interventricular_septum", "heart.left_ventricle", 0),
    ("heart.interatrial_septum", "heart.left_atrium", 0),
)
"""Ordered pairs whose spatial relation is measured, with the axis to compare on.

Axis 0 is left and right, 1 is superior and inferior, 2 is anterior and posterior. The
relation emitted depends on which way the measurement comes out, which is what carries
the variant.
"""

_AXIS_RELATIONS: tuple[tuple[str, str], ...] = (
    ("left_of", "right_of"),
    ("superior_to", "inferior_to"),
    ("anterior_to", "posterior_to"),
)

ADJACENCY_PAIRS: tuple[tuple[str, str], ...] = (
    ("heart.interventricular_septum", "heart.left_ventricle"),
    ("heart.interventricular_septum", "heart.right_ventricle"),
    ("heart.interatrial_septum", "heart.left_atrium"),
    ("heart.interatrial_septum", "heart.right_atrium"),
    ("heart.mitral_valve", "heart.left_atrium"),
    ("heart.mitral_valve", "heart.left_ventricle"),
    ("heart.tricuspid_valve", "heart.right_atrium"),
    ("heart.tricuspid_valve", "heart.right_ventricle"),
    ("heart.aortic_valve", "heart.left_ventricle"),
    ("heart.aortic_valve", "heart.right_ventricle"),
    ("heart.pulmonary_valve", "heart.left_ventricle"),
    ("heart.pulmonary_valve", "heart.right_ventricle"),
    ("heart.endocardium", "heart.myocardium"),
    ("heart.myocardium", "heart.epicardium"),
    ("heart.epicardium", "heart.pericardium"),
)
"""Pairs tested for contact. Both arterial valves are tested against both ventricles, so
the transposed variant shows up as a different adjacency, not as a different pair list."""

_ADJACENCY_TOLERANCE = 0.06


@dataclass(frozen=True, slots=True)
class MeasuredEdge:
    """One relation measured from a generated organ."""

    subject: str
    relation: str
    object: str

    def key(self) -> str:
        """Stable string key."""
        return f"{self.subject}|{self.relation}|{self.object}"


def _min_distance(first: FloatArray, second: FloatArray) -> float:
    if first.shape[0] == 0 or second.shape[0] == 0:
        return float("inf")
    sample_a = first[:: max(1, first.shape[0] // 200)]
    sample_b = second[:: max(1, second.shape[0] // 200)]
    deltas = sample_a[:, None, :] - sample_b[None, :, :]
    return float(np.sqrt((deltas**2).sum(axis=-1)).min())


def measure_relations(
    field: WholeOrganField,
    statistics: Mapping[str, Mapping[str, FloatArray]],
    rng: np.random.Generator,
    *,
    cloud_points: int = 600,
) -> tuple[MeasuredEdge, ...]:
    """Measure the relationship graph of one generated organ."""
    edges: list[MeasuredEdge] = []

    for subject, obj, axis in SPATIAL_PAIRS:
        if subject not in statistics or obj not in statistics:
            continue
        first = statistics[subject]["centroid"][axis]
        second = statistics[obj]["centroid"][axis]
        positive, negative = _AXIS_RELATIONS[axis]
        edges.append(MeasuredEdge(subject, positive if first > second else negative, obj))

    clouds: dict[str, FloatArray] = {}
    for entity_id in {name for pair in ADJACENCY_PAIRS for name in pair}:
        if entity_id not in statistics:
            continue
        points, owned = field.entity_points(entity_id, cloud_points, rng)
        clouds[entity_id] = points[owned]

    for subject, obj in ADJACENCY_PAIRS:
        if subject not in clouds or obj not in clouds:
            continue
        if _min_distance(clouds[subject], clouds[obj]) <= _ADJACENCY_TOLERANCE:
            edges.append(MeasuredEdge(subject, "adjacent_to", obj))

    # Functional edges follow the construction: each artery leaves through its own valve,
    # and the transposed variant swaps which ventricle that is.
    arterial = dict(field._arterial_pairs())  # noqa: SLF001 - the field owns this fact
    for valve, artery in (
        ("heart.aortic_valve", "heart.aorta"),
        ("heart.pulmonary_valve", "heart.pulmonary_trunk"),
    ):
        chamber = arterial[valve]
        edges.append(MeasuredEdge(chamber, "opens_into", valve))
        edges.append(MeasuredEdge(valve, "opens_into", artery))
    for chamber, valve, ventricle in (
        ("heart.left_atrium", "heart.mitral_valve", "heart.left_ventricle"),
        ("heart.right_atrium", "heart.tricuspid_valve", "heart.right_ventricle"),
    ):
        edges.append(MeasuredEdge(chamber, "opens_into", valve))
        edges.append(MeasuredEdge(valve, "opens_into", ventricle))
    for vessel, chamber in (
        ("heart.superior_vena_cava", "heart.right_atrium"),
        ("heart.inferior_vena_cava", "heart.right_atrium"),
        ("heart.pulmonary_veins", "heart.left_atrium"),
    ):
        edges.append(MeasuredEdge(chamber, "receives_from", vessel))
    edges.append(MeasuredEdge("heart.pulmonary_trunk", "exits_into", "heart.pulmonary_arteries"))
    return tuple(edges)


def relation_signature(edges: Sequence[MeasuredEdge]) -> str:
    """A stable signature of a relation set, for counting how many distinct graphs exist."""
    return "|".join(sorted(edge.key() for edge in edges))
