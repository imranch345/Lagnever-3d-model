"""Tier-0 synthetic heart corpus generator.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED. NOT CLINICAL. NOT FOR DIAGNOSIS.

Every renderable entity of Heart Ontology v0.1 is given one analytic primitive whose
position and size are derived from a small set of scene parameters. The parameters
are randomised per scene within ordered ranges, and a scene is **rejected and
resampled** if it violates the spatial relationships the ontology declares. So the
corpus has exact ground truth for identity, ownership, frames, geometry and the
relationships the first experiment measures.

What this corpus is for
-----------------------

Testing whether a structured representation gives better part control, better
relationship consistency and better edit locality than an appearance-driven one.
It is a controlled testbed with known structure.

What it is not
--------------

Anatomy. An ellipsoid is not a ventricle, the proportions are invented, and no
clinician has looked at any of it. No result measured here may be presented as
anatomical or medical accuracy.

Layout conventions: ``+x`` anatomical left, ``+y`` superior, ``+z`` anterior, the
organ inside the unit cube.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from awr.ontology import AnatomyOntology
from awr.relationships import GraphKind
from awr.schema import AnatomyType
from datasets.synthetic.primitives import (
    Capsule,
    Ellipsoid,
    Primitive,
    Shell,
    build_primitive,
)

__all__ = [
    "DATA_LABEL",
    "OWNERSHIP_PRIORITY",
    "REQUIRED_SPATIAL_RELATIONS",
    "HeartParameters",
    "SceneSpec",
    "sample_parameters",
    "build_primitives",
    "build_scene",
    "check_relations",
]

FloatArray = npt.NDArray[np.float64]

DATA_LABEL = "SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED"
"""Stamped into every scene and every manifest."""


OWNERSHIP_PRIORITY: Mapping[AnatomyType, int] = {
    AnatomyType.CONDUCTION_NODE: 0,
    AnatomyType.SURFACE_FEATURE: 0,
    AnatomyType.CONNECTIVE_STRUCTURE: 1,
    AnatomyType.MUSCLE: 1,
    AnatomyType.CONDUCTION_PATHWAY: 2,
    AnatomyType.VALVE: 2,
    AnatomyType.SEPTUM: 3,
    AnatomyType.VESSEL: 3,
    AnatomyType.VASCULAR_SYSTEM: 4,
    AnatomyType.CONDUCTION_SYSTEM: 4,
    AnatomyType.CHAMBER: 5,
    AnatomyType.WALL_LAYER: 6,
    AnatomyType.ORGAN: 7,
}
"""Which entity owns a point when primitives overlap. Smaller structures win.

Overlap is unavoidable: wall layers enclose chambers, papillary muscles sit inside
ventricles. Part ownership has to be a single label per point, so the tie-break is
declared here rather than left to whichever primitive happens to be tested first.
"""


REQUIRED_SPATIAL_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("heart.left_atrium", "left_of", "heart.right_atrium"),
    ("heart.left_ventricle", "left_of", "heart.right_ventricle"),
    ("heart.right_atrium", "superior_to", "heart.right_ventricle"),
    ("heart.left_atrium", "superior_to", "heart.left_ventricle"),
    ("heart.right_ventricle", "anterior_to", "heart.left_ventricle"),
    ("heart.left_atrium", "posterior_to", "heart.right_atrium"),
    ("heart.pulmonary_trunk", "anterior_to", "heart.aorta"),
    ("heart.aorta", "right_of", "heart.pulmonary_trunk"),
    ("heart.superior_vena_cava", "superior_to", "heart.right_atrium"),
    ("heart.inferior_vena_cava", "inferior_to", "heart.right_atrium"),
    ("heart.sinoatrial_node", "superior_to", "heart.atrioventricular_node"),
    ("heart.left_ventricle", "adjacent_to", "heart.interventricular_septum"),
    ("heart.right_ventricle", "adjacent_to", "heart.interventricular_septum"),
    ("heart.left_atrium", "adjacent_to", "heart.interatrial_septum"),
    ("heart.right_atrium", "adjacent_to", "heart.interatrial_septum"),
    ("heart.tricuspid_valve", "adjacent_to", "heart.right_atrium"),
    ("heart.tricuspid_valve", "adjacent_to", "heart.right_ventricle"),
    ("heart.mitral_valve", "adjacent_to", "heart.left_atrium"),
    ("heart.mitral_valve", "adjacent_to", "heart.left_ventricle"),
    ("heart.pulmonary_valve", "adjacent_to", "heart.right_ventricle"),
    ("heart.pulmonary_valve", "adjacent_to", "heart.pulmonary_trunk"),
    ("heart.aortic_valve", "adjacent_to", "heart.left_ventricle"),
    ("heart.aortic_valve", "adjacent_to", "heart.aorta"),
    ("heart.pericardium", "surrounds", "heart"),
    ("heart.epicardium", "surrounds", "heart.myocardium"),
    ("heart.myocardium", "surrounds", "heart.endocardium"),
)
"""Relationships the generator commits to satisfying, and the experiment measures.

The ontology declares more than this. Some are not checkable on a corpus where one
primitive stands for a bilateral structure: the single ``papillary_muscles`` entity
cannot be inside both ventricles at once. Those are recorded as unverified with a
reason rather than quietly counted as satisfied.
"""

_ADJACENCY_TOLERANCE = 0.075
"""Surface separation below which two primitives count as adjacent, in scene units."""


@dataclass(frozen=True, slots=True)
class HeartParameters:
    """Scene parameters. A family sets the archetype; jitter varies within it."""

    family_id: int
    global_scale: float
    lateral_separation: float
    """Factor on the chamber radius that sets how far the left and right sides sit apart."""
    atrioventricular_gap: float
    """Factor on the summed radii that sets the atrium-to-ventricle spacing."""
    anteroposterior_offset: float
    lv_radii: tuple[float, float, float]
    rv_radii: tuple[float, float, float]
    la_radii: tuple[float, float, float]
    ra_radii: tuple[float, float, float]
    wall_thickness: float
    vessel_radius: float
    valve_radius: float
    valve_thickness: float
    envelope_radii: tuple[float, float, float]
    orientation: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "family_id": self.family_id,
            "global_scale": self.global_scale,
            "lateral_separation": self.lateral_separation,
            "atrioventricular_gap": self.atrioventricular_gap,
            "anteroposterior_offset": self.anteroposterior_offset,
            "lv_radii": list(self.lv_radii),
            "rv_radii": list(self.rv_radii),
            "la_radii": list(self.la_radii),
            "ra_radii": list(self.ra_radii),
            "wall_thickness": self.wall_thickness,
            "vessel_radius": self.vessel_radius,
            "valve_radius": self.valve_radius,
            "valve_thickness": self.valve_thickness,
            "envelope_radii": list(self.envelope_radii),
            "orientation": list(self.orientation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HeartParameters:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            family_id=int(payload["family_id"]),
            global_scale=float(payload["global_scale"]),
            lateral_separation=float(payload["lateral_separation"]),
            atrioventricular_gap=float(payload["atrioventricular_gap"]),
            anteroposterior_offset=float(payload["anteroposterior_offset"]),
            lv_radii=_triple(payload["lv_radii"]),
            rv_radii=_triple(payload["rv_radii"]),
            la_radii=_triple(payload["la_radii"]),
            ra_radii=_triple(payload["ra_radii"]),
            wall_thickness=float(payload["wall_thickness"]),
            vessel_radius=float(payload["vessel_radius"]),
            valve_radius=float(payload["valve_radius"]),
            valve_thickness=float(payload["valve_thickness"]),
            envelope_radii=_triple(payload["envelope_radii"]),
            orientation=_triple(payload["orientation"]),
        )


def _triple(values: Any) -> tuple[float, float, float]:
    """Coerce a serialised sequence back into a three-component tuple."""
    first, second, third = (float(value) for value in values)
    return (first, second, third)


def _jitter(rng: np.random.Generator, base: float, spread: float) -> float:
    return float(base * (1.0 + rng.uniform(-spread, spread)))


def _jitter_triple(
    rng: np.random.Generator, base: tuple[float, float, float], spread: float
) -> tuple[float, float, float]:
    return (
        _jitter(rng, base[0], spread),
        _jitter(rng, base[1], spread),
        _jitter(rng, base[2], spread),
    )


def sample_parameters(rng: np.random.Generator, family_id: int) -> HeartParameters:
    """Sample scene parameters for a family.

    The family seeds an archetype (its own proportions) and the scene then jitters
    within it, so a split by family is a genuine generalisation test rather than a
    test on near-duplicates.
    """
    family_rng = np.random.default_rng(1_000_003 + family_id)
    archetype_scale = _jitter(family_rng, 1.0, 0.12)
    archetype_lv = _jitter_triple(family_rng, (0.22, 0.30, 0.22), 0.12)
    archetype_rv = _jitter_triple(family_rng, (0.20, 0.28, 0.20), 0.12)
    archetype_la = _jitter_triple(family_rng, (0.16, 0.15, 0.16), 0.12)
    archetype_ra = _jitter_triple(family_rng, (0.17, 0.16, 0.17), 0.12)

    return HeartParameters(
        family_id=family_id,
        global_scale=_jitter(rng, archetype_scale, 0.05),
        lateral_separation=_jitter(rng, 0.98, 0.08),
        atrioventricular_gap=_jitter(rng, 0.96, 0.05),
        anteroposterior_offset=_jitter(rng, 0.15, 0.15),
        lv_radii=_jitter_triple(rng, archetype_lv, 0.06),
        rv_radii=_jitter_triple(rng, archetype_rv, 0.06),
        la_radii=_jitter_triple(rng, archetype_la, 0.06),
        ra_radii=_jitter_triple(rng, archetype_ra, 0.06),
        wall_thickness=_jitter(rng, 0.045, 0.20),
        vessel_radius=_jitter(rng, 0.058, 0.15),
        valve_radius=_jitter(rng, 0.085, 0.12),
        valve_thickness=_jitter(rng, 0.020, 0.20),
        envelope_radii=_jitter_triple(rng, (0.52, 0.62, 0.42), 0.06),
        orientation=(
            float(rng.uniform(-0.10, 0.10)),
            float(rng.uniform(-0.08, 0.08)),
            float(rng.uniform(-0.08, 0.08)),
        ),
    )


def _disc(centre: FloatArray, radius: float, thickness: float, axis: int) -> Ellipsoid:
    radii = np.full(3, radius, dtype=np.float64)
    radii[axis] = thickness
    return Ellipsoid(centre_point=centre, radii=radii)


def build_primitives(params: HeartParameters) -> dict[str, Primitive]:
    """Build one primitive per renderable entity of Heart Ontology v0.1.

    Anchors are the four chamber centres; everything else is placed relative to them,
    which is what lets the ontology's spatial relationships hold by construction
    instead of by luck.
    """
    scale = params.global_scale
    separation = params.lateral_separation
    stacking = params.atrioventricular_gap
    ap = params.anteroposterior_offset * scale

    lv_r = np.array(params.lv_radii) * scale
    rv_r = np.array(params.rv_radii) * scale
    la_r = np.array(params.la_radii) * scale
    ra_r = np.array(params.ra_radii) * scale

    # Anchors are derived from the chamber radii rather than sampled independently, so
    # the ontology's spatial relationships hold by construction. Rejection sampling was
    # the alternative and it could not satisfy valve adjacency for thin-atrium families.
    lv_c = np.array([lv_r[0] * separation, -lv_r[1] * 0.55, -ap * 0.33])
    rv_c = np.array([-rv_r[0] * separation, -rv_r[1] * 0.55, ap * 0.65])
    la_c = np.array(
        [
            la_r[0] * separation * 0.95,
            lv_c[1] + (lv_r[1] + la_r[1]) * stacking,
            -ap * 0.80,
        ]
    )
    ra_c = np.array(
        [
            -ra_r[0] * separation,
            rv_c[1] + (rv_r[1] + ra_r[1]) * stacking,
            ap * 0.15,
        ]
    )
    envelope = np.array(params.envelope_radii) * scale
    reach = np.maximum.reduce(
        [
            np.abs(lv_c) + lv_r,
            np.abs(rv_c) + rv_r,
            np.abs(la_c) + la_r,
            np.abs(ra_c) + ra_r,
        ]
    )
    envelope = np.maximum(envelope, reach * 1.12)
    vessel_r = params.vessel_radius * scale
    valve_r = params.valve_radius * scale
    valve_t = params.valve_thickness * scale

    primitives: dict[str, Primitive] = {}

    # --- organ envelope and wall layers ------------------------------------
    primitives["heart"] = Ellipsoid(centre_point=np.zeros(3), radii=envelope)
    primitives["heart.endocardium"] = Shell(
        centre_point=np.zeros(3),
        outer_radii=envelope * 0.88,
        thickness=params.wall_thickness * 0.4 * scale,
    )
    primitives["heart.myocardium"] = Shell(
        centre_point=np.zeros(3),
        outer_radii=envelope * 0.94,
        thickness=params.wall_thickness * scale,
    )
    primitives["heart.epicardium"] = Shell(
        centre_point=np.zeros(3),
        outer_radii=envelope * 0.985,
        thickness=params.wall_thickness * 0.35 * scale,
    )
    primitives["heart.pericardium"] = Shell(
        centre_point=np.zeros(3),
        outer_radii=envelope * 1.07,
        thickness=params.wall_thickness * 0.5 * scale,
    )

    # --- chambers -----------------------------------------------------------
    primitives["heart.left_ventricle"] = Ellipsoid(centre_point=lv_c, radii=lv_r)
    primitives["heart.right_ventricle"] = Ellipsoid(centre_point=rv_c, radii=rv_r)
    primitives["heart.left_atrium"] = Ellipsoid(centre_point=la_c, radii=la_r)
    primitives["heart.right_atrium"] = Ellipsoid(centre_point=ra_c, radii=ra_r)

    # --- septa: thin partitions on the midline -----------------------------
    ivs_c = (lv_c + rv_c) / 2.0
    ias_c = (la_c + ra_c) / 2.0
    primitives["heart.interventricular_septum"] = Ellipsoid(
        centre_point=ivs_c,
        radii=np.array([params.wall_thickness * 0.55 * scale, lv_r[1] * 0.95, lv_r[2] * 0.9]),
    )
    primitives["heart.interatrial_septum"] = Ellipsoid(
        centre_point=ias_c,
        radii=np.array([params.wall_thickness * 0.45 * scale, la_r[1] * 0.95, la_r[2] * 0.9]),
    )

    # --- atrioventricular valves: on the interface between the two surfaces -
    def _interface(
        first_centre: FloatArray,
        first_radii: FloatArray,
        second_centre: FloatArray,
        second_radii: FloatArray,
    ) -> FloatArray:
        """Midpoint of the gap between two ellipsoid surfaces along their axis.

        Using the midpoint of the two centres instead leaves the valve far from a
        small atrium's surface, which breaks the adjacency the ontology declares.
        """
        axis = second_centre - first_centre
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            return np.asarray((first_centre + second_centre) / 2.0, dtype=np.float64)
        direction = axis / length
        first_reach = float(np.linalg.norm(direction * first_radii))
        second_reach = float(np.linalg.norm(direction * second_radii))
        near = first_centre + direction * min(first_reach, length)
        far = second_centre - direction * min(second_reach, length)
        return np.asarray((near + far) / 2.0, dtype=np.float64)

    tricuspid_c = _interface(ra_c, ra_r, rv_c, rv_r)
    mitral_c = _interface(la_c, la_r, lv_c, lv_r)
    primitives["heart.tricuspid_valve"] = _disc(tricuspid_c, valve_r, valve_t, axis=1)
    primitives["heart.mitral_valve"] = _disc(mitral_c, valve_r, valve_t, axis=1)

    # --- semilunar valves at the outflow tracts ----------------------------
    pulmonary_valve_c = rv_c + np.array([rv_r[0] * 0.45, rv_r[1] * 0.92, rv_r[2] * 0.2])
    aortic_valve_c = lv_c + np.array([-lv_r[0] * 0.5, lv_r[1] * 0.95, lv_r[2] * 0.1])
    primitives["heart.pulmonary_valve"] = _disc(pulmonary_valve_c, valve_r * 0.85, valve_t, axis=1)
    primitives["heart.aortic_valve"] = _disc(aortic_valve_c, valve_r * 0.85, valve_t, axis=1)

    # --- great vessels. The arteries cross, so the aorta ends to the right. -
    primitives["heart.superior_vena_cava"] = Capsule(
        start=ra_c + np.array([0.0, ra_r[1] * 0.6, 0.0]),
        end=ra_c + np.array([0.0, ra_r[1] + 0.34 * scale, -0.02 * scale]),
        radius=vessel_r,
    )
    primitives["heart.inferior_vena_cava"] = Capsule(
        start=ra_c - np.array([0.0, ra_r[1] * 0.6, 0.0]),
        end=ra_c - np.array([0.0, ra_r[1] + 0.32 * scale, 0.02 * scale]),
        radius=vessel_r * 1.05,
    )
    trunk_start = pulmonary_valve_c + np.array([0.0, valve_t + 0.01 * scale, 0.0])
    trunk_end = np.array([0.14, 0.58, 0.24]) * scale
    primitives["heart.pulmonary_trunk"] = Capsule(
        start=trunk_start, end=trunk_end, radius=vessel_r * 1.15
    )
    primitives["heart.pulmonary_arteries"] = Capsule(
        start=trunk_end + np.array([-0.20, 0.03, -0.01]) * scale,
        end=trunk_end + np.array([0.18, 0.04, -0.01]) * scale,
        radius=vessel_r * 0.85,
    )
    primitives["heart.pulmonary_veins"] = Capsule(
        start=la_c + np.array([la_r[0] * 0.7, 0.02 * scale, -la_r[2] * 0.5]),
        end=la_c + np.array([la_r[0] + 0.26 * scale, 0.06 * scale, -la_r[2] - 0.16 * scale]),
        radius=vessel_r * 0.9,
    )
    aorta_start = aortic_valve_c + np.array([0.0, valve_t + 0.01 * scale, 0.0])
    aorta_end = np.array([-0.16, 0.60, -0.16]) * scale
    primitives["heart.aorta"] = Capsule(
        start=aorta_start, end=aorta_end, radius=vessel_r * 1.2
    )

    # --- internal structures ------------------------------------------------
    papillary_c = lv_c + np.array([0.0, -lv_r[1] * 0.45, 0.0])
    primitives["heart.papillary_muscles"] = Ellipsoid(
        centre_point=papillary_c,
        radii=np.array([lv_r[0] * 0.3, lv_r[1] * 0.3, lv_r[2] * 0.3]),
    )
    primitives["heart.chordae_tendineae"] = Capsule(
        start=papillary_c + np.array([0.0, lv_r[1] * 0.25, 0.0]),
        end=mitral_c - np.array([0.0, valve_t, 0.0]),
        radius=0.012 * scale,
    )
    primitives["heart.trabeculae_carneae"] = Ellipsoid(
        centre_point=rv_c + np.array([0.0, -rv_r[1] * 0.4, rv_r[2] * 0.25]),
        radii=np.array([rv_r[0] * 0.34, rv_r[1] * 0.3, rv_r[2] * 0.28]),
    )
    primitives["heart.moderator_band"] = Capsule(
        start=ivs_c + np.array([-params.wall_thickness * scale, -lv_r[1] * 0.3, 0.0]),
        end=rv_c + np.array([-rv_r[0] * 0.2, -rv_r[1] * 0.35, rv_r[2] * 0.2]),
        radius=0.015 * scale,
    )
    primitives["heart.fossa_ovalis"] = _disc(
        ias_c + np.array([0.0, la_r[1] * 0.1, la_r[2] * 0.15]),
        radius=la_r[1] * 0.22,
        thickness=params.wall_thickness * 0.2 * scale,
        axis=0,
    )

    # --- coronary circulation ----------------------------------------------
    groove_y = (lv_c[1] + la_c[1]) / 2.0
    primitives["heart.coronary_circulation"] = Capsule(
        start=np.array([-envelope[0] * 0.82, groove_y, envelope[2] * 0.5]),
        end=np.array([envelope[0] * 0.82, groove_y, envelope[2] * 0.5]),
        radius=0.03 * scale,
    )
    primitives["heart.left_coronary_artery"] = Capsule(
        start=aorta_start + np.array([0.02 * scale, 0.01 * scale, 0.0]),
        end=np.array([envelope[0] * 0.75, groove_y - 0.05 * scale, envelope[2] * 0.45]),
        radius=0.018 * scale,
    )
    primitives["heart.right_coronary_artery"] = Capsule(
        start=aorta_start - np.array([0.02 * scale, -0.01 * scale, 0.0]),
        end=np.array([-envelope[0] * 0.75, groove_y - 0.05 * scale, envelope[2] * 0.45]),
        radius=0.018 * scale,
    )
    primitives["heart.coronary_sinus"] = Capsule(
        start=np.array([envelope[0] * 0.45, groove_y - 0.02 * scale, -envelope[2] * 0.5]),
        end=ra_c + np.array([0.0, -ra_r[1] * 0.55, -ra_r[2] * 0.4]),
        radius=0.025 * scale,
    )

    # --- conduction system --------------------------------------------------
    sa_c = ra_c + np.array([0.0, ra_r[1] * 0.62, ra_r[2] * 0.35])
    av_c = ias_c + np.array([0.0, -la_r[1] * 0.75, la_r[2] * 0.1])
    primitives["heart.sinoatrial_node"] = Ellipsoid(
        centre_point=sa_c, radii=np.full(3, 0.026 * scale)
    )
    primitives["heart.atrioventricular_node"] = Ellipsoid(
        centre_point=av_c, radii=np.full(3, 0.022 * scale)
    )
    his_end = ivs_c + np.array([0.0, lv_r[1] * 0.45, 0.0])
    primitives["heart.bundle_of_his"] = Capsule(start=av_c, end=his_end, radius=0.013 * scale)
    branches_end = ivs_c + np.array([0.0, -lv_r[1] * 0.6, 0.0])
    primitives["heart.bundle_branches"] = Capsule(
        start=his_end, end=branches_end, radius=0.012 * scale
    )
    primitives["heart.purkinje_fibers"] = Capsule(
        start=branches_end,
        end=lv_c + np.array([lv_r[0] * 0.7, -lv_r[1] * 0.7, 0.0]),
        radius=0.010 * scale,
    )
    primitives["heart.cardiac_conduction_system"] = Capsule(
        start=sa_c, end=branches_end, radius=0.010 * scale
    )
    return primitives


def _centroid_relation(relation: str, subject: FloatArray, obj: FloatArray) -> bool:
    axis_sign = {
        "left_of": (0, 1.0),
        "right_of": (0, -1.0),
        "superior_to": (1, 1.0),
        "inferior_to": (1, -1.0),
        "anterior_to": (2, 1.0),
        "posterior_to": (2, -1.0),
    }
    axis, sign = axis_sign[relation]
    return bool(sign * (subject[axis] - obj[axis]) > 1e-4)


def check_relations(
    primitives: Mapping[str, Primitive],
    relations: Sequence[tuple[str, str, str]] = REQUIRED_SPATIAL_RELATIONS,
    *,
    rng: np.random.Generator | None = None,
) -> dict[tuple[str, str, str], bool]:
    """Check each relation against the built geometry.

    Centroid comparisons for the directional relations, sampled surface distance for
    adjacency, and volume containment for ``surrounds``.
    """
    generator = rng or np.random.default_rng(0)
    results: dict[tuple[str, str, str], bool] = {}
    for subject, relation, obj in relations:
        if subject not in primitives or obj not in primitives:
            results[(subject, relation, obj)] = False
            continue
        first, second = primitives[subject], primitives[obj]
        if relation in {
            "left_of", "right_of", "superior_to", "inferior_to", "anterior_to", "posterior_to"
        }:
            ok = _centroid_relation(relation, first.centre, second.centre)
        elif relation == "adjacent_to":
            ok = first.distance_to(second, generator) <= _ADJACENCY_TOLERANCE
        elif relation == "surrounds":
            ok = bool(np.all(first.extent >= second.extent * 0.995))
        elif relation == "inside":
            ok = bool(second.occupancy(first.centre.reshape(1, 3))[0])
        else:
            ok = False
        results[(subject, relation, obj)] = ok
    return results


@dataclass(slots=True)
class SceneSpec:
    """One generated scene: parameters, primitives, frames and verified relations."""

    scene_id: str
    family_id: int
    seed: int
    ontology_id: str
    ontology_version: str
    active_lod: int
    parameters: HeartParameters
    primitives: dict[str, Primitive]
    ownership_priority: dict[str, int]
    verified_relations: dict[str, bool]
    unverified_relations: dict[str, str] = field(default_factory=dict)
    data_label: str = DATA_LABEL

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Entities with geometry in this scene, in declaration order."""
        return tuple(self.primitives)

    def visible_entity_ids(
        self, ontology: AnatomyOntology, lod: int | None = None
    ) -> tuple[str, ...]:
        """Entities visible by default at a level of detail."""
        level = self.active_lod if lod is None else lod
        out: list[str] = []
        for entity_id in self.primitives:
            policy = ontology.get(entity_id).lod_policy
            if policy.includes(level):
                out.append(entity_id)
        return tuple(out)

    def frames(self) -> dict[str, list[float]]:
        """Canonical frame per entity: translation, log-scale, 6D rotation."""
        return {
            entity_id: primitive.frame().tolist()
            for entity_id, primitive in self.primitives.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "scene_id": self.scene_id,
            "family_id": self.family_id,
            "seed": self.seed,
            "data_label": self.data_label,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "active_lod": self.active_lod,
            "parameters": self.parameters.to_dict(),
            "geometry": {
                entity_id: primitive.to_dict()
                for entity_id, primitive in self.primitives.items()
            },
            "frames": self.frames(),
            "ownership_priority": dict(self.ownership_priority),
            "relationships": {
                "source": f"ontology:{self.ontology_id}@{self.ontology_version}",
                "verified": dict(self.verified_relations),
                "unverified": dict(self.unverified_relations),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SceneSpec:
        """Rebuild from :meth:`to_dict` output."""
        relationships = payload.get("relationships", {})
        return cls(
            scene_id=str(payload["scene_id"]),
            family_id=int(payload["family_id"]),
            seed=int(payload["seed"]),
            ontology_id=str(payload["ontology_id"]),
            ontology_version=str(payload["ontology_version"]),
            active_lod=int(payload["active_lod"]),
            parameters=HeartParameters.from_dict(payload["parameters"]),
            primitives={
                entity_id: build_primitive(spec)
                for entity_id, spec in payload["geometry"].items()
            },
            ownership_priority={str(k): int(v) for k, v in payload["ownership_priority"].items()},
            verified_relations={
                str(k): bool(v) for k, v in relationships.get("verified", {}).items()
            },
            unverified_relations={
                str(k): str(v) for k, v in relationships.get("unverified", {}).items()
            },
            data_label=str(payload.get("data_label", DATA_LABEL)),
        )


def _unverifiable_relations(ontology: AnatomyOntology) -> dict[str, str]:
    """Ontology relations the tier-0 corpus cannot verify, with the reason."""
    required = {f"{s}|{r}|{o}" for s, r, o in REQUIRED_SPATIAL_RELATIONS}
    out: dict[str, str] = {}
    spatial = ontology.relationships.graph(GraphKind.SPATIAL)
    for edge in spatial.edges:
        key = f"{edge.subject}|{edge.relation}|{edge.object}"
        if key in required:
            continue
        out[key] = (
            "not enforced by the tier-0 generator: one primitive stands for a merged or "
            "bilateral structure, so the relation is not decidable on this corpus"
        )
    return out


def build_scene(
    ontology: AnatomyOntology,
    *,
    scene_index: int,
    family_id: int,
    lod: int,
    max_attempts: int = 40,
) -> SceneSpec:
    """Generate one scene, resampling until the required relationships hold.

    Raises:
        RuntimeError: if no sample satisfies the constraints within ``max_attempts``.
            A generator that cannot satisfy its own ontology is a bug worth failing on,
            not a reason to emit an invalid scene.

    """
    for attempt in range(max_attempts):
        seed = scene_index * 1_000 + attempt
        rng = np.random.default_rng(seed)
        params = sample_parameters(rng, family_id)
        primitives = build_primitives(params)
        checks = check_relations(primitives, rng=np.random.default_rng(seed + 7))
        if all(checks.values()):
            priority = {
                entity_id: OWNERSHIP_PRIORITY[ontology.get(entity_id).anatomy_type]
                for entity_id in primitives
            }
            return SceneSpec(
                scene_id=f"tier0-{scene_index:06d}",
                family_id=family_id,
                seed=seed,
                ontology_id=ontology.ontology_id,
                ontology_version=ontology.version,
                active_lod=lod,
                parameters=params,
                primitives=primitives,
                ownership_priority=priority,
                verified_relations={
                    f"{s}|{r}|{o}": ok for (s, r, o), ok in checks.items()
                },
                unverified_relations=_unverifiable_relations(ontology),
            )
    failures = [
        f"{s} {r} {o}"
        for (s, r, o), ok in check_relations(build_primitives(params)).items()
        if not ok
    ]
    raise RuntimeError(
        f"Could not satisfy the required relationships in {max_attempts} attempts. "
        f"Failing relations: {failures}. Fix the generator rather than relaxing the check."
    )


def rotate_scene(primitives: Mapping[str, Primitive], orientation: Sequence[float]) -> None:
    """Placeholder for whole-scene rotation.

    Not applied: rotating the scene would break the axis-aligned relationship checks
    that give this corpus its exact ground truth. Orientation jitter is stored in the
    parameters and left for the data-augmentation step in a later experiment, where
    the relationship checks would move into the rotated frame with it.
    """
    raise NotImplementedError(
        "Whole-scene rotation is deliberately not applied in tier-0; see the docstring."
    )
