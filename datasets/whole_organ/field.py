"""The whole-organ field: built as one object, then carved and labelled.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED.

Construction order matters and is the point of this module:

1. an **organ envelope** from global parameters, tapered toward the apex;
2. four **cavities** carved inside it, positioned in the organ's own frame;
3. **valve annuli** derived as the regions between adjacent cavities;
4. **vessels** attached at openings computed from the cavities;
5. **septa** derived as the myocardium that lies between two cavities;
6. **wall layers** derived from distance to the cavity and outer surfaces;
7. **ownership** assigned by the rule fixed in ``docs/step7/01_experimental_plan.md``.

Nothing is placed independently, so an entity-factorised representation has to discover
the parts of a coherent object rather than being handed them. Septa and valves in
particular have no existence of their own: they are residuals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from datasets.whole_organ.parameters import OrganParameters, Variant

__all__ = [
    "WHOLE_ORGAN_ENTITIES",
    "CHAMBERS",
    "VALVES",
    "VESSELS",
    "SEPTA",
    "WALL_LAYERS",
    "BACKGROUND",
    "WholeOrganField",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

CHAMBERS: tuple[str, ...] = (
    "heart.right_atrium",
    "heart.right_ventricle",
    "heart.left_atrium",
    "heart.left_ventricle",
)
VALVES: tuple[str, ...] = (
    "heart.tricuspid_valve",
    "heart.pulmonary_valve",
    "heart.mitral_valve",
    "heart.aortic_valve",
)
VESSELS: tuple[str, ...] = (
    "heart.superior_vena_cava",
    "heart.inferior_vena_cava",
    "heart.pulmonary_trunk",
    "heart.pulmonary_arteries",
    "heart.pulmonary_veins",
    "heart.aorta",
)
SEPTA: tuple[str, ...] = ("heart.interatrial_septum", "heart.interventricular_septum")
WALL_LAYERS: tuple[str, ...] = (
    "heart.endocardium",
    "heart.myocardium",
    "heart.epicardium",
    "heart.pericardium",
)

WHOLE_ORGAN_ENTITIES: tuple[str, ...] = (
    *CHAMBERS,
    *VALVES,
    *VESSELS,
    *SEPTA,
    *WALL_LAYERS,
)
"""The 20 entities this corpus instantiates.

Deliberately a subset of Heart Ontology v0.1's 36 renderable entities. Papillary
muscles, trabeculae, chordae, the moderator band, the fossa ovalis, the coronary vessels
and the conduction system are **not** represented: they cannot be derived from a
whole-organ field without placing them independently, which is exactly what this corpus
exists to avoid. Their absence is recorded in every scene rather than hidden.
"""

BACKGROUND = -1


def _normalise(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _surface_reach(direction: FloatArray, radii: FloatArray) -> float:
    """Distance from an ellipsoid's centre to its surface along a unit direction.

    The naive ``norm(direction * radii)`` is wrong for anisotropic radii and can place a
    valve annulus inside the cavity it is meant to sit on, where the ownership rule gives
    it to the chamber and the valve vanishes.
    """
    scaled = direction / np.maximum(radii, 1e-9)
    return float(1.0 / max(float(np.sqrt((scaled**2).sum())), 1e-9))


@dataclass(frozen=True, slots=True)
class _Cavity:
    """One blood pool, positioned in the organ frame."""

    entity_id: str
    centre: FloatArray
    radii: FloatArray

    def normalised_distance(self, local: FloatArray) -> FloatArray:
        """Radial distance in cavity units: 1.0 is the surface."""
        scaled = (local - self.centre) / np.maximum(self.radii, 1e-6)
        return np.asarray(np.sqrt((scaled**2).sum(axis=-1)), dtype=np.float64)

    def surface_gap(self, local: FloatArray) -> FloatArray:
        """Approximate distance from the cavity surface, in scene units."""
        mean_radius = float(np.mean(self.radii))
        return (self.normalised_distance(local) - 1.0) * mean_radius


@dataclass(frozen=True, slots=True)
class _Tube:
    """One vessel, attached at an opening computed from a cavity."""

    entity_id: str
    start: FloatArray
    end: FloatArray
    radius: float
    attached_to: str

    def distance(self, local: FloatArray) -> FloatArray:
        """Distance from the tube axis."""
        axis = self.end - self.start
        length_squared = float(axis @ axis)
        if length_squared < 1e-12:
            return np.asarray(np.linalg.norm(local - self.start, axis=-1), dtype=np.float64)
        t = np.clip(((local - self.start) @ axis) / length_squared, 0.0, 1.0)
        nearest = self.start + t[:, None] * axis
        return np.asarray(np.linalg.norm(local - nearest, axis=-1), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Annulus:
    """A valve: the region between two cavities."""

    entity_id: str
    upper: str
    lower: str
    centre: FloatArray
    axis: FloatArray
    radius: float
    thickness: float

    def occupancy(self, local: FloatArray) -> BoolArray:
        """Points within the annulus disc."""
        offset = local - self.centre
        along = offset @ self.axis
        radial = np.linalg.norm(offset - along[:, None] * self.axis, axis=-1)
        return np.asarray(
            (np.abs(along) <= self.thickness) & (radial <= self.radius), dtype=np.bool_
        )


class WholeOrganField:
    """A heart built as one object, then carved into labelled entities."""

    def __init__(self, parameters: OrganParameters) -> None:
        self.parameters = parameters
        self.rotation = parameters.rotation()
        self.mirror = -1.0 if parameters.variant is Variant.MIRRORED else 1.0
        self.radii = np.asarray(parameters.organ_radii, dtype=np.float64)
        self.cavities = self._build_cavities()
        self.cavity_index = {cavity.entity_id: cavity for cavity in self.cavities}
        self.annuli = self._build_annuli()
        self.tubes = self._build_tubes()
        self.entity_ids = WHOLE_ORGAN_ENTITIES
        self.slot_of = {entity_id: index for index, entity_id in enumerate(self.entity_ids)}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_cavities(self) -> tuple[_Cavity, ...]:
        parameters = self.parameters
        radii = self.radii
        ventricle = np.asarray(parameters.ventricle_cavity) * radii
        atrium = np.asarray(parameters.atrium_cavity) * radii
        side = radii[0] * 0.45 * self.mirror
        drop = -radii[1] * parameters.apex_drop
        lift = radii[1] * parameters.av_plane * 2.0
        depth = radii[2] * parameters.anterior_offset

        min_gap = parameters.septum_band * float(np.mean(radii)) * 0.9
        left_x = side + parameters.septal_offset
        right_x = -side + parameters.septal_offset
        span = abs(left_x - right_x)
        limit = max(0.02, (span - min_gap) / 2.0)
        ventricle = np.array([min(ventricle[0], limit), ventricle[1], ventricle[2]])
        atrium = np.array([min(atrium[0], limit), atrium[1], atrium[2]])

        def cavity(entity_id: str, centre: FloatArray, radii: FloatArray) -> _Cavity:
            return _Cavity(entity_id, centre, radii * parameters.scale_for(entity_id))

        return (
            cavity(
                "heart.left_ventricle",
                np.array([side + parameters.septal_offset, drop, -depth * 0.35]),
                ventricle,
            ),
            cavity(
                "heart.right_ventricle",
                np.array([-side + parameters.septal_offset, drop * 0.94, depth * 0.72]),
                ventricle * np.array([0.92, 0.94, 0.92]),
            ),
            cavity(
                "heart.left_atrium",
                np.array([side * 0.86 + parameters.septal_offset, lift, -depth * 0.80]),
                atrium,
            ),
            cavity(
                "heart.right_atrium",
                np.array([-side * 0.94 + parameters.septal_offset, lift * 0.94, depth * 0.22]),
                atrium * np.array([1.05, 1.02, 1.05]),
            ),
        )

    def _build_annuli(self) -> tuple[_Annulus, ...]:
        parameters = self.parameters
        pairs = (
            ("heart.mitral_valve", "heart.left_atrium", "heart.left_ventricle"),
            ("heart.tricuspid_valve", "heart.right_atrium", "heart.right_ventricle"),
        )
        annuli: list[_Annulus] = []
        for entity_id, upper, lower in pairs:
            top = self.cavity_index[upper]
            bottom = self.cavity_index[lower]
            axis = _normalise(bottom.centre - top.centre)
            reach_top = _surface_reach(axis, top.radii)
            reach_bottom = _surface_reach(axis, bottom.radii)
            near = top.centre + axis * reach_top
            far = bottom.centre - axis * reach_bottom
            centre = (near + far) / 2.0
            radius = (
                float(np.mean(np.minimum(top.radii, bottom.radii)))
                * parameters.valve_radius_fraction
                * 1.6
                * parameters.scale_for(entity_id)
            )
            annuli.append(
                _Annulus(
                    entity_id=entity_id,
                    upper=upper,
                    lower=lower,
                    centre=centre,
                    axis=axis,
                    radius=radius,
                    thickness=parameters.valve_thickness
                    * float(np.mean(self.radii))
                    * parameters.scale_for(entity_id),
                )
            )

        for entity_id, chamber in self._arterial_pairs():
            cavity = self.cavity_index[chamber]
            # Aim up and toward the midline. Aiming up and posterior lands the annulus
            # inside the atrium above, where the ownership rule gives it to the chamber.
            medial = -float(np.sign(cavity.centre[0])) or 1.0
            lean = 0.55 if "pulmonary" in entity_id else 0.25
            axis = _normalise(np.array([medial * 0.5, 1.0, lean]))
            reach = _surface_reach(axis, cavity.radii)
            thickness = parameters.valve_thickness * float(np.mean(self.radii)) * 0.8
            # Sit the annulus just outside the cavity surface. Straddling it would put
            # half the disc inside the blood pool, which the ownership rule gives to the
            # chamber, and the valve would lose most of its volume.
            centre = cavity.centre + axis * (reach + thickness)
            annuli.append(
                _Annulus(
                    entity_id=entity_id.replace("trunk", "valve")
                    if entity_id == "heart.pulmonary_trunk"
                    else entity_id,
                    upper=chamber,
                    lower=chamber,
                    centre=centre,
                    axis=axis,
                    radius=float(np.mean(cavity.radii)) * parameters.valve_radius_fraction,
                    thickness=thickness * parameters.scale_for(entity_id),
                )
            )
        return tuple(annuli)

    def _arterial_pairs(self) -> tuple[tuple[str, str], ...]:
        """Which ventricle each great artery leaves.

        The transposed variant swaps them, which is the whole point: the entity set is
        unchanged and only the connection differs.
        """
        if self.parameters.variant is Variant.TRANSPOSED:
            return (
                ("heart.pulmonary_valve", "heart.left_ventricle"),
                ("heart.aortic_valve", "heart.right_ventricle"),
            )
        return (
            ("heart.pulmonary_valve", "heart.right_ventricle"),
            ("heart.aortic_valve", "heart.left_ventricle"),
        )

    def _build_tubes(self) -> tuple[_Tube, ...]:
        parameters = self.parameters
        radii = self.radii
        vessel_radius = float(np.mean(radii)) * parameters.vessel_radius_fraction * 0.45
        length = float(np.mean(radii)) * parameters.vessel_length_fraction

        arterial = dict(self._arterial_pairs())
        tubes: list[_Tube] = []

        for entity_id, valve in (
            ("heart.aorta", "heart.aortic_valve"),
            ("heart.pulmonary_trunk", "heart.pulmonary_valve"),
        ):
            annulus = next(a for a in self.annuli if a.entity_id == valve)
            chamber = arterial[valve]
            # The artery leaves through its own valve, so attachment is by construction.
            start = annulus.centre + annulus.axis * annulus.thickness * 1.2
            tubes.append(
                _Tube(
                    entity_id,
                    start,
                    start + annulus.axis * length,
                    vessel_radius * 1.15 * parameters.scale_for(entity_id),
                    chamber,
                )
            )

        right_atrium = self.cavity_index["heart.right_atrium"]
        left_atrium = self.cavity_index["heart.left_atrium"]
        for entity_id, cavity, direction in (
            ("heart.superior_vena_cava", right_atrium, np.array([0.05 * self.mirror, 1.0, -0.1])),
            ("heart.inferior_vena_cava", right_atrium, np.array([0.02 * self.mirror, -1.0, -0.1])),
            ("heart.pulmonary_veins", left_atrium, np.array([0.55 * self.mirror, 0.15, -1.0])),
        ):
            unit = _normalise(direction)
            start = cavity.centre + unit * _surface_reach(unit, cavity.radii)
            tubes.append(
                _Tube(
                    entity_id,
                    start,
                    start + unit * length,
                    vessel_radius * parameters.scale_for(entity_id),
                    cavity.entity_id,
                )
            )

        trunk = next(tube for tube in tubes if tube.entity_id == "heart.pulmonary_trunk")
        branch_direction = _normalise(np.array([-0.9 * self.mirror, 0.25, 0.1]))
        tubes.append(
            _Tube(
                "heart.pulmonary_arteries",
                trunk.end,
                trunk.end + branch_direction * length * 0.9,
                vessel_radius * 0.85,
                "heart.pulmonary_trunk",
            )
        )
        return tuple(tubes)

    # ------------------------------------------------------------------
    # geometry queries
    # ------------------------------------------------------------------
    def to_local(self, points: FloatArray) -> FloatArray:
        """Scene space to organ space."""
        return np.asarray(points @ self.rotation, dtype=np.float64)

    def _taper(self, local: FloatArray) -> FloatArray:
        height = np.clip(-local[:, 1] / max(self.radii[1], 1e-6), 0.0, 1.0)
        return np.asarray(1.0 - self.parameters.apex_taper * height**1.5, dtype=np.float64)

    def envelope_value(self, local: FloatArray, scale: float = 1.0) -> FloatArray:
        """Implicit value of the organ envelope: below 1.0 is inside."""
        taper = self._taper(local)
        radii = self.radii * scale
        scaled = local / np.stack([radii[0] * taper, np.full_like(taper, radii[1]), radii[2] * taper], axis=-1)
        return np.asarray((scaled**2).sum(axis=-1), dtype=np.float64)

    def organ_occupancy(self, points: FloatArray) -> BoolArray:
        """Whole-organ occupancy, including cavities. The coherent object."""
        return np.asarray(self.envelope_value(self.to_local(points)) <= 1.0, dtype=np.bool_)

    def tissue_occupancy(self, points: FloatArray) -> BoolArray:
        """Organ minus the blood pools: what a surface render would show as muscle."""
        local = self.to_local(points)
        inside = self.envelope_value(local) <= 1.0
        for cavity in self.cavities:
            inside &= cavity.normalised_distance(local) > 1.0
        return np.asarray(inside, dtype=np.bool_)

    # ------------------------------------------------------------------
    # ownership, exactly as fixed in docs/step7/01_experimental_plan.md
    # ------------------------------------------------------------------
    def ownership(self, points: FloatArray) -> IntArray:
        """Owner slot for every point, or ``BACKGROUND``."""
        parameters = self.parameters
        local = self.to_local(points)
        scale = float(np.mean(self.radii))
        labels = np.full(points.shape[0], BACKGROUND, dtype=np.int64)

        envelope = self.envelope_value(local)
        inside_organ = envelope <= 1.0
        inside_sac = self.envelope_value(local, 1.0 + parameters.pericardium_gap) <= 1.0

        # 2. inside the sac but outside the organ: pericardium
        labels[inside_sac & ~inside_organ] = self.slot_of["heart.pericardium"]

        gaps = np.stack([cavity.surface_gap(local) for cavity in self.cavities], axis=0)
        inside_cavity = gaps <= 0.0

        # 6. myocardial tissue, resolved before the higher-priority classes overwrite it
        tissue = inside_organ & ~inside_cavity.any(axis=0)
        nearest_gap = gaps.min(axis=0)
        septum_band = parameters.septum_band * scale
        endocardium = parameters.endocardium_fraction * parameters.wall_fraction * scale * 2.0
        epicardium = parameters.epicardium_fraction * parameters.wall_fraction * scale * 2.0
        outer_gap = (np.sqrt(np.maximum(envelope, 0.0)) - 1.0) * scale

        labels[tissue] = self.slot_of["heart.myocardium"]
        labels[tissue & (np.abs(outer_gap) <= epicardium)] = self.slot_of["heart.epicardium"]
        labels[tissue & (nearest_gap <= endocardium)] = self.slot_of["heart.endocardium"]

        # 6a. septa: tissue within the band of BOTH cavities of a septal pair, exactly as
        # the pre-registered rule states. An earlier version also required the pair to be
        # the two nearest cavities overall, which is a condition the rule does not make
        # and which lost the septum whenever an atrium happened to be closer.
        gap_of = {
            cavity.entity_id: gaps[index] for index, cavity in enumerate(self.cavities)
        }
        septal_pairs = (
            (
                "heart.interventricular_septum",
                gap_of["heart.left_ventricle"],
                gap_of["heart.right_ventricle"],
            ),
            (
                "heart.interatrial_septum",
                gap_of["heart.left_atrium"],
                gap_of["heart.right_atrium"],
            ),
        )
        scores = np.stack([np.maximum(first, second) for _, first, second in septal_pairs])
        best = scores.argmin(axis=0)
        for index, (entity_id, _, _) in enumerate(septal_pairs):
            selected = tissue & (scores[index] <= septum_band) & (best == index)
            labels[selected] = self.slot_of[entity_id]

        # 5. vessels
        for tube in self.tubes:
            distance = tube.distance(local)
            labels[(distance <= tube.radius) & inside_sac] = self.slot_of[tube.entity_id]
            labels[(distance <= tube.radius) & ~inside_sac] = self.slot_of[tube.entity_id]

        # 4. valve annuli. Not masked to the organ envelope: the pre-registered rule
        # places valves above vessels and below cavities and says nothing about the
        # envelope, and an outflow valve legitimately sits at the boundary.
        for annulus in self.annuli:
            labels[annulus.occupancy(local)] = self.slot_of[annulus.entity_id]

        # 3. cavities, highest priority among interior classes
        for index, cavity in enumerate(self.cavities):
            labels[inside_cavity[index]] = self.slot_of[cavity.entity_id]
        return labels

    def entity_occupancy(self, entity_id: str, points: FloatArray) -> BoolArray:
        """Points owned by one entity."""
        return np.asarray(self.ownership(points) == self.slot_of[entity_id], dtype=np.bool_)

    def scene_occupancy(self, points: FloatArray, entity_ids: Sequence[str] | None = None) -> BoolArray:
        """Occupancy of the entities visible at a level of detail.

        Level-specific: the union of what that level shows, which is what gives the
        level-of-detail objective a genuinely different target per level.
        """
        labels = self.ownership(points)
        if entity_ids is None:
            return np.asarray(labels != BACKGROUND, dtype=np.bool_)
        wanted = np.array([self.slot_of[entity_id] for entity_id in entity_ids], dtype=np.int64)
        return np.asarray(np.isin(labels, wanted), dtype=np.bool_)

    # ------------------------------------------------------------------
    def sample_points(self, count: int, rng: np.random.Generator, *, jitter: float = 0.04) -> FloatArray:
        """Query points concentrated where the organ is."""
        uniform = rng.uniform(-1.0, 1.0, size=(count // 3, 3))
        directions = rng.normal(size=(count - count // 3, 3))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
        radii = rng.uniform(0.2, 1.15, size=(directions.shape[0], 1))
        shell = directions * radii * self.radii * (1.0 + self.parameters.pericardium_gap)
        shell = shell @ self.rotation.T + rng.normal(scale=jitter, size=shell.shape)
        return np.clip(np.concatenate([uniform, shell]), -1.0, 1.0)

    def proposal_points(
        self, entity_id: str, count: int, rng: np.random.Generator
    ) -> FloatArray:
        """Candidate points near where an entity was constructed.

        Thin structures such as valves and septa occupy a fraction of a percent of the
        volume, so uniform sampling finds almost none of them. Proposals come from the
        construction geometry: a cavity ellipsoid, a valve disc, a vessel axis, or the
        organ shell for wall layers. They are only proposals; the ownership rule still
        decides who owns each point, and a proposal that loses is simply a negative.
        """
        scale = float(np.mean(self.radii))
        if entity_id in self.cavity_index:
            cavity = self.cavity_index[entity_id]
            directions = rng.normal(size=(count, 3))
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
            radii = rng.uniform(0.0, 1.35, size=(count, 1))
            local = cavity.centre + directions * radii * cavity.radii
        elif any(annulus.entity_id == entity_id for annulus in self.annuli):
            annulus = next(a for a in self.annuli if a.entity_id == entity_id)
            helper = np.array([0.0, 0.0, 1.0]) if abs(annulus.axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            first = _normalise(np.cross(annulus.axis, helper))
            second = np.cross(annulus.axis, first)
            angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
            radius = np.sqrt(rng.uniform(0.0, 1.0, size=count)) * annulus.radius * 1.3
            along = rng.uniform(-1.6, 1.6, size=count) * annulus.thickness
            local = (
                annulus.centre
                + radius[:, None] * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)
                + along[:, None] * annulus.axis
            )
        elif any(tube.entity_id == entity_id for tube in self.tubes):
            tube = next(t for t in self.tubes if t.entity_id == entity_id)
            axis = tube.end - tube.start
            along = rng.uniform(-0.1, 1.1, size=(count, 1))
            offset = rng.normal(size=(count, 3)) * tube.radius * 0.8
            local = tube.start + along * axis + offset
        else:
            directions = rng.normal(size=(count, 3))
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
            if entity_id == "heart.pericardium":
                radii = rng.uniform(1.0, 1.0 + self.parameters.pericardium_gap, size=(count, 1))
            elif entity_id == "heart.epicardium":
                radii = rng.uniform(0.9, 1.02, size=(count, 1))
            elif entity_id in {
                "heart.interatrial_septum",
                "heart.interventricular_septum",
            }:
                pair = (
                    ("heart.left_atrium", "heart.right_atrium")
                    if "interatrial" in entity_id
                    else ("heart.left_ventricle", "heart.right_ventricle")
                )
                first_cavity = self.cavity_index[pair[0]]
                second_cavity = self.cavity_index[pair[1]]
                midpoint = (first_cavity.centre + second_cavity.centre) / 2.0
                spread = np.maximum(first_cavity.radii, second_cavity.radii)
                local = midpoint + rng.normal(size=(count, 3)) * spread * np.array([0.45, 0.8, 0.8])
                return np.clip(local @ self.rotation.T, -1.0, 1.0)
            else:
                radii = rng.uniform(0.35, 0.95, size=(count, 1))
            local = directions * radii * self.radii + rng.normal(size=(count, 3)) * scale * 0.03
        return np.clip(local @ self.rotation.T, -1.0, 1.0)

    def entity_points(
        self, entity_id: str, count: int, rng: np.random.Generator, *, oversample: int = 6
    ) -> tuple[FloatArray, BoolArray]:
        """Points near an entity, with the exact label of whether it owns each one."""
        proposals = self.proposal_points(entity_id, count * oversample, rng)
        labels = self.ownership(proposals)
        owned = labels == self.slot_of[entity_id]
        positive = proposals[owned]
        negative = proposals[~owned]
        wanted_positive = min(count // 2, positive.shape[0])
        chosen = [positive[:wanted_positive], negative[: count - wanted_positive]]
        points = np.concatenate([part for part in chosen if part.shape[0]])
        if points.shape[0] < count:
            filler = self.proposal_points(entity_id, count - points.shape[0], rng)
            points = np.concatenate([points, filler])
        points = points[:count]
        return points, self.ownership(points) == self.slot_of[entity_id]

    def entity_statistics(
        self, rng: np.random.Generator, count: int = 4000
    ) -> dict[str, dict[str, FloatArray]]:
        """Centroid and extent of every entity, measured from the labelled field.

        Uses per-entity proposals so thin structures are measured from enough points to
        be meaningful, rather than from the handful a uniform sample would find.
        """
        out: dict[str, dict[str, FloatArray]] = {}
        for entity_id, slot in self.slot_of.items():
            proposals = self.proposal_points(entity_id, count, rng)
            owned = proposals[self.ownership(proposals) == slot]
            if owned.shape[0] < 8:
                continue
            out[entity_id] = {
                "centroid": owned.mean(axis=0),
                "extent": owned.std(axis=0) * 2.0,
                "count": np.asarray([owned.shape[0]], dtype=np.float64),
            }
        return out

    def frames(self, rng: np.random.Generator, count: int = 6000) -> dict[str, list[float]]:
        """Canonical frame per entity, measured rather than declared."""
        statistics = self.entity_statistics(rng, count)
        frames: dict[str, list[float]] = {}
        for entity_id, values in statistics.items():
            centroid = values["centroid"]
            extent = np.maximum(values["extent"], 1e-3)
            frame = np.concatenate([centroid, np.log(extent), np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])])
            frames[entity_id] = [float(x) for x in frame]
        return frames

    def present_entities(self, statistics: Mapping[str, Mapping[str, FloatArray]]) -> tuple[str, ...]:
        """Entities that actually own enough volume to be measurable."""
        return tuple(
            entity_id for entity_id in self.entity_ids if entity_id in statistics
        )
