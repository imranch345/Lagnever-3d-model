"""Global parameters of a whole-organ synthetic heart, and the anatomical variants.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED.

Every entity in this corpus is derived from the parameters here. Nothing is placed
independently: the organ envelope, the cavity positions, the valve annuli and the wall
layers are all functions of one global parameter set, so the whole determines the parts.

The **variants** are what make the Step 7 experiments possible. Presence, entity
identities and text features are identical across variants; only the arrangement, and
therefore the measured relationship graph, differs. A model that cannot read relations
has no way to tell the variants apart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DATA_LABEL",
    "Variant",
    "OrganParameters",
    "sample_parameters",
    "rotation_matrix",
]

FloatArray = npt.NDArray[np.float64]

DATA_LABEL = "SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED"


class Variant(StrEnum):
    """Anatomical arrangement of a scene.

    Each variant changes where structures sit relative to one another, and therefore
    changes the relationships measured from the finished organ. It never changes which
    entities exist.
    """

    NORMAL = "normal"
    MIRRORED = "mirrored"
    """Reflected across the midline: the left chambers sit on the anatomical right."""
    TRANSPOSED = "transposed"
    """The great arteries swap ventricles: the aorta leaves the right ventricle."""
    ROTATED = "rotated"
    """Rotated about the vertical axis, which changes anterior and posterior relations."""

    @property
    def index(self) -> int:
        """Stable integer index, used only for reporting."""
        return list(Variant).index(self)


def rotation_matrix(yaw: float, pitch: float, roll: float) -> FloatArray:
    """Rotation matrix from yaw about y, pitch about z and roll about x."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    about_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    about_z = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return np.asarray(about_y @ about_z @ about_x, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class OrganParameters:
    """One organ, described globally.

    Cavity sizes are fractions of the organ radii, so changing the organ changes every
    structure inside it. That is the property the Step 6 corpus lacked.
    """

    family_id: int
    variant: Variant
    organ_radii: tuple[float, float, float]
    apex_taper: float
    tilt: tuple[float, float, float]
    wall_fraction: float
    endocardium_fraction: float
    epicardium_fraction: float
    pericardium_gap: float
    septum_band: float
    """Half-width of the band in which tissue between two cavities counts as septum."""
    ventricle_cavity: tuple[float, float, float]
    atrium_cavity: tuple[float, float, float]
    septal_offset: float
    av_plane: float
    apex_drop: float
    anterior_offset: float
    valve_thickness: float
    valve_radius_fraction: float
    vessel_radius_fraction: float
    vessel_length_fraction: float
    entity_scale: tuple[tuple[str, float], ...] = ()
    """Per-entity size overrides. Empty for a generated scene; an edit sets exactly one.

    The organ stays coherent: scaling one cavity moves the tissue around it, which is why
    edit locality has to be measured against what the generator actually changed rather
    than assumed from the instruction.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "family_id": self.family_id,
            "variant": str(self.variant),
            "organ_radii": list(self.organ_radii),
            "apex_taper": self.apex_taper,
            "tilt": list(self.tilt),
            "wall_fraction": self.wall_fraction,
            "endocardium_fraction": self.endocardium_fraction,
            "epicardium_fraction": self.epicardium_fraction,
            "pericardium_gap": self.pericardium_gap,
            "septum_band": self.septum_band,
            "ventricle_cavity": list(self.ventricle_cavity),
            "atrium_cavity": list(self.atrium_cavity),
            "septal_offset": self.septal_offset,
            "av_plane": self.av_plane,
            "apex_drop": self.apex_drop,
            "anterior_offset": self.anterior_offset,
            "valve_thickness": self.valve_thickness,
            "valve_radius_fraction": self.valve_radius_fraction,
            "vessel_radius_fraction": self.vessel_radius_fraction,
            "vessel_length_fraction": self.vessel_length_fraction,
            "entity_scale": [[name, value] for name, value in self.entity_scale],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OrganParameters:
        """Rebuild from :meth:`to_dict` output."""

        def triple(values: Any) -> tuple[float, float, float]:
            first, second, third = (float(value) for value in values)
            return (first, second, third)

        return cls(
            family_id=int(payload["family_id"]),
            variant=Variant(payload["variant"]),
            organ_radii=triple(payload["organ_radii"]),
            apex_taper=float(payload["apex_taper"]),
            tilt=triple(payload["tilt"]),
            wall_fraction=float(payload["wall_fraction"]),
            endocardium_fraction=float(payload["endocardium_fraction"]),
            epicardium_fraction=float(payload["epicardium_fraction"]),
            pericardium_gap=float(payload["pericardium_gap"]),
            septum_band=float(payload["septum_band"]),
            ventricle_cavity=triple(payload["ventricle_cavity"]),
            atrium_cavity=triple(payload["atrium_cavity"]),
            septal_offset=float(payload["septal_offset"]),
            av_plane=float(payload["av_plane"]),
            apex_drop=float(payload["apex_drop"]),
            anterior_offset=float(payload["anterior_offset"]),
            valve_thickness=float(payload["valve_thickness"]),
            valve_radius_fraction=float(payload["valve_radius_fraction"]),
            vessel_radius_fraction=float(payload["vessel_radius_fraction"]),
            vessel_length_fraction=float(payload["vessel_length_fraction"]),
            entity_scale=tuple(
                (str(name), float(value)) for name, value in payload.get("entity_scale", ())
            ),
        )

    def scale_for(self, entity_id: str) -> float:
        """Size override for one entity, or 1.0."""
        for name, value in self.entity_scale:
            if name == entity_id:
                return value
        return 1.0

    def rotation(self) -> FloatArray:
        """Organ-to-scene rotation."""
        return rotation_matrix(*self.tilt)

    def replace_scalar(self, field: str, value: float) -> OrganParameters:
        """Return a copy with one scalar parameter changed, used to build edit pairs."""
        from dataclasses import replace

        return replace(self, **{field: value})


def _jitter(rng: np.random.Generator, base: float, spread: float) -> float:
    return float(base * (1.0 + rng.uniform(-spread, spread)))


def sample_parameters(
    rng: np.random.Generator, family_id: int, variant: Variant
) -> OrganParameters:
    """Sample one organ.

    The family fixes an archetype's proportions; the scene jitters within it. Splitting
    by family is therefore a genuine generalisation test rather than a test on
    near-duplicates.
    """
    family_rng = np.random.default_rng(7_919_000 + family_id)
    archetype = (
        _jitter(family_rng, 0.42, 0.14),
        _jitter(family_rng, 0.56, 0.12),
        _jitter(family_rng, 0.38, 0.14),
    )
    archetype_wall = _jitter(family_rng, 0.20, 0.25)
    archetype_ventricle = (
        _jitter(family_rng, 0.40, 0.14),
        _jitter(family_rng, 0.46, 0.12),
        _jitter(family_rng, 0.40, 0.14),
    )

    yaw = float(rng.uniform(-0.12, 0.12))
    if variant is Variant.ROTATED:
        yaw = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.38, 0.52))
    return OrganParameters(
        family_id=family_id,
        variant=variant,
        organ_radii=(
            _jitter(rng, archetype[0], 0.06),
            _jitter(rng, archetype[1], 0.06),
            _jitter(rng, archetype[2], 0.06),
        ),
        apex_taper=_jitter(rng, 0.45, 0.15),
        tilt=(yaw, float(rng.uniform(-0.10, 0.10)), float(rng.uniform(-0.08, 0.08))),
        wall_fraction=_jitter(rng, archetype_wall, 0.12),
        endocardium_fraction=_jitter(rng, 0.22, 0.15),
        epicardium_fraction=_jitter(rng, 0.20, 0.15),
        pericardium_gap=_jitter(rng, 0.09, 0.20),
        septum_band=_jitter(rng, 0.20, 0.15),
        ventricle_cavity=(
            _jitter(rng, archetype_ventricle[0], 0.07),
            _jitter(rng, archetype_ventricle[1], 0.07),
            _jitter(rng, archetype_ventricle[2], 0.07),
        ),
        atrium_cavity=(
            _jitter(rng, 0.30, 0.10),
            _jitter(rng, 0.24, 0.10),
            _jitter(rng, 0.30, 0.10),
        ),
        septal_offset=float(rng.uniform(-0.03, 0.03)),
        av_plane=_jitter(rng, 0.12, 0.20),
        apex_drop=_jitter(rng, 0.30, 0.12),
        anterior_offset=_jitter(rng, 0.30, 0.18),
        valve_thickness=_jitter(rng, 0.10, 0.15),
        valve_radius_fraction=_jitter(rng, 0.62, 0.12),
        vessel_radius_fraction=_jitter(rng, 0.30, 0.14),
        vessel_length_fraction=_jitter(rng, 0.75, 0.15),
    )
