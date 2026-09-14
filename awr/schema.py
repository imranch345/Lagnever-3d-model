"""Core value types of the Anatomical World Representation (AWR).

This module defines the *typed* building blocks that every other layer shares:
controlled vocabularies, per-entity state, and the reference that ties an
anatomical entity to a future geometry component.

Design rules enforced here:

* Nothing is an untyped dictionary. Free-form data is confined to the explicit
  ``metadata`` / ``extra`` mappings, which are never used for semantics the
  system reasons about.
* Controlled vocabularies are closed enums. Extending the vocabulary is a
  deliberate one-line change plus an ontology bump, not an accident: unknown
  values raise instead of being silently accepted.
* Every type round-trips through ``to_dict`` / ``from_dict`` so that a scene can
  be persisted and restored without losing identity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Self, TypeVar

from awr.errors import InvalidStateError, SchemaError

__all__ = [
    "AWR_SCHEMA_VERSION",
    "AnatomyType",
    "Laterality",
    "EducationalLevel",
    "GeometryRepresentationKind",
    "VisibilitySource",
    "LodLevel",
    "LodPolicy",
    "Transform",
    "MaterialState",
    "AnimationState",
    "GeometryReference",
    "EntityState",
    "coerce_enum",
    "unique",
]

AWR_SCHEMA_VERSION = "awr-scene-1"
"""Version tag written into every serialised scene payload."""

LodLevel = int
"""Anatomical level of detail. Provisional ladder 0-4, defined in configs/heart.yaml."""


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def unique(values: Iterable[str]) -> tuple[str, ...]:
    """Return ``values`` without duplicates, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def coerce_enum(enum_cls: type[_EnumT], value: str | _EnumT, *, field_name: str) -> _EnumT:
    """Convert a string to a member of ``enum_cls`` with an actionable error."""
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(m.value for m in enum_cls))
        raise SchemaError(
            f"{field_name}={value!r} is not a registered {enum_cls.__name__}. "
            f"Allowed values: {allowed}. Extend the enum in awr/schema.py if the "
            "vocabulary genuinely needs to grow."
        ) from exc


class AnatomyType(StrEnum):
    """Kind of anatomical entity.

    Closed vocabulary on purpose: an unrecognised ``anatomy_type`` in ontology
    data is a data error, not a new concept to be invented at load time.
    """

    ORGAN = "organ"
    GROUP = "group"
    CHAMBER = "chamber"
    VALVE = "valve"
    VESSEL = "vessel"
    SEPTUM = "septum"
    WALL_LAYER = "wall_layer"
    MUSCLE = "muscle"
    CONNECTIVE_STRUCTURE = "connective_structure"
    SURFACE_FEATURE = "surface_feature"
    VASCULAR_SYSTEM = "vascular_system"
    CONDUCTION_SYSTEM = "conduction_system"
    CONDUCTION_NODE = "conduction_node"
    CONDUCTION_PATHWAY = "conduction_pathway"


class Laterality(StrEnum):
    """Body-side of an entity, using anatomical (subject) left and right."""

    LEFT = "left"
    RIGHT = "right"
    MEDIAN = "median"
    NONE = "none"


class EducationalLevel(StrEnum):
    """Audience level. Mapped to an LOD by configs/heart.yaml."""

    PRIMARY = "primary"
    SCHOOL = "school"
    HIGH_SCHOOL = "high_school"
    UNIVERSITY = "university"
    MEDICAL = "medical"


class GeometryRepresentationKind(StrEnum):
    """How a geometry component is represented.

    ``SYMBOLIC_PLACEHOLDER`` means "a component slot exists and is reserved for
    this entity, but no geometry has been produced". The prototype only ever
    produces symbolic placeholders; every other member is declared so that the
    correspondence layer is stable when Step 5 chooses a representation.
    """

    SYMBOLIC_PLACEHOLDER = "symbolic_placeholder"
    MESH = "mesh"
    IMPLICIT_FIELD = "implicit_field"
    POINT_CLOUD = "point_cloud"
    VOXEL_GRID = "voxel_grid"
    GAUSSIAN_SPLAT = "gaussian_splat"


class VisibilitySource(StrEnum):
    """Why an entity currently has its visibility value.

    ``LOD_DEFAULT`` means the value came from the LOD policy; ``MANUAL`` means a
    user command overrode it. A level-of-detail change recomputes ``LOD_DEFAULT``
    entities and clears ``MANUAL`` overrides, which keeps "show more detail"
    predictable.
    """

    LOD_DEFAULT = "lod_default"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class LodPolicy:
    """Level-of-detail window in which an entity is visible by default.

    The entity exists in the AWR at every LOD; this policy only decides default
    visibility. ``max_lod`` is inclusive and ``None`` means "no upper bound".
    """

    min_lod: LodLevel = 0
    max_lod: LodLevel | None = None

    def __post_init__(self) -> None:
        if self.min_lod < 0:
            raise InvalidStateError(f"min_lod must be >= 0, got {self.min_lod}.")
        if self.max_lod is not None and self.max_lod < self.min_lod:
            raise InvalidStateError(
                f"max_lod ({self.max_lod}) must be >= min_lod ({self.min_lod})."
            )

    def includes(self, lod: LodLevel) -> bool:
        """Return whether the entity is within its default-visible LOD window."""
        if lod < self.min_lod:
            return False
        return self.max_lod is None or lod <= self.max_lod

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {"min_lod": self.min_lod, "max_lod": self.max_lod}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(min_lod=int(payload.get("min_lod", 0)), max_lod=payload.get("max_lod"))


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Transform:
    """Rigid transform of an entity in scene space.

    Rotation is a ``(w, x, y, z)`` quaternion. The prototype never sets a
    non-identity transform; the type exists so that transform edits and the
    future geometry decoder have a stable contract.
    """

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = (1.0, 0.0, 0.0, 0.0)
    scale: Vec3 = (1.0, 1.0, 1.0)

    @property
    def is_identity(self) -> bool:
        """Whether this is the identity transform."""
        return (
            self.translation == (0.0, 0.0, 0.0)
            and self.rotation == (1.0, 0.0, 0.0, 0.0)
            and self.scale == (1.0, 1.0, 1.0)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "translation": list(self.translation),
            "rotation": list(self.rotation),
            "scale": list(self.scale),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""

        def _vec(key: str, size: int, default: Sequence[float]) -> tuple[float, ...]:
            raw = payload.get(key, default)
            values = tuple(float(v) for v in raw)
            if len(values) != size:
                raise SchemaError(f"Transform.{key} must have {size} components, got {values!r}.")
            return values

        return cls(
            translation=_vec("translation", 3, (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            rotation=_vec("rotation", 4, (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            scale=_vec("scale", 3, (1.0, 1.0, 1.0)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MaterialState:
    """Material appearance of an entity, excluding opacity.

    Opacity lives on :class:`EntityState` because it is a first-class semantic
    control ("make the left ventricle transparent") rather than a rendering
    detail. Everything here is reserved for the future material decoder and is
    ``None`` in the deterministic prototype.
    """

    preset: str | None = None
    base_color: Vec3 | None = None
    roughness: float | None = None
    metallic: float | None = None
    tissue_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "preset": self.preset,
            "base_color": list(self.base_color) if self.base_color is not None else None,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "tissue_class": self.tissue_class,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        color = payload.get("base_color")
        return cls(
            preset=payload.get("preset"),
            base_color=tuple(float(c) for c in color) if color else None,  # type: ignore[arg-type]
            roughness=payload.get("roughness"),
            metallic=payload.get("metallic"),
            tissue_class=payload.get("tissue_class"),
        )


@dataclass(frozen=True, slots=True)
class AnimationState:
    """Animation binding of an entity.

    The prototype sets semantic animation state only (which clip an entity takes
    part in, which phase, what role). No geometry is deformed and no renderer is
    driven; that is the animation decoder's job in a later step.
    """

    clip_id: str | None = None
    role: str | None = None
    phase: str | None = None
    normalized_time: float | None = None
    playing: bool = False

    def __post_init__(self) -> None:
        if self.normalized_time is not None and not 0.0 <= self.normalized_time <= 1.0:
            raise InvalidStateError(
                f"normalized_time must be within [0, 1], got {self.normalized_time}."
            )

    @property
    def is_animated(self) -> bool:
        """Whether the entity is bound to any clip."""
        return self.clip_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "clip_id": self.clip_id,
            "role": self.role,
            "phase": self.phase,
            "normalized_time": self.normalized_time,
            "playing": self.playing,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            clip_id=payload.get("clip_id"),
            role=payload.get("role"),
            phase=payload.get("phase"),
            normalized_time=payload.get("normalized_time"),
            playing=bool(payload.get("playing", False)),
        )


@dataclass(frozen=True, slots=True)
class GeometryReference:
    """Stable link from an anatomical entity to a geometry component.

    This is the load-bearing piece of the "a mesh must never lose semantic part
    identity" requirement. The component id is allocated when the entity enters
    the scene, long before any geometry exists, and never changes afterwards.
    """

    component_id: str
    kind: GeometryRepresentationKind = GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER
    uri: str | None = None
    checksum: str | None = None
    producer: str | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise InvalidStateError("GeometryReference.component_id must be non-empty.")

    @property
    def is_resolved(self) -> bool:
        """Whether actual geometry data backs this reference."""
        return self.kind is not GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER and bool(self.uri)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "component_id": self.component_id,
            "kind": str(self.kind),
            "uri": self.uri,
            "checksum": self.checksum,
            "producer": self.producer,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            component_id=str(payload["component_id"]),
            kind=coerce_enum(
                GeometryRepresentationKind,
                payload.get("kind", GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER),
                field_name="GeometryReference.kind",
            ),
            uri=payload.get("uri"),
            checksum=payload.get("checksum"),
            producer=payload.get("producer"),
        )


@dataclass(slots=True)
class EntityState:
    """Mutable per-entity state inside a scene.

    Mutated only through editing operations (see ``editing/operations.py``), so
    that every change is recorded in scene history. Identity and semantics are
    deliberately *not* here: they live on the immutable entity definition, which
    is why an edit can never change an entity id.
    """

    visibility: bool = False
    opacity: float = 1.0
    visibility_source: VisibilitySource = VisibilitySource.LOD_DEFAULT
    material: MaterialState = field(default_factory=MaterialState)
    transform: Transform = field(default_factory=Transform)
    animation: AnimationState = field(default_factory=AnimationState)
    geometry_reference: GeometryReference | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`InvalidStateError` if any field is out of domain."""
        if not 0.0 <= self.opacity <= 1.0:
            raise InvalidStateError(
                f"opacity must be within [0.0, 1.0], got {self.opacity!r}."
            )

    def copy(self) -> EntityState:
        """Return an independent copy (used for history snapshots)."""
        return replace(self, extra=dict(self.extra))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "visibility": self.visibility,
            "opacity": self.opacity,
            "visibility_source": str(self.visibility_source),
            "material": self.material.to_dict(),
            "transform": self.transform.to_dict(),
            "animation": self.animation.to_dict(),
            "geometry_reference": (
                self.geometry_reference.to_dict() if self.geometry_reference else None
            ),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        geometry = payload.get("geometry_reference")
        return cls(
            visibility=bool(payload.get("visibility", False)),
            opacity=float(payload.get("opacity", 1.0)),
            visibility_source=coerce_enum(
                VisibilitySource,
                payload.get("visibility_source", VisibilitySource.LOD_DEFAULT),
                field_name="EntityState.visibility_source",
            ),
            material=MaterialState.from_dict(payload.get("material", {})),
            transform=Transform.from_dict(payload.get("transform", {})),
            animation=AnimationState.from_dict(payload.get("animation", {})),
            geometry_reference=GeometryReference.from_dict(geometry) if geometry else None,
            extra=dict(payload.get("extra", {})),
        )
