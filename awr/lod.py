"""Anatomical level of detail (LOD).

The governing principle is that **the internal representation is always richer
than the visible output**. Every entity of the ontology exists in the AWR at
every LOD; the LOD only decides which entities are visible *by default*. A
school-level scene therefore still holds the conduction system, it simply does
not show it.

The ladder itself (level numbers, names, descriptions) and the audience mapping
are configuration, not code: see ``configs/heart.yaml``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from awr.errors import ConfigError, InvalidStateError
from awr.schema import EducationalLevel, LodLevel, coerce_enum

__all__ = [
    "LodLevelSpec",
    "LodLadder",
    "EducationalLevelSpec",
    "EducationalLevelMap",
    "default_visibility_at",
]


@dataclass(frozen=True, slots=True)
class LodLevelSpec:
    """One rung of the LOD ladder."""

    level: LodLevel
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class LodLadder:
    """The ordered set of valid LOD levels for a domain."""

    levels: tuple[LodLevelSpec, ...]

    def __post_init__(self) -> None:
        if not self.levels:
            raise ConfigError("The LOD ladder must define at least one level.")
        numbers = [spec.level for spec in self.levels]
        if numbers != sorted(numbers):
            raise ConfigError(f"LOD levels must be declared in ascending order, got {numbers}.")
        if len(set(numbers)) != len(numbers):
            raise ConfigError(f"LOD levels must be unique, got {numbers}.")

    @property
    def min_level(self) -> LodLevel:
        """Lowest valid LOD."""
        return self.levels[0].level

    @property
    def max_level(self) -> LodLevel:
        """Highest valid LOD."""
        return self.levels[-1].level

    def __contains__(self, level: object) -> bool:
        return any(spec.level == level for spec in self.levels)

    def spec(self, level: LodLevel) -> LodLevelSpec:
        """Return the spec for ``level`` or raise."""
        for candidate in self.levels:
            if candidate.level == level:
                return candidate
        raise InvalidStateError(
            f"LOD {level} is not defined. Valid levels: "
            f"{', '.join(str(s.level) for s in self.levels)}."
        )

    def name_of(self, level: LodLevel) -> str:
        """Human-readable name of a level."""
        return self.spec(level).name

    def validate(self, level: LodLevel) -> LodLevel:
        """Return ``level`` if valid, else raise :class:`InvalidStateError`."""
        self.spec(level)
        return level

    def step(self, level: LodLevel, delta: int) -> LodLevel:
        """Move ``delta`` rungs up or down, clamped to the ladder.

        Used by relative commands such as "show more detail".
        """
        target = level + delta
        return max(self.min_level, min(self.max_level, target))

    @classmethod
    def from_mapping(cls, payload: Mapping[Any, Mapping[str, Any]]) -> LodLadder:
        """Build a ladder from the ``lod_levels`` section of a config file."""
        specs: list[LodLevelSpec] = []
        for raw_level, body in sorted(payload.items(), key=lambda kv: int(kv[0])):
            specs.append(
                LodLevelSpec(
                    level=int(raw_level),
                    name=str(body.get("name", f"lod_{raw_level}")),
                    description=str(body.get("description", "")),
                )
            )
        return cls(tuple(specs))


@dataclass(frozen=True, slots=True)
class EducationalLevelSpec:
    """Mapping of one audience level onto an LOD."""

    level: EducationalLevel
    lod: LodLevel
    description: str


@dataclass(frozen=True, slots=True)
class EducationalLevelMap:
    """Audience level to LOD mapping, e.g. ``medical -> 4``."""

    specs: tuple[EducationalLevelSpec, ...]

    def __post_init__(self) -> None:
        if not self.specs:
            raise ConfigError("At least one educational level must be configured.")

    def lod_for(self, level: EducationalLevel) -> LodLevel:
        """LOD that an audience level maps to."""
        for spec in self.specs:
            if spec.level is level:
                return spec.lod
        raise ConfigError(f"Educational level {level!r} is not configured.")

    def level_for_lod(self, lod: LodLevel) -> EducationalLevel | None:
        """Audience level whose LOD equals ``lod``, if any.

        Purely for reporting: the LOD is the authoritative scene field, and an
        LOD reached by a relative command may map to no audience level at all.
        """
        for spec in self.specs:
            if spec.lod == lod:
                return spec.level
        return None

    def levels(self) -> tuple[EducationalLevel, ...]:
        """All configured audience levels."""
        return tuple(spec.level for spec in self.specs)

    def describe(self, level: EducationalLevel) -> str:
        """Description of an audience level."""
        for spec in self.specs:
            if spec.level is level:
                return spec.description
        return ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Mapping[str, Any]]) -> EducationalLevelMap:
        """Build the map from the ``educational_levels`` config section."""
        specs = [
            EducationalLevelSpec(
                level=coerce_enum(EducationalLevel, name, field_name="educational_levels"),
                lod=int(body["lod"]),
                description=str(body.get("description", "")),
            )
            for name, body in payload.items()
        ]
        specs.sort(key=lambda spec: spec.lod)
        return cls(tuple(specs))


def default_visibility_at(
    *, renderable: bool, min_lod: LodLevel, max_lod: LodLevel | None, lod: LodLevel
) -> bool:
    """Default visibility of an entity at a given LOD.

    Groups (``renderable=False``) are never visible themselves; commands that
    target a group act on its renderable descendants.
    """
    if not renderable:
        return False
    if lod < min_lod:
        return False
    return max_lod is None or lod <= max_lod


def visible_ids_at(
    entities: Iterable[tuple[str, bool, LodLevel, LodLevel | None]], lod: LodLevel
) -> tuple[str, ...]:
    """Ids that are visible by default at ``lod``.

    Each input tuple is ``(entity_id, renderable, min_lod, max_lod)``. Kept as a
    free function so the LOD policy can be unit tested without a scene.
    """
    return tuple(
        entity_id
        for entity_id, renderable, min_lod, max_lod in entities
        if default_visibility_at(renderable=renderable, min_lod=min_lod, max_lod=max_lod, lod=lod)
    )
