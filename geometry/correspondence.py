"""Entity-to-geometry correspondence.

This is the hinge of the whole architecture. A generated mesh is only useful to
Lagnav if every piece of it still knows which anatomical entity it is, so that
"make the left ventricle transparent" can be answered years later on geometry
produced by a model that does not exist yet.

Rules enforced here:

* A component belongs to exactly one entity. Sharing a component between two
  entities is an error, not a warning.
* An entity may own several components (future LOD variants or sub-parts).
* Component ids are allocated deterministically from the ontology order, so the
  same ontology always yields the same ids and correspondence files stay
  comparable between runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from awr.errors import GeometryError
from awr.schema import GeometryReference, GeometryRepresentationKind

__all__ = ["ComponentIdAllocator", "CorrespondenceEntry", "GeometryCorrespondence"]


@dataclass(frozen=True, slots=True)
class ComponentIdAllocator:
    """Deterministic component-id generator.

    ``geometry_part_0042``-style ids are produced by zero-padding a counter. The
    counter follows ontology declaration order, so ids are reproducible.
    """

    prefix: str = "geometry_part_"
    digits: int = 4
    start: int = 0

    def id_for_index(self, index: int) -> str:
        """Component id for a zero-based slot index."""
        if index < 0:
            raise GeometryError(f"Component index must be non-negative, got {index}.")
        return f"{self.prefix}{index + self.start:0{self.digits}d}"

    def allocate(self, entity_ids: Sequence[str]) -> dict[str, str]:
        """Map each entity id to a freshly allocated component id, in order."""
        return {entity_id: self.id_for_index(i) for i, entity_id in enumerate(entity_ids)}


@dataclass(frozen=True, slots=True)
class CorrespondenceEntry:
    """One anatomical entity to geometry component link."""

    entity_id: str
    component_id: str
    kind: GeometryRepresentationKind = GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER
    lod: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_reference(self, *, producer: str | None = None) -> GeometryReference:
        """Build the :class:`~awr.schema.GeometryReference` stored on the entity."""
        return GeometryReference(component_id=self.component_id, kind=self.kind, producer=producer)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "entity_id": self.entity_id,
            "component_id": self.component_id,
            "kind": str(self.kind),
            "lod": self.lod,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            entity_id=str(payload["entity_id"]),
            component_id=str(payload["component_id"]),
            kind=GeometryRepresentationKind(
                payload.get("kind", GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER)
            ),
            lod=payload.get("lod"),
            metadata=dict(payload.get("metadata", {})),
        )


class GeometryCorrespondence:
    """Bidirectional, validated map between entities and geometry components."""

    def __init__(self, entries: Iterable[CorrespondenceEntry] = ()) -> None:
        self._by_component: dict[str, CorrespondenceEntry] = {}
        self._by_entity: dict[str, list[CorrespondenceEntry]] = {}
        for entry in entries:
            self.assign(entry)

    def assign(self, entry: CorrespondenceEntry) -> CorrespondenceEntry:
        """Record a link, rejecting a component already owned by another entity."""
        existing = self._by_component.get(entry.component_id)
        if existing is not None:
            if existing.entity_id != entry.entity_id:
                raise GeometryError(
                    f"Geometry component {entry.component_id!r} is already assigned to "
                    f"{existing.entity_id!r} and cannot be reassigned to {entry.entity_id!r}. "
                    "Component to entity correspondence must stay unambiguous."
                )
            return existing
        self._by_component[entry.component_id] = entry
        self._by_entity.setdefault(entry.entity_id, []).append(entry)
        return entry

    def entity_for(self, component_id: str) -> str:
        """Anatomical entity that owns a component."""
        try:
            return self._by_component[component_id].entity_id
        except KeyError:
            raise GeometryError(
                f"Geometry component {component_id!r} has no anatomical owner. A component "
                "without an entity means part identity has been lost."
            ) from None

    def components_for(self, entity_id: str) -> tuple[str, ...]:
        """Component ids owned by one entity."""
        return tuple(entry.component_id for entry in self._by_entity.get(entity_id, ()))

    def entries(self) -> tuple[CorrespondenceEntry, ...]:
        """All links, in assignment order."""
        return tuple(self._by_component.values())

    def entity_ids(self) -> tuple[str, ...]:
        """Entities that own at least one component."""
        return tuple(self._by_entity)

    def __len__(self) -> int:
        return len(self._by_component)

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._by_component

    def coverage(self, expected_entity_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
        """Compare the map against a set of entities that should be covered."""
        expected = tuple(expected_entity_ids)
        missing = tuple(e for e in expected if not self.components_for(e))
        unexpected = tuple(e for e in self._by_entity if e not in set(expected))
        return {
            "covered": tuple(e for e in expected if self.components_for(e)),
            "missing": missing,
            "unexpected": unexpected,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {"entries": [entry.to_dict() for entry in self.entries()]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeometryCorrespondence:
        """Rebuild from :meth:`to_dict` output."""
        return cls(CorrespondenceEntry.from_dict(e) for e in payload.get("entries", ()))
