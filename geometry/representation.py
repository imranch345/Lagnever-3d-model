"""3D representation interfaces.

Nothing in this module generates geometry. That is deliberate: the Lagnav
geometry model does not exist yet, and inventing a mesh generator now would make
visual output the source of anatomical truth, which the project explicitly
rejects.

What does exist is the *contract*:

* :class:`Geometry3DRepresentation` - the abstract surface every future
  representation must expose (mesh, implicit field, point cloud, splats).
* :class:`GeometryComponent` - one addressable piece of geometry, permanently
  tagged with the anatomical entity it belongs to.
* :class:`SymbolicGeometry` - the only concrete implementation in this
  milestone. It holds component *slots* with stable ids and no geometric data,
  so that entity-to-geometry correspondence can be built, persisted and tested
  long before any decoder exists.

Declared-but-unimplemented representations raise
:class:`~awr.errors.NotYetImplementedError` rather than silently returning
something plausible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from awr.errors import GeometryError, NotYetImplementedError
from awr.schema import GeometryRepresentationKind

__all__ = [
    "GeometryComponent",
    "Geometry3DRepresentation",
    "ThreeDRepresentation",
    "SymbolicGeometry",
    "MeshRepresentation",
    "ImplicitFieldRepresentation",
]


@dataclass(frozen=True, slots=True)
class GeometryComponent:
    """One addressable geometry component bound to one anatomical entity.

    The binding is part of the component's identity. A future decoder may fill
    in ``uri`` and payload statistics, but it may never change ``entity_id``:
    that is what stops a finished mesh from losing semantic part identity.
    """

    component_id: str
    entity_id: str
    kind: GeometryRepresentationKind = GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER
    uri: str | None = None
    checksum: str | None = None
    lod: int | None = None
    stats: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id:
            raise GeometryError("GeometryComponent.component_id must be non-empty.")
        if not self.entity_id:
            raise GeometryError(
                f"Component {self.component_id!r} must name the anatomical entity it belongs to."
            )

    @property
    def has_payload(self) -> bool:
        """Whether real geometry data backs this component."""
        return self.kind is not GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER and bool(self.uri)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "component_id": self.component_id,
            "entity_id": self.entity_id,
            "kind": str(self.kind),
            "uri": self.uri,
            "checksum": self.checksum,
            "lod": self.lod,
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            component_id=str(payload["component_id"]),
            entity_id=str(payload["entity_id"]),
            kind=GeometryRepresentationKind(
                payload.get("kind", GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER)
            ),
            uri=payload.get("uri"),
            checksum=payload.get("checksum"),
            lod=payload.get("lod"),
            stats=dict(payload.get("stats", {})),
        )


class Geometry3DRepresentation(ABC):
    """Abstract 3D representation of a structured anatomical scene.

    Implementations must be *part-addressable*: a caller can always ask which
    components exist and which anatomical entity each belongs to. A
    representation that can only answer "here is one mesh" does not satisfy this
    interface and cannot be used by Lagnav.
    """

    @property
    @abstractmethod
    def kind(self) -> GeometryRepresentationKind:
        """Representation kind."""

    @abstractmethod
    def components(self) -> tuple[GeometryComponent, ...]:
        """All components, in stable order."""

    @abstractmethod
    def component(self, component_id: str) -> GeometryComponent:
        """Return one component by id."""

    def component_ids(self) -> tuple[str, ...]:
        """Ids of all components."""
        return tuple(component.component_id for component in self.components())

    def entity_ids(self) -> tuple[str, ...]:
        """Distinct anatomical entities represented, in stable order."""
        seen: set[str] = set()
        out: list[str] = []
        for component in self.components():
            if component.entity_id not in seen:
                seen.add(component.entity_id)
                out.append(component.entity_id)
        return tuple(out)

    def components_for(self, entity_id: str) -> tuple[GeometryComponent, ...]:
        """All components belonging to one anatomical entity."""
        return tuple(c for c in self.components() if c.entity_id == entity_id)

    def stats(self) -> dict[str, Any]:
        """Summary statistics, for reports and evaluation."""
        return {
            "kind": str(self.kind),
            "component_count": len(self.components()),
            "entity_count": len(self.entity_ids()),
            "components_with_payload": sum(1 for c in self.components() if c.has_payload),
        }

    def __len__(self) -> int:
        return len(self.components())

    def __iter__(self) -> Iterator[GeometryComponent]:
        return iter(self.components())


ThreeDRepresentation = Geometry3DRepresentation
"""Readable alias for :class:`Geometry3DRepresentation` (a class name cannot start with a digit)."""


class SymbolicGeometry(Geometry3DRepresentation):
    """Component slots with stable ids and no geometric payload.

    This is what the deterministic prototype produces. It exists so that the
    anatomy-to-geometry correspondence is real, persisted and testable today,
    while the representation that eventually carries vertices is left entirely
    to Step 5.
    """

    def __init__(self, components: Iterable[GeometryComponent] = ()) -> None:
        self._components: dict[str, GeometryComponent] = {}
        for component in components:
            self.add(component)

    @property
    def kind(self) -> GeometryRepresentationKind:
        """Always :attr:`GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER`."""
        return GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER

    def add(self, component: GeometryComponent) -> GeometryComponent:
        """Register a component slot; duplicate ids are rejected."""
        if component.component_id in self._components:
            raise GeometryError(
                f"Component id {component.component_id!r} is already registered "
                f"to {self._components[component.component_id].entity_id!r}."
            )
        self._components[component.component_id] = component
        return component

    def components(self) -> tuple[GeometryComponent, ...]:
        """All component slots in insertion order."""
        return tuple(self._components.values())

    def component(self, component_id: str) -> GeometryComponent:
        """Return one component slot by id."""
        try:
            return self._components[component_id]
        except KeyError:
            raise GeometryError(f"Unknown geometry component {component_id!r}.") from None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "kind": str(self.kind),
            "components": [component.to_dict() for component in self.components()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SymbolicGeometry:
        """Rebuild from :meth:`to_dict` output."""
        return cls(GeometryComponent.from_dict(c) for c in payload.get("components", ()))


class MeshRepresentation(Geometry3DRepresentation):
    """Triangle-mesh representation. Declared, not implemented.

    TODO(step-5): Step 5 decides whether meshes are the primary output of the
    geometry decoder or a conversion target from an implicit representation, and
    defines per-part topology guarantees, UV handling and units.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError("MeshRepresentation", planned_in="Step 5 and later")

    @property
    def kind(self) -> GeometryRepresentationKind:  # pragma: no cover - unreachable
        """Unreachable: construction always raises."""
        return GeometryRepresentationKind.MESH

    def components(self) -> tuple[GeometryComponent, ...]:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("MeshRepresentation.components")

    def component(self, component_id: str) -> GeometryComponent:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("MeshRepresentation.component")


class ImplicitFieldRepresentation(Geometry3DRepresentation):
    """Implicit / neural-field representation. Declared, not implemented.

    TODO(step-5): Step 5 decides the field parameterisation, how per-entity part
    identity is preserved inside a shared field, and the extraction strategy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError("ImplicitFieldRepresentation", planned_in="Step 5 and later")

    @property
    def kind(self) -> GeometryRepresentationKind:  # pragma: no cover - unreachable
        """Unreachable: construction always raises."""
        return GeometryRepresentationKind.IMPLICIT_FIELD

    def components(self) -> tuple[GeometryComponent, ...]:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("ImplicitFieldRepresentation.components")

    def component(self, component_id: str) -> GeometryComponent:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("ImplicitFieldRepresentation.component")
