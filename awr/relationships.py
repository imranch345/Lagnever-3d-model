"""Typed relationships and the three AWR graphs.

The AWR keeps three *separate* typed graphs rather than one untyped edge list:

* ``STRUCTURE``  - what is part of what, what connects to what.
* ``SPATIAL``    - where things are relative to each other.
* ``FUNCTIONAL`` - what flows or conducts to what.

Every relation name is declared once in :data:`DEFAULT_RELATION_TYPES` together
with the graph it belongs to, whether it is symmetric, its inverse, and whether
it carries physiological flow. Ontology data is validated against that registry,
so an unregistered relation name is a load-time error rather than a string that
quietly means nothing.

Adding a fourth graph kind (for example DEVELOPMENTAL or PATHOLOGICAL) requires
only a new :class:`GraphKind` member and new relation specs; no query code
changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self

from awr.errors import RelationshipError, UnknownRelationTypeError
from awr.schema import unique

__all__ = [
    "GraphKind",
    "FlowSemantics",
    "FlowMedium",
    "Granularity",
    "RelationTypeSpec",
    "RelationRegistry",
    "DEFAULT_RELATION_TYPES",
    "default_relation_registry",
    "Relationship",
    "RelationshipGraph",
    "RelationshipStore",
]


class GraphKind(StrEnum):
    """One of the AWR relationship graphs."""

    STRUCTURE = "structure"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"


class FlowSemantics(StrEnum):
    """Whether a relation carries physiological flow, and in which direction.

    ``FORWARD`` means flow goes subject -> object (``left_ventricle pumps_to
    aorta``). ``REVERSE`` means flow goes object -> subject (``right_atrium
    receives_from superior_vena_cava``). The animation layer uses this to build a
    correctly directed flow graph without hard-coding relation names.
    """

    NONE = "none"
    FORWARD = "forward"
    REVERSE = "reverse"


class FlowMedium(StrEnum):
    """What a flow-carrying relation transports.

    Blood flow and electrical conduction share the functional graph but must never
    be traversed as one path, so the medium is declared per relation type instead
    of being guessed from relation names.
    """

    BLOOD = "blood"
    IMPULSE = "impulse"


class Granularity(StrEnum):
    """Detail level at which a functional edge is valid.

    ``SUMMARY`` edges are chamber-level shortcuts (``right_atrium opens_into
    right_ventricle``); ``DETAILED`` edges route through valves. Selecting one
    granularity prevents a traversal from counting two parallel paths.
    """

    ANY = "any"
    SUMMARY = "summary"
    DETAILED = "detailed"


@dataclass(frozen=True, slots=True)
class RelationTypeSpec:
    """Declaration of one relation type."""

    name: str
    graph: GraphKind
    description: str
    symmetric: bool = False
    inverse: str | None = None
    transitive: bool = False
    flow: FlowSemantics = FlowSemantics.NONE
    medium: FlowMedium | None = None

    def __post_init__(self) -> None:
        if self.symmetric and self.inverse is not None:
            raise RelationshipError(
                f"Relation {self.name!r} cannot be both symmetric and have an inverse."
            )
        if self.flow is FlowSemantics.NONE and self.medium is not None:
            raise RelationshipError(
                f"Relation {self.name!r} declares a medium but carries no flow."
            )
        if self.flow is not FlowSemantics.NONE and self.medium is None:
            raise RelationshipError(
                f"Flow-carrying relation {self.name!r} must declare a medium."
            )


DEFAULT_RELATION_TYPES: tuple[RelationTypeSpec, ...] = (
    # --- structure -------------------------------------------------------
    RelationTypeSpec(
        "part_of",
        GraphKind.STRUCTURE,
        "The subject is anatomically a part of the object.",
        inverse="contains",
        transitive=True,
    ),
    RelationTypeSpec(
        "contains",
        GraphKind.STRUCTURE,
        "The subject anatomically contains the object.",
        inverse="part_of",
        transitive=True,
    ),
    RelationTypeSpec(
        "connects_to",
        GraphKind.STRUCTURE,
        "The two entities are structurally joined at a junction.",
        symmetric=True,
    ),
    RelationTypeSpec(
        "continuous_with",
        GraphKind.STRUCTURE,
        "The two entities form one continuous tissue or lumen.",
        symmetric=True,
    ),
    # --- spatial ---------------------------------------------------------
    RelationTypeSpec(
        "adjacent_to", GraphKind.SPATIAL, "The entities lie next to each other.", symmetric=True
    ),
    RelationTypeSpec(
        "inside", GraphKind.SPATIAL, "The subject lies within the object.", inverse="surrounds"
    ),
    RelationTypeSpec(
        "surrounds", GraphKind.SPATIAL, "The subject encloses the object.", inverse="inside"
    ),
    RelationTypeSpec(
        "anterior_to", GraphKind.SPATIAL, "The subject lies in front of the object.",
        inverse="posterior_to",
    ),
    RelationTypeSpec(
        "posterior_to", GraphKind.SPATIAL, "The subject lies behind the object.",
        inverse="anterior_to",
    ),
    RelationTypeSpec(
        "superior_to", GraphKind.SPATIAL, "The subject lies above the object.",
        inverse="inferior_to",
    ),
    RelationTypeSpec(
        "inferior_to", GraphKind.SPATIAL, "The subject lies below the object.",
        inverse="superior_to",
    ),
    RelationTypeSpec(
        "left_of", GraphKind.SPATIAL, "The subject lies to the anatomical left of the object.",
        inverse="right_of",
    ),
    RelationTypeSpec(
        "right_of", GraphKind.SPATIAL, "The subject lies to the anatomical right of the object.",
        inverse="left_of",
    ),
    # --- functional ------------------------------------------------------
    RelationTypeSpec(
        "receives_from",
        GraphKind.FUNCTIONAL,
        "The subject receives blood from the object.",
        flow=FlowSemantics.REVERSE,
        medium=FlowMedium.BLOOD,
    ),
    RelationTypeSpec(
        "pumps_to",
        GraphKind.FUNCTIONAL,
        "The subject actively pumps blood into the object.",
        flow=FlowSemantics.FORWARD,
        medium=FlowMedium.BLOOD,
    ),
    RelationTypeSpec(
        "opens_into",
        GraphKind.FUNCTIONAL,
        "The subject's lumen opens into the object.",
        flow=FlowSemantics.FORWARD,
        medium=FlowMedium.BLOOD,
    ),
    RelationTypeSpec(
        "exits_into",
        GraphKind.FUNCTIONAL,
        "Flow leaves the subject into the object.",
        flow=FlowSemantics.FORWARD,
        medium=FlowMedium.BLOOD,
    ),
    RelationTypeSpec(
        "conducts_to",
        GraphKind.FUNCTIONAL,
        "The subject conducts the cardiac impulse to the object.",
        flow=FlowSemantics.FORWARD,
        medium=FlowMedium.IMPULSE,
    ),
)


class RelationRegistry:
    """Mutable registry of relation types.

    Instances are cheap and independent: call :func:`default_relation_registry`
    for a fresh copy of the canonical vocabulary and extend that, so no module
    level state is shared between scenes or tests.
    """

    def __init__(self, specs: Iterable[RelationTypeSpec] = ()) -> None:
        self._specs: dict[str, RelationTypeSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: RelationTypeSpec, *, overwrite: bool = False) -> None:
        """Add a relation type. Raises unless ``overwrite`` is set."""
        if spec.name in self._specs and not overwrite:
            raise RelationshipError(f"Relation type {spec.name!r} is already registered.")
        self._specs[spec.name] = spec

    def get(self, name: str) -> RelationTypeSpec:
        """Return the spec for ``name`` or raise :class:`UnknownRelationTypeError`."""
        try:
            return self._specs[name]
        except KeyError:
            raise UnknownRelationTypeError(name, tuple(self._specs)) from None

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[RelationTypeSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def names(self) -> tuple[str, ...]:
        """All registered relation names, sorted."""
        return tuple(sorted(self._specs))

    def by_graph(self, graph: GraphKind) -> tuple[RelationTypeSpec, ...]:
        """All relation types belonging to one graph."""
        return tuple(s for s in self._specs.values() if s.graph is graph)

    def graph_of(self, name: str) -> GraphKind:
        """Graph that relation ``name`` belongs to."""
        return self.get(name).graph

    def copy(self) -> RelationRegistry:
        """Return an independent copy."""
        return RelationRegistry(self._specs.values())

    def validate_inverses(self) -> None:
        """Check that declared inverses exist, are mutual and share a graph."""
        for spec in self._specs.values():
            if spec.inverse is None:
                continue
            other = self.get(spec.inverse)
            if other.inverse != spec.name:
                raise RelationshipError(
                    f"Inverse of {spec.name!r} is {spec.inverse!r} but the inverse of "
                    f"{other.name!r} is {other.inverse!r}; inverses must be mutual."
                )
            if other.graph is not spec.graph:
                raise RelationshipError(
                    f"Relation {spec.name!r} ({spec.graph}) and its inverse {other.name!r} "
                    f"({other.graph}) must belong to the same graph."
                )


def default_relation_registry() -> RelationRegistry:
    """Return a fresh registry holding the canonical Lagnav relation vocabulary."""
    registry = RelationRegistry(DEFAULT_RELATION_TYPES)
    registry.validate_inverses()
    return registry


@dataclass(frozen=True, slots=True)
class Relationship:
    """One typed, directed edge between two entities."""

    subject: str
    relation: str
    object: str
    granularity: Granularity = Granularity.ANY
    note: str | None = None
    derived_from_hierarchy: bool = False
    source: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """Identity of the edge, used for de-duplication."""
        return (self.subject, self.relation, self.object)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "granularity": str(self.granularity),
            "note": self.note,
            "derived_from_hierarchy": self.derived_from_hierarchy,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            subject=str(payload["subject"]),
            relation=str(payload["relation"]),
            object=str(payload["object"]),
            granularity=Granularity(payload.get("granularity", Granularity.ANY)),
            note=payload.get("note"),
            derived_from_hierarchy=bool(payload.get("derived_from_hierarchy", False)),
            source=payload.get("source"),
        )


@dataclass(slots=True)
class RelationshipGraph:
    """All edges of a single graph kind, with symmetry- and inverse-aware queries."""

    kind: GraphKind
    registry: RelationRegistry
    _edges: list[Relationship] = field(default_factory=list, repr=False)
    _keys: set[tuple[str, str, str]] = field(default_factory=set, repr=False)

    def add(self, edge: Relationship) -> bool:
        """Add an edge; return ``False`` if the identical triple already exists."""
        spec = self.registry.get(edge.relation)
        if spec.graph is not self.kind:
            raise RelationshipError(
                f"Relation {edge.relation!r} belongs to the {spec.graph} graph, "
                f"not {self.kind}."
            )
        if edge.subject == edge.object:
            raise RelationshipError(
                f"Self-referential edge {edge.subject!r} --{edge.relation}--> itself is invalid."
            )
        if edge.key in self._keys:
            return False
        self._edges.append(edge)
        self._keys.add(edge.key)
        return True

    @property
    def edges(self) -> tuple[Relationship, ...]:
        """All stored edges in insertion order."""
        return tuple(self._edges)

    def __len__(self) -> int:
        return len(self._edges)

    def __iter__(self) -> Iterator[Relationship]:
        return iter(self._edges)

    def outgoing(self, subject: str, relation: str | None = None) -> tuple[Relationship, ...]:
        """Stored edges whose subject is ``subject`` (no inverse resolution)."""
        return tuple(
            e
            for e in self._edges
            if e.subject == subject and (relation is None or e.relation == relation)
        )

    def incoming(self, obj: str, relation: str | None = None) -> tuple[Relationship, ...]:
        """Stored edges whose object is ``obj`` (no inverse resolution)."""
        return tuple(
            e
            for e in self._edges
            if e.object == obj and (relation is None or e.relation == relation)
        )

    def related(
        self,
        subject: str,
        relation: str,
        *,
        granularity: Granularity | None = None,
    ) -> tuple[str, ...]:
        """Entities reached from ``subject`` by ``relation``.

        Resolves symmetric relations in both directions and declared inverse
        relations from stored edges, so ``related(rv, "inferior_to")`` answers
        from a stored ``ra superior_to rv`` edge without duplicating data.
        """
        spec = self.registry.get(relation)
        found: list[str] = []

        def _accept(edge: Relationship) -> bool:
            return granularity is None or edge.granularity in (granularity, Granularity.ANY)

        for edge in self._edges:
            if not _accept(edge):
                continue
            if edge.relation == relation and edge.subject == subject:
                found.append(edge.object)
            elif spec.symmetric and edge.relation == relation and edge.object == subject:
                found.append(edge.subject)
            elif spec.inverse and edge.relation == spec.inverse and edge.object == subject:
                found.append(edge.subject)
        return unique(found)  # deterministic order, no duplicates

    def incident(self, entity: str) -> tuple[Relationship, ...]:
        """Every stored edge that mentions ``entity`` in either position."""
        return tuple(e for e in self._edges if entity in (e.subject, e.object))

    def entity_ids(self) -> set[str]:
        """All entity ids mentioned by this graph."""
        ids: set[str] = set()
        for edge in self._edges:
            ids.add(edge.subject)
            ids.add(edge.object)
        return ids


class RelationshipStore:
    """The three (or more) AWR graphs, kept separate and queried by kind."""

    def __init__(self, registry: RelationRegistry | None = None) -> None:
        self.registry = registry or default_relation_registry()
        self._graphs: dict[GraphKind, RelationshipGraph] = {
            kind: RelationshipGraph(kind, self.registry) for kind in GraphKind
        }

    def graph(self, kind: GraphKind) -> RelationshipGraph:
        """Return one graph by kind."""
        return self._graphs[kind]

    @property
    def kinds(self) -> tuple[GraphKind, ...]:
        """Graph kinds held by this store."""
        return tuple(self._graphs)

    @property
    def structure(self) -> RelationshipGraph:
        """The structure graph."""
        return self._graphs[GraphKind.STRUCTURE]

    @property
    def spatial(self) -> RelationshipGraph:
        """The spatial graph."""
        return self._graphs[GraphKind.SPATIAL]

    @property
    def functional(self) -> RelationshipGraph:
        """The functional graph."""
        return self._graphs[GraphKind.FUNCTIONAL]

    def add(self, edge: Relationship) -> bool:
        """Route an edge into the graph its relation type belongs to."""
        kind = self.registry.graph_of(edge.relation)
        if kind not in self._graphs:
            self._graphs[kind] = RelationshipGraph(kind, self.registry)
        return self._graphs[kind].add(edge)

    def add_many(self, edges: Iterable[Relationship]) -> int:
        """Add several edges; return how many were newly inserted."""
        return sum(1 for edge in edges if self.add(edge))

    def related(
        self, subject: str, relation: str, *, granularity: Granularity | None = None
    ) -> tuple[str, ...]:
        """Query ``related`` on whichever graph owns ``relation``."""
        kind = self.registry.graph_of(relation)
        return self._graphs[kind].related(subject, relation, granularity=granularity)

    def incident(self, entity: str) -> tuple[Relationship, ...]:
        """Every edge mentioning ``entity``, across all graphs."""
        return tuple(e for graph in self._graphs.values() for e in graph.incident(entity))

    def all_edges(self) -> tuple[Relationship, ...]:
        """Every edge across all graphs, grouped by graph kind."""
        return tuple(e for graph in self._graphs.values() for e in graph.edges)

    def counts(self) -> dict[str, int]:
        """Edge count per graph kind, for reporting and tests."""
        return {str(kind): len(graph) for kind, graph in self._graphs.items()}

    def __len__(self) -> int:
        return sum(len(g) for g in self._graphs.values())

    def to_list(self) -> list[dict[str, Any]]:
        """Serialise every edge to JSON-compatible dictionaries."""
        return [edge.to_dict() for edge in self.all_edges()]

    @classmethod
    def from_list(
        cls, payload: Sequence[Mapping[str, Any]], registry: RelationRegistry | None = None
    ) -> RelationshipStore:
        """Rebuild a store from :meth:`to_list` output."""
        store = cls(registry)
        store.add_many(Relationship.from_dict(item) for item in payload)
        return store
