"""Typed loader for versioned anatomical ontology data.

An ontology is three JSON files that must agree on their version:

* ``anatomy.json``       - the entity catalogue (identity, semantics, LOD policy).
* ``hierarchy.json``     - the single-parent organisational tree.
* ``relationships.json`` - typed edges for the structure, spatial and functional graphs.

The loader is strict on purpose. Anything that would let a malformed ontology
through - an unknown ``anatomy_type``, an unregistered relation, an edge pointing
at a missing entity, two parents for one entity, a cycle - raises
:class:`~awr.errors.OntologyError` with the offending ids named.

``part_of`` / ``contains`` edges for the organisational tree are *derived* from
``hierarchy.json`` rather than repeated in ``relationships.json``, so the tree and
the structure graph cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awr.entities import AnatomicalEntity
from awr.errors import EntityNotFoundError, OntologyError, SchemaError
from awr.relationships import (
    Granularity,
    GraphKind,
    RelationRegistry,
    Relationship,
    RelationshipStore,
    default_relation_registry,
)
from awr.schema import AnatomyType, Laterality, LodLevel, LodPolicy, coerce_enum

__all__ = ["OntologyProvenance", "AnatomyOntology", "load_ontology"]

_REQUIRED_FILES = ("anatomy.json", "hierarchy.json", "relationships.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OntologyError(f"Ontology file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OntologyError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise OntologyError(f"{path.name} must contain a JSON object at the top level.")
    return payload


@dataclass(frozen=True, slots=True)
class OntologyProvenance:
    """Where an ontology came from and whether it has been reviewed."""

    status: str
    medically_validated: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_draft(self) -> bool:
        """Whether the ontology is still a draft (v0.1 always is)."""
        return not self.medically_validated


@dataclass(slots=True)
class AnatomyOntology:
    """An immutable, validated anatomical ontology."""

    ontology_id: str
    title: str
    version: str
    root_id: str
    entities: Mapping[str, AnatomicalEntity]
    parent_of: Mapping[str, str | None]
    children_of: Mapping[str, tuple[str, ...]]
    relationships: RelationshipStore
    registry: RelationRegistry
    provenance: OntologyProvenance
    source_dir: Path | None = None

    # --- lookup -----------------------------------------------------------
    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self.entities

    def __len__(self) -> int:
        return len(self.entities)

    def get(self, entity_id: str) -> AnatomicalEntity:
        """Return an entity definition or raise :class:`EntityNotFoundError`."""
        try:
            return self.entities[entity_id]
        except KeyError:
            raise EntityNotFoundError(entity_id, available=len(self.entities)) from None

    def ids(self) -> tuple[str, ...]:
        """All entity ids in declaration order."""
        return tuple(self.entities)

    def iter_entities(self) -> Iterator[AnatomicalEntity]:
        """Iterate entity definitions in declaration order."""
        return iter(self.entities.values())

    # --- tree -------------------------------------------------------------
    def parent(self, entity_id: str) -> str | None:
        """Parent id, or ``None`` for the root."""
        self.get(entity_id)
        return self.parent_of.get(entity_id)

    def children(self, entity_id: str) -> tuple[str, ...]:
        """Direct children in declaration order."""
        self.get(entity_id)
        return self.children_of.get(entity_id, ())

    def descendants(self, entity_id: str, *, include_self: bool = False) -> tuple[str, ...]:
        """All descendants, depth first, in declaration order."""
        out: list[str] = [entity_id] if include_self else []
        stack = list(reversed(self.children(entity_id)))
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self.children(current)))
        return tuple(out)

    def ancestors(self, entity_id: str) -> tuple[str, ...]:
        """Ancestor ids from the direct parent up to the root."""
        out: list[str] = []
        current = self.parent(entity_id)
        while current is not None:
            out.append(current)
            current = self.parent_of.get(current)
        return tuple(out)

    def renderable_descendants(
        self, entity_id: str, *, include_self: bool = True
    ) -> tuple[str, ...]:
        """Renderable entities at or under ``entity_id``.

        This is how a group reference such as "the valves" expands to the things
        that can actually be shown or hidden.
        """
        candidates = self.descendants(entity_id, include_self=include_self)
        return tuple(cid for cid in candidates if self.get(cid).renderable)

    # --- selections -------------------------------------------------------
    def groups(self) -> tuple[str, ...]:
        """Ids of organisational group entities."""
        return tuple(e.entity_id for e in self.iter_entities() if e.is_group)

    def by_type(self, anatomy_type: AnatomyType) -> tuple[str, ...]:
        """Ids of every entity of one anatomy type."""
        return tuple(e.entity_id for e in self.iter_entities() if e.anatomy_type is anatomy_type)

    def max_lod(self) -> LodLevel:
        """Highest ``min_lod`` declared by any entity."""
        return max(e.lod_policy.min_lod for e in self.iter_entities())

    def graph(self, kind: GraphKind) -> Any:
        """Shortcut to one relationship graph."""
        return self.relationships.graph(kind)

    def summary(self) -> dict[str, Any]:
        """Compact description used by reports, tests and the demo script."""
        by_type: dict[str, int] = {}
        for entity in self.iter_entities():
            by_type[str(entity.anatomy_type)] = by_type.get(str(entity.anatomy_type), 0) + 1
        return {
            "ontology_id": self.ontology_id,
            "title": self.title,
            "version": self.version,
            "status": self.provenance.status,
            "medically_validated": self.provenance.medically_validated,
            "entity_count": len(self.entities),
            "group_count": len(self.groups()),
            "entities_by_type": by_type,
            "relationship_counts": self.relationships.counts(),
            "max_lod": self.max_lod(),
        }


def _build_entity(raw: Mapping[str, Any], *, index: int) -> AnatomicalEntity:
    try:
        entity_id = str(raw["id"])
        canonical = str(raw["canonical_name"])
    except KeyError as exc:
        raise OntologyError(
            f"anatomy.json entity #{index} is missing required field {exc.args[0]!r}."
        ) from exc
    try:
        anatomy_type = coerce_enum(AnatomyType, str(raw["anatomy_type"]), field_name="anatomy_type")
        laterality = coerce_enum(
            Laterality, str(raw.get("laterality", "none")), field_name="laterality"
        )
    except (SchemaError, KeyError) as exc:
        raise OntologyError(f"Entity {entity_id!r}: {exc}") from exc

    metadata = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "id",
            "canonical_name",
            "display_name",
            "anatomy_type",
            "semantic_role",
            "laterality",
            "synonyms",
            "abbreviations",
            "is_group",
            "renderable",
            "min_lod",
            "max_lod",
            "notes",
            "source_refs",
        }
    }
    return AnatomicalEntity(
        entity_id=entity_id,
        canonical_name=canonical,
        display_name=str(raw.get("display_name", canonical.replace("_", " ").capitalize())),
        anatomy_type=anatomy_type,
        semantic_role=str(raw.get("semantic_role", "")),
        lod_policy=LodPolicy(min_lod=int(raw.get("min_lod", 0)), max_lod=raw.get("max_lod")),
        laterality=laterality,
        synonyms=tuple(str(s) for s in raw.get("synonyms", ())),
        abbreviations=tuple(str(s) for s in raw.get("abbreviations", ())),
        is_group=bool(raw.get("is_group", False)),
        renderable=bool(raw.get("renderable", True)),
        notes=raw.get("notes"),
        source_refs=tuple(str(s) for s in raw.get("source_refs", ())),
        metadata=metadata,
    )


def _build_tree(
    hierarchy: Mapping[str, Any], entities: Mapping[str, AnatomicalEntity]
) -> tuple[str, dict[str, str | None], dict[str, tuple[str, ...]]]:
    root_id = str(hierarchy.get("root", ""))
    if root_id not in entities:
        raise OntologyError(
            f"hierarchy.json declares root {root_id!r}, which is not defined in anatomy.json."
        )
    nodes: Sequence[Mapping[str, Any]] = hierarchy.get("nodes", ())
    parent_of: dict[str, str | None] = {root_id: None}
    children_of: dict[str, tuple[str, ...]] = {}

    for node in nodes:
        parent = str(node["parent"])
        if parent not in entities:
            raise OntologyError(f"hierarchy.json refers to unknown parent {parent!r}.")
        if parent in children_of:
            raise OntologyError(f"hierarchy.json declares parent {parent!r} more than once.")
        children: list[str] = []
        for raw_child in node.get("children", ()):
            child = str(raw_child)
            if child not in entities:
                raise OntologyError(
                    f"hierarchy.json refers to unknown child {child!r} under {parent!r}."
                )
            if child in parent_of:
                raise OntologyError(
                    f"Entity {child!r} has more than one parent "
                    f"({parent_of[child]!r} and {parent!r}); the tree must be single-parent."
                )
            if child == parent:
                raise OntologyError(f"Entity {parent!r} cannot be its own parent.")
            parent_of[child] = parent
            children.append(child)
        children_of[parent] = tuple(children)

    unplaced = sorted(set(entities) - set(parent_of))
    if unplaced:
        raise OntologyError(
            "Every entity must appear in the hierarchy. Unplaced entities: "
            f"{', '.join(unplaced)}."
        )

    # Cycle / reachability check: walking up from every node must reach the root.
    for entity_id in entities:
        seen: set[str] = set()
        current: str | None = entity_id
        while current is not None:
            if current in seen:
                raise OntologyError(
                    f"Hierarchy cycle detected while walking up from {entity_id!r}: "
                    f"{' -> '.join(sorted(seen))}."
                )
            seen.add(current)
            current = parent_of.get(current)
        if root_id not in seen:
            raise OntologyError(f"Entity {entity_id!r} is not connected to root {root_id!r}.")

    return root_id, parent_of, children_of


def _build_relationships(
    payload: Mapping[str, Any],
    entities: Mapping[str, AnatomicalEntity],
    parent_of: Mapping[str, str | None],
    registry: RelationRegistry,
) -> RelationshipStore:
    store = RelationshipStore(registry)

    # Derived organisational partonomy: keeps tree and structure graph in sync.
    for child, parent in parent_of.items():
        if parent is None:
            continue
        store.add(
            Relationship(
                subject=child,
                relation="part_of",
                object=parent,
                derived_from_hierarchy=True,
                source="hierarchy.json",
            )
        )

    for index, raw in enumerate(payload.get("edges", ())):
        try:
            subject = str(raw["subject"])
            relation = str(raw["relation"])
            obj = str(raw["object"])
        except KeyError as exc:
            raise OntologyError(
                f"relationships.json edge #{index} is missing field {exc.args[0]!r}."
            ) from exc
        for role, value in (("subject", subject), ("object", obj)):
            if value not in entities:
                raise OntologyError(
                    f"relationships.json edge #{index} ({subject} {relation} {obj}) has an "
                    f"unknown {role} {value!r}."
                )
        if relation not in registry:
            raise OntologyError(
                f"relationships.json edge #{index} uses unregistered relation {relation!r}. "
                f"Registered relations: {', '.join(registry.names())}."
            )
        try:
            granularity = Granularity(str(raw.get("granularity", Granularity.ANY)))
        except ValueError as exc:
            raise OntologyError(
                f"relationships.json edge #{index} has invalid granularity "
                f"{raw.get('granularity')!r}."
            ) from exc
        store.add(
            Relationship(
                subject=subject,
                relation=relation,
                object=obj,
                granularity=granularity,
                note=raw.get("note"),
                source="relationships.json",
            )
        )
    return store


def load_ontology(
    directory: str | Path,
    *,
    expected_version: str | None = None,
    registry: RelationRegistry | None = None,
) -> AnatomyOntology:
    """Load and validate an ontology from a directory of three JSON files.

    Args:
        directory: Folder containing ``anatomy.json``, ``hierarchy.json`` and
            ``relationships.json``.
        expected_version: If given, every file must declare this version.
        registry: Relation vocabulary to validate against. Defaults to the
            canonical Lagnav registry.

    Raises:
        OntologyError: If any file is missing, malformed, version-mismatched or
            internally inconsistent.

    """
    base = Path(directory)
    payloads = {name: _read_json(base / name) for name in _REQUIRED_FILES}

    versions = {name: str(payload.get("version", "")) for name, payload in payloads.items()}
    if len(set(versions.values())) != 1:
        raise OntologyError(
            "Ontology files disagree on version: "
            + ", ".join(f"{name}={version!r}" for name, version in versions.items())
        )
    version = next(iter(versions.values()))
    if expected_version is not None and version != expected_version:
        raise OntologyError(
            f"Ontology version mismatch: configuration expects {expected_version!r} but "
            f"{base} declares {version!r}."
        )

    anatomy = payloads["anatomy.json"]
    entities: dict[str, AnatomicalEntity] = {}
    for index, raw in enumerate(anatomy.get("entities", ())):
        entity = _build_entity(raw, index=index)
        if entity.entity_id in entities:
            raise OntologyError(f"Duplicate entity id {entity.entity_id!r} in anatomy.json.")
        entities[entity.entity_id] = entity
    if not entities:
        raise OntologyError(f"{base / 'anatomy.json'} declares no entities.")

    ontology_ids = {str(payload.get("ontology_id", "")) for payload in payloads.values()}
    if len(ontology_ids) != 1:
        raise OntologyError(f"Ontology files disagree on ontology_id: {sorted(ontology_ids)}.")

    active_registry = registry or default_relation_registry()
    root_id, parent_of, children_of = _build_tree(payloads["hierarchy.json"], entities)
    relationships = _build_relationships(
        payloads["relationships.json"], entities, parent_of, active_registry
    )

    return AnatomyOntology(
        ontology_id=next(iter(ontology_ids)),
        title=str(anatomy.get("title", next(iter(ontology_ids)))),
        version=version,
        root_id=root_id,
        entities=entities,
        parent_of=parent_of,
        children_of=children_of,
        relationships=relationships,
        registry=active_registry,
        provenance=OntologyProvenance(
            status=str(anatomy.get("status", "unknown")),
            medically_validated=bool(anatomy.get("medically_validated", False)),
            details=dict(anatomy.get("provenance", {})),
        ),
        source_dir=base,
    )
