"""The AWR scene: persistent anatomical memory.

A scene is the single source of truth for "what exists and what state is it in".
It is not a picture and it is not a mesh. Geometry, when it eventually exists,
hangs off entities by reference; deleting the geometry would not delete the
anatomy.

Key invariants, all covered by tests:

* An entity id is assigned once, at construction, and never changes. Edits touch
  :class:`~awr.schema.EntityState` only.
* Every ontology entity is present at every LOD. The LOD decides visibility, not
  existence.
* Every state change is committed as a :class:`SceneEvent` carrying explicit
  before/after deltas, so scene history is an auditable log rather than a diff
  of two pictures.
* The scene round-trips through JSON without losing identity, state or history.

The low-level ``set_*`` methods here are the primitives that editing operations
compose; they do not write history by themselves. Application code should go
through ``editing.scene_editor`` or the command engine instead.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from awr.entities import SceneEntity
from awr.errors import EntityNotFoundError, SceneError, SchemaError
from awr.lod import EducationalLevelMap, LodLadder, LodLevelSpec, default_visibility_at
from awr.ontology import AnatomyOntology
from awr.relationships import RelationshipStore
from awr.schema import (
    AWR_SCHEMA_VERSION,
    AnimationState,
    EducationalLevel,
    EntityState,
    GeometryReference,
    LodLevel,
    MaterialState,
    Transform,
    VisibilitySource,
    coerce_enum,
)

__all__ = ["EntityDelta", "SceneEvent", "AWRScene"]


@dataclass(frozen=True, slots=True)
class EntityDelta:
    """One recorded field change on one entity."""

    entity_id: str
    field_name: str
    before: Any
    after: Any

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"{self.entity_id}.{self.field_name}: {self.before!r} -> {self.after!r}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "entity_id": self.entity_id,
            "field": self.field_name,
            "before": self.before,
            "after": self.after,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            entity_id=str(payload["entity_id"]),
            field_name=str(payload["field"]),
            before=payload.get("before"),
            after=payload.get("after"),
        )


@dataclass(frozen=True, slots=True)
class SceneEvent:
    """One committed change-set in scene history."""

    version: int
    operation: str
    summary: str
    deltas: tuple[EntityDelta, ...] = ()
    command_text: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @property
    def touched_entities(self) -> tuple[str, ...]:
        """Entity ids affected by this event, in first-touched order."""
        seen: set[str] = set()
        out: list[str] = []
        for delta in self.deltas:
            if delta.entity_id not in seen:
                seen.add(delta.entity_id)
                out.append(delta.entity_id)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "version": self.version,
            "operation": self.operation,
            "summary": self.summary,
            "deltas": [d.to_dict() for d in self.deltas],
            "command_text": self.command_text,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            version=int(payload["version"]),
            operation=str(payload["operation"]),
            summary=str(payload.get("summary", "")),
            deltas=tuple(EntityDelta.from_dict(d) for d in payload.get("deltas", ())),
            command_text=payload.get("command_text"),
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(slots=True)
class AWRScene:
    """A persistent Anatomical World Representation scene."""

    scene_id: str
    name: str
    domain_id: str
    ontology_id: str
    ontology_version: str
    entities: dict[str, SceneEntity]
    relationships: RelationshipStore
    root_id: str
    lod_ladder: LodLadder
    educational_levels: EducationalLevelMap
    active_lod: LodLevel
    educational_level: EducationalLevel
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    history: list[SceneEvent] = field(default_factory=list)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_ontology(
        cls,
        ontology: AnatomyOntology,
        *,
        lod_ladder: LodLadder,
        educational_levels: EducationalLevelMap,
        active_lod: LodLevel,
        educational_level: EducationalLevel,
        name: str,
        domain_id: str,
        default_opacity: float = 1.0,
        scene_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AWRScene:
        """Instantiate every ontology entity into a fresh scene.

        All entities are created regardless of LOD; only their default
        visibility depends on ``active_lod``. Ids come straight from the
        ontology, which is what makes them stable across sessions.
        """
        lod_ladder.validate(active_lod)
        entities: dict[str, SceneEntity] = {}
        for definition in ontology.iter_entities():
            visible = default_visibility_at(
                renderable=definition.renderable,
                min_lod=definition.lod_policy.min_lod,
                max_lod=definition.lod_policy.max_lod,
                lod=active_lod,
            )
            entities[definition.entity_id] = SceneEntity(
                definition=definition,
                parent_id=ontology.parent_of.get(definition.entity_id),
                children=ontology.children_of.get(definition.entity_id, ()),
                state=EntityState(
                    visibility=visible,
                    opacity=default_opacity,
                    visibility_source=VisibilitySource.LOD_DEFAULT,
                ),
            )
        resolved_id = scene_id or cls._derive_scene_id(
            ontology.ontology_id, ontology.version, name, active_lod, educational_level
        )
        return cls(
            scene_id=resolved_id,
            name=name,
            domain_id=domain_id,
            ontology_id=ontology.ontology_id,
            ontology_version=ontology.version,
            entities=entities,
            relationships=ontology.relationships,
            root_id=ontology.root_id,
            lod_ladder=lod_ladder,
            educational_levels=educational_levels,
            active_lod=active_lod,
            educational_level=educational_level,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _derive_scene_id(
        ontology_id: str,
        ontology_version: str,
        name: str,
        lod: LodLevel,
        educational_level: EducationalLevel,
    ) -> str:
        """Deterministic scene id.

        The prototype must be reproducible, so ids are content-derived rather
        than random: the same request in the same configuration always produces
        the same scene id.
        """
        seed = f"{ontology_id}|{ontology_version}|{name}|{lod}|{educational_level}"
        digest = hashlib.blake2s(seed.encode("utf-8"), digest_size=4).hexdigest()
        return f"scene.{ontology_id.split('.')[-1]}.{digest}"

    # ------------------------------------------------------------------
    # lookup and traversal
    # ------------------------------------------------------------------
    def __contains__(self, entity_id: object) -> bool:
        return entity_id in self.entities

    def __len__(self) -> int:
        return len(self.entities)

    def get(self, entity_id: str) -> SceneEntity:
        """Return a scene entity or raise :class:`EntityNotFoundError`."""
        try:
            return self.entities[entity_id]
        except KeyError:
            raise EntityNotFoundError(entity_id, available=len(self.entities)) from None

    def ids(self) -> tuple[str, ...]:
        """All entity ids in ontology declaration order."""
        return tuple(self.entities)

    def iter_entities(self) -> Iterator[SceneEntity]:
        """Iterate scene entities in ontology declaration order."""
        return iter(self.entities.values())

    def children(self, entity_id: str) -> tuple[str, ...]:
        """Direct children of an entity."""
        return self.get(entity_id).children

    def parent(self, entity_id: str) -> str | None:
        """Parent id, or ``None`` for the root."""
        return self.get(entity_id).parent_id

    def descendants(self, entity_id: str, *, include_self: bool = False) -> tuple[str, ...]:
        """Depth-first descendants."""
        out: list[str] = [entity_id] if include_self else []
        stack = list(reversed(self.children(entity_id)))
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(reversed(self.children(current)))
        return tuple(out)

    def ancestors(self, entity_id: str) -> tuple[str, ...]:
        """Ancestors from parent to root."""
        out: list[str] = []
        current = self.parent(entity_id)
        while current is not None:
            out.append(current)
            current = self.parent(current)
        return tuple(out)

    def expand(self, entity_ids: Iterable[str], *, renderable_only: bool = True) -> tuple[str, ...]:
        """Expand ids to the entities an operation should actually touch.

        A group id expands to its renderable descendants; a renderable entity
        expands to itself. This is what makes "show the valves" act on the four
        valves rather than on an abstract container.
        """
        out: list[str] = []
        seen: set[str] = set()
        for entity_id in entity_ids:
            entity = self.get(entity_id)
            candidates = (
                self.descendants(entity_id, include_self=True)
                if entity.is_group
                else (entity_id,)
            )
            for candidate in candidates:
                if renderable_only and not self.get(candidate).renderable:
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    out.append(candidate)
        return tuple(out)

    def renderable_ids(self) -> tuple[str, ...]:
        """Every entity that can be drawn."""
        return tuple(e.entity_id for e in self.iter_entities() if e.renderable)

    def visible_ids(self) -> tuple[str, ...]:
        """Every currently visible entity id."""
        return tuple(e.entity_id for e in self.iter_entities() if e.visibility)

    def default_visibility(self, entity_id: str, lod: LodLevel | None = None) -> bool:
        """Visibility the LOD policy would give an entity at ``lod``."""
        entity = self.get(entity_id)
        return default_visibility_at(
            renderable=entity.renderable,
            min_lod=entity.lod_policy.min_lod,
            max_lod=entity.lod_policy.max_lod,
            lod=self.active_lod if lod is None else lod,
        )

    def lod_spec(self) -> LodLevelSpec:
        """Spec of the currently active LOD."""
        return self.lod_ladder.spec(self.active_lod)

    # ------------------------------------------------------------------
    # state primitives (used by editing operations; no history written here)
    # ------------------------------------------------------------------
    def set_entity_visibility(
        self, entity_id: str, value: bool, *, source: VisibilitySource
    ) -> EntityDelta | None:
        """Set visibility. Returns ``None`` when nothing actually changed."""
        entity = self.get(entity_id)
        if not entity.renderable and value:
            raise SceneError(
                f"{entity_id!r} is an organisational group and cannot be made visible itself; "
                "expand it to its renderable descendants first."
            )
        if entity.state.visibility == value and entity.state.visibility_source is source:
            return None
        before = entity.state.visibility
        entity.state.visibility = value
        entity.state.visibility_source = source
        if before == value:
            return None
        return EntityDelta(entity_id, "visibility", before, value)

    def set_entity_opacity(self, entity_id: str, value: float) -> EntityDelta | None:
        """Set opacity in ``[0, 1]``. Returns ``None`` when unchanged."""
        entity = self.get(entity_id)
        before = entity.state.opacity
        if before == value:
            return None
        entity.state.opacity = float(value)
        entity.state.validate()
        return EntityDelta(entity_id, "opacity", before, float(value))

    def set_entity_material(self, entity_id: str, material: MaterialState) -> EntityDelta | None:
        """Replace the material state."""
        entity = self.get(entity_id)
        before = entity.state.material
        if before == material:
            return None
        entity.state.material = material
        return EntityDelta(entity_id, "material", before.to_dict(), material.to_dict())

    def set_entity_transform(self, entity_id: str, transform: Transform) -> EntityDelta | None:
        """Replace the transform."""
        entity = self.get(entity_id)
        before = entity.state.transform
        if before == transform:
            return None
        entity.state.transform = transform
        return EntityDelta(entity_id, "transform", before.to_dict(), transform.to_dict())

    def set_entity_animation(self, entity_id: str, animation: AnimationState) -> EntityDelta | None:
        """Replace the animation binding."""
        entity = self.get(entity_id)
        before = entity.state.animation
        if before == animation:
            return None
        entity.state.animation = animation
        return EntityDelta(entity_id, "animation", before.to_dict(), animation.to_dict())

    def set_entity_geometry_reference(
        self, entity_id: str, reference: GeometryReference | None
    ) -> EntityDelta | None:
        """Attach or replace the geometry reference of an entity."""
        entity = self.get(entity_id)
        before = entity.state.geometry_reference
        if before == reference:
            return None
        entity.state.geometry_reference = reference
        return EntityDelta(
            entity_id,
            "geometry_reference",
            before.to_dict() if before else None,
            reference.to_dict() if reference else None,
        )

    def set_active_lod(self, lod: LodLevel) -> LodLevel:
        """Set the active LOD after validating it against the ladder."""
        self.lod_ladder.validate(lod)
        self.active_lod = lod
        return lod

    def set_educational_level(self, level: EducationalLevel) -> LodLevel:
        """Set the audience level and return the LOD it maps to."""
        lod = self.educational_levels.lod_for(level)
        self.educational_level = level
        return lod

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------
    def commit(
        self,
        *,
        operation: str,
        summary: str,
        deltas: Sequence[EntityDelta] = (),
        command_text: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> SceneEvent:
        """Record a change-set and bump the scene version."""
        self.version += 1
        event = SceneEvent(
            version=self.version,
            operation=operation,
            summary=summary,
            deltas=tuple(deltas),
            command_text=command_text,
            parameters=dict(parameters or {}),
        )
        self.history.append(event)
        return event

    def last_event(self) -> SceneEvent | None:
        """Most recent committed event, if any."""
        return self.history[-1] if self.history else None

    def state_snapshot(self) -> dict[str, dict[str, Any]]:
        """Compact per-entity state snapshot, used for diffing and tests."""
        return {
            entity.entity_id: {
                "visibility": entity.visibility,
                "opacity": entity.opacity,
                "geometry_component": (
                    entity.geometry_reference.component_id if entity.geometry_reference else None
                ),
                "animation_clip": entity.animation_state.clip_id,
            }
            for entity in self.iter_entities()
        }

    def describe(self) -> dict[str, Any]:
        """Compact scene description for reports and the demo script."""
        visible = self.visible_ids()
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "ontology": f"{self.ontology_id}@{self.ontology_version}",
            "active_lod": self.active_lod,
            "lod_name": self.lod_spec().name,
            "educational_level": str(self.educational_level),
            "entity_count": len(self.entities),
            "renderable_count": len(self.renderable_ids()),
            "visible_count": len(visible),
            "version": self.version,
            "events": len(self.history),
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole scene, including history, to plain data."""
        return {
            "schema_version": AWR_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "name": self.name,
            "domain_id": self.domain_id,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "root_id": self.root_id,
            "active_lod": self.active_lod,
            "educational_level": str(self.educational_level),
            "lod_ladder": [
                {"level": spec.level, "name": spec.name, "description": spec.description}
                for spec in self.lod_ladder.levels
            ],
            "educational_levels": [
                {"level": str(spec.level), "lod": spec.lod, "description": spec.description}
                for spec in self.educational_levels.specs
            ],
            "entities": [entity.to_dict() for entity in self.iter_entities()],
            "relationships": self.relationships.to_list(),
            "metadata": dict(self.metadata),
            "version": self.version,
            "history": [event.to_dict() for event in self.history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AWRScene:
        """Rebuild a scene from :meth:`to_dict` output."""
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != AWR_SCHEMA_VERSION:
            raise SchemaError(
                f"Scene payload has schema_version {schema_version!r}, expected "
                f"{AWR_SCHEMA_VERSION!r}."
            )
        from awr.lod import EducationalLevelSpec  # local import: avoids a cycle at module load

        entities = {
            entity["definition"]["entity_id"]: SceneEntity.from_dict(entity)
            for entity in payload.get("entities", ())
        }
        ladder = LodLadder(
            tuple(
                LodLevelSpec(int(s["level"]), str(s["name"]), str(s.get("description", "")))
                for s in payload.get("lod_ladder", ())
            )
        )
        educational = EducationalLevelMap(
            tuple(
                EducationalLevelSpec(
                    coerce_enum(EducationalLevel, str(s["level"]), field_name="educational_level"),
                    int(s["lod"]),
                    str(s.get("description", "")),
                )
                for s in payload.get("educational_levels", ())
            )
        )
        return cls(
            scene_id=str(payload["scene_id"]),
            name=str(payload.get("name", "")),
            domain_id=str(payload.get("domain_id", "")),
            ontology_id=str(payload.get("ontology_id", "")),
            ontology_version=str(payload.get("ontology_version", "")),
            entities=entities,
            relationships=RelationshipStore.from_list(payload.get("relationships", ())),
            root_id=str(payload["root_id"]),
            lod_ladder=ladder,
            educational_levels=educational,
            active_lod=int(payload["active_lod"]),
            educational_level=coerce_enum(
                EducationalLevel, str(payload["educational_level"]), field_name="educational_level"
            ),
            metadata=dict(payload.get("metadata", {})),
            version=int(payload.get("version", 0)),
            history=[SceneEvent.from_dict(e) for e in payload.get("history", ())],
        )

    def save(self, path: str | Path) -> Path:
        """Write the scene to a JSON file and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> AWRScene:
        """Read a scene previously written by :meth:`save`."""
        source = Path(path)
        if not source.is_file():
            raise SceneError(f"Scene file not found: {source}")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
