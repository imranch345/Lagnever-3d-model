"""Anatomical entities: immutable identity plus mutable scene state.

The AWR splits every entity into two halves on purpose:

* :class:`AnatomicalEntity` - identity and semantics from the ontology. Frozen.
  An edit can never change an entity id, because the identity object is not
  writable at all.
* :class:`SceneEntity` - the entity as it exists inside one scene: its place in
  the tree plus its mutable :class:`~awr.schema.EntityState`.

The fields required by the AWR entity model are all reachable from a
:class:`SceneEntity`: ``entity_id``, ``anatomy_type``, ``display_name``,
``canonical_name``, ``synonyms``, ``parent_id``, ``children``, ``semantic_role``,
``geometry_reference``, ``material_state``, ``transform``, ``visibility``,
``opacity``, ``lod``, ``animation_state`` and ``metadata``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from awr.schema import (
    AnatomyType,
    AnimationState,
    EntityState,
    GeometryReference,
    Laterality,
    LodLevel,
    LodPolicy,
    MaterialState,
    Transform,
    coerce_enum,
    unique,
)

__all__ = ["AnatomicalEntity", "SceneEntity"]


@dataclass(frozen=True, slots=True)
class AnatomicalEntity:
    """Ontology-defined identity and semantics of one anatomical entity.

    Frozen: this object is the persistent identity of the thing. Scene edits
    operate on :class:`SceneEntity.state`, never here.
    """

    entity_id: str
    canonical_name: str
    display_name: str
    anatomy_type: AnatomyType
    semantic_role: str
    lod_policy: LodPolicy = field(default_factory=LodPolicy)
    laterality: Laterality = Laterality.NONE
    synonyms: tuple[str, ...] = ()
    abbreviations: tuple[str, ...] = ()
    is_group: bool = False
    renderable: bool = True
    notes: str | None = None
    source_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def lod(self) -> LodLevel:
        """Level of detail tier the entity belongs to (its ``min_lod``)."""
        return self.lod_policy.min_lod

    def surface_forms(self) -> tuple[str, ...]:
        """Every textual form that should resolve to this entity."""
        forms = [
            self.canonical_name,
            self.canonical_name.replace("_", " "),
            self.display_name,
            *self.synonyms,
            *self.abbreviations,
        ]
        return unique(form for form in forms if form)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "anatomy_type": str(self.anatomy_type),
            "semantic_role": self.semantic_role,
            "lod_policy": self.lod_policy.to_dict(),
            "laterality": str(self.laterality),
            "synonyms": list(self.synonyms),
            "abbreviations": list(self.abbreviations),
            "is_group": self.is_group,
            "renderable": self.renderable,
            "notes": self.notes,
            "source_refs": list(self.source_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            entity_id=str(payload["entity_id"]),
            canonical_name=str(payload["canonical_name"]),
            display_name=str(payload["display_name"]),
            anatomy_type=coerce_enum(
                AnatomyType, payload["anatomy_type"], field_name="anatomy_type"
            ),
            semantic_role=str(payload.get("semantic_role", "")),
            lod_policy=LodPolicy.from_dict(payload.get("lod_policy", {})),
            laterality=coerce_enum(
                Laterality, payload.get("laterality", Laterality.NONE), field_name="laterality"
            ),
            synonyms=tuple(payload.get("synonyms", ())),
            abbreviations=tuple(payload.get("abbreviations", ())),
            is_group=bool(payload.get("is_group", False)),
            renderable=bool(payload.get("renderable", True)),
            notes=payload.get("notes"),
            source_refs=tuple(payload.get("source_refs", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class SceneEntity:
    """An anatomical entity as it exists inside a scene.

    ``definition`` carries identity; ``state`` carries everything a command can
    change. ``parent_id`` and ``children`` place the entity in the scene tree.
    """

    definition: AnatomicalEntity
    parent_id: str | None = None
    children: tuple[str, ...] = ()
    state: EntityState = field(default_factory=EntityState)

    # --- identity passthrough (read-only) --------------------------------
    @property
    def entity_id(self) -> str:
        """Persistent entity identifier. Never changes for the life of the entity."""
        return self.definition.entity_id

    @property
    def canonical_name(self) -> str:
        """Canonical machine name."""
        return self.definition.canonical_name

    @property
    def display_name(self) -> str:
        """Human-facing name."""
        return self.definition.display_name

    @property
    def anatomy_type(self) -> AnatomyType:
        """Kind of anatomical entity."""
        return self.definition.anatomy_type

    @property
    def semantic_role(self) -> str:
        """Functional role of the entity within the organ."""
        return self.definition.semantic_role

    @property
    def synonyms(self) -> tuple[str, ...]:
        """Alternative names accepted by the entity resolver."""
        return self.definition.synonyms

    @property
    def abbreviations(self) -> tuple[str, ...]:
        """Accepted abbreviations, such as ``LV``."""
        return self.definition.abbreviations

    @property
    def is_group(self) -> bool:
        """Whether this is an organisational group rather than tissue."""
        return self.definition.is_group

    @property
    def renderable(self) -> bool:
        """Whether the entity can ever be drawn (groups cannot)."""
        return self.definition.renderable

    @property
    def lod(self) -> LodLevel:
        """LOD tier of the entity."""
        return self.definition.lod

    @property
    def lod_policy(self) -> LodPolicy:
        """Default-visibility LOD window."""
        return self.definition.lod_policy

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Ontology metadata (notes, simplifications, external continuations)."""
        return self.definition.metadata

    # --- state passthrough (read-only; mutate via editing operations) -----
    @property
    def visibility(self) -> bool:
        """Whether the entity is currently visible."""
        return self.state.visibility

    @property
    def opacity(self) -> float:
        """Current opacity in ``[0, 1]``."""
        return self.state.opacity

    @property
    def geometry_reference(self) -> GeometryReference | None:
        """Link to the entity's geometry component, if one has been allocated."""
        return self.state.geometry_reference

    @property
    def material_state(self) -> MaterialState:
        """Current material state."""
        return self.state.material

    @property
    def transform(self) -> Transform:
        """Current transform."""
        return self.state.transform

    @property
    def animation_state(self) -> AnimationState:
        """Current animation binding."""
        return self.state.animation

    def with_children(self, children: Sequence[str]) -> SceneEntity:
        """Return a copy with a different child list (used while building a scene)."""
        return SceneEntity(
            definition=self.definition,
            parent_id=self.parent_id,
            children=tuple(children),
            state=self.state.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "definition": self.definition.to_dict(),
            "parent_id": self.parent_id,
            "children": list(self.children),
            "state": self.state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            definition=AnatomicalEntity.from_dict(payload["definition"]),
            parent_id=payload.get("parent_id"),
            children=tuple(payload.get("children", ())),
            state=EntityState.from_dict(payload.get("state", {})),
        )
