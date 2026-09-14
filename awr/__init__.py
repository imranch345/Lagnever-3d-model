"""Anatomical World Representation (AWR).

The AWR is the source of truth of Lagnav 3D: a persistent, typed representation
of anatomical entities, their hierarchy, their relationships across three graphs,
and their state. Geometry hangs off it by reference and is never the truth.

Modules:

* ``schema``        - value types and controlled vocabularies.
* ``entities``      - immutable entity identity plus mutable scene state.
* ``relationships`` - relation registry and the structure/spatial/functional graphs.
* ``ontology``      - strict loader for versioned ontology data.
* ``lod``           - level-of-detail policy.
* ``scene``         - the persistent scene, its history and its persistence.
* ``validation``    - reporting validators for ontologies and scenes.
* ``config``        - typed configuration loading.
* ``paths``         - repository-relative path resolution.
* ``errors``        - the typed exception hierarchy.
"""

from awr.config import DomainConfig, ModelConfig, load_domain_config, load_model_config
from awr.entities import AnatomicalEntity, SceneEntity
from awr.lod import EducationalLevelMap, LodLadder
from awr.ontology import AnatomyOntology, load_ontology
from awr.relationships import (
    DEFAULT_RELATION_TYPES,
    FlowMedium,
    FlowSemantics,
    Granularity,
    GraphKind,
    RelationRegistry,
    Relationship,
    RelationshipGraph,
    RelationshipStore,
    RelationTypeSpec,
    default_relation_registry,
)
from awr.scene import AWRScene, EntityDelta, SceneEvent
from awr.schema import (
    AWR_SCHEMA_VERSION,
    AnatomyType,
    AnimationState,
    EducationalLevel,
    EntityState,
    GeometryReference,
    GeometryRepresentationKind,
    Laterality,
    LodPolicy,
    MaterialState,
    Transform,
    VisibilitySource,
)
from awr.validation import ValidationReport, validate_ontology, validate_scene

__version__ = "0.1.0"

__all__ = [
    "AWRScene",
    "AWR_SCHEMA_VERSION",
    "AnatomicalEntity",
    "AnatomyOntology",
    "AnatomyType",
    "AnimationState",
    "DEFAULT_RELATION_TYPES",
    "DomainConfig",
    "EducationalLevel",
    "EducationalLevelMap",
    "EntityDelta",
    "EntityState",
    "FlowMedium",
    "FlowSemantics",
    "GeometryReference",
    "GeometryRepresentationKind",
    "Granularity",
    "GraphKind",
    "Laterality",
    "LodLadder",
    "LodPolicy",
    "MaterialState",
    "ModelConfig",
    "RelationRegistry",
    "RelationTypeSpec",
    "Relationship",
    "RelationshipGraph",
    "RelationshipStore",
    "SceneEntity",
    "SceneEvent",
    "Transform",
    "ValidationReport",
    "VisibilitySource",
    "__version__",
    "default_relation_registry",
    "load_domain_config",
    "load_model_config",
    "load_ontology",
    "validate_ontology",
    "validate_scene",
]
