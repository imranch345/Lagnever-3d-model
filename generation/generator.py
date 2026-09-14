"""Scene generation: text to a structured AWR scene.

This is the deterministic stand-in for the future generative pipeline. It wires
the real stages together in the real order:

    text -> LanguageEncoder -> AnatomicalReasoner -> AWR scene
         -> geometry component allocation (symbolic, no geometry produced)

What it does **not** do is produce geometry, call an external text-to-3D service,
or sample anything. The milestone being proven here is representational: given a
natural-language request, Lagnav builds a semantically structured heart in which
every entity has persistent identity, hierarchy, typed relationships, state and a
geometry reference reserved for later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from awr.config import DomainConfig, load_domain_config
from awr.ontology import AnatomyOntology, load_ontology
from awr.scene import AWRScene
from awr.schema import EducationalLevel, LodLevel
from editing.scene_editor import SceneEditor
from generation.anatomical_reasoner import (
    AnatomicalReasoner,
    AWRConstructionPlan,
    OntologyDrivenReasoner,
)
from generation.text_encoder import LanguageEncoder, LanguageEncoding, LexicalLanguageEncoder
from geometry.correspondence import ComponentIdAllocator, GeometryCorrespondence
from geometry.decoder import GeometryDecoder, GeometryDecodeRequest, SymbolicGeometryDecoder

__all__ = ["GenerationRequest", "GenerationResult", "SceneGenerator", "HeartSceneGenerator"]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A natural-language generation request with optional explicit overrides."""

    text: str
    lod: LodLevel | None = None
    educational_level: EducationalLevel | None = None
    scene_id: str | None = None
    allocate_geometry: bool = True


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Everything one generation produced, including its intermediate stages."""

    scene: AWRScene
    plan: AWRConstructionPlan
    encoding: LanguageEncoding
    correspondence: GeometryCorrespondence
    notes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact description for reports and tests."""
        return {
            "scene": self.scene.describe(),
            "plan": self.plan.summary(),
            "encoder": self.encoding.encoder_id,
            "geometry_links": len(self.correspondence),
            "geometry_payload": "none (symbolic component slots only)",
            "notes": list(self.notes),
        }


class SceneGenerator(ABC):
    """Interface for anything that can build a scene from text."""

    generator_id: str = "abstract"

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Build a scene for a request."""


class HeartSceneGenerator(SceneGenerator):
    """Deterministic generator for the human-heart domain."""

    generator_id = "heart-deterministic-v0.1"

    def __init__(
        self,
        *,
        config: DomainConfig | None = None,
        ontology: AnatomyOntology | None = None,
        encoder: LanguageEncoder | None = None,
        reasoner: AnatomicalReasoner | None = None,
        geometry_decoder: GeometryDecoder | None = None,
    ) -> None:
        self.config = config or load_domain_config()
        self.ontology = ontology or load_ontology(
            self.config.domain.ontology_dir,
            expected_version=self.config.domain.ontology_version,
        )
        self.encoder = encoder or LexicalLanguageEncoder(self.ontology)
        self.reasoner = reasoner or OntologyDrivenReasoner(
            default_lod=self.config.scene_defaults.active_lod,
            default_level=self.config.scene_defaults.educational_level,
        )
        self.geometry_decoder = geometry_decoder or SymbolicGeometryDecoder(
            ComponentIdAllocator(
                prefix=self.config.geometry.component_id_prefix,
                digits=self.config.geometry.component_id_digits,
            )
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Encode the request, plan the anatomy, build the scene, reserve geometry."""
        encoding = self.encoder.encode(request.text)
        plan = self._plan(encoding, request)
        scene = AWRScene.from_ontology(
            self.ontology,
            lod_ladder=self.config.lod_ladder,
            educational_levels=self.config.educational_levels,
            active_lod=plan.initial_lod,
            educational_level=plan.educational_level,
            name=self.config.scene_defaults.scene_name,
            domain_id=self.config.domain.id,
            default_opacity=self.config.scene_defaults.default_opacity,
            scene_id=request.scene_id,
            metadata={
                "generated_from": request.text,
                "generator": self.generator_id,
                "encoder": encoding.encoder_id,
                "reasoner": plan.reasoner_id,
                "focus": list(plan.focus_entity_ids),
                "geometry": "symbolic component slots only; no geometry has been generated",
            },
        )

        correspondence = GeometryCorrespondence()
        notes = list(plan.rationale)
        if request.allocate_geometry:
            decode = self.geometry_decoder.decode(
                GeometryDecodeRequest(
                    scene_id=scene.scene_id,
                    entity_ids=tuple(
                        entity.entity_id for entity in scene.iter_entities() if entity.renderable
                    ),
                    lod=scene.active_lod,
                )
            )
            correspondence = decode.correspondence
            SceneEditor(scene).attach_geometry(
                decode.references(), command_text=request.text
            )
            notes.extend(decode.notes)

        return GenerationResult(
            scene=scene,
            plan=plan,
            encoding=encoding,
            correspondence=correspondence,
            notes=tuple(notes),
            metadata={"generator": self.generator_id},
        )

    def _plan(self, encoding: LanguageEncoding, request: GenerationRequest) -> AWRConstructionPlan:
        """Ask the reasoner for a plan, honouring explicit request overrides."""
        if isinstance(self.reasoner, OntologyDrivenReasoner):
            return self.reasoner.plan(
                encoding,
                self.ontology,
                lod=request.lod,
                educational_level=request.educational_level,
            )
        return self.reasoner.plan(encoding, self.ontology)
