"""Anatomical reasoning: from an encoded request to an AWR construction plan.

The reasoner decides *what anatomy the scene should contain* before anything is
built. Separating it from the scene builder matters: the plan is inspectable,
comparable between runs, and testable without constructing anything.

The deterministic implementation reasons over the ontology only. A learned
reasoner that operates on graphs or embeddings is Step 5 work and slots in behind
the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from awr.errors import NotYetImplementedError, OntologyError
from awr.ontology import AnatomyOntology
from awr.schema import EducationalLevel, LodLevel
from generation.text_encoder import LanguageEncoding

__all__ = [
    "AWRConstructionPlan",
    "AnatomicalReasoner",
    "OntologyDrivenReasoner",
    "NeuralAnatomicalReasoner",
]


@dataclass(frozen=True, slots=True)
class AWRConstructionPlan:
    """What the AWR should contain for one request.

    ``entity_ids`` is the full internal content of the scene and is deliberately
    richer than what will be visible: the LOD decides visibility, the plan decides
    existence.
    """

    domain_id: str
    root_entity_id: str
    entity_ids: tuple[str, ...]
    initial_lod: LodLevel
    educational_level: EducationalLevel
    focus_entity_ids: tuple[str, ...] = ()
    reasoner_id: str = "unknown"
    rationale: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact description for reports and tests."""
        return {
            "domain": self.domain_id,
            "root": self.root_entity_id,
            "entities": len(self.entity_ids),
            "initial_lod": self.initial_lod,
            "educational_level": str(self.educational_level),
            "focus": list(self.focus_entity_ids),
            "reasoner": self.reasoner_id,
        }


class AnatomicalReasoner(ABC):
    """Interface for every Lagnav anatomical reasoner.

    TODO(step-5): the learned reasoner's architecture, whether it message-passes
    over the AWR graphs, its entity and relation embedding sizes, and how it is
    supervised are all open questions. The contract that will not change is this
    one: an encoding plus an ontology in, a construction plan out.
    """

    reasoner_id: str = "abstract"

    @abstractmethod
    def plan(self, encoding: LanguageEncoding, ontology: AnatomyOntology) -> AWRConstructionPlan:
        """Produce a construction plan for an encoded request."""


class OntologyDrivenReasoner(AnatomicalReasoner):
    """Deterministic reasoner: the ontology is the plan.

    For a whole-organ request it instantiates the entire ontology, so the AWR
    always holds the full internal structure regardless of the audience level.
    Entities named in the request become the focus, which the generator uses to
    decide what the learner is looking at.
    """

    reasoner_id = "ontology-driven-v0.1"

    def __init__(self, *, default_lod: LodLevel, default_level: EducationalLevel) -> None:
        self.default_lod = default_lod
        self.default_level = default_level

    def plan(
        self,
        encoding: LanguageEncoding,
        ontology: AnatomyOntology,
        *,
        lod: LodLevel | None = None,
        educational_level: EducationalLevel | None = None,
    ) -> AWRConstructionPlan:
        """Build a plan covering the whole organ, focused on mentioned entities."""
        mentioned = encoding.entity_ids
        if ontology.root_id not in mentioned and not mentioned:
            raise OntologyError(
                f"The request {encoding.raw_text!r} does not mention any entity of the "
                f"{ontology.ontology_id!r} ontology. The v0.1 prototype covers the human heart "
                "only; other anatomy is not modelled yet."
            )
        rationale = [
            f"Matched ontology terms: {', '.join(mentioned) if mentioned else 'none'}.",
            "Instantiating the complete ontology: the AWR always holds more than it shows.",
        ]
        return AWRConstructionPlan(
            domain_id=ontology.ontology_id.split(".")[-1],
            root_entity_id=ontology.root_id,
            entity_ids=ontology.ids(),
            initial_lod=self.default_lod if lod is None else lod,
            educational_level=(
                self.default_level if educational_level is None else educational_level
            ),
            focus_entity_ids=mentioned,
            reasoner_id=self.reasoner_id,
            rationale=tuple(rationale),
            metadata={"encoder": encoding.encoder_id},
        )


class NeuralAnatomicalReasoner(AnatomicalReasoner):
    """The Lagnav anatomical reasoning engine. Declared, not implemented.

    TODO(step-5): model architecture, graph reasoning strategy, entity and
    relation embeddings, training objectives and the supervision signal for
    anatomical correctness are all undecided.
    """

    reasoner_id = "lagnav-anatomical-reasoner-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "NeuralAnatomicalReasoner", planned_in="Step 5 (Lagnav neural architecture)"
        )

    def plan(
        self, encoding: LanguageEncoding, ontology: AnatomyOntology
    ) -> AWRConstructionPlan:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("NeuralAnatomicalReasoner.plan", planned_in="Step 5")
