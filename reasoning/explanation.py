"""Deterministic explanation of scene state, derived from the AWR graphs.

"Explain what is happening" is answered by *reading the representation*, not by
generating prose from a model. Every sentence below traces back to a typed edge,
an ontology field or a scene state value, which makes the explanation auditable:
if it says the left ventricle pumps to the aorta, there is a functional edge that
says so.

This is exactly the property a later language model should be held to. The
interface is therefore explicit, and a learned explainer can replace
:class:`TemplateSceneExplainer` without touching the AWR.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.errors import NotYetImplementedError
from awr.relationships import GraphKind
from awr.scene import AWRScene

__all__ = ["Explanation", "SceneExplainer", "TemplateSceneExplainer", "LanguageModelExplainer"]


@dataclass(frozen=True, slots=True)
class Explanation:
    """A structured explanation with its provenance."""

    text: str
    points: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    source: str = "graph-derived"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def rendered(self) -> str:
        """Full text with bullet points appended."""
        if not self.points:
            return self.text
        bullets = "\n".join(f"- {point}" for point in self.points)
        return f"{self.text}\n{bullets}"


class SceneExplainer(ABC):
    """Interface for anything that explains AWR state in words."""

    explainer_id: str = "abstract"

    @abstractmethod
    def explain_scene(self, scene: AWRScene) -> Explanation:
        """Explain the current state of the whole scene."""

    @abstractmethod
    def explain_entity(self, scene: AWRScene, entity_id: str) -> Explanation:
        """Explain one entity in its anatomical context."""


class TemplateSceneExplainer(SceneExplainer):
    """Template explainer that verbalises the three graphs and scene state."""

    explainer_id = "template-v0.1"

    _RELATION_PHRASES: Mapping[str, str] = {
        "receives_from": "receives blood from",
        "pumps_to": "pumps blood to",
        "opens_into": "opens into",
        "exits_into": "empties into",
        "conducts_to": "conducts the cardiac impulse to",
        "connects_to": "is joined to",
        "continuous_with": "is continuous with",
        "part_of": "is part of",
        "contains": "contains",
        "adjacent_to": "lies next to",
        "inside": "lies inside",
        "surrounds": "surrounds",
        "anterior_to": "lies in front of",
        "posterior_to": "lies behind",
        "superior_to": "lies above",
        "inferior_to": "lies below",
        "left_of": "lies to the anatomical left of",
        "right_of": "lies to the anatomical right of",
    }

    def _name(self, scene: AWRScene, entity_id: str) -> str:
        return scene.get(entity_id).display_name.lower()

    def explain_scene(self, scene: AWRScene) -> Explanation:
        """Describe what the scene currently holds, shows and animates."""
        spec = scene.lod_spec()
        visible = scene.visible_ids()
        hidden_but_present = len(scene.renderable_ids()) - len(visible)
        points = [
            f"Level of detail {scene.active_lod} ({spec.name}): {spec.description}",
            f"Audience level: {scene.educational_level}.",
            (
                f"{len(visible)} structures are visible; {hidden_but_present} more exist in the "
                "representation but are not shown at this level."
            ),
        ]
        transparent = [
            entity.entity_id for entity in scene.iter_entities() if 0.0 < entity.opacity < 1.0
        ]
        if transparent:
            names = ", ".join(self._name(scene, entity_id) for entity_id in transparent)
            points.append(f"Made partly transparent: {names}.")

        animated = [
            entity for entity in scene.iter_entities() if entity.animation_state.is_animated
        ]
        if animated:
            clip = animated[0].animation_state.clip_id
            phases = sorted({e.animation_state.phase for e in animated if e.animation_state.phase})
            points.append(
                f"Animation clip {clip!r} is bound to {len(animated)} structures across "
                f"{len(phases)} phases: {', '.join(phases)}."
            )
            for entity in animated[:4]:
                points.append(
                    f"{entity.display_name} is {entity.animation_state.role} during "
                    f"{entity.animation_state.phase}."
                )
        else:
            points.append("No animation is bound; the scene is static.")

        return Explanation(
            text=(
                f"This is {scene.name.lower()}, held as an anatomical representation of "
                f"{len(scene)} entities with {len(scene.relationships)} typed relationships."
            ),
            points=tuple(points),
            entity_ids=visible,
            metadata={"scene_version": scene.version, "explainer": self.explainer_id},
        )

    def explain_entity(self, scene: AWRScene, entity_id: str) -> Explanation:
        """Describe one entity using its ontology facts and graph edges."""
        entity = scene.get(entity_id)
        store = scene.relationships
        points: list[str] = []

        parent = entity.parent_id
        if parent:
            points.append(f"It sits under {self._name(scene, parent)} in the scene tree.")
        if entity.children:
            children = ", ".join(self._name(scene, child) for child in entity.children)
            points.append(f"It holds: {children}.")

        for kind, label in (
            (GraphKind.FUNCTIONAL, "Function"),
            (GraphKind.STRUCTURE, "Structure"),
            (GraphKind.SPATIAL, "Position"),
        ):
            graph = store.graph(kind)
            statements: list[str] = []
            for relation in sorted({edge.relation for edge in graph.edges}):
                targets = graph.related(entity_id, relation)
                if not targets:
                    continue
                phrase = self._RELATION_PHRASES.get(relation, relation.replace("_", " "))
                names = ", ".join(self._name(scene, target) for target in targets)
                statements.append(f"{phrase} {names}")
            if statements:
                points.append(f"{label}: it " + "; it ".join(statements) + ".")

        state = (
            f"Currently {'visible' if entity.visibility else 'hidden'} with opacity "
            f"{entity.opacity:g}; it appears by default from level of detail "
            f"{entity.lod_policy.min_lod}."
        )
        points.append(state)
        if entity.geometry_reference:
            points.append(
                f"Geometry component {entity.geometry_reference.component_id} is reserved for it "
                f"({entity.geometry_reference.kind}; no geometry has been generated yet)."
            )
        if entity.definition.notes:
            points.append(f"Note: {entity.definition.notes}")

        article = "The" if not entity.is_group else "The"
        return Explanation(
            text=(
                f"{article} {entity.display_name.lower()} ({entity.entity_id}) is a "
                f"{str(entity.anatomy_type).replace('_', ' ')} with the role "
                f"{entity.semantic_role.replace('_', ' ')}."
            ),
            points=tuple(points),
            entity_ids=(entity_id,),
            metadata={"explainer": self.explainer_id},
        )

    def explain_entities(self, scene: AWRScene, entity_ids: Sequence[str]) -> Explanation:
        """Explain several entities in one answer."""
        if len(entity_ids) == 1:
            return self.explain_entity(scene, entity_ids[0])
        parts = [self.explain_entity(scene, entity_id) for entity_id in entity_ids]
        return Explanation(
            text=" ".join(part.text for part in parts),
            points=tuple(point for part in parts for point in part.points),
            entity_ids=tuple(entity_ids),
            metadata={"explainer": self.explainer_id},
        )


class LanguageModelExplainer(SceneExplainer):
    """Learned explainer. Declared, not implemented.

    TODO(step-5 and later): a learned explainer must be grounded in the AWR and
    checkable against it, so that an explanation cannot assert anatomy the graphs
    do not contain. The grounding and verification strategy is an open question.
    """

    explainer_id = "language-model-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "LanguageModelExplainer", planned_in="a later R&D step"
        )

    def explain_scene(self, scene: AWRScene) -> Explanation:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("LanguageModelExplainer.explain_scene")

    def explain_entity(self, scene: AWRScene, entity_id: str) -> Explanation:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("LanguageModelExplainer.explain_entity")
