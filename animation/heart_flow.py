"""Semantic blood-flow and conduction model for the heart.

"Animate blood flow" is answered here, and it is answered from the **functional
graph**, not from hand-written animation data. The flow paths are discovered by
traversing typed relationships whose relation type declares that it carries
blood; the cardiac phases come from configuration; the participating entities and
their per-phase roles are derived from the ontology.

That distinction matters for the research claim. Nothing here draws or deforms
anything: the output is a semantic :class:`AnimationClip` that binds anatomical
entities to phases and roles. A future animation decoder consumes exactly this
structure and turns it into motion.

Two properties are deliberate:

* **Granularity aware.** At LOD 0-1 the valves are not part of the scene, so the
  traversal uses chamber-level ``summary`` edges. From LOD 2 the valves exist and
  the traversal routes through them. Parallel paths are never double-counted.
* **Honest about boundaries.** The path from the right ventricle ends at the
  pulmonary arteries, because the pulmonary capillary bed is outside the heart
  ontology. That boundary is reported, not silently bridged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from awr.config import CardiacCycleConfig
from awr.errors import SceneError
from awr.relationships import FlowMedium, FlowSemantics, Granularity, GraphKind
from awr.scene import AWRScene
from awr.schema import AnatomyType, AnimationState, LodLevel

__all__ = [
    "FlowRole",
    "FlowSegment",
    "FlowPath",
    "CyclePhase",
    "AnimationClip",
    "CardiacFlowModel",
]


class FlowRole(StrEnum):
    """Role an entity plays in an animation phase."""

    CONTRACTING = "contracting"
    RELAXING = "relaxing"
    FILLING = "filling"
    EJECTING = "ejecting"
    VALVE_OPEN = "valve_open"
    VALVE_CLOSED = "valve_closed"
    CONDUCTING = "conducting"
    CARRYING_FLOW = "carrying_flow"


@dataclass(frozen=True, slots=True)
class FlowSegment:
    """One directed step of flow between two entities."""

    source: str
    target: str
    relation: str
    granularity: Granularity = Granularity.ANY

    def describe(self) -> str:
        """One-line human-readable description."""
        return f"{self.source} --{self.relation}--> {self.target}"


@dataclass(frozen=True, slots=True)
class FlowPath:
    """An ordered chain of flow segments from an inflow to an outflow boundary."""

    name: str
    medium: FlowMedium
    segments: tuple[FlowSegment, ...]
    enters_from: str | None = None
    exits_to: str | None = None

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Entities on the path, in flow order."""
        if not self.segments:
            return ()
        out = [self.segments[0].source]
        out.extend(segment.target for segment in self.segments)
        return tuple(out)

    def describe(self) -> str:
        """Arrow-form description of the whole path."""
        body = " -> ".join(entity.split(".")[-1] for entity in self.entity_ids)
        prefix = f"[{self.enters_from}] -> " if self.enters_from else ""
        suffix = f" -> [{self.exits_to}]" if self.exits_to else ""
        return f"{prefix}{body}{suffix}"


@dataclass(frozen=True, slots=True)
class CyclePhase:
    """One phase of the cardiac cycle, with the roles entities take in it."""

    name: str
    description: str
    start: float
    end: float
    roles: Mapping[str, FlowRole] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Normalised duration of the phase."""
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        """Normalised midpoint, used when binding entity animation state."""
        return (self.start + self.end) / 2.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "start": self.start,
            "end": self.end,
            "roles": {entity: str(role) for entity, role in self.roles.items()},
        }


@dataclass(frozen=True, slots=True)
class AnimationClip:
    """A semantic animation clip: phases, roles and flow paths. No geometry.

    This is what the prototype produces for "animate blood flow". It is a
    schedule over anatomical entities, ready for a future animation decoder.
    """

    clip_id: str
    display_name: str
    duration_seconds: float
    phases: tuple[CyclePhase, ...]
    paths: tuple[FlowPath, ...]
    lod: LodLevel
    granularity: Granularity
    notes: tuple[str, ...] = ()

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Every entity taking part, in stable order."""
        seen: set[str] = set()
        out: list[str] = []
        for path in self.paths:
            for entity_id in path.entity_ids:
                if entity_id not in seen:
                    seen.add(entity_id)
                    out.append(entity_id)
        for phase in self.phases:
            for entity_id in phase.roles:
                if entity_id not in seen:
                    seen.add(entity_id)
                    out.append(entity_id)
        return tuple(out)

    def bindings(self) -> dict[str, AnimationState]:
        """Animation state to store on each participating entity.

        Each entity is bound to this clip with the role it plays in the first
        phase where it is active, which keeps the binding deterministic.
        """
        out: dict[str, AnimationState] = {}
        for phase in self.phases:
            for entity_id, role in phase.roles.items():
                if entity_id in out:
                    continue
                out[entity_id] = AnimationState(
                    clip_id=self.clip_id,
                    role=str(role),
                    phase=phase.name,
                    normalized_time=phase.midpoint,
                    playing=True,
                )
        for path in self.paths:
            for entity_id in path.entity_ids:
                out.setdefault(
                    entity_id,
                    AnimationState(
                        clip_id=self.clip_id,
                        role=str(FlowRole.CARRYING_FLOW),
                        phase=self.phases[0].name if self.phases else None,
                        normalized_time=0.0,
                        playing=True,
                    ),
                )
        return out

    def summary(self) -> dict[str, Any]:
        """Compact description for reports and tests."""
        return {
            "clip_id": self.clip_id,
            "duration_seconds": self.duration_seconds,
            "phases": [phase.name for phase in self.phases],
            "paths": [path.describe() for path in self.paths],
            "entities": len(self.entity_ids),
            "lod": self.lod,
            "granularity": str(self.granularity),
            "notes": list(self.notes),
        }


class CardiacFlowModel:
    """Builds semantic flow paths and cycle clips from a scene's functional graph."""

    def __init__(self, scene: AWRScene, cycle: CardiacCycleConfig | None = None) -> None:
        self.scene = scene
        self.cycle = cycle or CardiacCycleConfig()

    # ------------------------------------------------------------------
    # flow traversal
    # ------------------------------------------------------------------
    def granularity_for_lod(self, lod: LodLevel | None = None) -> Granularity:
        """Choose the traversal granularity for a level of detail.

        Below the LOD where valves exist, chamber-level ``summary`` edges are
        used; from that LOD on, the detailed valve chain is used.
        """
        active = self.scene.active_lod if lod is None else lod
        valve_lods = [
            entity.lod_policy.min_lod
            for entity in self.scene.iter_entities()
            if entity.anatomy_type is AnatomyType.VALVE
        ]
        threshold = min(valve_lods) if valve_lods else 0
        return Granularity.DETAILED if active >= threshold else Granularity.SUMMARY

    def _directed_edges(
        self, medium: FlowMedium, granularity: Granularity
    ) -> list[FlowSegment]:
        """Flow-carrying edges of one medium, oriented in the direction of flow."""
        registry = self.scene.relationships.registry
        segments: list[FlowSegment] = []
        for edge in self.scene.relationships.graph(GraphKind.FUNCTIONAL).edges:
            spec = registry.get(edge.relation)
            if spec.medium is not medium or spec.flow is FlowSemantics.NONE:
                continue
            if edge.granularity not in (granularity, Granularity.ANY):
                continue
            if spec.flow is FlowSemantics.FORWARD:
                segments.append(
                    FlowSegment(edge.subject, edge.object, edge.relation, edge.granularity)
                )
            else:  # REVERSE: the object is upstream of the subject
                segments.append(
                    FlowSegment(edge.object, edge.subject, edge.relation, edge.granularity)
                )
        return segments

    def flow_paths(
        self,
        *,
        medium: FlowMedium = FlowMedium.BLOOD,
        granularity: Granularity | None = None,
        lod: LodLevel | None = None,
    ) -> tuple[FlowPath, ...]:
        """Enumerate flow paths from every inflow source to every outflow sink."""
        active_granularity = granularity or self.granularity_for_lod(lod)
        segments = self._directed_edges(medium, active_granularity)
        if not segments:
            raise SceneError(
                f"The functional graph holds no {medium} edges at granularity "
                f"{active_granularity}; cannot derive flow paths."
            )
        outgoing: dict[str, list[FlowSegment]] = {}
        has_incoming: set[str] = set()
        for segment in segments:
            outgoing.setdefault(segment.source, []).append(segment)
            has_incoming.add(segment.target)

        order = {entity_id: index for index, entity_id in enumerate(self.scene.ids())}
        sources = sorted(
            {s.source for s in segments if s.source not in has_incoming},
            key=lambda entity_id: order.get(entity_id, len(order)),
        )

        paths: list[FlowPath] = []
        for source in sources:
            for chain in self._walk(source, outgoing, order):
                terminal = chain[-1].target
                paths.append(
                    FlowPath(
                        name=(
                            f"{self.scene.get(source).display_name} to "
                            f"{self.scene.get(terminal).display_name}"
                        ),
                        medium=medium,
                        segments=tuple(chain),
                        enters_from=self._boundary(source, "external_origin"),
                        exits_to=self._boundary(terminal, "external_continuation"),
                    )
                )
        return tuple(paths)

    def _walk(
        self,
        node: str,
        outgoing: Mapping[str, Sequence[FlowSegment]],
        order: Mapping[str, int],
        visited: tuple[str, ...] = (),
    ) -> list[list[FlowSegment]]:
        """Depth-first enumeration of simple paths from ``node`` to every sink."""
        if node in visited:
            return []
        next_segments = sorted(
            outgoing.get(node, ()), key=lambda s: order.get(s.target, len(order))
        )
        if not next_segments:
            return [[]]
        chains: list[list[FlowSegment]] = []
        for segment in next_segments:
            for tail in self._walk(segment.target, outgoing, order, (*visited, node)):
                chains.append([segment, *tail])
        return chains

    def _boundary(self, entity_id: str, key: str) -> str | None:
        """Read an entity's declared external boundary, if it has one."""
        value = self.scene.get(entity_id).metadata.get(key)
        return str(value) if value else None

    def conduction_paths(self, lod: LodLevel | None = None) -> tuple[FlowPath, ...]:
        """Electrical conduction paths, derived the same way as blood flow."""
        return self.flow_paths(medium=FlowMedium.IMPULSE, granularity=Granularity.ANY, lod=lod)

    # ------------------------------------------------------------------
    # cycle clip
    # ------------------------------------------------------------------
    def _classify(self) -> dict[str, tuple[str, ...]]:
        """Group entities by the role they can play, using ontology facts only."""
        atria: list[str] = []
        ventricles: list[str] = []
        av_valves: list[str] = []
        semilunar_valves: list[str] = []
        outflow_vessels: list[str] = []
        inflow_vessels: list[str] = []
        store = self.scene.relationships
        for entity in self.scene.iter_entities():
            if entity.anatomy_type is AnatomyType.CHAMBER:
                if entity.semantic_role == "inflow_chamber":
                    atria.append(entity.entity_id)
                else:
                    ventricles.append(entity.entity_id)
            elif entity.anatomy_type is AnatomyType.VALVE:
                downstream = store.related(entity.entity_id, "opens_into")
                if any(
                    self.scene.get(target).anatomy_type is AnatomyType.CHAMBER
                    for target in downstream
                ):
                    av_valves.append(entity.entity_id)
                else:
                    semilunar_valves.append(entity.entity_id)
            elif entity.anatomy_type is AnatomyType.VESSEL:
                if entity.semantic_role.endswith("outflow"):
                    outflow_vessels.append(entity.entity_id)
                elif entity.semantic_role.endswith("inflow"):
                    inflow_vessels.append(entity.entity_id)
        return {
            "atria": tuple(atria),
            "ventricles": tuple(ventricles),
            "av_valves": tuple(av_valves),
            "semilunar_valves": tuple(semilunar_valves),
            "outflow_vessels": tuple(outflow_vessels),
            "inflow_vessels": tuple(inflow_vessels),
        }

    def _phase_roles(
        self, phase_name: str, groups: Mapping[str, tuple[str, ...]]
    ) -> dict[str, FlowRole]:
        """Roles for one named phase.

        The phase names come from configuration; the mapping from a phase to the
        roles of atria, ventricles and valve classes is the domain knowledge this
        module contributes. Entity ids are never hard-coded.
        """
        roles: dict[str, FlowRole] = {}

        def assign(keys: Sequence[str], role: FlowRole) -> None:
            """Give every entity in the named groups the same role."""
            for key in keys:
                for entity_id in groups.get(key, ()):
                    roles[entity_id] = role

        if phase_name == "atrial_systole":
            assign(["atria"], FlowRole.CONTRACTING)
            assign(["ventricles"], FlowRole.FILLING)
            assign(["av_valves"], FlowRole.VALVE_OPEN)
            assign(["semilunar_valves"], FlowRole.VALVE_CLOSED)
            assign(["inflow_vessels"], FlowRole.CARRYING_FLOW)
        elif phase_name == "isovolumetric_contraction":
            assign(["ventricles"], FlowRole.CONTRACTING)
            assign(["atria"], FlowRole.RELAXING)
            assign(["av_valves", "semilunar_valves"], FlowRole.VALVE_CLOSED)
        elif phase_name == "ventricular_ejection":
            assign(["ventricles"], FlowRole.EJECTING)
            assign(["atria"], FlowRole.FILLING)
            assign(["av_valves"], FlowRole.VALVE_CLOSED)
            assign(["semilunar_valves"], FlowRole.VALVE_OPEN)
            assign(["outflow_vessels"], FlowRole.CARRYING_FLOW)
        elif phase_name == "isovolumetric_relaxation":
            assign(["ventricles"], FlowRole.RELAXING)
            assign(["atria"], FlowRole.FILLING)
            assign(["av_valves", "semilunar_valves"], FlowRole.VALVE_CLOSED)
        elif phase_name == "ventricular_filling":
            assign(["ventricles"], FlowRole.FILLING)
            assign(["atria"], FlowRole.RELAXING)
            assign(["av_valves"], FlowRole.VALVE_OPEN)
            assign(["semilunar_valves"], FlowRole.VALVE_CLOSED)
            assign(["inflow_vessels"], FlowRole.CARRYING_FLOW)
        return roles

    def build_clip(self, *, lod: LodLevel | None = None) -> AnimationClip:
        """Build the blood-flow clip for the scene's current level of detail."""
        if not self.cycle.phases:
            raise SceneError(
                "No cardiac-cycle phases are configured; see the cardiac_cycle section "
                "of configs/heart.yaml."
            )
        granularity = self.granularity_for_lod(lod)
        paths = self.flow_paths(granularity=granularity, lod=lod)
        groups = self._classify()

        phases: list[CyclePhase] = []
        cursor = 0.0
        for spec in self.cycle.phases:
            start, end = cursor, cursor + spec.fraction
            cursor = end
            phases.append(
                CyclePhase(
                    name=spec.name,
                    description=spec.description,
                    start=round(start, 6),
                    end=round(end, 6),
                    roles=self._phase_roles(spec.name, groups),
                )
            )

        boundaries = tuple(
            f"{path.name} continues outside the heart ontology into {path.exits_to}."
            for path in paths
            if path.exits_to
        )
        return AnimationClip(
            clip_id=self.cycle.clip_id,
            display_name="Blood flow through the heart",
            duration_seconds=self.cycle.cycle_duration_seconds,
            phases=tuple(phases),
            paths=paths,
            lod=self.scene.active_lod if lod is None else lod,
            granularity=granularity,
            notes=(
                "Semantic clip only: entities, phases and roles. No geometry is deformed and "
                "no renderer is driven; the animation decoder is future work.",
                "Phase fractions are provisional configuration, not a validated haemodynamic "
                "model.",
                *boundaries,
            ),
        )
