"""Batches from the whole-organ corpus, with one relationship graph per scene.

The difference from ``generation.neural.nn.tensors`` is the whole point of Step 7. There
the AWR structure was cached per level of detail, because every scene carried the same
ontology graph. Here the graph is measured from each organ, so each scene contributes
its own edges, adjacency and typed relations, and the tensors carry a batch axis.

That also makes the counterfactual experiments possible: swapping one scene's relation
graph for another's, with entities and presence held fixed, is a one-line change here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from awr.relationships import GraphKind
from awr.scene import AWRScene
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.relations import MeasuredEdge
from datasets.whole_organ.sampling import (
    SampledWholeOrgan,
    WholeOrganSampling,
    sample_scene,
)
from generation.neural.features import AWRFeatureExtractor
from generation.neural.nn.tensors import AWRStructure, PrototypeBatch

__all__ = ["WholeOrganBatch", "WholeOrganBatchBuilder"]


@dataclass(slots=True)
class WholeOrganBatch:
    """A prototype batch plus the whole-organ extras the Step 7 metrics need."""

    batch: PrototypeBatch
    scenes: tuple[WholeOrganScene, ...]
    sampled: tuple[SampledWholeOrgan, ...]
    lod_targets: Mapping[int, torch.Tensor]
    variant_index: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of scenes."""
        return len(self.scenes)


class WholeOrganBatchBuilder:
    """Turns whole-organ scenes into batches with per-scene relationship graphs."""

    def __init__(
        self,
        ontology: AnatomyOntology,
        config: DomainConfig,
        *,
        sampling: WholeOrganSampling | None = None,
        max_edges: int = 256,
    ) -> None:
        self.ontology = ontology
        self.config = config
        self.sampling = sampling or WholeOrganSampling()
        self._structure_cache: tuple[MeasuredEdge, ...] | None = None
        self.extractor = AWRFeatureExtractor(ontology)
        self.slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        self.max_edges = max_edges
        self.entity_slots = self.sampling.entity_slots
        self._entity_features = self._build_entity_features()

    # ------------------------------------------------------------------
    def vocabulary_sizes(self) -> Mapping[str, int]:
        """Vocabulary sizes for embedding tables."""
        return self.extractor.vocabulary_sizes()

    def inverse_relation_table(self) -> torch.Tensor:
        """Relation id to inverse relation id, or -1."""
        registry = self.ontology.registry
        table = torch.full((len(self.extractor.relations),), -1, dtype=torch.int64)
        for index, name in enumerate(self.extractor.relations.symbols):
            spec = registry.get(name)
            if spec.symmetric:
                table[index] = index
            elif spec.inverse is not None:
                table[index] = self.extractor.relations.index(spec.inverse)
        return table

    def _build_entity_features(self) -> dict[str, torch.Tensor]:
        """Per-entity symbolic features. Identical across scenes: only relations vary."""
        scene = AWRScene.from_ontology(
            self.ontology,
            lod_ladder=self.config.lod_ladder,
            educational_levels=self.config.educational_levels,
            active_lod=self.config.lod_ladder.max_level,
            educational_level=self.config.scene_defaults.educational_level,
            name=self.config.scene_defaults.scene_name,
            domain_id=self.config.domain.id,
        )
        payload = self.extractor.encode_batch([scene])
        wanted = (
            "entity_ids",
            "entity_type",
            "semantic_role",
            "laterality",
            "parent_index",
            "depth",
            "lod_min",
            "lod_max",
            "entity_mask",
        )
        out: dict[str, torch.Tensor] = {}
        for name in wanted:
            array = payload[name]
            dtype = torch.bool if str(array.dtype) == "bool" else torch.int64
            out[name] = torch.tensor(array.data, dtype=dtype).reshape(array.shape)[0]
        return out

    # ------------------------------------------------------------------
    def _edge_tensors(self, edge_sets: Sequence[Sequence[MeasuredEdge]]) -> dict[str, torch.Tensor]:
        batch = len(edge_sets)
        entities = int(self._entity_features["entity_ids"].shape[0])
        source = torch.full((batch, self.max_edges), -1, dtype=torch.int64)
        target = torch.full((batch, self.max_edges), -1, dtype=torch.int64)
        relation = torch.full((batch, self.max_edges), -1, dtype=torch.int64)
        graph = torch.full((batch, self.max_edges), -1, dtype=torch.int64)
        mask = torch.zeros((batch, self.max_edges), dtype=torch.bool)
        adjacency = torch.zeros((batch, len(GraphKind), entities, entities), dtype=torch.bool)

        for index, edges in enumerate(edge_sets):
            if len(edges) > self.max_edges:
                raise ValueError(
                    f"Scene {index} has {len(edges)} edges, more than the {self.max_edges} slots."
                )
            for position, edge in enumerate(edges):
                spec = self.ontology.registry.get(edge.relation)
                graph_index = self.extractor.graphs.index(str(spec.graph))
                subject_slot = self.slot_of[edge.subject]
                object_slot = self.slot_of[edge.object]
                source[index, position] = subject_slot
                target[index, position] = object_slot
                relation[index, position] = self.extractor.relations.index(edge.relation)
                graph[index, position] = graph_index
                mask[index, position] = True
                adjacency[index, graph_index, subject_slot, object_slot] = True
        return {
            "edge_source": source,
            "edge_target": target,
            "edge_relation": relation,
            "edge_graph": graph,
            "edge_mask": mask,
            "graph_adjacency": adjacency,
        }

    def _structure_edges(self, scene: WholeOrganScene) -> list[MeasuredEdge]:
        """The ontology's structural edges between entities this scene contains."""
        if self._structure_cache is None:
            self._structure_cache = tuple(
                MeasuredEdge(subject=edge.subject, relation=edge.relation, object=edge.object)
                for edge in self.ontology.graph(GraphKind.STRUCTURE).edges
            )
        present = set(scene.entity_ids)
        return [
            edge
            for edge in self._structure_cache
            if edge.subject in present and edge.object in present
        ]

    def _state(self, sampled: Sequence[SampledWholeOrgan]) -> torch.Tensor:
        """Explicit presentation state. Visibility only; nothing that leaks the variant."""
        batch = len(sampled)
        entities = int(self._entity_features["entity_ids"].shape[0])
        state = torch.zeros((batch, entities, 9), dtype=torch.float32)
        for index, item in enumerate(sampled):
            visible = torch.tensor(item.visible, dtype=torch.float32)
            state[index, :, 0] = visible
            state[index, :, 1] = 1.0
            state[index, :, 7] = visible
        return state

    def build(
        self,
        scenes: Sequence[WholeOrganScene],
        *,
        epoch: int = 0,
        edge_override: Sequence[Sequence[MeasuredEdge]] | None = None,
    ) -> WholeOrganBatch:
        """Build a batch. Every scene must share one level of detail.

        Args:
            scenes: The scenes to batch.
            epoch: Changes the sampling seed when point resampling is wanted.
            edge_override: Replacement relationship graphs, one per scene. This is how
                the counterfactual experiments swap relations while holding entities,
                presence and text features fixed.

        """
        if not scenes:
            raise ValueError("A batch needs at least one scene.")
        levels = {scene.active_lod for scene in scenes}
        if len(levels) != 1:
            raise ValueError(f"A batch must share one level of detail, got {sorted(levels)}.")
        level = next(iter(levels))

        sampled = [
            sample_scene(scene, self.slot_of, self.sampling, epoch=epoch) for scene in scenes
        ]
        batch_size = len(scenes)
        entities = int(self._entity_features["entity_ids"].shape[0])

        measured = list(edge_override) if edge_override is not None else [s.edges for s in scenes]
        # The structural graph is ontology knowledge, not a scene measurement: which
        # structures connect to and are continuous with which does not change when the
        # organ is mirrored. Without it the structure graph is empty and the
        # structure-only ablation degenerates into a no-graph arm that still pays for a
        # graph encoder. Because it is the same for every scene it can only act as a
        # prior, never as a channel that distinguishes one scene from another, and the
        # ablation results must be read in that light.
        edge_sets = [
            [*self._structure_edges(scene), *edges]
            for scene, edges in zip(scenes, measured, strict=True)
        ]
        edges = self._edge_tensors(edge_sets)

        def expand(name: str) -> torch.Tensor:
            return self._entity_features[name].unsqueeze(0).expand(batch_size, -1).contiguous()

        present = torch.tensor(np.stack([item.visible for item in sampled]), dtype=torch.bool)
        visible_slots = present[0].nonzero().flatten()
        structure = AWRStructure(
            lod=level,
            entity_ids=expand("entity_ids"),
            entity_type=expand("entity_type"),
            semantic_role=expand("semantic_role"),
            laterality=expand("laterality"),
            parent_index=expand("parent_index"),
            depth=expand("depth"),
            lod_min=expand("lod_min"),
            lod_max=expand("lod_max"),
            entity_state=self._state(sampled),
            entity_mask=expand("entity_mask"),
            edge_source=edges["edge_source"],
            edge_target=edges["edge_target"],
            edge_relation=edges["edge_relation"],
            edge_graph=edges["edge_graph"],
            edge_mask=edges["edge_mask"],
            graph_adjacency=edges["graph_adjacency"],
            visible_slots=visible_slots,
        )

        text_features = torch.zeros((batch_size, entities + 1), dtype=torch.float32)
        for index, item in enumerate(sampled):
            text_features[index, :entities] = torch.tensor(item.visible, dtype=torch.float32)
            text_features[index, entities] = 1.0

        prototype = PrototypeBatch(
            structure=structure,
            scene_ids=tuple(scene.scene_id for scene in scenes),
            scene_points=torch.tensor(
                np.stack([item.scene_points for item in sampled]), dtype=torch.float32
            ),
            scene_occupancy=torch.tensor(
                np.stack([item.scene_occupancy for item in sampled]), dtype=torch.float32
            ),
            part_owner=torch.tensor(
                np.stack([item.part_owner for item in sampled]), dtype=torch.int64
            ),
            entity_points=torch.tensor(
                np.stack([item.entity_points for item in sampled]), dtype=torch.float32
            ),
            entity_occupancy=torch.tensor(
                np.stack([item.entity_occupancy for item in sampled]), dtype=torch.float32
            ),
            entity_frames=torch.tensor(
                np.stack([item.entity_frames for item in sampled]), dtype=torch.float32
            ),
            entity_present=present,
            entity_presence_target=present.to(torch.float32),
            text_features=text_features,
        )
        lod_targets = {
            level_key: torch.tensor(
                np.stack([item.lod_occupancy[level_key] for item in sampled]), dtype=torch.float32
            )
            for level_key in (1, 2, 3)
        }
        variant_index = torch.tensor(
            [scene.variant.position for scene in scenes], dtype=torch.int64
        )
        return WholeOrganBatch(
            batch=prototype,
            scenes=tuple(scenes),
            sampled=tuple(sampled),
            lod_targets=lod_targets,
            variant_index=variant_index,
        )
