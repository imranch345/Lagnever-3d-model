"""Batch construction: AWR scenes and sampled geometry into PyTorch tensors.

The AWR features come from the Step 5 extractor, unchanged, so the tensors the model
sees are the tensors the contracts describe. What this module adds is the parts a
training loop needs: torch conversion, batching, dense graph masks, and the sampled
query points with their exact labels.

One efficiency decision worth stating: the AWR feature payload depends only on the
ontology and the scene's level of detail, not on its geometry, so it is built once
per level and cached. Scenes within a level share it by construction, which is also
a useful invariant: two scenes at the same level really do present the model with the
same anatomical structure and differ only in shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from awr.relationships import GraphKind
from awr.scene import AWRScene
from datasets.synthetic.heart_corpus import SceneSpec
from datasets.synthetic.sampling import SampledScene, SamplingConfig, sample_scene
from generation.neural.contracts import PlainArray
from generation.neural.features import AWRFeatureExtractor
from generation.neural.graph_encoder import HeadAllocation

__all__ = ["AWRStructure", "PrototypeBatch", "BatchBuilder"]


def _to_tensor(array: PlainArray) -> torch.Tensor:
    dtype = {
        "bool": torch.bool,
        "int32": torch.int64,
        "int64": torch.int64,
        "float32": torch.float32,
        "float16": torch.float32,
    }[str(array.dtype)]
    flat = torch.tensor(array.data, dtype=dtype)
    return flat.reshape(array.shape)


@dataclass(frozen=True, slots=True)
class AWRStructure:
    """The level-of-detail-specific AWR structure shared by a batch of scenes.

    Tensors are either shared across the batch (Step 6, one ontology graph for every
    scene) or per scene (Step 7, a graph measured from each organ). ``is_batched`` says
    which, and every consumer handles both.
    """

    lod: int
    entity_ids: torch.Tensor
    entity_type: torch.Tensor
    semantic_role: torch.Tensor
    laterality: torch.Tensor
    parent_index: torch.Tensor
    depth: torch.Tensor
    lod_min: torch.Tensor
    lod_max: torch.Tensor
    entity_state: torch.Tensor
    entity_mask: torch.Tensor
    edge_source: torch.Tensor
    edge_target: torch.Tensor
    edge_relation: torch.Tensor
    edge_graph: torch.Tensor
    edge_mask: torch.Tensor
    graph_adjacency: torch.Tensor
    visible_slots: torch.Tensor

    @property
    def entity_count(self) -> int:
        """Number of entity slots, whether the structure is shared or per scene."""
        return int(self.entity_ids.shape[-1])

    @property
    def is_batched(self) -> bool:
        """Whether this structure carries one graph per scene."""
        return self.entity_ids.dim() == 2

    @property
    def visible_count(self) -> int:
        """Number of entities visible by default at this level of detail."""
        return int(self.visible_slots.shape[0])


@dataclass(slots=True)
class PrototypeBatch:
    """One training batch: AWR structure, sampled points and exact labels."""

    structure: AWRStructure
    scene_ids: tuple[str, ...]
    scene_points: torch.Tensor
    scene_occupancy: torch.Tensor
    part_owner: torch.Tensor
    entity_points: torch.Tensor
    entity_occupancy: torch.Tensor
    entity_frames: torch.Tensor
    entity_present: torch.Tensor
    entity_presence_target: torch.Tensor
    text_features: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of scenes."""
        return int(self.scene_points.shape[0])

    def to(self, device: torch.device) -> PrototypeBatch:
        """Move every tensor to a device."""

        def move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device)

        structure = AWRStructure(
            lod=self.structure.lod,
            entity_ids=move(self.structure.entity_ids),
            entity_type=move(self.structure.entity_type),
            semantic_role=move(self.structure.semantic_role),
            laterality=move(self.structure.laterality),
            parent_index=move(self.structure.parent_index),
            depth=move(self.structure.depth),
            lod_min=move(self.structure.lod_min),
            lod_max=move(self.structure.lod_max),
            entity_state=move(self.structure.entity_state),
            entity_mask=move(self.structure.entity_mask),
            edge_source=move(self.structure.edge_source),
            edge_target=move(self.structure.edge_target),
            edge_relation=move(self.structure.edge_relation),
            edge_graph=move(self.structure.edge_graph),
            edge_mask=move(self.structure.edge_mask),
            graph_adjacency=move(self.structure.graph_adjacency),
            visible_slots=move(self.structure.visible_slots),
        )
        return PrototypeBatch(
            structure=structure,
            scene_ids=self.scene_ids,
            scene_points=move(self.scene_points),
            scene_occupancy=move(self.scene_occupancy),
            part_owner=move(self.part_owner),
            entity_points=move(self.entity_points),
            entity_occupancy=move(self.entity_occupancy),
            entity_frames=move(self.entity_frames),
            entity_present=move(self.entity_present),
            entity_presence_target=move(self.entity_presence_target),
            text_features=move(self.text_features),
        )


class BatchBuilder:
    """Turns scene specifications into batches, caching the AWR structure per level."""

    def __init__(
        self,
        ontology: AnatomyOntology,
        config: DomainConfig,
        *,
        sampling: SamplingConfig | None = None,
        heads: HeadAllocation | None = None,
    ) -> None:
        self.ontology = ontology
        self.config = config
        self.sampling = sampling or SamplingConfig()
        self.heads = heads or HeadAllocation.default()
        self.extractor = AWRFeatureExtractor(ontology)
        self.slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        self._structures: dict[int, AWRStructure] = {}

    # ------------------------------------------------------------------
    def vocabulary_sizes(self) -> Mapping[str, int]:
        """Vocabulary sizes for building embedding tables."""
        return self.extractor.vocabulary_sizes()

    def inverse_relation_table(self) -> torch.Tensor:
        """Relation id to inverse relation id, or -1 when there is no inverse.

        Symmetric relations map to themselves. This is what makes the encoder's
        reverse-direction bias use a different symbol from the forward direction.
        """
        registry = self.ontology.registry
        table = torch.full((len(self.extractor.relations),), -1, dtype=torch.int64)
        for index, name in enumerate(self.extractor.relations.symbols):
            spec = registry.get(name)
            if spec.symmetric:
                table[index] = index
            elif spec.inverse is not None:
                table[index] = self.extractor.relations.index(spec.inverse)
        return table

    def structure_for_lod(self, lod: int) -> AWRStructure:
        """AWR structure tensors for one level of detail, cached."""
        if lod in self._structures:
            return self._structures[lod]
        scene = AWRScene.from_ontology(
            self.ontology,
            lod_ladder=self.config.lod_ladder,
            educational_levels=self.config.educational_levels,
            active_lod=lod,
            educational_level=self.config.scene_defaults.educational_level,
            name=self.config.scene_defaults.scene_name,
            domain_id=self.config.domain.id,
            default_opacity=self.config.scene_defaults.default_opacity,
        )
        payload = self.extractor.encode_batch([scene])
        tensors = {name: _to_tensor(array)[0] for name, array in payload.items()}

        entity_count = int(tensors["entity_ids"].shape[0])
        adjacency = torch.zeros((len(GraphKind), entity_count, entity_count), dtype=torch.bool)
        for graph_index, kind in enumerate(GraphKind):
            dense = self.extractor.adjacency(payload, graph=kind)
            matrix = torch.tensor(dense, dtype=torch.int64)
            adjacency[graph_index] = matrix >= 0
        visible = torch.tensor(
            [
                self.slot_of[entity.entity_id]
                for entity in scene.iter_entities()
                if entity.visibility
            ],
            dtype=torch.int64,
        )
        structure = AWRStructure(
            lod=lod,
            entity_ids=tensors["entity_ids"],
            entity_type=tensors["entity_type"],
            semantic_role=tensors["semantic_role"],
            laterality=tensors["laterality"],
            parent_index=tensors["parent_index"],
            depth=tensors["depth"],
            lod_min=tensors["lod_min"],
            lod_max=tensors["lod_max"],
            entity_state=tensors["entity_state"],
            entity_mask=tensors["entity_mask"],
            edge_source=tensors["edge_source"],
            edge_target=tensors["edge_target"],
            edge_relation=tensors["edge_relation"],
            edge_graph=tensors["edge_graph"],
            edge_mask=tensors["edge_mask"],
            graph_adjacency=adjacency,
            visible_slots=visible,
        )
        self._structures[lod] = structure
        return structure

    # ------------------------------------------------------------------
    def _text_features(self, sampled: SampledScene, entity_count: int) -> torch.Tensor:
        """Deterministic text features: a bag of ontology terms for the request.

        This is the **oracle/control language path**, not a learned language model.
        Section 4 of the Step 6 brief allows a deterministic text representation for
        controlled experiments; what is learned on top of it is the mapping from these
        features to the requested entity set, which is the language-anatomy alignment
        objective. It is labelled as an oracle everywhere it appears.
        """
        features = torch.zeros(entity_count + 1, dtype=torch.float32)
        for slot in sampled.entity_slots:
            features[slot] = 1.0
        features[entity_count] = 1.0
        return features

    def build(self, specs: Sequence[SceneSpec], *, epoch: int = 0) -> PrototypeBatch:
        """Build a batch. Every scene must share one level of detail."""
        if not specs:
            raise ValueError("A batch needs at least one scene.")
        lods = {spec.active_lod for spec in specs}
        if len(lods) != 1:
            raise ValueError(
                f"A batch must share one level of detail, got {sorted(lods)}. Batches are "
                "bucketed by level so the visible entity set is constant."
            )
        structure = self.structure_for_lod(next(iter(lods)))
        entity_count = structure.entity_count

        sampled = [sample_scene(spec, self.ontology, self.sampling, epoch=epoch) for spec in specs]
        scene_points = torch.tensor(
            np.stack([item.scene_points for item in sampled]), dtype=torch.float32
        )
        scene_occupancy = torch.tensor(
            np.stack([item.scene_occupancy for item in sampled]), dtype=torch.float32
        )
        part_owner = torch.tensor(
            np.stack([item.part_owner for item in sampled]), dtype=torch.int64
        )
        entity_points = torch.tensor(
            np.stack([item.entity_points for item in sampled]), dtype=torch.float32
        )
        entity_occupancy = torch.tensor(
            np.stack([item.entity_occupancy for item in sampled]), dtype=torch.float32
        )
        entity_frames = torch.tensor(
            np.stack([item.entity_frames for item in sampled]), dtype=torch.float32
        )
        entity_present = torch.tensor(
            np.stack([item.visible_mask for item in sampled]), dtype=torch.bool
        )
        text_features = torch.stack(
            [self._text_features(item, entity_count) for item in sampled]
        )
        return PrototypeBatch(
            structure=structure,
            scene_ids=tuple(spec.scene_id for spec in specs),
            scene_points=scene_points,
            scene_occupancy=scene_occupancy,
            part_owner=part_owner,
            entity_points=entity_points,
            entity_occupancy=entity_occupancy,
            entity_frames=entity_frames,
            entity_present=entity_present,
            entity_presence_target=entity_present.to(torch.float32),
            text_features=text_features,
        )


@lru_cache(maxsize=8)
def _cached_builder(ontology_id: str) -> None:  # pragma: no cover - placeholder guard
    """Reserved. Builders hold mutable caches and are not shared across processes."""
    raise NotImplementedError("Builders are constructed per process, not cached globally.")
