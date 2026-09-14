"""AWR to tensor features: the concrete bridge from Step 4 into Step 5.

STATUS: implemented and tested. This is the one part of the proposed neural
stack that can be built today without deciding anything about models, because it
is pure bookkeeping: turning a validated :class:`~awr.scene.AWRScene` into the
index arrays, adjacency and state channels that any model would consume.

Two properties matter more than the code:

* **The entity axis is the identity axis.** Slot ``i`` of every per-entity tensor
  is the same anatomical entity, resolved through a stable
  :class:`EntityCodebook` built from ontology declaration order. Nothing
  downstream has to learn which slot is the left ventricle.
* **Vocabularies are closed and versioned.** Anatomy types, semantic roles,
  relation types, graphs, flow semantics and granularity are all mapped to
  integer ids deterministically, and a codebook records the ontology version it
  was built from, so a mismatch is detectable rather than silently wrong.

State channels are kept *out* of identity on purpose: see
``generation/neural/latent.py`` for why visibility and opacity condition decoding
instead of entering the entity latent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from awr.errors import ContractError
from awr.ontology import AnatomyOntology
from awr.relationships import FlowSemantics, Granularity, GraphKind
from awr.scene import AWRScene
from awr.schema import AnatomyType, EducationalLevel, Laterality
from generation.neural.contracts import (
    CONTRACTS,
    ChannelLayout,
    Contract,
    DType,
    PlainArray,
    TensorSpec,
    dim,
)


def _declared_size(name: str) -> int:
    """Prototype size of a named dimension, which must be statically declared."""
    size = dim(name).size
    if size is None:
        raise ContractError(f"Dimension {name!r} has no declared prototype size.")
    return size

__all__ = [
    "ENTITY_STATE_LAYOUT",
    "AWR_BATCH_CONTRACT",
    "Vocabulary",
    "EntityCodebook",
    "AWRFeatureExtractor",
]


ENTITY_STATE_LAYOUT = ChannelLayout(
    name="entity_state",
    channels=(
        ("visibility", "1.0 when the entity is currently visible."),
        ("opacity", "Opacity in [0, 1]."),
        ("lod_min_norm", "Entity's minimum level of detail, divided by the ladder maximum."),
        ("lod_span_norm", "Upper level-of-detail bound, or 1.0 when unbounded."),
        ("visibility_manual", "1.0 when visibility came from a user command, not the LOD policy."),
        ("animation_playing", "1.0 when the entity is bound to a playing clip."),
        ("animation_time", "Normalised cardiac-cycle time in [0, 1], 0.0 when not animated."),
        ("renderable", "1.0 when the entity can be drawn at all."),
        ("is_group", "1.0 for organisational groups, which are never drawn."),
    ),
)
"""Channel layout of the explicit entity state vector."""


def _entity_tensor(name: str, dtype: DType, semantics: str, padding: object) -> TensorSpec:
    return TensorSpec(
        name=name,
        dims=("B", "N_ENT"),
        dtype=dtype,
        semantics=semantics,
        mask="entity_mask",
        padding_value=padding,
        normalization="none",
    )


def _edge_tensor(name: str, dtype: DType, semantics: str) -> TensorSpec:
    return TensorSpec(
        name=name,
        dims=("B", "E_REL"),
        dtype=dtype,
        semantics=semantics,
        mask="edge_mask",
        padding_value=-1,
        normalization="none",
    )


AWR_BATCH_CONTRACT = CONTRACTS.register(
    Contract(
        name="awr_batch",
        purpose=(
            "One batch of Anatomical World Representation scenes, as the symbolic input to "
            "every neural stage. Produced by AWRFeatureExtractor from a Step 4 scene."
        ),
        specs=(
            TensorSpec(
                "entity_mask",
                ("B", "N_ENT"),
                DType.BOOL,
                "True for real entity slots, False for padding.",
            ),
            _entity_tensor(
                "entity_ids",
                DType.INT32,
                "Index into the entity codebook. The identity axis of the architecture.",
                -1,
            ),
            _entity_tensor("entity_type", DType.INT32, "Anatomy type vocabulary index.", -1),
            _entity_tensor("semantic_role", DType.INT32, "Semantic role vocabulary index.", -1),
            _entity_tensor("laterality", DType.INT32, "Laterality vocabulary index.", -1),
            _entity_tensor(
                "parent_index",
                DType.INT32,
                "Slot index of the parent entity; -1 for the root and for padding.",
                -1,
            ),
            _entity_tensor("depth", DType.INT32, "Depth in the presentation hierarchy.", -1),
            _entity_tensor(
                "lod_min", DType.INT32, "Lowest level of detail that shows the entity.", -1
            ),
            _entity_tensor(
                "lod_max",
                DType.INT32,
                "Highest level of detail that shows the entity; -1 means unbounded.",
                -1,
            ),
            TensorSpec(
                "entity_state",
                ("B", "N_ENT", "N_STATE"),
                DType.FLOAT32,
                "Explicit presentation state. Conditions decoding; never part of identity.",
                normalization="per-channel, see ENTITY_STATE_LAYOUT",
                mask="entity_mask",
                padding_value=0.0,
                value_range=(-1.0, 1.0),
                channels=tuple(name for name, _ in ENTITY_STATE_LAYOUT.channels),
            ),
            TensorSpec(
                "edge_mask", ("B", "E_REL"), DType.BOOL, "True for real edges, False for padding."
            ),
            _edge_tensor("edge_source", DType.INT32, "Entity slot index of the edge subject."),
            _edge_tensor("edge_target", DType.INT32, "Entity slot index of the edge object."),
            _edge_tensor("edge_relation", DType.INT32, "Relation type vocabulary index."),
            _edge_tensor(
                "edge_graph",
                DType.INT32,
                "Graph the edge belongs to: structure, spatial or functional.",
            ),
            _edge_tensor(
                "edge_flow",
                DType.INT32,
                "Flow semantics of the relation type: none, forward or reverse.",
            ),
            _edge_tensor(
                "edge_granularity",
                DType.INT32,
                "Granularity tag: any, summary or detailed. Lets a traversal pick one route.",
            ),
            TensorSpec(
                "active_lod",
                ("B",),
                DType.INT32,
                "Active level of detail of the scene.",
            ),
            TensorSpec(
                "educational_level",
                ("B",),
                DType.INT32,
                "Audience level vocabulary index.",
            ),
        ),
    )
)
"""Contract for a batch of AWR scenes."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """A closed, ordered mapping from symbol to integer id."""

    name: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.symbols)) != len(self.symbols):
            raise ContractError(f"Vocabulary {self.name!r} has duplicate symbols.")

    def __len__(self) -> int:
        return len(self.symbols)

    def index(self, symbol: str) -> int:
        """Return the id of a symbol, or raise listing the vocabulary."""
        try:
            return self.symbols.index(symbol)
        except ValueError:
            raise ContractError(
                f"{self.name}: unknown symbol {symbol!r}. Known: {', '.join(self.symbols)}."
            ) from None

    def symbol(self, index: int) -> str:
        """Return the symbol for an id."""
        if not 0 <= index < len(self.symbols):
            raise ContractError(f"{self.name}: id {index} out of range.")
        return self.symbols[index]

    @classmethod
    def from_enum(cls, name: str, members: Iterable[str]) -> Vocabulary:
        """Build a vocabulary from an enumeration's declaration order."""
        return cls(name, tuple(str(member) for member in members))


@dataclass(frozen=True, slots=True)
class EntityCodebook:
    """Stable mapping from persistent entity id to latent slot index.

    The codebook is the anchor of the whole architecture: it is what the learned
    identity embedding table is indexed by, what multimodal alignment retrieves
    against, and what ties a geometry token block to an anatomical entity.
    """

    ontology_id: str
    ontology_version: str
    entity_ids: tuple[str, ...]

    @classmethod
    def from_ontology(cls, ontology: AnatomyOntology) -> EntityCodebook:
        """Build a codebook in ontology declaration order."""
        return cls(ontology.ontology_id, ontology.version, ontology.ids())

    def __len__(self) -> int:
        return len(self.entity_ids)

    def index(self, entity_id: str) -> int:
        """Latent slot index of an entity."""
        try:
            return self.entity_ids.index(entity_id)
        except ValueError:
            raise ContractError(
                f"Entity {entity_id!r} is not in the codebook for "
                f"{self.ontology_id}@{self.ontology_version}."
            ) from None

    def entity(self, index: int) -> str:
        """Entity id at a slot index."""
        if not 0 <= index < len(self.entity_ids):
            raise ContractError(f"Codebook index {index} out of range.")
        return self.entity_ids[index]

    def matches(self, scene: AWRScene) -> bool:
        """Whether a scene was built from the ontology this codebook came from."""
        return (
            scene.ontology_id == self.ontology_id
            and scene.ontology_version == self.ontology_version
        )


class AWRFeatureExtractor:
    """Turns AWR scenes into contract-checked feature arrays."""

    def __init__(
        self,
        ontology: AnatomyOntology,
        *,
        max_entities: int | None = None,
        max_edges: int | None = None,
    ) -> None:
        self.ontology = ontology
        self.codebook = EntityCodebook.from_ontology(ontology)
        self.max_entities = max_entities or _declared_size("N_ENT")
        self.max_edges = max_edges or _declared_size("E_REL")
        self.anatomy_types = Vocabulary.from_enum("anatomy_type", AnatomyType)
        self.laterality = Vocabulary.from_enum("laterality", Laterality)
        self.educational_levels = Vocabulary.from_enum("educational_level", EducationalLevel)
        self.graphs = Vocabulary.from_enum("graph", GraphKind)
        self.flow = Vocabulary.from_enum("flow", FlowSemantics)
        self.granularity = Vocabulary.from_enum("granularity", Granularity)
        self.relations = Vocabulary("relation", ontology.registry.names())
        self.semantic_roles = Vocabulary(
            "semantic_role",
            tuple(sorted({entity.semantic_role for entity in ontology.iter_entities()})),
        )
        self.max_lod = max(1, ontology.max_lod())

    # ------------------------------------------------------------------
    def vocabulary_sizes(self) -> dict[str, int]:
        """Size of every closed vocabulary, for model configuration."""
        return {
            "entity": len(self.codebook),
            "anatomy_type": len(self.anatomy_types),
            "semantic_role": len(self.semantic_roles),
            "laterality": len(self.laterality),
            "relation": len(self.relations),
            "graph": len(self.graphs),
            "flow": len(self.flow),
            "granularity": len(self.granularity),
            "educational_level": len(self.educational_levels),
        }

    def _state_vector(self, scene: AWRScene, entity_id: str) -> list[float]:
        entity = scene.get(entity_id)
        animation = entity.animation_state
        max_lod = float(self.max_lod)
        upper = entity.lod_policy.max_lod
        return [
            1.0 if entity.visibility else 0.0,
            float(entity.opacity),
            float(entity.lod_policy.min_lod) / max_lod,
            1.0 if upper is None else float(upper) / max_lod,
            1.0 if str(entity.state.visibility_source) == "manual" else 0.0,
            1.0 if animation.playing else 0.0,
            float(animation.normalized_time or 0.0),
            1.0 if entity.renderable else 0.0,
            1.0 if entity.is_group else 0.0,
        ]

    def encode_scene(self, scene: AWRScene) -> dict[str, list[object]]:
        """Encode one scene into per-tensor row values (no batch axis yet)."""
        if not self.codebook.matches(scene):
            raise ContractError(
                f"Scene was built from {scene.ontology_id}@{scene.ontology_version}, but this "
                f"extractor holds a codebook for {self.codebook.ontology_id}@"
                f"{self.codebook.ontology_version}."
            )
        ids = scene.ids()
        if len(ids) > self.max_entities:
            raise ContractError(
                f"Scene holds {len(ids)} entities, more than the {self.max_entities} entity "
                "slots declared by the contract."
            )
        slot_of = {entity_id: position for position, entity_id in enumerate(ids)}
        pad = self.max_entities - len(ids)

        entity_mask: list[object] = [*([True] * len(ids)), *([False] * pad)]
        entity_ids: list[object] = [
            *(self.codebook.index(entity_id) for entity_id in ids),
            *([-1] * pad),
        ]
        entity_type: list[object] = []
        semantic_role: list[object] = []
        laterality: list[object] = []
        parent_index: list[object] = []
        depth: list[object] = []
        lod_min: list[object] = []
        lod_max: list[object] = []
        state: list[object] = []

        for entity_id in ids:
            entity = scene.get(entity_id)
            entity_type.append(self.anatomy_types.index(str(entity.anatomy_type)))
            semantic_role.append(self.semantic_roles.index(entity.semantic_role))
            laterality.append(self.laterality.index(str(entity.definition.laterality)))
            parent = entity.parent_id
            parent_index.append(slot_of[parent] if parent is not None else -1)
            depth.append(len(scene.ancestors(entity_id)))
            lod_min.append(entity.lod_policy.min_lod)
            lod_max.append(-1 if entity.lod_policy.max_lod is None else entity.lod_policy.max_lod)
            state.extend(self._state_vector(scene, entity_id))

        for _ in range(pad):
            entity_type.append(-1)
            semantic_role.append(-1)
            laterality.append(-1)
            parent_index.append(-1)
            depth.append(-1)
            lod_min.append(-1)
            lod_max.append(-1)
            state.extend([0.0] * ENTITY_STATE_LAYOUT.width)

        edges = scene.relationships.all_edges()
        if len(edges) > self.max_edges:
            raise ContractError(
                f"Scene holds {len(edges)} relationship edges, more than the {self.max_edges} "
                "edge slots declared by the contract."
            )
        registry = scene.relationships.registry
        edge_pad = self.max_edges - len(edges)
        edge_mask: list[object] = [*([True] * len(edges)), *([False] * edge_pad)]
        edge_source: list[object] = []
        edge_target: list[object] = []
        edge_relation: list[object] = []
        edge_graph: list[object] = []
        edge_flow: list[object] = []
        edge_granularity: list[object] = []
        for edge in edges:
            spec = registry.get(edge.relation)
            edge_source.append(slot_of[edge.subject])
            edge_target.append(slot_of[edge.object])
            edge_relation.append(self.relations.index(edge.relation))
            edge_graph.append(self.graphs.index(str(spec.graph)))
            edge_flow.append(self.flow.index(str(spec.flow)))
            edge_granularity.append(self.granularity.index(str(edge.granularity)))
        for _ in range(edge_pad):
            edge_source.append(-1)
            edge_target.append(-1)
            edge_relation.append(-1)
            edge_graph.append(-1)
            edge_flow.append(-1)
            edge_granularity.append(-1)

        return {
            "entity_mask": entity_mask,
            "entity_ids": entity_ids,
            "entity_type": entity_type,
            "semantic_role": semantic_role,
            "laterality": laterality,
            "parent_index": parent_index,
            "depth": depth,
            "lod_min": lod_min,
            "lod_max": lod_max,
            "entity_state": state,
            "edge_mask": edge_mask,
            "edge_source": edge_source,
            "edge_target": edge_target,
            "edge_relation": edge_relation,
            "edge_graph": edge_graph,
            "edge_flow": edge_flow,
            "edge_granularity": edge_granularity,
            "active_lod": [scene.active_lod],
            "educational_level": [
                self.educational_levels.index(str(scene.educational_level))
            ],
        }

    def encode_batch(self, scenes: Sequence[AWRScene]) -> dict[str, PlainArray]:
        """Encode several scenes into a contract-valid batch."""
        if not scenes:
            raise ContractError("A batch must contain at least one scene.")
        rows = [self.encode_scene(scene) for scene in scenes]
        batch_size = len(scenes)
        payload: dict[str, PlainArray] = {}
        for spec in AWR_BATCH_CONTRACT:
            values: list[object] = []
            for row in rows:
                values.extend(row[spec.name])
            shape = (batch_size, *spec.resolve({"B": batch_size})[1:])
            payload[spec.name] = PlainArray(shape, spec.dtype, tuple(values))
        AWR_BATCH_CONTRACT.validate_payload(payload, bindings={"B": batch_size})
        return payload

    def adjacency(
        self, payload: Mapping[str, PlainArray], *, graph: GraphKind, batch_index: int = 0
    ) -> list[list[int]]:
        """Dense adjacency of one graph for one scene, as relation ids.

        Entry ``[i][j]`` holds the relation id of the edge from slot ``i`` to slot
        ``j``, or ``-1`` when there is no edge. This is the structure the proposed
        graph encoder turns into a typed attention bias, and having it here means
        the bias can be tested without a model.
        """
        wanted = self.graphs.index(str(graph))
        size = payload["entity_ids"].shape[1]
        dense = [[-1] * size for _ in range(size)]
        edge_count = payload["edge_mask"].shape[1]
        for position in range(edge_count):
            if not payload["edge_mask"].at(batch_index, position):
                continue
            if payload["edge_graph"].at(batch_index, position) != wanted:
                continue
            source = int(payload["edge_source"].at(batch_index, position))
            target = int(payload["edge_target"].at(batch_index, position))
            dense[source][target] = int(payload["edge_relation"].at(batch_index, position))
        return dense
