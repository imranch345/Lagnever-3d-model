"""Proposed encoder for the three anatomical graphs.

STATUS: PROPOSED, UNVALIDATED. The head allocation and bias construction here are
implemented and tested as *structure*; no weights exist.

The requirement that drives the design
--------------------------------------

``left_ventricle --pumps_to--> aorta`` must not be representable in the same way
as ``left_ventricle --adjacent_to--> aorta``. That rules out any encoder where a
relation only decides *whether* two entities attend to each other. The relation
type has to change *what* is communicated.

Proposed encoder
----------------

A single **relational graph transformer** over entity tokens, with attention
heads partitioned by graph:

* structure heads attend only along structure edges,
* spatial heads only along spatial edges,
* functional heads only along functional edges,
* global heads attend everywhere, unmasked.

Every head adds a **typed bias** derived from the relation embedding of the edge
it traverses, and direction is preserved because the forward and reverse
directions of an edge use different relation embeddings (an inverse relation is a
different symbol). So the encoder sees, per head, both the connectivity and the
meaning of the connection.

Why one encoder with partitioned heads
--------------------------------------

The scale makes it cheap. Heart Ontology v0.1 has 42 entities and 130 edges, so a
dense 64x64 attention matrix is nothing; there is no efficiency argument for
sparse message passing yet, and there is a real argument for global context,
which message passing only reaches after several hops.

Sharing one encoder across graphs shares the parameters that should be shared
(what an entity is, how anatomy composes) while the head partition keeps the
parts that must stay distinct. Separate per-graph encoders trained independently
are the main alternative and are recorded in the decision record: they triple the
parameters, and they make cross-graph reasoning, such as "the septum is adjacent
to both ventricles *and* separates their flow", a fusion problem instead of an
attention problem.

Write-protected identity
------------------------

The encoder updates the context slice of ``z_entity`` only. The identity slice is
re-attached unchanged after every layer. This is what makes identity preservation
structural rather than something the model has to be trained not to break.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from awr.errors import ContractError, NotYetImplementedError
from awr.relationships import GraphKind
from generation.neural.contracts import (
    CONTRACTS,
    Contract,
    DType,
    PlainArray,
    TensorSpec,
    dim,
)
from generation.neural.features import AWRFeatureExtractor

__all__ = [
    "HeadRole",
    "HeadAllocation",
    "GraphEncoderSpec",
    "GRAPH_ENCODER_CONTRACT",
    "build_attention_mask",
    "AnatomicalGraphEncoder",
]


class HeadRole(StrEnum):
    """What a group of attention heads is allowed to attend along."""

    STRUCTURE = "structure"
    SPATIAL = "spatial"
    FUNCTIONAL = "functional"
    GLOBAL = "global"
    HIERARCHY = "hierarchy"

    @property
    def graph(self) -> GraphKind | None:
        """Graph this role is masked to, or ``None`` when it is unmasked."""
        mapping = {
            HeadRole.STRUCTURE: GraphKind.STRUCTURE,
            HeadRole.SPATIAL: GraphKind.SPATIAL,
            HeadRole.FUNCTIONAL: GraphKind.FUNCTIONAL,
        }
        return mapping.get(self)


@dataclass(frozen=True, slots=True)
class HeadAllocation:
    """How attention heads are divided between graphs.

    The allocation is a hypothesis, not a result. It is declared as data so that
    an ablation can change it in configuration rather than in code.
    """

    counts: Mapping[HeadRole, int]

    def __post_init__(self) -> None:
        if any(count < 0 for count in self.counts.values()):
            raise ContractError("Head counts must be non-negative.")
        if self.total == 0:
            raise ContractError("A head allocation must declare at least one head.")

    @property
    def total(self) -> int:
        """Total number of heads."""
        return sum(self.counts.values())

    def roles(self) -> tuple[HeadRole, ...]:
        """Role of each head, in head order."""
        out: list[HeadRole] = []
        for role in HeadRole:
            out.extend([role] * self.counts.get(role, 0))
        return tuple(out)

    def validate_against(self, dimension: str = "N_HEAD") -> None:
        """Check the allocation matches the declared head count."""
        declared = dim(dimension).size
        if declared is not None and declared != self.total:
            raise ContractError(
                f"Head allocation totals {self.total} but {dimension!r} is declared as "
                f"{declared}."
            )

    @classmethod
    def default(cls) -> HeadAllocation:
        """Proposed prototype allocation: two heads per graph, two global."""
        return cls(
            {
                HeadRole.STRUCTURE: 2,
                HeadRole.SPATIAL: 2,
                HeadRole.FUNCTIONAL: 2,
                HeadRole.GLOBAL: 2,
            }
        )

    def table(self) -> str:
        """Markdown table of the allocation."""
        lines = ["| head role | heads | attends along |", "| --- | --- | --- |"]
        for role in HeadRole:
            count = self.counts.get(role, 0)
            if not count:
                continue
            graph = role.graph
            scope = f"{graph} edges" if graph else "all entity pairs"
            lines.append(f"| `{role}` | {count} | {scope} |")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GraphEncoderSpec:
    """Configuration of the proposed anatomical graph encoder."""

    layers: int = 4
    heads: HeadAllocation = field(default_factory=lambda: HeadAllocation.default())
    entity_width: int = 256
    relation_width: int = 64
    mlp_ratio: float = 4.0
    residual_identity: bool = True
    share_relation_embeddings_across_layers: bool = True
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ContractError("The graph encoder needs at least one layer.")
        if self.entity_width % self.heads.total != 0:
            raise ContractError(
                f"Entity width {self.entity_width} must divide evenly among "
                f"{self.heads.total} heads."
            )

    @property
    def head_width(self) -> int:
        """Width of one attention head."""
        return self.entity_width // self.heads.total


GRAPH_ENCODER_CONTRACT = CONTRACTS.register(
    Contract(
        name="graph_encoder_io",
        purpose="Inputs and outputs of the anatomical graph encoder.",
        specs=(
            TensorSpec(
                "entity_mask",
                ("B", "N_ENT"),
                DType.BOOL,
                "True for real entity slots.",
            ),
            TensorSpec(
                "attention_mask",
                ("B", "N_HEAD", "N_ENT", "N_ENT"),
                DType.BOOL,
                "Per-head connectivity. A head masked to a graph may attend only along that "
                "graph's edges, plus the diagonal.",
                mask="entity_mask",
                padding_value=False,
            ),
            TensorSpec(
                "relation_bias",
                ("B", "N_HEAD", "N_ENT", "N_ENT"),
                DType.FLOAT32,
                "Additive attention bias projected from the relation embedding of each edge. "
                "This is what makes pumps_to and adjacent_to different messages.",
                normalization="unbounded logit space",
                mask="entity_mask",
                padding_value=0.0,
            ),
            TensorSpec(
                "z_entity_in",
                ("B", "N_ENT", "D_ENT"),
                DType.FLOAT32,
                "Composed entity latents before contextualisation.",
                mask="entity_mask",
            ),
            TensorSpec(
                "z_entity_out",
                ("B", "N_ENT", "D_ENT"),
                DType.FLOAT32,
                "Contextualised entity latents. The identity slice is bit-identical to the input.",
                mask="entity_mask",
            ),
        ),
    )
)
"""Contract for graph encoder inputs and outputs."""


def build_attention_mask(
    payload: Mapping[str, PlainArray],
    extractor: AWRFeatureExtractor,
    allocation: HeadAllocation | None = None,
    *,
    batch_index: int = 0,
    include_self: bool = True,
    symmetric_reachability: bool = True,
) -> list[list[list[bool]]]:
    """Build the per-head attention mask for one scene.

    Implemented because it is pure bookkeeping over the AWR, and because it is the
    part of the encoder that is easy to get subtly wrong: a head that silently
    attends everywhere would erase the distinction the architecture is built on.

    Args:
        payload: An AWR batch from :class:`AWRFeatureExtractor`.
        extractor: The extractor that produced the batch, for its vocabularies.
        allocation: Head allocation; defaults to the proposed prototype split.
        batch_index: Which scene in the batch to build the mask for.
        include_self: Whether every entity may attend to itself.
        symmetric_reachability: Whether a directed edge also permits attention in
            the reverse direction. Direction is still carried by the relation
            bias, which uses the inverse relation's embedding backwards.

    Returns:
        A ``[head][i][j]`` nested list of booleans.

    """
    heads = allocation or HeadAllocation.default()
    size = payload["entity_ids"].shape[1]
    adjacency = {
        kind: extractor.adjacency(payload, graph=kind, batch_index=batch_index)
        for kind in GraphKind
    }
    valid = [bool(payload["entity_mask"].at(batch_index, i)) for i in range(size)]

    mask: list[list[list[bool]]] = []
    for role in heads.roles():
        plane = [[False] * size for _ in range(size)]
        graph = role.graph
        for i in range(size):
            if not valid[i]:
                continue
            if include_self:
                plane[i][i] = True
            for j in range(size):
                if not valid[j]:
                    continue
                if graph is None:
                    plane[i][j] = True
                    continue
                if adjacency[graph][i][j] >= 0:
                    plane[i][j] = True
                elif symmetric_reachability and adjacency[graph][j][i] >= 0:
                    plane[i][j] = True
        mask.append(plane)
    return mask


def relation_bias_indices(
    payload: Mapping[str, PlainArray],
    *,
    batch_index: int = 0,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Relation ids for every ordered, graph-qualified entity pair with an edge.

    Keys are ``(graph, source_slot, target_slot)`` and values are every relation
    id on that pair. A pair can legitimately carry more than one relation, so the
    proposal is that the encoder **sums** their projected biases rather than
    picking one: two structures that both adjoin and connect are more strongly
    related than two that only adjoin.

    Returned sparsely because the graph is sparse. Densifying into
    ``[B, N_HEAD, N_ENT, N_ENT]`` is the model's job.
    """
    out: dict[tuple[int, int, int], list[int]] = {}
    edges = payload["edge_mask"].shape[1]
    for position in range(edges):
        if not payload["edge_mask"].at(batch_index, position):
            continue
        key = (
            int(payload["edge_graph"].at(batch_index, position)),
            int(payload["edge_source"].at(batch_index, position)),
            int(payload["edge_target"].at(batch_index, position)),
        )
        out.setdefault(key, []).append(
            int(payload["edge_relation"].at(batch_index, position))
        )
    return {key: tuple(values) for key, values in out.items()}


class AnatomicalGraphEncoder:
    """Interface for the proposed graph encoder. Declared, not implemented.

    A concrete encoder is a Step 6 deliverable in whichever framework is chosen.
    What is fixed here is the contract it must satisfy, including the identity
    guarantee, so that an implementation can be checked against it.
    """

    encoder_id = "lagnav-graph-encoder-undefined"

    def __init__(self, spec: GraphEncoderSpec | None = None) -> None:
        self.spec = spec or GraphEncoderSpec()
        raise NotYetImplementedError(
            "AnatomicalGraphEncoder",
            planned_in="Step 6, once a framework is chosen",
        )

    def encode(
        self, payload: Mapping[str, PlainArray], *, bindings: Mapping[str, int] | None = None
    ) -> Mapping[str, PlainArray]:  # pragma: no cover - construction raises
        """Contextualise entity latents over the three graphs."""
        raise NotYetImplementedError("AnatomicalGraphEncoder.encode", planned_in="Step 6")

    @staticmethod
    def output_specs() -> Sequence[TensorSpec]:
        """Tensors any implementation must produce."""
        return (GRAPH_ENCODER_CONTRACT.spec("z_entity_out"),)
