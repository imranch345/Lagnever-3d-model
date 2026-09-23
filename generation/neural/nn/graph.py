"""The partitioned relational graph transformer.

Implements ADR 0002. Attention heads are partitioned by graph: structure heads may
only attend along structure edges, spatial heads along spatial edges, functional
heads along functional edges, and global heads everywhere. Each head adds a typed
bias derived from the relation embedding of the edge it traverses, and the reverse
direction of an edge uses the **inverse** relation's embedding, so direction is
carried by a different symbol rather than lost.

Two properties are enforced by construction and checked by test:

* the identity slice of the entity latent is bit-identical after encoding,
* padded entity slots neither attend nor are attended to.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from awr.relationships import GraphKind
from generation.neural.graph_encoder import HeadAllocation, HeadRole
from generation.neural.nn.tensors import AWRStructure

__all__ = ["GraphEncoderConfig", "PartitionedGraphEncoder"]

_NEG_INF = -1.0e9


@dataclass(frozen=True, slots=True)
class GraphEncoderConfig:
    """Configuration of the encoder."""

    width: int = 256
    identity_width: int = 96
    layers: int = 4
    heads: HeadAllocation = HeadAllocation({HeadRole.GLOBAL: 1})
    relation_vocabulary: int = 18
    relation_width: int = 64
    relation_values: bool = False
    """Whether a relation contributes a message, not only an attention weight.

    Off reproduces every result up to Step 11, where the measurement was that relation type
    moved the answer by 0.0003 degrees because its only channel was a scalar bias that trained
    to 0.55% of the attention logit scale. On, each relation also adds a learned vector to what
    a neighbour contributes, which is the smallest change that lets a relation say something
    rather than only say it louder.
    """
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def context_width(self) -> int:
        """Width the encoder is allowed to write."""
        return self.width - self.identity_width

    @property
    def head_width(self) -> int:
        """Width of one attention head."""
        return self.width // self.heads.total


@dataclass(frozen=True, slots=True)
class _RelationRouting:
    """Where each live relation sits, precomputed once and reused by every layer.

    The same index lists :meth:`PartitionedGraphEncoder.build_bias` walks, kept so the value
    term is routed exactly as the bias term is: per head, respecting each head's graph scope,
    with the reverse direction carrying the inverse relation.
    """

    scene: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    relation: torch.Tensor


class _GraphAttentionLayer(nn.Module):
    """One layer: masked multi-head attention with typed bias, then an MLP.

    Both sublayers write only to the context slice.
    """

    def __init__(self, config: GraphEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_attention = nn.LayerNorm(config.width)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.query = nn.Linear(config.width, config.width)
        self.key = nn.Linear(config.width, config.width)
        self.value = nn.Linear(config.width, config.width)
        self.out = nn.Linear(config.width, config.context_width)
        hidden = int(config.width * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.context_width),
        )

    def forward(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor,
        routing: _RelationRouting | None = None,
        relation_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update the context slice of ``latent``."""
        batch, entities, _ = latent.shape
        heads = self.config.heads.total
        head_width = self.config.head_width

        normed = self.norm_attention(latent)
        q = self.query(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        k = self.key(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        v = self.value(normed).view(batch, entities, heads, head_width).transpose(1, 2)

        logits = torch.einsum("bhid,bhjd->bhij", q, k) / (head_width**0.5)
        logits = logits + bias
        logits = logits.masked_fill(~mask, _NEG_INF)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attended = torch.einsum("bhij,bhjd->bhid", weights, v)
        if routing is not None and relation_value is not None:
            # sum_j alpha_ij r_ij, computed as attention mass per relation times the relation's
            # vector. Accumulating the mass first keeps this the size of [B, H, N, R] instead
            # of a value for every pair.
            vocabulary = relation_value.shape[0]
            mass = torch.zeros(
                (batch, heads, entities, vocabulary), device=latent.device, dtype=weights.dtype
            )
            mass.index_put_(
                (routing.scene, routing.head, routing.query, routing.relation),
                weights[routing.scene, routing.head, routing.query, routing.key],
                accumulate=True,
            )
            attended = attended + torch.einsum(
                "bhir,rhd->bhid", mass, relation_value.view(vocabulary, heads, head_width)
            )
        attended = attended.transpose(1, 2).reshape(batch, entities, self.config.width)

        identity_width = self.config.identity_width
        context = latent[..., identity_width:] + self.out(attended)
        updated = torch.cat([latent[..., :identity_width], context], dim=-1)
        context = context + self.mlp(self.norm_mlp(updated))
        return torch.cat([latent[..., :identity_width], context], dim=-1)


class PartitionedGraphEncoder(nn.Module):
    """Contextualises entity latents over the three AWR graphs."""

    inverse_relations: torch.Tensor
    head_graph: torch.Tensor

    def __init__(self, config: GraphEncoderConfig, inverse_relations: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        if config.width % config.heads.total:
            raise ValueError(
                f"Width {config.width} must divide among {config.heads.total} heads."
            )
        self.register_buffer("inverse_relations", inverse_relations, persistent=False)
        self.relation_bias = nn.Embedding(config.relation_vocabulary, config.heads.total)
        nn.init.zeros_(self.relation_bias.weight)
        # Zero-initialised, and created inside a forked generator, so switching the flag on
        # leaves every other parameter bit-identical. Without the fork the extra embedding
        # would consume random numbers and shift the whole initialisation stream, and the
        # control would differ from the treatment by its seed as well as by the new channel —
        # indistinguishable from the effect, since A3's seed spread is about a degree.
        self.relation_value: nn.Embedding | None = None
        if config.relation_values:
            with torch.random.fork_rng(devices=[]):
                self.relation_value = nn.Embedding(config.relation_vocabulary, config.width)
            nn.init.zeros_(self.relation_value.weight)
        self.layers = nn.ModuleList(
            [_GraphAttentionLayer(config) for _ in range(config.layers)]
        )
        roles = config.heads.roles()
        self.roles = roles
        graph_index = {kind: index for index, kind in enumerate(GraphKind)}
        head_graph = torch.full((len(roles),), -1, dtype=torch.int64)
        for position, role in enumerate(roles):
            graph = role.graph
            if graph is not None:
                head_graph[position] = graph_index[graph]
        self.register_buffer("head_graph", head_graph, persistent=False)

    # ------------------------------------------------------------------
    def build_mask(self, structure: AWRStructure, batch_size: int) -> torch.Tensor:
        """Per-head attention mask of shape ``[B, H, N, N]``.

        A graph-masked head sees an edge in either direction plus the diagonal; a
        global head sees every real entity pair. Padded slots are excluded from both.

        Handles both structure forms: shared across the batch, or one graph per scene,
        which is what the Step 7 corpus provides.
        """
        valid = structure.entity_mask
        batched = valid.dim() == 2
        if not batched:
            valid = valid.unsqueeze(0)
        entities = valid.shape[1]
        pair = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(entities, dtype=torch.bool, device=valid.device).unsqueeze(0)
        adjacency = structure.graph_adjacency
        if adjacency.dim() == 3:
            adjacency = adjacency.unsqueeze(0)
        planes: list[torch.Tensor] = []
        for position in range(len(self.roles)):
            graph = int(self.head_graph[position].item())
            if graph < 0:
                planes.append(pair)
                continue
            edges = adjacency[:, graph]
            planes.append((edges | edges.transpose(1, 2) | (eye & valid.unsqueeze(2))) & pair)
        mask = torch.stack(planes, dim=1)
        return mask if mask.shape[0] == batch_size else mask.expand(batch_size, -1, -1, -1)

    def build_bias(self, structure: AWRStructure, batch_size: int) -> torch.Tensor:
        """Typed relation bias of shape ``[B, H, N, N]``.

        Accumulated per edge, so a pair carrying two relations sums both biases, and
        the reverse direction uses the inverse relation's embedding.
        """
        heads = self.config.heads.total
        batched = structure.edge_mask.dim() == 2
        entity_mask = structure.entity_mask if batched else structure.entity_mask.unsqueeze(0)
        entities = entity_mask.shape[1]
        device = entity_mask.device
        scenes = entity_mask.shape[0] if batched else 1
        bias = torch.zeros((scenes, heads, entities, entities), device=device)

        edge_mask = structure.edge_mask if batched else structure.edge_mask.unsqueeze(0)
        live = edge_mask
        if not bool(live.any()):
            return bias if scenes == batch_size else bias.expand(batch_size, -1, -1, -1)
        edge_source = structure.edge_source if batched else structure.edge_source.unsqueeze(0)
        edge_target = structure.edge_target if batched else structure.edge_target.unsqueeze(0)
        edge_relation = structure.edge_relation if batched else structure.edge_relation.unsqueeze(0)
        edge_graph = structure.edge_graph if batched else structure.edge_graph.unsqueeze(0)
        scene_index, edge_index = live.nonzero(as_tuple=True)
        source = edge_source[scene_index, edge_index]
        target = edge_target[scene_index, edge_index]
        relation = edge_relation[scene_index, edge_index]
        graph = edge_graph[scene_index, edge_index]

        forward_bias = self.relation_bias(relation)
        inverse_ids = self.inverse_relations[relation]
        has_inverse = inverse_ids >= 0
        inverse_bias = self.relation_bias(inverse_ids.clamp(min=0))

        head_graph = self.head_graph
        for position in range(heads):
            head_scope = int(head_graph[position].item())
            if head_scope < 0:
                continue
            selected = graph == head_scope
            if not bool(selected.any()):
                continue
            head_index = torch.full_like(source[selected], position)
            bias.index_put_(
                (
                    scene_index[selected],
                    head_index,
                    source[selected],
                    target[selected],
                ),
                forward_bias[selected, position],
                accumulate=True,
            )
            reverse = selected & has_inverse
            if bool(reverse.any()):
                head_index_rev = torch.full_like(source[reverse], position)
                bias.index_put_(
                    (
                        scene_index[reverse],
                        head_index_rev,
                        target[reverse],
                        source[reverse],
                    ),
                    inverse_bias[reverse, position],
                    accumulate=True,
                )
        return bias if scenes == batch_size else bias.expand(batch_size, -1, -1, -1)

    def build_routing(
        self, structure: AWRStructure, batch_size: int
    ) -> _RelationRouting | None:
        """Index lists placing each live relation on the pairs its head may see.

        Built once and reused by every layer, and deliberately the same walk
        :meth:`build_bias` performs: a head only carries edges of its own graph kind, and the
        reverse direction carries the inverse relation where one exists. If the value term were
        routed differently from the bias term the two would disagree about what a relation is.
        """
        if self.relation_value is None:
            return None
        batched = structure.edge_mask.dim() == 2
        edge_mask = structure.edge_mask if batched else structure.edge_mask.unsqueeze(0)
        if not bool(edge_mask.any()):
            return None
        edge_source = structure.edge_source if batched else structure.edge_source.unsqueeze(0)
        edge_target = structure.edge_target if batched else structure.edge_target.unsqueeze(0)
        edge_relation = structure.edge_relation if batched else structure.edge_relation.unsqueeze(0)
        edge_graph = structure.edge_graph if batched else structure.edge_graph.unsqueeze(0)
        scene_index, edge_index = edge_mask.nonzero(as_tuple=True)
        source = edge_source[scene_index, edge_index]
        target = edge_target[scene_index, edge_index]
        relation = edge_relation[scene_index, edge_index]
        graph = edge_graph[scene_index, edge_index]
        inverse_ids = self.inverse_relations[relation]
        has_inverse = inverse_ids >= 0

        scenes: list[torch.Tensor] = []
        heads: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        relations: list[torch.Tensor] = []
        for position in range(self.config.heads.total):
            head_scope = int(self.head_graph[position].item())
            if head_scope < 0:
                continue
            selected = graph == head_scope
            if bool(selected.any()):
                scenes.append(scene_index[selected])
                heads.append(torch.full_like(source[selected], position))
                queries.append(source[selected])
                keys.append(target[selected])
                relations.append(relation[selected])
            reverse = selected & has_inverse
            if bool(reverse.any()):
                scenes.append(scene_index[reverse])
                heads.append(torch.full_like(source[reverse], position))
                queries.append(target[reverse])
                keys.append(source[reverse])
                relations.append(inverse_ids[reverse])
        if not scenes:
            return None
        routing = _RelationRouting(
            scene=torch.cat(scenes),
            head=torch.cat(heads),
            query=torch.cat(queries),
            key=torch.cat(keys),
            relation=torch.cat(relations),
        )
        if edge_mask.shape[0] == batch_size:
            return routing
        # One structure shared by the whole batch: repeat the routing across scenes.
        offsets = torch.arange(batch_size, device=routing.scene.device)
        count = routing.scene.shape[0]
        return _RelationRouting(
            scene=offsets.repeat_interleave(count),
            head=routing.head.repeat(batch_size),
            query=routing.query.repeat(batch_size),
            key=routing.key.repeat(batch_size),
            relation=routing.relation.repeat(batch_size),
        )

    def forward(self, latent: torch.Tensor, structure: AWRStructure) -> torch.Tensor:
        """Contextualise entity latents, leaving the identity slice untouched."""
        batch_size = latent.shape[0]
        mask = self.build_mask(structure, batch_size)
        bias = self.build_bias(structure, batch_size)
        routing = self.build_routing(structure, batch_size)
        weight = None if self.relation_value is None else self.relation_value.weight
        current = latent
        for layer in self.layers:
            current = layer(current, mask, bias, routing, weight)
        identity_width = self.config.identity_width
        return torch.cat([latent[..., :identity_width], current[..., identity_width:]], dim=-1)
