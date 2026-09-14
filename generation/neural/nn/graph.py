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
        self, latent: torch.Tensor, mask: torch.Tensor, bias: torch.Tensor
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

    def forward(self, latent: torch.Tensor, structure: AWRStructure) -> torch.Tensor:
        """Contextualise entity latents, leaving the identity slice untouched."""
        batch_size = latent.shape[0]
        mask = self.build_mask(structure, batch_size)
        bias = self.build_bias(structure, batch_size)
        current = latent
        for layer in self.layers:
            current = layer(current, mask, bias)
        identity_width = self.config.identity_width
        return torch.cat([latent[..., :identity_width], current[..., identity_width:]], dim=-1)
