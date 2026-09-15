"""Step 8: untyped graph attention, the lightweight replacement for the graph transformer.

Step 7 measured what the partitioned relational graph transformer was actually reading.
The answer was narrow: the model responded strongly to **which entities are connected**
and almost not at all to **what the relation between them says**. Flipping every spatial
relation to its opposite moved entity ownership IoU by 0.0002; removing the same edges
cost 0.0406, two hundred times more.

So this encoder keeps the part that was doing the work and drops the rest:

kept
    masked attention over the graph's connectivity, so an entity attends to its
    neighbours and not to unrelated structures;
    the write-protected identity subspace, untouched by every layer;
    endpoint identity, so "A connects to B" and "A connects to C" differ.

dropped
    the relation-type embedding and the typed attention bias;
    head partitioning by graph kind, which existed to keep typed relations separate;
    the depth and width the typed machinery needed.

What remains is ordinary multi-head attention under a graph mask built from the **union**
of the three adjacencies. The union is deliberate: Step 7 found each of the three graphs
contributes when removed, and nothing found that keeping them in separate heads helped.

The encoder must not be able to tell relation types apart. That is a property worth
testing rather than assuming, so :meth:`is_relation_type_blind` reports it and a test
checks that permuting every relation label leaves the output bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from generation.neural.nn.tensors import AWRStructure

__all__ = ["GraphLiteConfig", "UntypedGraphEncoder"]

_NEG_INF = -1.0e9


@dataclass(frozen=True, slots=True)
class GraphLiteConfig:
    """Shape of the untyped encoder.

    Defaults land within 4% of A3L's graph budget. A3L's whole model was 1,094,415
    parameters against A1's 427,655, so its encoder alone was about 666,760; these
    defaults give 692,864. Matching the budget is what makes the comparison a question
    about relational structure rather than about capacity.
    """

    width: int = 256
    identity_width: int = 96
    layers: int = 2
    heads: int = 4
    mlp_ratio: float = 1.0
    include_self: bool = True
    """Whether an entity attends to itself as well as its neighbours.

    An isolated entity would otherwise have nothing to attend to and produce NaN.
    """

    def __post_init__(self) -> None:
        if self.width % self.heads:
            raise ValueError(f"Width {self.width} must divide among {self.heads} heads.")
        if self.identity_width >= self.width:
            raise ValueError(
                f"Identity subspace {self.identity_width} must be narrower than "
                f"width {self.width}."
            )

    @property
    def context_width(self) -> int:
        """Width the encoder is allowed to write. The identity subspace is not."""
        return self.width - self.identity_width

    @property
    def head_width(self) -> int:
        """Width of one attention head."""
        return self.width // self.heads


class _UntypedLayer(nn.Module):
    """Masked multi-head attention and an MLP, both writing only the context slice."""

    def __init__(self, config: GraphLiteConfig) -> None:
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

    def forward(self, latent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Update the context slice of ``latent`` under one shared graph mask."""
        batch, entities, _ = latent.shape
        heads = self.config.heads
        head_width = self.config.head_width

        normed = self.norm_attention(latent)
        q = self.query(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        k = self.key(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        v = self.value(normed).view(batch, entities, heads, head_width).transpose(1, 2)

        logits = torch.einsum("bhid,bhjd->bhij", q, k) / (head_width**0.5)
        # One mask for every head. There is no typed partitioning to give heads
        # different views, which is the whole point of this encoder.
        logits = logits.masked_fill(~mask.unsqueeze(1), _NEG_INF)
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        attended = torch.einsum("bhij,bhjd->bhid", weights, v)
        attended = attended.transpose(1, 2).reshape(batch, entities, self.config.width)

        identity_width = self.config.identity_width
        context = latent[..., identity_width:] + self.out(attended)
        updated = torch.cat([latent[..., :identity_width], context], dim=-1)
        context = context + self.mlp(self.norm_mlp(updated))
        return torch.cat([latent[..., :identity_width], context], dim=-1)


class UntypedGraphEncoder(nn.Module):
    """Contextualises entity latents over the union of the AWR adjacencies."""

    def __init__(self, config: GraphLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([_UntypedLayer(config) for _ in range(config.layers)])

    @staticmethod
    def is_relation_type_blind() -> bool:
        """Report that relation types are unreadable here, so a test can check it."""
        return True

    def build_mask(self, structure: AWRStructure, batch_size: int) -> torch.Tensor:
        """Attention mask of shape ``[B, N, N]`` from the union of the three graphs.

        An entity may attend to anything it shares an edge with, in either direction,
        plus itself. Padded slots are excluded on both sides, so a padded entity can
        neither be attended to nor attend.
        """
        valid = structure.entity_mask
        batched = valid.dim() == 2
        if not batched:
            valid = valid.unsqueeze(0)
        entities = valid.shape[1]
        adjacency = structure.graph_adjacency
        if adjacency.dim() == 3:
            adjacency = adjacency.unsqueeze(0)
        # Union over the graph axis. Direction is symmetrised: a relation and its
        # inverse are the same edge read two ways, and Step 7 showed the encoder already
        # treated them identically.
        linked = adjacency.any(dim=1)
        linked = linked | linked.transpose(-1, -2)
        if self.config.include_self:
            eye = torch.eye(entities, dtype=torch.bool, device=valid.device).unsqueeze(0)
            linked = linked | eye
        pair = valid.unsqueeze(2) & valid.unsqueeze(1)
        mask = linked & pair
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        return mask

    def forward(self, entity_latent: torch.Tensor, structure: AWRStructure) -> torch.Tensor:
        """Return contextualised latents, identity subspace untouched."""
        mask = self.build_mask(structure, entity_latent.shape[0])
        latent = entity_latent
        for layer in self.layers:
            latent = layer(latent, mask)
        return latent

    def parameter_groups(self) -> dict[str, int]:
        """Parameter count, for the capacity-matched comparison."""
        return {"graph": sum(p.numel() for p in self.parameters())}
