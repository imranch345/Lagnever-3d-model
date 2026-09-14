"""Per-entity geometry: tokens, canonical frames and implicit fields.

Implements ADR 0003 and ADR 0006.

* Each entity owns a block of ``K_GEO`` geometry tokens, produced from its entity
  latent and the scene latent, and from nothing else. No entity's tokens can read
  another's, which is what makes local editing local by construction.
* Each entity's field is decoded in its own canonical frame, predicted as
  translation, log-scale and a 6D rotation basis.
* A level of detail reads a **prefix** of the same token block, so detail changes
  cannot change identity.

**Field choice: occupancy.** Occupancy with a binary cross-entropy objective is
chosen over a signed distance field because the tier-0 primitives give exact inside
and outside labels everywhere, while exact signed distances to a union of primitives
are not available analytically. A signed distance field would also need an eikonal
term, which Step 5 deferred. Recorded in the Step 6 report as a prototype decision,
not a final one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE

__all__ = [
    "GeometryConfig",
    "frame_rotation",
    "points_to_local",
    "FrameHead",
    "GeometryTokenGenerator",
    "OccupancyFieldDecoder",
    "compose_scene",
]


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Geometry tokeniser and field decoder sizes."""

    entity_width: int = 256
    scene_width: int = 256
    tokens: int = 32
    token_width: int = 64
    token_layers: int = 2
    token_heads: int = 4
    field_width: int = 128
    field_layers: int = 3
    frequencies: int = 6
    lod_schedule: tuple[int, ...] = LOD_TOKEN_SCHEDULE
    per_lod_blocks: bool = False
    """Ablation A5: separate token blocks per level of detail instead of prefixes."""

    def tokens_at(self, lod: int) -> int:
        """Token prefix length read at a level of detail."""
        index = max(0, min(lod, len(self.lod_schedule) - 1))
        return self.lod_schedule[index]


def frame_rotation(frame: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from the 6D part of a frame tensor.

    Gram-Schmidt on the two stored rows, which is the standard continuous
    parameterisation. Returns ``[..., 3, 3]`` whose rows are the basis vectors, the
    same convention the corpus generator stores.
    """
    first = frame[..., 6:9]
    second = frame[..., 9:12]
    b1 = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    projection = (b1 * second).sum(dim=-1, keepdim=True)
    residual = second - projection * b1
    b2 = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    stacked: torch.Tensor = torch.stack([b1, b2, b3], dim=-2)
    return stacked


def points_to_local(points: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Transform scene-space points into an entity's canonical frame.

    Args:
        points: ``[..., P, 3]`` in scene space.
        frame: ``[..., 12]`` translation, log-scale and 6D rotation.

    Returns:
        ``[..., P, 3]`` normalised local coordinates, where the unit sphere is the
        entity's own extent.

    """
    translation = frame[..., 0:3].unsqueeze(-2)
    scale = frame[..., 3:6].exp().clamp_min(1e-4).unsqueeze(-2)
    rotation = frame_rotation(frame)
    centred = points - translation
    rotated: torch.Tensor = torch.einsum("...pi,...ij->...pj", centred, rotation)
    return rotated / scale


class FrameHead(nn.Module):
    """Predicts each entity's canonical frame from its latent."""

    def __init__(self, entity_width: int, initial_log_scale: float = -1.9) -> None:
        super().__init__()
        self.projection = nn.Linear(entity_width, 12)
        nn.init.zeros_(self.projection.weight)
        with torch.no_grad():
            bias = torch.zeros(12)
            bias[3:6] = initial_log_scale
            bias[6] = 1.0
            bias[10] = 1.0
            self.projection.bias.copy_(bias)

    def forward(self, entity_latent: torch.Tensor) -> torch.Tensor:
        """Return ``[B, N, 12]`` frames."""
        frames: torch.Tensor = self.projection(entity_latent)
        return frames


class _TokenRefinementBlock(nn.Module):
    """Self-attention over one entity's token block, then a small feed-forward."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Refine a token block."""
        normed = self.norm(tokens)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        refined: torch.Tensor = tokens + self.mlp(self.mlp_norm(tokens))
        return refined


class GeometryTokenGenerator(nn.Module):
    """Produces a coarse-to-fine geometry token block per entity."""

    def __init__(self, config: GeometryConfig) -> None:
        super().__init__()
        self.config = config
        levels = len(config.lod_schedule) if config.per_lod_blocks else 1
        self.queries = nn.Parameter(torch.randn(levels, config.tokens, config.token_width) * 0.02)
        self.entity_projection = nn.Linear(config.entity_width, config.token_width)
        self.scene_projection = nn.Linear(config.scene_width, config.token_width)
        self.blocks = nn.ModuleList(
            [
                _TokenRefinementBlock(config.token_width, config.token_heads)
                for _ in range(config.token_layers)
            ]
        )

    def forward(
        self, entity_latent: torch.Tensor, scene_latent: torch.Tensor, *, lod: int
    ) -> torch.Tensor:
        """Return ``[B, N, K, D_GEO]`` geometry tokens."""
        batch, entities, _ = entity_latent.shape
        level = (
            self.config.lod_schedule.index(self.config.tokens_at(lod))
            if self.config.per_lod_blocks
            else 0
        )
        queries = self.queries[level]
        tokens = (
            queries.view(1, 1, -1, self.config.token_width)
            + self.entity_projection(entity_latent).unsqueeze(2)
            + self.scene_projection(scene_latent).view(batch, 1, 1, -1)
        )
        flat: torch.Tensor = tokens.reshape(
            batch * entities, self.config.tokens, self.config.token_width
        )
        for block in self.blocks:
            refined: torch.Tensor = block(flat)
            flat = refined
        return flat.view(batch, entities, self.config.tokens, self.config.token_width)


class OccupancyFieldDecoder(nn.Module):
    """Decodes occupancy at query points from a token block.

    One set of weights, shared across entities. What differs per entity is the token
    block and the canonical frame, which is the point: a shared decoder plus
    per-entity tokens is what makes the representation factorised rather than
    fifty separate networks.
    """

    frequencies: torch.Tensor

    def __init__(self, config: GeometryConfig) -> None:
        super().__init__()
        self.config = config
        encoded_width = 3 + 6 * config.frequencies
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(config.frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.point_projection = nn.Linear(encoded_width, config.field_width)
        self.token_key = nn.Linear(config.token_width, config.field_width)
        self.token_value = nn.Linear(config.token_width, config.field_width)
        layers: list[nn.Module] = []
        for _ in range(config.field_layers):
            layers.extend([nn.Linear(config.field_width, config.field_width), nn.GELU()])
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(config.field_width, 1)

    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        """Positional encoding of query points."""
        scaled: torch.Tensor = points.unsqueeze(-1) * self.frequencies
        features: torch.Tensor = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        flattened = features.flatten(start_dim=-2)
        encoded: torch.Tensor = torch.cat([points, flattened], dim=-1)
        return encoded

    def features(
        self, points: torch.Tensor, tokens: torch.Tensor, *, active_tokens: int
    ) -> torch.Tensor:
        """Conditioned hidden features at each query point.

        Exposed separately because every head that reads the field must read the same
        conditioned features. An auxiliary head fed only the raw point encoding would
        have no access to the scene at all, which would quietly cripple whichever arm
        used it.

        Args:
            points: ``[B, N, P, 3]`` local coordinates.
            tokens: ``[B, N, K, D_GEO]`` token block.
            active_tokens: prefix length read from the block.

        Returns:
            ``[B, N, P, field_width]`` hidden features.

        """
        prefix = tokens[:, :, :active_tokens]
        query = self.point_projection(self.encode_points(points))
        keys = self.token_key(prefix)
        values = self.token_value(prefix)
        logits = torch.einsum("bnpd,bnkd->bnpk", query, keys) / (self.config.field_width**0.5)
        weights = torch.softmax(logits, dim=-1)
        conditioned = query + torch.einsum("bnpk,bnkd->bnpd", weights, values)
        hidden: torch.Tensor = self.trunk(conditioned)
        return hidden

    def forward(
        self, points: torch.Tensor, tokens: torch.Tensor, *, active_tokens: int
    ) -> torch.Tensor:
        """Occupancy logits of shape ``[B, N, P]``."""
        batch, entities, count, _ = points.shape
        hidden = self.features(points, tokens, active_tokens=active_tokens)
        logits: torch.Tensor = self.head(hidden)
        return logits.squeeze(-1).view(batch, entities, count)


def compose_scene(
    entity_logits: torch.Tensor, present: torch.Tensor, *, temperature: float = 8.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose per-entity fields into a scene field and part logits.

    Args:
        entity_logits: ``[B, N, P]`` per-entity occupancy logits.
        present: ``[B, N]`` boolean, which entities are in the scene.
        temperature: sharpness of the soft union.

    Returns:
        ``(scene_logits [B, P], part_logits [B, P, N + 1])``. The extra class is
        background. The part logits *are* the composition, which is how geometry
        correspondence stays exact: the winning channel names the owning entity.

    """
    masked = entity_logits.masked_fill(~present.unsqueeze(-1), -1.0e9)
    scene_logits = torch.logsumexp(masked * temperature, dim=1) / temperature
    background = torch.zeros_like(scene_logits).unsqueeze(-1)
    part_logits = torch.cat([masked.transpose(1, 2), background], dim=-1)
    return scene_logits, part_logits
