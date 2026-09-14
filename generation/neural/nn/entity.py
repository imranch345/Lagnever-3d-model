"""Entity latent composition with a write-protected identity subspace.

Implements ADR 0001. The entity latent is assembled from the named subspaces the
Step 5 design declares, and the identity slice is written once from the codebook
embedding. Everything downstream receives ``[identity | context]`` and is only
allowed to modify the context part, which is what makes identity preservation a
property of the tensor layout rather than of training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from generation.neural.latent import ENTITY_LATENT_LAYOUT
from generation.neural.nn.tensors import AWRStructure

__all__ = ["EntityLatentConfig", "EntityLatentComposer"]


@dataclass(frozen=True, slots=True)
class EntityLatentConfig:
    """Sizes needed to build the embedding tables."""

    entity_vocabulary: int
    anatomy_type_vocabulary: int
    semantic_role_vocabulary: int
    laterality_vocabulary: int
    max_depth: int = 8
    width: int = 256

    def __post_init__(self) -> None:
        if self.width != ENTITY_LATENT_LAYOUT.width:
            raise ValueError(
                f"Entity latent width {self.width} must match the declared layout "
                f"({ENTITY_LATENT_LAYOUT.width}). The layout is the contract."
            )


class EntityLatentComposer(nn.Module):
    """Builds ``z_entity`` from the AWR's symbolic features.

    The subspace widths and their order come from
    :data:`generation.neural.latent.ENTITY_LATENT_LAYOUT`, so the tensor layout and
    the design document cannot drift apart.

    Structures may be shared across a batch (Step 6, where every scene had the same
    graph) or per scene (Step 7, where the graph is measured from each organ). Both are
    accepted; the difference is whether the AWR tensors carry a leading batch axis.

    The ``geometry_summary`` subspace is fed by a learned constant in this prototype.
    It is designed to carry pooled geometry tokens, which do not exist at the point
    the entity latent is composed in the generation direction. That is a measured
    limitation, recorded in the Step 6 report, not a silent omission.
    """

    def __init__(self, config: EntityLatentConfig) -> None:
        super().__init__()
        self.config = config
        layout = ENTITY_LATENT_LAYOUT
        self.identity_span = layout.span("identity")
        self.type_span = layout.span("type")
        self.hierarchy_span = layout.span("hierarchy")
        self.geometry_span = layout.span("geometry_summary")
        self.free_span = layout.span("free")

        identity_width = self.identity_span[1] - self.identity_span[0]
        type_width = self.type_span[1] - self.type_span[0]
        hierarchy_width = self.hierarchy_span[1] - self.hierarchy_span[0]
        geometry_width = self.geometry_span[1] - self.geometry_span[0]
        free_width = self.free_span[1] - self.free_span[0]

        self.identity = nn.Embedding(config.entity_vocabulary, identity_width)
        self.anatomy_type = nn.Embedding(config.anatomy_type_vocabulary, type_width)
        self.semantic_role = nn.Embedding(config.semantic_role_vocabulary, type_width)
        self.laterality = nn.Embedding(config.laterality_vocabulary, type_width)
        self.depth = nn.Embedding(config.max_depth, hierarchy_width)
        self.parent_projection = nn.Linear(identity_width, hierarchy_width)
        self.no_parent = nn.Parameter(torch.zeros(hierarchy_width))
        self.geometry_unknown = nn.Parameter(torch.zeros(geometry_width))
        self.geometry_projection = nn.Linear(64, geometry_width)
        self.free = nn.Embedding(config.entity_vocabulary, free_width)

    @property
    def identity_width(self) -> int:
        """Width of the protected identity slice."""
        return self.identity_span[1] - self.identity_span[0]

    def identity_slice(self, latent: torch.Tensor) -> torch.Tensor:
        """The protected identity slice of a latent tensor."""
        start, end = self.identity_span
        return latent[..., start:end]

    def forward(
        self,
        structure: AWRStructure,
        batch_size: int,
        *,
        geometry_summary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compose ``z_entity`` of shape ``[B, N_ENT, D_ENT]``."""
        mask = structure.entity_mask
        safe_ids = structure.entity_ids.clamp(min=0)
        identity = self.identity(safe_ids)

        type_vector = (
            self.anatomy_type(structure.entity_type.clamp(min=0))
            + self.semantic_role(structure.semantic_role.clamp(min=0))
            + self.laterality(structure.laterality.clamp(min=0))
        )

        depth = self.depth(structure.depth.clamp(min=0, max=self.config.max_depth - 1))
        parent_index = structure.parent_index
        has_parent = parent_index >= 0
        safe_parent = parent_index.clamp(min=0)
        if identity.dim() == 3:
            # Per-scene structure: gather along the entity axis of each scene.
            gather_index = safe_parent.unsqueeze(-1).expand(-1, -1, identity.shape[-1])
            parent_identity = torch.gather(identity, 1, gather_index)
        else:
            parent_identity = identity[safe_parent]
        hierarchy = depth + torch.where(
            has_parent.unsqueeze(-1),
            self.parent_projection(parent_identity),
            self.no_parent.expand_as(depth),
        )

        if geometry_summary is None:
            geometry = self.geometry_unknown.expand(*identity.shape[:-1], -1)
        else:
            geometry = self.geometry_projection(geometry_summary)

        free = self.free(safe_ids)

        parts = [identity, type_vector, hierarchy, geometry, free]
        composed = torch.cat(parts, dim=-1)
        composed = composed * mask.unsqueeze(-1)
        if composed.dim() == 2:
            composed = composed.unsqueeze(0).expand(batch_size, -1, -1)
        return composed.contiguous()
