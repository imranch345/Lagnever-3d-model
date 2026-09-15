"""The local edit head: Step 7 Experiment 8.

Step 6 left persistent geometric editing unimplemented, so the claim that the
per-entity geometry token block buys *local* editing was never tested. This module
supplies the smallest mechanism that could make the claim true, and the two controls
that could show it false.

The head reads the entity latents, the current geometry tokens, and an encoding of the
requested edit, and produces a **delta** on the geometry token block. What differs
between the arms is only which entities the delta is allowed to reach:

``target``
    The delta is multiplied by the target mask, so entities other than the edited one
    keep their token block bit for bit. Locality is a property of the architecture.
``free``
    The same head with the same capacity, applied to every entity. Locality, if it
    appears, was learned rather than built in.

Neither arm can guarantee locality in the *rendered* field, and that is deliberate.
Scene occupancy is composed by competition between entities, so enlarging one structure
legitimately takes points from its neighbours. The locality metric measures the rendered
field, not the token block, precisely so that this real coupling is visible.

The identity subspace is never written. It is read as conditioning and carried through
unchanged, which is what makes an edited entity still the same entity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

__all__ = [
    "EditScope",
    "EditHeadConfig",
    "LocalEditHead",
    "encode_edit",
]

EditScope = Literal["target", "free"]


@dataclass(frozen=True, slots=True)
class EditHeadConfig:
    """Shape and scope of the edit head."""

    entity_width: int
    token_width: int
    tokens: int
    operations: int
    identity_width: int = 96
    hidden: int = 128
    scope: EditScope = "target"

    def __post_init__(self) -> None:
        if self.identity_width > self.entity_width:
            raise ValueError(
                f"Identity subspace {self.identity_width} exceeds entity width "
                f"{self.entity_width}."
            )
        if self.scope not in ("target", "free"):
            raise ValueError(f"Unknown edit scope {self.scope!r}.")


def encode_edit(
    operation_index: Sequence[int] | torch.Tensor,
    magnitude: Sequence[float] | torch.Tensor,
    *,
    operations: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode an edit request as ``[B, operations + 1]``.

    The magnitude rides alongside the one-hot operation rather than being folded into
    it, so that the same operation at two strengths is the same instruction applied
    twice as hard, not two unrelated instructions.
    """
    index = torch.as_tensor(operation_index, dtype=torch.long, device=device)
    scale = torch.as_tensor(magnitude, dtype=torch.float32, device=device)
    if index.dim() != 1 or scale.dim() != 1 or index.shape != scale.shape:
        raise ValueError("Operation indices and magnitudes must be matching 1-D sequences.")
    one_hot = torch.zeros(index.shape[0], operations, device=index.device)
    one_hot.scatter_(1, index.unsqueeze(1), 1.0)
    return torch.cat([one_hot, scale.unsqueeze(1)], dim=1)


class LocalEditHead(nn.Module):
    """Rewrites geometry token blocks in response to an edit instruction."""

    def __init__(self, config: EditHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.instruction = nn.Sequential(
            nn.Linear(config.operations + 1, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
        )
        self.entity_projection = nn.Linear(config.entity_width, config.hidden)
        self.token_projection = nn.Linear(config.token_width, config.hidden)
        self.delta = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.token_width),
        )
        # Start as the identity map. An untrained head must leave the scene alone, so
        # that any measured change is something the head learned to do.
        final = self.delta[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.gate = nn.Linear(config.hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(
        self,
        tokens: torch.Tensor,
        entity_latent: torch.Tensor,
        edit: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return edited geometry tokens, shaped like ``tokens``.

        ``tokens`` is ``[B, N, K, D]``, ``entity_latent`` ``[B, N, W]``, ``edit``
        ``[B, F]`` and ``target_mask`` ``[B, N]`` with 1 on the entity being edited.
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B, N, K, D] tokens, got {tuple(tokens.shape)}.")
        if target_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"Target mask {tuple(target_mask.shape)} does not match "
                f"{tuple(tokens.shape[:2])}."
            )
        instruction = self.instruction(edit).unsqueeze(1).unsqueeze(2)
        entity = self.entity_projection(entity_latent).unsqueeze(2)
        features = self.token_projection(tokens) + entity + instruction
        features = torch.nn.functional.gelu(features)
        delta = self.delta(features)
        gate = torch.sigmoid(self.gate(features))
        update = gate * delta
        if self.config.scope == "target":
            update = update * target_mask.unsqueeze(-1).unsqueeze(-1)
        return torch.as_tensor(tokens + update)

    def parameter_groups(self) -> Mapping[str, int]:
        """Parameter count, for the matched-capacity comparison between arms."""
        return {
            "instruction": sum(p.numel() for p in self.instruction.parameters()),
            "projections": sum(p.numel() for p in self.entity_projection.parameters())
            + sum(p.numel() for p in self.token_projection.parameters()),
            "delta": sum(p.numel() for p in self.delta.parameters()),
            "gate": sum(p.numel() for p in self.gate.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
