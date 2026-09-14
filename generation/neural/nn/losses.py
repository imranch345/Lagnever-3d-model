"""The seven first-wave training objectives from Step 5, implemented.

Only those seven. The deferred objectives (topology, multi-view, cross-modal,
animation, style invariance) are not implemented and not approximated, and the three
relationship objectives are **held out of training** so the experiment measures
something the model was not optimised for.

Two objectives are structural and have no counterpart in the appearance-driven arm:
``entity_identity`` needs an entity axis and ``entity_frame`` needs canonical frames.
Rather than inventing a fake equivalent, the loss reports them as not applicable and
the experiment report states the asymmetry as a threat to validity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as functional_ops

from awr.errors import ContractError
from generation.neural.losses import LOSSES, LossRole
from generation.neural.nn.model import ModelOutput
from generation.neural.nn.tensors import PrototypeBatch

__all__ = ["FIRST_WAVE_OBJECTIVES", "LossInputs", "LossBreakdown", "PrototypeLoss"]

FIRST_WAVE_OBJECTIVES: tuple[str, ...] = (
    "language_anatomy_alignment",
    "entity_identity",
    "geometry_reconstruction",
    "semantic_part_correspondence",
    "entity_frame",
    "lod_consistency",
    "edit_consistency",
)
"""The objectives Step 5 selected for the first experiment, and the only ones here."""

_HELD_OUT = {
    spec.name for spec in LOSSES if spec.role_in_first_experiment is LossRole.EVALUATE_ONLY
}


def validate_weights(weights: Mapping[str, float]) -> None:
    """Reject a loss configuration that trains a held-out or unimplemented objective."""
    declared = {spec.name for spec in LOSSES}
    unknown = sorted(set(weights) - declared)
    if unknown:
        raise ContractError(f"Loss configuration names undeclared objectives {unknown}.")
    forbidden = sorted(name for name in weights if name in _HELD_OUT and weights[name] > 0)
    if forbidden:
        raise ContractError(
            f"Objectives {forbidden} are held out for evaluation in this experiment. Training "
            "on them would make the hypothesis untestable."
        )
    unimplemented = sorted(
        name for name, weight in weights.items() if weight > 0 and name not in FIRST_WAVE_OBJECTIVES
    )
    if unimplemented:
        raise ContractError(
            f"Objectives {unimplemented} are declared but not implemented in the Step 6 "
            f"prototype. Implemented objectives: {list(FIRST_WAVE_OBJECTIVES)}."
        )


@dataclass(slots=True)
class LossInputs:
    """Everything the loss needs, including the optional extra forward passes."""

    output: ModelOutput
    batch: PrototypeBatch
    prefix_output: ModelOutput | None = None
    edited_output: ModelOutput | None = None
    edited_untouched_mask: torch.Tensor | None = None
    geometry_state_invariant: bool = True


@dataclass(slots=True)
class LossBreakdown:
    """Total loss plus each objective's unweighted value."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    def as_floats(self) -> dict[str, float]:
        """Scalar values for logging."""
        out = {"total": float(self.total.detach())}
        out.update({name: float(value.detach()) for name, value in self.terms.items()})
        return out


class PrototypeLoss(nn.Module):
    """Computes the seven first-wave objectives."""

    hard_negatives: torch.Tensor | None

    def __init__(
        self,
        weights: Mapping[str, float],
        *,
        hard_negatives: torch.Tensor | None = None,
        identity_temperature: float = 0.07,
        alignment_mode: str = "prototype",
    ) -> None:
        super().__init__()
        validate_weights(weights)
        self.weights = dict(weights)
        self.identity_temperature = identity_temperature
        self.alignment_mode = alignment_mode
        if hard_negatives is not None:
            self.register_buffer("hard_negatives", hard_negatives, persistent=False)
        else:
            self.register_buffer("hard_negatives", None, persistent=False)

    # ------------------------------------------------------------------
    def _language_alignment(self, inputs: LossInputs) -> torch.Tensor:
        target = inputs.batch.entity_presence_target[:, : inputs.output.presence_logits.shape[1]]
        return functional_ops.binary_cross_entropy_with_logits(
            inputs.output.presence_logits, target
        )

    def _identity(self, inputs: LossInputs, prototypes: torch.Tensor) -> torch.Tensor:
        alignment = inputs.output.alignment
        if alignment is None:
            raise ContractError("entity_identity needs an entity axis; this arm has none.")
        present = inputs.batch.entity_present
        vocabulary = prototypes.shape[0]
        embeddings = functional_ops.normalize(alignment[:, :vocabulary], dim=-1)
        reference = functional_ops.normalize(prototypes, dim=-1)
        logits = embeddings @ reference.t() / self.identity_temperature
        targets = torch.arange(vocabulary, device=logits.device).expand(logits.shape[0], -1)
        mask = present[:, :vocabulary]
        if not bool(mask.any()):
            return logits.sum() * 0.0
        flat_logits = logits[mask]
        flat_targets = targets[mask]
        loss = functional_ops.cross_entropy(flat_logits, flat_targets)
        if self.hard_negatives is not None:
            negatives = self.hard_negatives[flat_targets]
            gathered = torch.gather(flat_logits, 1, negatives.clamp(min=0))
            gathered = gathered.masked_fill(negatives < 0, -1.0e9)
            positive = flat_logits.gather(1, flat_targets.unsqueeze(1))
            restricted = torch.cat([positive, gathered], dim=1)
            hard_target = torch.zeros(
                restricted.shape[0], dtype=torch.int64, device=restricted.device
            )
            loss = 0.5 * loss + 0.5 * functional_ops.cross_entropy(restricted, hard_target)
        return loss

    def _pairwise_identity(self, inputs: LossInputs) -> torch.Tensor:
        """Ablation A6: in-batch instance pairs instead of prototypes."""
        alignment = inputs.output.alignment
        if alignment is None:
            raise ContractError("A6 alignment needs an entity axis; this arm has none.")
        present = inputs.batch.entity_present
        batch, entities, _ = alignment.shape
        if batch < 2:
            return alignment.sum() * 0.0
        embeddings = functional_ops.normalize(alignment, dim=-1)
        anchor = embeddings[0]
        positive = embeddings[1]
        logits = anchor @ positive.t() / self.identity_temperature
        targets = torch.arange(entities, device=logits.device)
        mask = present[0] & present[1]
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return functional_ops.cross_entropy(logits[mask], targets[mask])

    def _geometry(self, inputs: LossInputs) -> torch.Tensor:
        output, batch = inputs.output, inputs.batch
        scene = functional_ops.binary_cross_entropy_with_logits(
            output.scene_logits, batch.scene_occupancy
        )
        if output.entity_field_logits is None:
            return scene
        present = batch.entity_present.unsqueeze(-1).expand_as(output.entity_field_logits)
        per_entity = functional_ops.binary_cross_entropy_with_logits(
            output.entity_field_logits, batch.entity_occupancy, reduction="none"
        )
        masked = (per_entity * present).sum() / present.sum().clamp_min(1)
        return 0.5 * scene + 0.5 * masked

    def _part(self, inputs: LossInputs) -> torch.Tensor:
        output, batch = inputs.output, inputs.batch
        classes = output.part_logits.shape[-1]
        background = classes - 1
        target = torch.where(batch.part_owner >= 0, batch.part_owner, background)
        return functional_ops.cross_entropy(
            output.part_logits.reshape(-1, classes), target.reshape(-1)
        )

    def _frame(self, inputs: LossInputs) -> torch.Tensor:
        frames = inputs.output.frames
        if frames is None:
            raise ContractError("entity_frame needs predicted frames; this arm has none.")
        present = inputs.batch.entity_present.unsqueeze(-1)
        target = inputs.batch.entity_frames
        pose = (frames[..., :6] - target[..., :6]).abs()
        rotation = (frames[..., 6:] - target[..., 6:]).abs()
        weighted = pose.sum(dim=-1, keepdim=True) + 0.5 * rotation.sum(dim=-1, keepdim=True)
        return (weighted * present).sum() / present.sum().clamp_min(1) / 6.0

    def _lod(self, inputs: LossInputs) -> torch.Tensor:
        if inputs.prefix_output is None:
            return inputs.output.scene_logits.sum() * 0.0
        coarse = inputs.prefix_output.scene_logits
        fine = inputs.output.scene_logits.detach()
        return functional_ops.binary_cross_entropy_with_logits(coarse, torch.sigmoid(fine))

    def _edit(self, inputs: LossInputs) -> torch.Tensor:
        if inputs.geometry_state_invariant or inputs.edited_output is None:
            return inputs.output.scene_logits.sum() * 0.0
        mask = inputs.edited_untouched_mask
        if mask is None:
            mask = torch.ones_like(inputs.output.scene_logits, dtype=torch.bool)
        before = torch.sigmoid(inputs.output.scene_logits.detach())
        after = torch.sigmoid(inputs.edited_output.scene_logits)
        difference = (after - before).abs() * mask
        return difference.sum() / mask.sum().clamp_min(1)

    # ------------------------------------------------------------------
    def forward(self, inputs: LossInputs, prototypes: torch.Tensor | None = None) -> LossBreakdown:
        """Compute the weighted total and every term."""
        terms: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        total = inputs.output.scene_logits.sum() * 0.0

        for name, weight in self.weights.items():
            if weight <= 0.0:
                continue
            try:
                if name == "language_anatomy_alignment":
                    value = self._language_alignment(inputs)
                elif name == "entity_identity":
                    value = (
                        self._pairwise_identity(inputs)
                        if self.alignment_mode == "pairwise"
                        else self._identity(inputs, self._require(prototypes))
                    )
                elif name == "geometry_reconstruction":
                    value = self._geometry(inputs)
                elif name == "semantic_part_correspondence":
                    value = self._part(inputs)
                elif name == "entity_frame":
                    value = self._frame(inputs)
                elif name == "lod_consistency":
                    value = self._lod(inputs)
                elif name == "edit_consistency":
                    value = self._edit(inputs)
                else:  # pragma: no cover - validate_weights rejects this earlier
                    raise ContractError(f"Objective {name!r} has no implementation.")
            except ContractError:
                skipped.append(name)
                continue
            terms[name] = value
            total = total + weight * value
        return LossBreakdown(total=total, terms=terms, skipped=tuple(skipped))

    @staticmethod
    def _require(prototypes: torch.Tensor | None) -> torch.Tensor:
        if prototypes is None:
            raise ContractError("Prototype alignment needs the prototype table.")
        return prototypes
