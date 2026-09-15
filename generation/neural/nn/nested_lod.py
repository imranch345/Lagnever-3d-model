"""Step 8: objectives and metrics that make the level-of-detail prefixes genuinely nested.

Step 7 found the token prefixes were read and used badly. Detail gain was negative at
every arm while the token delta was 0.16, meaning the finer prefixes changed the output
substantially and made it worse. The diagnosis recorded then: nothing in the objective
required the coarse prefix to be a *prefix* of the fine decode rather than an independent
summary, so the model was free to use the first eight tokens for one thing and the next
eight to overwrite it.

The mechanism was already nested. ``OccupancyFieldDecoder`` reads ``tokens[:, :, :K]``, a
literal prefix, and the corpus's own targets nest too: the ten entities exposed at level 1
are a subset of the sixteen at level 2, which are a subset of the twenty at level 3. So
the truth satisfies

    occupancy(L1) subset-of occupancy(L2) subset-of occupancy(L3)

and a representation that behaved like the truth would too. What was missing was an
objective that asked for it.

Three terms, each answering a different failure:

``level_reconstruction``
    Each prefix is scored against **its own** level's target, not the finest one. Step 7
    added this; it is kept because without it a coarse prefix is asked for detail it
    cannot represent.
``containment``
    A point the coarse prefix calls occupied must stay occupied when tokens are added.
    Adding detail may refine a boundary; it may not delete structure.
``preservation``
    Where the coarse prefix is **already correct**, the fine prefix must agree with it.
    Masking by correctness is the point: it forbids breaking what is right while leaving
    the model free to change what is wrong, which is exactly the freedom refinement needs.

Deliberately absent: any per-level projection head. Separate heads would let each level
learn an unrelated output while the prefix structure became decorative, which is the
failure this module exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as functional_ops

__all__ = [
    "NestedLodWeights",
    "containment_loss",
    "preservation_loss",
    "nested_lod_metrics",
]


@dataclass(frozen=True, slots=True)
class NestedLodWeights:
    """Relative weight of each nesting term. Fixed before the Step 8 runs."""

    reconstruction: float = 1.0
    containment: float = 0.5
    preservation: float = 0.5

    def total(self) -> float:
        """Sum, for reporting how much of the objective the nesting terms carry."""
        return self.reconstruction + self.containment + self.preservation


def containment_loss(coarse_logits: torch.Tensor, fine_logits: torch.Tensor) -> torch.Tensor:
    """Penalise occupancy the coarse prefix asserted and the fine prefix withdrew.

    One-sided on purpose. The fine prefix may add occupancy freely, which is what adding
    detail means; it may not remove what the coarse prefix already established.

    The coarse side is detached, so the gradient moves the fine decode toward the coarse
    one rather than dragging the coarse decode down to meet it. Without that, the cheapest
    way to satisfy the term is for both to predict nothing.
    """
    coarse = torch.sigmoid(coarse_logits).detach()
    fine = torch.sigmoid(fine_logits)
    return functional_ops.relu(coarse - fine).mean()


def preservation_loss(
    coarse_logits: torch.Tensor,
    fine_logits: torch.Tensor,
    coarse_target: torch.Tensor,
    fine_target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalise the fine prefix for disagreeing where the coarse prefix was right.

    Args:
        coarse_logits: scene occupancy logits at the coarse prefix.
        fine_logits: the same at the fine prefix.
        coarse_target: the coarse level's occupancy target.
        fine_target: the fine level's occupancy target.
        threshold: probability above which a decode counts as asserting occupancy.

    Two masks, and both are necessary.

    The first keeps points where the coarse decode was **right**, so the term forbids
    breaking what is correct while leaving the fine prefix free to fix what is wrong.

    The second keeps points where the two levels' targets **agree**. Points the finer
    level legitimately adds are exactly where the fine decode is supposed to differ, and
    a term that punished the difference there would be punishing refinement itself. That
    is the mistake this signature exists to avoid; an earlier version took one target and
    scored a correct refinement as a failure.

    """
    coarse = torch.sigmoid(coarse_logits).detach()
    fine = torch.sigmoid(fine_logits)
    was_right = (coarse > threshold) == (coarse_target > threshold)
    levels_agree = (coarse_target > threshold) == (fine_target > threshold)
    keep = was_right & levels_agree
    if not bool(keep.any()):
        return fine.sum() * 0.0
    return ((coarse - fine).abs() * keep.to(fine.dtype)).sum() / keep.sum().clamp_min(1)


@torch.no_grad()
def nested_lod_metrics(
    decodes: Mapping[int, torch.Tensor],
    targets: Mapping[int, torch.Tensor],
    *,
    ownership: Mapping[int, torch.Tensor] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Measure whether the prefixes behave as a nested coarse-to-fine representation.

    Args:
        decodes: scene occupancy logits per level, keyed by level.
        targets: that level's own occupancy target, keyed by level.
        ownership: optional predicted part labels per level, for identity consistency.
        threshold: probability above which a decode asserts occupancy.

    Returns metrics whose names say which pair of levels they compare, because a single
    "consistency" number cannot distinguish a model that is coarse-to-fine from one that
    is merely stable.

    """
    levels = sorted(decodes)
    out: dict[str, float] = {}

    for level in levels:
        predicted = torch.sigmoid(decodes[level]) > threshold
        truth = targets[level] > threshold
        intersection = (predicted & truth).sum(dim=-1).to(torch.float32)
        union = (predicted | truth).sum(dim=-1).to(torch.float32)
        iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
        out[f"lod{level}_iou"] = float(iou.mean())

        # Occupancy IoU is not comparable across levels on its own. A finer level exposes
        # more entities, so its target covers more of the space, and a model that simply
        # over-predicts scores higher there. On this corpus the level-1 target covers
        # about 14% of sampled points and the level-3 target about 55%, so a rising IoU
        # can be an artefact of density rather than evidence of refinement.
        #
        # The trivial predictor calls every point occupied and scores exactly the target
        # density. Reporting that alongside, and the share of the remaining headroom the
        # model takes, makes the comparison across levels honest.
        density = truth.to(torch.float32).mean(dim=-1)
        out[f"lod{level}_target_density"] = float(density.mean())
        headroom = (1.0 - density).clamp_min(1e-6)
        out[f"lod{level}_iou_over_trivial"] = float(((iou - density) / headroom).mean())

    for first, second in zip(levels, levels[1:], strict=False):
        coarse = torch.sigmoid(decodes[first]) > threshold
        fine = torch.sigmoid(decodes[second]) > threshold
        asserted = coarse.sum().clamp_min(1)
        # Containment: of the points the coarse prefix called occupied, how many does the
        # fine prefix still call occupied? 1.0 means nothing was withdrawn.
        out[f"lod{first}_to_{second}_containment"] = float(
            (coarse & fine).sum().to(torch.float32) / asserted
        )
        # Preservation: of the points the coarse prefix got right *and* where the two
        # levels agree on the answer, how many are still right? Restricting to points
        # where the targets agree is essential: the finer level legitimately adds
        # structure, and scoring those as broken would penalise refinement itself.
        coarse_truth = targets[first] > threshold
        fine_truth = targets[second] > threshold
        keep = (coarse == coarse_truth) & (coarse_truth == fine_truth)
        still_right = keep & (fine == fine_truth)
        out[f"lod{first}_to_{second}_preservation"] = float(
            still_right.sum().to(torch.float32) / keep.sum().clamp_min(1)
        )
        # Added detail: points the fine prefix adds that the coarse one did not have.
        out[f"lod{first}_to_{second}_added"] = float(
            (fine & ~coarse).sum().to(torch.float32) / fine.sum().clamp_min(1)
        )
        out[f"lod{first}_to_{second}_detail_gain"] = (
            out[f"lod{second}_iou"] - out[f"lod{first}_iou"]
        )

    if ownership is not None:
        for first, second in zip(levels, levels[1:], strict=False):
            coarse_owner = ownership[first]
            fine_owner = ownership[second]
            owned = coarse_owner >= 0
            if not bool(owned.any()):
                continue
            agree = (coarse_owner == fine_owner) & owned
            out[f"lod{first}_to_{second}_identity_consistency"] = float(
                agree.sum().to(torch.float32) / owned.sum().clamp_min(1)
            )

    if len(levels) >= 2:
        out["lod_detail_gain"] = out[f"lod{levels[-1]}_iou"] - out[f"lod{levels[0]}_iou"]
        # The density-corrected version of the same comparison. This is the one to read.
        out["lod_detail_gain_over_trivial"] = (
            out[f"lod{levels[-1]}_iou_over_trivial"] - out[f"lod{levels[0]}_iou_over_trivial"]
        )
        containments = [
            out[key] for key in out if key.endswith("_containment")
        ]
        preservations = [out[key] for key in out if key.endswith("_preservation")]
        out["lod_containment_mean"] = float(sum(containments) / len(containments))
        out["lod_preservation_mean"] = float(sum(preservations) / len(preservations))
    return out


def prefix_is_nested(schedule: Sequence[int]) -> bool:
    """Whether a token schedule is strictly increasing, so each level extends the last."""
    return all(first < second for first, second in zip(schedule, schedule[1:], strict=False))
