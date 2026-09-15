"""Step 8 nested level of detail: the prefix property and the objectives that enforce it.

Step 7 found the token prefixes were read and used badly: detail gain was negative while
the token delta was large, meaning finer prefixes changed the output substantially and
made it worse. The mechanism was already a literal prefix; what was missing was an
objective asking the prefix to behave like a coarse summary.
"""

from __future__ import annotations

import pytest
import torch

from datasets.whole_organ.corpus import LOD_ENTITIES
from generation.neural.nn.geometry import GeometryConfig
from generation.neural.nn.nested_lod import (
    NestedLodWeights,
    containment_loss,
    nested_lod_metrics,
    prefix_is_nested,
    preservation_loss,
)
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE


def _logits(mask: torch.Tensor) -> torch.Tensor:
    return (mask.float() - 0.5) * 20.0


def test_the_token_schedule_is_strictly_increasing() -> None:
    """Each level must extend the last, or "prefix" means nothing."""
    assert prefix_is_nested(LOD_TOKEN_SCHEDULE)


def test_each_level_reads_a_literal_prefix_of_the_next() -> None:
    """tokens[0:K1] must be contained in tokens[0:K2], not a separate projection."""
    config = GeometryConfig()
    counts = [config.tokens_at(level) for level in (1, 2, 3)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)
    tokens = torch.randn(2, 3, config.tokens, config.token_width)
    coarse = tokens[:, :, : counts[0]]
    fine = tokens[:, :, : counts[1]]
    assert torch.equal(fine[:, :, : counts[0]], coarse)


def test_per_level_blocks_are_off_by_default() -> None:
    """Separate blocks per level would make the prefix structure decorative."""
    assert GeometryConfig().per_lod_blocks is False


def test_the_entity_sets_nest_too() -> None:
    """The targets must have the property the representation is asked to have."""
    assert set(LOD_ENTITIES[1]) < set(LOD_ENTITIES[2]) < set(LOD_ENTITIES[3])


def test_ideal_refinement_scores_perfectly() -> None:
    """Adding structure and keeping the old structure is what refinement means."""
    coarse_truth = torch.zeros(2, 100)
    coarse_truth[:, :30] = 1.0
    fine_truth = torch.zeros(2, 100)
    fine_truth[:, :50] = 1.0
    metrics = nested_lod_metrics(
        {1: _logits(coarse_truth > 0.5), 2: _logits(fine_truth > 0.5)},
        {1: coarse_truth, 2: fine_truth},
    )
    assert metrics["lod1_to_2_containment"] == 1.0
    assert metrics["lod1_to_2_preservation"] == 1.0
    assert metrics["lod1_to_2_added"] > 0.0


def test_refinement_is_not_punished_by_the_preservation_term() -> None:
    """The finer level legitimately adds structure; scoring that as broken is a bug.

    An earlier version of this term took one target and marked every point the finer
    level added as a failure, which penalised exactly the behaviour being asked for.
    """
    coarse_truth = torch.zeros(1, 60)
    coarse_truth[:, :20] = 1.0
    fine_truth = torch.zeros(1, 60)
    fine_truth[:, :40] = 1.0
    loss = preservation_loss(
        _logits(coarse_truth > 0.5), _logits(fine_truth > 0.5), coarse_truth, fine_truth
    )
    assert float(loss) == 0.0


def test_withdrawing_coarse_structure_is_penalised() -> None:
    """Adding detail may refine a boundary; it may not delete what was there."""
    coarse_truth = torch.zeros(1, 60)
    coarse_truth[:, :30] = 1.0
    fine_truth = torch.zeros(1, 60)
    fine_truth[:, :40] = 1.0
    deleting = torch.zeros(1, 60)
    deleting[:, 30:40] = 1.0
    loss = containment_loss(_logits(coarse_truth > 0.5), _logits(deleting > 0.5))
    assert float(loss) > 0.2
    metrics = nested_lod_metrics(
        {1: _logits(coarse_truth > 0.5), 2: _logits(deleting > 0.5)},
        {1: coarse_truth, 2: fine_truth},
    )
    assert metrics["lod1_to_2_containment"] == 0.0


def test_containment_does_not_punish_adding_detail() -> None:
    """The term is one-sided: extra occupancy at the fine level is free."""
    coarse = torch.zeros(1, 40)
    coarse[:, :10] = 1.0
    fine = torch.zeros(1, 40)
    fine[:, :25] = 1.0
    assert float(containment_loss(_logits(coarse > 0.5), _logits(fine > 0.5))) == 0.0


def test_the_coarse_side_of_each_term_is_detached() -> None:
    """Otherwise the cheapest way to satisfy nesting is for both levels to predict nothing."""
    coarse = torch.zeros(1, 20, requires_grad=True)
    fine = torch.zeros(1, 20, requires_grad=True)
    target = torch.zeros(1, 20)
    loss = containment_loss(coarse, fine) + preservation_loss(coarse, fine, target, target)
    loss.backward()
    assert coarse.grad is None or float(coarse.grad.abs().sum()) == 0.0


def test_identity_consistency_is_reported_when_ownership_is_supplied() -> None:
    """Entity identity must not change as detail is added."""
    coarse_truth = torch.zeros(1, 20)
    coarse_truth[:, :10] = 1.0
    owner = torch.full((1, 20), -1)
    owner[:, :10] = 3
    metrics = nested_lod_metrics(
        {1: _logits(coarse_truth > 0.5), 2: _logits(coarse_truth > 0.5)},
        {1: coarse_truth, 2: coarse_truth},
        ownership={1: owner, 2: owner},
    )
    assert metrics["lod1_to_2_identity_consistency"] == 1.0
    swapped = owner.clone()
    swapped[:, :5] = 7
    changed = nested_lod_metrics(
        {1: _logits(coarse_truth > 0.5), 2: _logits(coarse_truth > 0.5)},
        {1: coarse_truth, 2: coarse_truth},
        ownership={1: owner, 2: swapped},
    )
    assert changed["lod1_to_2_identity_consistency"] < 1.0


def test_the_nesting_weights_are_declared() -> None:
    """The objective's composition must be visible in the run manifest."""
    weights = NestedLodWeights()
    assert weights.containment > 0.0
    assert weights.preservation > 0.0
    assert weights.total() > weights.reconstruction


def test_raw_detail_gain_is_confounded_by_target_density() -> None:
    """A trivial predictor must not appear to refine.

    A finer level exposes more entities, so its target covers more of the space. On this
    corpus level 1 covers about 14% of sampled points and level 3 about 55%. A model that
    simply calls everything occupied therefore scores a large positive raw detail gain
    while adding nothing at all, which is why the density-corrected number is the one to
    read.
    """
    coarse_truth = torch.zeros(1, 100)
    coarse_truth[:, :14] = 1.0
    fine_truth = torch.zeros(1, 100)
    fine_truth[:, :55] = 1.0
    everything = torch.full((1, 100), 10.0)
    metrics = nested_lod_metrics(
        {1: everything, 3: everything}, {1: coarse_truth, 3: fine_truth}
    )
    assert metrics["lod_detail_gain"] > 0.3, "the confound this test describes is absent"
    assert abs(metrics["lod_detail_gain_over_trivial"]) < 1e-6


def test_the_trivial_baseline_is_the_target_density() -> None:
    """Predicting everything occupied scores exactly the density, by construction."""
    truth = torch.zeros(1, 200)
    truth[:, :50] = 1.0
    metrics = nested_lod_metrics({1: torch.full((1, 200), 10.0)}, {1: truth})
    assert metrics["lod1_target_density"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["lod1_iou"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["lod1_iou_over_trivial"] == pytest.approx(0.0, abs=1e-6)


def test_a_perfect_decode_takes_all_the_headroom() -> None:
    """The corrected measure must reach 1.0 when the decode is exactly right."""
    truth = torch.zeros(1, 200)
    truth[:, :50] = 1.0
    metrics = nested_lod_metrics({1: (truth - 0.5) * 20.0}, {1: truth})
    assert metrics["lod1_iou_over_trivial"] == pytest.approx(1.0, abs=1e-6)
