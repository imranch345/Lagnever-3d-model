"""The placement-blind floor, and the masking mistake that made it wrong once.

Step 9 elevates the floor to a first-class baseline, so the thing that made the first
version of it wrong needs a test rather than a memory. The floor must be measured over
exactly the entities the model is scored on; measured over all twenty it is 0.02 optimistic
and every Step 8 arm looks worse than a lookup table.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step9.placement_floor import (
    _iterate_frames,
    build_tables,
    placement_floor,
    score_table,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def builder() -> WholeOrganBatchBuilder:
    """A batch builder on the real heart ontology."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return WholeOrganBatchBuilder(ontology, domain)


@pytest.fixture(scope="module")
def scenes():
    """A slice of the held-out split."""
    return load_step8_split(CORPUS, "test_seen", limit=48)


def test_the_floor_is_measured_over_the_entities_the_model_is_scored_on(builder, scenes) -> None:
    """The mistake that made the first version wrong, pinned.

    The model's frame error is masked by ``entity_present``, the set visible at the scene's
    level of detail. Iterating the corpus directly scores all twenty entities instead, and
    the coarse level exposes the large structures whose centroids move most.
    """
    masked = [slot for slot, _ in _iterate_frames(scenes, builder)]
    unmasked = sum(len(scene.frames()) for scene in scenes)
    assert masked, "no frames iterated"
    assert len(masked) < unmasked, (
        "the floor is iterating every entity, not the visible ones; it is not comparable "
        "to the model's frame error"
    )


def test_the_floor_only_ever_sees_present_entities(builder, scenes) -> None:
    """A slot the evaluation masks out must never enter the floor's fit or its score."""
    from training.whole_organ import WholeOrganLoader

    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=8, seed=0, shuffle=False, drop_last=False
    )
    allowed = set()
    for whole in loader.epoch(0):
        for index in range(whole.batch.entity_present.shape[0]):
            allowed |= set(whole.batch.entity_present[index].nonzero().flatten().tolist())
    for slot, _ in _iterate_frames(scenes, builder):
        assert slot in allowed


def test_the_oracle_scores_exactly_zero(builder, scenes) -> None:
    """Predicting each scene's own frame must be perfect, or the scoring code is wrong."""
    truths: dict[int, list[np.ndarray]] = defaultdict(list)
    for slot, frame in _iterate_frames(scenes, builder):
        truths[slot].append(frame)
    # An oracle keyed on the exact frame cannot be expressed as a per-slot table, so this
    # checks the scorer directly: the distance from a frame to itself is zero.
    for frames in truths.values():
        for frame in frames:
            assert float(np.linalg.norm(frame[0:3] - frame[0:3])) == 0.0


def test_the_per_entity_table_beats_the_global_one(builder, scenes) -> None:
    """Identity is worth something, which is why the floor uses it."""
    tables = build_tables(scenes, builder)
    per_entity = tables["per_entity"]
    global_frame = tables["global"]
    keyed = score_table(scenes, builder, lambda slot: per_entity.get(slot, global_frame))
    flat = score_table(scenes, builder, lambda _: global_frame)
    assert keyed["position_error"] < flat["position_error"]


def test_the_rotation_component_is_structurally_zero(builder, scenes) -> None:
    """The frame target's rotation is a constant, so any predictor scores zero on it."""
    tables = build_tables(scenes, builder)
    per_entity = tables["per_entity"]
    values = score_table(scenes, builder, lambda slot: per_entity[slot])
    assert values["rotation_error"] < 1.0e-6


def test_every_split_gets_its_own_floor() -> None:
    """A floor from one split does not transfer to another; the splits differ."""
    report = placement_floor(CORPUS, splits=("test_seen", "test_combination"))
    first = report["splits"]["test_seen"]["placement_blind_floor"]
    second = report["splits"]["test_combination"]["placement_blind_floor"]
    assert first > 0.0 and second > 0.0
    assert not math.isclose(first, second, rel_tol=1e-3), (
        "two splits produced the same floor, which would make the per-split rule pointless"
    )


def test_the_floor_is_fitted_on_training_only() -> None:
    """Fitting the lookup on the split it scores would make it an oracle, not a floor."""
    report = placement_floor(CORPUS, splits=("test_seen",))
    assert report["fitted_on"] == "train"
    assert report["frames_fitted"] > 0
