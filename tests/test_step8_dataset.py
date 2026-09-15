"""Step 8 corpus: continuous arrangements, hold-out regions and leakage controls.

The Step 7 corpus had four arrangements, which a model can classify. These tests check
the properties that make arrangement classification useless as a shortcut, and that each
held-out split really is held out.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from datasets.whole_organ.arrangement import (
    ARRANGEMENT_FIELDS,
    YAW_EXTRAPOLATION_EDGE,
    YAW_INTERPOLATION_HOLE,
    Arrangement,
    arrangement_distance,
    classify_arrangement,
    nearest_neighbour_distances,
    sample_arrangement,
)
from datasets.whole_organ.continuous_corpus import (
    STEP8_SPLITS,
    TEST_SPLITS,
    generate_step8_corpus,
    load_step8_split,
    split_regions,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    """A small Step 8 corpus, generated once for the whole module."""
    target = tmp_path_factory.mktemp("step8")
    manifest = generate_step8_corpus(
        target,
        train_scenes=48,
        validation_scenes=16,
        test_scenes=16,
        families=16,
        corpus_id="test-step8",
    )
    return target, manifest


def test_arrangements_are_continuous_not_categorical(corpus) -> None:
    """Two scenes must essentially never share an arrangement.

    This is the property that makes arrangement classification useless. With four
    discrete variants a model could learn "this is variant 2" and place from memory.
    """
    target, _ = corpus
    scenes = load_step8_split(target, "train")
    coordinates = {scene.parameters.arrangement.coordinates() for scene in scenes}
    assert len(coordinates) == len(scenes), "two scenes share an exact arrangement"


def test_every_continuous_axis_actually_varies(corpus) -> None:
    """An axis that never moves is a field in a dataclass, not an experiment."""
    target, _ = corpus
    scenes = load_step8_split(target, "train")
    values = np.asarray([s.parameters.arrangement.coordinates() for s in scenes])
    for position, name in enumerate(ARRANGEMENT_FIELDS):
        spread = float(values[:, position].std())
        assert spread > 1e-3, f"arrangement axis {name} is effectively constant"


def test_each_split_lands_in_its_own_region(corpus) -> None:
    """A split whose scenes fall in the wrong region is a leak."""
    target, _ = corpus
    regions = split_regions(target)
    for split in ("train", "validation", "test_seen"):
        assert set(regions[split]) == {"in_distribution"}, f"{split}: {regions[split]}"
    for split in ("test_arrangement", "test_transform", "test_combination"):
        assert set(regions[split]) == {split}, f"{split}: {regions[split]}"


def test_held_out_families_never_appear_in_training(corpus) -> None:
    """``test_seen`` asks about unfamiliar organ shapes, so its families must be new."""
    target, manifest = corpus
    train = {scene.family_id for scene in load_step8_split(target, "train")}
    unseen = {scene.family_id for scene in load_step8_split(target, "test_seen")}
    assert train, "no training families"
    assert unseen, "no held-out families"
    assert not (train & unseen), f"families {sorted(train & unseen)} appear in both"
    assert set(manifest.split_families["test_seen"]) == unseen


def test_arrangement_holdouts_reuse_training_families(corpus) -> None:
    """The three arrangement hold-outs must not also change the organ shape.

    If they held out families as well, a failure could not be attributed to the
    arrangement rather than to an unfamiliar organ.
    """
    target, _ = corpus
    train = {scene.family_id for scene in load_step8_split(target, "train")}
    for split in ("test_arrangement", "test_transform", "test_combination"):
        families = {scene.family_id for scene in load_step8_split(target, split)}
        assert families <= train, f"{split} uses families training never saw"


def test_no_training_arrangement_falls_in_a_held_out_band(corpus) -> None:
    """The bands removed from training must really be absent from it."""
    target, _ = corpus
    for scene in load_step8_split(target, "train"):
        arrangement = scene.parameters.arrangement
        assert classify_arrangement(arrangement) == "in_distribution"
        assert not (
            YAW_INTERPOLATION_HOLE[0] <= abs(arrangement.yaw) <= YAW_INTERPOLATION_HOLE[1]
        )
        assert abs(arrangement.yaw) <= YAW_EXTRAPOLATION_EDGE
        assert not (arrangement.mirror and arrangement.transpose)


def test_held_out_arrangements_are_far_from_every_training_arrangement(corpus) -> None:
    """The leakage check: a held-out scene must not have a training near-duplicate."""
    target, _ = corpus
    train = [s.parameters.arrangement for s in load_step8_split(target, "train")]
    for split in TEST_SPLITS:
        held = [s.parameters.arrangement for s in load_step8_split(target, split)]
        distances = nearest_neighbour_distances(held, train)
        closest = min(distances)
        assert closest > 1e-6, f"{split} contains a training arrangement exactly"
        if split == "test_combination":
            assert math.isinf(closest), (
                "the combination hold-out must be unreachable from training, because its "
                "discrete pair never occurs there"
            )


def test_the_combination_holdout_isolates_the_discrete_pair(corpus) -> None:
    """Its continuous coordinates must stay in distribution, or two hold-outs confound."""
    target, _ = corpus
    for scene in load_step8_split(target, "test_combination"):
        arrangement = scene.parameters.arrangement
        assert arrangement.mirror and arrangement.transpose
        upright = Arrangement(
            mirror=False,
            transpose=False,
            **{name: getattr(arrangement, name) for name in ARRANGEMENT_FIELDS},
        )
        assert classify_arrangement(upright) == "in_distribution"


def test_the_transform_holdout_is_an_extrapolation(corpus) -> None:
    """Its rotations must exceed anything training contained."""
    target, _ = corpus
    train_max = max(
        abs(s.parameters.arrangement.yaw) for s in load_step8_split(target, "train")
    )
    for scene in load_step8_split(target, "test_transform"):
        assert abs(scene.parameters.arrangement.yaw) > train_max


def test_presence_is_identical_across_arrangements(corpus) -> None:
    """The relationship graph must stay the only channel carrying the arrangement."""
    target, _ = corpus
    presence = set()
    for split in STEP8_SPLITS:
        for scene in load_step8_split(target, split, limit=8):
            presence.add(scene.entity_ids)
    assert len(presence) == 1, "entity presence varies with the arrangement"


def test_relation_graphs_are_diverse(corpus) -> None:
    """A corpus with few distinct graphs cannot test relational inference."""
    target, manifest = corpus
    assert manifest.distinct_relation_graphs > 0.2 * manifest.scenes, (
        f"only {manifest.distinct_relation_graphs} distinct graphs in {manifest.scenes} "
        "scenes; the relations barely vary"
    )


def test_arrangement_distance_separates_the_discrete_part(corpus) -> None:
    """A mirrored organ is not a small perturbation of an unmirrored one."""
    zero = {name: 0.0 for name in ARRANGEMENT_FIELDS}
    first = Arrangement(mirror=False, transpose=False, **zero)
    second = Arrangement(mirror=True, transpose=False, **zero)
    assert math.isinf(arrangement_distance(first, second))
    assert arrangement_distance(first, first) == 0.0


def test_levels_of_detail_are_crossed_with_arrangements(corpus) -> None:
    """Level of detail must not be confounded with the arrangement region."""
    target, _ = corpus
    for split in STEP8_SPLITS:
        levels = Counter(scene.active_lod for scene in load_step8_split(target, split))
        assert set(levels) == {1, 2, 3}, f"{split} is missing a level: {dict(levels)}"


def test_sampling_respects_the_requested_region() -> None:
    """The generator and the tests must read one definition of the boundaries."""
    rng = np.random.default_rng(0)
    for region in ("in_distribution", "test_arrangement", "test_transform"):
        for _ in range(40):
            assert classify_arrangement(sample_arrangement(rng, region=region)) == region
