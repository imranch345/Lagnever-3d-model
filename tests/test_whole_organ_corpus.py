"""Step 7 whole-organ corpus: construction, crossing, and measured relations."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace as dataclass_replace

import numpy as np
import pytest

from datasets.whole_organ.corpus import (
    LOD_ENTITIES,
    WHOLE_ORGAN_ENTITIES,
    assign_splits,
    build_scene,
    generate_corpus,
    load_split,
)
from datasets.whole_organ.field import WholeOrganField
from datasets.whole_organ.parameters import Variant, sample_parameters
from datasets.whole_organ.relations import measure_relations
from experiments.step7.relation_baseline import (
    blind_centroids,
    oracle_centroids,
    relation_baseline,
    score_against_measured_relations,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    """A small corpus that still crosses every family with every variant."""
    target = tmp_path_factory.mktemp("whole-organ")
    manifest = generate_corpus(target, scenes=96, families=8, corpus_id="test-corpus")
    return target, manifest


def test_every_family_is_seen_in_every_variant(corpus) -> None:
    """The contrast the graph experiments rely on must actually be presented.

    If a family only ever appears in one arrangement, then "same organ, different
    relations" is never shown and any graph result measures family differences instead.
    """
    target, manifest = corpus
    assert manifest.variants_per_family, "crossing was not recorded"
    assert set(manifest.variants_per_family.values()) == {len(list(Variant))}

    seen: dict[int, set[str]] = {}
    for split in ("train", "validation", "test"):
        for scene in load_split(target, split):
            seen.setdefault(scene.family_id, set()).add(str(scene.variant))
    assert seen, "corpus produced no scenes"
    for family, variants in sorted(seen.items()):
        assert variants == {str(v) for v in Variant}, f"family {family} saw only {variants}"


def test_every_split_contains_every_variant(corpus) -> None:
    """A split missing an arrangement cannot test generalisation across arrangements."""
    target, manifest = corpus
    for split in ("train", "validation", "test"):
        scenes = load_split(target, split)
        assert scenes, f"split {split} is empty"
        counts = Counter(str(scene.variant) for scene in scenes)
        assert set(counts) == {str(v) for v in Variant}, f"{split} holds only {set(counts)}"
        assert dict(manifest.split_variants[split]) == dict(counts)


def test_variant_is_not_predictable_from_family(corpus) -> None:
    """Guards the specific aliasing defect: variant derived from the family index."""
    target, _ = corpus
    scenes = [s for split in ("train", "validation", "test") for s in load_split(target, split)]
    by_family: dict[int, Counter] = {}
    for scene in scenes:
        by_family.setdefault(scene.family_id, Counter())[str(scene.variant)] += 1
    for family, counts in by_family.items():
        assert len(counts) > 1, f"family {family} is locked to a single variant"


def test_variant_and_level_of_detail_are_crossed(corpus) -> None:
    """Level of detail must not be confounded with arrangement either."""
    target, _ = corpus
    scenes = [s for split in ("train", "validation", "test") for s in load_split(target, split)]
    cells = Counter((str(s.variant), s.active_lod) for s in scenes)
    levels = {scene.active_lod for scene in scenes}
    assert levels == {1, 2, 3}, f"unexpected levels {sorted(levels)}"
    assert len(cells) == len(list(Variant)) * len(levels), "a variant is missing a level"
    assert min(cells.values()) > 0


def test_generation_crosses_when_variants_divide_families(tmp_path) -> None:
    """Eight families and four variants is the exact aliasing case the old code hit.

    The variant count dividing the family count is what made the old expression a
    function of the family index. It must now still cross.
    """
    manifest = generate_corpus(tmp_path / "ok", scenes=96, families=8, corpus_id="crossed")
    assert set(manifest.variants_per_family.values()) == {len(list(Variant))}
    for counts in manifest.split_variants.values():
        assert set(counts) == {str(variant) for variant in Variant}


def test_every_entity_owns_volume_in_every_scene(corpus) -> None:
    """Wall layers included: a scene where an entity owns nothing breaks the metrics."""
    target, _ = corpus
    for scene in load_split(target, "test")[:12]:
        expected = LOD_ENTITIES[scene.active_lod]
        for entity in expected:
            assert scene.counts.get(entity, 0) > 0, f"{entity} owns nothing in {scene.scene_id}"


def test_relations_are_measured_not_asserted() -> None:
    """Mirroring the organ must flip the left/right relations that were measured."""
    normal = sample_parameters(np.random.default_rng(3), family_id=3, variant=Variant.NORMAL)
    mirrored = dataclass_replace(
        normal, arrangement=dataclass_replace(normal.arrangement, mirror=True)
    )

    def edges_of(parameters):
        organ = WholeOrganField(parameters)
        statistics = organ.entity_statistics(np.random.default_rng(11))
        return {
            (e.subject, e.object): e.relation
            for e in measure_relations(organ, statistics, np.random.default_rng(13))
        }

    first, second = edges_of(normal), edges_of(mirrored)
    shared = set(first) & set(second)
    flipped = {
        pair
        for pair in shared
        if {first[pair], second[pair]} in ({"left_of", "right_of"},)
    }
    assert flipped, "mirroring changed no left/right relation, so the variant is inert"


def test_oracle_scores_perfectly_and_blind_does_not(corpus) -> None:
    """The metric must be able to tell a relation-user from a relation-ignorer."""
    target, _ = corpus
    scenes = load_split(target, "test")
    oracle = score_against_measured_relations(scenes, oracle_centroids)
    assert math.isclose(oracle["accuracy"], 1.0), "scoring code is wrong, not the model"

    cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    blind = score_against_measured_relations(scenes, lambda s: blind_centroids(s, cache))
    assert blind["accuracy"] < oracle["accuracy"], "blind predictor is indistinguishable"


def test_relation_baseline_reports_headroom(corpus) -> None:
    """The reported headroom is what a claim about relational competence must clear."""
    target, _ = corpus
    baseline = relation_baseline(target, "test")
    assert math.isclose(baseline["oracle"]["accuracy"], 1.0)
    assert 0.0 < baseline["discriminative_headroom"] <= 1.0
    assert baseline["blind_normal_arrangement"]["by_variant"]["normal"]["accuracy"] > (
        baseline["blind_normal_arrangement"]["by_variant"]["mirrored"]["accuracy"]
    ), "mirroring must cost the blind predictor more than leaving the organ alone"


def test_split_assignment_is_family_disjoint() -> None:
    """Families must not straddle splits, or the held-out test is not held out."""
    assignment = assign_splits(40, {"train": 0.8, "validation": 0.1, "test": 0.1})
    assert len(assignment) == 40
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_scene_round_trips_through_serialisation() -> None:
    """A scene read back from disk must carry its measured relations unchanged."""
    scene = build_scene(scene_index=5, family_id=2, variant=Variant.ROTATED, lod=3)
    restored = type(scene).from_dict(scene.to_dict())
    assert restored.scene_id == scene.scene_id
    assert restored.variant is scene.variant
    assert [(e.subject, e.relation, e.object) for e in restored.edges] == [
        (e.subject, e.relation, e.object) for e in scene.edges
    ]


def test_whole_organ_entity_set_is_complete() -> None:
    """The finest level must expose every entity the corpus claims to contain."""
    assert set(LOD_ENTITIES[max(LOD_ENTITIES)]) == set(WHOLE_ORGAN_ENTITIES)
    assert len(WHOLE_ORGAN_ENTITIES) == 20


def test_part_target_never_names_a_hidden_entity(corpus) -> None:
    """The part-correspondence target must only name entities the level exposes.

    The composition masks absent entities to -1e9, so a target pointing at one costs
    about 1e9 for that point. At level 1 this affected roughly 39% of points and the
    resulting loss term, around 4e8, swamped every other objective for every arm.
    """
    from awr.config import load_domain_config
    from awr.ontology import load_ontology
    from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

    target, _ = corpus
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = load_split(target, "test")
    for level in (1, 2, 3):
        group = [scene for scene in scenes if scene.active_lod == level][:4]
        if not group:
            continue
        batch = builder.build(group).batch
        for index in range(batch.part_owner.shape[0]):
            for slot in batch.part_owner[index].unique().tolist():
                if slot < 0:
                    continue
                assert bool(batch.entity_present[index, slot]), (
                    f"level {level} labels points with hidden entity slot {slot}"
                )


def test_part_target_agrees_with_level_occupancy(corpus) -> None:
    """A point is background in the part target exactly when the level says it is empty."""
    from datasets.whole_organ.sampling import sample_scene

    target, _ = corpus
    slot_of = {entity_id: index for index, entity_id in enumerate(WHOLE_ORGAN_ENTITIES)}
    del slot_of
    from awr.config import load_domain_config
    from awr.ontology import load_ontology

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    slots = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    for scene in load_split(target, "test")[:9]:
        sampled = sample_scene(scene, slots)
        level = min(max(scene.active_lod, 1), 3)
        occupied = sampled.lod_occupancy[level]
        assert ((sampled.part_owner >= 0) == occupied).all(), (
            f"{scene.scene_id}: part target and level occupancy disagree"
        )


def test_all_three_typed_graphs_carry_edges(corpus) -> None:
    """An empty typed graph turns its ablation into a differently-named no-graph arm.

    The structure graph was empty for the whole of the first two suites, which made the
    structure-only ablation meaningless while it still paid for a graph encoder.
    """
    from awr.config import load_domain_config
    from awr.ontology import load_ontology
    from experiments.step7.perturbations import GRAPH_INDEX
    from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

    target, _ = corpus
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = load_split(target, "test")
    for level in (1, 2, 3):
        group = [scene for scene in scenes if scene.active_lod == level][:4]
        if not group:
            continue
        structure = builder.build(group).batch.structure
        live = structure.edge_mask
        for name, index in GRAPH_INDEX.items():
            count = int(((structure.edge_graph == index) & live).sum())
            assert count > 0, f"level {level}: the {name} graph has no edges"


def test_structure_graph_is_scene_independent(corpus) -> None:
    """The structural edges are ontology knowledge, so they must not vary by scene.

    Recorded as a test because it is what makes the structure-only arm a constant prior
    rather than a channel that can distinguish arrangements.
    """
    from awr.config import load_domain_config
    from awr.ontology import load_ontology
    from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

    target, _ = corpus
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = [scene for scene in load_split(target, "test") if scene.active_lod == 3][:6]
    signatures = {
        tuple(sorted((e.subject, e.relation, e.object) for e in builder._structure_edges(scene)))
        for scene in scenes
    }
    assert len(signatures) == 1, "structural edges differ between scenes of the same level"
    assert signatures.pop(), "no structural edges at all"
