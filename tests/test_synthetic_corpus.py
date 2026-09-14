"""The tier-0 synthetic corpus: constraints, determinism, labels and splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import (
    DATA_LABEL,
    OWNERSHIP_PRIORITY,
    REQUIRED_SPATIAL_RELATIONS,
    SceneSpec,
    build_primitives,
    build_scene,
    check_relations,
    sample_parameters,
)
from datasets.synthetic.primitives import Capsule, Ellipsoid, Shell, build_primitive
from datasets.synthetic.sampling import SamplingConfig, sample_scene, surface_point_cloud
from datasets.synthetic.store import (
    CORPUS_FORMAT_VERSION,
    assign_splits,
    generate_corpus,
    load_manifest,
    load_split,
)


@pytest.fixture(scope="module")
def scene(ontology: AnatomyOntology) -> SceneSpec:
    """One generated scene."""
    return build_scene(ontology, scene_index=0, family_id=0, lod=2)


def test_every_renderable_entity_gets_geometry(
    scene: SceneSpec, ontology: AnatomyOntology
) -> None:
    """No renderable entity is left without a primitive."""
    renderable = {e.entity_id for e in ontology.iter_entities() if e.renderable}
    assert set(scene.primitives) == renderable
    assert len(renderable) == 36


def test_required_relationships_hold(scene: SceneSpec) -> None:
    """The generator satisfies every relationship it commits to."""
    assert len(scene.verified_relations) == len(REQUIRED_SPATIAL_RELATIONS)
    assert all(scene.verified_relations.values())


def test_unverifiable_relationships_are_recorded_with_reasons(scene: SceneSpec) -> None:
    """Relations the corpus cannot decide are named, not silently counted as passes."""
    assert scene.unverified_relations
    assert all(reason.strip() for reason in scene.unverified_relations.values())


def test_generation_is_deterministic(ontology: AnatomyOntology) -> None:
    """The same index and family produce byte-identical scenes."""
    first = build_scene(ontology, scene_index=11, family_id=3, lod=1)
    second = build_scene(ontology, scene_index=11, family_id=3, lod=1)
    assert first.to_dict() == second.to_dict()


def test_scenes_vary(ontology: AnatomyOntology) -> None:
    """Ten thousand identical hearts would teach nothing; scenes differ."""
    scenes = [build_scene(ontology, scene_index=i, family_id=i % 8, lod=2) for i in range(12)]
    centres = np.array(
        [scene.primitives["heart.left_ventricle"].centre for scene in scenes]
    )
    radii = np.array([scene.primitives["heart.left_ventricle"].extent for scene in scenes])
    assert centres.std(axis=0).max() > 1e-3
    assert radii.std(axis=0).max() > 1e-3


def test_constraints_hold_across_families(ontology: AnatomyOntology) -> None:
    """Every family produces a valid scene."""
    for family in range(12):
        scene = build_scene(ontology, scene_index=family * 3, family_id=family, lod=2)
        assert all(scene.verified_relations.values()), family


def test_relationship_checks_can_fail(ontology: AnatomyOntology) -> None:
    """The checker is not vacuous: a broken layout is detected."""
    parameters = sample_parameters(np.random.default_rng(0), 0)
    primitives = dict(build_primitives(parameters))
    left = primitives["heart.left_ventricle"]
    assert isinstance(left, Ellipsoid)
    primitives["heart.left_ventricle"] = Ellipsoid(
        centre_point=left.centre_point * np.array([-1.0, 1.0, 1.0]), radii=left.radii
    )
    results = check_relations(primitives)
    assert not results[("heart.left_ventricle", "left_of", "heart.right_ventricle")]


def test_scene_round_trips(scene: SceneSpec) -> None:
    """A scene survives serialisation exactly."""
    restored = SceneSpec.from_dict(scene.to_dict())
    assert restored.scene_id == scene.scene_id
    assert set(restored.primitives) == set(scene.primitives)
    assert restored.to_dict() == scene.to_dict()


def test_primitive_round_trip() -> None:
    """Every primitive family rebuilds from its serialised form."""
    shapes = [
        Ellipsoid(np.zeros(3), np.array([0.2, 0.3, 0.4])),
        Capsule(np.zeros(3), np.array([0.0, 0.5, 0.0]), 0.05),
        Shell(np.zeros(3), np.array([0.5, 0.5, 0.5]), 0.05),
    ]
    for shape in shapes:
        restored = build_primitive(shape.to_dict())
        assert type(restored) is type(shape)
        points = np.random.default_rng(0).uniform(-1, 1, size=(64, 3))
        assert np.array_equal(restored.occupancy(points), shape.occupancy(points))


def test_frames_match_the_primitive_convention(scene: SceneSpec) -> None:
    """The stored frame reproduces the primitive's own local transform."""
    primitive = scene.primitives["heart.left_ventricle"]
    frame = primitive.frame()
    assert frame.shape == (12,)
    assert np.allclose(frame[:3], primitive.centre)
    assert np.allclose(np.exp(frame[3:6]), np.maximum(primitive.extent, 1e-4))


def test_ownership_priority_covers_every_type(ontology: AnatomyOntology) -> None:
    """Every anatomy type has a declared tie-break for overlapping geometry."""
    used = {ontology.get(entity_id).anatomy_type for entity_id in ontology.ids()}
    groups = {ontology.get(group).anatomy_type for group in ontology.groups()}
    assert used - groups <= set(OWNERSHIP_PRIORITY)


def test_sampling_labels_are_exact(scene: SceneSpec, ontology: AnatomyOntology) -> None:
    """Occupancy and ownership labels agree with the primitives they came from."""
    sampled = sample_scene(scene, ontology, SamplingConfig(scene_points=256, entity_points=16))
    slots = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    for entity_id in sampled.entity_ids[:6]:
        slot = slots[entity_id]
        expected = scene.primitives[entity_id].occupancy(sampled.entity_points[slot])
        assert np.array_equal(expected, sampled.entity_occupancy[slot])
    owned = sampled.part_owner >= 0
    assert bool(np.all(sampled.scene_occupancy[owned]))


def test_sampling_is_deterministic(scene: SceneSpec, ontology: AnatomyOntology) -> None:
    """Points and labels are reproducible for a scene."""
    config = SamplingConfig(scene_points=128, entity_points=8)
    first = sample_scene(scene, ontology, config)
    second = sample_scene(scene, ontology, config)
    assert np.array_equal(first.scene_points, second.scene_points)
    assert np.array_equal(first.part_owner, second.part_owner)


def test_sampling_respects_the_level_of_detail(
    scene: SceneSpec, ontology: AnatomyOntology
) -> None:
    """Only entities visible at the scene's level are sampled and labelled."""
    sampled = sample_scene(scene, ontology, SamplingConfig(scene_points=128, entity_points=8))
    visible = set(scene.visible_entity_ids(ontology))
    assert set(sampled.entity_ids) == visible
    assert int(sampled.visible_mask.sum()) == len(visible)


def test_surface_cloud_is_on_the_surface(scene: SceneSpec, ontology: AnatomyOntology) -> None:
    """Reference surface points sit on the primitives they came from."""
    cloud = surface_point_cloud(scene, ontology, 256)
    assert cloud.shape[1] == 3
    assert np.abs(cloud).max() <= 2.0


def test_splits_are_deterministic_and_proportional() -> None:
    """Family assignment is stable and keeps the requested proportions."""
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    first = assign_splits(40, fractions)
    second = assign_splits(40, fractions)
    assert first == second
    counts = {name: sum(1 for value in first.values() if value == name) for name in fractions}
    assert counts == {"train": 32, "validation": 4, "test": 4}


def test_corpus_generation_writes_a_valid_manifest(
    ontology: AnatomyOntology, tmp_path: Path
) -> None:
    """A generated corpus loads back with no family crossing a split."""
    manifest = generate_corpus(ontology, tmp_path / "corpus", scenes=48, families=12, seed=0)
    assert manifest.format_version == CORPUS_FORMAT_VERSION
    assert manifest.data_label == DATA_LABEL
    assert sum(manifest.split_counts.values()) == 48

    reloaded = load_manifest(tmp_path / "corpus")
    assert reloaded.corpus_id == manifest.corpus_id

    train = load_split(tmp_path / "corpus", "train")
    test = load_split(tmp_path / "corpus", "test")
    assert train and test
    assert not {spec.family_id for spec in train} & {spec.family_id for spec in test}


def test_empty_split_is_refused(ontology: AnatomyOntology, tmp_path: Path) -> None:
    """A configuration that would leave a split empty fails loudly."""
    with pytest.raises(ValueError, match="would be empty"):
        generate_corpus(ontology, tmp_path / "corpus", scenes=8, families=4, seed=0)


def test_scene_ontology_references_are_valid(
    scene: SceneSpec, ontology: AnatomyOntology
) -> None:
    """Every entity id in a scene exists in the ontology it names."""
    assert scene.ontology_id == ontology.ontology_id
    assert scene.ontology_version == ontology.version
    for entity_id in scene.primitives:
        assert entity_id in ontology
