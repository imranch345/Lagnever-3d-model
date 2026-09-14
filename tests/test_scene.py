"""Scene construction, identity, LOD defaults, history and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from awr.errors import EntityNotFoundError, InvalidStateError, SceneError, SchemaError
from awr.ontology import AnatomyOntology
from awr.scene import AWRScene
from awr.schema import EntityState, VisibilitySource
from awr.validation import validate_scene, validate_scene_roundtrip
from editing.scene_editor import SceneEditor


def test_scene_holds_every_entity_at_every_lod(scene: AWRScene, ontology: AnatomyOntology) -> None:
    """Existence is independent of the level of detail."""
    assert set(scene.ids()) == set(ontology.ids())
    for lod in (0, 1, 2, 3, 4):
        scene.set_active_lod(lod)
        assert len(scene) == len(ontology)


def test_lod_defaults_match_the_specified_ladder(scene: AWRScene) -> None:
    """Each LOD reveals the structures the specification assigns to it."""
    editor = SceneEditor(scene)

    editor.set_lod(0)
    assert scene.visible_ids() == ("heart",)

    editor.set_lod(1)
    visible = set(scene.visible_ids())
    assert set(scene.expand(["heart.chambers"])) <= visible
    assert set(scene.expand(["heart.great_vessels"])) <= visible
    assert not set(scene.expand(["heart.valves"])) & visible

    editor.set_lod(2)
    visible = set(scene.visible_ids())
    assert set(scene.expand(["heart.valves"])) <= visible
    assert set(scene.expand(["heart.septa"])) <= visible
    assert not set(scene.expand(["heart.wall_layers"])) & visible

    editor.set_lod(3)
    assert set(scene.expand(["heart.wall_layers"])) <= set(scene.visible_ids())

    editor.set_lod(4)
    assert scene.get("heart.sinoatrial_node").visibility is True


def test_hidden_entities_still_exist(scene: AWRScene) -> None:
    """A school-level scene still holds medical-level anatomy."""
    assert scene.active_lod == 1
    node = scene.get("heart.sinoatrial_node")
    assert node.visibility is False
    assert node.entity_id in scene
    assert scene.relationships.incident("heart.sinoatrial_node")


def test_groups_are_never_visible(scene: AWRScene) -> None:
    """Organisational groups are not drawable at any LOD."""
    editor = SceneEditor(scene)
    for lod in (0, 1, 2, 3, 4):
        editor.set_lod(lod)
        for group_id in [e.entity_id for e in scene.iter_entities() if e.is_group]:
            assert scene.get(group_id).visibility is False


def test_making_a_group_visible_is_refused(scene: AWRScene) -> None:
    """The low-level primitive refuses to draw a group."""
    with pytest.raises(SceneError, match="organisational group"):
        scene.set_entity_visibility("heart.valves", True, source=VisibilitySource.MANUAL)


def test_expand_resolves_groups_to_renderable_members(scene: AWRScene) -> None:
    """Group expansion yields renderable entities only."""
    expanded = scene.expand(["heart.valves"])
    assert len(expanded) == 4
    assert all(scene.get(entity_id).renderable for entity_id in expanded)


def test_tree_links_are_mutual(scene: AWRScene) -> None:
    """Parent and child links agree in both directions."""
    for entity in scene.iter_entities():
        if entity.parent_id is not None:
            assert entity.entity_id in scene.get(entity.parent_id).children
        for child in entity.children:
            assert scene.get(child).parent_id == entity.entity_id


def test_scene_id_is_deterministic(generator) -> None:
    """The same request in the same configuration yields the same scene id."""
    from generation.generator import GenerationRequest

    first = generator.generate(GenerationRequest(text="generate a human heart")).scene
    second = generator.generate(GenerationRequest(text="generate a human heart")).scene
    assert first.scene_id == second.scene_id


def test_history_records_every_change(scene: AWRScene) -> None:
    """Each state-changing operation appends exactly one event."""
    editor = SceneEditor(scene)
    start = scene.version
    editor.hide("heart.aorta", command_text="hide the aorta")
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    assert scene.version == start + 2
    assert [event.version for event in scene.history] == list(range(1, scene.version + 1))
    assert scene.history[-1].deltas[0].entity_id == "heart.left_ventricle"


def test_no_op_does_not_bump_the_version(scene: AWRScene) -> None:
    """Hiding something already hidden changes nothing and records nothing."""
    editor = SceneEditor(scene)
    editor.hide("heart.sinoatrial_node")
    version = scene.version
    applied = editor.hide("heart.sinoatrial_node")
    assert applied.result.changed is False
    assert applied.committed is False
    assert scene.version == version


def test_scene_round_trips_through_json(scene: AWRScene, tmp_path: Path) -> None:
    """A saved scene reloads with identity, state and history intact."""
    editor = SceneEditor(scene)
    editor.isolate("heart.chambers")
    editor.set_opacity(["heart.left_ventricle"], 0.3, preset_name="transparent")

    path = scene.save(tmp_path / "scene.json")
    restored = AWRScene.load(path)

    assert restored.scene_id == scene.scene_id
    assert restored.ids() == scene.ids()
    assert restored.state_snapshot() == scene.state_snapshot()
    assert restored.version == scene.version
    assert len(restored.history) == len(scene.history)
    assert restored.get("heart.left_ventricle").opacity == 0.3
    assert validate_scene_roundtrip(scene).is_valid


def test_relationships_survive_serialisation(scene: AWRScene) -> None:
    """Graph structure is preserved across a save and load cycle."""
    restored = AWRScene.from_dict(scene.to_dict())
    assert restored.relationships.counts() == scene.relationships.counts()
    assert "heart.aorta" in restored.relationships.related("heart.left_ventricle", "pumps_to")


def test_scene_schema_version_is_checked() -> None:
    """A payload from an unknown schema version is refused."""
    with pytest.raises(SchemaError, match="schema_version"):
        AWRScene.from_dict({"schema_version": "something-else"})


def test_missing_entity_lookup_is_explicit(scene: AWRScene) -> None:
    """An unknown id raises with the id in the message."""
    with pytest.raises(EntityNotFoundError, match="heart.gallbladder"):
        scene.get("heart.gallbladder")


def test_opacity_is_range_checked() -> None:
    """Entity state refuses an impossible opacity."""
    with pytest.raises(InvalidStateError, match=r"\[0.0, 1.0\]"):
        EntityState(opacity=1.4)


def test_scene_validation_is_clean(scene: AWRScene) -> None:
    """A freshly generated scene has no validation errors."""
    report = validate_scene(scene)
    assert report.is_valid, str(report)
