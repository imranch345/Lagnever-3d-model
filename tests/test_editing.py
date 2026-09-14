"""Editing operations: visibility, opacity, isolation, LOD and planned operations."""

from __future__ import annotations

import pytest

from awr.errors import EntityNotFoundError, NotYetImplementedError, OperationError
from awr.scene import AWRScene
from awr.schema import MaterialState, Transform, VisibilitySource
from editing.operations import (
    PLANNED_OPERATIONS,
    HideEntities,
    IsolateEntities,
    RevealEntities,
    SetLod,
    SetOpacity,
    ShowEntities,
)
from editing.scene_editor import SceneEditor


def test_show_and_hide_touch_only_their_targets(scene: AWRScene, editor: SceneEditor) -> None:
    """An edit changes exactly the entities it names."""
    before = scene.state_snapshot()
    applied = editor.hide("heart.aorta")
    after = scene.state_snapshot()
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {"heart.aorta"}
    assert applied.result.affected == ("heart.aorta",)


def test_group_targets_expand(scene: AWRScene, editor: SceneEditor) -> None:
    """Hiding a group hides its members, not the group itself."""
    editor.set_lod(2)
    editor.hide("heart.valves")
    assert not any(scene.get(v).visibility for v in scene.expand(["heart.valves"]))


def test_isolate_shows_only_the_target(scene: AWRScene, editor: SceneEditor) -> None:
    """Isolation hides every other renderable entity."""
    editor.apply(IsolateEntities(("heart.chambers",)))
    assert set(scene.visible_ids()) == set(scene.expand(["heart.chambers"]))


def test_reveal_includes_descendants(scene: AWRScene, editor: SceneEditor) -> None:
    """Reveal shows an entity together with everything beneath it."""
    editor.apply(IsolateEntities(("heart.left_ventricle",)))
    editor.apply(RevealEntities(("heart.coronary_circulation",)))
    assert scene.get("heart.left_coronary_artery").visibility is True


def test_opacity_edit_keeps_identity_and_visibility(scene: AWRScene, editor: SceneEditor) -> None:
    """Transparency changes opacity only."""
    before = scene.get("heart.left_ventricle")
    editor.set_opacity(["heart.left_ventricle"], 0.3, preset_name="transparent")
    after = scene.get("heart.left_ventricle")
    assert after.entity_id == before.entity_id == "heart.left_ventricle"
    assert after.opacity == 0.3
    assert after.visibility is True


def test_opacity_out_of_range_is_rejected_before_applying(
    scene: AWRScene, editor: SceneEditor
) -> None:
    """An invalid value fails validation and leaves the scene untouched."""
    before = scene.state_snapshot()
    with pytest.raises(OperationError, match=r"\[0.0, 1.0\]"):
        editor.apply(SetOpacity(("heart.left_ventricle",), 30.0))
    assert scene.state_snapshot() == before
    assert scene.version == 1


def test_unknown_entity_in_an_operation_raises(scene: AWRScene, editor: SceneEditor) -> None:
    """Operations validate their targets against the scene."""
    before = scene.state_snapshot()
    with pytest.raises(EntityNotFoundError):
        editor.apply(ShowEntities(("heart.pancreas",)))
    assert scene.state_snapshot() == before


def test_empty_target_list_is_rejected(editor: SceneEditor) -> None:
    """An operation with no targets is an error, not a silent no-op."""
    with pytest.raises(OperationError, match="No target entities"):
        editor.apply(HideEntities(()))


def test_lod_change_clears_manual_visibility_but_keeps_opacity(
    scene: AWRScene, editor: SceneEditor
) -> None:
    """The documented LOD rule: visibility is recomputed, material state is not."""
    editor.apply(IsolateEntities(("heart.chambers",)))
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    applied = editor.set_lod(2)

    assert scene.get("heart.left_ventricle").opacity == 0.3
    assert scene.get("heart.tricuspid_valve").visibility is True
    assert scene.get("heart.aorta").visibility is True
    assert applied.result.parameters["cleared_overrides"] > 0
    assert all(
        entity.state.visibility_source is VisibilitySource.LOD_DEFAULT
        for entity in scene.iter_entities()
        if entity.renderable
    )


def test_invalid_lod_is_rejected(scene: AWRScene, editor: SceneEditor) -> None:
    """An LOD outside the ladder fails with the valid levels listed."""
    with pytest.raises(OperationError, match="Valid levels"):
        editor.apply(SetLod(9))
    assert scene.active_lod == 1


def test_manual_visibility_can_exceed_the_lod(scene: AWRScene, editor: SceneEditor) -> None:
    """Deeper anatomy can be shown without changing the level of detail."""
    editor.show("heart.sinoatrial_node")
    assert scene.get("heart.sinoatrial_node").visibility is True
    assert scene.active_lod == 1


def test_material_and_transform_are_semantic_state(scene: AWRScene, editor: SceneEditor) -> None:
    """Material and transform are recorded on the entity, without rendering."""
    editor.set_material(["heart.myocardium"], MaterialState(tissue_class="cardiac_muscle"))
    assert scene.get("heart.myocardium").material_state.tissue_class == "cardiac_muscle"

    from editing.operations import SetTransform

    editor.apply(SetTransform(("heart.aorta",), Transform(translation=(1.0, 0.0, 0.0))))
    assert scene.get("heart.aorta").transform.is_identity is False


def test_reset_restores_lod_defaults(scene: AWRScene, editor: SceneEditor) -> None:
    """Reset discards manual visibility and opacity edits."""
    editor.apply(IsolateEntities(("heart.left_ventricle",)))
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    editor.reset_view()
    assert scene.get("heart.left_ventricle").opacity == 1.0
    assert set(scene.visible_ids()) == {
        entity.entity_id
        for entity in scene.iter_entities()
        if scene.default_visibility(entity.entity_id)
    }


@pytest.mark.parametrize("operation_cls", PLANNED_OPERATIONS)
def test_planned_operations_are_declared_but_refuse_to_run(operation_cls: type) -> None:
    """Future operations exist as typed declarations and never fake a result."""
    with pytest.raises(NotYetImplementedError):
        operation_cls()


def test_planned_operations_cover_the_specified_roadmap() -> None:
    """Every future operation named in the specification is declared."""
    names = {cls.name for cls in PLANNED_OPERATIONS}
    assert {
        "segment",
        "cross_section",
        "explode",
        "add_label",
        "replace_component",
        "regenerate_component",
        "edit_geometry",
    } <= names
