"""End-to-end natural-language command behaviour.

These tests cover the scenarios the project specification names, against one
persistent scene: generation, visibility, isolation, transparency, level
switching, identity stability, persistence across commands and useful errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from awr.scene import AWRScene
from awr.validation import validate_scene
from reasoning.command_engine import CommandEngine
from reasoning.parser import Intent


def test_generate_builds_the_full_awr(engine: CommandEngine) -> None:
    """"Generate a human heart" constructs the whole representation."""
    response = engine.execute("Generate a human heart")
    assert response.ok
    assert response.intent is Intent.GENERATE
    scene = engine.scene
    assert scene is not None
    assert len(scene) == 42
    assert len(scene.relationships) > 100
    assert scene.active_lod == 1


def test_commands_before_generation_explain_what_to_do(engine: CommandEngine) -> None:
    """Editing with no scene gives an instruction, not a crash."""
    response = engine.execute("show the valves")
    assert response.ok is False
    assert "generate a human heart" in response.message


def test_show_the_four_chambers(live_engine: CommandEngine) -> None:
    """The four chambers become visible and are reported as chambers."""
    live_engine.execute("hide everything except the aorta")
    response = live_engine.execute("Show the four chambers")
    assert response.ok
    scene = live_engine.scene
    assert scene is not None
    for chamber in ("right_atrium", "right_ventricle", "left_atrium", "left_ventricle"):
        assert scene.get(f"heart.{chamber}").visibility is True


def test_hide_everything_except_the_chambers(live_engine: CommandEngine) -> None:
    """Isolation leaves exactly the four chambers visible."""
    response = live_engine.execute("Hide everything except the chambers")
    assert response.ok
    scene = live_engine.scene
    assert scene is not None
    assert set(scene.visible_ids()) == set(scene.expand(["heart.chambers"]))
    assert len(scene) == 42


def test_make_the_left_ventricle_transparent_keeps_the_same_entity(
    live_engine: CommandEngine,
) -> None:
    """Opacity changes on the same persistent entity id."""
    scene = live_engine.scene
    assert scene is not None
    before = scene.get("heart.left_ventricle")
    assert before.opacity == 1.0

    response = live_engine.execute("Make the left ventricle transparent")
    assert response.ok
    after = scene.get("heart.left_ventricle")
    assert after.entity_id == "heart.left_ventricle"
    assert after.opacity == 0.3
    assert after.visibility is True
    assert response.affected == ("heart.left_ventricle",)


def test_make_the_left_ventricle_opaque_again(live_engine: CommandEngine) -> None:
    """The inverse command restores full opacity."""
    live_engine.execute("Make the left ventricle transparent")
    live_engine.execute("Make the left ventricle opaque")
    scene = live_engine.scene
    assert scene is not None
    assert scene.get("heart.left_ventricle").opacity == 1.0


def test_show_and_hide_the_valves(live_engine: CommandEngine) -> None:
    """Valves can be shown below their LOD and hidden again."""
    scene = live_engine.scene
    assert scene is not None
    live_engine.execute("Show the valves")
    assert all(scene.get(v).visibility for v in scene.expand(["heart.valves"]))
    live_engine.execute("Hide the valves")
    assert not any(scene.get(v).visibility for v in scene.expand(["heart.valves"]))


def test_show_and_hide_a_single_entity(live_engine: CommandEngine) -> None:
    """Single-entity visibility commands work by name."""
    scene = live_engine.scene
    assert scene is not None
    live_engine.execute("Hide the left ventricle")
    assert scene.get("heart.left_ventricle").visibility is False
    live_engine.execute("Show the left ventricle")
    assert scene.get("heart.left_ventricle").visibility is True


def test_show_more_detail_steps_up_the_ladder(live_engine: CommandEngine) -> None:
    """Relative detail commands move one rung and clamp at the top."""
    scene = live_engine.scene
    assert scene is not None
    assert scene.active_lod == 1
    live_engine.execute("Show more detail")
    assert scene.active_lod == 2
    for _ in range(5):
        live_engine.execute("Show more detail")
    assert scene.active_lod == scene.lod_ladder.max_level
    response = live_engine.execute("Show more detail")
    assert response.ok
    assert "highest" in response.message


def test_switch_between_school_and_medical_level(live_engine: CommandEngine) -> None:
    """Audience level drives the level of detail through configuration."""
    scene = live_engine.scene
    assert scene is not None
    live_engine.execute("Switch to medical level")
    assert str(scene.educational_level) == "medical"
    assert scene.active_lod == 4
    assert scene.get("heart.purkinje_fibers").visibility is True

    live_engine.execute("Switch to school level")
    assert str(scene.educational_level) == "school"
    assert scene.active_lod == 1
    assert scene.get("heart.purkinje_fibers").visibility is False
    assert "heart.purkinje_fibers" in scene


def test_scene_state_persists_across_a_command_sequence(live_engine: CommandEngine) -> None:
    """The specification's session runs against one scene, not several."""
    scene_id = live_engine.scene.scene_id if live_engine.scene else None
    responses = live_engine.run(
        (
            "Show the four chambers",
            "Hide everything except the chambers",
            "Make the left ventricle transparent",
            "Show the valves",
            "Show more detail",
        )
    )
    assert all(response.ok for response in responses)
    scene = live_engine.scene
    assert scene is not None
    assert scene.scene_id == scene_id
    assert scene.get("heart.left_ventricle").opacity == 0.3
    assert len(scene.history) == scene.version
    assert validate_scene(scene).is_valid


def test_entity_ids_never_change_across_a_session(live_engine: CommandEngine) -> None:
    """Identity is stable across every command in a session."""
    scene = live_engine.scene
    assert scene is not None
    before = scene.ids()
    live_engine.run(
        (
            "Hide everything except the chambers",
            "Make the left ventricle transparent",
            "Switch to medical level",
            "Animate blood flow",
            "Switch to school level",
        )
    )
    assert scene.ids() == before


def test_session_survives_save_and_reload(live_engine: CommandEngine, tmp_path: Path) -> None:
    """A session can be persisted mid-way and continued after reloading."""
    live_engine.run(("Hide everything except the chambers", "Make the left ventricle transparent"))
    scene = live_engine.scene
    assert scene is not None
    path = scene.save(tmp_path / "session.json")

    resumed = CommandEngine(config=live_engine.config, ontology=live_engine.ontology)
    resumed.attach_scene(AWRScene.load(path))
    response = resumed.execute("Show the valves")

    assert response.ok
    assert resumed.scene is not None
    assert resumed.scene.get("heart.left_ventricle").opacity == 0.3
    assert resumed.scene.version == scene.version + 1


def test_unknown_structure_gives_a_useful_error(live_engine: CommandEngine) -> None:
    """An invalid anatomical reference never corrupts the scene."""
    scene = live_engine.scene
    assert scene is not None
    before = scene.state_snapshot()
    response = live_engine.execute("Show the spleen")
    assert response.ok is False
    assert response.error_code == "UnknownEntityReferenceError"
    assert "spleen" in response.message
    assert scene.state_snapshot() == before
    assert scene.version == 1


def test_ambiguous_structure_lists_candidates(live_engine: CommandEngine) -> None:
    """An ambiguous reference asks for precision instead of guessing."""
    response = live_engine.execute("Make the ventricle transparent")
    assert response.ok is False
    assert response.error_code == "AmbiguousEntityReferenceError"
    assert "heart.left_ventricle" in response.message
    assert "heart.right_ventricle" in response.message


def test_uninterpretable_command_is_reported(live_engine: CommandEngine) -> None:
    """An unsupported sentence is refused with examples of what works."""
    response = live_engine.execute("teach me about quantum mechanics")
    assert response.ok is False
    assert response.error_code == "CommandError"
    assert "generate a human heart" in response.message


def test_out_of_range_opacity_is_explained(live_engine: CommandEngine) -> None:
    """A percentage given as a bare number is rejected with a hint."""
    response = live_engine.execute("set the left ventricle opacity to 30")
    assert response.ok is False
    assert "percentage" in response.message


def test_percentage_opacity_is_accepted(live_engine: CommandEngine) -> None:
    """An explicit percentage is converted."""
    response = live_engine.execute("set the left ventricle opacity to 45%")
    assert response.ok
    scene = live_engine.scene
    assert scene is not None
    assert scene.get("heart.left_ventricle").opacity == pytest.approx(0.45)


def test_explain_is_read_only(live_engine: CommandEngine) -> None:
    """Explaining the scene reports state without changing it."""
    scene = live_engine.scene
    assert scene is not None
    before = scene.state_snapshot()
    version = scene.version
    response = live_engine.execute("Explain what is happening")
    assert response.ok
    assert "level of detail" in response.message.lower()
    assert scene.state_snapshot() == before
    assert scene.version == version


def test_explain_an_entity_uses_the_graphs(live_engine: CommandEngine) -> None:
    """An entity explanation is derived from typed relationships."""
    response = live_engine.execute("Explain the left ventricle")
    assert response.ok
    assert "pumps blood to aorta" in response.message
    assert "interventricular septum" in response.message


def test_animate_blood_flow_binds_the_cycle(live_engine: CommandEngine) -> None:
    """Animation binds entities semantically and reports its flow paths."""
    live_engine.execute("Show more detail")
    response = live_engine.execute("Animate blood flow")
    assert response.ok
    scene = live_engine.scene
    assert scene is not None
    assert scene.get("heart.left_ventricle").animation_state.clip_id == "blood_flow"
    clip = response.data["clip"]
    assert len(clip["phases"]) == 5
    assert any("tricuspid_valve" in path for path in clip["paths"])

    live_engine.execute("Stop the animation")
    assert scene.get("heart.left_ventricle").animation_state.is_animated is False


def test_reset_returns_to_defaults(live_engine: CommandEngine) -> None:
    """A reset discards manual edits but keeps the scene and its history."""
    live_engine.run(("Hide everything except the chambers", "Make the left ventricle transparent"))
    response = live_engine.execute("Reset the scene")
    assert response.ok
    scene = live_engine.scene
    assert scene is not None
    assert scene.get("heart.left_ventricle").opacity == 1.0
    assert scene.get("heart.aorta").visibility is True
    assert len(scene.history) == scene.version


def test_strict_mode_raises_instead_of_reporting(live_engine: CommandEngine) -> None:
    """Programmatic callers can opt into exceptions."""
    from awr.errors import UnknownEntityReferenceError

    with pytest.raises(UnknownEntityReferenceError):
        live_engine.execute("show the spleen", strict=True)
