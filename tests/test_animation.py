"""Blood flow and conduction derived from the functional graph."""

from __future__ import annotations

import pytest

from animation.heart_flow import CardiacFlowModel, FlowRole
from awr.config import DomainConfig
from awr.relationships import FlowMedium, Granularity
from awr.scene import AWRScene
from editing.scene_editor import SceneEditor


@pytest.fixture
def model(scene: AWRScene, config: DomainConfig) -> CardiacFlowModel:
    """Flow model over a generated scene."""
    return CardiacFlowModel(scene, config.cardiac_cycle)


def test_flow_follows_the_functional_graph(scene: AWRScene, model: CardiacFlowModel) -> None:
    """The right-heart path is discovered, not hard-coded."""
    SceneEditor(scene).set_lod(2)
    paths = {path.describe() for path in model.flow_paths()}
    assert any(
        "superior_vena_cava -> right_atrium -> tricuspid_valve -> right_ventricle "
        "-> pulmonary_valve -> pulmonary_trunk" in path
        for path in paths
    )
    assert any(
        "pulmonary_veins -> left_atrium -> mitral_valve -> left_ventricle -> aortic_valve -> aorta"
        in path
        for path in paths
    )


def test_granularity_follows_the_level_of_detail(scene: AWRScene, model: CardiacFlowModel) -> None:
    """Without valves in scope the traversal uses chamber-level edges."""
    editor = SceneEditor(scene)
    editor.set_lod(1)
    assert model.granularity_for_lod() is Granularity.SUMMARY
    summary = model.flow_paths()
    assert all("valve" not in path.describe() for path in summary)

    editor.set_lod(2)
    assert model.granularity_for_lod() is Granularity.DETAILED
    assert any("valve" in path.describe() for path in model.flow_paths())


def test_parallel_paths_are_not_double_counted(scene: AWRScene, model: CardiacFlowModel) -> None:
    """A path never mixes the summary shortcut with the valve chain."""
    SceneEditor(scene).set_lod(2)
    for path in model.flow_paths():
        entities = path.entity_ids
        if "heart.tricuspid_valve" in entities:
            index = entities.index("heart.right_atrium")
            assert entities[index + 1] == "heart.tricuspid_valve"


def test_external_boundaries_are_reported(model: CardiacFlowModel) -> None:
    """The pulmonary capillary bed is outside the ontology and is named as such."""
    paths = model.flow_paths()
    assert any(path.exits_to == "pulmonary_capillary_bed" for path in paths)
    assert any(path.enters_from == "pulmonary_capillary_bed" for path in paths)


def test_conduction_is_a_separate_medium(model: CardiacFlowModel) -> None:
    """Electrical conduction never appears in a blood-flow path."""
    blood = model.flow_paths(medium=FlowMedium.BLOOD)
    conduction = model.conduction_paths()
    assert all("heart.sinoatrial_node" not in path.entity_ids for path in blood)
    assert any("heart.purkinje_fibers" in path.entity_ids for path in conduction)


def test_clip_phases_come_from_configuration(model: CardiacFlowModel, config: DomainConfig) -> None:
    """Phase names and durations are configuration, not constants in code."""
    clip = model.build_clip()
    assert [phase.name for phase in clip.phases] == [
        phase.name for phase in config.cardiac_cycle.phases
    ]
    assert clip.phases[0].start == 0.0
    assert clip.phases[-1].end == pytest.approx(1.0)
    assert clip.duration_seconds == config.cardiac_cycle.cycle_duration_seconds


def test_valve_roles_are_derived_from_the_graph(scene: AWRScene, model: CardiacFlowModel) -> None:
    """Atrioventricular and semilunar valves behave differently per phase."""
    SceneEditor(scene).set_lod(2)
    clip = model.build_clip()
    ejection = next(phase for phase in clip.phases if phase.name == "ventricular_ejection")
    assert ejection.roles["heart.mitral_valve"] is FlowRole.VALVE_CLOSED
    assert ejection.roles["heart.aortic_valve"] is FlowRole.VALVE_OPEN
    assert ejection.roles["heart.left_ventricle"] is FlowRole.EJECTING

    filling = next(phase for phase in clip.phases if phase.name == "ventricular_filling")
    assert filling.roles["heart.mitral_valve"] is FlowRole.VALVE_OPEN
    assert filling.roles["heart.aortic_valve"] is FlowRole.VALVE_CLOSED


def test_clip_is_semantic_only(model: CardiacFlowModel) -> None:
    """The clip says plainly that it animates nothing."""
    clip = model.build_clip()
    assert any("No geometry is deformed" in note for note in clip.notes)
    bindings = clip.bindings()
    assert all(state.clip_id == clip.clip_id for state in bindings.values())
    assert all(0.0 <= (state.normalized_time or 0.0) <= 1.0 for state in bindings.values())


def test_binding_animation_does_not_change_visibility(
    scene: AWRScene, model: CardiacFlowModel
) -> None:
    """Animating is a separate concern from what is shown."""
    editor = SceneEditor(scene)
    before = set(scene.visible_ids())
    editor.bind_animation(model.build_clip().bindings(), "blood_flow")
    assert set(scene.visible_ids()) == before
