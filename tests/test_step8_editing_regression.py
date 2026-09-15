"""Step 8 persistent-editing regression.

Step 8 is not about editing. Its changes are large enough to break editing by accident,
though: the arrangement space replaced the variant enum, placement became predicted, and
the level-of-detail objective changed. This suite checks that the editing interfaces still
do what Step 4 and Step 7 established, and nothing more ambitious than that.

A full editing redesign is deliberately out of scope. Step 7 measured that the edit head's
locality works and its accuracy does not, and recorded that as REVISE; spending Step 8's
compute on it would have come out of the placement experiments that Step 8 exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.scene import AWRScene
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.sampling import EditOperation
from editing.scene_editor import SceneEditor
from experiments.step7.editing import OPERATIONS, build_edit_case
from experiments.step8.editing_regression import editing_regression
from generation.neural.nn.editing_head import EditHeadConfig, LocalEditHead, encode_edit


def test_editing_one_entity_leaves_the_others_alone(editor: SceneEditor, scene: AWRScene) -> None:
    """The Step 4 guarantee: a presentation edit touches one entity."""
    before = {entity.entity_id: entity.state.opacity for entity in scene.iter_entities()}
    editor.set_opacity(["heart.left_ventricle"], 0.25)
    after = {entity.entity_id: entity.state.opacity for entity in scene.iter_entities()}
    assert after["heart.left_ventricle"] == pytest.approx(0.25)
    for entity_id, opacity in before.items():
        if entity_id != "heart.left_ventricle":
            assert after[entity_id] == pytest.approx(opacity), entity_id


def test_an_edit_survives_saving_and_reloading(editor: SceneEditor, scene: AWRScene) -> None:
    """Persistence is what makes the scene a memory rather than a render."""
    editor.set_opacity(["heart.aorta"], 0.4)
    editor.hide("heart.pericardium")
    restored = AWRScene.from_dict(scene.to_dict())
    assert restored.get("heart.aorta").state.opacity == pytest.approx(0.4)
    assert restored.get("heart.pericardium").state.visibility is False


def test_a_second_edit_does_not_undo_the_first(editor: SceneEditor, scene: AWRScene) -> None:
    """Edits accumulate. An edit forgotten when the next arrives is not persistent."""
    editor.set_opacity(["heart.left_atrium"], 0.3)
    editor.set_opacity(["heart.right_atrium"], 0.6)
    assert scene.get("heart.left_atrium").state.opacity == pytest.approx(0.3)
    assert scene.get("heart.right_atrium").state.opacity == pytest.approx(0.6)


def test_edit_then_reload_then_edit_again(editor: SceneEditor, scene: AWRScene) -> None:
    """The loop Step 8 was asked to regression-test, end to end."""
    editor.set_opacity(["heart.mitral_valve"], 0.2)
    restored = AWRScene.from_dict(scene.to_dict())
    second = SceneEditor(restored)
    second.set_opacity(["heart.tricuspid_valve"], 0.8)
    assert restored.get("heart.mitral_valve").state.opacity == pytest.approx(0.2)
    assert restored.get("heart.tricuspid_valve").state.opacity == pytest.approx(0.8)


def test_entity_identity_survives_every_edit(editor: SceneEditor, scene: AWRScene) -> None:
    """Identity is what makes the edited entity the same entity."""
    before = [entity.entity_id for entity in scene.iter_entities()]
    editor.set_opacity(["heart.aorta"], 0.5)
    editor.hide("heart.myocardium")
    editor.set_lod(2)
    after = [entity.entity_id for entity in scene.iter_entities()]
    assert before == after


def test_the_local_edit_head_still_scopes_to_its_target() -> None:
    """The Step 7 locality guarantee, checked against the Step 8 model shapes."""
    head = LocalEditHead(
        EditHeadConfig(
            entity_width=256,
            token_width=64,
            tokens=32,
            operations=len(OPERATIONS),
            identity_width=96,
            scope="target",
        )
    )
    with torch.no_grad():
        for parameter in head.delta[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        head.gate.bias.add_(4.0)
    tokens = torch.randn(1, 6, 32, 64)
    latent = torch.randn(1, 6, 256)
    mask = torch.zeros(1, 6)
    mask[0, 2] = 1.0
    edited = head(tokens, latent, encode_edit([0], [1.4], operations=len(OPERATIONS)), mask)
    assert not torch.equal(edited[0, 2], tokens[0, 2])
    for slot in (0, 1, 3, 4, 5):
        assert torch.equal(edited[0, slot], tokens[0, slot]), slot


def test_edit_cases_still_build_on_continuous_arrangements() -> None:
    """The arrangement change must not have broken the Step 7 edit-pair construction."""
    rng = np.random.default_rng(0)
    scene = build_scene(scene_index=0, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    case = build_edit_case(scene, EditOperation.THICKEN_VALVE)
    assert case.target == "heart.mitral_valve"
    assert case.change_fraction[case.target] > 0.05
    assert case.moved and case.still


def test_the_regression_helper_reports_every_check() -> None:
    """The report the Step 8 document cites must come from a run, not from prose."""
    report = editing_regression()
    assert report["checks"], "no checks ran"
    assert all(entry["passed"] for entry in report["checks"].values()), report
    assert report["all_passed"] is True
