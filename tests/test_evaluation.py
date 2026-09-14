"""Evaluation layer: benchmarks, consistency and honest reporting of gaps."""

from __future__ import annotations

from awr.scene import AWRScene
from editing.scene_editor import SceneEditor
from evaluation.benchmarks import HEART_V01_BENCHMARK, run_benchmark, run_benchmark_suite
from evaluation.consistency import (
    evaluate_edit_consistency,
    evaluate_history_integrity,
    evaluate_identity_stability,
    evaluate_persistence,
)
from evaluation.geometry import evaluate_correspondence


def test_benchmark_suite_is_green() -> None:
    """Every specified workflow passes end to end."""
    report = run_benchmark_suite()
    assert report.is_green, report.summary()
    assert report.passed == len(HEART_V01_BENCHMARK)


def test_benchmarks_are_deterministic() -> None:
    """Running a case twice gives the same outcome and the same scene version."""
    first = run_benchmark(HEART_V01_BENCHMARK[1])
    second = run_benchmark(HEART_V01_BENCHMARK[1])
    assert first.passed == second.passed
    assert first.scene_version == second.scene_version


def test_identity_stability_after_edits(scene: AWRScene) -> None:
    """Ids are unchanged by a sequence of edits."""
    before = scene.ids()
    editor = SceneEditor(scene)
    editor.isolate("heart.chambers")
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    editor.set_lod(3)
    assert evaluate_identity_stability(before, scene).is_valid


def test_identity_loss_is_reported(scene: AWRScene) -> None:
    """Removing an entity would be flagged as an identity failure."""
    before = scene.ids()
    del scene.entities["heart.aorta"]
    report = evaluate_identity_stability(before, scene)
    assert not report.is_valid
    assert any(issue.code == "ENTITY_ID_LOST" for issue in report.errors)


def test_edit_consistency_matches_reported_deltas(scene: AWRScene) -> None:
    """An operation changes exactly the entities it reports."""
    editor = SceneEditor(scene)
    before = scene.state_snapshot()
    applied = editor.hide("heart.valves")
    after = scene.state_snapshot()
    report = evaluate_edit_consistency(
        before, after, expected_changed=applied.result.affected, context="hide valves"
    )
    assert report.is_valid, str(report)


def test_undeclared_change_is_detected(scene: AWRScene) -> None:
    """A change an operation did not declare is an error."""
    before = scene.state_snapshot()
    scene.entities["heart.aorta"].state.opacity = 0.2
    after = scene.state_snapshot()
    report = evaluate_edit_consistency(before, after, expected_changed=("heart.left_ventricle",))
    assert any(issue.code == "UNDECLARED_CHANGE" for issue in report.errors)


def test_persistence_and_history_integrity(scene: AWRScene) -> None:
    """A scene round-trips and its history is a contiguous log."""
    editor = SceneEditor(scene)
    editor.isolate("heart.chambers")
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    assert evaluate_persistence(scene).is_valid
    assert evaluate_history_integrity(scene).is_valid


def test_correspondence_report_states_that_no_geometry_exists(scene: AWRScene) -> None:
    """The correspondence report never implies geometry has been produced."""
    report = evaluate_correspondence(scene)
    assert report.is_valid
    assert any(issue.code == "NO_GEOMETRY_PAYLOAD" for issue in report.issues)


def test_missing_geometry_reference_is_reported(scene: AWRScene) -> None:
    """A renderable entity without a reserved component is an error."""
    scene.entities["heart.aorta"].state.geometry_reference = None
    report = evaluate_correspondence(scene)
    assert any(issue.code == "MISSING_GEOMETRY_REFERENCE" for issue in report.errors)
