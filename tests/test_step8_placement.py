"""Step 8 predicted placement: the frame head, the curriculum, and the shortcuts it closes.

Step 7 supplied every entity's true frame at evaluation time, so the frame head was never
exercised. These tests check that the head produces a usable frame, that the schedule does
what it says, and above all that an evaluation cannot see a true frame however the
schedule is set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from experiments.step8.frame_metrics import (
    COMPOSITE_WEIGHTS,
    frame_errors,
    position_error,
    rotation_error,
    scale_error,
)
from generation.neural.nn.geometry import FramePredictor
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.step8 import Step8Config, teacher_forcing_at


@pytest.fixture(scope="module")
def setup():
    """A batch from continuous arrangements, and the builder that made it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    rng = np.random.default_rng(3)
    scenes = [
        build_scene(scene_index=i, family_id=i % 3, lod=3, arrangement=sample_arrangement(rng))
        for i in range(4)
    ]
    return builder.build(scenes).batch, builder


def _canonical(shape: tuple[int, ...]) -> torch.Tensor:
    frame = torch.zeros(*shape, 12)
    frame[..., 3:6] = -1.9
    frame[..., 6] = 1.0
    frame[..., 10] = 1.0
    return frame


def test_the_frame_predictor_keeps_the_twelve_number_contract() -> None:
    """Changing the contract would invalidate every decoder and metric downstream."""
    head = FramePredictor(256)
    out = head(torch.randn(2, 5, 256))
    assert out.shape == (2, 5, 12)


def test_an_untrained_head_starts_at_the_canonical_frame() -> None:
    """Training should move away from a sane scene, not climb out of a numerical hole."""
    head = FramePredictor(256)
    out = head(torch.randn(2, 5, 256))
    assert torch.allclose(out, _canonical((2, 5)))


def test_the_head_has_enough_capacity_to_be_worth_training() -> None:
    """Step 7's head was 3,084 parameters and was never asked to do anything."""
    assert sum(p.numel() for p in FramePredictor(256).parameters()) > 50_000


def test_position_scale_and_rotation_errors_measure_what_they_say() -> None:
    """Each part is on its own terms; a single number would hide which one is wrong."""
    truth = _canonical((2, 4))
    shifted = truth.clone()
    shifted[..., 0] = 0.25
    assert math.isclose(float(position_error(shifted, truth).mean()), 0.25, rel_tol=1e-5)

    scaled = truth.clone()
    scaled[..., 3:6] = -1.9 + math.log(2.0)
    assert math.isclose(float(scale_error(scaled, truth).mean()), math.log(2.0), rel_tol=1e-4)

    turned = truth.clone()
    turned[..., 6:9] = torch.tensor([0.0, 1.0, 0.0])
    turned[..., 9:12] = torch.tensor([-1.0, 0.0, 0.0])
    assert math.isclose(float(rotation_error(turned, truth).mean()), math.pi / 2, rel_tol=1e-4)


def test_the_composite_error_is_the_documented_combination() -> None:
    """The weights are fixed in advance and reported, so they must be what is computed."""
    truth = _canonical((1, 3))
    predicted = truth.clone()
    predicted[..., 0] = 0.2
    predicted[..., 3:6] = -1.9 + 0.4
    present = torch.ones(1, 3, dtype=torch.bool)
    values = frame_errors(predicted, truth, present)
    expected = (
        COMPOSITE_WEIGHTS["position"] * 0.2
        + COMPOSITE_WEIGHTS["scale"] * 0.4
        + COMPOSITE_WEIGHTS["rotation"] * 0.0
    )
    assert math.isclose(values["composite_frame_error"], expected, rel_tol=1e-4)


def test_padded_entities_are_excluded_from_frame_errors() -> None:
    """Padding carries no frame; including it would dilute every number."""
    truth = _canonical((1, 4))
    predicted = truth.clone()
    predicted[0, 3, 0] = 5.0
    present = torch.tensor([[True, True, True, False]])
    values = frame_errors(predicted, truth, present)
    assert values["position_error"] == 0.0
    assert values["frame_entities"] == 3.0


def test_frame_errors_are_undefined_rather_than_zero_when_nothing_is_present() -> None:
    """A model that places nothing must not score a perfect placement."""
    truth = _canonical((1, 2))
    values = frame_errors(truth, truth, torch.zeros(1, 2, dtype=torch.bool))
    assert math.isnan(values["position_error"])


def test_the_curriculum_runs_from_supplied_to_predicted() -> None:
    """P0 supplies frames, P1 ramps down, P2 supplies none."""
    config = Step8Config(steps=1000, stage_p0=0.2, stage_p2=0.6)
    assert teacher_forcing_at(0, 1000, config) == 1.0
    assert teacher_forcing_at(199, 1000, config) == 1.0
    middle = teacher_forcing_at(400, 1000, config)
    assert 0.0 < middle < 1.0
    assert teacher_forcing_at(600, 1000, config) == 0.0
    assert teacher_forcing_at(999, 1000, config) == 0.0


def test_the_curriculum_ends_at_zero_teacher_forcing() -> None:
    """The headline result must not be produced with any frames supplied."""
    config = Step8Config(steps=800)
    assert teacher_forcing_at(config.steps - 1, config.steps, config) == 0.0
    assert config.to_dict()["curriculum"]["final_teacher_forcing"] == 0.0
    assert config.to_dict()["curriculum"]["inference_uses_predicted_frames"] is True


def test_predicted_frames_override_the_schedule(setup) -> None:
    """An evaluation must not be able to see a true frame, however the ratio is left.

    This is the guard against the Step 7 defect returning by accident: a teacher-forcing
    ratio left at 1.0 must not turn a headline result into an oracle result.
    """
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        strict = model(batch, use_predicted_frames=True, teacher_forcing=1.0)
        plain = model(batch, use_predicted_frames=True, teacher_forcing=0.0)
    assert torch.equal(strict.scene_logits, plain.scene_logits)


def test_supplied_and_predicted_placement_differ(setup) -> None:
    """If they matched, the condition would not be measuring anything."""
    batch, builder = setup
    model, _ = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    model.eval()
    with torch.no_grad():
        inferred = model(batch, use_predicted_frames=True).scene_logits
        supplied = model(batch, use_predicted_frames=False).scene_logits
    assert not torch.equal(inferred, supplied)


def test_every_arm_predicts_frames(setup) -> None:
    """A no-graph arm must still place entities, or the comparison is not like for like."""
    batch, builder = setup
    for arm in ("A1", "A1M", "A3Lite", "A3L", "A3"):
        model, groups = build_model(arm, builder, text_features=int(batch.text_features.shape[1]))
        model.eval()
        with torch.no_grad():
            output = model(batch, use_predicted_frames=True)
        assert output.frames is not None, arm
        assert output.frames.shape[-1] == 12, arm
        assert groups["frame_head"] > 50_000, arm
