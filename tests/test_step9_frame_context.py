"""Step 9: the frame head's scene context, its objective, and the Step 8 guarantees.

Two switches are under test, and the most important property of both is that leaving them
off reproduces Step 8 exactly. An experiment whose control is not the previous system is
not a controlled experiment.
"""

from __future__ import annotations

import math

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step9.frame_report import (
    ROTATION_ZERO_TOLERANCE,
    decompose,
    describe_rotation,
    floor_gap,
)
from generation.neural.nn.geometry import FramePredictor
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.step9 import Step9Trainer, default_config, with_variant

CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def setup():
    """A batch at one level of detail, and the builder that made it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    loaded = load_step8_split(CORPUS, "test_seen", limit=48)
    level = loaded[0].active_lod
    scenes = [scene for scene in loaded if scene.active_lod == level][:3]
    return builder.build(scenes).batch, builder


def test_the_default_reproduces_the_step_8_head_exactly(setup) -> None:
    """The control must be the previous system, not a near neighbour of it."""
    batch, builder = setup
    _, groups = build_model("A3Lite", builder, text_features=int(batch.text_features.shape[1]))
    assert groups["frame_head"] == 135_180
    assert default_config().frame_scene_context is False
    assert default_config().frame_objective == "l1"


def test_scene_context_changes_only_the_frame_head(setup) -> None:
    """Everything else must stay identical, or the comparison is not isolating one thing."""
    batch, builder = setup
    features = int(batch.text_features.shape[1])
    _, plain = build_model("A3Lite", builder, text_features=features)
    _, context = build_model(
        "A3Lite", builder, text_features=features, overrides={"frame_scene_context": True}
    )
    for group, value in plain.items():
        if group in ("frame_head", "total"):
            continue
        assert context[group] == value, group
    assert context["frame_head"] > plain["frame_head"]


def test_the_scene_summary_is_permutation_invariant(setup) -> None:
    """A scene summary that depends on slot order would be a positional shortcut."""
    batch, builder = setup
    model, _ = build_model(
        "A3Lite",
        builder,
        text_features=int(batch.text_features.shape[1]),
        overrides={"frame_scene_context": True},
    )
    latent = torch.randn(batch.batch_size, batch.structure.entity_count, 256)
    order = torch.randperm(batch.structure.entity_count)
    direct = model.scene_summary(latent, batch.entity_present)
    permuted = model.scene_summary(latent[:, order], batch.entity_present[:, order])
    assert torch.allclose(direct, permuted, atol=1e-5)


def test_padded_slots_cannot_dilute_the_scene_summary(setup) -> None:
    """Otherwise a scene's summary would depend on how many slots happen to be padded."""
    batch, builder = setup
    model, _ = build_model(
        "A3Lite",
        builder,
        text_features=int(batch.text_features.shape[1]),
        overrides={"frame_scene_context": True},
    )
    latent = torch.randn(batch.batch_size, batch.structure.entity_count, 256)
    poisoned = latent.clone()
    poisoned[~batch.entity_present] = 1_000.0
    assert torch.allclose(
        model.scene_summary(latent, batch.entity_present),
        model.scene_summary(poisoned, batch.entity_present),
        atol=1e-5,
    )


def test_a_context_mismatch_fails_loudly() -> None:
    """Silently dropping the mechanism under test would invalidate the experiment."""
    with pytest.raises(ValueError, match="needs a scene summary"):
        FramePredictor(64, scene_context=True)(torch.randn(2, 4, 64))
    with pytest.raises(ValueError, match="without scene context"):
        FramePredictor(64)(torch.randn(2, 4, 64), torch.randn(2, 64))


def test_both_heads_start_at_the_canonical_frame() -> None:
    """Training must move away from a sane scene in both configurations."""
    plain = FramePredictor(64)
    context = FramePredictor(64, scene_context=True)
    latent = torch.randn(2, 4, 64)
    assert torch.allclose(plain(latent), context(latent, torch.randn(2, 64)))


def test_the_euclidean_objective_differs_from_l1(setup) -> None:
    """If the two agreed, the alignment experiment would be measuring nothing."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train = load_step8_split(CORPUS, "train", limit=32)
    evaluation = load_step8_split(CORPUS, "validation", limit=16)
    from dataclasses import replace

    losses = {}
    for objective in ("l1", "euclidean"):
        config = replace(
            with_variant(default_config(), scene_context=False, objective=objective, seed=0),
            steps=2,
            batch_size=4,
            eval_every=0,
        )
        trainer = Step9Trainer(config, ontology, domain, train, evaluation)
        batch = trainer.train_loader.builder.build(
            [s for s in train if s.active_lod == train[0].active_lod][:4]
        ).batch
        with torch.no_grad():
            output = trainer.model(batch, use_predicted_frames=True)
            losses[objective] = float(trainer._frame_loss(output, batch))
    assert not math.isclose(losses["l1"], losses["euclidean"], rel_tol=1e-6), losses


def test_the_decomposition_reports_the_rotation_as_carrying_nothing() -> None:
    """The corrected metric and the Step 8 composite must differ by the rotation term."""
    values = decompose(
        {
            "position_error": 0.1561,
            "scale_error": 0.1306,
            "rotation_error": 0.0,
            "composite_frame_error": 0.2214,
        }
    )
    assert values["rotation_information"] == 0.0
    assert math.isclose(values["placement_error"], 0.1561 + 0.5 * 0.1306, rel_tol=1e-9)
    assert "structurally zero" in describe_rotation(0.0)
    assert "non-zero" in describe_rotation(0.5)
    assert ROTATION_ZERO_TOLERANCE > 0.0


def test_the_floor_gap_has_the_sign_a_reader_expects() -> None:
    """Positive means better than a lookup table keyed on identity alone."""
    assert floor_gap(0.1561, 0.1605) > 0.0
    assert floor_gap(0.1676, 0.1605) < 0.0
    assert math.isnan(floor_gap(0.1, 0.0))


def test_run_ids_say_what_the_run_varies() -> None:
    """A directory of checkpoints should be readable without opening the manifests."""
    from dataclasses import replace

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train = load_step8_split(CORPUS, "train", limit=16)
    config = replace(
        with_variant(default_config(), scene_context=True, objective="euclidean", seed=2),
        steps=1,
        batch_size=4,
        eval_every=0,
    )
    trainer = Step9Trainer(config, ontology, domain, train, train)
    assert trainer.manifest.run_id == "s9-A3Lite-ctx-euclidean-seed2"
    assert trainer.manifest.config["step9"]["frame_scene_context"] is True
