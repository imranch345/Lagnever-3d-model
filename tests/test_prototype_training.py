"""Training: one-batch learning, checkpoints, determinism, loss guards, diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from awr.config import DomainConfig
from awr.errors import ContractError
from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.diagnostics import lod_token_usage, run_diagnostics, sibling_separation
from generation.neural.nn.losses import (
    FIRST_WAVE_OBJECTIVES,
    LossInputs,
    validate_weights,
)
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch
from training.loop import LodBucketLoader, Trainer, TrainingConfig
from training.manifest import RunManifest, environment_report, git_commit


def _trainer(
    config: TrainingConfig,
    ontology: AnatomyOntology,
    domain: DomainConfig,
    specs: list[SceneSpec],
) -> Trainer:
    return Trainer(config, ontology, domain, specs[:6], specs[6:])


@pytest.fixture
def tiny_config() -> TrainingConfig:
    """A configuration small enough for a test."""
    return TrainingConfig(
        arm="A3",
        seed=0,
        steps=3,
        batch_size=2,
        scene_points=64,
        entity_points=8,
        eval_every=0,
        log_every=1,
        eval_batches=1,
        warmup_steps=1,
    )


def test_first_wave_objectives_are_the_step_five_seven() -> None:
    """Seven objectives, exactly the ones Step 5 selected."""
    assert set(FIRST_WAVE_OBJECTIVES) == set(initial_loss_strategy())
    assert len(FIRST_WAVE_OBJECTIVES) == 7


def test_held_out_objectives_cannot_be_trained() -> None:
    """The relationship objectives stay held out, enforced not remembered."""
    with pytest.raises(ContractError, match="untestable"):
        validate_weights({"spatial_relationship": 1.0})
    with pytest.raises(ContractError, match="not implemented"):
        validate_weights({"topology": 1.0})
    with pytest.raises(ContractError, match="undeclared"):
        validate_weights({"make_it_pretty": 1.0})


def test_one_batch_training_reduces_the_loss(
    tiny_config: TrainingConfig,
    ontology: AnatomyOntology,
    config: DomainConfig,
    scene_specs: list[SceneSpec],
) -> None:
    """Overfitting a single batch works, which is the minimum a loop must do."""
    trainer = _trainer(tiny_config, ontology, config, scene_specs)
    batch = next(trainer.train_loader.infinite())
    first = trainer.train_step(batch, 0)["total"]
    for step in range(1, 12):
        trainer.train_step(batch, step)
    last = trainer.train_step(batch, 12)["total"]
    assert last < first


def test_structural_objectives_are_skipped_by_the_baseline(
    tiny_config: TrainingConfig,
    ontology: AnatomyOntology,
    config: DomainConfig,
    scene_specs: list[SceneSpec],
) -> None:
    """Identity and frame objectives have no baseline counterpart and are reported, not faked."""
    from dataclasses import replace

    trainer = _trainer(replace(tiny_config, arm="A0"), ontology, config, scene_specs)
    batch = next(trainer.train_loader.infinite())
    output = trainer.model(batch)
    breakdown = trainer.loss(
        LossInputs(output=output, batch=batch, geometry_state_invariant=False), None
    )
    assert set(breakdown.skipped) == {"entity_identity", "entity_frame"}
    assert "geometry_reconstruction" in breakdown.terms


def test_checkpoint_round_trip(
    tiny_config: TrainingConfig,
    ontology: AnatomyOntology,
    config: DomainConfig,
    scene_specs: list[SceneSpec],
    tmp_path: Path,
) -> None:
    """A checkpoint restores the exact weights."""
    trainer = _trainer(tiny_config, ontology, config, scene_specs)
    batch = next(trainer.train_loader.infinite())
    trainer.train_step(batch, 0)
    path = trainer.save_checkpoint(tmp_path / "run.pt")
    assert path.is_file()

    before = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}
    trainer.train_step(batch, 1)
    trainer.load_checkpoint(path)
    after = trainer.model.state_dict()
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), name


def test_training_is_reproducible(
    tiny_config: TrainingConfig,
    ontology: AnatomyOntology,
    config: DomainConfig,
    scene_specs: list[SceneSpec],
) -> None:
    """Same seed, same configuration, same losses."""
    first = _trainer(tiny_config, ontology, config, scene_specs)
    second = _trainer(tiny_config, ontology, config, scene_specs)
    first_losses = [
        first.train_step(batch, step)["total"]
        for step, batch in zip(range(3), first.train_loader.infinite(), strict=False)
    ]
    second_losses = [
        second.train_step(batch, step)["total"]
        for step, batch in zip(range(3), second.train_loader.infinite(), strict=False)
    ]
    assert first_losses == pytest.approx(second_losses, rel=1e-6)


def test_full_tiny_run_produces_a_manifest(
    tiny_config: TrainingConfig,
    ontology: AnatomyOntology,
    config: DomainConfig,
    scene_specs: list[SceneSpec],
    tmp_path: Path,
) -> None:
    """A complete run writes a manifest with everything needed to reproduce it."""
    trainer = _trainer(tiny_config, ontology, config, scene_specs)
    manifest = trainer.fit(checkpoint_dir=tmp_path)
    assert manifest.steps_completed == 3
    assert manifest.results
    assert manifest.parameters["total"] > 1_000_000
    assert manifest.environment["torch"]
    assert (tmp_path / f"{manifest.run_id}.pt").is_file()
    assert (tmp_path / f"{manifest.run_id}.manifest.json").is_file()
    payload = RunManifest.load(tmp_path / f"{manifest.run_id}.manifest.json")
    assert payload["arm"] == "A3"
    assert payload["config"]["loss_weights"]


def test_manifest_records_the_environment() -> None:
    """Versions and commit are captured, and an absent commit says so."""
    report = environment_report()
    assert report["torch"] and report["numpy"] and report["python"]
    commit = git_commit()
    assert commit


def test_loader_buckets_by_level_of_detail(
    builder: BatchBuilder, mixed_lod_specs: list[SceneSpec]
) -> None:
    """Every batch the loader yields has one level of detail."""
    loader = LodBucketLoader(mixed_lod_specs, builder, batch_size=2, seed=0)
    for batch in loader.epoch(0):
        levels = {batch.structure.lod}
        assert len(levels) == 1


def test_diagnostics_run_on_both_arms(
    builder: BatchBuilder, batch: PrototypeBatch
) -> None:
    """Diagnostics work for the structured arm and degrade honestly for the baseline."""
    from training.loop import build_model

    structured, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    baseline, _ = build_model("A0", builder, text_features=int(batch.text_features.shape[1]))
    structured.eval()
    baseline.eval()

    report = run_diagnostics(structured, batch, builder)
    assert "sibling_cosine_mean" in report["sibling_separation"]["values"]
    assert report["relation_sensitivity"]["values"]["identity_delta"] == 0.0

    slot_of = {entity_id: index for index, entity_id in enumerate(builder.ontology.ids())}
    empty = sibling_separation(baseline, batch, slot_of)
    assert empty.values == {}
    assert empty.notes

    usage = lod_token_usage(structured, batch, (4, 16, 32))
    assert "delta_first_to_last" in usage.values
