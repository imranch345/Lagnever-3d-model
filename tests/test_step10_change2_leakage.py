"""Step 10 Change 2 §17: the model's inferred output owes nothing to the true frames.

Change 1 established this on the identity-rotation corpus. It has to hold again here, and
for a stronger reason: the frames now carry real rotations, so a leak would hand the model
the one thing Change 2 is trying to measure whether it can learn.

The test is the established noise-replacement one — every true frame in the batch is replaced
with noise, and nothing the metrics read may move — run against the rotated corpus with the
chordal objective active.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from datasets.whole_organ.hierarchy import parent_slots
from training.step10 import Step10Trainer, default_config, with_variant

CORPUS = "datasets/processed/step10_rotated"


@pytest.fixture(scope="module")
def ontology():
    """Heart Ontology v0.1."""
    domain = load_domain_config()
    return load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )


def _trainer(ontology, *, target: str = "global", convention: str = "isotropic"):
    domain = load_domain_config()
    train = load_step8_split(CORPUS, "train", limit=48)
    validation = load_step8_split(CORPUS, "validation", limit=16)
    config = replace(
        with_variant(
            default_config("A1"), placement_target=target, hierarchy="spatial", seed=0
        ),
        parent_convention=convention,
        rotation_objective="chordal",
        rotation_loss_weight=0.33,
        steps=2,
        batch_size=8,
        device="cpu",
        eval_every=10**6,
    )
    return Step10Trainer(config, ontology, domain, train, validation)


def _batch(trainer, split: str = "test_arrangement"):
    scenes = load_step8_split(CORPUS, split, limit=16)
    level = scenes[0].active_lod
    return trainer.builder.build([s for s in scenes if s.active_lod == level][:8]).batch


class TestNoFrameLeakage:
    """Inferred placement is a function of the model's inputs, never of the answer."""

    @pytest.mark.parametrize(
        ("target", "convention"),
        (("global", "isotropic"), ("parent_relative", "isotropic"), ("parent_relative", "rigid")),
    )
    def test_inferred_output_ignores_the_true_frames(
        self, ontology, target: str, convention: str
    ) -> None:
        """Every cell of the Change 2 matrix, including both parent conventions."""
        trainer = _trainer(ontology, target=target, convention=convention)
        trainer.model.eval()
        batch = _batch(trainer)
        noisy = replace(batch, entity_frames=torch.randn_like(batch.entity_frames) * 5.0)
        with torch.no_grad():
            clean = trainer.model(batch, use_predicted_frames=True)
            corrupted = trainer.model(noisy, use_predicted_frames=True)
        assert torch.equal(clean.frames, corrupted.frames)
        assert torch.equal(clean.part_logits, corrupted.part_logits)

    def test_rotating_the_true_frames_does_not_move_the_prediction(self, ontology) -> None:
        """A sharper probe than noise: a *valid* alternative rotation must not leak either."""
        trainer = _trainer(ontology)
        trainer.model.eval()
        batch = _batch(trainer)
        swapped = batch.entity_frames.clone()
        swapped[..., 6:9], swapped[..., 9:12] = (
            batch.entity_frames[..., 9:12],
            batch.entity_frames[..., 6:9],
        )
        with torch.no_grad():
            clean = trainer.model(batch, use_predicted_frames=True)
            rotated_batch = replace(batch, entity_frames=swapped)
            altered = trainer.model(rotated_batch, use_predicted_frames=True)
        assert torch.equal(clean.frames, altered.frames)

    def test_the_supplied_condition_is_the_only_path_that_sees_them(self, ontology) -> None:
        """The oracle condition must still differ, or the test above proves nothing."""
        trainer = _trainer(ontology)
        trainer.model.eval()
        batch = _batch(trainer)
        with torch.no_grad():
            inferred = trainer.model(batch, use_predicted_frames=True)
            supplied = trainer.model(batch, use_predicted_frames=False, teacher_forcing=1.0)
        assert not torch.equal(inferred.part_logits, supplied.part_logits)


class TestNoArrangementLeakage:
    """The parent table cannot carry the held-out arrangement into the model."""

    def test_the_table_is_the_same_on_every_split(self, ontology) -> None:
        """Built from the ontology, never from the scenes."""
        slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        config = with_variant(
            default_config("A1"), placement_target="parent_relative", seed=0
        ).resolved(ontology)
        assert config.placement_parents == parent_slots("spatial", slot_of)

    def test_the_arrangement_oracle_table_is_not_the_one_in_use(self, ontology) -> None:
        """The transposed table would encode the held-out arrangement; it is not used."""
        slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        config = with_variant(
            default_config("A1"), placement_target="parent_relative", seed=0
        ).resolved(ontology)
        assert config.placement_parents != parent_slots("spatial", slot_of, transpose=True)
