"""Step 10 Change 2 §3–§5: the rotation objective, and that it changes only rotation.

Four things are pinned here. The default reproduces Change 1 bit for bit, so the frozen
result stays reproducible. The chordal term is a distance on rotation *matrices*, so it
cannot be reduced by inflating the stored basis vectors. It is smooth where the geodesic
angle is not. And switching it on leaves the translation and scale terms untouched, so any
difference between the two objectives is the rotation term alone.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from generation.neural.nn.transforms import rotation_angle, rotation_chordal
from training.step10 import Step10Trainer, default_config, with_variant

ROTATED = "datasets/processed/step10_rotated"
IDENTITY_CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def ontology():
    """Heart Ontology v0.1."""
    domain = load_domain_config()
    return load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )


def _trainer(ontology, corpus: str, *, objective: str = "chordal", weight: float = 0.4):
    train = load_step8_split(corpus, "train", limit=48)
    validation = load_step8_split(corpus, "validation", limit=16)
    config = replace(
        with_variant(default_config("A1"), placement_target="global", hierarchy="spatial", seed=0),
        rotation_objective=objective,  # type: ignore[arg-type]
        rotation_loss_weight=weight,
        steps=2,
        batch_size=8,
        device="cpu",
        eval_every=10**6,
    )
    return Step10Trainer(config, ontology, load_domain_config(), train, validation)


def _frame(rotation: torch.Tensor, *, scale: float = 0.0) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(3, dtype=torch.float64),
            torch.full((3,), scale, dtype=torch.float64),
            rotation[0],
            rotation[1],
        ]
    )


def _about_z(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )


class TestTheDefaultReproducesChange1:
    """The frozen result must stay reproducible."""

    def test_the_default_objective_is_step8(self) -> None:
        """A configuration that says nothing gets Step 8's term."""
        assert default_config("A1").rotation_objective == "step8_6d_l1"

    def test_the_default_frame_loss_is_the_step9_one(self, ontology) -> None:
        """Bit-identical, on the corpus Change 1 ran on."""
        trainer = _trainer(ontology, IDENTITY_CORPUS, objective="step8_6d_l1")
        whole = next(iter(trainer.train_loader.epoch(0)))
        batch = whole.batch.to(trainer.device)
        with torch.no_grad():
            output = trainer.model(batch, teacher_forcing=1.0)
            from training.step9 import Step9Trainer

            mine = trainer._frame_loss(output, batch)
            theirs = Step9Trainer._frame_loss(trainer, output, batch)
        assert float(mine) == float(theirs)


class TestTheChordalTerm:
    """§4: a proper distance on rotations, smooth at zero."""

    def test_it_is_zero_for_equal_rotations(self) -> None:
        """No penalty when the rotation is right."""
        frame = _frame(_about_z(37.0))
        assert float(rotation_chordal(frame, frame)) < 1e-15

    def test_it_is_invariant_to_inflating_the_stored_basis(self) -> None:
        """The six-number term could be reduced by scaling the basis; this cannot.

        The stored vectors are Gram-Schmidted before the distance is taken, so a model cannot
        lower the rotation loss by shrinking its basis vectors towards the target's without
        changing the rotation they represent.
        """
        target = _frame(_about_z(20.0))
        predicted = _frame(_about_z(50.0))
        inflated = predicted.clone()
        inflated[6:12] *= 7.5
        assert float(rotation_chordal(predicted, target)) == pytest.approx(
            float(rotation_chordal(inflated, target)), abs=1e-12
        )
        raw = float((predicted[6:12] - target[6:12]).abs().sum())
        raw_inflated = float((inflated[6:12] - target[6:12]).abs().sum())
        assert raw != pytest.approx(raw_inflated, abs=1e-6)

    @pytest.mark.parametrize("degrees", (0.0, 1.0, 10.0, 45.0, 90.0, 135.0, 180.0))
    def test_it_is_monotone_in_the_angle(self, degrees: float) -> None:
        """Equivalent ordering to the geodesic angle, which is what makes it a valid proxy."""
        identity = _frame(torch.eye(3, dtype=torch.float64))
        rotated = _frame(_about_z(degrees))
        # ||Ra - Rb||_F^2 = 4 (1 - cos t), and the helper divides by 8.
        expected = (1.0 - math.cos(math.radians(degrees))) / 2.0
        assert float(rotation_chordal(identity, rotated)) == pytest.approx(expected, abs=1e-9)

    def test_it_is_bounded(self) -> None:
        """In [0, 1], so its weight means the same thing at every angle."""
        identity = _frame(torch.eye(3, dtype=torch.float64))
        for degrees in (0.0, 90.0, 180.0):
            value = float(rotation_chordal(identity, _frame(_about_z(degrees))))
            assert 0.0 <= value <= 1.0

    def test_its_gradient_is_finite_at_zero_where_the_geodesic_diverges(self) -> None:
        """§4's reason for descending this and reporting the other."""
        identity = _frame(torch.eye(3, dtype=torch.float64))
        near = _frame(_about_z(1e-3)).requires_grad_(True)
        rotation_chordal(identity, near).backward()
        chordal_grad = float(near.grad.abs().max())  # type: ignore[union-attr]
        other = _frame(_about_z(1e-3)).requires_grad_(True)
        rotation_angle(identity, other).backward()
        geodesic_grad = float(other.grad.abs().max())  # type: ignore[union-attr]
        assert math.isfinite(chordal_grad)
        assert chordal_grad < 0.01
        assert geodesic_grad > 100.0 * chordal_grad


class TestSwitchingItOnChangesOnlyRotation:
    """§6: the other terms must be untouched, or a comparison means nothing."""

    def test_translation_and_scale_terms_are_unchanged(self, ontology) -> None:
        """The two objectives share their translation and scale parts exactly."""
        trainer = _trainer(ontology, ROTATED)
        whole = next(iter(trainer.train_loader.epoch(0)))
        batch = whole.batch.to(trainer.device)
        with torch.no_grad():
            output = trainer.model(batch, teacher_forcing=1.0)
            parts = trainer._frame_loss_parts(output.frames, batch.entity_frames)
            truth = batch.entity_frames
            expected_translation = (
                (output.frames[..., 0:3] - truth[..., 0:3]).abs().sum(dim=-1, keepdim=True)
            )
            expected_scale = (
                (output.frames[..., 3:6] - truth[..., 3:6]).abs().sum(dim=-1, keepdim=True)
            )
        assert torch.equal(parts["translation"], expected_translation)
        assert torch.equal(parts["scale"], expected_scale)

    def test_the_weight_reaches_the_loss(self, ontology) -> None:
        """Doubling the weight moves the loss by the rotation term's contribution."""
        trainer = _trainer(ontology, ROTATED, weight=0.4)
        whole = next(iter(trainer.train_loader.epoch(0)))
        batch = whole.batch.to(trainer.device)
        with torch.no_grad():
            output = trainer.model(batch, teacher_forcing=1.0)
            first = float(trainer._frame_loss(output, batch))
            parts = trainer._frame_loss_parts(output.frames, batch.entity_frames)
            present = batch.entity_present.unsqueeze(-1).to(output.frames.dtype)
            contribution = float(
                (parts["rotation"] * present).sum() / present.sum().clamp_min(1)
            )
            trainer.config = replace(trainer.config, rotation_loss_weight=0.8)
            second = float(trainer._frame_loss(output, batch))
        # float32 frames: the tolerance is the dtype's, not the arithmetic's.
        assert second - first == pytest.approx(0.4 * contribution, rel=1e-5)

    def test_the_manifest_records_the_objective_and_the_weight(self, ontology) -> None:
        """§5: a run that does not say which weight it used is not reproducible."""
        trainer = _trainer(ontology, ROTATED, weight=0.37)
        step10 = trainer.manifest.config["step10"]
        assert step10["rotation_objective"] == "chordal"
        assert step10["rotation_loss_weight"] == 0.37


class TestTheRotationTermIsNoLongerDegenerate:
    """On the Change 1 corpus this term was identically zero; here it is not."""

    @staticmethod
    def _identity_like(frames: torch.Tensor) -> torch.Tensor:
        identity = torch.zeros_like(frames)
        identity[..., 6] = 1.0
        identity[..., 10] = 1.0
        return identity

    def test_the_target_rotations_are_not_constant(self, ontology) -> None:
        """Measured on the rotated corpus, through the loader."""
        trainer = _trainer(ontology, ROTATED)
        whole = next(iter(trainer.train_loader.epoch(0)))
        batch = whole.batch.to(trainer.device)
        present = batch.entity_present.bool()
        angles = rotation_angle(batch.entity_frames, self._identity_like(batch.entity_frames))
        assert float(angles[present].std()) > 0.1
        assert float(angles[present].mean()) > 0.3

    def test_the_same_term_is_zero_on_the_identity_corpus(self, ontology) -> None:
        """Which is exactly why Change 1 could not test rotation."""
        trainer = _trainer(ontology, IDENTITY_CORPUS)
        whole = next(iter(trainer.train_loader.epoch(0)))
        batch = whole.batch.to(trainer.device)
        present = batch.entity_present.bool()
        truth = batch.entity_frames
        identity = self._identity_like(truth)
        assert float(rotation_chordal(identity, truth)[present].max()) < 1e-12
