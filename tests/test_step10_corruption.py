"""Step 10 §27 and §9: corrupted hierarchy metadata is refused, and placement cannot leak.

Each corruption is applied deliberately and must raise :class:`PlacementIntegrityError`
from the gate the trainer runs before it trains. A corruption that merely makes a number
worse is not enough: the requirement is that it cannot pass through the pipeline silently.
Every corruption class also has its control — the correct metadata passing the same check —
so a gate that rejected everything would fail here too.
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

import generation.neural.nn.transforms as transforms
from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ import hierarchy as hierarchy_module
from datasets.whole_organ.continuous_corpus import load_step8_split
from datasets.whole_organ.hierarchy import ROOT, parent_slots
from generation.neural.nn.geometry import frame_rotation
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.placement_integrity import (
    ROUND_TRIP_TOLERANCE,
    PlacementIntegrityError,
    check_frames,
    check_order,
    check_round_trip,
    check_slot_table,
    check_source_table,
    validate_placement,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def ontology():
    """Heart Ontology v0.1."""
    domain = load_domain_config()
    return load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )


@pytest.fixture(scope="module")
def scene(ontology):
    """One batch of real corpus frames and the builder's slot map."""
    builder = WholeOrganBatchBuilder(ontology, load_domain_config())
    scenes = load_step8_split(CORPUS, "test_seen", limit=16)
    loader = WholeOrganLoader(
        scenes, builder, batch_size=8, seed=0, shuffle=False, drop_last=False
    )
    batch = next(iter(loader.epoch(0))).batch
    return dict(builder.slot_of), batch.entity_frames, batch.entity_present


def _placement(slot_of, width: int) -> HierarchicalPlacement:
    return HierarchicalPlacement(slot_of, slots=width)


def _shuffled(slot_of, width: int) -> list[int]:
    """The spatial table with its parent assignments reversed among parented slots."""
    table = list(parent_slots("spatial", slot_of, slots=width))
    parented = [slot for slot, parent in enumerate(table) if parent != ROOT]
    parents = [table[slot] for slot in parented]
    for slot, parent in zip(parented, reversed(parents), strict=True):
        table[slot] = parent
    return table


def _trainer(
    ontology,
    *,
    target: str = "parent_relative",
    hierarchy: str = "spatial",
    train_split: str = "train",
):
    train = load_step8_split(CORPUS, train_split, limit=16)
    validation = load_step8_split(CORPUS, "validation", limit=8)
    config = replace(
        with_variant(default_config("A1"), placement_target=target, hierarchy=hierarchy, seed=0),  # type: ignore[arg-type]
        steps=2,
        batch_size=8,
        device="cpu",
        eval_every=10**6,
    )
    return Step10Trainer(config, ontology, load_domain_config(), train, validation)


class TestTheCorrectMetadataPasses:
    """The controls: every check can pass, so a failure below is the corruption's doing."""

    def test_every_declared_hierarchy_matches_the_generator(self) -> None:
        """Spatial, alternative and taxonomic trees all agree with construction."""
        for name in ("spatial", "spatial_alternative", "taxonomic"):
            check_source_table(name)

    def test_the_spatial_placement_passes_the_whole_gate(self, scene) -> None:
        """The real table, order and convention pass every check on real frames."""
        slot_of, frames, present = scene
        report = validate_placement(
            _placement(slot_of, frames.shape[1]),
            slot_of=slot_of,
            hierarchy="spatial",
            frames=frames,
            present=present,
        )
        assert report["parented_slots"] == 10
        assert report["round_trip_error"] <= ROUND_TRIP_TOLERANCE
        assert report["probe_round_trip_error"] <= ROUND_TRIP_TOLERANCE

    def test_the_global_target_passes_with_nothing_to_compose(self, scene) -> None:
        """The control arm is checked too, and passes."""
        slot_of, frames, present = scene
        report = validate_placement(
            None, slot_of=slot_of, hierarchy="spatial", frames=frames, present=present
        )
        assert report["composition"] == "none: global target"

    def test_the_trainer_records_a_passing_gate(self, ontology) -> None:
        """The gate runs inside the trainer and its report reaches the manifest."""
        trainer = _trainer(ontology)
        assert trainer.integrity["parented_slots"] == 10
        assert trainer.manifest.config["step10"]["integrity"] == trainer.integrity

    def test_the_corpus_rotations_are_all_the_identity(self, scene) -> None:
        """The premise that makes the probe rotation necessary, measured rather than assumed."""
        _, frames, present = scene
        assert check_frames(frames, present)["max_rotation_deviation_from_identity"] == 0.0


class TestAShuffledParentTableIsRefused:
    """§27 A."""

    def test_a_shuffled_slot_table_fails_the_slot_check(self, scene) -> None:
        """Reversed parent assignments are named slot by slot."""
        slot_of, frames, _ = scene
        with pytest.raises(PlacementIntegrityError, match="hierarchy says"):
            check_slot_table(_shuffled(slot_of, frames.shape[1]), slot_of, "spatial")

    def test_a_shuffled_slot_table_fails_the_whole_gate(self, scene) -> None:
        """The same corruption is refused by the gate the trainer calls."""
        slot_of, frames, present = scene
        corrupt = HierarchicalPlacement.from_parents(_shuffled(slot_of, frames.shape[1]))
        with pytest.raises(PlacementIntegrityError):
            validate_placement(
                corrupt, slot_of=slot_of, hierarchy="spatial", frames=frames, present=present
            )

    def test_a_table_built_in_the_wrong_slot_ordering_is_refused(self, scene) -> None:
        """The hazard parent_slots documents: the 20-entity ordering on a 64-slot batch."""
        slot_of, frames, _ = scene
        wrong = parent_slots("spatial", None, slots=frames.shape[1])
        with pytest.raises(PlacementIntegrityError):
            check_slot_table(wrong, slot_of, "spatial")

    def test_a_shuffled_source_table_fails_against_the_generator(self, monkeypatch) -> None:
        """Checking the table against a copy of itself would pass; the generator does not."""
        source = dict(hierarchy_module.SPATIAL_PARENT)
        children = sorted(source)
        shuffled = dict(zip(children, reversed([source[c] for c in children]), strict=True))
        monkeypatch.setitem(hierarchy_module.HIERARCHIES, "spatial", shuffled)
        with pytest.raises(PlacementIntegrityError, match="generator builds it from"):
            check_source_table("spatial")

    def test_one_wrong_edge_is_enough(self, monkeypatch) -> None:
        """A single plausible-looking mistake, not only a wholesale shuffle."""
        edited = {**hierarchy_module.SPATIAL_PARENT, "heart.aorta": "heart.right_ventricle"}
        monkeypatch.setitem(hierarchy_module.HIERARCHIES, "spatial", edited)
        with pytest.raises(PlacementIntegrityError, match="heart.aorta"):
            check_source_table("spatial")

    def test_the_trainer_refuses_to_start_on_a_shuffled_table(self, ontology, monkeypatch) -> None:
        """The pipeline-level requirement: it cannot pass through training silently."""
        source = dict(hierarchy_module.SPATIAL_PARENT)
        children = sorted(source)
        shuffled = dict(zip(children, reversed([source[c] for c in children]), strict=True))
        monkeypatch.setitem(hierarchy_module.HIERARCHIES, "spatial", shuffled)
        with pytest.raises(PlacementIntegrityError):
            _trainer(ontology)


class TestATransposedRotationIsRefused:
    """§27 B."""

    @staticmethod
    def _transposing(frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``frame_to_matrix`` with the bug its first version had: the rotation transposed."""
        rotation = frame_rotation(frame).transpose(-1, -2)
        scale = frame[..., 3:6].exp().clamp_min(1e-6)
        return rotation * scale.unsqueeze(-2), frame[..., 0:3]

    def test_the_gate_catches_a_transposed_rotation(self, scene, monkeypatch) -> None:
        """The historic frame_to_matrix transpose bug is refused."""
        slot_of, frames, present = scene
        placement = _placement(slot_of, frames.shape[1])
        monkeypatch.setattr(transforms, "frame_to_matrix", self._transposing)
        with pytest.raises(PlacementIntegrityError, match="probe-rotated"):
            check_round_trip(placement, frames, present)

    def test_the_stored_frames_alone_could_not_have_caught_it(self, scene, monkeypatch) -> None:
        """Why the probe exists: on identity rotations the transpose is invisible."""
        slot_of, frames, present = scene
        placement = _placement(slot_of, frames.shape[1])
        monkeypatch.setattr(transforms, "frame_to_matrix", self._transposing)
        frames64 = frames.double()
        target = placement.to_local(frames64, present)
        rebuilt = placement.to_global(target.local, target.parents)
        error = float((rebuilt - frames64)[present.bool()].abs().max())
        assert error <= ROUND_TRIP_TOLERANCE

    def test_a_transposed_rotation_in_this_corpus_is_the_same_data(self, scene) -> None:
        """So the data-level form of this corruption cannot occur on this corpus at all."""
        _, frames, present = scene
        rotation = frame_rotation(frames.double())
        assert torch.equal(rotation[present.bool()], rotation.transpose(-1, -2)[present.bool()])

    def test_a_non_orthonormal_stored_basis_is_refused(self, scene) -> None:
        """The model would Gram-Schmidt this away; the gate reads the raw vectors."""
        slot_of, frames, present = scene
        corrupt = frames.clone()
        first = int(present[0].nonzero()[0])
        corrupt[0, first, 9:12] = torch.tensor([0.6, 0.8, 0.0])
        with pytest.raises(PlacementIntegrityError, match="not orthonormal"):
            check_frames(corrupt, present)


class TestABrokenOrderIsRefused:
    """§27 C."""

    def test_a_reversed_order_fails_the_order_check(self, scene) -> None:
        """Children composed before parents are refused."""
        slot_of, frames, _ = scene
        placement = _placement(slot_of, frames.shape[1])
        with pytest.raises(PlacementIntegrityError, match="before its parent"):
            check_order(placement.table, tuple(reversed(placement.order)))

    def test_an_order_that_skips_a_slot_is_refused(self, scene) -> None:
        """An order that composes one slot twice and another never is refused."""
        slot_of, frames, _ = scene
        placement = _placement(slot_of, frames.shape[1])
        order = list(placement.order)
        order[-1] = order[0]
        with pytest.raises(PlacementIntegrityError, match="permutation"):
            check_order(placement.table, order)

    def test_a_placement_holding_a_broken_order_fails_the_whole_gate(self, scene) -> None:
        """A corrupt order inside a real placement is refused by the full gate."""
        slot_of, frames, present = scene
        placement = _placement(slot_of, frames.shape[1])
        placement._order = torch.flip(placement._order, dims=[0])
        with pytest.raises(PlacementIntegrityError, match="before its parent"):
            validate_placement(
                placement, slot_of=slot_of, hierarchy="spatial", frames=frames, present=present
            )


class TestTheGateDoesNotPerturbTraining:
    """A gate that advanced a random generator would move every result it guards."""

    def test_every_random_source_is_restored(self, ontology) -> None:
        """Torch, NumPy and Python generators are unchanged by the gate."""
        trainer = _trainer(ontology)
        torch.manual_seed(11)
        np.random.seed(11)
        random.seed(11)
        before = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
        trainer.validate(load_step8_split(CORPUS, "train", limit=16))
        after = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
        assert torch.equal(before[0], after[0])
        assert np.array_equal(before[1], after[1])
        assert before[2] == after[2]


class TestNoPlacementLeakage:
    """§9: inferred placement is a function of the model's inputs, never of the answer."""

    def test_inferred_output_ignores_the_true_frames(self, ontology) -> None:
        """Replace every true frame with noise; nothing the metric reads may change."""
        trainer = _trainer(ontology)
        trainer.model.eval()
        scenes = load_step8_split(CORPUS, "test_arrangement", limit=16)
        level = scenes[0].active_lod
        batch = trainer.builder.build([s for s in scenes if s.active_lod == level][:8]).batch
        noisy = replace(batch, entity_frames=torch.randn_like(batch.entity_frames) * 5.0)
        with torch.no_grad():
            clean = trainer.model(batch, use_predicted_frames=True)
            corrupted = trainer.model(noisy, use_predicted_frames=True)
        assert torch.equal(clean.frames, corrupted.frames)
        assert torch.equal(clean.part_logits, corrupted.part_logits)

    def test_the_parent_table_does_not_depend_on_the_scenes(self, ontology) -> None:
        """Built on held-out arrangements or on training scenes, the table is the same."""
        on_train = _trainer(ontology, train_split="train")
        on_held_out = _trainer(ontology, train_split="test_arrangement")
        assert on_train.model.placement.table == on_held_out.model.placement.table
        assert on_train.config.placement_parents == on_held_out.config.placement_parents

    def test_the_default_table_is_not_the_arrangement_oracle(self, ontology) -> None:
        """The run's table is the fixed convention, never the transposed oracle."""
        slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        assert parent_slots("spatial", slot_of) != parent_slots("spatial", slot_of, transpose=True)
        config = with_variant(default_config("A1"), placement_target="parent_relative", seed=0)
        assert config.resolved(ontology).placement_parents == parent_slots("spatial", slot_of)
