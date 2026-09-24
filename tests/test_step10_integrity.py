"""Step 10 §27: the checks that fail when the placement pipeline is quietly corrupted.

Every defect Steps 7 to 9 lost time to was silent: the ablation whose arms received the
same input, the evaluation that supplied ground-truth frames, the floor measured over the
wrong entity set, the parent table that indexed the wrong entities. None of them raised.

So these tests do not check that the pipeline works. They corrupt it deliberately, one
thing at a time, and assert that the corruption is **detected**. A check that cannot fail
is not evidence, and a test that passes on broken data is worse than no test.
"""

from __future__ import annotations

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from datasets.whole_organ.hierarchy import ROOT, parent_slots, topological_order
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import (
    as_parent,
    compose_hierarchy,
    frame_to_matrix,
    relative,
    similarity_residual,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.step10 import Step10Config, default_config, with_variant
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
    """One batch of real frames, as the model would receive them."""
    domain = load_domain_config()
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = load_step8_split(CORPUS, "test_seen", limit=64)
    loader = WholeOrganLoader(
        scenes, builder, batch_size=8, seed=0, shuffle=False, drop_last=False
    )
    whole = next(iter(loader.epoch(0)))
    return builder, whole.batch.entity_frames.double(), whole.batch.entity_present.bool()


def _placement(builder, width: int) -> HierarchicalPlacement:
    return HierarchicalPlacement(dict(builder.slot_of), slots=width)


def _rotate(frames: torch.Tensor, angle: float) -> torch.Tensor:
    """Give every entity a real rotation, which the corpus itself does not have."""
    turn = torch.tensor(angle, dtype=frames.dtype)
    cos, sin = torch.cos(turn), torch.sin(turn)
    out = frames.clone()
    out[..., 6:9] = torch.tensor([cos, -sin, 0.0], dtype=frames.dtype)
    out[..., 9:12] = torch.tensor([sin, cos, 0.0], dtype=frames.dtype)
    return out


def _local_under(
    placement: HierarchicalPlacement, frames: torch.Tensor, present: torch.Tensor, convention: str
) -> torch.Tensor:
    """Build the parent-relative target under an explicit convention."""
    parents = placement.parents_for(present)
    local = frames.clone()
    rows = torch.arange(frames.shape[0])
    for slot in placement._order.tolist():
        column = parents[:, slot]
        mask = column >= 0
        if not bool(mask.any()):
            continue
        candidate = relative(frames[rows, column.clamp_min(0)], frames[:, slot], convention)
        local[:, slot] = torch.where(mask.unsqueeze(-1), candidate, frames[:, slot])
    return local


def _round_trip_error(placement: HierarchicalPlacement, frames, present) -> float:
    target = placement.to_local(frames, present)
    rebuilt = placement.to_global(target.local, target.parents)
    return float((frame_to_matrix(rebuilt)[1] - frame_to_matrix(frames)[1])[present].abs().max())


class TestCorruptingTheParentTable:
    """A wrong tree must not reconstruct the truth; if it does, the tree is not being used."""

    def test_a_correct_table_reconstructs(self, scene) -> None:
        """The control: the check can pass."""
        builder, frames, present = scene
        assert _round_trip_error(_placement(builder, frames.shape[1]), frames, present) < 1e-9

    def test_a_shuffled_table_is_detected(self, scene) -> None:
        """Reversing the parent assignments must change what is reconstructed."""
        builder, frames, present = scene
        table = list(parent_slots("spatial", dict(builder.slot_of), slots=frames.shape[1]))
        parented = [slot for slot, parent in enumerate(table) if parent != ROOT]
        for slot, other in zip(parented, reversed(parented), strict=True):
            table[slot] = table[other]
        corrupt = HierarchicalPlacement.from_parents(table)
        # Composing a target built on the true tree through the wrong tree must move things.
        true_placement = _placement(builder, frames.shape[1])
        target = true_placement.to_local(frames, present)
        rebuilt = corrupt.to_global(target.local, corrupt.parents_for(present))
        moved = (frame_to_matrix(rebuilt)[1] - frame_to_matrix(frames)[1])[present].abs().max()
        assert float(moved) > 1e-3, "the hierarchy is not affecting the result at all"

    def test_an_empty_table_is_refused_by_the_model(self) -> None:
        """A table that silently became empty would train the control arm under another name."""
        from generation.neural.nn.model import LagnavPrototype, PrototypeConfig

        config = PrototypeConfig(placement_target="parent_relative", placement_parents=())
        with pytest.raises(ValueError, match="needs placement_parents"):
            LagnavPrototype(config, torch.zeros(1, dtype=torch.long))

    def test_an_unknown_target_is_refused(self) -> None:
        """A typo must not fall through to the global control."""
        from generation.neural.nn.model import LagnavPrototype, PrototypeConfig

        config = PrototypeConfig(placement_target="relative")
        with pytest.raises(ValueError, match="Unknown placement_target"):
            LagnavPrototype(config, torch.zeros(1, dtype=torch.long))

    def test_a_cycle_is_refused(self, scene) -> None:
        """A cycle would compose against a stale frame rather than raise."""
        builder, frames, _ = scene
        table = list(parent_slots("spatial", dict(builder.slot_of), slots=frames.shape[1]))
        parented = [slot for slot, parent in enumerate(table) if parent != ROOT]
        table[table[parented[0]]] = parented[0]
        with pytest.raises(ValueError, match="cycle"):
            HierarchicalPlacement.from_parents(table)


class TestCorruptingTheOrder:
    """Composition reads a parent's already-composed frame, so the order is load-bearing."""

    def test_reversing_the_order_breaks_reconstruction(self, scene) -> None:
        """Walking children before parents must not still produce the truth."""
        builder, frames, present = scene
        placement = _placement(builder, frames.shape[1])
        target = placement.to_local(frames, present)
        parents = target.parents
        table = parent_slots("spatial", dict(builder.slot_of), slots=frames.shape[1])
        backwards = torch.tensor(list(reversed(topological_order(table))))
        rebuilt = compose_hierarchy(target.local, parents, backwards)
        deepest = [slot for slot, parent in enumerate(table) if parent != ROOT]
        error = (frame_to_matrix(rebuilt)[1] - frame_to_matrix(frames)[1])[:, deepest].abs().max()
        assert float(error) > 1e-6, "order is not affecting composition; the walk is wrong"


class TestCorruptingTheTransform:
    """The conventions the target's exactness depends on."""

    def test_swapping_the_stored_basis_vectors_is_detected(self, scene) -> None:
        """A corrupted local rotation must not compose back to the truth.

        Swapping the two stored vectors is *not* a transpose: on this corpus's identity
        rotations it produces a 180-degree turn about (1, 1, 0), while a real transpose of
        the identity changes nothing. The transpose itself is tested, with real rotations,
        in ``test_step10_corruption``.
        """
        builder, frames, present = scene
        placement = _placement(builder, frames.shape[1])
        target = placement.to_local(frames, present)
        flipped = target.local.clone()
        flipped[..., 6:9], flipped[..., 9:12] = target.local[..., 9:12], target.local[..., 6:9]
        rebuilt = placement.to_global(flipped, target.parents)
        moved = (frame_to_matrix(rebuilt)[1] - frame_to_matrix(frames)[1])[present].abs().max()
        assert float(moved) > 1e-6

    def test_the_unclosed_convention_loses_accuracy_once_rotations_are_real(self, scene) -> None:
        """The measurement behind the table in the transforms docstring.

        With the corpus's own anisotropic scales and a real rotation, the ``full``
        convention cannot reconstruct what it decomposed, and the closed one can. If this
        ever stops being true the isotropic reduction is unnecessary and should be removed.
        """
        builder, frames, present = scene
        rotated = _rotate(frames, 0.9)
        placement = _placement(builder, frames.shape[1])
        errors = {}
        for convention in ("isotropic", "full"):
            local = _local_under(placement, rotated, present, convention)
            rebuilt = compose_hierarchy(
                local, placement.parents_for(present), placement._order, convention
            )
            moved = (
                frame_to_matrix(rebuilt)[1] - frame_to_matrix(rotated)[1]
            )[present].norm(dim=-1)
            errors[convention] = float(moved.max())
        assert errors["isotropic"] < 1e-9
        assert errors["full"] > 1e-3, (
            "the unclosed convention is no longer losing anything, so the closure argument "
            "needs revisiting rather than keeping"
        )

    def test_the_closed_convention_leaves_no_shear_on_the_same_data(self, scene) -> None:
        """The convention actually used must leave nothing for the projection to discard."""
        builder, frames, present = scene
        placement = _placement(builder, frames.shape[1])
        target = placement.to_local(frames, present)
        rows = torch.arange(frames.shape[0])
        for slot in range(frames.shape[1]):
            column = target.parents[:, slot]
            if not bool((column >= 0).any()):
                continue
            mask = column >= 0
            parent_frames = as_parent(frames[rows, column.clamp_min(0)], placement.convention)
            linear, _ = frame_to_matrix(parent_frames[mask])
            product = linear @ frame_to_matrix(target.local[:, slot][mask])[0]
            assert float(similarity_residual(product).max()) < 1e-9


class TestTheControlArmIsReallyAControl:
    """The taxonomic arm must be numerically identical to the global one, not merely close."""

    def test_the_taxonomic_tree_has_no_parents(self, ontology) -> None:
        """AWR's hierarchy leaves every whole-organ entity a root."""
        resolved = with_variant(
            default_config("A3Lite"),
            placement_target="parent_relative",
            hierarchy="taxonomic",
            seed=0,
        ).resolved(ontology)
        assert all(slot == ROOT for slot in resolved.placement_parents)

    def test_composing_over_it_is_the_identity(self, scene, ontology) -> None:
        """The control must be the control exactly, not approximately."""
        builder, frames, present = scene
        taxonomic = HierarchicalPlacement.from_parents(
            parent_slots("taxonomic", dict(builder.slot_of), slots=frames.shape[1]),
            hierarchy="taxonomic",
        )
        target = taxonomic.to_local(frames, present)
        assert torch.equal(target.local, frames)
        assert torch.equal(taxonomic.to_global(target.local, target.parents), frames)

    def test_a_bad_hierarchy_name_is_refused(self) -> None:
        """Silently defaulting would run the wrong arm under the right name."""
        with pytest.raises(ValueError, match="Unknown hierarchy"):
            Step10Config(arm="A3Lite", loss_weights={}, placement_hierarchy="spacial")

    def test_a_bad_convention_name_is_refused(self) -> None:
        """Silently defaulting would run the wrong convention under the right name."""
        with pytest.raises(ValueError, match="Unknown parent convention"):
            Step10Config(arm="A3Lite", loss_weights={}, parent_convention="isotropic_scale")


class TestTheCorpusPremises:
    """Claims Step 10 rests on that live in the data, not in the code."""

    def test_relative_is_the_exact_inverse_of_compose_on_real_frames(self, scene) -> None:
        """The target-building and target-consuming halves must agree on real data."""
        builder, frames, present = scene
        placement = _placement(builder, frames.shape[1])
        rows = torch.arange(frames.shape[0])
        parents = placement.parents_for(present)
        for slot in range(frames.shape[1]):
            column = parents[:, slot]
            mask = column >= 0
            if not bool(mask.any()):
                continue
            parent_frames = frames[rows, column.clamp_min(0)][mask]
            child_frames = frames[mask, slot]
            local = relative(parent_frames, child_frames)
            rebuilt = placement.compose_one(parent_frames, local)
            assert torch.allclose(
                frame_to_matrix(rebuilt)[0], frame_to_matrix(child_frames)[0], atol=1e-9
            )


class TestTheScaleCascade:
    """A parent's scale error multiplies into its descendants; a position error only adds.

    This is a property of the target, not of any model, and it is why ``rigid`` is a cell in
    the Step 10 suite rather than a curiosity: it propagates a parent's orientation without
    its size, so the multiplicative term disappears.
    """

    @staticmethod
    def _chain_targets(convention: str, error: float) -> list[float]:
        """Scale error per depth after mispredicting the root's scale by ``error``."""
        parents = torch.tensor([-1, 0, 1, 2])
        order = torch.tensor([0, 1, 2, 3])
        generator = torch.Generator().manual_seed(4)
        truth = torch.randn(16, 4, 12, generator=generator, dtype=torch.float64)
        truth[..., 3:6] *= 0.4
        local = truth.clone()
        for slot in range(1, 4):
            local[:, slot] = relative(truth[:, slot - 1], truth[:, slot], convention)
        # the head gets the root's scale wrong and everything else exactly right
        predicted = local.clone()
        predicted[:, 0, 3:6] += error
        rebuilt = compose_hierarchy(predicted, parents, order, convention)
        return [
            float((rebuilt[:, slot, 3:6] - truth[:, slot, 3:6]).abs().mean())
            for slot in range(4)
        ]

    def test_isotropic_propagates_a_parents_scale_error(self) -> None:
        """Every descendant inherits the root's mistake, undiminished."""
        errors = self._chain_targets("isotropic", 0.5)
        assert errors[0] > 0.4
        for slot in range(1, 4):
            assert errors[slot] > 0.4, f"depth {slot} did not inherit the parent's scale error"

    def test_rigid_confines_a_scale_error_to_the_entity_that_made_it(self) -> None:
        """The reason T4_rigid is worth running."""
        errors = self._chain_targets("rigid", 0.5)
        assert errors[0] > 0.4
        for slot in range(1, 4):
            assert errors[slot] < 1e-9, (
                f"depth {slot} inherited a scale error under rigid, which should not "
                "propagate scale at all"
            )
