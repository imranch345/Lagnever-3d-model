"""Step 10: the spatial parent table, and the claims it rests on.

Change 1 asks for placement relative to a parent and offers ``left_ventricle ->
mitral_valve`` as the example. That tree is not in AWR, so this module writes the spatial
hierarchy down from the generator's construction order instead. Everything asserted about
why that was necessary is checked here against the ontology and the corpus, so the
justification cannot quietly stop being true.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from datasets.whole_organ.field import WHOLE_ORGAN_ENTITIES
from datasets.whole_organ.hierarchy import (
    HIERARCHIES,
    ROOT,
    SPATIAL_PARENT,
    describe,
    parent_of,
    parent_slots,
    topological_order,
)
from generation.neural.nn.transforms import (
    compose_hierarchy,
    frame_to_matrix,
    relative,
    resolve_parents,
    rotation_angle,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

CORPUS = "datasets/processed/step8_continuous"


@pytest.fixture(scope="module")
def builder() -> WholeOrganBatchBuilder:
    """A batch builder on the real heart ontology."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return WholeOrganBatchBuilder(ontology, domain)


@pytest.fixture(scope="module")
def slot_of(builder: WholeOrganBatchBuilder) -> dict[str, int]:
    """The batch builder's own mapping, which is the only correct one."""
    return dict(builder.slot_of)


@pytest.fixture(scope="module")
def batches(builder: WholeOrganBatchBuilder):
    """Frames and presence from the held-out split, as the model would see them."""
    scenes = load_step8_split(CORPUS, "test_seen", limit=64)
    loader = WholeOrganLoader(scenes, builder, batch_size=8, seed=0, shuffle=False, drop_last=False)
    return [
        (whole.batch.entity_frames.double(), whole.batch.entity_present.bool())
        for whole in loader.epoch(0)
    ]


class TestWhyTheTableExists:
    """The finding that made a new table necessary, pinned against the ontology."""

    def test_awr_has_no_spatial_parent_among_the_whole_organ_entities(self, builder) -> None:
        """The finding that made a separate table necessary."""
        whole_organ = set(WHOLE_ORGAN_ENTITIES)
        parent_of_id = builder.ontology.parent_of
        nested = [
            (child, parent_of_id.get(child))
            for child in sorted(whole_organ)
            if parent_of_id.get(child) in whole_organ
        ]
        assert nested == [], (
            "AWR now nests whole-organ entities inside each other; the spatial table was "
            "written because it did not, so revisit it rather than keeping both."
        )

    def test_the_briefs_example_tree_is_not_in_the_ontology(self, builder, slot_of) -> None:
        """``left_ventricle -> mitral_valve`` is the example Change 1 gives."""
        assert "heart.mitral_valve" in slot_of
        assert "heart.left_ventricle" in slot_of
        assert builder.ontology.parent_of["heart.mitral_valve"] == "heart.valves"
        assert parent_of("heart.mitral_valve", "taxonomic") is None

    def test_the_taxonomic_parents_carry_no_geometry(self, builder) -> None:
        """Composing against a category node would compose against a frame that does not exist."""
        parent_of_id = builder.ontology.parent_of
        for entity_id in WHOLE_ORGAN_ENTITIES:
            assert parent_of_id[entity_id] not in WHOLE_ORGAN_ENTITIES

    def test_the_spatial_table_supplies_it(self) -> None:
        """What AWR lacks, the construction-derived table provides."""
        assert parent_of("heart.mitral_valve") == "heart.left_ventricle"


class TestTableShape:
    """What the table contains, and the index bug it is built to prevent."""

    def test_every_entity_named_is_a_real_whole_organ_entity(self) -> None:
        """A typo in the table would silently leave an entity parentless."""
        known = set(WHOLE_ORGAN_ENTITIES)
        for table in HIERARCHIES.values():
            for child, parent in table.items():
                assert child in known, child
                assert parent in known, parent

    def test_the_taxonomic_control_has_no_parents(self) -> None:
        """The control arm must be AWR's hierarchy, not a softened version of it."""
        assert HIERARCHIES["taxonomic"] == {}

    def test_only_half_the_entities_have_a_parent(self, slot_of) -> None:
        """Change 1 can only move ten of twenty entities; the report must say so."""
        parents = parent_slots("spatial", slot_of)
        assert sum(1 for parent in parents if parent != ROOT) == 10

    def test_depth_is_three(self, slot_of) -> None:
        """The depth analysis quotes these counts."""
        shape = describe("spatial", slot_of)
        assert shape["max_depth"] == 3
        assert shape["depth_histogram"]["1"] == 7
        assert shape["depth_histogram"]["2"] == 2
        assert shape["depth_histogram"]["3"] == 1

    def test_slots_follow_the_builder_not_the_entity_tuple(self, slot_of) -> None:
        """Indexing a 64-slot batch by a position in WHOLE_ORGAN_ENTITIES is a silent bug."""
        by_builder = parent_slots("spatial", slot_of)
        by_tuple = parent_slots("spatial")
        assert by_builder != by_tuple
        child = slot_of["heart.mitral_valve"]
        assert by_builder[child] == slot_of["heart.left_ventricle"]

    def test_a_missing_entity_is_refused_rather_than_dropped(self) -> None:
        """A slot map that cannot express the table must raise."""
        with pytest.raises(KeyError, match="silently address the wrong entity"):
            parent_slots("spatial", {"heart.mitral_valve": 0})

    def test_topological_order_places_parents_first(self, slot_of) -> None:
        """Composition reads the parent's result, so the order must be real."""
        parents = parent_slots("spatial", slot_of)
        seen: set[int] = set()
        for slot in topological_order(parents):
            if parents[slot] != ROOT:
                assert parents[slot] in seen
            seen.add(slot)

    def test_a_cycle_is_refused(self) -> None:
        """A cycle would compose against a stale frame forever."""
        with pytest.raises(ValueError, match="cycle"):
            topological_order((1, 0))

    def test_the_alternative_convention_differs_only_at_the_av_valves(self) -> None:
        """The sensitivity arm must vary one choice, not several."""
        differing = {
            key
            for key in set(SPATIAL_PARENT) | set(HIERARCHIES["spatial_alternative"])
            if SPATIAL_PARENT.get(key) != HIERARCHIES["spatial_alternative"].get(key)
        }
        assert differing == {"heart.mitral_valve", "heart.tricuspid_valve"}


class TestAgainstTheCorpus:
    """Claims about the data, checked against the data."""

    def test_the_corpus_rotation_target_is_a_constant_identity(self, batches) -> None:
        """Change 2's premise, and the reason the closure defect stayed hidden."""
        for frames, present in batches:
            from generation.neural.nn.geometry import frame_rotation

            rotation = frame_rotation(frames)[present]
            identity = torch.eye(3, dtype=rotation.dtype).expand_as(rotation)
            assert torch.allclose(rotation, identity, atol=1e-12)

    def test_scales_are_strongly_anisotropic(self, batches) -> None:
        """Isotropic scales would have made the closure question moot."""
        spread = torch.cat(
            [
                (frames[..., 3:6].amax(-1) - frames[..., 3:6].amin(-1))[present]
                for frames, present in batches
            ]
        )
        assert float(spread.mean()) > 0.3

    def test_levels_of_detail_orphan_some_children(self, batches, slot_of) -> None:
        """The reason reparenting exists; if this stops being true, simplify."""
        parents = parent_slots("spatial", slot_of, slots=batches[0][1].shape[1])
        static = torch.tensor(parents)
        orphaned = children = 0
        for _, present in batches:
            has_parent = (static.unsqueeze(0) >= 0) & present
            children += int(has_parent.sum())
            absent = torch.zeros_like(present)
            for slot, parent in enumerate(parents):
                if parent != ROOT:
                    absent[:, slot] = ~present[:, parent]
            orphaned += int((has_parent & absent).sum())
        assert children > 0
        assert 0 < orphaned < children

    def test_reparenting_leaves_no_orphans(self, batches, slot_of) -> None:
        """Every present child ends up composed against a parent that is also present."""
        parents = parent_slots("spatial", slot_of, slots=batches[0][1].shape[1])
        static = torch.tensor(parents)
        order = torch.tensor(topological_order(parents))
        for _, present in batches:
            resolved = resolve_parents(static, present, order)
            had_parent = (static.unsqueeze(0) >= 0) & present
            # every present entity that had a static parent keeps a present one
            assert bool((resolved[had_parent] >= 0).all())
            kept = resolved[had_parent]
            assert bool(present.gather(1, resolved.clamp_min(0))[had_parent].all()), (
                "reparented onto an absent ancestor"
            )
            assert int(kept.numel()) > 0

    @pytest.mark.parametrize("degrees", [0.0, 30.0, 90.0])
    def test_parent_relative_target_reconstructs_the_global_frames(
        self, batches, slot_of, degrees: float
    ) -> None:
        """The property the whole of Change 1 rests on, at rotations Change 2 will create.

        If this drifts, the target is no longer the truth and every arm is scored against a
        moved goalpost.
        """
        parents = parent_slots("spatial", slot_of, slots=batches[0][1].shape[1])
        static = torch.tensor(parents)
        order = torch.tensor(topological_order(parents))
        generator = torch.Generator().manual_seed(0)
        for frames, present in batches:
            working = _with_rotation(frames, degrees, generator)
            resolved = resolve_parents(static, present, order)
            local = _to_local(working, resolved, order)
            rebuilt = compose_hierarchy(local, resolved, order)
            rebuilt_linear, rebuilt_translation = frame_to_matrix(rebuilt)
            true_linear, true_translation = frame_to_matrix(working)
            assert torch.allclose(
                rebuilt_translation[present], true_translation[present], atol=1e-9
            )
            assert torch.allclose(rebuilt_linear[present], true_linear[present], atol=1e-9)
            assert float(rotation_angle(rebuilt, working)[present].max()) < 1e-6


def _to_local(frames: torch.Tensor, resolved: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Build the parent-relative target the way training will."""
    local = frames.clone()
    rows = torch.arange(frames.shape[0])
    for slot in order.tolist():
        has = resolved[:, slot] >= 0
        if not bool(has.any()):
            continue
        candidate = relative(frames[rows, resolved[:, slot].clamp_min(0)], frames[:, slot])
        local[:, slot] = torch.where(has.unsqueeze(-1), candidate, frames[:, slot])
    return local


def _with_rotation(
    frames: torch.Tensor, degrees: float, generator: torch.Generator
) -> torch.Tensor:
    """Replace the corpus's constant identity rotation with a real one."""
    if degrees == 0.0:
        return frames
    shape = frames.shape[:-1]
    axis = torch.nn.functional.normalize(
        torch.randn(*shape, 3, generator=generator, dtype=frames.dtype), dim=-1
    )
    angle = torch.rand(*shape, 1, 1, generator=generator, dtype=frames.dtype) * np.deg2rad(degrees)
    cross = torch.zeros(*shape, 3, 3, dtype=frames.dtype)
    cross[..., 0, 1], cross[..., 0, 2] = -axis[..., 2], axis[..., 1]
    cross[..., 1, 0], cross[..., 1, 2] = axis[..., 2], -axis[..., 0]
    cross[..., 2, 0], cross[..., 2, 1] = -axis[..., 1], axis[..., 0]
    identity = torch.eye(3, dtype=frames.dtype).expand_as(cross)
    rotation = identity + torch.sin(angle) * cross + (1 - torch.cos(angle)) * (cross @ cross)
    out = frames.clone()
    out[..., 6:9], out[..., 9:12] = rotation[..., 0, :], rotation[..., 1, :]
    return out
