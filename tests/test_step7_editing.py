"""Step 7 Experiments 8 and 9: the edit head and the edit-pair construction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.parameters import Variant
from datasets.whole_organ.sampling import (
    EDIT_SPECIFICATIONS,
    EditOperation,
    changed_entities,
    edit_pair,
)
from experiments.step7.editing import (
    OPERATIONS,
    build_edit_batch,
    build_edit_case,
    edited_parameters,
    rebuild_scene,
)
from generation.neural.nn.editing_head import EditHeadConfig, LocalEditHead, encode_edit
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder


@pytest.fixture(scope="module")
def builder() -> WholeOrganBatchBuilder:
    """A batch builder on the real heart ontology."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    return WholeOrganBatchBuilder(ontology, domain)


@pytest.fixture(scope="module")
def scenes():
    """Three finest-level scenes, enough to batch."""
    return [
        build_scene(scene_index=index, family_id=index % 2, variant=Variant.NORMAL, lod=3)
        for index in range(3)
    ]


def _head(scope: str) -> LocalEditHead:
    return LocalEditHead(
        EditHeadConfig(
            entity_width=64,
            token_width=32,
            tokens=8,
            operations=len(OPERATIONS),
            identity_width=32,
            scope=scope,
        )
    )


def test_untrained_head_is_the_identity() -> None:
    """A head that has learned nothing must leave the scene exactly as it was.

    Otherwise a measured change could be initialisation noise rather than an edit.
    """
    head = _head("target")
    tokens = torch.randn(2, 5, 8, 32)
    latent = torch.randn(2, 5, 64)
    edit = encode_edit([0, 1], [1.4, 0.7], operations=len(OPERATIONS))
    mask = torch.zeros(2, 5)
    mask[0, 2] = 1.0
    mask[1, 3] = 1.0
    assert torch.equal(head(tokens, latent, edit, mask), tokens)


def test_target_scope_leaves_other_token_blocks_untouched() -> None:
    """The locality guarantee of the target-scoped arm, checked bit for bit."""
    head = _head("target")
    with torch.no_grad():
        for parameter in head.delta[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        head.gate.bias.add_(4.0)
    tokens = torch.randn(2, 5, 8, 32)
    latent = torch.randn(2, 5, 64)
    edit = encode_edit([0, 1], [1.4, 0.7], operations=len(OPERATIONS))
    mask = torch.zeros(2, 5)
    mask[0, 2] = 1.0
    mask[1, 3] = 1.0
    edited = head(tokens, latent, edit, mask)
    assert not torch.equal(edited[0, 2], tokens[0, 2]), "the target was not edited"
    for slot in (0, 1, 3, 4):
        assert torch.equal(edited[0, slot], tokens[0, slot]), f"slot {slot} leaked"


def test_free_scope_may_touch_every_entity() -> None:
    """The contrast arm must genuinely be free, or it is not a control."""
    head = _head("free")
    with torch.no_grad():
        for parameter in head.delta[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        head.gate.bias.add_(4.0)
    tokens = torch.randn(1, 4, 8, 32)
    latent = torch.randn(1, 4, 64)
    edit = encode_edit([2], [1.3], operations=len(OPERATIONS))
    mask = torch.zeros(1, 4)
    mask[0, 1] = 1.0
    edited = head(tokens, latent, edit, mask)
    untouched = [slot for slot in range(4) if torch.equal(edited[0, slot], tokens[0, slot])]
    assert not untouched, f"free scope left {untouched} untouched"


def test_head_rejects_a_mismatched_target_mask() -> None:
    """A silently broadcast mask would edit the wrong entity."""
    head = _head("target")
    with pytest.raises(ValueError, match="Target mask"):
        head(
            torch.randn(2, 5, 8, 32),
            torch.randn(2, 5, 64),
            encode_edit([0, 0], [1.0, 1.0], operations=len(OPERATIONS)),
            torch.zeros(2, 4),
        )


def test_edit_encoding_separates_operation_from_magnitude() -> None:
    """The same operation at two strengths must share its one-hot part."""
    encoded = encode_edit([1, 1], [1.5, 0.5], operations=len(OPERATIONS))
    assert torch.equal(encoded[0, :-1], encoded[1, :-1])
    assert encoded[0, -1] != encoded[1, -1]


def test_every_operation_changes_its_declared_target(scenes) -> None:
    """The declared target must be an entity the generator actually alters.

    The wall edits originally named the myocardium, which the parameter only changes as
    a side effect; this test is what would have caught that.
    """
    for operation in EditOperation:
        target = EDIT_SPECIFICATIONS[operation][2]
        fractions = []
        for scene in scenes:
            _, _, changes = edit_pair(scene, operation)
            fractions.append(changes[target])
        mean = float(np.mean(fractions))
        assert mean >= 0.08, f"{operation} barely changes its target {target} ({mean:.3f})"


def test_edit_changes_are_measured_not_assumed(scenes) -> None:
    """Some entities must be reported as changed and some as untouched."""
    _, _, changes = edit_pair(scenes[0], EditOperation.THICKEN_VALVE)
    moved, still = changed_entities(changes)
    assert moved, "no entity changed, so the edit is inert"
    assert still, "every entity changed, so there is nothing to score leakage against"
    assert set(moved) & set(still) == set()


def test_edited_parameters_compose_commutatively(scenes) -> None:
    """Disjoint edits applied in either order must reach the same parameters."""
    first, second = EditOperation.ENLARGE_ATRIUM, EditOperation.WIDEN_VESSEL
    forward = edited_parameters(edited_parameters(scenes[0].parameters, first), second)
    backward = edited_parameters(edited_parameters(scenes[0].parameters, second), first)
    assert dict(sorted(forward.entity_scale)) == dict(sorted(backward.entity_scale))


def test_rebuild_scene_remeasures_rather_than_copying(scenes) -> None:
    """The edited scene's statistics must come from the edited organ."""
    scene = scenes[0]
    edited = rebuild_scene(scene, edited_parameters(scene.parameters, EditOperation.WIDEN_VESSEL))
    assert edited.counts["heart.aorta"] != scene.counts["heart.aorta"]


def test_before_and_after_share_one_point_set(builder, scenes) -> None:
    """Resampling between before and after would make untouched entities look changed."""
    cases = [
        case
        for case in (build_edit_case(scene, EditOperation.ENLARGE_VENTRICLE) for scene in scenes)
        if case.is_measurable
    ]
    assert cases, "every case removed an entity, so there is nothing to batch"
    edits = build_edit_batch(cases, builder)
    assert edits.after_ownership.shape == edits.before.batch.scene_points.shape[:2]
    assert edits.after_occupancy.shape == edits.before.batch.scene_points.shape[:2]
    before = torch.from_numpy(np.stack([item.ownership for item in edits.before.sampled]))
    changed = float((before != edits.after_ownership).float().mean())
    assert changed > 0.0, "the edit changed no point's owner"
    assert changed < 0.9, "nearly every point changed, so this is not a local edit"


def test_edit_batch_marks_the_target_and_the_untouched(builder, scenes) -> None:
    """The masks the locality metric depends on must be consistent with the case."""
    cases = [build_edit_case(scene, EditOperation.THICKEN_VALVE) for scene in scenes]
    edits = build_edit_batch(cases, builder)
    for index, case in enumerate(cases):
        slot = builder.slot_of[case.target]
        assert float(edits.target_mask[index, slot]) == 1.0
        assert float(edits.target_mask[index].sum()) == 1.0
        assert float(edits.unchanged_mask[index, slot]) == 0.0
        assert float(edits.unchanged_mask[index].sum()) == len(case.still)


def test_edit_batch_refuses_mixed_operations(builder, scenes) -> None:
    """One encoding per batch, so a batch may not mix operations."""
    mixed = [
        build_edit_case(scenes[0], EditOperation.WIDEN_VESSEL),
        build_edit_case(scenes[1], EditOperation.THIN_LINING),
    ]
    with pytest.raises(ValueError, match="one operation"):
        build_edit_batch(mixed, builder)


def test_operation_order_is_fixed() -> None:
    """The encoding indexes into this order, so it must not drift."""
    assert OPERATIONS == tuple(EDIT_SPECIFICATIONS)
    assert len(OPERATIONS) == 7


def test_an_edit_that_deletes_a_structure_is_flagged_not_batched(builder, scenes) -> None:
    """Shrinking a cavity destroys the structures derived from its surface.

    The septum is the tissue between two cavities and the valve annuli are the regions
    between them, so shrinking the left ventricle enough makes them cease to exist. A
    locality measurement compares a before and an after over one entity set, so these
    cases have no comparison to make and must be excluded explicitly rather than
    silently producing a malformed batch.
    """
    cases = [build_edit_case(scene, EditOperation.SHRINK_VENTRICLE) for scene in scenes]
    deleting = [case for case in cases if not case.is_measurable]
    assert deleting, "shrinking a ventricle removed nothing, so this guard is untested"
    for case in deleting:
        assert case.lost, "a case flagged unmeasurable must name what it lost"
    with pytest.raises(ValueError, match="remove an entity entirely"):
        build_edit_batch(deleting, builder)
