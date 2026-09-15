"""Step 7 Experiments 8 and 9: persistent local geometric editing.

Step 6 shipped the editing *interfaces* and never tested them, so the claim that
per-entity geometry token blocks buy **local** editing was unsupported. This module
builds the data and the measurements that can support or refute it.

The construction that makes the measurement honest: the before-organ and the
after-organ are evaluated at the **same points**. An edit is then a change in the label
of a fixed point set, not a comparison between two independent samples, so a difference
cannot be an artefact of resampling.

What the generator says an edit did is measured, never assumed. `edit_pair` reports the
fraction of each entity's points whose ownership changed, and entities below the
threshold are the ones the edit genuinely left alone. Those are the entities against
which leakage is scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, cast

import numpy as np
import torch

from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.field import WholeOrganField
from datasets.whole_organ.parameters import OrganParameters
from datasets.whole_organ.relations import measure_relations
from datasets.whole_organ.sampling import (
    EDIT_SPECIFICATIONS,
    EditOperation,
    changed_entities,
    edit_pair,
)
from generation.neural.nn.editing_head import encode_edit
from generation.neural.nn.whole_organ import WholeOrganBatch, WholeOrganBatchBuilder

__all__ = [
    "OPERATIONS",
    "EditCase",
    "EditBatch",
    "edited_parameters",
    "rebuild_scene",
    "build_edit_case",
    "build_edit_batch",
]

#: Fixed operation order. The edit encoding indexes into this, so it must not be
#: reordered once a head has been trained.
OPERATIONS: tuple[EditOperation, ...] = tuple(EDIT_SPECIFICATIONS)


@dataclass(frozen=True, slots=True)
class EditCase:
    """One scene with one edit applied, before and after sharing a point set."""

    scene: WholeOrganScene
    after_scene: WholeOrganScene
    operation: EditOperation
    target: str
    moved: tuple[str, ...]
    still: tuple[str, ...]
    change_fraction: Mapping[str, float]
    #: Entities present before the edit that own no volume afterwards. An edit that
    #: deletes a structure is outside what a locality measurement can score, because the
    #: metric compares a before and an after over the same entity set.
    lost: tuple[str, ...] = ()

    @property
    def is_measurable(self) -> bool:
        """True when the edit changed the organ without removing any structure."""
        return not self.lost

    @property
    def operation_index(self) -> int:
        """Position in :data:`OPERATIONS`."""
        return OPERATIONS.index(self.operation)

    @property
    def magnitude(self) -> float:
        """The scale factor the generator applied."""
        return float(EDIT_SPECIFICATIONS[self.operation][1])


@dataclass(frozen=True, slots=True)
class EditBatch:
    """A before-batch, the after-truth on the same points, and the edit request."""

    before: WholeOrganBatch
    after: WholeOrganBatch
    cases: tuple[EditCase, ...]
    #: ``[B, F]`` operation one-hot plus magnitude.
    edit: torch.Tensor
    #: ``[B, N]`` with 1.0 on the edited entity.
    target_mask: torch.Tensor
    #: ``[B, P]`` true ownership after the edit, at the before-batch's scene points.
    after_ownership: torch.Tensor
    #: ``[B, P]`` true scene occupancy after the edit, at the same points.
    after_occupancy: torch.Tensor
    #: ``[B, N, Q]`` true occupancy after the edit, at the before-batch's entity points.
    after_entity_occupancy: torch.Tensor
    #: ``[B, N]`` entities the generator's measurement says the edit did not alter.
    unchanged_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of cases."""
        return len(self.cases)


def edited_parameters(parameters: OrganParameters, operation: EditOperation) -> OrganParameters:
    """Apply one controlled edit to the organ parameters.

    Kept in one place so the edit the head is trained on and the edit the regeneration
    control re-decodes are provably the same edit.
    """
    parameter, scale, target = EDIT_SPECIFICATIONS[operation]
    if parameter == "entity":
        return dataclass_replace(
            parameters, entity_scale=(*parameters.entity_scale, (target, scale))
        )
    return dataclass_replace(
        parameters,
        **cast(dict[str, Any], {parameter: float(getattr(parameters, parameter)) * scale}),
    )


def rebuild_scene(scene: WholeOrganScene, parameters: OrganParameters) -> WholeOrganScene:
    """Re-measure a scene's statistics and relations from edited parameters.

    The regeneration control needs a scene description that matches the edited organ,
    and every quantity in it is measured from that organ rather than carried over.
    """
    organ = WholeOrganField(parameters)
    statistics = organ.entity_statistics(np.random.default_rng(scene.seed + 101))
    edges = measure_relations(organ, statistics, np.random.default_rng(scene.seed + 103))
    return dataclass_replace(
        scene,
        parameters=parameters,
        centroids={name: value["centroid"].tolist() for name, value in statistics.items()},
        extents={name: value["extent"].tolist() for name, value in statistics.items()},
        counts={name: int(value["count"][0]) for name, value in statistics.items()},
        edges=edges,
    )


def build_edit_case(
    scene: WholeOrganScene, operation: EditOperation, *, threshold: float = 0.08
) -> EditCase:
    """Measure what one edit actually does to one scene."""
    _, _, changes = edit_pair(scene, operation, change_threshold=threshold)
    moved, still = changed_entities(changes, threshold=threshold)
    target = EDIT_SPECIFICATIONS[operation][2]
    after = rebuild_scene(scene, edited_parameters(scene.parameters, operation))
    lost = tuple(
        name
        for name in scene.entity_ids
        if scene.counts.get(name, 0) > 0 and after.counts.get(name, 0) == 0
    )
    return EditCase(
        scene=scene,
        after_scene=after,
        operation=operation,
        target=target,
        moved=moved,
        still=still,
        change_fraction=changes,
        lost=lost,
    )


def build_edit_batch(
    cases: Sequence[EditCase],
    builder: WholeOrganBatchBuilder,
    *,
    epoch: int = 0,
) -> EditBatch:
    """Batch a set of edit cases, with before and after sharing every point.

    The after-truth is obtained by relabelling the before-batch's own points under the
    edited organ. Sampling fresh points for the after-organ would make an unchanged
    entity look changed.
    """
    if not cases:
        raise ValueError("An edit batch needs at least one case.")
    operations = {case.operation for case in cases}
    if len(operations) != 1:
        raise ValueError(
            f"A batch must share one operation so the encoding is constant, got {operations}."
        )
    deleting = [case for case in cases if not case.is_measurable]
    if deleting:
        # Rendering these would mix entity sets within one batch, which the gathered
        # decode cannot represent. They are filtered out at case construction and
        # counted, rather than silently dropped here.
        raise ValueError(
            f"{len(deleting)} case(s) remove an entity entirely, for example "
            f"{deleting[0].lost}. Filter with EditCase.is_measurable before batching."
        )

    before = builder.build([case.scene for case in cases], epoch=epoch)
    after = builder.build([case.after_scene for case in cases], epoch=epoch)
    slot_of = builder.slot_of
    entities = int(before.batch.entity_present.shape[1])

    edit = encode_edit(
        [case.operation_index for case in cases],
        [case.magnitude for case in cases],
        operations=len(OPERATIONS),
    )
    target_mask = torch.zeros((len(cases), entities), dtype=torch.float32)
    unchanged_mask = torch.zeros((len(cases), entities), dtype=torch.float32)
    for index, case in enumerate(cases):
        target_mask[index, slot_of[case.target]] = 1.0
        for name in case.still:
            if name in slot_of:
                unchanged_mask[index, slot_of[name]] = 1.0

    points = before.batch.scene_points.numpy()
    entity_points = before.batch.entity_points.numpy()
    ownership = np.full(points.shape[:2], -1, dtype=np.int64)
    occupancy = np.zeros(points.shape[:2], dtype=np.float32)
    entity_occupancy = np.zeros(
        entity_points.shape[:2] + (entity_points.shape[2],), dtype=np.float32
    )
    for index, case in enumerate(cases):
        organ = WholeOrganField(case.after_scene.parameters)
        labels = organ.ownership(points[index])
        for name, local in organ.slot_of.items():
            ownership[index][labels == local] = slot_of[name]
        occupancy[index] = (labels >= 0).astype(np.float32)
        for name in case.after_scene.entity_ids:
            slot = slot_of[name]
            probe = entity_points[index, slot]
            entity_occupancy[index, slot] = (organ.ownership(probe) == organ.slot_of[name]).astype(
                np.float32
            )

    return EditBatch(
        before=before,
        after=after,
        cases=tuple(cases),
        edit=edit,
        target_mask=target_mask,
        after_ownership=torch.from_numpy(ownership),
        after_occupancy=torch.from_numpy(occupancy),
        after_entity_occupancy=torch.from_numpy(entity_occupancy),
        unchanged_mask=unchanged_mask,
    )
