"""Step 10 §27: the gate hierarchy metadata must pass before anything is trained or scored.

Parent-relative placement fails quietly. A wrong parent, a child composed before its parent,
or a rotation read transposed all still produce a frame of the right shape, the loss still
falls, and the only symptom is a placement number that is worse than it should be — which
is exactly the number Step 10 is trying to read. So the metadata is checked, and a failure
raises, before a run is allowed to start.

Six checks, each aimed at one way the pipeline can go wrong
-----------------------------------------------------------

``check_source_table``
    Every declared parent is an entity the generator actually builds that child from — an
    annulus from one of the two chambers it sits between, a vessel from its anchor or from
    the annulus it starts at. This compares the table against the construction code, not
    against a copy of itself, so an edit to the table is caught.

``check_slot_table``
    The table the model holds addresses the same entities as the declared hierarchy. The
    batch is indexed by the full ontology, not by the twenty whole-organ entities, and a
    table built in the wrong ordering is a shuffled table that raises nothing on its own.

``check_order``
    The composition order is a permutation in which every parent precedes its children.
    Composition reads a parent's already-composed frame, so a wrong order composes against
    a stale one.

``check_frames``
    Stored rotations are orthonormal and right-handed. This reads the raw stored vectors,
    because the model's own ``frame_rotation`` Gram-Schmidts them and would silently repair
    a corrupt basis.

``check_round_trip``
    Decomposing real frames into parent-relative targets and composing them back reproduces
    the frames. It runs twice: on the frames as stored, and on the same frames given a
    distinct real rotation per entity. The second pass is not optional on an
    identity-rotation corpus, where the transpose of the identity is the identity and a
    transposed-rotation defect is otherwise invisible.

``check_corpus_rotations``
    Every stored rotation is the one the generator's construction implies, compared against
    the scene's own parameters. Added for Change 2, and it is the only check that can see a
    rotation which is well formed but *wrong* — replaced by the identity, or transposed —
    because such a rotation passes every other check here. The Change 1 report named this
    gap; this closes it.

What this still cannot catch
----------------------------

A corpus whose stored generator parameters were themselves edited to match a corrupted
rotation. The comparison is against the parameters the scene carries, not against an
external record of what those parameters should have been; the corpus manifest's seed and
generator version are what pin that.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import numpy as np
import torch

from datasets.whole_organ.field import WholeOrganField
from datasets.whole_organ.hierarchy import HIERARCHIES, ROOT, parent_slots
from datasets.whole_organ.parameters import Variant, sample_parameters
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import frame_to_matrix

__all__ = [
    "PlacementIntegrityError",
    "check_corpus_rotations",
    "check_frames",
    "check_order",
    "check_round_trip",
    "check_slot_table",
    "check_source_table",
    "construction_parents",
    "validate_placement",
]

#: Loose enough for float32 frames, tight enough that any real defect is orders above it.
ROUND_TRIP_TOLERANCE = 1e-6
ORTHONORMAL_TOLERANCE = 1e-4


class PlacementIntegrityError(ValueError):
    """Hierarchy metadata that would train or score something other than what it names."""


@cache
def construction_parents() -> dict[str, frozenset[str]]:
    """Which entities the generator builds each constructed entity from.

    Read from a live :class:`WholeOrganField` for the normal arrangement, which is the
    arrangement the fixed parent table is defined on. An annulus is built between two
    cavities; a vessel is anchored to a cavity or trunk, and a great artery additionally
    starts at its own valve's annulus, which is recovered geometrically because the tube
    does not store it.
    """
    field = WholeOrganField(sample_parameters(np.random.default_rng(0), 0, Variant.NORMAL))
    allowed: dict[str, frozenset[str]] = {
        annulus.entity_id: frozenset({annulus.upper, annulus.lower}) for annulus in field.annuli
    }
    for tube in field.tubes:
        references = {tube.attached_to}
        for annulus in field.annuli:
            if _starts_on(tube.start, annulus.centre, annulus.axis, annulus.thickness):
                references.add(annulus.entity_id)
        allowed[tube.entity_id] = frozenset(references)
    return allowed


def _starts_on(start: Any, centre: Any, axis: Any, thickness: float) -> bool:
    """Whether a tube begins on an annulus's own axis, just off its face.

    Proximity alone is not enough: the inferior vena cava starts within two thicknesses of
    the tricuspid annulus without being built from it. A tube built from an annulus starts
    *on its axis*, which a neighbour does not.
    """
    offset = np.asarray(start - centre, dtype=np.float64)
    distance = float(np.linalg.norm(offset))
    if distance == 0.0 or distance > 2.0 * thickness:
        return False
    off_axis = float(np.linalg.norm(np.cross(offset / distance, axis)))
    return off_axis < 1e-6


def check_corpus_rotations(
    scenes: Sequence[Any], *, tolerance: float = 1e-6
) -> dict[str, Any]:
    """Every stored rotation is the one the generator's construction implies.

    This is the check the Change 1 report identified as missing and a rotated corpus needs.
    A rotation replaced by the identity, or stored transposed, is still a perfectly valid
    rotation: it is orthonormal, right-handed, and it composes and decomposes exactly, so no
    round trip and no well-formedness test can see it. Only a comparison against an
    independent record can, and the scene carries one — its generator parameters — from which
    the declared basis is reproducible exactly and without sampling.

    Raises:
        PlacementIntegrityError: if a scene carries no rotations at all, if an entity's
            rotation is missing, or if any stored basis differs from the generator's.

    """
    worst = 0.0
    worst_at = ""
    checked = 0
    for scene in scenes:
        if not scene.rotations:
            raise PlacementIntegrityError(
                f"{scene.scene_id} carries no rotations; this is not a rotated corpus"
            )
        declared = scene.field().entity_bases()
        for entity_id in scene.centroids:
            stored = scene.rotations.get(entity_id)
            if stored is None:
                raise PlacementIntegrityError(
                    f"{scene.scene_id}: {entity_id} has a frame but no rotation"
                )
            expected = np.asarray(declared[entity_id][:2], dtype=np.float64).reshape(-1)
            deviation = float(np.abs(np.asarray(stored, dtype=np.float64) - expected).max())
            checked += 1
            if deviation > worst:
                worst, worst_at = deviation, f"{scene.scene_id}/{entity_id}"
    if worst > tolerance:
        raise PlacementIntegrityError(
            f"a stored rotation is not the generator's: worst deviation {worst:.3e} at "
            f"{worst_at}; the frame would describe an orientation the geometry does not have"
        )
    return {"rotations_checked": checked, "worst_deviation": worst}


def check_source_table(hierarchy: str) -> None:
    """Every declared parent is one the generator actually builds that child from."""
    if hierarchy not in HIERARCHIES:
        raise PlacementIntegrityError(f"unknown hierarchy {hierarchy!r}")
    allowed = construction_parents()
    for child, parent in sorted(HIERARCHIES[hierarchy].items()):
        if child not in allowed:
            raise PlacementIntegrityError(
                f"{hierarchy}: {child!r} is not built from any other entity, so it cannot "
                f"have a construction parent, but the table gives it {parent!r}"
            )
        if parent not in allowed[child]:
            raise PlacementIntegrityError(
                f"{hierarchy}: {child!r} is parented to {parent!r}, but the generator builds "
                f"it from {sorted(allowed[child])}"
            )


def check_slot_table(
    table: Sequence[int], slot_of: Mapping[str, int], hierarchy: str
) -> None:
    """The slot table addresses the same entities as the declared hierarchy."""
    expected = parent_slots(hierarchy, slot_of, slots=len(table))
    if tuple(table) == expected:
        return
    name_of = {slot: entity_id for entity_id, slot in slot_of.items()}
    for slot, (have, want) in enumerate(zip(table, expected, strict=True)):
        if have != want:
            raise PlacementIntegrityError(
                f"slot {slot} ({name_of.get(slot, 'padding')}) is parented to slot {have} "
                f"({name_of.get(have, 'root') if have != ROOT else 'root'}), but the "
                f"{hierarchy} hierarchy says {want} "
                f"({name_of.get(want, 'root') if want != ROOT else 'root'})"
            )


def check_order(table: Sequence[int], order: Sequence[int]) -> None:
    """The order is a permutation of the slots with every parent before its children."""
    if sorted(order) != list(range(len(table))):
        raise PlacementIntegrityError(
            f"composition order is not a permutation of the {len(table)} slots"
        )
    position = {slot: index for index, slot in enumerate(order)}
    for slot, parent in enumerate(table):
        if parent != ROOT and position[parent] > position[slot]:
            raise PlacementIntegrityError(
                f"slot {slot} is composed before its parent, slot {parent}; it would be "
                "composed against a stale frame"
            )


def check_frames(frames: torch.Tensor, present: torch.Tensor) -> dict[str, float]:
    """Stored rotations are orthonormal and right-handed, and nothing is non-finite."""
    mask = present.bool()
    chosen = frames[mask].double()
    if not bool(torch.isfinite(chosen).all()):
        raise PlacementIntegrityError("a present entity has a non-finite frame")
    first, second = chosen[:, 6:9], chosen[:, 9:12]
    unit = max(
        float((first.norm(dim=-1) - 1.0).abs().max()),
        float((second.norm(dim=-1) - 1.0).abs().max()),
    )
    orthogonal = float((first * second).sum(dim=-1).abs().max())
    if unit > ORTHONORMAL_TOLERANCE or orthogonal > ORTHONORMAL_TOLERANCE:
        raise PlacementIntegrityError(
            f"a stored rotation basis is not orthonormal (unit deviation {unit:.2e}, "
            f"dot product {orthogonal:.2e}); the model would silently re-orthonormalise it"
        )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=chosen.dtype)
    return {
        "entities_checked": float(chosen.shape[0]),
        "max_rotation_deviation_from_identity": float(
            (chosen[:, 6:12] - identity).abs().max()
        ),
    }


def _probe_rotations(frames: torch.Tensor) -> torch.Tensor:
    """The same frames, each slot given its own real rotation.

    Distinct per slot on purpose: if every entity shared one rotation, every
    parent-relative rotation would be the identity again and the probe would be as blind
    as the stored frames.
    """
    out = frames.clone()
    slots = frames.shape[-2]
    axis = torch.tensor([0.36, 0.48, 0.80], dtype=frames.dtype)
    for slot in range(slots):
        angle = 0.35 + 0.23 * slot
        cos, sin = math.cos(angle), math.sin(angle)
        x, y, z = (float(value) for value in axis)
        rotation = torch.tensor(
            [
                [cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin],
                [y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin],
                [z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)],
            ],
            dtype=frames.dtype,
        )
        out[..., slot, 6:9] = rotation[0]
        out[..., slot, 9:12] = rotation[1]
    return out


def _round_trip(
    placement: HierarchicalPlacement, frames: torch.Tensor, present: torch.Tensor
) -> float:
    target = placement.to_local(frames, present)
    rebuilt = placement.to_global(target.local, target.parents)
    linear, translation = frame_to_matrix(rebuilt)
    true_linear, true_translation = frame_to_matrix(frames)
    mask = present.bool()
    return max(
        float((linear - true_linear)[mask].abs().max()),
        float((translation - true_translation)[mask].abs().max()),
    )


def check_round_trip(
    placement: HierarchicalPlacement, frames: torch.Tensor, present: torch.Tensor
) -> dict[str, float]:
    """Local targets compose back to the frames, with the stored and with real rotations."""
    frames = frames.double()
    stored = _round_trip(placement, frames, present)
    probe = _round_trip(placement, _probe_rotations(frames), present)
    for name, error in (("stored", stored), ("probe-rotated", probe)):
        if not error <= ROUND_TRIP_TOLERANCE:
            raise PlacementIntegrityError(
                f"parent-relative targets do not compose back to the {name} frames "
                f"(max error {error:.3e}); a rotation, the order or the convention is wrong"
            )
    return {"round_trip_error": stored, "probe_round_trip_error": probe}


def validate_placement(
    placement: HierarchicalPlacement | None,
    *,
    slot_of: Mapping[str, int],
    hierarchy: str,
    frames: torch.Tensor,
    present: torch.Tensor,
) -> dict[str, Any]:
    """Run every check that applies, raise on the first failure, report what passed.

    ``placement`` is ``None`` for the global target, which composes nothing; its frames and
    its declared hierarchy are still checked, because the depth analysis reads depth from
    that hierarchy for every cell.
    """
    report: dict[str, Any] = {"hierarchy": hierarchy}
    check_source_table(hierarchy)
    report["source_table"] = "matches generator construction"
    report.update(check_frames(frames, present))
    if placement is None:
        report["composition"] = "none: global target"
        return report
    if placement.hierarchy != hierarchy:
        raise PlacementIntegrityError(
            f"the model composes over {placement.hierarchy!r} but the run declares {hierarchy!r}"
        )
    table = placement.table
    check_slot_table(table, slot_of, hierarchy)
    check_order(table, placement.order)
    report.update(check_round_trip(placement, frames, present))
    report["parented_slots"] = sum(1 for parent in table if parent != ROOT)
    report["convention"] = placement.convention
    return report
