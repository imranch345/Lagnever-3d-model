"""Evaluation metrics for the first heart experiment.

Every metric here is computed the same way for both arms, from quantities both arms
produce: the composed scene field and the part-label logits. Nothing uses a structure
only one arm has, because a metric that did would decide the experiment by
construction.

**Occupancy-set Chamfer, defined precisely.** Both arms are evaluated on the same
query points ``Q`` (the standard sampling mixture: uniform, near-surface and
interior). Let ``P`` be the points the model marks occupied (sigmoid above 0.5) and
``G`` the points that are occupied in the ground truth. The metric is

    chamfer(P, G) = mean_{p in P} min_{g in G} ||p - g|| + mean_{g in G} min_{p in P} ||p - g||

If either set is empty the metric returns ``2 * sqrt(3)``, the diameter of the unit
cube, and the case is counted separately. This is a *point-set* Chamfer distance on
shared samples, not a surface Chamfer distance: no mesh is extracted, and the number
is comparable between arms but not with published surface metrics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from generation.neural.nn.model import ModelOutput
from generation.neural.nn.tensors import PrototypeBatch

__all__ = [
    "EMPTY_SET_PENALTY",
    "ADDRESSABLE_ENTITIES",
    "MetricResult",
    "part_control",
    "scene_geometry",
    "relationship_accuracy",
    "lod_consistency",
    "edit_drift",
]

EMPTY_SET_PENALTY = 2.0 * math.sqrt(3.0)
"""Returned by the Chamfer metric when either point set is empty."""

ADDRESSABLE_ENTITIES: tuple[str, ...] = (
    "heart.left_ventricle",
    "heart.right_ventricle",
    "heart.left_atrium",
    "heart.right_atrium",
    "heart.mitral_valve",
    "heart.aorta",
)
"""The entities the brief names for part-control evaluation."""


@dataclass(slots=True)
class MetricResult:
    """Scalar metrics plus any per-entity detail."""

    values: dict[str, float] = field(default_factory=dict)
    per_entity: dict[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def merge(self, other: MetricResult, prefix: str = "") -> None:
        """Merge another result in, optionally prefixing its keys."""
        for key, value in other.values.items():
            self.values[f"{prefix}{key}"] = value
        for key, value in other.per_entity.items():
            self.per_entity[f"{prefix}{key}"] = value
        self.notes = (*self.notes, *other.notes)


def _owned_mask(part_logits: torch.Tensor, slot: int) -> torch.Tensor:
    return part_logits.argmax(dim=-1) == slot


def part_control(
    output: ModelOutput,
    batch: PrototypeBatch,
    slot_names: Mapping[int, str],
    *,
    threshold: float = 0.5,
) -> MetricResult:
    """Measure whether the model can address the entity it was asked about.

    For every entity present in the scene, the predicted region is the set of query
    points whose part label argmax is that entity, and the target region is the set
    whose ground-truth owner is that entity. Success means intersection over union at
    or above ``threshold``.
    """
    predicted = output.part_logits.argmax(dim=-1)
    result = MetricResult()
    successes: list[float] = []
    addressable: list[float] = []
    for slot, name in slot_names.items():
        present = batch.entity_present[:, slot]
        if not bool(present.any()):
            continue
        target = batch.part_owner == slot
        prediction = predicted == slot
        intersection = (target & prediction).sum(dim=1).to(torch.float32)
        union = (target | prediction).sum(dim=1).to(torch.float32)
        valid = present & (target.sum(dim=1) > 0)
        if not bool(valid.any()):
            continue
        iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union))
        entity_iou = float(iou[valid].mean())
        entity_success = float((iou[valid] >= threshold).to(torch.float32).mean())
        result.per_entity[f"iou/{name}"] = entity_iou
        result.per_entity[f"success/{name}"] = entity_success
        successes.append(entity_success)
        if name in ADDRESSABLE_ENTITIES:
            addressable.append(entity_success)
    result.values["part_control_success"] = (
        float(sum(successes) / len(successes)) if successes else 0.0
    )
    result.values["part_control_success_addressable"] = (
        float(sum(addressable) / len(addressable)) if addressable else 0.0
    )
    result.values["part_iou_mean"] = (
        float(
            sum(v for k, v in result.per_entity.items() if k.startswith("iou/"))
            / max(1, len(successes))
        )
    )
    return result


def scene_geometry(output: ModelOutput, batch: PrototypeBatch) -> MetricResult:
    """Scene-level occupancy quality: intersection over union and Chamfer."""
    probability = torch.sigmoid(output.scene_logits)
    predicted = probability > 0.5
    target = batch.scene_occupancy > 0.5
    intersection = (predicted & target).sum(dim=1).to(torch.float32)
    union = (predicted | target).sum(dim=1).to(torch.float32)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))

    chamfers: list[float] = []
    empty = 0
    for index in range(batch.batch_size):
        points = batch.scene_points[index]
        p = points[predicted[index]]
        g = points[target[index]]
        if p.numel() == 0 or g.numel() == 0:
            chamfers.append(EMPTY_SET_PENALTY)
            empty += 1
            continue
        distances = torch.cdist(p, g)
        forward = distances.min(dim=1).values.mean()
        backward = distances.min(dim=0).values.mean()
        chamfers.append(float(forward + backward))
    return MetricResult(
        values={
            "scene_iou": float(iou.mean()),
            "occupancy_chamfer": float(sum(chamfers) / len(chamfers)),
            "chamfer_empty_sets": float(empty),
            "occupancy_accuracy": float(((probability > 0.5) == target).to(torch.float32).mean()),
        },
        notes=(
            "Chamfer is an occupancy-set distance on shared query points, not surface Chamfer.",
        ),
    )


def _predicted_centroids(
    output: ModelOutput, batch: PrototypeBatch, slots: Sequence[int], minimum_points: int = 3
) -> dict[int, torch.Tensor]:
    predicted = output.part_logits.argmax(dim=-1)
    centroids: dict[int, torch.Tensor] = {}
    for slot in slots:
        owned = predicted == slot
        counts = owned.sum(dim=1)
        if not bool((counts >= minimum_points).any()):
            continue
        weights = owned.to(torch.float32).unsqueeze(-1)
        summed = (batch.scene_points * weights).sum(dim=1)
        centroid = summed / counts.clamp_min(1).unsqueeze(-1).to(torch.float32)
        centroids[slot] = torch.where(
            (counts >= minimum_points).unsqueeze(-1),
            centroid,
            torch.full_like(centroid, float("nan")),
        )
    return centroids


def relationship_accuracy(
    output: ModelOutput,
    batch: PrototypeBatch,
    relations: Sequence[tuple[str, str, str]],
    slot_of: Mapping[str, int],
    *,
    adjacency_tolerance: float = 0.12,
) -> MetricResult:
    """Held-out relationship accuracy, measured from generated geometry.

    Never trained on. Directional relations are checked on the centroids of the
    predicted part regions; adjacency is checked on the minimum distance between two
    predicted regions.

    Two definitions are reported, because they differ sharply between arms and only
    reporting one would flatter whichever arm places fewer entities:

    * ``relationship_accuracy`` counts only relations whose two entities the model
      actually placed, so a model that locates two structures and gets their relation
      right scores well even if it cannot find anything else.
    * ``relationship_accuracy_strict`` counts every relation whose entities are present
      in the scene, so failing to place an entity counts as failing its relations.

    The strict form is the one used for the pre-registered verdict: a relation that
    cannot be recovered from the generated geometry has not been recovered.
    """
    predicted = output.part_logits.argmax(dim=-1)
    axis_sign = {
        "left_of": (0, 1.0),
        "right_of": (0, -1.0),
        "superior_to": (1, 1.0),
        "inferior_to": (1, -1.0),
        "anterior_to": (2, 1.0),
        "posterior_to": (2, -1.0),
    }
    slots = sorted(
        {
            slot_of[name]
            for triple in relations
            for name in (triple[0], triple[2])
            if name in slot_of
        }
    )
    centroids = _predicted_centroids(output, batch, slots)

    correct = 0
    total = 0
    skipped = 0
    possible = 0
    per_relation: dict[str, float] = {}
    for subject, relation, obj in relations:
        if subject not in slot_of or obj not in slot_of:
            skipped += 1
            continue
        first, second = slot_of[subject], slot_of[obj]
        first_present = bool(batch.entity_present[:, first].any())
        second_present = bool(batch.entity_present[:, second].any())
        if not first_present or not second_present:
            skipped += 1
            continue
        hits = 0
        counted = 0
        possible += int(
            (batch.entity_present[:, first] & batch.entity_present[:, second]).sum()
        )
        for index in range(batch.batch_size):
            if not (batch.entity_present[index, first] and batch.entity_present[index, second]):
                continue
            if relation in axis_sign:
                if first not in centroids or second not in centroids:
                    continue
                a = centroids[first][index]
                b = centroids[second][index]
                if bool(torch.isnan(a).any() or torch.isnan(b).any()):
                    continue
                axis, sign = axis_sign[relation]
                counted += 1
                hits += int(sign * float(a[axis] - b[axis]) > 0.0)
            elif relation == "adjacent_to":
                mask_a = predicted[index] == first
                mask_b = predicted[index] == second
                if int(mask_a.sum()) < 3 or int(mask_b.sum()) < 3:
                    continue
                distance = torch.cdist(
                    batch.scene_points[index][mask_a], batch.scene_points[index][mask_b]
                ).min()
                counted += 1
                hits += int(float(distance) <= adjacency_tolerance)
            else:
                continue
        if counted == 0:
            skipped += 1
            continue
        per_relation[f"{subject}|{relation}|{obj}"] = hits / counted
        correct += hits
        total += counted
    return MetricResult(
        values={
            "relationship_accuracy": float(correct / total) if total else 0.0,
            "relationship_accuracy_strict": float(correct / possible) if possible else 0.0,
            "relationship_checked": float(total),
            "relationship_possible": float(possible),
            "relationship_coverage": float(total / possible) if possible else 0.0,
            "relationship_skipped": float(skipped),
        },
        per_entity=per_relation,
        notes=("Relationship objectives are held out of training in this experiment.",),
    )


def lod_consistency(
    model: torch.nn.Module,
    batch: PrototypeBatch,
    *,
    prefixes: Sequence[int],
    slot_names: Mapping[int, str],
) -> MetricResult:
    """Measure whether more detail means more detail, not a different structure.

    Three quantities across token prefixes: whether the part attribution of each
    entity stays the same, how far the predicted region's centroid moves, and whether
    scene occupancy agreement with the ground truth improves monotonically.
    """
    with torch.no_grad():
        outputs = [model(batch, token_prefix=prefix) for prefix in prefixes]
    ious: list[float] = []
    agreements: list[float] = []
    shifts: list[float] = []
    reference = outputs[-1]
    reference_centroids = _predicted_centroids(reference, batch, list(slot_names))
    for output in outputs:
        ious.append(scene_geometry(output, batch).values["scene_iou"])
        agreement = (
            output.part_logits.argmax(dim=-1) == reference.part_logits.argmax(dim=-1)
        ).to(torch.float32).mean()
        agreements.append(float(agreement))
        centroids = _predicted_centroids(output, batch, list(slot_names))
        distances = [
            float(
                torch.nanmean(
                    torch.linalg.norm(centroids[slot] - reference_centroids[slot], dim=-1)
                )
            )
            for slot in centroids
            if slot in reference_centroids
        ]
        shifts.append(float(sum(distances) / len(distances)) if distances else 0.0)
    monotone = all(later >= earlier - 1e-3 for earlier, later in zip(ious, ious[1:], strict=False))
    return MetricResult(
        values={
            "lod_iou_monotone": float(monotone),
            "lod_part_agreement_mean": float(sum(agreements) / len(agreements)),
            "lod_centroid_shift_mean": float(sum(shifts) / len(shifts)),
            "lod_iou_coarsest": ious[0],
            "lod_iou_finest": ious[-1],
        },
        notes=("Identity is compared by part attribution, which both arms produce.",),
    )


def edit_drift(
    model: torch.nn.Module,
    batch: PrototypeBatch,
    *,
    hidden_slot: int,
) -> MetricResult:
    """Change in geometry at untouched entities after a presentation edit.

    The edit hides one entity by removing it from the request. Drift is the mean
    absolute change in occupancy probability at query points owned by entities the
    edit did not name. Zero is the target, and for a state-invariant geometry path it
    is exactly zero.
    """
    import copy

    with torch.no_grad():
        before = model(batch)
        edited = copy.copy(batch)
        edited.text_features = batch.text_features.clone()
        edited.text_features[:, hidden_slot] = 0.0
        edited.entity_presence_target = batch.entity_presence_target.clone()
        if hidden_slot < edited.entity_presence_target.shape[1]:
            edited.entity_presence_target[:, hidden_slot] = 0.0
        after = model(edited)

    untouched = (batch.part_owner >= 0) & (batch.part_owner != hidden_slot)
    probability_before = torch.sigmoid(before.scene_logits)
    probability_after = torch.sigmoid(after.scene_logits)
    difference = (probability_after - probability_before).abs()
    drift = float((difference * untouched).sum() / untouched.sum().clamp_min(1))
    touched = batch.part_owner == hidden_slot
    touched_change = (
        float((difference * touched).sum() / touched.sum().clamp_min(1))
        if bool(touched.any())
        else 0.0
    )
    token_identical = bool(torch.equal(before.geometry_tokens, after.geometry_tokens))
    return MetricResult(
        values={
            "untouched_drift": drift,
            "touched_change": touched_change,
            "geometry_tokens_identical": float(token_identical),
        },
        notes=("Drift is measured in occupancy probability, so both arms are comparable.",),
    )
