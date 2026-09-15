"""Step 7 metrics, defined in ``docs/step7/01_experimental_plan.md`` before any run.

Every metric is computed identically for every arm, from quantities every arm produces:
the composed occupancy field and the part-label logits. Nothing uses a structure that
only one arm has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.parameters import Variant
from datasets.whole_organ.relations import measure_relations
from generation.neural.nn.metrics import scene_geometry
from generation.neural.nn.whole_organ import WholeOrganBatch, WholeOrganBatchBuilder
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE

__all__ = [
    "predicted_centroids",
    "entity_ownership",
    "relation_accuracy",
    "evaluate_whole_organ",
    "counterfactual_analysis",
]

_AXIS_OF = {
    "left_of": (0, 1.0),
    "right_of": (0, -1.0),
    "superior_to": (1, 1.0),
    "inferior_to": (1, -1.0),
    "anterior_to": (2, 1.0),
    "posterior_to": (2, -1.0),
}


def predicted_centroids(
    part_logits: torch.Tensor, points: torch.Tensor, slots: Sequence[int], minimum: int = 3
) -> dict[int, torch.Tensor]:
    """Centroid of each entity's predicted ownership region, or NaN when unplaceable."""
    predicted = part_logits.argmax(dim=-1)
    out: dict[int, torch.Tensor] = {}
    for slot in slots:
        owned = predicted == slot
        counts = owned.sum(dim=1)
        weights = owned.to(points.dtype).unsqueeze(-1)
        centroid = (points * weights).sum(dim=1) / counts.clamp_min(1).unsqueeze(-1).to(
            points.dtype
        )
        out[slot] = torch.where(
            (counts >= minimum).unsqueeze(-1), centroid, torch.full_like(centroid, float("nan"))
        )
    return out


def entity_ownership(
    part_logits: torch.Tensor, target: torch.Tensor, slots: Sequence[int], *, threshold: float = 0.5
) -> dict[str, float]:
    """Per-entity ownership intersection over union, and the control-success rate."""
    predicted = part_logits.argmax(dim=-1)
    ious: list[float] = []
    successes: list[float] = []
    for slot in slots:
        truth = target == slot
        guess = predicted == slot
        valid = truth.sum(dim=1) > 0
        if not bool(valid.any()):
            continue
        intersection = (truth & guess).sum(dim=1).to(torch.float32)
        union = (truth | guess).sum(dim=1).to(torch.float32)
        iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union))
        ious.append(float(iou[valid].mean()))
        successes.append(float((iou[valid] >= threshold).to(torch.float32).mean()))
    return {
        "entity_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "part_control_success": float(np.mean(successes)) if successes else 0.0,
    }


def relation_accuracy(
    centroids: Mapping[int, torch.Tensor],
    scenes: Sequence[WholeOrganScene],
    slot_of: Mapping[str, int],
) -> dict[str, float]:
    """Share of each scene's **measured** spatial relations reproduced by the prediction.

    Never trained on. Uses the relations measured from that scene's own organ, so a
    mirrored scene is scored against mirrored relations.
    """
    correct = 0
    total = 0
    skipped = 0
    for index, scene in enumerate(scenes):
        for edge in scene.edges:
            if edge.relation not in _AXIS_OF:
                continue
            first = slot_of.get(edge.subject)
            second = slot_of.get(edge.object)
            if first is None or second is None or first not in centroids or second not in centroids:
                skipped += 1
                continue
            a = centroids[first][index]
            b = centroids[second][index]
            if bool(torch.isnan(a).any() or torch.isnan(b).any()):
                skipped += 1
                continue
            axis, sign = _AXIS_OF[edge.relation]
            total += 1
            correct += int(sign * float(a[axis] - b[axis]) > 0.0)
    possible = total + skipped
    return {
        "spatial_relation_accuracy": float(correct / total) if total else 0.0,
        "spatial_relation_accuracy_strict": float(correct / possible) if possible else 0.0,
        "spatial_relation_coverage": float(total / possible) if possible else 0.0,
    }


def placement_error(
    centroids: Mapping[int, torch.Tensor],
    scenes: Sequence[WholeOrganScene],
    slot_of: Mapping[str, int],
) -> float:
    """Mean distance between predicted and measured entity centroids."""
    errors: list[float] = []
    for index, scene in enumerate(scenes):
        for name, centroid in scene.centroids.items():
            slot = slot_of[name]
            if slot not in centroids:
                continue
            predicted = centroids[slot][index]
            if bool(torch.isnan(predicted).any()):
                continue
            truth = torch.tensor(centroid, dtype=predicted.dtype, device=predicted.device)
            errors.append(float(torch.linalg.norm(predicted - truth)))
    return float(np.mean(errors)) if errors else float("nan")


@torch.no_grad()
def evaluate_whole_organ(
    model: torch.nn.Module,
    whole: WholeOrganBatch,
    slot_of: Mapping[str, int],
    ids: Sequence[str],
    *,
    device: torch.device,
    full: bool = False,
    builder: WholeOrganBatchBuilder | None = None,
    use_predicted_frames: bool = False,
) -> dict[str, float]:
    """Evaluate one batch on every Step 7 category.

    ``use_predicted_frames`` selects the condition. With it false the scene's own entity
    frames are supplied, so the model is asked only what shape to put at a given place.
    With it true the model must infer placement from structure, relations and text, which
    is the condition in which a relationship graph could earn its parameters. Both are
    reported; see amendment A8.
    """
    batch = whole.batch.to(device)
    output = model(batch, use_predicted_frames=use_predicted_frames)
    slots = [int(slot) for slot in batch.structure.visible_slots]

    values = dict(scene_geometry(output, batch).values)
    values.update(entity_ownership(output.part_logits, batch.part_owner, slots))
    centroids = predicted_centroids(output.part_logits, batch.scene_points, slots)
    values.update(relation_accuracy(centroids, whole.scenes, slot_of))
    values["placement_error"] = placement_error(centroids, whole.scenes, slot_of)

    if full:
        values.update(
            _level_of_detail(model, whole, device, use_predicted_frames=use_predicted_frames)
        )
        if builder is not None:
            values.update(
                counterfactual_analysis(
                    model,
                    whole,
                    builder,
                    slot_of,
                    device=device,
                    use_predicted_frames=use_predicted_frames,
                )
            )
    return values


@torch.no_grad()
def _level_of_detail(
    model: torch.nn.Module,
    whole: WholeOrganBatch,
    device: torch.device,
    *,
    use_predicted_frames: bool = False,
) -> dict[str, float]:
    """Detail gain by level, scored against each level's own target."""
    batch = whole.batch.to(device)
    values: dict[str, float] = {}
    ious: list[float] = []
    for level in (1, 2, 3):
        prefix = LOD_TOKEN_SCHEDULE[min(level, len(LOD_TOKEN_SCHEDULE) - 1)]
        output = model(batch, token_prefix=prefix, use_predicted_frames=use_predicted_frames)
        target = whole.lod_targets[level].to(device) > 0.5
        predicted = torch.sigmoid(output.scene_logits) > 0.5
        intersection = (predicted & target).sum(dim=1).to(torch.float32)
        union = (predicted | target).sum(dim=1).to(torch.float32)
        iou = float(
            torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union)).mean()
        )
        values[f"lod{level}_iou"] = iou
        ious.append(iou)
    values["lod_detail_gain"] = ious[-1] - ious[0]
    coarse = torch.sigmoid(
        model(
            batch,
            token_prefix=LOD_TOKEN_SCHEDULE[1],
            use_predicted_frames=use_predicted_frames,
        ).scene_logits
    )
    fine = torch.sigmoid(
        model(
            batch,
            token_prefix=LOD_TOKEN_SCHEDULE[-1],
            use_predicted_frames=use_predicted_frames,
        ).scene_logits
    )
    values["lod_token_delta"] = float((fine - coarse).abs().mean())
    return values


@torch.no_grad()
def counterfactual_analysis(
    model: torch.nn.Module,
    whole: WholeOrganBatch,
    builder: WholeOrganBatchBuilder,
    slot_of: Mapping[str, int],
    *,
    device: torch.device,
    target_variant: Variant | None = None,
    invariant_tolerance: float = 0.02,
    use_predicted_frames: bool = False,
) -> dict[str, float]:
    """Swap each scene's relation graph for its counterfactual variant's and measure.

    Entities, presence and text features are held fixed; only the measured relations
    change. Sensitivity says whether the output moved at all. **Correctness** says
    whether it moved toward the counterfactual's true arrangement, which is the number
    that matters: a model can be sensitive without being right.
    """
    from dataclasses import replace as dataclass_replace

    from datasets.whole_organ.corpus import WholeOrganScene as Scene
    from datasets.whole_organ.field import WholeOrganField

    counterfactual_scenes: list[Scene] = []
    for scene in whole.scenes:
        # Flip the mirror bit and leave every other coordinate alone, so the
        # counterfactual differs from the scene in exactly one respect. Step 7 swapped a
        # whole variant label, which changed several things at once.
        flipped = dataclass_replace(
            scene.parameters.arrangement, mirror=not scene.parameters.arrangement.mirror
        )
        parameters = dataclass_replace(scene.parameters, arrangement=flipped)
        organ = WholeOrganField(parameters)
        statistics = organ.entity_statistics(np.random.default_rng(scene.seed + 101))
        edges = measure_relations(organ, statistics, np.random.default_rng(scene.seed + 103))
        counterfactual_scenes.append(
            dataclass_replace(
                scene,
                parameters=parameters,
                centroids={k: v["centroid"].tolist() for k, v in statistics.items()},
                extents={k: v["extent"].tolist() for k, v in statistics.items()},
                counts={k: int(v["count"][0]) for k, v in statistics.items()},
                edges=edges,
            )
        )

    swapped = builder.build(
        list(whole.scenes), edge_override=[scene.edges for scene in counterfactual_scenes]
    )
    base = whole.batch.to(device)
    alternative = swapped.batch.to(device)
    slots = [int(slot) for slot in base.structure.visible_slots]

    original = predicted_centroids(
        model(base, use_predicted_frames=use_predicted_frames).part_logits,
        base.scene_points,
        slots,
    )
    perturbed = predicted_centroids(
        model(alternative, use_predicted_frames=use_predicted_frames).part_logits,
        alternative.scene_points,
        slots,
    )

    displacements: list[float] = []
    correct: list[float] = []
    invariant: list[float] = []
    for index, (scene, counterfactual) in enumerate(
        zip(whole.scenes, counterfactual_scenes, strict=True)
    ):
        for name, slot in slot_of.items():
            if slot not in original or name not in scene.centroids:
                continue
            before = original[slot][index]
            after = perturbed[slot][index]
            if bool(torch.isnan(before).any() or torch.isnan(after).any()):
                continue
            truth_original = torch.tensor(scene.centroids[name], dtype=after.dtype, device=device)
            truth_counterfactual = torch.tensor(
                counterfactual.centroids[name], dtype=after.dtype, device=device
            )
            moved = float(torch.linalg.norm(after - before))
            separation = float(torch.linalg.norm(truth_counterfactual - truth_original))
            if separation <= invariant_tolerance:
                invariant.append(moved)
                continue
            displacements.append(moved)
            to_counterfactual = float(torch.linalg.norm(after - truth_counterfactual))
            to_original = float(torch.linalg.norm(after - truth_original))
            correct.append(float(to_counterfactual < to_original))
    return {
        "counterfactual_relation_sensitivity": float(np.mean(displacements))
        if displacements
        else 0.0,
        "counterfactual_correctness": float(np.mean(correct)) if correct else 0.0,
        "invariant_preservation": float(np.mean(invariant)) if invariant else 0.0,
        "counterfactual_entities": float(len(displacements)),
    }
