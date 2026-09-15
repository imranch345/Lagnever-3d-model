"""Step 8 evaluation: everything reported under an explicitly named placement condition.

Step 7's most consequential defect was an evaluation that supplied ground-truth entity
frames, which hands the model the arrangement a relationship graph would otherwise have
to supply. Every function here takes ``use_predicted_frames`` and every result records it,
because a relational number without its placement condition is not interpretable.

The headline Step 8 condition is **inferred**. The supplied condition is kept and reported
as an oracle upper bound, never as a result in its own right.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import torch

from datasets.whole_organ.corpus import LOD_ENTITIES, WholeOrganScene
from experiments.step7.metrics import (
    entity_ownership,
    placement_error,
    predicted_centroids,
    relation_accuracy,
)
from experiments.step8.frame_metrics import frame_errors, frame_errors_by_group, summarise
from generation.neural.nn.metrics import scene_geometry
from generation.neural.nn.nested_lod import nested_lod_metrics
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE
from training.whole_organ import WholeOrganLoader

__all__ = ["ENTITY_GROUPS", "evaluate_step8", "entity_group_slots"]

#: Entity groups the frame breakdown is reported over. Thin structures are pooled
#: separately from large ones because a mean over twenty entities hides the case where a
#: model places four big chambers well and every small structure badly.
ENTITY_GROUPS: Mapping[str, tuple[str, ...]] = {
    "chambers": (
        "heart.left_ventricle",
        "heart.right_ventricle",
        "heart.left_atrium",
        "heart.right_atrium",
    ),
    "valves": (
        "heart.mitral_valve",
        "heart.tricuspid_valve",
        "heart.aortic_valve",
        "heart.pulmonary_valve",
    ),
    "vessels": (
        "heart.aorta",
        "heart.pulmonary_trunk",
        "heart.pulmonary_arteries",
        "heart.pulmonary_veins",
        "heart.superior_vena_cava",
        "heart.inferior_vena_cava",
    ),
    "septa": ("heart.interventricular_septum", "heart.interatrial_septum"),
    "walls": (
        "heart.myocardium",
        "heart.endocardium",
        "heart.epicardium",
        "heart.pericardium",
    ),
}


def entity_group_slots(slot_of: Mapping[str, int]) -> dict[str, list[int]]:
    """Codebook slots for each reported entity group."""
    return {
        name: [slot_of[entity] for entity in entities if entity in slot_of]
        for name, entities in ENTITY_GROUPS.items()
    }


@torch.no_grad()
def evaluate_step8(
    model: torch.nn.Module,
    scenes: Sequence[WholeOrganScene],
    builder: WholeOrganBatchBuilder,
    slot_of: Mapping[str, int],
    *,
    device: torch.device,
    batch_size: int = 8,
    use_predicted_frames: bool = True,
) -> dict[str, float]:
    """Evaluate one split under one placement condition.

    Covers the whole split. Step 7 capped the final evaluation at a batch count, which
    walked the level-of-detail buckets in order and stopped partway, biasing every number
    toward whichever levels came first.
    """
    if not scenes:
        raise ValueError("Nothing to evaluate.")
    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    groups = entity_group_slots(slot_of)

    totals: dict[str, list[float]] = {}
    frame_reports: list[Mapping[str, float]] = []
    scenes_seen = 0

    for whole in loader.epoch(0):
        batch = whole.batch.to(device)
        output = model(batch, use_predicted_frames=use_predicted_frames)
        slots = [int(slot) for slot in batch.structure.visible_slots]

        values = dict(scene_geometry(output, batch).values)
        values.update(entity_ownership(output.part_logits, batch.part_owner, slots))
        centroids = predicted_centroids(output.part_logits, batch.scene_points, slots)
        values.update(relation_accuracy(centroids, whole.scenes, slot_of))
        values["placement_error"] = placement_error(centroids, whole.scenes, slot_of)

        if output.frames is not None:
            report = frame_errors(output.frames, batch.entity_frames, batch.entity_present)
            report = {
                **report,
                **frame_errors_by_group(
                    output.frames, batch.entity_frames, batch.entity_present, groups
                ),
            }
            frame_reports.append(report)

        # Level of detail: all three prefixes decoded on the same batch, so the
        # containment relation between them is measurable rather than inferred across
        # separate passes.
        decodes: dict[int, torch.Tensor] = {}
        targets: dict[int, torch.Tensor] = {}
        ownership: dict[int, torch.Tensor] = {}
        for level in (1, 2, 3):
            prefix = LOD_TOKEN_SCHEDULE[min(level, len(LOD_TOKEN_SCHEDULE) - 1)]
            level_output = model(
                batch, token_prefix=prefix, use_predicted_frames=use_predicted_frames
            )
            decodes[level] = level_output.scene_logits
            targets[level] = whole.lod_targets[level].to(device)
            predicted_owner = level_output.part_logits.argmax(dim=-1)
            background = int(level_output.part_logits.shape[-1]) - 1
            ownership[level] = torch.where(
                predicted_owner == background,
                torch.full_like(predicted_owner, -1),
                predicted_owner,
            )
        values.update(nested_lod_metrics(decodes, targets, ownership=ownership))

        scenes_seen += len(whole.scenes)
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value))

    summary: dict[str, float] = {}
    for key, collected in totals.items():
        finite = [value for value in collected if math.isfinite(value)]
        summary[key] = statistics.fmean(finite) if finite else float("nan")
    summary.update(summarise(frame_reports))
    summary["eval_scenes"] = float(scenes_seen)
    summary["placement_supplied"] = 0.0 if use_predicted_frames else 1.0
    return summary


@torch.no_grad()
def evaluate_both_conditions(
    model: torch.nn.Module,
    scenes: Sequence[WholeOrganScene],
    builder: WholeOrganBatchBuilder,
    slot_of: Mapping[str, int],
    *,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Both placement conditions from the same weights.

    The supplied condition is an oracle upper bound, not a result. It says how well the
    model would do if placement were solved, which bounds what better placement could buy.
    """
    return {
        "inferred": evaluate_step8(
            model,
            scenes,
            builder,
            slot_of,
            device=device,
            batch_size=batch_size,
            use_predicted_frames=True,
        ),
        "supplied": evaluate_step8(
            model,
            scenes,
            builder,
            slot_of,
            device=device,
            batch_size=batch_size,
            use_predicted_frames=False,
        ),
    }


def lod_entity_counts() -> dict[int, int]:
    """How many entities each level exposes, for reading the level tables."""
    return {level: len(LOD_ENTITIES[level]) for level in (1, 2, 3)}


def relation_gap(accuracy: float, blind_floor: float) -> float:
    """Share of the available headroom above the relation-blind floor that was taken.

    Negative when a model scores below the floor, which means a fixed arrangement-blind
    guess would have done better. Step 7 found every arm in that position under inferred
    placement, so the sign of this number is the first thing to check.
    """
    headroom = 1.0 - blind_floor
    if headroom <= 1e-9:
        return float("nan")
    return float((accuracy - blind_floor) / headroom)


def aggregate_seeds(
    runs: Sequence[Mapping[str, float]], metrics: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Mean, spread and count per metric, skipping undefined values."""
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [
            float(run[metric])
            for run in runs
            if metric in run and math.isfinite(float(run[metric]))
        ]
        out[metric] = {
            "mean": statistics.fmean(values) if values else float("nan"),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "n": float(len(values)),
            "defined_n": float(len(values)),
        }
    return out
