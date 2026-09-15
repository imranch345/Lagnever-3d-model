"""Frame prediction metrics for Step 8.

The Step 7 evaluation supplied each entity's canonical frame and asked only what shape to
put there. Step 8 asks the model to place structures itself, so placement needs metrics of
its own rather than being visible only as a drop in ownership accuracy.

A frame is the 12-number contract the decoder already uses: three translation components,
three log-scale components and a 6D rotation. Each part is scored on its own terms,
because they are not commensurable and a single number would hide which one is wrong:

``position_error``
    Euclidean distance between predicted and true translation, in scene units. The scene
    occupies roughly [-1, 1] on each axis, so 0.1 is a tenth of a half-width.
``scale_error``
    Mean absolute difference in log-scale, which is a ratio error: 0.69 is a factor of two
    in either direction, and the measure is symmetric in a way a raw ratio is not.
``rotation_error``
    Geodesic angle in radians between the predicted and true rotation matrices, after
    Gram-Schmidt. The natural distance on rotations; a raw difference of 6D components is
    not one.
``composite_frame_error``
    ``position + 0.5 * scale + 0.25 * rotation``, documented here and fixed before any
    Step 8 run. The weights make the three parts comparable in magnitude on this corpus;
    they are not a claim about relative importance, and every component is also reported
    separately so a reader can reweigh them.

The number that actually answers the question is not any of these. A model can place every
entity a little wrong and still get the anatomy right, or place them all near the average
and score moderately while getting every relation wrong. Whether the model puts the correct
entity in the correct *relational* location is measured by spatial relation accuracy under
inferred placement, against the relation-blind floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from generation.neural.nn.geometry import frame_rotation

__all__ = [
    "COMPOSITE_WEIGHTS",
    "position_error",
    "scale_error",
    "rotation_error",
    "frame_errors",
    "frame_errors_by_group",
]

#: Fixed before any Step 8 run. See the module docstring for why these values.
COMPOSITE_WEIGHTS: Mapping[str, float] = {"position": 1.0, "scale": 0.5, "rotation": 0.25}


def position_error(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-entity Euclidean translation error, shaped like the leading dimensions."""
    error: torch.Tensor = torch.linalg.norm(predicted[..., 0:3] - truth[..., 0:3], dim=-1)
    return error


def scale_error(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-entity mean absolute log-scale error, which is a symmetric ratio error."""
    return (predicted[..., 3:6] - truth[..., 3:6]).abs().mean(dim=-1)


def rotation_error(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-entity geodesic angle in radians between two rotations."""
    first = frame_rotation(predicted)
    second = frame_rotation(truth)
    relative = torch.matmul(first, second.transpose(-1, -2))
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.arccos(cosine)


def frame_errors(
    predicted: torch.Tensor, truth: torch.Tensor, present: torch.Tensor
) -> dict[str, float]:
    """Mean frame errors over the entities actually in the scene.

    Padded slots carry no meaningful frame, so including them would dilute every number
    toward whatever the padding happens to hold.
    """
    mask = present.bool()
    if not bool(mask.any()):
        return {
            "position_error": float("nan"),
            "scale_error": float("nan"),
            "rotation_error": float("nan"),
            "composite_frame_error": float("nan"),
            "frame_entities": 0.0,
        }
    position = position_error(predicted, truth)[mask]
    scale = scale_error(predicted, truth)[mask]
    rotation = rotation_error(predicted, truth)[mask]
    composite = (
        COMPOSITE_WEIGHTS["position"] * position
        + COMPOSITE_WEIGHTS["scale"] * scale
        + COMPOSITE_WEIGHTS["rotation"] * rotation
    )
    return {
        "position_error": float(position.mean()),
        "scale_error": float(scale.mean()),
        "rotation_error": float(rotation.mean()),
        "composite_frame_error": float(composite.mean()),
        "frame_entities": float(int(mask.sum())),
    }


def frame_errors_by_group(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    present: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    """Frame errors broken down by entity group, flattened as ``<group>_<metric>``.

    Reported because a mean over twenty structures hides the case that matters: a model
    can place four large chambers well and every thin structure badly, and the pooled
    number looks respectable.
    """
    out: dict[str, float] = {}
    for name, slots in groups.items():
        if not slots:
            continue
        index = torch.tensor(list(slots), dtype=torch.long, device=present.device)
        sub_present = torch.zeros_like(present, dtype=torch.bool)
        sub_present[..., index] = present[..., index].bool()
        values = frame_errors(predicted, truth, sub_present)
        for key, value in values.items():
            if key == "frame_entities":
                continue
            out[f"{name}_{key}"] = value
    return out


def summarise(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average a sequence of per-batch frame reports, skipping undefined entries."""
    keys = {key for entry in values for key in entry}
    out: dict[str, float] = {}
    for key in sorted(keys):
        finite = [
            float(entry[key]) for entry in values if key in entry and np.isfinite(float(entry[key]))
        ]
        out[key] = float(np.mean(finite)) if finite else float("nan")
    return out
