"""Step 10 Change 2 §8: rotation reported as an angle, against this corpus's rotation floor.

The training loss is chordal because the geodesic angle's gradient diverges at zero. The
**report** is the geodesic angle in degrees, because that is the quantity anyone can
interpret: 10 degrees is 10 degrees. Both facts live in one place so they cannot drift apart.

Everything here is measured on frames the model produced with ``use_predicted_frames=True``,
the only honest condition, and bucketed nowhere except by the thresholds the brief names.

The floor is passed in rather than recomputed, so a caller cannot accidentally score against
a different split's floor — they are not interchangeable: 22.65 degrees on ``test_seen``
against 35.95 on ``test_transform``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from experiments.step8.frame_metrics import rotation_error
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = ["DEGREES", "rotation_report"]

DEGREES = 180.0 / math.pi


@torch.no_grad()
def rotation_report(
    model: Any,
    scenes: Sequence[Any],
    builder: WholeOrganBatchBuilder,
    *,
    device: torch.device,
    batch_size: int = 8,
    floor_radians: float,
) -> dict[str, Any]:
    """Geodesic rotation error in degrees, with the distribution the brief asks for.

    ``below_floor`` is the share of individual entities whose error is under the floor, and
    ``floor_margin_deg`` is the floor minus the mean: positive means the model beat a lookup
    table keyed on identity alone. Both are needed — a model can sit on the floor on average
    while being better on most entities and much worse on a few.
    """
    model.eval()
    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    errors: list[np.ndarray] = []
    for whole in loader.epoch(0):
        batch = whole.batch.to(device)
        output = model(batch, use_predicted_frames=True)
        if output.frames is None:
            raise ValueError("this arm produced no frames and cannot be scored for rotation")
        present = batch.entity_present.bool()
        radians = rotation_error(output.frames, batch.entity_frames)[present]
        errors.append(radians.detach().cpu().numpy())
    joined = np.concatenate(errors) * DEGREES
    floor_deg = floor_radians * DEGREES
    return {
        "entities_scored": int(joined.size),
        "mean_deg": float(joined.mean()),
        "median_deg": float(np.median(joined)),
        "std_deg": float(joined.std(ddof=1)) if joined.size > 1 else 0.0,
        "min_deg": float(joined.min()),
        "max_deg": float(joined.max()),
        "within_5_deg": float((joined <= 5.0).mean()),
        "within_10_deg": float((joined <= 10.0).mean()),
        "within_20_deg": float((joined <= 20.0).mean()),
        "below_floor": float((joined < floor_deg).mean()),
        "floor_deg": floor_deg,
        "floor_margin_deg": floor_deg - float(joined.mean()),
        "beats_floor": bool(float(joined.mean()) < floor_deg),
    }
