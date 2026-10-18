"""Resize one structure of a Level 2b thorax by moving its code (M3).

A request is a change of linear size by ``factor``, as in Level 2a. The target volume is
therefore ``factor``³ times the current one.

**How.** A structure's soft volume, the sum of its softmax probability over a grid around it,
is differentiable in the code. The code is optimised towards the target, with a prior keeping
it near where it started:

    loss = (V(z) / V_target − 1)² + prior · mean((z − z₀)²)

With a part field, only the structure's own part of the code moves; the other parts keep
theirs. Neighbours change only as much as the field couples them, which is how the training
thoraxes say they move together. No point can belong to two structures, whatever the code.

**Checks.** The result is decoded on the full grid and judged by ``fields.generate``'s checks:
- volume within the training range;
- stray pieces no worse than the training decodes;
- sides;
- closed meshes;
- one label per point.

The largest whole-percent change that passes is found by bisection, as in Level 2a, and the
volumes reached are reported.

    python -m fields.resize --checkpoint runs/gpu/m2-h512-focus-prior/checkpoint.pt
        --data datasets/processed/level2b --report runs/gpu/m2-h512-focus-prior/samples.json
        --subject s0029 --entity heart.left_ventricle --percent 15
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.ontology import AnatomyOntology, load_ontology
from fields.evaluate import Grid, load_checkpoint
from fields.generate import decode_thorax, judge
from fields.model import Field, pick_device

__all__ = ["ResizeResult", "largest_change", "resize_code", "soft_volume"]


def soft_volume(
    field: Field,
    code: torch.Tensor,
    points: torch.Tensor,
    index: int,
    voxel_ml: float,
    chunk: int = 32768,
    weight: float | None = None,
) -> float:
    """The structure's soft volume in mL; with ``weight``, also back-propagates weight · V."""
    total = 0.0
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk].unsqueeze(0)
        if weight is None:
            with torch.no_grad():
                share = torch.softmax(field(part, code), dim=-1)[0, :, index].sum()
        else:
            share = torch.softmax(field(part, code), dim=-1)[0, :, index].sum()
            (weight * share * voxel_ml).backward()
        total += float(share.detach()) * voxel_ml
    return total


def _box_points(
    labels: np.ndarray, grid: Grid, index: int, margin_mm: float, spacing_mm: float
) -> np.ndarray:
    """Grid points at ``spacing_mm`` in the structure's bounding box plus ``margin_mm``."""
    where = np.argwhere(labels == index)
    if len(where) == 0:
        raise ValueError("the structure is absent from the starting thorax")
    low = grid.low + where.min(axis=0) * grid.spacing - margin_mm
    high = grid.low + where.max(axis=0) * grid.spacing + margin_mm
    return Grid(low, high, spacing_mm).points()


def resize_code(
    field: Field,
    start: torch.Tensor,
    index: int,
    factor: float,
    grid: Grid,
    labels: np.ndarray,
    *,
    steps: int = 120,
    lr: float = 1e-2,
    prior: float = 1.0,
    margin_mm: float = 30.0,
    spacing_mm: float = 3.0,
    chunk: int = 32768,
) -> tuple[torch.Tensor, dict[str, float]]:
    """A code whose structure ``index`` has ``factor``³ its starting soft volume, if it can."""
    device = start.device
    points = torch.from_numpy(_box_points(labels, grid, index, margin_mm, spacing_mm)).to(device)
    voxel_ml = spacing_mm**3 / 1000.0
    before = soft_volume(field, start, points, index, voxel_ml, chunk)
    target = before * factor**3
    moving = torch.zeros(start.shape[-1], dtype=torch.bool, device=device)
    config = field.config
    if config.part_of:
        part = config.part_of[index - 1]
        moving[part * config.latent_dim : (part + 1) * config.latent_dim] = True
    else:
        moving[:] = True
    delta = torch.zeros_like(start, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    volume = before
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # A leaf per step: each chunk's backward adds into code.grad, and none of them has to
        # pass back through a graph an earlier chunk already freed.
        code = (start + delta.detach() * moving).requires_grad_(True)
        volume = soft_volume(field, code, points, index, voxel_ml, chunk)
        weight = 2.0 * (volume / target - 1.0) / target
        soft_volume(field, code, points, index, voxel_ml, chunk, weight=weight)
        penalty = prior * (delta * moving).pow(2).sum() / moving.sum()
        penalty.backward()
        assert delta.grad is not None and code.grad is not None
        delta.grad += code.grad * moving
        optimizer.step()
    final = (start + delta * moving).detach()
    after = soft_volume(field, final, points, index, voxel_ml, chunk)
    return final, {"soft_before_ml": before, "soft_target_ml": target, "soft_after_ml": after}


@dataclass
class ResizeResult:
    """One attempt: what was asked, whether it passes, and what it reached."""

    percent: int
    passes: bool
    failures: list[str]
    volume_ml: dict[str, float]
    linear_percent: float
    volume_percent: float
    soft: dict[str, float]


def _attempt(
    field: Field,
    start: torch.Tensor,
    classes: Sequence[str],
    entity: str,
    percent: int,
    grid: Grid,
    start_labels: np.ndarray,
    start_volumes: dict[str, float],
    limits: dict[str, Any],
    ontology: AnatomyOntology,
    options: dict[str, Any],
) -> ResizeResult:
    index = classes.index(entity) + 1
    code, soft = resize_code(field, start, index, 1 + percent / 100, grid, start_labels, **options)
    thorax = decode_thorax(field, code, grid, classes)
    failures = judge(thorax.measures, limits["volume_ml"], limits["stray"], ontology)
    volume = thorax.measures["volume_ml"][entity]
    ratio = volume / start_volumes[entity]
    return ResizeResult(
        percent=percent,
        passes=not failures,
        failures=[f"{kind}: {message}" for kind, message in failures],
        volume_ml={e: round(v, 1) for e, v in thorax.measures["volume_ml"].items()},
        linear_percent=round((ratio ** (1 / 3) - 1) * 100, 1),
        volume_percent=round((ratio - 1) * 100, 1),
        soft={k: round(v, 1) for k, v in soft.items()},
    )


def largest_change(
    checkpoint: Path,
    report: Path,
    data_dir: Path,
    subject: str,
    entity: str,
    percent: int,
    *,
    device: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The asked change if it passes; otherwise the largest whole percent that does."""
    field, ck = load_checkpoint(checkpoint, pick_device(device))
    device_ = next(field.parameters()).device
    classes = list(ck["classes"][1:])
    measured = json.loads(report.read_text(encoding="utf-8"))
    g = measured["grid"]
    high = np.asarray(g["low"]) + g["spacing_mm"] * (np.asarray(g["shape"]) - 1)
    grid = Grid(g["low"], high, g["spacing_mm"])
    limits = {
        "volume_ml": {e: tuple(v) for e, v in measured["limits"]["volume_ml"].items()},
        "stray": measured["limits"]["stray"],
    }
    ontology = load_ontology(Path("ontology/thorax"), expected_version="0.1.0")
    if subject not in ck["subjects"]:
        raise SystemExit(f"{subject} is not a training subject of this checkpoint")
    start = ck["codes"][ck["subjects"].index(subject)].unsqueeze(0).to(device_)
    begin = decode_thorax(field, start, grid, classes)
    start_volumes = dict(begin.measures["volume_ml"])
    start_failures = judge(begin.measures, limits["volume_ml"], limits["stray"], ontology)
    options = options or {}

    def attempt(p: int) -> ResizeResult:
        result = _attempt(
            field,
            start,
            classes,
            entity,
            p,
            grid,
            begin.volume,
            start_volumes,
            limits,
            ontology,
            options,
        )
        verdict = "passes" if result.passes else result.failures[0]
        print(
            f"{entity} {p:+d}%: reached {result.linear_percent:+.1f}% size, "
            f"{result.volume_percent:+.1f}% volume; {verdict}",
            flush=True,
        )
        return result

    tried = [attempt(percent)]
    if not tried[0].passes:
        low, high = 0, percent
        while high - low > 1:
            middle = (low + high) // 2
            result = attempt(middle)
            tried.append(result)
            if result.passes:
                low = middle
            else:
                high = middle
    passing = [t for t in tried if t.passes]
    best = max(passing, key=lambda t: t.percent) if passing else None
    return {
        "checkpoint": str(checkpoint),
        "subject": subject,
        "entity": entity,
        "asked_percent": percent,
        "start": {
            "volume_ml": {e: round(v, 1) for e, v in start_volumes.items()},
            "failures": [f"{k}: {m}" for k, m in start_failures],
        },
        "largest_passing": None if best is None else best.__dict__,
        "attempts": [t.__dict__ for t in tried],
        "options": options,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Resize from the command line."""
    parser = argparse.ArgumentParser(description="Resize one structure of a Level 2b thorax.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="fields.generate report")
    parser.add_argument("--data", type=Path, default=Path("datasets/processed/level2b"))
    parser.add_argument("--subject", default="s0029")
    parser.add_argument("--entity", default="heart.left_ventricle")
    parser.add_argument("--percent", type=int, default=15)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--prior", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = largest_change(
        args.checkpoint,
        args.report,
        args.data,
        args.subject,
        args.entity,
        args.percent,
        device=args.device,
        options={"steps": args.steps, "prior": args.prior},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    best = result["largest_passing"]
    print(
        "largest passing: none"
        if best is None
        else f"largest passing: asked {best['percent']:+d}%, reached "
        f"{best['linear_percent']:+.1f}% size ({best['volume_percent']:+.1f}% volume)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
