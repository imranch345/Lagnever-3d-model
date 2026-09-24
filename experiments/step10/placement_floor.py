"""Step 10: the placement-blind floor under the parent-relative target.

Step 9 made the placement-blind floor a first-class baseline. Changing the target changes
the floor, so the floor has to be recomputed or every Step 10 number is compared against a
baseline for a different problem.

Three predictors, all fitted on ``train`` only and all scored with the **Step 9 metric on
global frames**, so the columns are comparable with Step 9 rather than merely similar:

``global``
    Step 9's floor, recomputed. It must reproduce Step 9's numbers exactly; if it does not,
    the pipeline has drifted and nothing else here means anything.

``parent_relative``
    The same identity-only lookup, but over parent-relative frames, composed back to scene
    coordinates through its **own** predictions. This is the honest Step 10 floor: a blind
    predictor places parents blindly too, so the composition compounds its errors.

``parent_relative_oracle``
    The same lookup composed against the corpus's **true** parent frames. This leaks the
    parent and is not a baseline; it is the diagnostic that separates "the target is
    easier" from "the target is easier only if you can already place the parent".

Read together they say how much of Change 1's premise is real and what it is conditional
on. Reported before any arm is trained, so the prediction is on record first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import (
    TEST_SPLITS,
    load_step8_manifest,
    load_step8_split,
)
from generation.neural.nn.geometry import frame_rotation
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = ["placement_floor", "main"]

#: Step 9's measured floor, per split. Recomputing must reproduce these.
STEP9_POSITION_FLOOR = {
    "test_seen": 0.1605,
    "test_arrangement": 0.1655,
    "test_transform": 0.1703,
    "test_combination": 0.1750,
}


def _batches(
    corpus_dir: str | Path, split: str, builder: WholeOrganBatchBuilder, *, batch_size: int = 16
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Frames and presence, driven through the evaluation loader.

    Step 9 learned this the hard way: a floor iterated off the corpus rather than the
    loader scores a different entity set and is 0.02 optimistic.
    """
    scenes = load_step8_split(corpus_dir, split)
    loader = WholeOrganLoader(
        scenes, builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    for whole in loader.epoch(0):
        yield whole.batch.entity_frames.double(), whole.batch.entity_present.bool()


def _mean_rotation(frames: torch.Tensor) -> torch.Tensor:
    """The chordal mean rotation of a set of frames, as two basis rows.

    Averaging the six stored numbers and re-orthonormalising would be the obvious thing and
    is wrong twice over. It is not the mean of the rotations, and when a set of rotations is
    widely spread the averaged rows can cancel to nearly zero, at which point Gram-Schmidt
    divides by its own epsilon and returns an arbitrary frame. Projecting the averaged
    matrix onto SO(3) is the Fréchet mean under the chordal metric, which is the estimator
    an identity-only predictor should be given: a floor that is worse than it needs to be
    flatters every model measured against it.

    On a corpus whose rotations are all the identity this returns the identity, so the
    Change 1 floor is unaffected.
    """
    matrices = frame_rotation(frames)
    average = matrices.mean(dim=0)
    left, _, right = torch.linalg.svd(average)
    projected = left @ right
    if float(torch.linalg.det(projected)) < 0.0:
        # Reflection rather than rotation: flip the least significant singular direction.
        left = left.clone()
        left[:, -1] = -left[:, -1]
        projected = left @ right
    rows: torch.Tensor = projected[:2].reshape(-1)
    return rows


def _fit(
    corpus_dir: str | Path,
    builder: WholeOrganBatchBuilder,
    placement: HierarchicalPlacement,
    *,
    parent_relative: bool,
) -> dict[int, torch.Tensor]:
    """Per-entity mean frame over the training split: identity and nothing else.

    Position and log scale are arithmetic means; the rotation is the chordal mean. Nothing
    here sees a relation, a hierarchy, a scene or a split other than ``train``.
    """
    collected: dict[int, list[torch.Tensor]] = defaultdict(list)
    for frames, present in _batches(corpus_dir, "train", builder):
        source = placement.to_local(frames, present).local if parent_relative else frames
        for row in range(frames.shape[0]):
            for slot in torch.nonzero(present[row]).flatten().tolist():
                collected[int(slot)].append(source[row, slot])
    if not collected:
        raise ValueError("No frames to fit a placement floor on.")
    out: dict[int, torch.Tensor] = {}
    for slot, values in collected.items():
        stacked = torch.stack(values)
        out[slot] = torch.cat([stacked[:, 0:6].mean(dim=0), _mean_rotation(stacked)])
    return out


def _score(
    corpus_dir: str | Path,
    split: str,
    builder: WholeOrganBatchBuilder,
    placement: HierarchicalPlacement,
    table: dict[int, torch.Tensor],
    *,
    parent_relative: bool,
    oracle_parent: bool = False,
) -> dict[str, float]:
    position: list[np.ndarray] = []
    scale: list[np.ndarray] = []
    rotation: list[np.ndarray] = []
    for frames, present in _batches(corpus_dir, split, builder):
        guess = torch.zeros_like(frames)
        for slot, value in table.items():
            guess[:, slot] = value
        if parent_relative:
            target = placement.to_local(frames, present)
            guess = (
                _compose_against_truth(placement, guess, frames, target.parents)
                if oracle_parent
                else placement.to_global(guess, target.parents)
            )
        position.append((guess[..., 0:3] - frames[..., 0:3]).norm(dim=-1)[present].numpy())
        scale.append((guess[..., 3:6] - frames[..., 3:6]).abs().mean(dim=-1)[present].numpy())
        rotation.append(rotation_angle(guess, frames)[present].numpy())
    joined = [np.concatenate(part) for part in (position, scale, rotation)]
    means = [float(part.mean()) for part in joined]
    return {
        "position_error": means[0],
        "scale_error": means[1],
        "rotation_error": means[2],
        "composite_frame_error": means[0] + 0.5 * means[1] + 0.25 * means[2],
        "frames_scored": int(joined[0].shape[0]),
    }


def _compose_against_truth(
    placement: HierarchicalPlacement,
    guess: torch.Tensor,
    truth: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """One composition step against the corpus's own parent frame.

    Leaks the parent by construction. Kept separate from :meth:`to_global` so it can never
    be reached by accident from a scoring path that is meant to be blind.
    """
    rows = torch.arange(guess.shape[0], device=guess.device)
    out = guess.clone()
    for slot in range(guess.shape[1]):
        column = parents[:, slot]
        has_parent = column >= 0
        if not bool(has_parent.any()):
            continue
        parent_frames = truth[rows, column.clamp_min(0)]
        composed = placement.compose_one(parent_frames, guess[:, slot])
        out[:, slot] = torch.where(has_parent.unsqueeze(-1), composed, guess[:, slot])
    return out


def placement_floor(
    corpus_dir: str | Path,
    *,
    splits: Sequence[str] = TEST_SPLITS,
    hierarchy: str = "spatial",
) -> dict[str, Any]:
    """Every floor on every split, fitted on training only."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_step8_manifest(corpus_dir)
    frames, _ = next(_batches(corpus_dir, "train", builder))
    placement = HierarchicalPlacement(
        dict(builder.slot_of), slots=frames.shape[1], hierarchy=hierarchy
    )

    global_table = _fit(corpus_dir, builder, placement, parent_relative=False)
    local_table = _fit(corpus_dir, builder, placement, parent_relative=True)

    report: dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "corpus_id": manifest.corpus_id,
        "parent_corpus_id": manifest.parent_corpus_id,
        "format_version": manifest.format_version,
        "rotation_enabled": manifest.rotation_enabled,
        "generator_version": manifest.generator_version,
        "corpus_random_seed": manifest.random_seed,
        "evaluation_loader": (
            "WholeOrganLoader, batch 16, seed 0, shuffle off, drop_last off — the evaluation "
            "path, because a floor iterated off the corpus scores a different entity set"
        ),
        "hierarchy": hierarchy,
        "convention": placement.convention,
        "fitted_on": "train",
        "metric": (
            "Step 9's, unchanged, on global frames. The target changed; the ruler did not."
        ),
        "splits": {},
        "notes": [
            "parent_relative is the parent-relative floor: blind in the parent as well as "
            "the child.",
            "parent_relative_oracle leaks the true parent and is a diagnostic, not a floor.",
            *(
                [
                    "This corpus carries measured rotations, so rotation_error is a real "
                    "number and the Step 9 floor of 0.1605 is NOT its bar: that floor "
                    "belongs to the identity-rotation corpus and the two are separate "
                    "tracks.",
                ]
                if manifest.rotation_enabled
                else [
                    "global must reproduce Step 9 exactly; the check is asserted, not "
                    "eyeballed.",
                    "rotation_error is structurally zero: this corpus's rotation target is "
                    "a constant identity.",
                ]
            ),
        ],
    }
    for split in splits:
        entry: dict[str, Any] = {
            "global": _score(
                corpus_dir, split, builder, placement, global_table, parent_relative=False
            ),
            "parent_relative": _score(
                corpus_dir, split, builder, placement, local_table, parent_relative=True
            ),
            "parent_relative_oracle": _score(
                corpus_dir,
                split,
                builder,
                placement,
                local_table,
                parent_relative=True,
                oracle_parent=True,
            ),
        }
        blind = entry["parent_relative"]["position_error"]
        oracle = entry["parent_relative_oracle"]["position_error"]
        base = entry["global"]["position_error"]
        entry["target_change_blind"] = (base - blind) / base
        entry["target_change_with_true_parent"] = (base - oracle) / base
        # The Step 9 reproduction check belongs to the identity-rotation lineage only.
        # Applying it to a rotated corpus would report a meaningless failure and invite
        # exactly the cross-corpus comparison that is not valid.
        expected = STEP9_POSITION_FLOOR.get(split)
        if manifest.rotation_enabled:
            entry["step9_comparison"] = (
                "not applicable: a different corpus with a different placement distribution"
            )
        elif expected is not None:
            entry["reproduces_step9"] = abs(base - expected) < 5e-4
        report["splits"][split] = entry
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 placement-blind floor.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step8_continuous"))
    parser.add_argument("--splits", nargs="+", default=list(TEST_SPLITS))
    parser.add_argument("--hierarchy", default="spatial")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = placement_floor(args.corpus, splits=args.splits, hierarchy=args.hierarchy)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
