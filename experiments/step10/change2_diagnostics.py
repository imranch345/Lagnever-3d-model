"""Step 10 Change 2 §15–§16: where hierarchical rotation error comes from.

Diagnostic only. It trains nothing, changes no result and introduces no loss; it re-reads
saved checkpoints to separate two things the headline numbers cannot.

§15, error propagation with depth
---------------------------------

Under a parent-relative target a child's global rotation is its parent's composed with its
own, so a parent's rotation error necessarily reaches its children. The question is how much.
For every parented entity this records its parent's rotation error beside its own, per depth,
with the correlation between them.

The ``T0_global`` cell is the control that makes those numbers readable. There, nothing is
composed at all — each entity's frame is predicted directly — so any parent-child correlation
is shared representation, not propagation. Whatever the parent-relative cells show **above**
that baseline is what composition added.

§16, the true-parent oracle
---------------------------

For the parent-relative cells only, each child's predicted local frame is recomposed onto its
parent's **true** frame instead of its predicted one. The gap separates

* how well the model predicts a child's transform *relative to its parent*, from
* how much it loses by composing that onto a parent it also had to predict.

Change 1 found the second term dominant for position. Rotation is a different question,
because a rotation error does not shrink with distance the way a position error does.

The oracle leaks the true parent by construction, is reported as a diagnostic, and is never a
benchmark.

    python -m experiments.step10.change2_diagnostics
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step8.frame_metrics import position_error, rotation_error
from experiments.step10.depth_analysis import depth_table
from experiments.step10.rotation_metrics import DEGREES
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import compose
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["main", "propagation_and_oracle"]


def _local_frames(model: Any, batch: Any) -> torch.Tensor:
    """The frame head's raw output, before any composition."""
    staged = model.stage(batch)
    latent = staged.entity_latent
    if model.config.frame_scene_context:
        return cast(
            torch.Tensor,
            model.frame_head(latent, model.scene_summary(latent, batch.entity_present)),
        )
    return cast(torch.Tensor, model.frame_head(latent))


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return float("nan")
    matrix = np.corrcoef(np.asarray(first), np.asarray(second))
    return float(matrix[0, 1])


@torch.no_grad()
def propagation_and_oracle(
    model: Any,
    scenes: Sequence[Any],
    builder: Any,
    *,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Parent-child rotation error by depth, and the true-parent recomposition."""
    model.eval()
    placement = cast(HierarchicalPlacement | None, getattr(model, "placement", None))
    composing = placement is not None
    reference = placement or HierarchicalPlacement(
        dict(builder.slot_of), slots=len(builder.slot_of)
    )
    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    pairs: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    depths: list[int] | None = None
    reproduction = 0.0

    for whole in loader.epoch(0):
        batch = whole.batch.to(device)
        present = batch.entity_present.bool()
        if depths is None:
            depths = depth_table(dict(builder.slot_of), width=int(present.shape[1]))
        output = model(batch, use_predicted_frames=True)
        predicted = output.frames
        truth = batch.entity_frames
        parents = reference.parents_for(present)
        rows = torch.arange(truth.shape[0], device=truth.device)
        local = _local_frames(model, batch) if composing else None
        if composing and local is not None:
            rebuilt = reference.to_global(local, parents)
            reproduction = max(reproduction, float((rebuilt - predicted)[present].abs().max()))

        child_rotation = rotation_error(predicted, truth) * DEGREES
        child_position = position_error(predicted, truth)
        for slot in range(present.shape[1]):
            column = parents[:, slot]
            mask = present[:, slot] & (column >= 0)
            if not bool(mask.any()):
                continue
            safe = column.clamp_min(0)
            parent_rotation = (
                rotation_error(predicted[rows, safe], truth[rows, safe]) * DEGREES
            )
            bucket = pairs[depths[slot]]
            bucket["parent_rotation"].extend(parent_rotation[mask].tolist())
            bucket["child_rotation"].extend(child_rotation[mask, slot].tolist())
            bucket["child_position"].extend(child_position[mask, slot].tolist())
            if composing and local is not None:
                true_parent = truth[rows, safe]
                oracle = compose(true_parent, local[:, slot], reference.convention)
                bucket["oracle_rotation"].extend(
                    (rotation_error(oracle, truth[:, slot]) * DEGREES)[mask].tolist()
                )
                bucket["oracle_position"].extend(
                    position_error(oracle, truth[:, slot])[mask].tolist()
                )

    out: dict[str, Any] = {
        "composes_a_hierarchy": composing,
        "reproduces_model_output": reproduction,
        "convention": reference.convention if composing else "none: global target",
        "depths": {},
    }
    for depth, values in sorted(pairs.items()):
        entry: dict[str, Any] = {
            "entities_scored": len(values["child_rotation"]),
            "parent_rotation_deg": float(np.mean(values["parent_rotation"])),
            "child_rotation_deg": float(np.mean(values["child_rotation"])),
            "child_position": float(np.mean(values["child_position"])),
            "parent_child_correlation": _correlation(
                values["parent_rotation"], values["child_rotation"]
            ),
        }
        if values["oracle_rotation"]:
            entry["oracle_rotation_deg"] = float(np.mean(values["oracle_rotation"]))
            entry["oracle_position"] = float(np.mean(values["oracle_position"]))
            entry["rotation_lost_to_predicted_parent_deg"] = (
                entry["child_rotation_deg"] - entry["oracle_rotation_deg"]
            )
            entry["position_lost_to_predicted_parent"] = (
                entry["child_position"] - entry["oracle_position"]
            )
        out["depths"][str(depth)] = entry
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run both diagnostics over every saved Change 2 checkpoint."""
    parser = argparse.ArgumentParser(description="Step 10 Change 2 diagnostics (post hoc).")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs/step10-change2"))
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args(argv)

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small_train = load_step8_split(args.corpus, "train", limit=48)
    small_validation = load_step8_split(args.corpus, "validation", limit=16)
    scenes = load_step8_split(args.corpus, args.split)
    report = json.loads((args.runs / "change2_report.json").read_text(encoding="utf-8"))

    from experiments.step10.run_change2 import (
        CELLS,
        ROTATION_LOSS_WEIGHT,
        ROTATION_OBJECTIVE,
    )

    results: dict[str, Any] = {}
    for run in report["runs"]:
        target, hierarchy, convention, _role, _question = CELLS[run["cell"]]
        config = replace(
            with_variant(
                default_config(run["arm"]),
                placement_target=target,  # type: ignore[arg-type]
                hierarchy=hierarchy,
                seed=int(run["seed"]),
            ),
            parent_convention=convention,
            rotation_objective=ROTATION_OBJECTIVE,  # type: ignore[arg-type]
            rotation_loss_weight=ROTATION_LOSS_WEIGHT,
            device="cpu",
        )
        trainer = Step10Trainer(config, ontology, domain, small_train, small_validation)
        checkpoint = torch.load(
            args.runs / "checkpoints" / f"{run['run_id']}.pt", map_location="cpu"
        )
        trainer.model.load_state_dict(checkpoint["model"])
        found = propagation_and_oracle(
            trainer.model, scenes, trainer.builder, device=trainer.device
        )
        print(
            f"[diagnostics] {run['arm']} {run['cell']} seed {run['seed']}: "
            f"reproduces output to {found['reproduces_model_output']:.1e}",
            flush=True,
        )
        results.setdefault(run["cell"], {}).setdefault(run["arm"], {})[str(run["seed"])] = found

    summary: dict[str, Any] = {}
    for cell, arms in results.items():
        summary[cell] = {}
        for arm, seeds in arms.items():
            depths = sorted({d for entry in seeds.values() for d in entry["depths"]}, key=int)
            summary[cell][arm] = {
                depth: {
                    field: float(
                        np.mean(
                            [
                                entry["depths"][depth][field]
                                for entry in seeds.values()
                                if field in entry["depths"][depth]
                            ]
                        )
                    )
                    for field in (
                        "parent_rotation_deg",
                        "child_rotation_deg",
                        "child_position",
                        "parent_child_correlation",
                        "oracle_rotation_deg",
                        "oracle_position",
                        "rotation_lost_to_predicted_parent_deg",
                        "position_lost_to_predicted_parent",
                    )
                    if any(field in entry["depths"][depth] for entry in seeds.values())
                }
                for depth in depths
            }

    payload = {
        "status": "diagnostic only; no training, no loss, no benchmark",
        "split": args.split,
        "what": {
            "parent_rotation_deg": "the parent's own global rotation error",
            "child_rotation_deg": "the child's global rotation error, as trained",
            "oracle_rotation_deg": "the child recomposed onto its parent's TRUE frame",
            "rotation_lost_to_predicted_parent_deg": "as trained minus oracle",
            "parent_child_correlation": (
                "correlation between the two; in T0_global nothing is composed, so its value "
                "there is the shared-representation baseline"
            ),
        },
        "by_seed": results,
        "summary": summary,
    }
    out = args.runs / "change2_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
