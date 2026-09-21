"""Step 10, post hoc: how much of a child's placement error it inherits from its parent.

**Exploratory, and added after the confirmatory results were known.** It trains nothing
and changes no result; it re-reads the saved ``T1_spatial`` checkpoints to ask *why* the
parent-relative target placed worse, which the pre-registered tables show but cannot
explain.

Under the parent-relative target a child's global frame is its predicted local frame
composed onto its parent's *predicted* global frame, so its error has two sources: its own
local prediction, and whatever its parent got wrong. Recomposing each parented child three
ways separates them:

``as_trained``
    parent's predicted frame ∘ child's predicted local frame. Reproduces the headline, and
    is checked against the model's own output to 1e-6.
``true_parent``
    parent's **true** frame ∘ child's predicted local frame. What the child would score if
    its parent were placed perfectly: its own local error alone.
``true_parent_scale``
    parent's predicted frame with its **scale** replaced by the truth ∘ child's local frame.
    Isolates the isotropic scale term — the part of the cascade a rigid parent would not
    propagate.

Every number is the Step 9 translation error on global frames, bucketed by depth in the
spatial tree, so it reads directly against the ``T0_global`` depth columns.

    python -m experiments.step10.inheritance_diagnostic
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
from experiments.step8.frame_metrics import position_error
from experiments.step10.depth_analysis import depth_table
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.transforms import compose
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["inheritance", "main"]

VARIANTS: tuple[str, ...] = ("as_trained", "true_parent", "true_parent_scale")


def _local_frames(model: Any, batch: Any) -> torch.Tensor:
    """The frame head's raw output — local frames — before any composition."""
    staged = model.stage(batch)
    latent = staged.entity_latent
    if model.config.frame_scene_context:
        return cast(
            torch.Tensor,
            model.frame_head(latent, model.scene_summary(latent, batch.entity_present)),
        )
    return cast(torch.Tensor, model.frame_head(latent))


@torch.no_grad()
def inheritance(
    model: Any,
    scenes: Sequence[Any],
    builder: Any,
    *,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Per-depth translation error of parented children under the three recompositions."""
    model.eval()
    placement = cast(HierarchicalPlacement, model.placement)
    loader = WholeOrganLoader(
        list(scenes), builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    worst_reproduction = 0.0
    depths: list[int] | None = None

    for whole in loader.epoch(0):
        batch = whole.batch.to(device)
        present = batch.entity_present.bool()
        if depths is None:
            depths = depth_table(dict(builder.slot_of), width=int(present.shape[1]))
        local = _local_frames(model, batch)
        parents = placement.parents_for(present)
        predicted = placement.to_global(local, parents)
        reference = model(batch, use_predicted_frames=True).frames
        worst_reproduction = max(
            worst_reproduction, float((predicted - reference)[present].abs().max())
        )
        truth = batch.entity_frames
        rows = torch.arange(truth.shape[0], device=truth.device)

        for slot in range(present.shape[1]):
            column = parents[:, slot]
            mask = present[:, slot] & (column >= 0)
            if not bool(mask.any()):
                continue
            safe = column.clamp_min(0)
            parent_predicted = predicted[rows, safe]
            parent_true = truth[rows, safe]
            parent_true_scale = parent_predicted.clone()
            parent_true_scale[..., 3:6] = parent_true[..., 3:6]
            variants = {
                "as_trained": predicted[:, slot],
                "true_parent": compose(parent_true, local[:, slot], placement.convention),
                "true_parent_scale": compose(
                    parent_true_scale, local[:, slot], placement.convention
                ),
            }
            for name, frames in variants.items():
                error = position_error(frames, truth[:, slot])
                buckets[depths[slot]][name].extend(error[mask].tolist())

    return {
        "reproduces_model_output": worst_reproduction,
        "depths": {
            str(depth): {
                "entities_scored": len(values["as_trained"]),
                **{name: float(np.mean(values[name])) for name in VARIANTS},
            }
            for depth, values in sorted(buckets.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic over every saved ``T1_spatial`` checkpoint."""
    parser = argparse.ArgumentParser(description="Step 10 inheritance diagnostic (post hoc).")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step8_continuous"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs/step10-placement"))
    parser.add_argument("--split", default="test_seen")
    args = parser.parse_args(argv)

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small_train = load_step8_split(args.corpus, "train", limit=16)
    small_validation = load_step8_split(args.corpus, "validation", limit=8)
    scenes = load_step8_split(args.corpus, args.split)
    report = json.loads((args.runs / "placement_report.json").read_text(encoding="utf-8"))

    results: dict[str, Any] = {}
    for run in report["runs"]:
        if run["cell"] != "T1_spatial":
            continue
        config = replace(
            with_variant(
                default_config(run["arm"]),
                placement_target="parent_relative",
                hierarchy="spatial",
                seed=int(run["seed"]),
            ),
            device="cpu",
        )
        trainer = Step10Trainer(config, ontology, domain, small_train, small_validation)
        checkpoint = torch.load(
            args.runs / "checkpoints" / f"{run['run_id']}.pt", map_location="cpu"
        )
        trainer.model.load_state_dict(checkpoint["model"])
        found = inheritance(trainer.model, scenes, trainer.builder, device=trainer.device)
        headline = float(run["splits"][args.split]["inferred"]["translation_error"])
        print(
            f"[inheritance] {run['arm']} seed {run['seed']}: reproduces output to "
            f"{found['reproduces_model_output']:.1e}; headline {headline:.4f}",
            flush=True,
        )
        results.setdefault(run["arm"], {})[str(run["seed"])] = found

    summary: dict[str, Any] = {}
    for arm, by_seed in results.items():
        control = report["depth_aggregates"][f"{arm}/T0_global"][args.split]
        summary[arm] = {}
        for depth in sorted({d for seed in by_seed.values() for d in seed["depths"]}, key=int):
            entry: dict[str, Any] = {
                name: float(np.mean([seed["depths"][depth][name] for seed in by_seed.values()]))
                for name in VARIANTS
            }
            entry["T0_global"] = float(control[depth]["position_error"]["mean"])
            summary[arm][depth] = entry

    payload = {
        "status": "post hoc, exploratory; added after the confirmatory results were known",
        "split": args.split,
        "variants": {
            "as_trained": "predicted parent ∘ predicted local (the headline)",
            "true_parent": "true parent ∘ predicted local (the child's own local error)",
            "true_parent_scale": "predicted parent with true scale ∘ predicted local",
            "T0_global": "the global-target control at the same depth, from the report",
        },
        "by_seed": results,
        "summary": summary,
    }
    out = args.runs / "inheritance_diagnostic.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nPost hoc. Split {args.split}. Translation error of parented children, seed mean.\n")
    print("| arm | depth | T0_global | T1 as trained | T1, true parent | T1, true parent scale |")
    print("| --- | --- | --- | --- | --- | --- |")
    for arm, depths in summary.items():
        for depth, entry in depths.items():
            print(
                f"| {arm} | {depth} | {entry['T0_global']:.4f} | {entry['as_trained']:.4f} "
                f"| {entry['true_parent']:.4f} | {entry['true_parent_scale']:.4f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
