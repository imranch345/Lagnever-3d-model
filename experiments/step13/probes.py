"""Step 13 mechanism evidence: whether each treatment's mechanism carries what it claims.

The primary metric decides whether an arm helps. It cannot say *why*, and the decision rule
requires mechanism evidence alongside the number, so a treatment that improved for an unrelated
reason is not credited to its hypothesis.

Two probes, both following Step 11's discipline exactly -- same fitting code, same optimiser,
same 2000 steps, fitted on **train**, scored on **validation**, no test split read, no model
weight moved.

**Depth probe (H2, brief section 16).** Does running the stack twice put relational information
into the entity latent that a single pass left out? Taps ``entity_in`` (before the graph, the
control: identity and its attributes) and ``entity_latent`` (what the frame head reads), on A3
and A3-depth. If the latent gains information while rotation does not improve, that is a
downstream bottleneck and is recorded as such rather than acted on.

**Global-context probe (H6, brief section 15).** Does the scene summary contain anything about
the graph? One vector per scene is probed for the graph's own configuration (relation-type
histogram, per-kind edge counts), the scene's transformation (mean rotation), its placement
(centroid), and -- broadcast back to entities -- individual rotation.

Each global target carries a **permutation control**: the same probe, same capacity, on summaries
shuffled across scenes. That is the honest zero point, because a probe with 256 inputs and 2000
steps can fit a lot from nothing. A target is only "present" if the real probe beats its
permuted twin.

**A probe measures presence, never use.** Nothing here shows the model does anything with what a
probe can find.

    python -m experiments.step13.probes
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step11.representation_probe import (
    PROBE_BATCH,
    PROBE_LR,
    PROBE_STEPS,
    _fit_probe,
    _score_probe,
)
from experiments.step13.evaluate import SEEDS, build_arm
from training.whole_organ import WholeOrganLoader

__all__ = ["GLOBAL_TARGETS", "depth_latent_probe", "global_context_probe", "main"]

RELATION_VOCABULARY = 18
GRAPH_KINDS = 3

#: name -> what it asks of the scene summary, from brief section 15.
GLOBAL_TARGETS: dict[str, str] = {
    "relationship_configuration": "which relation types the scene contains, as a histogram",
    "graph_arrangement": "how many edges of each graph kind the scene has",
    "scene_transformation": "the scene's mean 6D rotation: its overall transformation",
    "entity_placement": "the centroid of the scene's entity positions",
    "entity_rotation": "an individual entity's 6D rotation, from its scene's summary alone",
}


@torch.no_grad()
def _collect(trainer: Any, scenes: Sequence[Any], *, batch_size: int = 16) -> dict[str, Any]:
    """Every tensor either probe needs, from one frozen forward pass per batch."""
    loader = WholeOrganLoader(
        scenes, trainer.builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    entity_in: list[torch.Tensor] = []
    entity_latent: list[torch.Tensor] = []
    entity_target: list[torch.Tensor] = []
    context: list[torch.Tensor] = []
    context_per_entity: list[torch.Tensor] = []
    relation_hist: list[torch.Tensor] = []
    kind_counts: list[torch.Tensor] = []
    scene_rotation: list[torch.Tensor] = []
    centroid: list[torch.Tensor] = []

    for whole in loader.epoch(0):
        batch = whole.batch
        staged = trainer.model.stage(batch)
        present = batch.entity_present.bool()
        entity_in.append(staged.entity_latent_in[present].detach())
        entity_latent.append(staged.entity_latent[present].detach())
        entity_target.append(batch.entity_frames[present][..., 6:12].detach())

        summary = trainer.model.scene_summary(staged.entity_latent, present).detach()
        context.append(summary)
        counts = present.sum(dim=1)
        context_per_entity.append(summary.repeat_interleave(counts, dim=0))

        structure = batch.structure
        mask = structure.edge_mask.bool()
        hist = torch.zeros(batch.batch_size, RELATION_VOCABULARY)
        kinds = torch.zeros(batch.batch_size, GRAPH_KINDS)
        for scene in range(batch.batch_size):
            live = mask[scene]
            if bool(live.any()):
                hist[scene] = torch.bincount(
                    structure.edge_relation[scene][live], minlength=RELATION_VOCABULARY
                ).float()
                kinds[scene] = torch.bincount(
                    structure.edge_graph[scene][live], minlength=GRAPH_KINDS
                ).float()
        relation_hist.append(hist / hist.sum(dim=-1, keepdim=True).clamp_min(1.0))
        kind_counts.append(kinds / kinds.sum(dim=-1, keepdim=True).clamp_min(1.0))

        weights = present.unsqueeze(-1).float()
        frames = batch.entity_frames
        denominator = weights.sum(dim=1).clamp_min(1.0)
        scene_rotation.append(((frames[..., 6:12] * weights).sum(dim=1) / denominator).detach())
        centroid.append(((frames[..., 0:3] * weights).sum(dim=1) / denominator).detach())

    return {
        "entity_in": torch.cat(entity_in),
        "entity_latent": torch.cat(entity_latent),
        "entity_target": torch.cat(entity_target),
        "context": torch.cat(context),
        "context_per_entity": torch.cat(context_per_entity),
        "relationship_configuration": torch.cat(relation_hist),
        "graph_arrangement": torch.cat(kind_counts),
        "scene_transformation": torch.cat(scene_rotation),
        "entity_placement": torch.cat(centroid),
    }


def _fit_probe_nd(
    inputs: torch.Tensor, target: torch.Tensor, *, seed: int, hidden: int | None
) -> torch.nn.Module:
    """Step 11's probe, for a target that is not six-dimensional.

    ``representation_probe._fit_probe`` hardcodes a 6-wide output because every Step 11 target
    was a 6D rotation. The global targets here are 18-, 3- and 6-wide, so the width is taken
    from the target. Every other hyperparameter is imported from Step 11 rather than retyped,
    so the two probes cannot drift apart.
    """
    torch.manual_seed(seed)
    width = inputs.shape[-1]
    outputs = target.shape[-1]
    probe: torch.nn.Module = (
        torch.nn.Linear(width, outputs)
        if hidden is None
        else torch.nn.Sequential(
            torch.nn.Linear(width, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, outputs)
        )
    )
    optimiser = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(PROBE_STEPS):
        index = torch.randint(0, inputs.shape[0], (PROBE_BATCH,), generator=generator)
        loss = (probe(inputs[index]) - target[index]).pow(2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return probe


def _relative_error(probe: torch.nn.Module, inputs: torch.Tensor, target: torch.Tensor) -> float:
    """Residual error as a share of the error of predicting the target's mean.

    1.0 means the probe learned nothing beyond the mean; 0.0 means it reconstructed the target.
    Scale-free, so the five targets are comparable to each other.
    """
    with torch.no_grad():
        predicted = probe(inputs)
    residual = (predicted - target).pow(2).sum()
    baseline = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    return float(residual / baseline.clamp_min(1e-12))


def depth_latent_probe(
    *, corpus_dir: Path, arms: Sequence[str] = ("A3", "A3-depth")
) -> dict[str, Any]:
    """Step 11's probe, run on the control and on the deeper arm."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    train_scenes = load_step8_split(corpus_dir, "train")
    valid_scenes = load_step8_split(corpus_dir, "validation")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        for seed in SEEDS:
            trainer = build_arm(arm, seed, ontology, domain, small)
            fitted = _collect(trainer, train_scenes)
            held = _collect(trainer, valid_scenes)
            for tap in ("entity_in", "entity_latent"):
                probe = _fit_probe(fitted[tap], fitted["entity_target"], seed=seed, hidden=None)
                score = _score_probe(probe, held[tap], held["entity_target"])
                results.setdefault(arm, {}).setdefault(tap, {})[str(seed)] = score
                print(f"[step13] {arm:<9} seed {seed} {tap:<14} {score:6.2f} deg", flush=True)

    summary: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            tap: {
                "per_seed_deg": scores,
                "mean_deg": statistics.fmean(scores.values()),
                "std_deg": statistics.stdev(scores.values()) if len(scores) > 1 else 0.0,
            }
            for tap, scores in taps.items()
        }
        for arm, taps in results.items()
    }
    gains: dict[str, float] = {
        arm: float(taps["entity_in"]["mean_deg"]) - float(taps["entity_latent"]["mean_deg"])
        for arm, taps in summary.items()
    }
    return {
        "experiment_id": "step13-depth-latent-probe",
        "status": "diagnostic; probes measure presence, never use",
        "question": (
            "does running the stack twice put relational information into the entity latent "
            "that one pass left out?"
        ),
        "split": "validation",
        "fitted_on": "train",
        "test_splits_read": [],
        "taps": summary,
        "graph_encoder_gain_deg": gains,
        "depth_added_information_deg": (
            gains.get("A3-depth", 0.0) - gains.get("A3", 0.0) if len(gains) > 1 else None
        ),
    }


def global_context_probe(*, corpus_dir: Path, arm: str = "A3-global") -> dict[str, Any]:
    """What the scene summary knows, against a permutation control."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    train_scenes = load_step8_split(corpus_dir, "train")
    valid_scenes = load_step8_split(corpus_dir, "validation")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        trainer = build_arm(arm, seed, ontology, domain, small)
        fitted = _collect(trainer, train_scenes)
        held = _collect(trainer, valid_scenes)
        generator = torch.Generator().manual_seed(13 + seed)
        for target in GLOBAL_TARGETS:
            per_entity = target == "entity_rotation"
            source = "context_per_entity" if per_entity else "context"
            truth_key = "entity_target" if per_entity else target
            inputs, truth = fitted[source], fitted[truth_key]
            held_inputs, held_truth = held[source], held[truth_key]

            shuffle = torch.randperm(inputs.shape[0], generator=generator)
            held_shuffle = torch.randperm(held_inputs.shape[0], generator=generator)

            for label, x, y, hx, hy in (
                ("real", inputs, truth, held_inputs, held_truth),
                ("permuted", inputs[shuffle], truth, held_inputs[held_shuffle], held_truth),
            ):
                probe = _fit_probe_nd(x, y, seed=seed, hidden=256)
                if per_entity:
                    score = _score_probe(probe, hx, hy)
                else:
                    score = _relative_error(probe, hx, hy)
                results.setdefault(target, {}).setdefault(label, {})[str(seed)] = score
            got = results[target]["real"][str(seed)]
            control = results[target]["permuted"][str(seed)]
            unit = "deg" if per_entity else "rel.err"
            print(
                f"[step13] {arm} seed {seed} {target:<28} real {got:7.3f} "
                f"permuted {control:7.3f} {unit}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for target, labels in results.items():
        real_scores = list(labels["real"].values())
        permuted_scores = list(labels["permuted"].values())
        margin = statistics.fmean(permuted_scores) - statistics.fmean(real_scores)
        summary[target] = {
            "what": GLOBAL_TARGETS[target],
            "metric": (
                "geodesic degrees"
                if target == "entity_rotation"
                else "error / mean-predictor error"
            ),
            "real_per_seed": labels["real"],
            "permuted_per_seed": labels["permuted"],
            "real_mean": statistics.fmean(real_scores),
            "permuted_mean": statistics.fmean(permuted_scores),
            "margin_over_permutation": margin,
            "beats_permutation_all_seeds": all(
                labels["real"][s] < labels["permuted"][s] for s in labels["real"]
            ),
        }
    return {
        "experiment_id": "step13-global-context-probe",
        "status": "diagnostic; probes measure presence, never use",
        "arm": arm,
        "split": "validation",
        "fitted_on": "train",
        "test_splits_read": [],
        "control": "the same probe on scene summaries shuffled across scenes",
        "targets": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 13 mechanism probes.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step13"))
    parser.add_argument("--skip-depth", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.skip_depth:
        depth = depth_latent_probe(corpus_dir=args.corpus)
        (args.out / "depth_latent_probe.json").write_text(
            json.dumps(depth, indent=2), encoding="utf-8"
        )
        print("\ndepth probe, graph encoder gain (entity_in - entity_latent):")
        for arm, gain in depth["graph_encoder_gain_deg"].items():
            print(f"  {arm:<10} {gain:+.2f} deg")
        print(f"  depth added: {depth['depth_added_information_deg']}")

    if not args.skip_global:
        glob = global_context_probe(corpus_dir=args.corpus)
        (args.out / "global_context_probe.json").write_text(
            json.dumps(glob, indent=2), encoding="utf-8"
        )
        print("\nglobal context probe:")
        for target, entry in glob["targets"].items():
            print(
                f"  {target:<28} real {entry['real_mean']:7.3f}  "
                f"permuted {entry['permuted_mean']:7.3f}  "
                f"beats control: {entry['beats_permutation_all_seeds']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
