"""Step 14: what the objective is actually made of, and whether the graph reaches it.

Steps 11 to 13 established that the information is available (a lookup on identity plus the
relationship graph reaches 11.57 degrees where A3 reaches 21.38) and that three architectural
pathways for carrying it change nothing. What was never measured is the objective itself.

Three measurements, all on frozen checkpoints, nothing trained:

**Decomposition.** Every term of the real total loss, raw and weighted, as a share of the
total. The total is reassembled here exactly as ``Step8Trainer.train_step`` assembles it --
``breakdown.total + frame_weight * frame_term + lod_weight * lod_term`` -- because the frame
term is added *outside* the loss module and reading only the module's own breakdown would miss
two thirds of the frame supervision.

**Gradients.** The magnitude of the gradient the objective delivers to each pathway: the
identity slice, the graph encoder, the relation parameters, the entity latent and the frame
head. This answers whether the objective sends any optimisation signal through the graph at
all.

**Scrambled graph.** The same batches, the same targets, the same weights, with the relation
graph permuted. If the objective barely moves, the loss does not notice which entities are
related, and no amount of architecture can make it care.

Train split only, for gradients and decomposition. No test split is read.

    python -m experiments.step14.objective_audit
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step7.perturbations import perturb_relations
from experiments.step13.evaluate import SEEDS, build_arm
from generation.neural.nn.losses import LossInputs
from training.step8 import teacher_forcing_at
from training.whole_organ import WholeOrganLoader

__all__ = ["PATHWAYS", "audit", "decompose_objective", "main"]

#: pathway -> the parameter-name prefixes that belong to it.
#:
#: Taken from ``named_parameters``, not from the model's own parameter-count grouping, which
#: uses different labels ("entity_composer" for the module actually named ``composer``,
#: "field_decoder" for ``field``). A prefix that matches nothing yields a gradient norm of
#: exactly zero, which reads as a finding rather than a typo, so :func:`_check_pathways`
#: refuses to measure until every prefix matches a real parameter.
PATHWAYS: dict[str, tuple[str, ...]] = {
    "entity_identity": ("composer",),
    "graph_encoder": ("graph_encoder",),
    # relation_bias only: A3 runs with relation_values=False, so Step 12's relation_value
    # embedding does not exist on this model at all.
    "relation_parameters": ("graph_encoder.relation_bias",),
    "frame_head": ("frame_head",),
    "geometry": ("tokens", "field"),
    "alignment": ("alignment_entity", "alignment_prototypes"),
    "scene_and_text": ("scene_encoder", "text_encoder", "presence_head"),
}


def _check_pathways(model: Any) -> None:
    """Refuse to report a zero that is really a misspelled prefix."""
    names = [name for name, _ in model.named_parameters()]
    unmatched = [
        prefix
        for prefixes in PATHWAYS.values()
        for prefix in prefixes
        if not any(name.startswith(prefix) for name in names)
    ]
    if unmatched:
        raise SystemExit(f"these pathway prefixes match no parameter: {sorted(set(unmatched))}")
    covered = {
        name for name in names for prefixes in PATHWAYS.values() if name.startswith(prefixes)
    }
    missing = sorted(set(names) - covered)
    if missing:
        raise SystemExit(f"these parameters belong to no pathway: {missing}")


BATCHES = 8


def _scalar(value: Any) -> float:
    """A float from a tensor that may still carry a graph, without warning about it."""
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _total_loss(trainer: Any, whole: Any, step: int) -> dict[str, Any]:
    """The real total, reassembled exactly as the trainer assembles it."""
    config = trainer.config
    ratio = teacher_forcing_at(step, config.steps, config)
    batch = whole.batch.to(trainer.device)
    output = trainer.model(batch, teacher_forcing=ratio)
    breakdown = trainer.loss(
        LossInputs(
            output=output,
            batch=batch,
            prefix_output=None,
            geometry_state_invariant=bool(
                getattr(trainer.model, "geometry_is_state_invariant", False)
            ),
        ),
        trainer._prototypes(),
    )
    frame_term = trainer._frame_loss(output, batch)
    lod_term, _ = trainer._nested_lod_loss(whole, batch, ratio)
    total = breakdown.total + config.frame_weight * frame_term + config.lod_weight * lod_term
    parts = trainer._frame_loss_parts(output.frames, batch.entity_frames)
    present = batch.entity_present.unsqueeze(-1).to(output.frames.dtype)
    denominator = present.sum().clamp_min(1)
    frame_split = {
        name: float(((value * present).sum() / denominator).detach())
        for name, value in parts.items()
    }
    return {
        "total": total,
        "module_terms": {name: value for name, value in breakdown.terms.items()},
        "frame_term": frame_term,
        "lod_term": lod_term,
        "frame_split_raw": frame_split,
        "teacher_forcing": ratio,
    }


def decompose_objective(
    trainer: Any, scenes: Sequence[Any], *, batches: int = BATCHES
) -> dict[str, Any]:
    """Every weighted term as a share of the total, averaged over batches."""
    config = trainer.config
    loader = WholeOrganLoader(
        scenes, trainer.builder, batch_size=config.batch_size, seed=0, shuffle=False
    )
    raw: dict[str, list[float]] = {}
    weighted: dict[str, list[float]] = {}
    totals: list[float] = []
    frame_raw: dict[str, list[float]] = {}

    for index, whole in enumerate(loader.epoch(0)):
        if index >= batches:
            break
        # Step 1199: the schedule's final state, where teacher forcing is off and the frame
        # head is fully responsible for placement. Measuring at step 0 would describe a
        # regime the model spends almost none of its training in.
        computed = _total_loss(trainer, whole, config.steps - 1)
        totals.append(_scalar(computed["total"]))
        for name, value in computed["module_terms"].items():
            raw.setdefault(name, []).append(_scalar(value))
            weighted.setdefault(name, []).append(config.loss_weights[name] * _scalar(value))
        raw.setdefault("entity_frame_chordal", []).append(_scalar(computed["frame_term"]))
        weighted.setdefault("entity_frame_chordal", []).append(
            config.frame_weight * _scalar(computed["frame_term"])
        )
        raw.setdefault("nested_lod", []).append(_scalar(computed["lod_term"]))
        weighted.setdefault("nested_lod", []).append(
            config.lod_weight * _scalar(computed["lod_term"])
        )
        for name, value in computed["frame_split_raw"].items():  # noqa: PERF403
            frame_raw.setdefault(name, []).append(value)

    total_mean = statistics.fmean(totals)
    components = {
        name: {
            "raw": statistics.fmean(raw[name]),
            "weight": (
                config.frame_weight
                if name == "entity_frame_chordal"
                else config.lod_weight
                if name == "nested_lod"
                else config.loss_weights[name]
            ),
            "weighted": statistics.fmean(values),
            "share_of_total": statistics.fmean(values) / total_mean,
        }
        for name, values in weighted.items()
    }
    # Inside the chordal frame term: translation + 0.5 * scale + rotation_loss_weight * rotation.
    inner = {
        "translation": 1.0,
        "scale": 0.5,
        "rotation": config.rotation_loss_weight,
    }
    frame_inner = {
        name: {
            "raw": statistics.fmean(values),
            "coefficient": inner[name],
            "weighted_inside_term": inner[name] * statistics.fmean(values),
            "share_of_total": config.frame_weight
            * inner[name]
            * statistics.fmean(values)
            / total_mean,
        }
        for name, values in frame_raw.items()
    }
    return {
        "total": total_mean,
        "components": components,
        "chordal_frame_term_breakdown": frame_inner,
        "batches": len(totals),
    }


def _gradient_norms(trainer: Any, whole: Any, step: int) -> dict[str, float]:
    """L2 norm of the gradient the objective delivers to each pathway."""
    trainer.model.zero_grad(set_to_none=True)
    computed = _total_loss(trainer, whole, step)
    total = cast(Any, computed["total"])
    total.backward()
    norms: dict[str, float] = {}
    for pathway, prefixes in PATHWAYS.items():
        squared = 0.0
        for name, parameter in trainer.model.named_parameters():
            if parameter.grad is None or not name.startswith(prefixes):
                continue
            squared += float(parameter.grad.pow(2).sum())
        norms[pathway] = squared**0.5
    everything = sum(
        float(p.grad.pow(2).sum()) for p in trainer.model.parameters() if p.grad is not None
    )
    norms["all_parameters"] = everything**0.5
    trainer.model.zero_grad(set_to_none=True)
    return norms


def _scrambled_objective(trainer: Any, whole: Any, step: int) -> dict[str, float]:
    """The objective with the correct graph, and with the relations permuted."""
    from dataclasses import replace as dc_replace

    inverse = trainer.builder.inverse_relation_table()
    generator = torch.Generator().manual_seed(14)
    correct = _scalar(_total_loss(trainer, whole, step)["total"])
    damaged_batch = perturb_relations(
        whole.batch, "randomise_spatial_endpoints", inverse_relations=inverse, generator=generator
    )
    scrambled = _scalar(_total_loss(trainer, dc_replace(whole, batch=damaged_batch), step)["total"])
    correct_parts = _total_loss(trainer, whole, step)["frame_split_raw"]
    damaged_parts = _total_loss(trainer, dc_replace(whole, batch=damaged_batch), step)[
        "frame_split_raw"
    ]
    return {
        "objective_correct_graph": correct,
        "objective_scrambled_graph": scrambled,
        "objective_change": scrambled - correct,
        "objective_relative_change": (scrambled - correct) / abs(correct),
        "rotation_term_correct": correct_parts["rotation"],
        "rotation_term_scrambled": damaged_parts["rotation"],
        "rotation_term_relative_change": (damaged_parts["rotation"] - correct_parts["rotation"])
        / abs(correct_parts["rotation"]),
    }


def audit(*, corpus_dir: Path, arm: str = "A3", batches: int = BATCHES) -> dict[str, Any]:
    """Decomposition, gradients and the scrambled-graph test, for every seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    train_scenes = load_step8_split(corpus_dir, "train")

    decompositions: dict[str, Any] = {}
    gradients: dict[str, dict[str, list[float]]] = {}
    scrambles: dict[str, dict[str, list[float]]] = {}

    for seed in SEEDS:
        trainer = build_arm(arm, seed, ontology, domain, small)
        _check_pathways(trainer.model)
        trainer.model.train()
        decompositions[str(seed)] = decompose_objective(trainer, train_scenes, batches=batches)
        print(f"[step14] seed {seed} objective decomposed", flush=True)

        loader = WholeOrganLoader(
            train_scenes,
            trainer.builder,
            batch_size=trainer.config.batch_size,
            seed=0,
            shuffle=False,
        )
        step = trainer.config.steps - 1
        for index, whole in enumerate(loader.epoch(0)):
            if index >= batches:
                break
            for key, value in _gradient_norms(trainer, whole, step).items():
                gradients.setdefault(str(seed), {}).setdefault(key, []).append(value)
            for key, value in _scrambled_objective(trainer, whole, step).items():
                scrambles.setdefault(str(seed), {}).setdefault(key, []).append(value)
        print(f"[step14] seed {seed} gradients and scramble measured", flush=True)

    gradient_summary = {
        pathway: {
            "per_seed_mean": {s: statistics.fmean(v[pathway]) for s, v in gradients.items()},
            "mean": statistics.fmean([statistics.fmean(v[pathway]) for v in gradients.values()]),
            "median": statistics.median([x for v in gradients.values() for x in v[pathway]]),
            "std": statistics.stdev([x for v in gradients.values() for x in v[pathway]]),
        }
        for pathway in list(PATHWAYS) + ["all_parameters"]
    }
    scramble_summary = {
        key: {
            "per_seed_mean": {s: statistics.fmean(v[key]) for s, v in scrambles.items()},
            "mean": statistics.fmean([statistics.fmean(v[key]) for v in scrambles.values()]),
        }
        for key in next(iter(scrambles.values()))
    }
    return {
        "experiment_id": "step14-objective-audit",
        "status": "diagnostic; nothing was trained and no coefficient was changed",
        "preregistration": "docs/STEP_14_OBJECTIVE_DIAGNOSTIC_PLAN.md",
        "arm": arm,
        "split": "train",
        "test_splits_read": [],
        "measured_at_step": "steps - 1 (teacher forcing off, the regime training mostly runs in)",
        "decomposition": decompositions,
        "gradients": gradient_summary,
        "scrambled_graph": scramble_summary,
        "notes": [
            "The total is reassembled as Step8Trainer.train_step does, including the frame and "
            "nested-LOD terms that are added outside the loss module.",
            "Gradients are measured on frozen weights and discarded; no optimiser step runs.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 14 objective audit.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step14/objective_audit.json")
    )
    args = parser.parse_args(argv)
    report = audit(corpus_dir=args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    first = report["decomposition"]["0"]
    print(f"\n=== objective decomposition (seed 0, total {first['total']:.4f}) ===")
    print(f"{'component':<30}{'raw':>10}{'weight':>8}{'weighted':>11}{'share':>9}")
    for name, entry in sorted(first["components"].items(), key=lambda kv: -kv[1]["share_of_total"]):
        print(
            f"{name:<30}{entry['raw']:10.4f}{entry['weight']:8.2f}"
            f"{entry['weighted']:11.4f}{entry['share_of_total'] * 100:8.1f}%"
        )
    print("\ninside the chordal frame term:")
    for name, entry in first["chordal_frame_term_breakdown"].items():
        print(
            f"  {name:<14} raw {entry['raw']:8.4f}  coeff {entry['coefficient']:4.1f}  "
            f"share of total {entry['share_of_total'] * 100:5.1f}%"
        )
    print("\n=== gradient norms ===")
    for pathway, entry in report["gradients"].items():
        print(f"  {pathway:<22} mean {entry['mean']:10.5f}  median {entry['median']:10.5f}")
    print("\n=== scrambled graph ===")
    for key, entry in report["scrambled_graph"].items():
        print(f"  {key:<34} {entry['mean']:+.6f}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
