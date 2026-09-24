"""Step 10 Change 2 follow-up: whether rotation learns at a larger weight.

Change 2 put the rotation term at 7% of the frame loss — the pre-registered calibration rule
applied to Step 8's inherited weights — and no arm beat the rotation floor. That leaves two
explanations the experiment could not separate: the architecture does not learn rotation, or
rotation was 7% of the objective. The addendum to the Change 2 report shows the headroom is
real and reachable — a lookup on entity identity and the relationship graph halves the error,
22.65° to 11.57° on ``test_seen`` — so the question is worth settling.

This screens the weight. It is **not** a confirmatory experiment and produces no headline
result.

The rule, fixed before the first run
------------------------------------

Choose the **lowest** weight whose validation rotation error is within 0.1° of the best,
subject to validation position error degrading by no more than **2%** against the current
weight of 0.33. Lowest rather than best, because a larger weight buys rotation with position
and the tie-break should favour the least disturbance to a result that already works; 0.1°
because anything finer is below the effect worth acting on, and without a tolerance the far
decimals of two indistinguishable runs would decide it.

**No test split is read.** The runner refuses one. A weight chosen against test data would
make every later number on this corpus untrustworthy, and Change 2's confirmatory runs are
already on record at 0.33.

    python -m experiments.step10.weight_study --arms A3 --weights 0.33 1.0 3.0 10.0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.continuous_corpus import (
    load_step8_manifest,
    load_step8_split,
)
from experiments.step8.evaluation import evaluate_step8
from experiments.step9.frame_report import decompose
from experiments.step10.placement_floor import placement_floor
from experiments.step10.rotation_metrics import rotation_report
from training.manifest import environment_report, git_commit
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = [
    "DEGRADATION_ALLOWED",
    "TIE_TOLERANCE_DEG",
    "WEIGHTS",
    "choose_weight",
    "main",
    "run_study",
]

#: The weights screened. 0.33 is Change 2's; the rest span 7% to about 70% of the frame loss.
WEIGHTS: tuple[float, ...] = (0.33, 1.0, 3.0, 10.0)

#: How much validation position error may degrade before a weight is disqualified.
DEGRADATION_ALLOWED = 0.02

#: Weights whose validation rotation is within this of the best count as tied, and the tie
#: goes to the lower weight. A tenth of a degree is far below any effect worth acting on —
#: the gap this study is chasing is about 11 — and without it the far decimals of two
#: indistinguishable runs would decide the outcome.
TIE_TOLERANCE_DEG = 0.1

#: The weight Change 2 ran at, and the baseline every other weight is judged against.
BASELINE_WEIGHT = 0.33

SPLIT = "validation"


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_study(
    *,
    corpus_dir: Path,
    arms: Sequence[str],
    weights: Sequence[float],
    seeds: Sequence[int],
    steps: int,
    batch_size: int,
    device: str,
    output_dir: Path,
) -> list[Path]:
    """Train one run per arm, weight and seed, scoring on validation only."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    scored = load_step8_split(corpus_dir, SPLIT)
    train = load_step8_split(corpus_dir, "train")
    floors = placement_floor(corpus_dir, splits=[SPLIT])["splits"][SPLIT]["global"]
    runs_dir = output_dir / "runs"
    written: list[Path] = []

    for seed in seeds:
        for arm in arms:
            for weight in weights:
                label = f"{arm} w={weight} seed {seed}"
                path = runs_dir / f"{arm}__w{weight}__seed{seed}.json"
                if path.exists():
                    print(f"[weights] {label}: already complete, reusing", flush=True)
                    written.append(path)
                    continue
                config = replace(
                    with_variant(
                        default_config(arm),
                        placement_target="global",
                        hierarchy="spatial",
                        seed=seed,
                    ),
                    rotation_objective="chordal",
                    rotation_loss_weight=float(weight),
                    steps=steps,
                    batch_size=batch_size,
                    device=device,
                    eval_every=max(1, steps // 3),
                )
                print(f"[weights] {label} ...", flush=True)
                started = time.time()
                trainer = Step10Trainer(
                    config,
                    ontology,
                    domain,
                    train,
                    scored,
                    dataset_info={
                        "corpus_id": manifest.corpus_id,
                        "corpus_dir": str(corpus_dir),
                        "scored_split": SPLIT,
                    },
                )
                run = trainer.fit(checkpoint_dir=output_dir / "checkpoints")
                values = decompose(
                    evaluate_step8(
                        trainer.model,
                        scored,
                        trainer.builder,
                        trainer.slot_of,
                        device=trainer.device,
                        batch_size=batch_size,
                        use_predicted_frames=True,
                    )
                )
                rotation = rotation_report(
                    trainer.model,
                    scored,
                    trainer.builder,
                    device=trainer.device,
                    batch_size=batch_size,
                    floor_radians=float(floors["rotation_error"]),
                )
                record: dict[str, Any] = {
                    "run_id": run.run_id,
                    "arm": arm,
                    "seed": seed,
                    "rotation_loss_weight": float(weight),
                    "scored_split": SPLIT,
                    "screening_only": True,
                    "position": float(values["translation_error"]),
                    "scale": float(values["scale_error"]),
                    "rotation_deg": float(rotation["mean_deg"]),
                    "rotation": rotation,
                    "floors": {
                        "position": float(floors["position_error"]),
                        "rotation_deg": rotation["floor_deg"],
                        "scale": float(floors["scale_error"]),
                    },
                    "parameters": dict(trainer.parameter_groups),
                    "integrity": trainer.integrity,
                    "protocol": {
                        "steps": steps,
                        "batch_size": batch_size,
                        "device": device,
                        "corpus_id": manifest.corpus_id,
                        "torch_threads": torch.get_num_threads(),
                        "rotation_objective": "chordal",
                    },
                    "seconds": round(time.time() - started, 1),
                }
                _write_atomic(path, record)
                written.append(path)
                print(
                    f"[weights]   {label}: position={record['position']:.4f} "
                    f"(floor {record['floors']['position']:.4f}) "
                    f"rotation={record['rotation_deg']:.2f}deg "
                    f"(floor {record['floors']['rotation_deg']:.2f}) "
                    f"scale={record['scale']:.4f}",
                    flush=True,
                )
    return written


def choose_weight(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the pre-registered rule to the screened runs."""
    by_weight: dict[float, list[Mapping[str, Any]]] = {}
    for run in runs:
        by_weight.setdefault(float(run["rotation_loss_weight"]), []).append(run)
    summary: dict[float, dict[str, float]] = {
        weight: {
            "runs": float(len(group)),
            "position": statistics.fmean(float(run["position"]) for run in group),
            "rotation_deg": statistics.fmean(float(run["rotation_deg"]) for run in group),
            "scale": statistics.fmean(float(run["scale"]) for run in group),
        }
        for weight, group in sorted(by_weight.items())
    }
    if BASELINE_WEIGHT not in summary:
        return {"summary": summary, "chosen": None, "why": "the baseline weight was not run"}

    baseline = summary[BASELINE_WEIGHT]["position"]
    allowed = baseline * (1.0 + DEGRADATION_ALLOWED)
    eligible = {w: e for w, e in summary.items() if e["position"] <= allowed}
    if not eligible:
        return {
            "summary": summary,
            "chosen": None,
            "why": "every weight degraded validation position by more than the allowance",
        }
    best = min(entry["rotation_deg"] for entry in eligible.values())
    chosen = min(
        w for w, entry in eligible.items()
        if entry["rotation_deg"] <= best + TIE_TOLERANCE_DEG
    )
    return {
        "summary": summary,
        "baseline_position": baseline,
        "position_allowance": allowed,
        "eligible_weights": sorted(eligible),
        "chosen": chosen,
        "why": (
            "lowest weight attaining the minimum validation rotation error among those "
            "within the position allowance"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Screen the rotation loss weight.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--arms", nargs="+", default=["A3", "A1"])
    parser.add_argument("--weights", nargs="+", type=float, default=list(WEIGHTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step10-weight-study"))
    parser.add_argument("--decide-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.decide_only:
        run_study(
            corpus_dir=args.corpus,
            arms=args.arms,
            weights=args.weights,
            seeds=args.seeds,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.out,
        )

    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.out / "runs").glob("*.json"))
    ]
    if not runs:
        raise SystemExit(f"no run files under {args.out / 'runs'}")
    decision = choose_weight(runs)
    report = {
        "experiment_id": "step10-rotation-weight-study",
        "status": "screening only; not a confirmatory result",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "scored_split": SPLIT,
        "test_splits_read": [],
        "rule": (
            "lowest weight minimising validation rotation error, subject to validation "
            f"position degrading no more than {DEGRADATION_ALLOWED:.0%} against weight "
            f"{BASELINE_WEIGHT}; weights within {TIE_TOLERANCE_DEG} deg of the best count "
            "as tied and the tie goes to the lower weight"
        ),
        "rule_fixed_before_results": True,
        "decision": decision,
        "runs": runs,
    }
    _write_atomic(args.out / "weight_study.json", report)
    print("\nweight   position   rotation°   scale")
    for weight, entry in sorted(decision["summary"].items()):
        print(
            f"{float(weight):>6}   {float(entry['position']):.4f}     "
            f"{float(entry['rotation_deg']):6.2f}   {float(entry['scale']):.4f}"
        )
    print(f"\nchosen: {decision['chosen']} — {decision['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
