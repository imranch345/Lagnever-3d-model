"""Step 13: score all three arms through one evaluation path.

The three arms are the frozen A3 control, A3-depth (the four-layer stack run twice, sharing
weights) and A3-global (the frame head reading a presence-masked scene summary). The decision
rule is a 1.0 degree difference, which is smaller than the gap between the two rotation
aggregates a run record carries, so the arms cannot be compared by reading their training-time
records -- the control's was written by a different invocation than the treatments'. Everything
here is recomputed now, by the same functions the runner uses:

* rotation from :func:`rotation_report`, the ``rotation.<split>.mean_deg`` the frozen
  references quote;
* position and scale from :func:`evaluate_step8` with ``use_predicted_frames=True``, read
  against the placement floors.

``--splits`` decides what is read, and nothing defaults to a test split. Validation carries the
primary result and every diagnostic; the test splits are read exactly once, after the
architecture is frozen, and cannot change anything.

    python -m experiments.step13.evaluate --splits validation
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step8.evaluation import evaluate_step8
from experiments.step9.frame_report import decompose
from experiments.step10.placement_floor import placement_floor
from experiments.step10.rotation_metrics import rotation_report
from training.step10 import Step10Trainer, default_config, with_variant

__all__ = ["ARMS", "MINIMUM_MEANINGFUL_DEG", "evaluate_arms", "main"]

#: arm -> (runs directory, checkpoint stem, graph_recurrence, frame_scene_context)
ARMS: dict[str, tuple[str, str, int, bool]] = {
    "A3": ("experiments/runs/step10-change2-w3", "s10-A3-noctx-l1-globalframe", 1, False),
    "A3-depth": ("experiments/runs/step13/depth", "s10-A3-noctx-l1-globalframe", 2, False),
    "A3-global": ("experiments/runs/step13/global", "s10-A3-ctx-l1-globalframe", 1, True),
}

#: A difference smaller than A3's own seed spread (1.03 deg) cannot be told apart from it.
MINIMUM_MEANINGFUL_DEG = 1.0
WEIGHT = 3.0
SEEDS = (0, 1, 2)


def build_arm(arm: str, seed: int, ontology: Any, domain: Any, small: Sequence[Any]) -> Any:
    """A trainer whose model holds the frozen weights of one arm and seed."""
    runs_dir, stem, recurrence, context = ARMS[arm]
    config = replace(
        with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
        ),
        rotation_objective="chordal",
        rotation_loss_weight=WEIGHT,
        graph_recurrence=recurrence,
        frame_scene_context=context,
        device="cpu",
    )
    trainer = Step10Trainer(config, ontology, domain, small, small)
    state = torch.load(Path(runs_dir) / "checkpoints" / f"{stem}-seed{seed}.pt", map_location="cpu")
    # strict: a silent shape mismatch would score a different architecture than the label says.
    trainer.model.load_state_dict(state["model"])
    trainer.model.eval()
    return trainer


def _score(
    trainer: Any, scenes: Sequence[Any], floor: dict[str, Any], batch_size: int
) -> dict[str, Any]:
    """Rotation, position and scale for one arm, one seed, one split."""
    rotation = rotation_report(
        trainer.model,
        scenes,
        trainer.builder,
        device=trainer.device,
        batch_size=batch_size,
        floor_radians=float(floor["rotation_error"]),
    )
    placement = decompose(
        evaluate_step8(
            trainer.model,
            scenes,
            trainer.builder,
            trainer.slot_of,
            device=trainer.device,
            batch_size=batch_size,
            use_predicted_frames=True,
        )
    )
    return {
        "rotation_deg": float(rotation["mean_deg"]),
        "rotation_median_deg": float(rotation["median_deg"]),
        "rotation_floor_deg": float(rotation["floor_deg"]),
        "rotation_beats_floor": bool(rotation["beats_floor"]),
        "position": float(placement["translation_error"]),
        "position_floor": float(floor["position_error"]),
        "scale": float(placement["scale_error"]),
        "scale_floor": float(floor["scale_error"]),
        "entities_scored": int(rotation["entities_scored"]),
    }


def _summarise(per_seed: dict[str, dict[str, float]], key: str) -> dict[str, Any]:
    """Mean, spread and per-seed values for one metric."""
    values = [per_seed[str(s)][key] for s in SEEDS if str(s) in per_seed]
    return {
        "per_seed": {str(s): per_seed[str(s)][key] for s in SEEDS if str(s) in per_seed},
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def evaluate_arms(
    *,
    corpus_dir: Path,
    splits: Sequence[str],
    arms: Sequence[str] = tuple(ARMS),
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score every arm and seed on every requested split."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    small = load_step8_split(corpus_dir, "train", limit=48)
    scenes = {split: load_step8_split(corpus_dir, split) for split in splits}
    floors = placement_floor(corpus_dir, splits=list(splits))

    raw: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for arm in arms:
        for seed in SEEDS:
            trainer = build_arm(arm, seed, ontology, domain, small)
            for split in splits:
                entry = _score(
                    trainer, scenes[split], floors["splits"][split]["global"], batch_size
                )
                raw.setdefault(arm, {}).setdefault(split, {})[str(seed)] = entry
                print(
                    f"[step13] {arm:<10} seed {seed} {split:<18} "
                    f"rotation {entry['rotation_deg']:6.2f}  position {entry['position']:.4f}  "
                    f"scale {entry['scale']:.4f}",
                    flush=True,
                )

    summary: dict[str, Any] = {}
    for arm, by_split in raw.items():
        for split, per_seed in by_split.items():
            summary.setdefault(split, {})[arm] = {
                metric: _summarise(per_seed, metric)
                for metric in ("rotation_deg", "position", "scale")
            }

    decisions: dict[str, Any] = {}
    for split, arms_here in summary.items():
        control = arms_here["A3"]["rotation_deg"]
        for arm, entry in arms_here.items():
            if arm == "A3":
                continue
            gains = {
                seed: control["per_seed"][seed] - value
                for seed, value in entry["rotation_deg"]["per_seed"].items()
            }
            mean_gain = statistics.fmean(gains.values())
            same_sign = all(g > 0 for g in gains.values()) or all(g < 0 for g in gains.values())
            decisions.setdefault(split, {})[arm] = {
                "control_mean_deg": control["mean"],
                "arm_mean_deg": entry["rotation_deg"]["mean"],
                "per_seed_gain_deg": gains,
                "mean_gain_deg": mean_gain,
                "all_seeds_same_sign": same_sign,
                "exceeds_minimum": bool(abs(mean_gain) >= MINIMUM_MEANINGFUL_DEG and same_sign),
                "direction": "improves" if mean_gain > 0 else "worse",
            }

    return {
        "experiment_id": "step13-depth-vs-global-context",
        "preregistration": "docs/STEP_13_GRAPH_DEPTH_GLOBAL_CONTEXT_PLAN.md",
        "splits": list(splits),
        "test_splits_read": [s for s in splits if s.startswith("test")],
        "rotation_loss_weight": WEIGHT,
        "minimum_meaningful_deg": MINIMUM_MEANINGFUL_DEG,
        "rule": (
            "a treatment is supported only if mean rotation improves by at least "
            f"{MINIMUM_MEANINGFUL_DEG} deg, all three seeds agree in sign, its mechanism shows "
            "measurable evidence, and integrity and leakage pass"
        ),
        "arms": {
            name: {"runs_dir": v[0], "graph_recurrence": v[2], "frame_scene_context": v[3]}
            for name, v in ARMS.items()
        },
        "per_seed": raw,
        "summary": summary,
        "decisions": decisions,
        "notes": [
            "All arms recomputed here through one path; no training-time record is compared.",
            "Checkpoints load with strict=True, so the scored architecture matches the label.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 13 arm evaluation.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_arms(corpus_dir=args.corpus, splits=args.splits)
    label = "test" if any(s.startswith("test") for s in args.splits) else "validation"
    out = args.out or Path(f"experiments/runs/step13/evaluation_{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for split, arms_here in report["summary"].items():
        print(f"\n=== {split} ===")
        print(f"{'arm':<11} {'rotation':>16} {'position':>16} {'scale':>14}")
        for arm, entry in arms_here.items():
            r, p, s = entry["rotation_deg"], entry["position"], entry["scale"]
            print(
                f"{arm:<11} {r['mean']:8.2f} ±{r['std']:4.2f} {p['mean']:10.4f} ±{p['std']:.4f} "
                f"{s['mean']:8.4f} ±{s['std']:.4f}"
            )
        for arm, d in report["decisions"].get(split, {}).items():
            print(
                f"  {arm}: {d['mean_gain_deg']:+.2f} deg, seeds agree {d['all_seeds_same_sign']}, "
                f"supported-by-metric {d['exceeds_minimum']}"
            )
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
