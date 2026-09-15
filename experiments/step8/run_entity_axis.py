"""Step 8 entity-axis stress test, run without teacher-forced placement.

Step 7 found the entity axis was the component with the largest effect by far: an
appearance baseline at the same parameter budget reached 0.0016 part control against 0.72.
But that was measured with ground-truth placement supplied, which is the easy condition.
The open question Step 8 inherits is whether factorisation survives when placement has to
be inferred, or whether it was only carrying the easy case.

A ladder of three, each removing one thing from the one above:

``A3Lite``
    entity axis, per-entity geometry token blocks, untyped relational attention.
``A4``
    entity axis kept, per-entity geometry **removed**: one shared token block and a part
    head that labels points afterwards.
``A0``
    no entity axis at all: a flat latent and a shared field, the appearance baseline.

`A0` has no per-entity frames, so supplied and inferred placement are the same thing for
it. That is reported rather than hidden: it is why the baseline is unaffected by the
condition that halves everything else.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.continuous_corpus import (
    TEST_SPLITS,
    load_step8_manifest,
    load_step8_split,
)
from experiments.step8.evaluation import evaluate_step8
from training.manifest import environment_report, git_commit
from training.step8 import Step8Config, Step8Trainer

__all__ = ["LADDER", "run_entity_axis", "main"]

#: The ladder, and what removing each rung tests.
LADDER: dict[str, str] = {
    "A3Lite": "entity axis, per-entity geometry, untyped relational attention",
    "A4": "entity axis kept, per-entity geometry removed: one shared token block",
    "A0": "no entity axis: flat latent and shared field, the appearance baseline",
}


def run_entity_axis(
    *,
    corpus_dir: Path,
    steps: int,
    seeds: Sequence[int],
    arms: Sequence[str],
    batch_size: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Train each rung of the ladder and evaluate it under both placement conditions."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    manifest = load_step8_manifest(corpus_dir)
    train = load_step8_split(corpus_dir, "train")
    validation = load_step8_split(corpus_dir, "validation")

    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            print(f"[step8-entity] {arm} seed {seed} ...", flush=True)
            config = Step8Config(
                arm=arm,
                seed=seed,
                steps=steps,
                batch_size=batch_size,
                device=device,
                eval_every=0,
            )
            trainer = Step8Trainer(
                config,
                ontology,
                domain,
                train,
                validation,
                dataset_info={"corpus_id": manifest.corpus_id, "corpus_dir": str(corpus_dir)},
            )
            run = trainer.fit(checkpoint_dir=output_dir / "checkpoints")
            record = run.to_dict()
            splits: dict[str, dict[str, dict[str, float]]] = {}
            for split in TEST_SPLITS:
                scenes = load_step8_split(corpus_dir, split)
                splits[split] = {
                    condition: evaluate_step8(
                        trainer.model,
                        scenes,
                        trainer.builder,
                        trainer.slot_of,
                        device=trainer.device,
                        batch_size=batch_size,
                        use_predicted_frames=(condition == "inferred"),
                    )
                    for condition in ("inferred", "supplied")
                }
            record["splits"] = splits
            record["ladder_note"] = LADDER.get(arm, "")
            runs.append(record)
            summary = splits["test_seen"]["inferred"]
            print(
                f"[step8-entity]   {arm} seed {seed}: "
                f"iou={summary['entity_iou_mean']:.4f} "
                f"ctrl={summary['part_control_success']:.4f} "
                f"scene={summary['scene_iou']:.4f}",
                flush=True,
            )

    report = {
        "experiment_id": "step8-entity-axis",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "ladder": LADDER,
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "data_label": manifest.data_label,
        },
        "training": {"steps": steps, "seeds": list(seeds), "arms": list(arms)},
        "runs": runs,
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "A0 has no per-entity frames, so both placement conditions give it the same "
            "numbers. That is why it is unaffected by the condition that halves the rest.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "entity_axis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 entity-axis stress test.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--arms", nargs="+", default=["A3Lite", "A4", "A0"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step8-entity-axis"))
    args = parser.parse_args(argv)

    run_entity_axis(
        corpus_dir=args.corpus,
        steps=args.steps,
        seeds=args.seeds,
        arms=args.arms,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
