"""Train one arm of the heart prototype.

python -m training.cli --arm A3 --seed 0 --steps 600 \
        --corpus datasets/processed/heart_tier0_2000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.synthetic.heart_corpus import SceneSpec
from datasets.synthetic.store import load_manifest, load_split
from training.loop import Trainer, TrainingConfig

__all__ = ["main", "load_corpus"]


def load_corpus(
    corpus_dir: Path, *, train_limit: int | None, eval_limit: int | None
) -> tuple[list[SceneSpec], list[SceneSpec], dict[str, Any]]:
    """Load the train and evaluation splits plus the corpus manifest."""
    manifest = load_manifest(corpus_dir)
    train = load_split(corpus_dir, "train", limit=train_limit)
    evaluation = load_split(corpus_dir, "validation", limit=eval_limit)
    if not evaluation:
        raise ValueError(f"The validation split of {corpus_dir} is empty.")
    info = {
        "corpus_dir": str(corpus_dir),
        "corpus_id": manifest.corpus_id,
        "format_version": manifest.format_version,
        "ontology": f"{manifest.ontology_id}@{manifest.ontology_version}",
        "scenes_total": manifest.scenes,
        "families": manifest.families,
        "train_scenes_used": len(train),
        "validation_scenes_used": len(evaluation),
        "data_label": manifest.data_label,
    }
    return train, evaluation, info


def main(argv: Sequence[str] | None = None) -> int:
    """Train one arm and print its manifest."""
    parser = argparse.ArgumentParser(description="Train one arm of the Lagnav heart prototype.")
    parser.add_argument("--arm", default="A3", choices=["A0", "A1", "A2", "A3", "A4", "A5", "A6"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--scene-points", type=int, default=256)
    parser.add_argument("--entity-points", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs"))
    args = parser.parse_args(argv)

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train, evaluation, info = load_corpus(
        args.corpus, train_limit=args.train_limit, eval_limit=args.eval_limit
    )
    config = TrainingConfig(
        arm=args.arm,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        scene_points=args.scene_points,
        entity_points=args.entity_points,
        device=args.device,
    )
    trainer = Trainer(config, ontology, domain, train, evaluation, dataset_info=info)
    manifest = trainer.fit(checkpoint_dir=args.out)
    print(json.dumps({"run": manifest.run_id, "results": manifest.results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
