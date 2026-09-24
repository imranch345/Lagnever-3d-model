"""How much of the rotation target the model's inputs can determine at all.

The placement-blind floor is the best predictor keyed on **entity identity**. This is the
best predictor keyed on entity identity **and the relationship graph** — which, by the
corpus's own design, is the only channel that carries the arrangement to the model. Presence,
text features and entity ordering are identical across arrangements.

If that predictor scores no better than the identity-only floor, then no model reading the
graph can do much better either, and a rotation result at the floor says the information is
absent from the inputs rather than that the architecture or the loss failed to use it.

Fitted on ``train`` only. Scenes whose graph is unseen in training fall back to the
entity-only mean, and the coverage is reported, because a predictor that memorised the
training graphs and abstained elsewhere would otherwise look better than it is.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import TEST_SPLITS, load_step8_split
from datasets.whole_organ.relations import relation_signature
from experiments.step10.placement_floor import _mean_rotation
from experiments.step10.rotation_metrics import DEGREES
from generation.neural.nn.transforms import rotation_angle
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

__all__ = ["graph_conditional_floor", "main"]


def _frames_by_scene(
    corpus_dir: str | Path, split: str, builder: WholeOrganBatchBuilder
) -> list[tuple[str, dict[int, torch.Tensor]]]:
    """Per scene: its relation signature and each present entity's frame, as the model sees it."""
    out: list[tuple[str, dict[int, torch.Tensor]]] = []
    for scene in load_step8_split(corpus_dir, split):
        whole = builder.build([scene])
        frames = whole.batch.entity_frames.double()[0]
        present = whole.batch.entity_present.bool()[0]
        entities = {
            int(slot): frames[slot] for slot in torch.nonzero(present).flatten().tolist()
        }
        out.append((relation_signature(scene.edges), entities))
    return out


def graph_conditional_floor(
    corpus_dir: str | Path, *, splits: Sequence[str] = TEST_SPLITS
) -> dict[str, Any]:
    """Rotation error of the best predictor keyed on entity identity and relation graph."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)

    train = _frames_by_scene(corpus_dir, "train", builder)
    by_entity: dict[int, list[torch.Tensor]] = defaultdict(list)
    by_pair: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    for signature, entities in train:
        for slot, frame in entities.items():
            by_entity[slot].append(frame)
            by_pair[(signature, slot)].append(frame)
    entity_table = {slot: _mean_rotation(torch.stack(v)) for slot, v in by_entity.items()}
    pair_table = {key: _mean_rotation(torch.stack(v)) for key, v in by_pair.items()}
    samples = {key: len(v) for key, v in by_pair.items()}

    out: dict[str, Any] = {
        "corpus_dir": str(corpus_dir),
        "distinct_train_graphs": len({signature for signature, _ in train}),
        "train_scenes": len(train),
        "splits": {},
    }
    for split in splits:
        identity_only: list[float] = []
        graph_aware: list[float] = []
        covered = 0
        total = 0
        for signature, entities in _frames_by_scene(corpus_dir, split, builder):
            for slot, frame in entities.items():
                total += 1
                blank = torch.zeros(12, dtype=torch.float64)
                guess_entity = blank.clone()
                guess_entity[6:12] = entity_table[slot]
                identity_only.append(float(rotation_angle(guess_entity, frame)) * DEGREES)
                key = (signature, slot)
                if key in pair_table and samples[key] >= 2:
                    covered += 1
                    guess_pair = blank.clone()
                    guess_pair[6:12] = pair_table[key]
                    graph_aware.append(float(rotation_angle(guess_pair, frame)) * DEGREES)
                else:
                    graph_aware.append(identity_only[-1])
        out["splits"][split] = {
            "entities_scored": total,
            "graph_seen_in_training": covered / total if total else 0.0,
            "identity_only_deg": float(np.mean(identity_only)),
            "identity_and_graph_deg": float(np.mean(graph_aware)),
            "improvement_deg": float(np.mean(identity_only)) - float(np.mean(graph_aware)),
        }
    out["notes"] = [
        "Fitted on train only; the relation graph is the only channel carrying the "
        "arrangement, so this bounds what any graph-reading model can recover.",
        "Rotations are averaged as chordal means, the same estimator the floor uses.",
    ]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Graph-conditional rotation bound.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step10-change2/graph_conditional_floor.json"),
    )
    args = parser.parse_args(argv)
    report = graph_conditional_floor(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"distinct relation graphs in train: {report['distinct_train_graphs']}")
    for split, entry in report["splits"].items():
        print(
            f"  {split:<18} identity-only {entry['identity_only_deg']:6.2f}  "
            f"+graph {entry['identity_and_graph_deg']:6.2f}  "
            f"gain {entry['improvement_deg']:+5.2f}  "
            f"graph seen {entry['graph_seen_in_training']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
