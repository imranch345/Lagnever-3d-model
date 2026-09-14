"""Run the research diagnostics on trained checkpoints.

python -m experiments.heart.run_diagnostics \
        --checkpoints experiments/runs/heart-001/checkpoints \
        --corpus datasets/processed/heart_tier0_2000
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.synthetic.sampling import SamplingConfig
from datasets.synthetic.store import load_split
from generation.neural.nn.diagnostics import run_diagnostics
from generation.neural.nn.tensors import BatchBuilder
from training.loop import ArmName, build_model

__all__ = ["diagnose_checkpoint", "main"]


def diagnose_checkpoint(
    checkpoint: Path, corpus_dir: Path, *, batch_size: int = 8, lod: int = 2
) -> dict[str, Any]:
    """Load one checkpoint and run every diagnostic on a held-out batch."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = cast(ArmName, str(payload["config"]["arm"]))
    builder = BatchBuilder(
        ontology, domain, sampling=SamplingConfig(scene_points=256, entity_points=32)
    )
    specs = [
        spec
        for spec in load_split(corpus_dir, "test", limit=200)
        if spec.active_lod == lod
    ][:batch_size]
    batch = builder.build(specs)
    model, _ = build_model(arm, builder, text_features=int(batch.text_features.shape[1]))
    model.load_state_dict(payload["model"])
    model.eval()
    report = run_diagnostics(model, batch, builder)
    report["arm"] = arm
    report["checkpoint"] = checkpoint.name
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Diagnose every checkpoint in a directory."""
    parser = argparse.ArgumentParser(description="Run Lagnav research diagnostics.")
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    reports = []
    for checkpoint in sorted(args.checkpoints.glob("*.pt")):
        reports.append(diagnose_checkpoint(checkpoint, args.corpus))
    payload = {"diagnostics": reports}
    text = json.dumps(payload, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
