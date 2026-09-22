"""Step 11 E2: whether the relational signal is in the latent, or was never written.

The ablation established that A3's output barely moves when the relationship graph's *content*
is destroyed. Two readings survive that, and they call for different fixes:

* the graph encoder does write relational information into the entity latent, and the frame
  head ignores it — a head problem (H1);
* the graph encoder never writes it, so there is nothing for the head to ignore — an encoder
  problem (H2 or H3).

A probe separates them. Two taps on the same frozen forward pass:

``entity_in``
    before the graph encoder. Identity, anatomy type, semantic role, laterality. The control.
``entity_latent``
    after the graph encoder. Exactly what the frame head reads.

If the second beats the first, the encoder wrote something. If the second also beats A3's own
error, the information is present and unused.

A probe measures **presence, never use.** Nothing here is evidence that the model does
anything with what a probe can find.

Fitted on ``train``, reported on ``validation``. No test split is read. No model weight moves.

    python -m experiments.step11.representation_probe
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
from torch import nn

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.continuous_corpus import load_step8_split
from experiments.step8.frame_metrics import rotation_error
from experiments.step10.rotation_metrics import DEGREES
from training.step10 import Step10Trainer, default_config, with_variant
from training.whole_organ import WholeOrganLoader

__all__ = ["TAPS", "probe_all", "main"]

TAPS: tuple[str, ...] = ("entity_in", "entity_latent")
SPLIT = "validation"
WEIGHT = 3.0

#: Fixed in the Step 11 plan before any probe ran.
PROBE_STEPS = 2000
PROBE_BATCH = 64
PROBE_LR = 1e-3


@torch.no_grad()
def _features(
    model: Any, scenes: Sequence[Any], builder: Any, *, batch_size: int = 16
) -> dict[str, torch.Tensor]:
    """Both taps and the rotation target, for every present entity."""
    loader = WholeOrganLoader(
        scenes, builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    collected: dict[str, list[torch.Tensor]] = {tap: [] for tap in TAPS}
    targets: list[torch.Tensor] = []
    for whole in loader.epoch(0):
        batch = whole.batch
        staged = model.stage(batch)
        present = batch.entity_present.bool()
        collected["entity_in"].append(staged.entity_latent_in[present].detach())
        collected["entity_latent"].append(staged.entity_latent[present].detach())
        targets.append(batch.entity_frames[present][..., 6:12].detach())
    out = {tap: torch.cat(values) for tap, values in collected.items()}
    out["target"] = torch.cat(targets)
    return out


def _fit_probe(
    inputs: torch.Tensor, target: torch.Tensor, *, seed: int, hidden: int | None
) -> nn.Module:
    """Fit one probe. ``hidden=None`` is the linear probe."""
    torch.manual_seed(seed)
    width = inputs.shape[-1]
    probe: nn.Module = (
        nn.Linear(width, 6)
        if hidden is None
        else nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 6))
    )
    optimiser = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(PROBE_STEPS):
        index = torch.randint(0, inputs.shape[0], (PROBE_BATCH,), generator=generator)
        predicted = probe(inputs[index])
        loss = (predicted - target[index]).pow(2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    return probe


@torch.no_grad()
def _score_probe(probe: nn.Module, inputs: torch.Tensor, target: torch.Tensor) -> float:
    """Geodesic rotation error in degrees, the metric every report uses."""
    predicted = probe(inputs)
    blank = torch.zeros(predicted.shape[0], 12, dtype=predicted.dtype)
    frame_pred = blank.clone()
    frame_pred[:, 6:12] = predicted
    frame_true = blank.clone()
    frame_true[:, 6:12] = target
    return float((rotation_error(frame_pred, frame_true) * DEGREES).mean())


def probe_all(*, corpus_dir: Path, runs_dir: Path) -> dict[str, Any]:
    """Fit and score both probes on both taps, for every frozen A3 seed."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    train_scenes = load_step8_split(corpus_dir, "train")
    valid_scenes = load_step8_split(corpus_dir, SPLIT)
    small = load_step8_split(corpus_dir, "train", limit=48)
    report = json.loads((runs_dir / "change2_report.json").read_text(encoding="utf-8"))
    a3 = sorted(
        (r for r in report["runs"] if r["arm"] == "A3" and r["cell"] == "T0_global"),
        key=lambda r: int(r["seed"]),
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for run in a3:
        seed = int(run["seed"])
        config = replace(
            with_variant(
                default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
            ),
            rotation_objective="chordal",
            rotation_loss_weight=WEIGHT,
            device="cpu",
        )
        trainer = Step10Trainer(config, ontology, domain, small, small)
        checkpoint = torch.load(
            runs_dir / "checkpoints" / f"{run['run_id']}.pt", map_location="cpu"
        )
        trainer.model.load_state_dict(checkpoint["model"])
        trainer.model.eval()
        fitted = _features(trainer.model, train_scenes, trainer.builder)
        held = _features(trainer.model, valid_scenes, trainer.builder)
        for tap in TAPS:
            for name, hidden in (("linear", None), ("mlp", 256)):
                probe = _fit_probe(fitted[tap], fitted["target"], seed=seed, hidden=hidden)
                score = _score_probe(probe, held[tap], held["target"])
                results.setdefault(tap, {}).setdefault(name, {})[str(seed)] = score
                print(f"[probe] seed {seed} {tap:<14} {name:<6} {score:6.2f} deg", flush=True)

    summary: dict[str, Any] = {}
    for tap, kinds in results.items():
        summary[tap] = {
            name: {
                "per_seed_deg": scores,
                "mean_deg": statistics.fmean(scores.values()),
                "std_deg": statistics.stdev(scores.values()) if len(scores) > 1 else 0.0,
            }
            for name, scores in kinds.items()
        }
    graph_gain = {
        name: summary["entity_in"][name]["mean_deg"] - summary["entity_latent"][name]["mean_deg"]
        for name in ("linear", "mlp")
    }
    return {
        "experiment_id": "step11-representation-probe",
        "status": "diagnostic; probes measure presence, never use",
        "split": SPLIT,
        "test_splits_read": [],
        "fitted_on": "train",
        "protocol": {
            "steps": PROBE_STEPS,
            "batch": PROBE_BATCH,
            "learning_rate": PROBE_LR,
            "optimiser": "AdamW",
            "target": "the 6D rotation, scored as geodesic degrees",
        },
        "taps": summary,
        "graph_encoder_gain_deg": graph_gain,
        "notes": [
            "entity_in is the control: identity and its attributes, before any graph layer.",
            "entity_latent is exactly what the frame head reads.",
            "A probe beating the model is evidence of presence, not of use.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 11 representation probe.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs/step10-change2-w3"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step11/representation_probe.json")
    )
    args = parser.parse_args(argv)

    report = probe_all(corpus_dir=args.corpus, runs_dir=args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\ntap             linear    mlp")
    for tap, kinds in report["taps"].items():
        print(f"{tap:<14} {kinds['linear']['mean_deg']:6.2f}  {kinds['mlp']['mean_deg']:6.2f}")
    print(f"\ngraph encoder gain: {report['graph_encoder_gain_deg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
