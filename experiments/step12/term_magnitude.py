r"""Step 12: how large the relation message is, next to the message it joins.

Step 11 explained an inert channel by measuring it: relation type entered only as a scalar
attention bias, and that bias carried 0.55% of the logit scale. Step 12's negative result needs
the same treatment, because "the weights trained and nothing happened" is not yet an
explanation.

The quantity here is the direct analogue. Inside each layer the attended value is

    attended = sum_j alpha_ij v_j          +    sum_r mass_ir V_rel[r]
               \_______ ordinary ________/       \____ the new term ____/

so the question is the norm of the second against the norm of the first, measured on a real
validation batch through a trained checkpoint. The computation is replicated from
``_GraphAttentionLayer.forward`` rather than hooked, so the numbers are the ones the model saw.

    python -m experiments.step12.term_magnitude
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
from generation.neural.nn.graph import _NEG_INF, PartitionedGraphEncoder
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

__all__ = ["measure", "main"]

CHECKPOINT = "s10-A3-noctx-l1-globalframe-seed{seed}.pt"


@torch.no_grad()
def _per_layer(
    encoder: PartitionedGraphEncoder,
    latent: torch.Tensor,
    structure: Any,
    batch_size: int,
    _return_latent: bool = False,
) -> Any:
    """The two terms' magnitudes, layer by layer."""
    mask = encoder.build_mask(structure, batch_size)
    bias = encoder.build_bias(structure, batch_size)
    routing = encoder.build_routing(structure, batch_size)
    assert routing is not None, "the batch has no live relations to route"
    assert encoder.relation_value is not None, "the checkpoint was trained without the term"
    relation_value = encoder.relation_value.weight
    heads = encoder.config.heads.total
    head_width = encoder.config.head_width

    rows: list[dict[str, Any]] = []
    current = latent
    for index, raw_layer in enumerate(encoder.layers):
        # The ModuleList's element type is Module; the concrete layer is private to graph.py.
        layer = cast(Any, raw_layer)
        batch, entities, _ = current.shape
        normed = layer.norm_attention(current)
        q = layer.query(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        k = layer.key(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        v = layer.value(normed).view(batch, entities, heads, head_width).transpose(1, 2)
        logits = torch.einsum("bhid,bhjd->bhij", q, k) / (head_width**0.5)
        logits = (logits + bias).masked_fill(~mask, _NEG_INF)
        weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        ordinary = torch.einsum("bhij,bhjd->bhid", weights, v)

        vocabulary = relation_value.shape[0]
        mass = torch.zeros((batch, heads, entities, vocabulary), dtype=weights.dtype)
        mass.index_put_(
            (routing.scene, routing.head, routing.query, routing.relation),
            weights[routing.scene, routing.head, routing.query, routing.key],
            accumulate=True,
        )
        message = torch.einsum(
            "bhir,rhd->bhid", mass, relation_value.view(vocabulary, heads, head_width)
        )

        live = structure.entity_mask.bool()
        scoped = torch.tensor([h for h in range(heads) if int(encoder.head_graph[h].item()) >= 0])
        ordinary_norm = ordinary[:, scoped][:, :, live[0]].norm(dim=-1)
        message_norm = message[:, scoped][:, :, live[0]].norm(dim=-1)
        rows.append(
            {
                "layer": index,
                "ordinary_term_norm": float(ordinary_norm.mean()),
                "relation_message_norm": float(message_norm.mean()),
                "share_of_attended": float(message_norm.mean() / ordinary_norm.mean()),
            }
        )
        attended = (
            (ordinary + message).transpose(1, 2).reshape(batch, entities, encoder.config.width)
        )
        width = encoder.config.identity_width
        context = current[..., width:] + layer.out(attended)
        updated = torch.cat([current[..., :width], context], dim=-1)
        context = context + layer.mlp(layer.norm_mlp(updated))
        current = torch.cat([current[..., :width], context], dim=-1)
    if _return_latent:
        # The encoder returns the identity slice from its own input, not the last layer's.
        identity = encoder.config.identity_width
        final = torch.cat([latent[..., :identity], current[..., identity:]], dim=-1)
        return {"layers": rows, "latent": final}
    return rows


def measure(
    *, corpus_dir: Path, runs_dir: Path, seeds: Sequence[int] = (0, 1, 2)
) -> dict[str, Any]:
    """Measure the relation message against the ordinary attended value."""
    from dataclasses import replace as dc_replace

    from training.step10 import Step10Trainer, default_config, with_variant

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = load_step8_split(corpus_dir, "validation", limit=64)
    level = scenes[0].active_lod
    batch = builder.build([s for s in scenes if s.active_lod == level][:16]).batch
    small = load_step8_split(corpus_dir, "train", limit=48)

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        config = dc_replace(
            with_variant(
                default_config("A3"), placement_target="global", hierarchy="spatial", seed=seed
            ),
            rotation_objective="chordal",
            rotation_loss_weight=3.0,
            relation_values=True,
            device="cpu",
        )
        trainer = Step10Trainer(config, ontology, domain, small, small)
        state = torch.load(
            runs_dir / "checkpoints" / CHECKPOINT.format(seed=seed), map_location="cpu"
        )
        trainer.model.load_state_dict(state["model"])
        trainer.model.eval()
        with torch.no_grad():
            staged = cast(Any, trainer.model).stage(batch)
        encoder = cast(PartitionedGraphEncoder, cast(Any, trainer.model).graph_encoder)
        latent = staged.entity_latent_in
        rows = _per_layer(encoder, latent, batch.structure, batch.batch_size)
        assert encoder.relation_value is not None
        weight = encoder.relation_value.weight.detach()
        per_seed[str(seed)] = {
            "layers": rows,
            "mean_share_of_attended": statistics.fmean(r["share_of_attended"] for r in rows),
            "relation_value_abs_max": float(weight.abs().max()),
            "relation_value_std": float(weight.std()),
            "relation_bias_abs_max": float(encoder.relation_bias.weight.detach().abs().max()),
        }
        print(
            f"[step12] seed {seed} relation message is "
            f"{per_seed[str(seed)]['mean_share_of_attended'] * 100:.2f}% of the attended value",
            flush=True,
        )

    shares = [entry["mean_share_of_attended"] for entry in per_seed.values()]
    return {
        "experiment_id": "step12-term-magnitude",
        "status": "diagnostic; explains the Step 12 negative result, decides nothing",
        "split": "validation",
        "test_splits_read": [],
        "per_seed": per_seed,
        "mean_share_of_attended": statistics.fmean(shares),
        "step11_comparison": {
            "relation_bias_share_of_logit_scale": 0.0055,
            "note": (
                "Step 11 section 11 measured the scalar bias channel at 0.55% of the "
                "attention logit scale"
            ),
        },
        "notes": [
            "Replicated from the layer's own forward pass, not hooked, so these are its numbers.",
            "Only heads scoped to a graph are counted; global heads carry no relation message.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 12 relation message magnitude.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--runs", type=Path, default=Path("experiments/runs/step12-relation-values")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step12-relation-values/term_magnitude.json"),
    )
    args = parser.parse_args(argv)
    report = measure(corpus_dir=args.corpus, runs_dir=args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nmean share of attended value: {report['mean_share_of_attended'] * 100:.2f}%")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
