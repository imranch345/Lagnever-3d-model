"""Research diagnostics: measuring whether the representation behaves as designed.

These answer questions the headline metrics cannot:

* **Sibling separation.** Do the four chambers occupy distinct directions in the
  identity space, or have they collapsed into one representation?
* **Relation sensitivity.** Does changing a typed relation move the latent, and does
  a relation in one graph move it differently from the same pair in another?
* **Edit locality.** How much do touched and untouched entities change?
* **Level-of-detail token usage.** Do the later tokens in a nested block contribute
  anything, or is the prefix doing all the work?

The last one exists because Step 5 recorded nested prefixes as an open question and
warned against assuming they work merely because the architecture permits them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as functional_ops

from generation.neural.nn.metrics import edit_drift
from generation.neural.nn.tensors import AWRStructure, BatchBuilder, PrototypeBatch

__all__ = [
    "DiagnosticReport",
    "sibling_separation",
    "relation_sensitivity",
    "lod_token_usage",
    "run_diagnostics",
]

SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "heart.left_ventricle",
        "heart.right_ventricle",
        "heart.left_atrium",
        "heart.right_atrium",
    ),
    (
        "heart.tricuspid_valve",
        "heart.pulmonary_valve",
        "heart.mitral_valve",
        "heart.aortic_valve",
    ),
)


@dataclass(slots=True)
class DiagnosticReport:
    """Diagnostic values and any notes."""

    values: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {"values": dict(self.values), "detail": dict(self.detail), "notes": list(self.notes)}


@torch.no_grad()
def sibling_separation(
    model: torch.nn.Module,
    batch: PrototypeBatch,
    slot_of: Mapping[str, int],
) -> DiagnosticReport:
    """Cosine similarity between sibling entities in the alignment space.

    High similarity for the four chambers would mean the representation has collapsed
    them, which is the failure the ontology-derived hard negatives exist to prevent.
    """
    output = model(batch)
    report = DiagnosticReport()
    if output.alignment is None:
        report.notes = ("This arm has no entity axis, so sibling separation is undefined.",)
        return report
    embeddings = functional_ops.normalize(output.alignment[0], dim=-1)
    similarities: list[float] = []
    for group in SIBLING_GROUPS:
        members = [slot_of[name] for name in group if name in slot_of]
        for first_index, first in enumerate(members):
            for second in members[first_index + 1 :]:
                value = float(embeddings[first] @ embeddings[second])
                report.detail[
                    f"{group[members.index(first)].split('.')[-1]}|{group[members.index(second)].split('.')[-1]}"
                ] = round(value, 4)
                similarities.append(value)
    if similarities:
        report.values["sibling_cosine_mean"] = sum(similarities) / len(similarities)
        report.values["sibling_cosine_max"] = max(similarities)
    random_pairs = torch.randperm(embeddings.shape[0])[:16]
    baseline = [
        float(embeddings[int(a)] @ embeddings[int(b)])
        for a, b in zip(random_pairs[::2], random_pairs[1::2], strict=False)
    ]
    if baseline:
        report.values["random_cosine_mean"] = sum(baseline) / len(baseline)
    return report


@torch.no_grad()
def relation_sensitivity(
    model: torch.nn.Module,
    batch: PrototypeBatch,
    builder: BatchBuilder,
    *,
    subject: str,
    obj: str,
    relation: str = "adjacent_to",
) -> DiagnosticReport:
    """Measure whether adding one typed edge changes the latent, and by how much.

    The edge is added to the AWR structure only. If the graph encoder is doing
    anything, the two entities' context slices move and their identity slices do not.
    """
    report = DiagnosticReport()
    from generation.neural.nn.model import LagnavPrototype

    if not isinstance(model, LagnavPrototype) or model.graph_encoder is None:
        report.notes = ("This arm has no graph encoder, so relation sensitivity is undefined.",)
        return report

    slot_of = {entity_id: index for index, entity_id in enumerate(builder.ontology.ids())}
    structure = batch.structure
    free = (~structure.edge_mask).nonzero().flatten()
    if free.numel() == 0:
        report.notes = ("No free edge slot to add a relation into.",)
        return report

    position = int(free[0].item())
    edge_mask = structure.edge_mask.clone()
    edge_source = structure.edge_source.clone()
    edge_target = structure.edge_target.clone()
    edge_relation = structure.edge_relation.clone()
    edge_graph = structure.edge_graph.clone()
    adjacency = structure.graph_adjacency.clone()

    spec = builder.ontology.registry.get(relation)
    graph_position = builder.extractor.graphs.index(str(spec.graph))
    edge_mask[position] = True
    edge_source[position] = slot_of[subject]
    edge_target[position] = slot_of[obj]
    edge_relation[position] = builder.extractor.relations.index(relation)
    edge_graph[position] = graph_position
    adjacency[graph_position, slot_of[subject], slot_of[obj]] = True

    modified = AWRStructure(
        lod=structure.lod,
        entity_ids=structure.entity_ids,
        entity_type=structure.entity_type,
        semantic_role=structure.semantic_role,
        laterality=structure.laterality,
        parent_index=structure.parent_index,
        depth=structure.depth,
        lod_min=structure.lod_min,
        lod_max=structure.lod_max,
        entity_state=structure.entity_state,
        entity_mask=structure.entity_mask,
        edge_source=edge_source,
        edge_target=edge_target,
        edge_relation=edge_relation,
        edge_graph=edge_graph,
        edge_mask=edge_mask,
        graph_adjacency=adjacency,
        visible_slots=structure.visible_slots,
    )
    import copy

    altered = copy.copy(batch)
    altered.structure = modified
    before = model(batch)
    after = model(altered)
    assert before.entity_latent is not None and after.entity_latent is not None
    identity_width = model.config.identity_width
    delta = (after.entity_latent - before.entity_latent).abs()
    subject_slot, object_slot = slot_of[subject], slot_of[obj]
    report.values["subject_context_delta"] = float(delta[0, subject_slot, identity_width:].mean())
    report.values["object_context_delta"] = float(delta[0, object_slot, identity_width:].mean())
    report.values["identity_delta"] = float(delta[0, :, :identity_width].max())
    others = [
        index
        for index in range(delta.shape[1])
        if index not in (subject_slot, object_slot)
    ]
    report.values["other_entity_context_delta"] = float(
        delta[0, others, identity_width:].mean()
    )
    report.detail["edge_added"] = f"{subject} --{relation}--> {obj}"
    report.notes = (
        "Identity delta must be exactly zero: the encoder never writes the identity slice.",
    )
    return report


@torch.no_grad()
def lod_token_usage(
    model: torch.nn.Module, batch: PrototypeBatch, prefixes: Sequence[int]
) -> DiagnosticReport:
    """Measure whether later geometry tokens contribute anything.

    Decodes the same scene at each prefix and reports how far the occupancy field moves
    between consecutive prefixes. A flat curve means the nested block's later tokens are
    unused and the level-of-detail design is not doing what it claims.
    """
    report = DiagnosticReport()
    probabilities = [
        torch.sigmoid(model(batch, token_prefix=prefix).scene_logits) for prefix in prefixes
    ]
    for index in range(1, len(prefixes)):
        change = float((probabilities[index] - probabilities[index - 1]).abs().mean())
        report.values[f"delta_{prefixes[index - 1]}_to_{prefixes[index]}"] = change
    report.values["delta_first_to_last"] = float(
        (probabilities[-1] - probabilities[0]).abs().mean()
    )
    report.detail["prefixes"] = list(prefixes)
    return report


@torch.no_grad()
def run_diagnostics(
    model: torch.nn.Module,
    batch: PrototypeBatch,
    builder: BatchBuilder,
    *,
    prefixes: Sequence[int] = (4, 8, 16, 24, 32),
) -> dict[str, Any]:
    """Run every diagnostic and return a plain-data report."""
    slot_of = {entity_id: index for index, entity_id in enumerate(builder.ontology.ids())}
    hidden = int(batch.structure.visible_slots[-1].item())
    out: dict[str, Any] = {
        "sibling_separation": sibling_separation(model, batch, slot_of).to_dict(),
        "lod_token_usage": lod_token_usage(model, batch, prefixes).to_dict(),
        "edit_locality": edit_drift(model, batch, hidden_slot=hidden).values,
    }
    try:
        out["relation_sensitivity"] = relation_sensitivity(
            model,
            batch,
            builder,
            subject="heart.left_ventricle",
            obj="heart.superior_vena_cava",
        ).to_dict()
    except (AttributeError, KeyError, ValueError) as exc:  # pragma: no cover - arm dependent
        out["relation_sensitivity"] = {"values": {}, "notes": [f"unavailable: {exc}"]}
    return out
