"""Step 7 Experiments 1, 2 and 3: perturbing the Anatomical World Representation.

The whole-organ benchmark asks how well a model reproduces an organ. These experiments
ask a narrower question: **which parts of the representation is the model actually
reading?** The method is to damage one channel at a time at evaluation time, on a model
that was trained on intact input, and see what stops working.

A channel the model depends on must degrade the output when it is corrupted. A channel
that can be deleted with no measurable effect is not being used, whatever role the
architecture assigns it. That is the evidence a KEEP or REMOVE decision needs.

Every perturbation here alters only the relationship tensors. Entities, presence, text
features and sampled points are untouched, so a change in the output can only have come
through the graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace as dataclass_replace

import torch

from generation.neural.nn.tensors import AWRStructure, PrototypeBatch

__all__ = [
    "GRAPH_INDEX",
    "RELATION_PERTURBATIONS",
    "AWR_CASES",
    "perturb_relations",
    "restrict_graphs",
    "describe",
]

#: Index of each typed graph along the adjacency axis, matching the feature extractor.
GRAPH_INDEX: Mapping[str, int] = {"structure": 0, "spatial": 1, "functional": 2}

#: Experiment 1 and 2 perturbation categories, declared before running.
RELATION_PERTURBATIONS: tuple[str, ...] = (
    "intact",
    "drop_spatial",
    "invert_spatial",
    "invert_spatial_labels",
    "shuffle_spatial_types",
    "randomise_spatial_endpoints",
    "drop_functional",
    "drop_structure",
)

#: Experiment 3 partial-AWR cases, declared before running.
AWR_CASES: Mapping[str, tuple[str, ...]] = {
    "A_full": ("structure", "spatial", "functional"),
    "B_no_spatial": ("structure", "functional"),
    "C_no_functional": ("structure", "spatial"),
    "D_no_structure": ("spatial", "functional"),
    "E_entities_only": (),
}


def describe(name: str) -> str:
    """One line on what a perturbation does, for the report."""
    return {
        "intact": "unmodified relationship graph; the control",
        "drop_spatial": "every spatial edge removed",
        "invert_spatial": (
            "every spatial relation replaced by its inverse, endpoints swapped with it. "
            "A structural no-op: the encoder adds the inverse relation's embedding in "
            "the reverse direction, so the relation bias is bit-identical"
        ),
        "invert_spatial_labels": (
            "every spatial relation replaced by its inverse with the endpoints left in "
            "place, so the graph asserts the opposite of what the organ shows"
        ),
        "shuffle_spatial_types": "spatial relation types permuted among the same edges",
        "randomise_spatial_endpoints": "spatial edges rewired to random entity pairs",
        "drop_functional": "every functional edge removed",
        "drop_structure": "every structural edge removed",
        "A_full": "the whole representation",
        "B_no_spatial": "structure and function, no space",
        "C_no_functional": "structure and space, no function",
        "D_no_structure": "space and function, no partonomy",
        "E_entities_only": "entities alone, no relationships at all",
    }[name]


def _graph_mask(structure: AWRStructure, graph: str) -> torch.Tensor:
    """Boolean mask over the edge axis selecting one typed graph's live edges."""
    wanted = GRAPH_INDEX[graph]
    return (structure.edge_graph == wanted) & structure.edge_mask


def _blank_edges(structure: AWRStructure, selection: torch.Tensor) -> AWRStructure:
    """Remove the selected edges from both the edge list and the adjacency."""
    edge_mask = structure.edge_mask & ~selection
    adjacency = structure.graph_adjacency.clone()
    for graph, index in GRAPH_INDEX.items():
        del graph
        keep = selection & (structure.edge_graph == index)
        if not bool(keep.any()):
            continue
        adjacency = _clear_adjacency(adjacency, structure, keep, index)
    return dataclass_replace(structure, edge_mask=edge_mask, graph_adjacency=adjacency)


def _clear_adjacency(
    adjacency: torch.Tensor, structure: AWRStructure, selection: torch.Tensor, graph: int
) -> torch.Tensor:
    """Zero the adjacency entries belonging to the selected edges."""
    positions = selection.nonzero()
    if positions.numel() == 0:
        return adjacency
    if structure.edge_source.dim() == 1:
        rows = structure.edge_source[positions[:, 0]]
        columns = structure.edge_target[positions[:, 0]]
        adjacency[graph, rows, columns] = 0.0
        return adjacency
    scenes = positions[:, 0]
    index = positions[:, 1]
    rows = structure.edge_source[scenes, index]
    columns = structure.edge_target[scenes, index]
    adjacency[scenes, graph, rows, columns] = 0.0
    return adjacency


def restrict_graphs(batch: PrototypeBatch, keep: Sequence[str]) -> PrototypeBatch:
    """Keep only the named typed graphs; delete the rest.

    This is the Experiment 3 construction: the model is asked to work from part of the
    representation and the loss of accuracy says what the missing part was worth.
    """
    unknown = sorted(set(keep) - set(GRAPH_INDEX))
    if unknown:
        raise ValueError(f"Unknown graphs {unknown}.")
    structure = batch.structure
    drop = [graph for graph in GRAPH_INDEX if graph not in set(keep)]
    if not drop:
        return batch
    selection = torch.zeros_like(structure.edge_mask, dtype=torch.bool)
    for graph in drop:
        selection = selection | _graph_mask(structure, graph)
    return dataclass_replace(batch, structure=_blank_edges(structure, selection))


def perturb_relations(
    batch: PrototypeBatch,
    kind: str,
    *,
    inverse_relations: torch.Tensor,
    generator: torch.Generator | None = None,
) -> PrototypeBatch:
    """Apply one declared perturbation to a batch's relationship graph.

    Args:
        batch: The batch to damage. Never modified in place.
        kind: One of :data:`RELATION_PERTURBATIONS`.
        inverse_relations: Relation id to inverse relation id, or -1 where none exists.
        generator: Seeded source for the randomising perturbations.

    """
    if kind not in RELATION_PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {kind!r}.")
    structure = batch.structure
    if kind == "intact":
        return batch
    if kind == "drop_spatial":
        return dataclass_replace(
            batch, structure=_blank_edges(structure, _graph_mask(structure, "spatial"))
        )
    if kind == "drop_functional":
        return dataclass_replace(
            batch, structure=_blank_edges(structure, _graph_mask(structure, "functional"))
        )
    if kind == "drop_structure":
        return dataclass_replace(
            batch, structure=_blank_edges(structure, _graph_mask(structure, "structure"))
        )

    spatial = _graph_mask(structure, "spatial")
    if kind == "invert_spatial_labels":
        # Flip the relation and leave the endpoints alone, so the graph asserts the
        # opposite of what the organ shows. Unlike ``invert_spatial`` this is visible to
        # the encoder, because the bias for (a, left_of, b) is not the bias for
        # (a, right_of, b). It states something the ontology forbids, which is the point:
        # a corruption test asks what the model does with a graph that is wrong.
        table = inverse_relations.to(structure.edge_relation.device)
        flipped = table[structure.edge_relation.clamp_min(0)]
        relation = torch.where(spatial & (flipped >= 0), flipped, structure.edge_relation)
        return dataclass_replace(
            batch, structure=dataclass_replace(structure, edge_relation=relation)
        )

    if kind == "invert_spatial":
        table = inverse_relations.to(structure.edge_relation.device)
        flipped = table[structure.edge_relation.clamp_min(0)]
        # An edge with no declared inverse keeps its relation; inverting it would be a
        # different corruption than the one being measured.
        relation = torch.where(spatial & (flipped >= 0), flipped, structure.edge_relation)
        # The inverse of a relation is the same edge read backwards, so the endpoints
        # swap too. Leaving them put would assert something the ontology forbids.
        source = torch.where(spatial & (flipped >= 0), structure.edge_target, structure.edge_source)
        target = torch.where(spatial & (flipped >= 0), structure.edge_source, structure.edge_target)
        return dataclass_replace(
            batch,
            structure=dataclass_replace(
                structure, edge_relation=relation, edge_source=source, edge_target=target
            ),
        )

    if generator is None:
        raise ValueError(f"Perturbation {kind!r} needs a seeded generator.")

    if kind == "shuffle_spatial_types":
        relation = structure.edge_relation.clone()
        positions = spatial.nonzero()
        if positions.numel():
            values = relation[spatial]
            order = torch.randperm(values.numel(), generator=generator)
            relation[spatial] = values[order]
        return dataclass_replace(
            batch, structure=dataclass_replace(structure, edge_relation=relation)
        )

    # randomise_spatial_endpoints
    entities = structure.entity_count
    source = structure.edge_source.clone()
    target = structure.edge_target.clone()
    count = int(spatial.sum())
    if count:
        source[spatial] = torch.randint(entities, (count,), generator=generator)
        target[spatial] = torch.randint(entities, (count,), generator=generator)
    adjacency = structure.graph_adjacency.clone()
    index = GRAPH_INDEX["spatial"]
    if adjacency.dim() == 3:
        adjacency[index] = 0.0
    else:
        adjacency[:, index] = 0.0
    rebuilt = dataclass_replace(
        structure, edge_source=source, edge_target=target, graph_adjacency=adjacency
    )
    return dataclass_replace(batch, structure=_rebuild_adjacency(rebuilt, spatial, index))


def _rebuild_adjacency(
    structure: AWRStructure, selection: torch.Tensor, graph: int
) -> AWRStructure:
    """Write the rewired edges back into the adjacency they belong to."""
    positions = selection.nonzero()
    if positions.numel() == 0:
        return structure
    adjacency = structure.graph_adjacency
    if structure.edge_source.dim() == 1:
        rows = structure.edge_source[positions[:, 0]]
        columns = structure.edge_target[positions[:, 0]]
        adjacency[graph, rows, columns] = 1.0
        return structure
    scenes = positions[:, 0]
    index = positions[:, 1]
    rows = structure.edge_source[scenes, index]
    columns = structure.edge_target[scenes, index]
    adjacency[scenes, graph, rows, columns] = 1.0
    return structure
