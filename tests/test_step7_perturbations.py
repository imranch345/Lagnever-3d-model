"""Step 7 Experiments 1, 2 and 3: the perturbation harness itself.

These tests check that a perturbation does what its name says and nothing else. A
perturbation that quietly alters entities or points would make any measured change
uninterpretable, and a perturbation that alters nothing would make a model look robust
when the harness was simply inert.
"""

from __future__ import annotations

import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.corpus import load_split
from experiments.step7.perturbations import (
    AWR_CASES,
    GRAPH_INDEX,
    RELATION_PERTURBATIONS,
    describe,
    perturb_relations,
    restrict_graphs,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder

CORPUS = "datasets/processed/whole_organ_1600_v2"


@pytest.fixture(scope="module")
def setup():
    """A real batch, the builder that made it, and the inverse-relation table."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    scenes = [s for s in load_split(CORPUS, "test") if s.active_lod == 3][:4]
    return builder.build(scenes).batch, builder.inverse_relation_table()


def _live(structure, graph: str) -> int:
    return int(((structure.edge_graph == GRAPH_INDEX[graph]) & structure.edge_mask).sum())


def test_every_typed_graph_starts_non_empty(setup) -> None:
    """A perturbation on an empty graph measures nothing."""
    batch, _ = setup
    for graph in GRAPH_INDEX:
        assert _live(batch.structure, graph) > 0, f"{graph} graph is empty before perturbation"


def test_dropping_a_graph_removes_exactly_that_graph(setup) -> None:
    """Collateral damage to another graph would misattribute the effect."""
    batch, inverse = setup
    for graph, kind in (
        ("spatial", "drop_spatial"),
        ("functional", "drop_functional"),
        ("structure", "drop_structure"),
    ):
        out = perturb_relations(batch, kind, inverse_relations=inverse)
        assert _live(out.structure, graph) == 0, f"{kind} left {graph} edges behind"
        for other in GRAPH_INDEX:
            if other == graph:
                continue
            assert _live(out.structure, other) == _live(batch.structure, other), (
                f"{kind} also changed the {other} graph"
            )


def test_perturbations_never_touch_entities_or_points(setup) -> None:
    """Any measured change must have arrived through the relationship graph."""
    batch, inverse = setup
    for kind in RELATION_PERTURBATIONS:
        generator = torch.Generator().manual_seed(7)
        out = perturb_relations(batch, kind, inverse_relations=inverse, generator=generator)
        assert torch.equal(out.scene_points, batch.scene_points), kind
        assert torch.equal(out.entity_points, batch.entity_points), kind
        assert torch.equal(out.text_features, batch.text_features), kind
        assert torch.equal(out.entity_present, batch.entity_present), kind
        assert torch.equal(out.entity_frames, batch.entity_frames), kind
        assert torch.equal(out.part_owner, batch.part_owner), kind
        assert torch.equal(out.structure.entity_ids, batch.structure.entity_ids), kind


def test_perturbations_do_not_modify_the_original(setup) -> None:
    """The control must stay the control across a whole sweep."""
    batch, inverse = setup
    before = {graph: _live(batch.structure, graph) for graph in GRAPH_INDEX}
    adjacency = batch.structure.graph_adjacency.clone()
    relations = batch.structure.edge_relation.clone()
    for kind in RELATION_PERTURBATIONS:
        generator = torch.Generator().manual_seed(3)
        perturb_relations(batch, kind, inverse_relations=inverse, generator=generator)
    assert {graph: _live(batch.structure, graph) for graph in GRAPH_INDEX} == before
    assert torch.equal(batch.structure.graph_adjacency, adjacency)
    assert torch.equal(batch.structure.edge_relation, relations)


def test_inverting_spatial_relations_changes_them(setup) -> None:
    """An inversion that leaves the relations alone is not an inversion."""
    batch, inverse = setup
    out = perturb_relations(batch, "invert_spatial", inverse_relations=inverse)
    spatial = (batch.structure.edge_graph == GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    changed = (out.structure.edge_relation != batch.structure.edge_relation) & spatial
    assert int(changed.sum()) > 0, "no spatial relation was inverted"
    untouched = (batch.structure.edge_graph != GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    assert torch.equal(
        out.structure.edge_relation[untouched], batch.structure.edge_relation[untouched]
    )


def test_inverting_also_swaps_the_endpoints(setup) -> None:
    """The inverse of a relation is the same edge read backwards.

    Inverting the label without swapping the endpoints asserts something the ontology
    forbids, which is a different corruption than the one being measured.
    """
    batch, inverse = setup
    out = perturb_relations(batch, "invert_spatial", inverse_relations=inverse)
    spatial = (batch.structure.edge_graph == GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    flipped = spatial & (out.structure.edge_relation != batch.structure.edge_relation)
    assert torch.equal(out.structure.edge_source[flipped], batch.structure.edge_target[flipped])
    assert torch.equal(out.structure.edge_target[flipped], batch.structure.edge_source[flipped])


def test_shuffling_types_keeps_the_multiset_of_relations(setup) -> None:
    """Shuffling must permute labels, not invent or lose them."""
    batch, inverse = setup
    generator = torch.Generator().manual_seed(11)
    out = perturb_relations(
        batch, "shuffle_spatial_types", inverse_relations=inverse, generator=generator
    )
    spatial = (batch.structure.edge_graph == GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    before = sorted(batch.structure.edge_relation[spatial].tolist())
    after = sorted(out.structure.edge_relation[spatial].tolist())
    assert before == after


def test_randomising_endpoints_keeps_the_edge_count(setup) -> None:
    """Rewiring must move edges, not delete them."""
    batch, inverse = setup
    generator = torch.Generator().manual_seed(5)
    out = perturb_relations(
        batch, "randomise_spatial_endpoints", inverse_relations=inverse, generator=generator
    )
    assert _live(out.structure, "spatial") == _live(batch.structure, "spatial")
    spatial = (batch.structure.edge_graph == GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    moved = (out.structure.edge_source != batch.structure.edge_source) & spatial
    assert int(moved.sum()) > 0, "no edge was rewired"


def test_randomising_endpoints_needs_a_generator(setup) -> None:
    """A randomising perturbation without a seed would be unreproducible."""
    batch, inverse = setup
    with pytest.raises(ValueError, match="seeded generator"):
        perturb_relations(batch, "randomise_spatial_endpoints", inverse_relations=inverse)


def test_partial_representation_cases_keep_what_they_name(setup) -> None:
    """Each case must retain exactly the graphs it declares."""
    batch, _ = setup
    for case, keep in AWR_CASES.items():
        out = restrict_graphs(batch, keep)
        for graph in GRAPH_INDEX:
            live = _live(out.structure, graph)
            if graph in keep:
                assert live > 0, f"{case} lost the {graph} graph it declared"
            else:
                assert live == 0, f"{case} kept {graph} edges it did not declare"


def test_entities_only_case_removes_every_edge(setup) -> None:
    """The floor case must really be the floor."""
    batch, _ = setup
    out = restrict_graphs(batch, AWR_CASES["E_entities_only"])
    assert int(out.structure.edge_mask.sum()) == 0
    assert float(out.structure.graph_adjacency.sum()) == 0.0


def test_unknown_names_are_rejected(setup) -> None:
    """A typo must fail loudly rather than silently evaluate the control."""
    batch, inverse = setup
    with pytest.raises(ValueError, match="Unknown perturbation"):
        perturb_relations(batch, "drop_everything", inverse_relations=inverse)
    with pytest.raises(ValueError, match="Unknown graphs"):
        restrict_graphs(batch, ("spatial", "temporal"))


def test_every_declared_name_has_a_description() -> None:
    """The report prints these; a missing one would raise while writing results."""
    for name in (*RELATION_PERTURBATIONS, *AWR_CASES):
        assert describe(name)


def test_arrangement_disagreement_is_zero_for_identical_labels() -> None:
    """A model producing one organ regardless must score zero, not something small."""
    import numpy as np

    from experiments.step7.arrangement_response import _disagreement

    same = np.array([0, 1, 2, 3, 4])
    assert _disagreement({name: same for name in ("a", "b", "c", "d")}) == 0.0


def test_arrangement_disagreement_grows_with_difference() -> None:
    """The measure must order a small change below a large one."""
    import numpy as np

    from experiments.step7.arrangement_response import _disagreement

    base = np.arange(10)
    small = base.copy()
    small[0] = 99
    large = base[::-1].copy()
    assert _disagreement({"a": base, "b": small}) < _disagreement({"a": base, "b": large})


def test_arrangement_disagreement_of_one_arrangement_is_zero() -> None:
    """With nothing to compare against there is no disagreement to report."""
    import numpy as np

    from experiments.step7.arrangement_response import _disagreement

    assert _disagreement({"only": np.arange(4)}) == 0.0


def test_label_only_inversion_changes_what_the_graph_says(setup) -> None:
    """Endpoints stay, relations flip, so the graph now asserts the opposite."""
    batch, inverse = setup
    out = perturb_relations(batch, "invert_spatial_labels", inverse_relations=inverse)
    assert torch.equal(out.structure.edge_source, batch.structure.edge_source)
    assert torch.equal(out.structure.edge_target, batch.structure.edge_target)
    spatial = (batch.structure.edge_graph == GRAPH_INDEX["spatial"]) & batch.structure.edge_mask
    changed = (out.structure.edge_relation != batch.structure.edge_relation) & spatial
    assert int(changed.sum()) > 0, "no relation label was flipped"


def test_endpoint_swapping_inversion_is_a_structural_no_op(setup) -> None:
    """Documents why `invert_spatial` cannot test relation sensitivity.

    The encoder adds the inverse relation's embedding in the reverse direction, so
    (a, left_of, b) and (b, right_of, a) produce a bit-identical relation bias. The
    perturbation is therefore invisible by construction, and a null result from it says
    nothing about whether the model reads relation types. `invert_spatial_labels` is the
    perturbation that tests that.
    """
    from awr.config import load_domain_config
    from awr.ontology import load_ontology
    from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
    from training.loop import build_model

    batch, inverse = setup
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    model, _ = build_model("A3", builder, text_features=int(batch.text_features.shape[1]))
    encoder = model.graph_encoder
    assert encoder is not None
    # A freshly built encoder may start with zero relation embeddings, which would make
    # every bias identical and the comparison vacuous. Give them values first.
    with torch.no_grad():
        for parameter in encoder.parameters():
            if parameter.dim() >= 2:
                parameter.normal_(0.0, 0.1)
    base = encoder.build_bias(batch.structure, batch.batch_size)
    swapped = perturb_relations(batch, "invert_spatial", inverse_relations=inverse)
    assert torch.allclose(encoder.build_bias(swapped.structure, swapped.batch_size), base)

    labels = perturb_relations(batch, "invert_spatial_labels", inverse_relations=inverse)
    assert not torch.allclose(encoder.build_bias(labels.structure, labels.batch_size), base)
