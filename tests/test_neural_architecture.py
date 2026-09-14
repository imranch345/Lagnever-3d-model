"""Architecture design checks: latent layout, graph heads, programs, geometry, editing."""

from __future__ import annotations

import pytest

from awr.config import DomainConfig
from awr.errors import ContractError, NotYetImplementedError
from awr.ontology import AnatomyOntology
from awr.scene import AWRScene
from editing.scene_editor import SceneEditor
from generation.neural.anatomical_encoder import (
    SUBSPACE_ROUTING,
    AnatomicalReasoningEncoder,
    validate_routing,
)
from generation.neural.contracts import dim
from generation.neural.editing import EditKind, plan_edit
from generation.neural.features import AWRFeatureExtractor
from generation.neural.graph_encoder import (
    AnatomicalGraphEncoder,
    GraphEncoderSpec,
    HeadAllocation,
    HeadRole,
    build_attention_mask,
    relation_bias_indices,
)
from generation.neural.language_encoder import (
    AWRProgram,
    ProgramOp,
    ProgramOpType,
    ProgramTranscoder,
    TextToProgramModel,
)
from generation.neural.latent import ENTITY_LATENT_LAYOUT, LATENT_TIERS
from generation.neural.multimodal import (
    ALIGNMENT_METHODS,
    RECOMMENDED_ALIGNMENT,
    OntologyNegativeSampler,
)
from generation.neural.three_d_latent import (
    CANDIDATE_REPRESENTATIONS,
    RECOMMENDED_REPRESENTATION,
    Criterion,
    GeometryTokenSpec,
    rank_candidates,
    tokens_for_lod,
    validate_token_schedule,
)

# --- anatomical latent -------------------------------------------------------


def test_entity_subspaces_fill_the_declared_width() -> None:
    """The subspace layout and the declared entity width are one decision."""
    assert ENTITY_LATENT_LAYOUT.width == dim("D_ENT").size
    ENTITY_LATENT_LAYOUT.validate_against()


def test_identity_subspace_is_write_protected() -> None:
    """Identity preservation is a property of the layout, not of training."""
    assert ENTITY_LATENT_LAYOUT.protected_spans() == ((0, 96),)
    identity = next(s for s in ENTITY_LATENT_LAYOUT.subspaces if s.name == "identity")
    assert identity.write_protected is True
    assert ENTITY_LATENT_LAYOUT.context_width() == 160


def test_subspace_spans_do_not_overlap() -> None:
    """Named subspaces tile the latent exactly once."""
    covered: list[int] = []
    for subspace in ENTITY_LATENT_LAYOUT.subspaces:
        start, end = ENTITY_LATENT_LAYOUT.span(subspace.name)
        covered.extend(range(start, end))
    assert covered == list(range(ENTITY_LATENT_LAYOUT.width))


def test_every_subspace_states_why_it_exists() -> None:
    """A subspace without a rationale is capacity nobody can defend."""
    for subspace in ENTITY_LATENT_LAYOUT.subspaces:
        assert len(subspace.rationale) > 40, subspace.name


def test_latent_tiers_record_the_merge_alternative() -> None:
    """Each tier says what would be lost by folding it into another."""
    assert len(LATENT_TIERS) == 4
    for tier in LATENT_TIERS:
        assert tier.alternative_if_merged.strip()


def test_state_is_not_part_of_the_entity_latent() -> None:
    """Presentation state must not be able to alter identity."""
    names = {subspace.name for subspace in ENTITY_LATENT_LAYOUT.subspaces}
    assert "state" not in names
    assert "visibility" not in names
    assert "opacity" not in names


def test_subspace_routing_is_complete_and_valid() -> None:
    """Every subspace declares its source and every source exists."""
    validate_routing()
    assert set(SUBSPACE_ROUTING) == {s.name for s in ENTITY_LATENT_LAYOUT.subspaces}
    assert SUBSPACE_ROUTING["identity"] == ("entity_ids",)


# --- graph encoder -----------------------------------------------------------


def test_head_allocation_matches_the_declared_head_count() -> None:
    """Heads are partitioned across the graphs without leftovers."""
    allocation = HeadAllocation.default()
    allocation.validate_against()
    assert allocation.total == dim("N_HEAD").size
    assert set(allocation.roles()) == {
        HeadRole.STRUCTURE,
        HeadRole.SPATIAL,
        HeadRole.FUNCTIONAL,
        HeadRole.GLOBAL,
    }


def test_graph_heads_cannot_see_other_graphs(
    ontology: AnatomyOntology, scene: AWRScene
) -> None:
    """A functional head sees flow edges and not adjacency, which is the core requirement."""
    extractor = AWRFeatureExtractor(ontology)
    payload = extractor.encode_batch([scene])
    allocation = HeadAllocation.default()
    mask = build_attention_mask(payload, extractor, allocation)
    roles = allocation.roles()

    ids = list(scene.ids())
    lv, aorta = ids.index("heart.left_ventricle"), ids.index("heart.aorta")
    septum = ids.index("heart.interventricular_septum")
    functional = roles.index(HeadRole.FUNCTIONAL)
    spatial = roles.index(HeadRole.SPATIAL)

    assert mask[functional][lv][aorta] is True
    assert mask[functional][lv][septum] is False
    assert mask[spatial][lv][septum] is True
    assert mask[spatial][lv][aorta] is False


def test_global_heads_see_everything_real(
    ontology: AnatomyOntology, scene: AWRScene
) -> None:
    """Global heads are unmasked over real entities and closed over padding."""
    extractor = AWRFeatureExtractor(ontology)
    payload = extractor.encode_batch([scene])
    allocation = HeadAllocation.default()
    mask = build_attention_mask(payload, extractor, allocation)
    head = allocation.roles().index(HeadRole.GLOBAL)
    assert all(mask[head][0][j] for j in range(len(scene.ids())))
    assert not any(mask[head][63])
    assert not any(row[63] for row in mask[head])


def test_relation_bias_keeps_multiple_relations_on_one_pair(
    ontology: AnatomyOntology, scene: AWRScene
) -> None:
    """Two structures that both adjoin and connect keep both relations."""
    extractor = AWRFeatureExtractor(ontology)
    payload = extractor.encode_batch([scene])
    index = relation_bias_indices(payload)
    assert sum(len(v) for v in index.values()) == len(scene.relationships.all_edges())
    assert all(len(v) >= 1 for v in index.values())


def test_graph_encoder_spec_requires_divisible_width() -> None:
    """A width that does not divide among heads is a configuration error."""
    with pytest.raises(ContractError, match="divide evenly"):
        GraphEncoderSpec(entity_width=100)


def test_graph_encoder_is_declared_only() -> None:
    """No graph encoder exists yet."""
    with pytest.raises(NotYetImplementedError, match="Step 6"):
        AnatomicalGraphEncoder()


# --- language control path ---------------------------------------------------


def test_programs_use_closed_entity_arguments(
    ontology: AnatomyOntology, config: DomainConfig
) -> None:
    """Entity arguments are codebook ids, so anatomy cannot be invented."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    program = transcoder.transcode("make the left ventricle transparent")
    assert program.ops[0].op_type is ProgramOpType.SET_OPACITY
    assert program.ops[0].entity_args == ("heart.left_ventricle",)
    assert program.ops[0].scalars["opacity"] == pytest.approx(0.3)


def test_group_references_stay_as_groups(
    ontology: AnatomyOntology, config: DomainConfig
) -> None:
    """A program names the group, and the AWR engine expands it."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    program = transcoder.transcode("show the four chambers")
    assert program.ops[0].entity_args == ("heart.chambers",)


def test_programs_round_trip_through_tensors(
    ontology: AnatomyOntology, config: DomainConfig
) -> None:
    """Encoding and decoding a program is lossless."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    texts = (
        "generate a human heart",
        "hide everything except the chambers",
        "make the left ventricle transparent",
        "switch to medical level",
        "set lod 3",
    )
    programs = [transcoder.transcode(text) for text in texts]
    payload = transcoder.encode(programs)
    for index, original in enumerate(programs):
        restored = transcoder.decode(payload, index)
        assert [op.op_type for op in restored.ops] == [op.op_type for op in original.ops]
        assert [op.entity_args for op in restored.ops] == [op.entity_args for op in original.ops]


def test_unparsed_text_never_becomes_a_program(
    ontology: AnatomyOntology, config: DomainConfig
) -> None:
    """An utterance the parser rejects cannot be turned into an action."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    with pytest.raises(ContractError, match="recognised no intent"):
        transcoder.transcode("please make it prettier")


def test_program_argument_limit_is_enforced() -> None:
    """Argument slots are declared, and exceeding them is an error."""
    with pytest.raises(ContractError, match="entity arguments exceed"):
        ProgramOp(ProgramOpType.SHOW, tuple(f"heart.e{i}" for i in range(9)))


def test_program_length_limit_is_enforced() -> None:
    """Program length is bounded by the declared slot count."""
    op = ProgramOp(ProgramOpType.SHOW, ("heart",))
    with pytest.raises(ContractError, match="more than the 16 declared slots"):
        AWRProgram(tuple([op] * 17))


def test_every_step4_intent_has_a_program_operation(
    ontology: AnatomyOntology, config: DomainConfig
) -> None:
    """The control path covers the whole deterministic command vocabulary."""
    from reasoning.parser import RuleBasedCommandParser

    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    for utterance in RuleBasedCommandParser.supported_forms():
        program = transcoder.transcode(utterance)
        assert len(program) == 1, utterance


def test_learned_parser_is_declared_only() -> None:
    """The learned control path does not exist yet."""
    with pytest.raises(NotYetImplementedError, match="Step 6"):
        TextToProgramModel()


# --- 3D representation -------------------------------------------------------


def test_candidate_comparison_is_complete() -> None:
    """Every candidate is scored on every criterion and argued both ways."""
    assert len(CANDIDATE_REPRESENTATIONS) >= 7
    for candidate in CANDIDATE_REPRESENTATIONS:
        assert set(candidate.scores) == set(Criterion)
        assert candidate.advantages
        assert candidate.disadvantages


def test_recommendation_follows_from_the_declared_weights() -> None:
    """The recommendation is the top-ranked candidate under the stated priorities."""
    assert rank_candidates()[0][0] == RECOMMENDED_REPRESENTATION


def test_changing_the_weights_can_change_the_recommendation() -> None:
    """The ranking is a consequence of priorities, not an objective fact."""
    quality_only = dict.fromkeys(Criterion, 0.0)
    quality_only[Criterion.DECODING_QUALITY] = 1.0
    quality_only[Criterion.MULTI_VIEW_CONSISTENCY] = 1.0
    assert rank_candidates(quality_only)[0][0] != RECOMMENDED_REPRESENTATION


def test_lod_token_schedule_is_nested_and_monotone() -> None:
    """More detail reads more tokens, never different ones."""
    validate_token_schedule()
    counts = [tokens_for_lod(level) for level in range(5)]
    assert counts == sorted(counts)
    assert counts[-1] == dim("K_GEO").size
    with pytest.raises(ContractError, match="non-decreasing"):
        validate_token_schedule((8, 4))


def test_lod_outside_the_schedule_is_rejected() -> None:
    """An undefined level of detail is an error, not a clamp."""
    with pytest.raises(ContractError, match="outside the token schedule"):
        tokens_for_lod(9)


def test_geometry_token_spec_rejects_unsupported_fields() -> None:
    """The field kind is a declared choice, not a free string."""
    with pytest.raises(ContractError, match="Unsupported field kind"):
        GeometryTokenSpec(field_kind="gaussians")


# --- multimodal alignment ----------------------------------------------------


def test_alignment_records_a_rejected_alternative() -> None:
    """The recommendation is stated against explicit alternatives."""
    verdicts = {method.name: method.verdict for method in ALIGNMENT_METHODS}
    assert verdicts["prototype_hub"].startswith("RECOMMENDED")
    assert verdicts["pairwise_contrastive"].startswith("REJECTED")
    assert RECOMMENDED_ALIGNMENT == "prototype_hub"


def test_hard_negatives_are_anatomically_informative(ontology: AnatomyOntology) -> None:
    """The contralateral structure is the first negative, not a random one."""
    sampler = OntologyNegativeSampler(ontology)
    negatives = sampler.hard_negatives("heart.left_ventricle")
    assert "heart.right_ventricle" in negatives
    assert "heart.left_atrium" in negatives
    assert all(not ontology.get(entity).is_group for entity in negatives)


def test_hard_negatives_exclude_the_entity_itself(ontology: AnatomyOntology) -> None:
    """An entity is never its own negative."""
    sampler = OntologyNegativeSampler(ontology)
    for entity in ontology.iter_entities():
        if entity.renderable:
            assert entity.entity_id not in sampler.hard_negatives(entity.entity_id)


def test_most_entities_have_hard_negatives(ontology: AnatomyOntology) -> None:
    """Coverage is good enough for the objective to be meaningful."""
    coverage = OntologyNegativeSampler(ontology).coverage()
    assert sum(1 for count in coverage.values() if count >= 3) >= 30


# --- editing -----------------------------------------------------------------


def test_state_edits_recompute_no_geometry(
    ontology: AnatomyOntology, config: DomainConfig, scene: AWRScene
) -> None:
    """The central editing claim, in testable form."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    for utterance in (
        "make the left ventricle transparent",
        "hide everything except the chambers",
        "show the valves",
        "animate blood flow",
    ):
        plan = plan_edit(transcoder.transcode(utterance), scene)
        assert plan.kind is EditKind.STATE_ONLY, utterance
        assert plan.recompute_entities == (), utterance
        assert plan.recompute_fraction == 0.0
        assert len(plan.frozen_entities) == len(scene.renderable_ids())


def test_lazy_decoding_is_reported_rather_than_hidden(
    ontology: AnatomyOntology, config: DomainConfig, scene: AWRScene
) -> None:
    """With nothing cached, showing a structure does require decoding it, and says so."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    plan = plan_edit(transcoder.transcode("show the valves"), scene, decoded_entities=())
    assert plan.kind is EditKind.STATE_ONLY
    assert len(plan.recompute_entities) == 4
    assert plan.notes


def test_level_of_detail_change_is_a_refinement_not_a_regeneration(
    ontology: AnatomyOntology, config: DomainConfig, scene: AWRScene
) -> None:
    """Increasing detail refines existing entities and keeps their identity."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    plan = plan_edit(transcoder.transcode("show more detail"), scene, target_lod=2)
    assert plan.kind is EditKind.LOD_REFINEMENT
    assert plan.requires_full_regeneration is False
    assert 0.0 < plan.recompute_fraction < 1.0
    assert any("identity is unchanged" in guarantee for guarantee in plan.guarantees)


def test_generation_is_the_only_full_rebuild(
    ontology: AnatomyOntology, config: DomainConfig, scene: AWRScene
) -> None:
    """Only an explicit generate request rebuilds everything."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    plan = plan_edit(transcoder.transcode("generate a human heart"), scene)
    assert plan.requires_full_regeneration is True
    assert plan.recompute_fraction == 1.0


def test_edit_plan_matches_what_step4_actually_changed(
    ontology: AnatomyOntology, config: DomainConfig, scene: AWRScene
) -> None:
    """The plan's touched set agrees with the deterministic engine's deltas."""
    transcoder = ProgramTranscoder(ontology, opacity_presets=config.opacity_presets.values)
    plan = plan_edit(transcoder.transcode("make the left ventricle transparent"), scene)
    applied = SceneEditor(scene).set_opacity(["heart.left_ventricle"], 0.3)
    assert set(applied.result.affected) == set(plan.touched_entities)


def test_local_editor_is_declared_only() -> None:
    """No learned editor exists yet, and its guarantees are written down."""
    from generation.neural.editing import LocalGeometryEditor

    with pytest.raises(NotYetImplementedError, match="Step 6"):
        LocalGeometryEditor()
    assert any("bit-identical" in rule for rule in LocalGeometryEditor.locality_guarantee())


def test_anatomical_encoder_is_declared_only() -> None:
    """The encoder is an interface with stated invariants."""
    with pytest.raises(NotYetImplementedError, match="Step 6"):
        AnatomicalReasoningEncoder()
    assert any(
        "identity subspace" in invariant
        for invariant in AnatomicalReasoningEncoder.invariants()
    )
