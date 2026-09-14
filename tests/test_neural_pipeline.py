"""Pipeline composition, the end-to-end shape trace, scales and configuration."""

from __future__ import annotations

import pytest

from awr.errors import ConfigError, ContractError, NotYetImplementedError
from awr.ontology import AnatomyOntology
from awr.scene import AWRScene
from generation.neural.config import available_neural_configs, load_neural_config
from generation.neural.curriculum import (
    CURRICULUM,
    stages_without_external_data,
    validate_curriculum,
)
from generation.neural.features import AWRFeatureExtractor
from generation.neural.losses import (
    LOSSES,
    LossRole,
    evaluation_only_losses,
    first_experiment_losses,
    initial_loss_strategy,
    validate_strategy,
)
from generation.neural.model import LAGNAV_PIPELINE, LagnavArchitecture, trace_pipeline
from generation.neural.scales import (
    SCALES,
    estimate_parameters,
    estimate_training_flops,
    scale,
    validate_prototype_dimensions,
)

# --- pipeline ----------------------------------------------------------------


def test_pipeline_composes(ontology: AnatomyOntology) -> None:
    """Every stage's inputs are produced upstream or supplied with the batch."""
    from generation.neural.features import AWR_BATCH_CONTRACT
    from generation.neural.model import DEFAULT_SUPPLIED_TENSORS

    LagnavArchitecture().validate(
        supplied=(*DEFAULT_SUPPLIED_TENSORS, *AWR_BATCH_CONTRACT.names())
    )
    assert len(LAGNAV_PIPELINE) == 8


def test_pipeline_with_a_missing_producer_is_rejected() -> None:
    """A stage consuming a tensor nobody emits fails the composition check."""
    with pytest.raises(ContractError, match="which no earlier stage produces"):
        LagnavArchitecture().validate(supplied=())


def test_shape_trace_runs_end_to_end(ontology: AnatomyOntology, scene: AWRScene) -> None:
    """The architecture's smoke test: a real scene in, consistent shapes throughout."""
    report = trace_pipeline(scene, ontology, batch_size=3, query_points=512)

    assert report["bindings"]["B"] == 3
    assert report["bindings"]["P_QUERY"] == 512
    stages = report["stages"]
    assert stages["awr_features"]["entity_ids"] == (3, 64)
    assert stages["graph_encoder"]["z_entity_out"] == (3, 64, 256)
    assert stages["anatomical_encoder"]["z_scene"] == (3, 256)
    assert stages["geometry_tokeniser"]["z_geometry"] == (3, 64, 32, 64)
    assert stages["field_decoder"]["entity_occupancy_logits"] == (3, 64, 512)
    assert stages["field_decoder"]["part_logits"] == (3, 512, 65)
    assert stages["alignment_heads"]["entity_prototypes"] == (42, 256)


def test_shape_trace_reports_what_is_real_and_what_is_declared(
    ontology: AnatomyOntology, scene: AWRScene
) -> None:
    """Only the feature stage actually ran; the report says so."""
    report = trace_pipeline(scene, ontology)
    summary = report["summary"]
    assert summary["implemented"] == ["awr_features"]
    assert len(summary["declared"]) == 7
    assert summary["status"] == "PROPOSED"
    assert report["real_tensors"]["entity_ids"] == (2, 64)


def test_geometry_axis_is_the_entity_axis(ontology: AnatomyOntology, scene: AWRScene) -> None:
    """Correspondence is a tensor axis, which the traced shapes show directly."""
    report = trace_pipeline(scene, ontology, batch_size=1)
    geometry = report["stages"]["geometry_tokeniser"]["z_geometry"]
    entities = report["stages"]["awr_features"]["entity_ids"]
    assert geometry[1] == entities[1]


def test_part_logits_cover_every_entity_plus_background(
    ontology: AnatomyOntology, scene: AWRScene
) -> None:
    """Every point can be attributed to an entity or to nothing."""
    report = trace_pipeline(scene, ontology, batch_size=1, query_points=64)
    part_logits = report["stages"]["field_decoder"]["part_logits"]
    entity_slots = report["stages"]["awr_features"]["entity_ids"][1]
    assert part_logits[-1] == entity_slots + 1


# --- scales ------------------------------------------------------------------


def test_prototype_scale_matches_the_declared_dimensions() -> None:
    """The dimension registry and the prototype scale cannot drift apart."""
    validate_prototype_dimensions()


def test_parameter_estimates_are_ordered_by_scale(ontology: AnatomyOntology) -> None:
    """Prototype, research and production are separated by orders of magnitude."""
    sizes = AWRFeatureExtractor(ontology).vocabulary_sizes()
    totals = [estimate_parameters(spec, sizes)["total"] for spec in SCALES]
    assert totals == sorted(totals)
    assert 5e6 < totals[0] < 25e6
    assert 100e6 < totals[1] < 500e6
    assert totals[2] > 1e9


def test_parameter_breakdown_is_complete(ontology: AnatomyOntology) -> None:
    """The estimate names where the parameters go, not just how many there are."""
    sizes = AWRFeatureExtractor(ontology).vocabulary_sizes()
    breakdown = estimate_parameters(scale("prototype"), sizes)
    expected = {
        "embeddings",
        "graph_encoder",
        "language",
        "geometry_tokeniser",
        "field_decoder",
        "frame_head",
        "alignment",
    }
    assert expected <= set(breakdown)
    assert breakdown["total"] == sum(v for k, v in breakdown.items() if k != "total")


def test_prototype_fits_one_gpu_by_design() -> None:
    """The first experiment is meant to be cheap enough to repeat."""
    compute = scale("prototype").compute
    assert compute.gpu_count == (1, 1)
    assert "estimate" in compute.confidence


def test_flops_estimate_scales_with_work() -> None:
    """The estimator is monotone in dataset size and epochs."""
    spec = scale("prototype")
    small = estimate_training_flops(spec, scenes=1000, epochs=10)
    large = estimate_training_flops(spec, scenes=1000, epochs=20)
    assert large == pytest.approx(2 * small)
    with pytest.raises(ContractError):
        estimate_training_flops(spec, scenes=0, epochs=1)


# --- objectives and curriculum ----------------------------------------------


def test_every_objective_declares_its_data_and_its_failure_mode() -> None:
    """A loss without a stated data requirement is a plan without a schedule."""
    for spec in LOSSES:
        assert spec.teaches.strip()
        assert spec.requires.strip()
        assert spec.prevents.strip()
        assert spec.stages


def test_relationship_objectives_are_held_out_of_training() -> None:
    """The hypothesis is judged on relationships, so they are not trained on."""
    held_out = {spec.name for spec in evaluation_only_losses()}
    assert held_out == {
        "structural_relationship",
        "spatial_relationship",
        "functional_relationship",
    }
    for name in held_out:
        assert name not in initial_loss_strategy()


def test_training_on_a_held_out_objective_is_refused() -> None:
    """The methodological rule is enforced, not just written down."""
    with pytest.raises(ContractError, match="untestable"):
        validate_strategy({"spatial_relationship": 1.0})


def test_unknown_objective_in_a_strategy_is_refused() -> None:
    """Loss configuration cannot name something that does not exist."""
    with pytest.raises(ContractError, match="undeclared objectives"):
        validate_strategy({"make_it_prettier": 1.0})


def test_first_experiment_objectives_have_weights() -> None:
    """Everything trained has a proposed weight."""
    for spec in first_experiment_losses():
        assert spec.role_in_first_experiment is LossRole.TRAIN
        assert spec.proposed_weight and spec.proposed_weight > 0


def test_curriculum_is_consistent_and_ordered() -> None:
    """Stage dependencies point backwards and objectives agree with stages."""
    validate_curriculum()
    assert [stage.key for stage in CURRICULUM] == ["S0", "S1", "S2", "S3", "S4", "S5"]


def test_first_two_stages_need_no_external_data() -> None:
    """Work can start before any dataset or licence decision."""
    assert stages_without_external_data() == ("S0", "S1")


def test_every_stage_states_how_it_ends() -> None:
    """A stage without exit criteria is a stage that never finishes."""
    for stage_spec in CURRICULUM:
        assert stage_spec.exit_criteria
        assert stage_spec.goal.strip()


# --- configuration -----------------------------------------------------------


def test_shipped_configurations_load_and_validate() -> None:
    """Every configuration in the repository is coherent."""
    names = available_neural_configs()
    assert {"prototype", "research", "baseline"} <= set(names)
    for name in names:
        config = load_neural_config(name)
        assert config.is_proposed
        config.validate()


def test_configuration_references_a_scale_rather_than_repeating_it() -> None:
    """Architecture widths live in code; configuration names them."""
    config = load_neural_config("prototype")
    assert config.scale.name == "prototype"
    assert config.scale.entity_width == 256


def test_baseline_arm_declares_what_it_skips() -> None:
    """The difference between the two arms is explicit and auditable."""
    baseline = load_neural_config("baseline")
    assert baseline.arm == "appearance_baseline"
    assert set(baseline.skipped_stages) == {"S0", "S1"}
    assert all(reason.strip() for reason in baseline.skipped_stages.values())
    assert baseline.architecture_overrides["entity_axis"] is False
    assert baseline.architecture_overrides["graph_encoder"] is False


def test_both_arms_share_the_experiment_and_the_budget() -> None:
    """A comparison with different budgets would measure the budget."""
    lagnav = load_neural_config("prototype")
    baseline = load_neural_config("baseline")
    assert lagnav.experiment_id == baseline.experiment_id
    assert lagnav.optimization.max_steps == baseline.optimization.max_steps
    assert lagnav.data.scenes == baseline.data.scenes
    assert lagnav.optimization.seed == baseline.optimization.seed


def test_configuration_rejects_an_undeclared_loss(tmp_path) -> None:
    """A configuration cannot invent an objective."""
    from generation.neural.config import NeuralExperimentConfig

    with pytest.raises(ContractError, match="undeclared objectives"):
        NeuralExperimentConfig.from_mapping(
            {
                "scale": "prototype",
                "curriculum": {"stages": ["S0"]},
                "losses": {"vibes": 1.0},
            }
        )


def test_configuration_rejects_out_of_order_stages() -> None:
    """Stages must run in dependency order."""
    from generation.neural.config import NeuralExperimentConfig

    with pytest.raises(ConfigError, match="out of dependency order"):
        NeuralExperimentConfig.from_mapping(
            {"scale": "prototype", "curriculum": {"stages": ["S3", "S0"]}, "losses": {}}
        )


def test_synthetic_data_cannot_claim_licensed_assets() -> None:
    """Provenance stories are kept distinct."""
    from generation.neural.config import DataConfig

    with pytest.raises(ConfigError, match="licensed assets"):
        DataConfig(tier="synthetic", licensed_assets=True)


def test_declared_components_all_refuse_to_run() -> None:
    """Nothing in the proposed stack can be accidentally instantiated."""
    from generation.neural.graph_encoder import AnatomicalGraphEncoder
    from generation.neural.language_encoder import (
        ConditioningLanguageEncoder,
        TextToProgramModel,
    )
    from generation.neural.multimodal import ModalityEncoder
    from generation.neural.three_d_latent import PerEntityFieldDecoder

    for component in (
        AnatomicalGraphEncoder,
        TextToProgramModel,
        ConditioningLanguageEncoder,
        PerEntityFieldDecoder,
        ModalityEncoder,
    ):
        with pytest.raises(NotYetImplementedError):
            component()
