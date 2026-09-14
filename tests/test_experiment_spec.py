"""The pre-registered experiment, its metrics and its baselines."""

from __future__ import annotations

import pytest

from awr.errors import ContractError, NotYetImplementedError
from evaluation.baselines import (
    BASELINES,
    BaselineAdapter,
    BaselineKind,
    Capability,
    CapabilityMatrix,
    ComparisonAxis,
)
from evaluation.neural_metrics import (
    EXPERIMENT_METRICS,
    computable_today,
    compute,
    metric,
    primary_metrics,
)
from experiments.heart.experiment_001_structured_vs_appearance import EXPERIMENT_001


def test_experiment_is_pre_registered_and_unrun() -> None:
    """The plan exists, the result does not."""
    assert EXPERIMENT_001.status == "PRE-REGISTERED, NOT RUN"
    assert EXPERIMENT_001.results is None
    EXPERIMENT_001.validate()


def test_experiment_has_a_null_hypothesis() -> None:
    """A hypothesis with no null is not a hypothesis."""
    keys = {key for key, _ in EXPERIMENT_001.hypotheses}
    assert "H0" in keys
    assert {"H1a", "H1b", "H1c", "H1d"} <= keys


def test_every_success_criterion_states_how_it_could_fail() -> None:
    """Pre-registration means writing down what would refute the claim."""
    for criterion in EXPERIMENT_001.success_criteria:
        assert criterion.refuted_if.strip()
        assert criterion.threshold.strip()
        metric(criterion.metric_name)


def test_geometry_guard_prevents_buying_semantics_with_quality() -> None:
    """A semantic win that wrecks geometry does not count."""
    guard = next(c for c in EXPERIMENT_001.success_criteria if c.hypothesis == "guard")
    assert guard.metric_name == "chamfer_distance"


def test_arms_are_controlled() -> None:
    """Both arms hold data, steps, seeds and parameters constant."""
    assert len(EXPERIMENT_001.arms) == 2
    for arm in EXPERIMENT_001.arms:
        assert "matched parameters" in arm.controls
        assert "matched data" in arm.controls
    assert any("Three seeds" in control for control in EXPERIMENT_001.controls)


def test_ablations_isolate_each_architectural_claim() -> None:
    """Each design decision has an ablation that could contradict it."""
    keys = {ablation.key for ablation in EXPERIMENT_001.ablations}
    assert {"A0", "A1", "A2", "A3", "A4", "A5", "A6"} == keys
    for ablation in EXPERIMENT_001.ablations:
        assert ablation.isolates.strip()


def test_synthetic_tier_states_its_limits() -> None:
    """The first corpus is not anatomy and the plan says so."""
    tier0 = EXPERIMENT_001.datasets[0]
    assert tier0.licensing == "no licensing exposure"
    assert any("NOT anatomically realistic" in caveat for caveat in tier0.caveats)


def test_real_data_is_not_acquired() -> None:
    """No dataset has been downloaded, and the plan records that."""
    tier1 = EXPERIMENT_001.datasets[1]
    assert "NOT ACQUIRED" in tier1.licensing


def test_risks_include_a_null_result() -> None:
    """The plan admits the hypothesis may not hold."""
    assert any("null result" in risk for risk in EXPERIMENT_001.risks)


def test_experiment_renders() -> None:
    """The plan prints as a readable document."""
    rendered = EXPERIMENT_001.render()
    assert "PRE-REGISTERED, NOT RUN" in rendered
    assert "Pre-registered success criteria" in rendered


def test_undeclared_metric_is_rejected() -> None:
    """An experiment cannot score itself on an undefined metric."""
    with pytest.raises(ContractError, match="Unknown metric"):
        metric("looks_nice")


def test_primary_metrics_cover_control_consistency_and_quality() -> None:
    """The hypothesis is judged on more than one axis."""
    families = {spec.family for spec in primary_metrics()}
    assert len(families) >= 3


def test_metrics_that_need_a_model_cannot_be_computed_yet() -> None:
    """No metric returns a number before the thing it measures exists."""
    assert len(computable_today()) >= 3
    for spec in EXPERIMENT_METRICS:
        with pytest.raises(NotYetImplementedError):
            compute(spec.name)


def test_baselines_separate_controlled_from_reference_comparisons() -> None:
    """The internal ablation and external systems are not the same kind of evidence."""
    kinds = {baseline.name: baseline.kind for baseline in BASELINES}
    assert kinds["appearance_driven_ablation"] is BaselineKind.INTERNAL_ABLATION
    assert kinds["external_text_to_3d"] is BaselineKind.EXTERNAL_SYSTEM
    assert kinds["step4_deterministic"] is BaselineKind.DETERMINISTIC_REFERENCE


def test_external_systems_are_not_integrated() -> None:
    """No external model is wrapped, and its licence is unreviewed."""
    external = next(b for b in BASELINES if b.kind is BaselineKind.EXTERNAL_SYSTEM)
    assert external.integration_status.startswith("not integrated")
    assert "reviewed" in external.licence_status
    assert any("Never part of the Lagnav architecture" in c for c in external.cautions)


def test_unsupported_axes_are_reported_as_unsupported() -> None:
    """A system that cannot do the task does not score zero at it."""
    matrix = CapabilityMatrix()
    assert "external_text_to_3d" in matrix.unsupported(ComparisonAxis.PERSISTENT_EDITING)
    assert (
        matrix.baselines[2].capability(ComparisonAxis.SEMANTIC_PART_CONTROL)
        is Capability.NOT_SUPPORTED
    )
    assert any("never as a score of zero" in rule for rule in BaselineAdapter.reporting_rules())


def test_step4_reference_records_why_its_perfect_scores_prove_nothing() -> None:
    """The deterministic ceiling is labelled as a ceiling, not as a result."""
    reference = next(b for b in BASELINES if b.kind is BaselineKind.DETERMINISTIC_REFERENCE)
    assert any("not evidence" in caution for caution in reference.cautions)


def test_baseline_adapter_is_declared_only() -> None:
    """No baseline is runnable from this repository."""
    with pytest.raises(NotYetImplementedError):
        BaselineAdapter()
