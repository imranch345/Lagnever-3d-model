"""Metrics for the first heart experiment, declared before any model exists.

STATUS: declarations. A metric is defined here together with what it needs, so
that the experiment cannot be scored on whatever happens to be easy to compute
when the time comes.

Three of these metrics can already be computed on the Step 4 deterministic system,
where they are exactly 1.0 by construction. That is useful as a reference ceiling
and useless as evidence: the open question is whether a *learned* system retains
those properties, not whether a symbolic one has them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from awr.errors import ContractError, NotYetImplementedError

__all__ = ["MetricFamily", "MetricSpec", "EXPERIMENT_METRICS", "metric", "primary_metrics"]


class MetricFamily(StrEnum):
    """What a metric is about."""

    SEMANTIC_CONTROL = "semantic_control"
    ANATOMICAL_CONSISTENCY = "anatomical_consistency"
    GEOMETRY_QUALITY = "geometry_quality"
    EFFICIENCY = "efficiency"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One declared metric."""

    name: str
    family: MetricFamily
    definition: str
    requires: str
    higher_is_better: bool
    primary: bool
    computable_today: bool
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.definition.strip():
            raise ContractError(f"Metric {self.name!r} needs a definition.")


EXPERIMENT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="part_control_success",
        family=MetricFamily.SEMANTIC_CONTROL,
        definition=(
            "Share of instructions where the named structure changed as asked and no other "
            "structure changed beyond tolerance."
        ),
        requires="Generated scenes before and after an instruction, with part attribution.",
        higher_is_better=True,
        primary=True,
        computable_today=True,
        notes="1.0 on the Step 4 system by construction; the question is whether a model keeps it.",
    ),
    MetricSpec(
        name="edit_locality",
        family=MetricFamily.SEMANTIC_CONTROL,
        definition=(
            "Mean geometric change on entities the instruction did not name. Zero is the target."
        ),
        requires="Paired before and after geometry per entity.",
        higher_is_better=False,
        primary=True,
        computable_today=True,
        notes="Exactly zero for state edits in the proposed architecture, by construction.",
    ),
    MetricSpec(
        name="identity_persistence",
        family=MetricFamily.SEMANTIC_CONTROL,
        definition="Share of entity ids that survive a sequence of edits unchanged.",
        requires="A multi-turn session with per-entity attribution.",
        higher_is_better=True,
        primary=True,
        computable_today=True,
        notes=(
            "Structural for Lagnav, and the main thing an appearance-driven baseline is "
            "expected to lose. Report the baseline's value, not just ours."
        ),
    ),
    MetricSpec(
        name="structural_relationship_accuracy",
        family=MetricFamily.ANATOMICAL_CONSISTENCY,
        definition="Share of ontology structural relations recoverable from generated geometry.",
        requires="Generated geometry plus a relation extractor.",
        higher_is_better=True,
        primary=True,
        computable_today=False,
        notes="Held out of training in the first experiment so the comparison stays honest.",
    ),
    MetricSpec(
        name="spatial_relationship_accuracy",
        family=MetricFamily.ANATOMICAL_CONSISTENCY,
        definition="Share of spatial relations satisfied by generated entity frames.",
        requires="Generated entity frames or part-labelled geometry.",
        higher_is_better=True,
        primary=True,
        computable_today=False,
    ),
    MetricSpec(
        name="functional_relationship_accuracy",
        family=MetricFamily.ANATOMICAL_CONSISTENCY,
        definition="Share of flow relations whose lumen connectivity is present in geometry.",
        requires="Generated geometry and a connectivity test.",
        higher_is_better=True,
        primary=False,
        computable_today=False,
        notes="The hardest to measure reliably; treat early numbers with suspicion.",
    ),
    MetricSpec(
        name="chamfer_distance",
        family=MetricFamily.GEOMETRY_QUALITY,
        definition="Bidirectional surface distance between generated and reference geometry.",
        requires="Reference geometry.",
        higher_is_better=False,
        primary=True,
        computable_today=False,
        notes=(
            "A guard rail, not a target. Lagnav must not win on semantics by giving up geometry, "
            "so a large regression here invalidates a semantic win."
        ),
    ),
    MetricSpec(
        name="part_segmentation_iou",
        family=MetricFamily.GEOMETRY_QUALITY,
        definition="Intersection over union of predicted and reference part labels.",
        requires="Part-labelled reference geometry.",
        higher_is_better=True,
        primary=True,
        computable_today=False,
    ),
    MetricSpec(
        name="watertight_fraction",
        family=MetricFamily.GEOMETRY_QUALITY,
        definition="Share of extracted meshes that are closed surfaces.",
        requires="Extracted meshes.",
        higher_is_better=True,
        primary=False,
        computable_today=False,
    ),
    MetricSpec(
        name="edit_latency",
        family=MetricFamily.EFFICIENCY,
        definition="Wall-clock time to apply one instruction to an existing scene.",
        requires="A running system.",
        higher_is_better=False,
        primary=False,
        computable_today=True,
        notes=(
            "The practical consequence of locality: a state edit should be microseconds because "
            "it touches no model at all."
        ),
    ),
    MetricSpec(
        name="sample_efficiency",
        family=MetricFamily.EFFICIENCY,
        definition="Metric value as a function of training-set size.",
        requires="Runs at several dataset sizes.",
        higher_is_better=True,
        primary=False,
        computable_today=False,
        notes=(
            "Where structure is most likely to pay off. If it helps anywhere, it should help "
            "most in the small-data regime."
        ),
    ),
)
"""Metrics for the first experiment."""


def metric(name: str) -> MetricSpec:
    """Return one metric by name."""
    for spec in EXPERIMENT_METRICS:
        if spec.name == name:
            return spec
    raise ContractError(
        f"Unknown metric {name!r}. Declared: {', '.join(m.name for m in EXPERIMENT_METRICS)}."
    )


def primary_metrics() -> tuple[MetricSpec, ...]:
    """Metrics the hypothesis is judged on."""
    return tuple(spec for spec in EXPERIMENT_METRICS if spec.primary)


def computable_today() -> tuple[MetricSpec, ...]:
    """Metrics that can be computed without a trained model."""
    return tuple(spec for spec in EXPERIMENT_METRICS if spec.computable_today)


def compute(name: str, *args: object, **kwargs: object) -> float:
    """Compute a metric.

    Always raises for now. Metric implementations arrive with the model they
    measure; a metric that returns a number before then would be measuring
    nothing.
    """
    spec = metric(name)
    raise NotYetImplementedError(
        f"Metric {spec.name!r} ({spec.definition})",
        planned_in="Step 6, alongside the component it measures",
    )
