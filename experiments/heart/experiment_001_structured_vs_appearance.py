"""Experiment 001: does a structured anatomical representation help.

STATUS: SPECIFICATION, PRE-REGISTERED, NOT RUN. No model exists, no data has been
collected, and no result is claimed.

The point of writing this before building anything is that the success criteria
are fixed in advance. A threshold chosen after seeing the numbers is not a
threshold.

Run ``python -m experiments.heart.experiment_001_structured_vs_appearance`` to
print the plan.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field

from awr.errors import ContractError
from evaluation.neural_metrics import EXPERIMENT_METRICS, metric

__all__ = [
    "ExperimentArm",
    "AblationSpec",
    "SuccessCriterion",
    "DatasetSpec",
    "ExperimentSpec",
    "EXPERIMENT_001",
    "main",
]


@dataclass(frozen=True, slots=True)
class ExperimentArm:
    """One arm of the comparison."""

    key: str
    name: str
    config: str
    description: str
    controls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AblationSpec:
    """One ablation and the question it isolates."""

    key: str
    name: str
    change: str
    isolates: str


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """A pre-registered criterion, with the outcome that would refute it."""

    hypothesis: str
    metric_name: str
    comparison: str
    threshold: str
    refuted_if: str

    def __post_init__(self) -> None:
        metric(self.metric_name)  # raises if the metric is not declared


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """What data the experiment needs, and where it is allowed to come from."""

    tier: str
    description: str
    scenes: tuple[int, int]
    provenance: str
    licensing: str
    splits: str
    caveats: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A complete, pre-registered experiment plan."""

    experiment_id: str
    question: str
    hypotheses: tuple[tuple[str, str], ...]
    arms: tuple[ExperimentArm, ...]
    datasets: tuple[DatasetSpec, ...]
    metrics: tuple[str, ...]
    ablations: tuple[AblationSpec, ...]
    success_criteria: tuple[SuccessCriterion, ...]
    controls: tuple[str, ...]
    risks: tuple[str, ...]
    status: str = "PRE-REGISTERED, NOT RUN"
    results: None = field(default=None, init=False)
    """Always ``None``. A result field that could be filled in by hand is a result
    field that will be filled in by hand."""

    def validate(self) -> None:
        """Check the plan references declared metrics and has two arms to compare."""
        declared = {spec.name for spec in EXPERIMENT_METRICS}
        unknown = sorted(set(self.metrics) - declared)
        if unknown:
            raise ContractError(f"Experiment references undeclared metrics {unknown}.")
        if len(self.arms) < 2:
            raise ContractError("An ablation needs at least two arms.")
        if not self.success_criteria:
            raise ContractError(
                "An experiment without pre-registered success criteria cannot refute anything."
            )

    def render(self) -> str:
        """Render the plan as readable text."""
        lines = [
            f"Experiment {self.experiment_id}  [{self.status}]",
            "",
            f"Question: {self.question}",
            "",
            "Hypotheses:",
        ]
        lines.extend(f"  {key}: {text}" for key, text in self.hypotheses)
        lines.append("")
        lines.append("Arms:")
        for arm in self.arms:
            lines.append(f"  {arm.key:22s} {arm.name} ({arm.config})")
            lines.append(f"  {'':22s} {arm.description}")
        lines.append("")
        lines.append("Data:")
        for dataset in self.datasets:
            lines.append(
                f"  {dataset.tier:10s} {dataset.scenes[0]}-{dataset.scenes[1]} scenes, "
                f"{dataset.licensing}"
            )
            lines.append(f"  {'':10s} {dataset.description}")
        lines.append("")
        lines.append("Ablations:")
        lines.extend(f"  {a.key}  {a.name:34s} isolates: {a.isolates}" for a in self.ablations)
        lines.append("")
        lines.append("Pre-registered success criteria:")
        for criterion in self.success_criteria:
            lines.append(f"  {criterion.hypothesis}: {criterion.metric_name}")
            lines.append(f"      supported if {criterion.threshold}")
            lines.append(f"      refuted if   {criterion.refuted_if}")
        lines.append("")
        lines.append("Controls:")
        lines.extend(f"  - {control}" for control in self.controls)
        lines.append("")
        lines.append("Risks:")
        lines.extend(f"  - {risk}" for risk in self.risks)
        return "\n".join(lines)


EXPERIMENT_001 = ExperimentSpec(
    experiment_id="heart-001-structured-vs-appearance",
    question=(
        "Does a structured anatomical latent representation improve semantic control and "
        "anatomical consistency compared with an appearance-driven representation trained on "
        "the same data with the same budget?"
    ),
    hypotheses=(
        (
            "H1a",
            "The structured arm achieves higher part-level control success than the "
            "appearance-driven arm.",
        ),
        (
            "H1b",
            "The structured arm produces geometry that satisfies more held-out anatomical "
            "relationships.",
        ),
        (
            "H1c",
            "The structured arm keeps entity identity and edit locality across a multi-turn "
            "session, where the appearance-driven arm degrades.",
        ),
        (
            "H1d",
            "Any advantage is larger in the small-data regime, because structure substitutes "
            "for data.",
        ),
        (
            "H0",
            "Null hypothesis: no difference beyond run-to-run variance on any primary metric.",
        ),
    ),
    arms=(
        ExperimentArm(
            key="lagnav_structured",
            name="Lagnav structured anatomical representation",
            config="configs/neural/prototype.yaml",
            description=(
                "Entity axis, three typed graphs, per-entity geometry tokens and fields, "
                "nested level-of-detail prefixes."
            ),
            controls=("matched parameters", "matched data", "matched steps", "matched seeds"),
        ),
        ExperimentArm(
            key="appearance_baseline",
            name="Appearance-driven baseline",
            config="configs/neural/baseline.yaml",
            description=(
                "One undifferentiated latent token set, no graphs, part identity recovered "
                "afterwards by a segmentation head."
            ),
            controls=("matched parameters", "matched data", "matched steps", "matched seeds"),
        ),
    ),
    datasets=(
        DatasetSpec(
            tier="tier-0",
            description=(
                "Procedurally generated part-labelled hearts: one primitive per ontology entity, "
                "varied in size, position and proportion within anatomically ordered ranges. "
                "Ground-truth part labels, frames and relationships are exact because the "
                "generator knows them."
            ),
            scenes=(2_000, 10_000),
            provenance="generated by this project, no external source",
            licensing="no licensing exposure",
            splits="80/10/10 by procedural family, so no family appears in two splits",
            caveats=(
                "NOT anatomically realistic. It measures representation properties, not clinical "
                "fidelity, and no result from it may be described as anatomical accuracy.",
                "A structured generator could favour a structured model. Ablation A4 exists "
                "partly to detect that.",
            ),
        ),
        DatasetSpec(
            tier="tier-1",
            description=(
                "Real segmented cardiac data with per-structure labels, for a follow-up run once "
                "tier-0 has shown whether the effect exists at all."
            ),
            scenes=(50, 500),
            provenance="public cardiac segmentation corpora, each reviewed individually",
            licensing="NOT ACQUIRED. No dataset may be downloaded before its terms are reviewed "
            "and recorded, per the Step 4 dataset policy.",
            splits="held out by subject and by source, not by scene",
            caveats=(
                "Small, heterogeneous, and labelled to clinical rather than ontological "
                "conventions; a label-mapping step is required and is itself a source of error.",
            ),
        ),
    ),
    metrics=(
        "part_control_success",
        "edit_locality",
        "identity_persistence",
        "structural_relationship_accuracy",
        "spatial_relationship_accuracy",
        "functional_relationship_accuracy",
        "chamfer_distance",
        "part_segmentation_iou",
        "watertight_fraction",
        "sample_efficiency",
    ),
    ablations=(
        AblationSpec(
            key="A0",
            name="Appearance-driven baseline",
            change="No entity axis, no graphs, no per-entity geometry.",
            isolates="The value of the whole structured representation.",
        ),
        AblationSpec(
            key="A1",
            name="Entity axis only",
            change="Per-entity tokens, graph encoder removed.",
            isolates="How much comes from factorisation alone, before any relational reasoning.",
        ),
        AblationSpec(
            key="A2",
            name="Structure graph only",
            change="Spatial and functional graphs removed.",
            isolates="Whether hierarchy alone accounts for the effect.",
        ),
        AblationSpec(
            key="A3",
            name="All three graphs",
            change="The full proposed encoder.",
            isolates="The contribution of spatial and functional relations over structure.",
        ),
        AblationSpec(
            key="A4",
            name="Shared field with part head",
            change="One field for the scene, part identity from a segmentation head.",
            isolates=(
                "Whether per-entity fields matter, or whether a part head on a shared field "
                "is enough."
            ),
        ),
        AblationSpec(
            key="A5",
            name="Independent level-of-detail latents",
            change="Nested token prefixes replaced by separate latents per level.",
            isolates="Whether nesting is what keeps identity stable across detail changes.",
        ),
        AblationSpec(
            key="A6",
            name="Pairwise contrastive alignment",
            change="Prototype hub replaced by instance-pair contrastive learning.",
            isolates="Whether ontology-anchored alignment is worth its extra labelling.",
        ),
    ),
    success_criteria=(
        SuccessCriterion(
            hypothesis="H1a",
            metric_name="part_control_success",
            comparison="structured arm versus appearance baseline, tier-0 test split",
            threshold=(
                "the structured arm exceeds the baseline by at least 20 percentage points, with "
                "non-overlapping intervals across three seeds"
            ),
            refuted_if="the gap is under 5 points, or the intervals overlap",
        ),
        SuccessCriterion(
            hypothesis="H1b",
            metric_name="structural_relationship_accuracy",
            comparison="held-out relationship checks, never trained on",
            threshold="the structured arm is at least 15 points higher",
            refuted_if="the gap is under 5 points",
        ),
        SuccessCriterion(
            hypothesis="H1c",
            metric_name="edit_locality",
            comparison="ten-instruction sessions on the test split",
            threshold=(
                "the structured arm changes untouched entities by essentially nothing, and the "
                "baseline changes them measurably"
            ),
            refuted_if="the baseline also achieves near-zero drift on untouched entities",
        ),
        SuccessCriterion(
            hypothesis="guard",
            metric_name="chamfer_distance",
            comparison="structured arm versus appearance baseline",
            threshold=(
                "the structured arm is no more than 20 percent worse; semantics must not be "
                "bought with geometry"
            ),
            refuted_if=(
                "geometry is more than 20 percent worse, in which case a semantic win does not "
                "count as support for the hypothesis"
            ),
        ),
        SuccessCriterion(
            hypothesis="H1d",
            metric_name="sample_efficiency",
            comparison="runs at 10, 30 and 100 percent of the training set",
            threshold="the structured arm's advantage grows as the training set shrinks",
            refuted_if="the advantage is flat or shrinks in the small-data regime",
        ),
    ),
    controls=(
        "Matched parameter count within 5 percent, verified by the estimator before training.",
        "Identical data, splits, steps, optimiser and seeds across arms.",
        "Three seeds per arm; single-seed differences are not reported as results.",
        "Relationship objectives held out of training in both arms.",
        "The Step 4 deterministic system reported alongside as the semantic ceiling.",
    ),
    risks=(
        "The synthetic tier may favour structured models because it was generated structurally. "
        "Mitigation: ablation A4, plus a tier-1 replication before any general claim.",
        "Per-entity fields may produce visible seams, hurting geometry metrics for reasons "
        "unrelated to the hypothesis.",
        "Part-control success depends on a part attribution step, which is easier for the "
        "structured arm by construction; the metric definition must not smuggle in the answer.",
        "A null result is a real possibility and must be reported as one.",
    ),
)
"""The pre-registered plan for the first heart experiment."""

EXPERIMENT_001.validate()


def main(argv: Sequence[str] | None = None) -> int:
    """Print the experiment plan."""
    parser = argparse.ArgumentParser(description="Print the Lagnav heart experiment 001 plan.")
    parser.parse_args(argv)
    print(EXPERIMENT_001.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
