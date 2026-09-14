"""Candidate training objectives, declared as data rather than code.

STATUS: PROPOSED. Nothing here computes a gradient. Each objective records what it
teaches, what data it needs, what failure it prevents, and whether it belongs in
the first heart experiment, so that the loss strategy can be argued about and
changed without touching model code.

One methodological rule shapes the selection
--------------------------------------------

**Do not train on the metric the hypothesis is judged by.** The research question
is whether a structured anatomical representation produces better anatomical
consistency *by virtue of its structure*. If relationship correctness is optimised
directly, a better score proves only that the objective worked. In the first
experiment the three relationship objectives are therefore held out of training
and used as evaluation only, and both arms of the ablation are trained on the same
objectives. If Lagnav wins under those conditions, the win is attributable to the
architecture.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from awr.errors import ContractError

__all__ = [
    "LossRole",
    "LossSpec",
    "LOSSES",
    "loss",
    "first_experiment_losses",
    "evaluation_only_losses",
    "initial_loss_strategy",
]


class LossRole(StrEnum):
    """How an objective is used in a given phase."""

    TRAIN = "train"
    EVALUATE_ONLY = "evaluate_only"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class LossSpec:
    """One candidate objective."""

    name: str
    teaches: str
    requires: str
    prevents: str
    role_in_first_experiment: LossRole
    stages: tuple[str, ...]
    proposed_weight: float | None = None
    form: str = "undecided"
    notes: str | None = None
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.proposed_weight is not None and self.proposed_weight <= 0:
            raise ContractError(f"{self.name}: a proposed weight must be positive.")
        if self.role_in_first_experiment is LossRole.TRAIN and self.proposed_weight is None:
            raise ContractError(
                f"{self.name}: objectives trained in the first experiment need a proposed weight."
            )


LOSSES: tuple[LossSpec, ...] = (
    LossSpec(
        name="language_anatomy_alignment",
        teaches=(
            "Map an utterance to the correct AWR program: the right operation and the right "
            "entity arguments."
        ),
        requires=(
            "Utterance and program pairs. Generated free of charge by the Step 4 transcoder "
            "over templated and paraphrased commands; no external data."
        ),
        prevents=(
            "The model acting on the wrong structure, and inventing anatomy that the ontology "
            "does not contain."
        ),
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S1",),
        proposed_weight=1.0,
        form="cross-entropy over the operation vocabulary plus a pointer over the entity codebook",
    ),
    LossSpec(
        name="entity_identity",
        teaches="Each entity's prototype is distinct and each modality lands on the right one.",
        requires="Entity-labelled samples in any modality. The ontology supplies the labels.",
        prevents=(
            "Sibling structures collapsing, which shows up later as edits affecting the wrong "
            "chamber."
        ),
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S0", "S1", "S2"),
        proposed_weight=1.0,
        form="supervised contrastive against prototypes, with ontology-derived hard negatives",
        notes=(
            "Identity probing of z_entity is exact by construction, so probe accuracy is not "
            "evidence. This objective is about the alignment space, not the identity slice."
        ),
    ),
    LossSpec(
        name="structural_relationship",
        teaches="Predict part_of, connects_to and continuous_with edges from entity latents.",
        requires="The ontology graph alone.",
        prevents="Latents that ignore composition, producing parts that do not belong together.",
        role_in_first_experiment=LossRole.EVALUATE_ONLY,
        stages=("S0",),
        proposed_weight=0.5,
        form="binary cross-entropy over typed edge prediction",
        notes=(
            "Trained in S0 pretraining; held out during the ablation so it cannot rig the metric."
        ),
    ),
    LossSpec(
        name="spatial_relationship",
        teaches=(
            "Predicted entity frames satisfy the spatial graph: superior_to, adjacent_to, "
            "left_of and the rest."
        ),
        requires="The ontology graph plus predicted or ground-truth entity frames.",
        prevents="Anatomically impossible arrangements that still look plausible in one view.",
        role_in_first_experiment=LossRole.EVALUATE_ONLY,
        stages=("S0", "S3"),
        proposed_weight=0.5,
        form="margin loss on frame-derived relations, or edge classification from latents",
        notes=(
            "Computable analytically from entity frames, which is a direct benefit of decoding "
            "each entity in a canonical frame."
        ),
    ),
    LossSpec(
        name="functional_relationship",
        teaches="Flow and conduction structure is recoverable from the representation.",
        requires="The functional graph; later, lumen connectivity measured from geometry.",
        prevents="A heart whose chambers are shaped correctly but do not connect into a circuit.",
        role_in_first_experiment=LossRole.EVALUATE_ONLY,
        stages=("S0", "S3"),
        proposed_weight=0.5,
        form="typed edge prediction; geometric variant checks lumen connectivity",
    ),
    LossSpec(
        name="geometry_reconstruction",
        teaches="Decode each entity's field so it reconstructs the target shape.",
        requires="3D data with per-entity part labels. The main data dependency of the project.",
        prevents="A semantically perfect representation that decodes to nothing usable.",
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S2", "S3", "S4"),
        proposed_weight=1.0,
        form="binary cross-entropy on occupancy at sampled query points",
    ),
    LossSpec(
        name="semantic_part_correspondence",
        teaches="Every point in space is attributed to the entity that owns it.",
        requires="Part-labelled 3D data; labels are exact in the synthetic tier.",
        prevents=(
            "The failure this project is defined against: geometry that cannot say which part "
            "of it is the left ventricle."
        ),
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S2", "S3"),
        proposed_weight=1.0,
        form="cross-entropy over the part-label head, including a background class",
    ),
    LossSpec(
        name="entity_frame",
        teaches="Predict each entity's canonical position, scale and orientation.",
        requires="Per-entity poses, derivable from part-labelled geometry.",
        prevents=(
            "Shape and placement being entangled, which makes moving a structure require "
            "re-decoding its shape."
        ),
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S2", "S3"),
        proposed_weight=0.5,
        form="L1 on translation and log-scale, geodesic loss on rotation",
    ),
    LossSpec(
        name="topology",
        teaches="Produce closed, well-formed surfaces.",
        requires="Geometry only; no extra labels.",
        prevents="Fields that mesh into holes or disconnected shells at the export step.",
        role_in_first_experiment=LossRole.DEFERRED,
        stages=("S3",),
        proposed_weight=0.1,
        form="eikonal regularisation for signed distance, or surface smoothness for occupancy",
        notes=(
            "Deferred: topology regularisers are easy to weight badly, and their effect is "
            "hard to attribute while reconstruction quality is still moving."
        ),
    ),
    LossSpec(
        name="multi_view_consistency",
        teaches="Renderings agree across viewpoints.",
        requires="A differentiable renderer and multi-view supervision.",
        prevents="Representations that only look right from the training camera.",
        role_in_first_experiment=LossRole.DEFERRED,
        stages=("S5",),
        form="photometric or feature-space consistency across sampled views",
        notes=(
            "Largely moot for a field representation, which is view-consistent by construction. "
            "It matters for baselines that are not."
        ),
    ),
    LossSpec(
        name="cross_modal_alignment",
        teaches="Images and text of the same structure land near its prototype.",
        requires="Entity-labelled images, which need licensing.",
        prevents="A text-only interface that cannot be grounded in visual evidence later.",
        role_in_first_experiment=LossRole.DEFERRED,
        stages=("S5",),
        form="prototype contrastive per modality, with optional paired InfoNCE",
    ),
    LossSpec(
        name="lod_consistency",
        teaches=(
            "A coarse token prefix decodes a coarser but compatible version of the same shape."
        ),
        requires="Geometry only. Supervision is generated by decoding at several prefixes.",
        prevents=(
            "Level of detail changing identity: the failure where 'show more detail' returns a "
            "different structure rather than a finer one."
        ),
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S2", "S3"),
        proposed_weight=0.5,
        form="agreement between prefix decodings, plus nested-dropout style prefix sampling",
    ),
    LossSpec(
        name="edit_consistency",
        teaches="An edit changes the target entity and leaves every other entity alone.",
        requires="Paired before and after scenes, generated by the Step 4 engine.",
        prevents="Edits leaking into neighbouring structures.",
        role_in_first_experiment=LossRole.TRAIN,
        stages=("S4",),
        proposed_weight=1.0,
        form="reconstruction on the edited entity plus an explicit penalty on any change elsewhere",
        notes=(
            "For Lagnav the locality term should be zero by construction; it is kept because it "
            "is the only comparable way to train and measure the baseline."
        ),
    ),
    LossSpec(
        name="functional_animation_consistency",
        teaches="Deformation across the cardiac cycle preserves flow structure and volumes.",
        requires="Temporal 3D data or a physiological prior; neither is available yet.",
        prevents="Animation that looks like motion but violates the circuit.",
        role_in_first_experiment=LossRole.DEFERRED,
        stages=("S5",),
        form="volume-curve and connectivity constraints across phases",
    ),
    LossSpec(
        name="style_invariance",
        teaches="The same anatomy in different rendering styles maps to the same latent.",
        requires="Style-varied renderings of identical anatomy.",
        prevents=(
            "Anatomy being confounded with appearance, which is the central risk of any "
            "image-supervised route."
        ),
        role_in_first_experiment=LossRole.DEFERRED,
        stages=("S5",),
        form="invariance penalty across style augmentations of one scene",
    ),
)
"""Candidate objectives. Thirteen requested, plus entity frames and style invariance."""


def loss(name: str) -> LossSpec:
    """Return one objective by name."""
    for spec in LOSSES:
        if spec.name == name:
            return spec
    raise ContractError(
        f"Unknown objective {name!r}. Declared: {', '.join(s.name for s in LOSSES)}."
    )


def first_experiment_losses() -> tuple[LossSpec, ...]:
    """Objectives trained in the first heart experiment."""
    return tuple(s for s in LOSSES if s.role_in_first_experiment is LossRole.TRAIN)


def evaluation_only_losses() -> tuple[LossSpec, ...]:
    """Objectives measured but deliberately not trained in the first experiment."""
    return tuple(s for s in LOSSES if s.role_in_first_experiment is LossRole.EVALUATE_ONLY)


def initial_loss_strategy() -> Mapping[str, float]:
    """Proposed starting weights for the first experiment.

    Equal weight on the two objectives that define the deliverable (reconstruct
    the shape, know which part it is), half weight on the objectives that shape
    the representation, and nothing on anything the experiment measures.
    """
    return {spec.name: spec.proposed_weight or 0.0 for spec in first_experiment_losses()}


def validate_strategy(weights: Mapping[str, float], declared: Iterable[str] | None = None) -> None:
    """Check a loss configuration references only declared objectives."""
    known = {spec.name for spec in LOSSES} if declared is None else set(declared)
    unknown = sorted(set(weights) - known)
    if unknown:
        raise ContractError(
            f"Loss configuration references undeclared objectives {unknown}. Declare them in "
            "generation/neural/losses.py with their data requirements first."
        )
    trained_but_measured = sorted(
        name
        for name in weights
        if name in {s.name for s in evaluation_only_losses()} and weights[name] > 0
    )
    if trained_but_measured:
        raise ContractError(
            f"Objectives {trained_but_measured} are held out for evaluation in the first "
            "experiment; training on them would make the hypothesis untestable."
        )
