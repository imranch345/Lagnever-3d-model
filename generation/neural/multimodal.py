"""Proposed multimodal alignment: a hub, not a web.

STATUS: PROPOSED, UNVALIDATED. The hard-negative sampler is implemented because
it is pure ontology work; the encoders and losses are declared only.

The problem
-----------

The phrase "left ventricle", a textbook illustration of it, a 3D model of it, a
segmentation mask of it and the graph node ``heart.left_ventricle`` must all land
in the same place. The usual answer is pairwise contrastive learning between
modalities, in the style of image-text models.

Why not plain pairwise contrastive learning
-------------------------------------------

Three reasons, in order of importance:

1. **We already have the anchor.** Every modality can be labelled with a
   persistent entity id, because the AWR exists. Pairwise contrastive learning
   spends data discovering a correspondence that the ontology already states.
2. **Sample efficiency.** Pairwise objectives need large paired corpora to
   separate classes. The first experiment has hundreds of samples, not millions.
3. **It learns appearance.** With paired data alone, "left ventricle" aligns with
   whatever the images of it happen to look like: a particular rendering style, a
   particular textbook. That is the failure this project is defined against.

Proposed alignment: prototype hub
---------------------------------

The entity identity embedding table is the **hub**. Every modality encoder is a
spoke that projects into the shared alignment space ``D_ALIGN`` and is trained to
land on the correct entity prototype. Concretely: supervised contrastive learning
against class prototypes, not instance pairs.

* A text phrase aligns to the prototype of the entity it names.
* A geometry token block aligns to the prototype of the entity it decodes.
* An image crop or mask aligns to the prototype of the entity it depicts.
* The graph node *is* the prototype.

Cross-modal instance pairs are kept as an auxiliary objective where genuinely
paired data exists, because they capture within-entity variation that prototypes
average away. They are secondary, not the backbone.

Known risks, recorded rather than assumed away
----------------------------------------------

* **Prototype collapse.** Sibling structures may converge. Mitigation: hard
  negatives drawn from the ontology, below.
* **Prototypes absorb the dataset's bias.** If every left ventricle in training is
  a textbook cutaway, the prototype encodes cutaways. Mitigation: style
  augmentation and an explicit style-invariance objective, both open questions.
* **The hub makes identity easy to probe.** Identity probes will score highly by
  construction and must not be reported as evidence of learned anatomy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from awr.errors import NotYetImplementedError
from awr.ontology import AnatomyOntology
from awr.schema import Laterality
from generation.neural.contracts import CONTRACTS, Contract, DType, TensorSpec

__all__ = [
    "Modality",
    "AlignmentMethod",
    "ALIGNMENT_METHODS",
    "RECOMMENDED_ALIGNMENT",
    "ALIGNMENT_CONTRACT",
    "OntologyNegativeSampler",
    "ModalityEncoder",
]


class Modality(StrEnum):
    """A spoke of the alignment hub."""

    TEXT = "text"
    IMAGE = "image"
    MULTI_VIEW = "multi_view"
    GEOMETRY = "geometry"
    SEGMENTATION = "segmentation"
    GRAPH = "graph"

    @property
    def available_in_first_experiment(self) -> bool:
        """Whether the first heart experiment uses this modality."""
        return self in (Modality.TEXT, Modality.GEOMETRY, Modality.GRAPH)


@dataclass(frozen=True, slots=True)
class AlignmentMethod:
    """A candidate mechanism for aligning modalities."""

    name: str
    summary: str
    advantages: tuple[str, ...]
    disadvantages: tuple[str, ...]
    data_requirement: str
    verdict: str
    revisit_when: str | None = None


ALIGNMENT_METHODS: tuple[AlignmentMethod, ...] = (
    AlignmentMethod(
        name="pairwise_contrastive",
        summary="Image-text style InfoNCE between instance pairs from each modality.",
        advantages=(
            "No labels needed beyond pairing.",
            "Well understood, with strong results at large scale.",
        ),
        disadvantages=(
            "Needs large paired corpora, which this project does not have and may not licence.",
            "Aligns appearance, not anatomy; two renderings of one ventricle may separate.",
            "Ignores the entity labels the AWR already provides.",
        ),
        data_requirement="Large paired corpora per modality pair.",
        verdict="REJECTED as the primary mechanism",
        revisit_when="A large licensed paired corpus becomes available at research scale.",
    ),
    AlignmentMethod(
        name="cross_modal_transformer",
        summary="A joint transformer over concatenated modality tokens with masked modelling.",
        advantages=(
            "Rich interaction between modalities; handles missing modalities gracefully.",
        ),
        disadvantages=(
            "Expensive, and hard to interpret when alignment fails.",
            "Gives no explicit anchor, so identity is implicit in the weights.",
        ),
        data_requirement="Moderate to large multimodal corpora.",
        verdict="DEFERRED",
        revisit_when="After prototype alignment works and fusion quality becomes the bottleneck.",
    ),
    AlignmentMethod(
        name="prototype_hub",
        summary=(
            "Entity identity embeddings act as class prototypes; every modality encoder is "
            "trained to land on the right prototype, with ontology-derived hard negatives."
        ),
        advantages=(
            "Uses the labels the ontology already gives, so it is far more sample efficient.",
            "The shared space has an explicit, inspectable basis: one direction per entity.",
            "A new modality is a new spoke and does not disturb existing ones.",
            "Retrieval and classification are the same operation, which simplifies evaluation.",
        ),
        disadvantages=(
            "Prototypes average away within-entity variation.",
            "Requires every training sample to carry an entity label.",
            "Risks collapse between anatomically similar siblings.",
        ),
        data_requirement="Entity-labelled samples per modality; no cross-modal pairing required.",
        verdict="RECOMMENDED for the first prototype",
    ),
    AlignmentMethod(
        name="distillation_from_pretrained",
        summary="Distil a general vision-language model into the anatomical space.",
        advantages=("Cheap; inherits broad visual knowledge.",),
        disadvantages=(
            "Inherits the teacher's appearance bias and its licence constraints.",
            "Anatomical precision of general models is unverified for this domain.",
        ),
        data_requirement="Unlabelled images plus a teacher model with suitable licensing.",
        verdict="DEFERRED",
        revisit_when="Image data is scarce and a suitably licensed teacher is identified.",
    ),
)
"""Candidate alignment mechanisms with the reasoning for and against each."""


RECOMMENDED_ALIGNMENT = "prototype_hub"
"""The mechanism proposed for the first prototype. See ADR 0004."""


ALIGNMENT_CONTRACT = CONTRACTS.register(
    Contract(
        name="multimodal_alignment",
        purpose="Projections of every modality into the shared anatomical alignment space.",
        specs=(
            TensorSpec(
                "entity_prototypes",
                ("N_CODEBOOK", "D_ALIGN"),
                DType.FLOAT32,
                "The hub: one prototype per ontology entity, shared by every modality.",
                normalization="unit L2 norm",
            ),
            TensorSpec(
                "align_text",
                ("B", "D_ALIGN"),
                DType.FLOAT32,
                "Text projection, from a phrase or an utterance.",
                normalization="unit L2 norm",
            ),
            TensorSpec(
                "align_geometry",
                ("B", "N_ENT", "D_ALIGN"),
                DType.FLOAT32,
                "Projection of each entity's geometry token block.",
                normalization="unit L2 norm",
                mask="entity_mask",
            ),
            TensorSpec(
                "align_image",
                ("B", "N_VIEW", "D_ALIGN"),
                DType.FLOAT32,
                "Projection of each rendered or photographed view. Optional modality.",
                normalization="unit L2 norm",
            ),
            TensorSpec(
                "entity_mask", ("B", "N_ENT"), DType.BOOL, "True for real entity slots."
            ),
            TensorSpec(
                "alignment_target",
                ("B", "N_ENT"),
                DType.INT32,
                "Codebook index each projection should land on. The supervision signal.",
                mask="entity_mask",
                padding_value=-1,
            ),
        ),
    )
)
"""Contract for multimodal alignment."""


class OntologyNegativeSampler:
    """Draws hard negatives from the ontology rather than at random.

    Implemented, because it needs no model. Random negatives are easy: separating
    the left ventricle from the aorta teaches little. The informative negatives
    are the ones the ontology can name, in priority order:

    1. the mirrored structure across the body midline (left versus right),
    2. siblings under the same parent,
    3. other entities of the same anatomy type,
    4. spatially adjacent structures, which share context in every image.
    """

    def __init__(self, ontology: AnatomyOntology) -> None:
        self.ontology = ontology

    def mirrored(self, entity_id: str) -> tuple[str, ...]:
        """The contralateral structure, when the ontology declares one."""
        entity = self.ontology.get(entity_id)
        if entity.laterality not in (Laterality.LEFT, Laterality.RIGHT):
            return ()
        opposite = Laterality.RIGHT if entity.laterality is Laterality.LEFT else Laterality.LEFT
        return tuple(
            other.entity_id
            for other in self.ontology.iter_entities()
            if other.laterality is opposite
            and other.anatomy_type is entity.anatomy_type
            and other.entity_id != entity_id
        )

    def siblings(self, entity_id: str) -> tuple[str, ...]:
        """Entities sharing a parent in the presentation tree."""
        parent = self.ontology.parent(entity_id)
        if parent is None:
            return ()
        return tuple(child for child in self.ontology.children(parent) if child != entity_id)

    def same_type(self, entity_id: str) -> tuple[str, ...]:
        """Entities of the same anatomy type."""
        entity = self.ontology.get(entity_id)
        return tuple(
            other.entity_id
            for other in self.ontology.iter_entities()
            if other.anatomy_type is entity.anatomy_type and other.entity_id != entity_id
        )

    def adjacent(self, entity_id: str) -> tuple[str, ...]:
        """Spatially adjacent entities, which share visual context."""
        return self.ontology.relationships.related(entity_id, "adjacent_to")

    def hard_negatives(self, entity_id: str, limit: int = 8) -> tuple[str, ...]:
        """Ordered hard negatives for one entity, most informative first."""
        ordered: list[str] = []
        seen = {entity_id}
        for group in (
            self.mirrored(entity_id),
            self.siblings(entity_id),
            self.same_type(entity_id),
            self.adjacent(entity_id),
        ):
            for candidate in group:
                if candidate in seen:
                    continue
                if self.ontology.get(candidate).is_group:
                    continue
                seen.add(candidate)
                ordered.append(candidate)
                if len(ordered) == limit:
                    return tuple(ordered)
        return tuple(ordered)

    def coverage(self, limit: int = 8) -> Mapping[str, int]:
        """How many hard negatives each renderable entity has."""
        return {
            entity.entity_id: len(self.hard_negatives(entity.entity_id, limit))
            for entity in self.ontology.iter_entities()
            if entity.renderable
        }


class ModalityEncoder:
    """Interface for one spoke of the alignment hub. Declared, not implemented.

    Any implementation must project into ``D_ALIGN`` with unit norm and must be
    trainable against entity prototypes, so that adding a modality never requires
    retraining the modalities already aligned.
    """

    modality: Modality = Modality.TEXT
    encoder_id: str = "lagnav-modality-encoder-undefined"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotYetImplementedError(
            f"ModalityEncoder({self.modality})", planned_in="Step 6, curriculum stages S1 and S5"
        )

    def project(self, inputs: Sequence[object]) -> object:  # pragma: no cover
        """Project inputs into the alignment space."""
        raise NotYetImplementedError("ModalityEncoder.project", planned_in="Step 6")
