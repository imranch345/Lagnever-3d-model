"""Candidate 3D representations, the recommendation, and the geometry contracts.

STATUS: PROPOSED, UNVALIDATED. The comparison below is engineering judgement
recorded as data, not measurement. No representation has been trained or
benchmarked; the scores exist so the reasoning is inspectable and so a later
experiment can overturn it explicitly.

The question
------------

Which internal 3D representation should the first Lagnav prototype decode into?
The objective is deliberately not maximum visual quality. A representation that
renders beautifully but cannot say which points belong to the mitral valve fails
the project's first requirement.

The recommendation
------------------

**Per-entity latent token blocks decoded into per-entity implicit fields in
canonical entity frames, composed into a scene, with meshes extracted only at
export.**

* Each entity owns ``K_GEO`` tokens, ordered coarse to fine.
* Each entity's field is decoded in its own canonical frame; the frame itself
  (translation, scale, rotation) is predicted per entity and is explicit.
* The scene field is the composition of entity fields, and a part-label head
  attributes every query point to an entity.

Why this and not a triplane or a single field: correspondence becomes an *axis of
the tensor* rather than a learned association, local editing becomes re-decoding
one token block, and level of detail becomes a token prefix. The cost, recorded
honestly, is that a shared field can model inter-part surfaces more naturally,
and composition seams are a genuine risk this design has to manage.

Why canonical per-entity frames
-------------------------------

Decoding each entity in its own frame separates *where a structure is* from *what
shape it has*. Three things follow: shape decoding becomes a smaller, more
sample-efficient problem; spatial relationships such as ``superior_to`` become
computable from predicted frames and therefore directly supervisable; and moving
a structure does not require re-decoding its shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from awr.errors import ContractError, NotYetImplementedError
from generation.neural.contracts import CONTRACTS, Contract, DType, TensorSpec, dim

__all__ = [
    "Criterion",
    "DEFAULT_CRITERION_WEIGHTS",
    "RepresentationCandidate",
    "CANDIDATE_REPRESENTATIONS",
    "RECOMMENDED_REPRESENTATION",
    "rank_candidates",
    "LOD_TOKEN_SCHEDULE",
    "tokens_for_lod",
    "GEOMETRY_CONTRACT",
    "GeometryTokenSpec",
    "PerEntityFieldDecoder",
]


class Criterion(StrEnum):
    """Axes the candidate 3D representations are judged on."""

    ANATOMICAL_ACCURACY = "anatomical_accuracy"
    SEMANTIC_CORRESPONDENCE = "semantic_correspondence"
    EDITABILITY = "editability"
    MEMORY_EFFICIENCY = "memory_efficiency"
    TOPOLOGY = "topology"
    MULTI_VIEW_CONSISTENCY = "multi_view_consistency"
    LOCAL_EDITING = "local_editing"
    SCALABILITY = "scalability"
    TRAINING_SIMPLICITY = "training_simplicity"
    DECODING_QUALITY = "decoding_quality"


DEFAULT_CRITERION_WEIGHTS: Mapping[Criterion, float] = {
    Criterion.ANATOMICAL_ACCURACY: 1.0,
    Criterion.SEMANTIC_CORRESPONDENCE: 2.0,
    Criterion.EDITABILITY: 1.5,
    Criterion.MEMORY_EFFICIENCY: 0.75,
    Criterion.TOPOLOGY: 1.0,
    Criterion.MULTI_VIEW_CONSISTENCY: 0.75,
    Criterion.LOCAL_EDITING: 1.5,
    Criterion.SCALABILITY: 0.75,
    Criterion.TRAINING_SIMPLICITY: 1.0,
    Criterion.DECODING_QUALITY: 1.0,
}
"""Weights for the first prototype.

Correspondence and editing are weighted above visual quality because they are what
the research hypothesis is about. A different phase of the project should use
different weights, and changing them here changes the ranking, which is the point:
the recommendation is a consequence of stated priorities, not an objective fact.
"""


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    """One candidate 3D representation with judged scores and prose."""

    name: str
    summary: str
    scores: Mapping[Criterion, int]
    advantages: tuple[str, ...] = ()
    disadvantages: tuple[str, ...] = ()
    revisit_when: str | None = None

    def __post_init__(self) -> None:
        missing = sorted(str(c) for c in Criterion if c not in self.scores)
        if missing:
            raise ContractError(f"{self.name}: no score for criteria {missing}.")
        for criterion, score in self.scores.items():
            if not 1 <= score <= 5:
                raise ContractError(
                    f"{self.name}: score {score} for {criterion} is outside 1 to 5."
                )

    def weighted_total(
        self, weights: Mapping[Criterion, float] | None = None
    ) -> float:
        """Weighted sum of the judged scores."""
        active = weights or DEFAULT_CRITERION_WEIGHTS
        return sum(self.scores[c] * active.get(c, 0.0) for c in Criterion)


CANDIDATE_REPRESENTATIONS: tuple[RepresentationCandidate, ...] = (
    RepresentationCandidate(
        name="dense_voxels",
        summary="A regular occupancy grid over the scene.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 2,
            Criterion.SEMANTIC_CORRESPONDENCE: 4,
            Criterion.EDITABILITY: 3,
            Criterion.MEMORY_EFFICIENCY: 1,
            Criterion.TOPOLOGY: 2,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 4,
            Criterion.SCALABILITY: 1,
            Criterion.TRAINING_SIMPLICITY: 5,
            Criterion.DECODING_QUALITY: 2,
        },
        advantages=(
            "Per-voxel part labels are trivial, so correspondence is easy.",
            "Simplest possible training signal.",
        ),
        disadvantages=(
            "Memory grows with the cube of resolution; valve leaflets need resolution.",
            "Staircase surfaces make topology metrics close to meaningless.",
        ),
        revisit_when="A coarse occupancy prior is needed to bootstrap entity frames.",
    ),
    RepresentationCandidate(
        name="sparse_voxels",
        summary="An octree or hash grid holding only occupied space.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 3,
            Criterion.SEMANTIC_CORRESPONDENCE: 4,
            Criterion.EDITABILITY: 3,
            Criterion.MEMORY_EFFICIENCY: 3,
            Criterion.TOPOLOGY: 2,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 4,
            Criterion.SCALABILITY: 3,
            Criterion.TRAINING_SIMPLICITY: 3,
            Criterion.DECODING_QUALITY: 3,
        },
        advantages=("Much better memory behaviour than dense grids at the same resolution.",),
        disadvantages=(
            "Sparse machinery adds real implementation cost early in a project.",
            "Still discretised, so thin structures remain awkward.",
        ),
        revisit_when="Scaling beyond one organ makes dense fields too expensive.",
    ),
    RepresentationCandidate(
        name="point_cloud",
        summary="An unordered set of surface points, optionally with part labels.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 2,
            Criterion.SEMANTIC_CORRESPONDENCE: 4,
            Criterion.EDITABILITY: 3,
            Criterion.MEMORY_EFFICIENCY: 4,
            Criterion.TOPOLOGY: 1,
            Criterion.MULTI_VIEW_CONSISTENCY: 3,
            Criterion.LOCAL_EDITING: 4,
            Criterion.SCALABILITY: 4,
            Criterion.TRAINING_SIMPLICITY: 4,
            Criterion.DECODING_QUALITY: 2,
        },
        advantages=("Cheap, easy to supervise, easy to label per point.",),
        disadvantages=(
            "No surface and no topology, so 'is the chamber closed' cannot be asked.",
            "Rendering a transparent left ventricle from points is not a solved problem.",
        ),
        revisit_when="Used as an intermediate supervision target rather than the representation.",
    ),
    RepresentationCandidate(
        name="single_implicit_field",
        summary="One occupancy or signed-distance field for the whole organ.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 4,
            Criterion.SEMANTIC_CORRESPONDENCE: 2,
            Criterion.EDITABILITY: 2,
            Criterion.MEMORY_EFFICIENCY: 4,
            Criterion.TOPOLOGY: 4,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 1,
            Criterion.SCALABILITY: 4,
            Criterion.TRAINING_SIMPLICITY: 4,
            Criterion.DECODING_QUALITY: 4,
        },
        advantages=(
            "Continuous, resolution-free, good surfaces, well understood.",
            "Natural inter-part continuity: no composition seams.",
        ),
        disadvantages=(
            "Part identity must be recovered by a segmentation head after the fact.",
            "An edit to one structure changes the same weights that produce every other.",
        ),
        revisit_when=(
            "Composition seams between per-entity fields prove worse than segmentation error."
        ),
    ),
    RepresentationCandidate(
        name="triplane",
        summary="Three axis-aligned feature planes decoded by a small field network.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 4,
            Criterion.SEMANTIC_CORRESPONDENCE: 2,
            Criterion.EDITABILITY: 2,
            Criterion.MEMORY_EFFICIENCY: 4,
            Criterion.TOPOLOGY: 3,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 2,
            Criterion.SCALABILITY: 4,
            Criterion.TRAINING_SIMPLICITY: 3,
            Criterion.DECODING_QUALITY: 4,
        },
        advantages=(
            "Strong quality per unit of compute; well supported by existing literature.",
        ),
        disadvantages=(
            "Entities are entangled in shared planes; a local edit is a global write.",
            "Per-entity triplanes restore locality but multiply memory by the entity count.",
        ),
        revisit_when="Research scale, as a shared coarse context beneath per-entity tokens.",
    ),
    RepresentationCandidate(
        name="gaussian_splats",
        summary="A set of anisotropic Gaussians optimised for rendering.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 2,
            Criterion.SEMANTIC_CORRESPONDENCE: 2,
            Criterion.EDITABILITY: 2,
            Criterion.MEMORY_EFFICIENCY: 3,
            Criterion.TOPOLOGY: 1,
            Criterion.MULTI_VIEW_CONSISTENCY: 5,
            Criterion.LOCAL_EDITING: 3,
            Criterion.SCALABILITY: 3,
            Criterion.TRAINING_SIMPLICITY: 3,
            Criterion.DECODING_QUALITY: 5,
        },
        advantages=("Excellent appearance and view consistency.",),
        disadvantages=(
            "Appearance-first, with no surface and no topology.",
            "Optimises exactly the quantity this project refuses to treat as truth.",
        ),
        revisit_when="A presentation layer needs fast photorealistic preview of a fixed scene.",
    ),
    RepresentationCandidate(
        name="direct_mesh_generation",
        summary="Generating vertices and faces directly.",
        scores={
            Criterion.ANATOMICAL_ACCURACY: 3,
            Criterion.SEMANTIC_CORRESPONDENCE: 4,
            Criterion.EDITABILITY: 3,
            Criterion.MEMORY_EFFICIENCY: 4,
            Criterion.TOPOLOGY: 3,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 3,
            Criterion.SCALABILITY: 2,
            Criterion.TRAINING_SIMPLICITY: 1,
            Criterion.DECODING_QUALITY: 3,
        },
        advantages=("The export format, produced directly; per-face part labels are natural.",),
        disadvantages=(
            "Discrete connectivity is hard to generate and harder to train at small data scale.",
            "Topology errors are structural rather than a threshold choice.",
        ),
        revisit_when="Export quality becomes the bottleneck rather than representation quality.",
    ),
    RepresentationCandidate(
        name="per_entity_latent_fields",
        summary=(
            "Per-entity latent token blocks decoded into per-entity implicit fields in canonical "
            "entity frames, composed into a scene, meshed only for export."
        ),
        scores={
            Criterion.ANATOMICAL_ACCURACY: 4,
            Criterion.SEMANTIC_CORRESPONDENCE: 5,
            Criterion.EDITABILITY: 5,
            Criterion.MEMORY_EFFICIENCY: 4,
            Criterion.TOPOLOGY: 4,
            Criterion.MULTI_VIEW_CONSISTENCY: 4,
            Criterion.LOCAL_EDITING: 5,
            Criterion.SCALABILITY: 4,
            Criterion.TRAINING_SIMPLICITY: 3,
            Criterion.DECODING_QUALITY: 3,
        },
        advantages=(
            "Correspondence is an index, not an inference.",
            "Re-decoding one entity is the natural unit of work, so edits are local by design.",
            "Level of detail is a token prefix, so detail changes cannot change identity.",
            "Each field is a small local problem, which suits a small first dataset.",
        ),
        disadvantages=(
            "Composition seams between neighbouring entities are a real and open risk.",
            "Per-entity capacity is fixed in advance by K_GEO.",
            "Decoding quality is expected to trail a well-tuned single field at first.",
        ),
        revisit_when=(
            "If seam artefacts dominate the geometry metrics, fall back to a single field with "
            "a strong part head, which is ablation A4 in the first experiment."
        ),
    ),
)
"""Candidate representations, scored by engineering judgement rather than measurement."""


def rank_candidates(
    weights: Mapping[Criterion, float] | None = None,
) -> tuple[tuple[str, float], ...]:
    """Rank candidates by weighted score, highest first."""
    ranked = sorted(
        ((c.name, c.weighted_total(weights)) for c in CANDIDATE_REPRESENTATIONS),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ranked)


RECOMMENDED_REPRESENTATION = "per_entity_latent_fields"
"""The representation proposed for the first prototype. See ADR 0003."""


LOD_TOKEN_SCHEDULE: tuple[int, ...] = (4, 8, 16, 24, 32)
"""Geometry tokens read at each level of detail, from LOD 0 to LOD 4.

Nested rather than separate: a level of detail reads a *prefix* of the same token
block. One entity therefore has one geometry latent at every level, which is what
makes "show more detail" incapable of changing what a structure is. Separate
per-level latents were the alternative and are rejected in ADR 0006.
"""


def tokens_for_lod(lod: int, schedule: Sequence[int] | None = None) -> int:
    """Number of geometry tokens visible at a level of detail."""
    active = tuple(schedule or LOD_TOKEN_SCHEDULE)
    if not 0 <= lod < len(active):
        raise ContractError(
            f"Level of detail {lod} is outside the token schedule of length {len(active)}."
        )
    return active[lod]


def validate_token_schedule(schedule: Sequence[int] | None = None) -> None:
    """Check a token schedule is non-decreasing and ends at the declared token count."""
    active = tuple(schedule or LOD_TOKEN_SCHEDULE)
    if any(b < a for a, b in zip(active, active[1:], strict=False)):
        raise ContractError(
            f"Token schedule {active} must be non-decreasing: detail may not be removed by "
            "increasing the level of detail."
        )
    declared = dim("K_GEO").size
    if declared is not None and active[-1] != declared:
        raise ContractError(
            f"Token schedule ends at {active[-1]} but K_GEO is declared as {declared}."
        )


validate_token_schedule()


@dataclass(frozen=True, slots=True)
class GeometryTokenSpec:
    """Configuration of the per-entity geometry token block."""

    tokens_per_entity: int = 32
    token_width: int = 64
    lod_schedule: tuple[int, ...] = field(default_factory=lambda: LOD_TOKEN_SCHEDULE)
    field_kind: str = "occupancy"
    decoder_layers: int = 4
    decoder_width: int = 128
    canonical_frame: bool = True
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        validate_token_schedule(self.lod_schedule)
        if self.field_kind not in ("occupancy", "signed_distance"):
            raise ContractError(
                f"Unsupported field kind {self.field_kind!r}. The prototype proposes occupancy; "
                "signed distance is the recorded alternative."
            )

    def tokens_at(self, lod: int) -> int:
        """Tokens read at a level of detail."""
        return tokens_for_lod(lod, self.lod_schedule)


GEOMETRY_CONTRACT = CONTRACTS.register(
    Contract(
        name="geometry_decoding",
        purpose=(
            "Queries and outputs of the proposed per-entity field decoder, including the "
            "part-label head that makes correspondence checkable at any point in space."
        ),
        specs=(
            TensorSpec(
                "entity_mask", ("B", "N_ENT"), DType.BOOL, "True for real entity slots."
            ),
            TensorSpec(
                "query_points",
                ("B", "P_QUERY", "3D"),
                DType.FLOAT32,
                "Query positions in canonical scene space.",
                normalization="unit cube, each axis in [-1, 1]",
                value_range=(-1.0, 1.0),
            ),
            TensorSpec(
                "entity_query_points",
                ("B", "N_ENT", "P_QUERY", "3D"),
                DType.FLOAT32,
                "The same queries expressed in each entity's canonical frame.",
                normalization="unit cube per entity frame",
                mask="entity_mask",
                value_range=(-1.0, 1.0),
            ),
            TensorSpec(
                "entity_occupancy_logits",
                ("B", "N_ENT", "P_QUERY"),
                DType.FLOAT32,
                "Per-entity field value at each query point. This tensor is the geometry "
                "correspondence: axis 1 is the entity axis.",
                mask="entity_mask",
                padding_value=0.0,
            ),
            TensorSpec(
                "scene_occupancy_logits",
                ("B", "P_QUERY"),
                DType.FLOAT32,
                "Composed scene field, proposed as a smooth maximum over entity fields.",
            ),
            TensorSpec(
                "part_logits",
                ("B", "P_QUERY", "N_PART"),
                DType.FLOAT32,
                "Which entity owns each query point, plus a background class. Gives point-level "
                "correspondence and a directly supervisable target.",
            ),
            TensorSpec(
                "active_tokens",
                ("B", "N_ENT"),
                DType.INT32,
                "Token prefix length actually read per entity, set by the level of detail.",
                mask="entity_mask",
                padding_value=0,
            ),
        ),
    )
)
"""Contract for geometry decoding."""


class PerEntityFieldDecoder:
    """The proposed geometry decoder. Declared, not implemented.

    Contract for any implementation:

    * consume ``z_geometry[b, n, :k]`` where ``k`` comes from the level of detail,
    * decode in the entity's canonical frame given by ``entity_frame[b, n]``,
    * never read another entity's token block except through the scene latent,
      so that re-decoding one entity is provably local,
    * emit a part label for every query point.
    """

    decoder_id = "lagnav-per-entity-field-decoder-undefined"

    def __init__(self, spec: GeometryTokenSpec | None = None) -> None:
        self.spec = spec or GeometryTokenSpec()
        raise NotYetImplementedError(
            "PerEntityFieldDecoder", planned_in="Step 6, curriculum stage S2"
        )

    def decode(
        self, latent: Mapping[str, object], lod: int
    ) -> Mapping[str, object]:  # pragma: no cover - construction raises
        """Decode entity fields at a level of detail."""
        raise NotYetImplementedError("PerEntityFieldDecoder.decode", planned_in="Step 6")
