"""Step 5: the proposed Lagnav neural architecture.

STATUS: DESIGN. Everything in this package is either a contract, an interface, or
a piece of symbolic bookkeeping that needs no model. No network is trained, no
deep-learning framework is imported, and no performance claim is made.

What is implemented here, because it needs no model:

* ``contracts`` - named dimensions, tensor specifications and payload validation.
* ``features`` - the AWR to tensor bridge, with vocabularies and the entity codebook.
* ``graph_encoder`` - per-head attention masks and typed relation bias indices.
* ``language_encoder`` - the AWR program format and a transcoder that turns Step 4's
  deterministic parse into training supervision.
* ``three_d_latent`` - the candidate comparison and the nested level-of-detail schedule.
* ``multimodal`` - ontology-derived hard negatives.
* ``editing`` - the edit planner that says exactly what an instruction recomputes.
* ``losses``, ``curriculum``, ``scales`` - declared objectives, stages and estimates.
* ``model`` - pipeline composition and the end-to-end shape trace.

What is declared but not implemented: every learned component. Each raises
``awr.errors.NotYetImplementedError`` naming the step that will define it.
"""

from generation.neural.anatomical_encoder import (
    SUBSPACE_ROUTING,
    AnatomicalEncoderSpec,
    AnatomicalReasoningEncoder,
)
from generation.neural.contracts import (
    CONTRACTS,
    DIMENSIONS,
    Contract,
    ContractRegistry,
    Dim,
    DType,
    PlainArray,
    TensorSpec,
    dim,
)
from generation.neural.curriculum import CURRICULUM, CurriculumStage, stage
from generation.neural.editing import (
    EDIT_STRATEGIES,
    RECOMMENDED_EDIT_STRATEGY,
    EditKind,
    EditPlan,
    LocalGeometryEditor,
    plan_edit,
)
from generation.neural.features import (
    AWR_BATCH_CONTRACT,
    ENTITY_STATE_LAYOUT,
    AWRFeatureExtractor,
    EntityCodebook,
    Vocabulary,
)
from generation.neural.graph_encoder import (
    GRAPH_ENCODER_CONTRACT,
    AnatomicalGraphEncoder,
    GraphEncoderSpec,
    HeadAllocation,
    HeadRole,
    build_attention_mask,
)
from generation.neural.language_encoder import (
    PROGRAM_CONTRACT,
    TEXT_CONTRACT,
    AWRProgram,
    ProgramOp,
    ProgramOpType,
    ProgramTranscoder,
    TextToProgramModel,
)
from generation.neural.latent import (
    ENTITY_LATENT_LAYOUT,
    LATENT_CONTRACT,
    LATENT_TIERS,
    EntityLatentLayout,
)
from generation.neural.losses import (
    LOSSES,
    LossRole,
    LossSpec,
    first_experiment_losses,
    initial_loss_strategy,
)
from generation.neural.model import (
    LAGNAV_PIPELINE,
    LagnavArchitecture,
    PipelineStage,
    trace_pipeline,
)
from generation.neural.multimodal import (
    ALIGNMENT_METHODS,
    RECOMMENDED_ALIGNMENT,
    Modality,
    OntologyNegativeSampler,
)
from generation.neural.scales import SCALES, ComputeEstimate, ScaleSpec, estimate_parameters, scale
from generation.neural.three_d_latent import (
    CANDIDATE_REPRESENTATIONS,
    RECOMMENDED_REPRESENTATION,
    GeometryTokenSpec,
    PerEntityFieldDecoder,
    rank_candidates,
    tokens_for_lod,
)

__all__ = [
    "ALIGNMENT_METHODS",
    "AWRFeatureExtractor",
    "AWRProgram",
    "AWR_BATCH_CONTRACT",
    "AnatomicalEncoderSpec",
    "AnatomicalGraphEncoder",
    "AnatomicalReasoningEncoder",
    "CANDIDATE_REPRESENTATIONS",
    "CONTRACTS",
    "CURRICULUM",
    "ComputeEstimate",
    "Contract",
    "ContractRegistry",
    "CurriculumStage",
    "DIMENSIONS",
    "DType",
    "Dim",
    "EDIT_STRATEGIES",
    "ENTITY_LATENT_LAYOUT",
    "ENTITY_STATE_LAYOUT",
    "EditKind",
    "EditPlan",
    "EntityCodebook",
    "EntityLatentLayout",
    "GRAPH_ENCODER_CONTRACT",
    "GeometryTokenSpec",
    "GraphEncoderSpec",
    "HeadAllocation",
    "HeadRole",
    "LAGNAV_PIPELINE",
    "LATENT_CONTRACT",
    "LATENT_TIERS",
    "LOSSES",
    "LagnavArchitecture",
    "LocalGeometryEditor",
    "LossRole",
    "LossSpec",
    "Modality",
    "OntologyNegativeSampler",
    "PROGRAM_CONTRACT",
    "PerEntityFieldDecoder",
    "PipelineStage",
    "PlainArray",
    "ProgramOp",
    "ProgramOpType",
    "ProgramTranscoder",
    "RECOMMENDED_ALIGNMENT",
    "RECOMMENDED_EDIT_STRATEGY",
    "RECOMMENDED_REPRESENTATION",
    "SCALES",
    "SUBSPACE_ROUTING",
    "ScaleSpec",
    "TEXT_CONTRACT",
    "TensorSpec",
    "TextToProgramModel",
    "Vocabulary",
    "build_attention_mask",
    "dim",
    "estimate_parameters",
    "first_experiment_losses",
    "initial_loss_strategy",
    "plan_edit",
    "rank_candidates",
    "scale",
    "stage",
    "tokens_for_lod",
    "trace_pipeline",
]
