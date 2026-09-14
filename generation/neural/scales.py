"""Model scales, parameter estimates and compute estimates.

STATUS: ESTIMATES ONLY. Parameter counts are computed from declared widths by the
standard transformer formulas, so they are arithmetic rather than measurement.
Compute figures are engineering guesses with wide error bars and are labelled as
such; nothing here is a commitment, and no cluster is assumed.

The prototype scale is chosen to be trainable on **one** consumer or workstation
GPU, because the first experiment's value is in how fast it can be iterated, not
in how large it is. A hypothesis that takes a week per run gets tested once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from awr.errors import ContractError

__all__ = [
    "PROTOTYPE_DIMENSION_MAP",
    "ScaleSpec",
    "ComputeEstimate",
    "SCALES",
    "scale",
    "estimate_parameters",
    "estimate_training_flops",
]


@dataclass(frozen=True, slots=True)
class ComputeEstimate:
    """Rough compute expectations for one scale. Estimates, not measurements."""

    gpu_memory_gb: tuple[int, int]
    gpu_count: tuple[int, int]
    gpu_class: str
    wall_clock: str
    dataset_scenes: tuple[int, int]
    storage_gb: tuple[int, int]
    confidence: str = "rough estimate, order of magnitude only"

    def describe(self) -> str:
        """One-line human-readable description."""
        return (
            f"{self.gpu_count[0]}-{self.gpu_count[1]} x {self.gpu_class} "
            f"({self.gpu_memory_gb[0]}-{self.gpu_memory_gb[1]} GB), {self.wall_clock}, "
            f"{self.dataset_scenes[0]}-{self.dataset_scenes[1]} scenes, "
            f"{self.storage_gb[0]}-{self.storage_gb[1]} GB"
        )


@dataclass(frozen=True, slots=True)
class ScaleSpec:
    """Widths and depths of one model scale."""

    name: str
    purpose: str
    entity_width: int
    graph_layers: int
    graph_heads: int
    relation_width: int
    text_width: int
    text_layers: int
    text_vocabulary: int
    geometry_tokens: int
    geometry_width: int
    geometry_layers: int
    field_width: int
    field_layers: int
    align_width: int
    mlp_ratio: float = 4.0
    compute: ComputeEstimate = field(
        default_factory=lambda: ComputeEstimate((16, 24), (1, 1), "GPU", "hours", (1, 1), (1, 1))
    )
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.entity_width % self.graph_heads:
            raise ContractError(
                f"{self.name}: entity width {self.entity_width} must divide among "
                f"{self.graph_heads} heads."
            )


def _transformer_block(width: int, mlp_ratio: float) -> int:
    """Parameters in one pre-norm transformer block, ignoring norms and biases."""
    attention = 4 * width * width
    mlp = 2 * width * int(width * mlp_ratio)
    return attention + mlp


def estimate_parameters(
    spec: ScaleSpec, vocabulary_sizes: Mapping[str, int] | None = None
) -> dict[str, int]:
    """Estimate parameter counts per component.

    Arithmetic from declared widths, not a measurement of a built model. Useful
    for one decision in particular: seeing where the parameters actually go.
    """
    sizes = dict(vocabulary_sizes or {})
    entities = sizes.get("entity", 42)
    relations = sizes.get("relation", 18)
    types = (
        sizes.get("anatomy_type", 14)
        + sizes.get("semantic_role", 23)
        + sizes.get("laterality", 4)
        + sizes.get("educational_level", 5)
    )

    embeddings = (
        entities * spec.entity_width
        + types * spec.entity_width
        + relations * spec.relation_width
    )
    graph = spec.graph_layers * _transformer_block(spec.entity_width, spec.mlp_ratio)
    graph += spec.relation_width * spec.graph_heads  # relation to attention-bias projection

    language = spec.text_vocabulary * spec.text_width
    language += spec.text_layers * _transformer_block(spec.text_width, spec.mlp_ratio)
    language += spec.text_width * spec.entity_width  # pointer head over the entity codebook

    geometry_tokens = spec.geometry_tokens * spec.geometry_width  # learned token queries
    geometry_tokens += spec.geometry_layers * _transformer_block(
        spec.geometry_width, spec.mlp_ratio
    )
    geometry_tokens += 2 * spec.entity_width * spec.geometry_width  # cross-attention projections

    field = spec.field_layers * 2 * spec.field_width * spec.field_width
    field += 3 * 32 * spec.field_width  # positional encoding of a query point
    field += spec.geometry_width * spec.field_width  # token conditioning
    field += spec.field_width  # occupancy head
    field += spec.field_width * (entities + 1)  # part-label head

    frames = spec.entity_width * 12  # translation, log-scale and a 6D rotation basis
    alignment = (
        entities * spec.align_width
        + spec.entity_width * spec.align_width
        + spec.text_width * spec.align_width
        + spec.geometry_width * spec.align_width
    )

    breakdown = {
        "embeddings": embeddings,
        "graph_encoder": graph,
        "language": language,
        "geometry_tokeniser": geometry_tokens,
        "field_decoder": field,
        "frame_head": frames,
        "alignment": alignment,
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def estimate_training_flops(
    spec: ScaleSpec,
    *,
    scenes: int,
    epochs: int,
    queries_per_scene: int = 4096,
    vocabulary_sizes: Mapping[str, int] | None = None,
) -> float:
    """Very rough training FLOPs estimate.

    Uses the customary six FLOPs per parameter per token for forward and backward
    together, counting entity slots, text tokens and field queries as the tokens
    processed. Accurate to perhaps a factor of three, which is enough to tell a
    laptop-scale experiment from a cluster-scale one and not enough for anything
    else.
    """
    if scenes <= 0 or epochs <= 0:
        raise ContractError("Scenes and epochs must be positive.")
    params = estimate_parameters(spec, vocabulary_sizes)["total"]
    tokens_per_scene = 64 + spec.text_vocabulary * 0 + 128 + queries_per_scene
    return 6.0 * params * tokens_per_scene * scenes * epochs


SCALES: tuple[ScaleSpec, ...] = (
    ScaleSpec(
        name="prototype",
        purpose=(
            "The first heart experiment. Small enough to train and re-train on one GPU in "
            "hours, which is what makes an ablation affordable."
        ),
        entity_width=256,
        graph_layers=4,
        graph_heads=8,
        relation_width=64,
        text_width=384,
        text_layers=4,
        text_vocabulary=8192,
        geometry_tokens=32,
        geometry_width=64,
        geometry_layers=4,
        field_width=128,
        field_layers=4,
        align_width=256,
        compute=ComputeEstimate(
            gpu_memory_gb=(16, 24),
            gpu_count=(1, 1),
            gpu_class="single workstation GPU",
            wall_clock="2 to 12 hours per run",
            dataset_scenes=(2_000, 10_000),
            storage_gb=(20, 100),
        ),
    ),
    ScaleSpec(
        name="research",
        purpose="Multiple organs, real segmented data, image grounding.",
        entity_width=768,
        graph_layers=12,
        graph_heads=12,
        relation_width=128,
        text_width=768,
        text_layers=12,
        text_vocabulary=32_000,
        geometry_tokens=64,
        geometry_width=128,
        geometry_layers=8,
        field_width=256,
        field_layers=6,
        align_width=512,
        compute=ComputeEstimate(
            gpu_memory_gb=(40, 80),
            gpu_count=(4, 8),
            gpu_class="data-centre GPU",
            wall_clock="2 days to 2 weeks per run",
            dataset_scenes=(50_000, 500_000),
            storage_gb=(1_000, 10_000),
        ),
    ),
    ScaleSpec(
        name="production",
        purpose="Broad anatomy, high fidelity, interactive latency.",
        entity_width=1536,
        graph_layers=24,
        graph_heads=16,
        relation_width=256,
        text_width=2048,
        text_layers=24,
        text_vocabulary=64_000,
        geometry_tokens=128,
        geometry_width=256,
        geometry_layers=16,
        field_width=512,
        field_layers=8,
        align_width=1024,
        compute=ComputeEstimate(
            gpu_memory_gb=(80, 141),
            gpu_count=(32, 128),
            gpu_class="data-centre GPU",
            wall_clock="weeks per run",
            dataset_scenes=(1_000_000, 10_000_000),
            storage_gb=(10_000, 100_000),
        ),
    ),
)
"""Three proposed scales. Only the prototype is intended to be built soon."""


PROTOTYPE_DIMENSION_MAP: Mapping[str, str] = {
    "D_ENT": "entity_width",
    "D_REL": "relation_width",
    "D_TEXT": "text_width",
    "K_GEO": "geometry_tokens",
    "D_GEO": "geometry_width",
    "D_ALIGN": "align_width",
    "N_HEAD": "graph_heads",
}
"""Which declared dimension each prototype-scale field must equal.

The dimension registry and the prototype scale are two statements of the same
decision. They are cross-checked by :func:`validate_prototype_dimensions` so they
cannot drift, which is a mistake that only shows up as a shape error much later.
"""


def validate_prototype_dimensions(spec: ScaleSpec | None = None) -> None:
    """Check the prototype scale agrees with the declared dimensions."""
    from generation.neural.contracts import dim

    active = spec or scale("prototype")
    for dimension, attribute in PROTOTYPE_DIMENSION_MAP.items():
        declared = dim(dimension).size
        configured = getattr(active, attribute)
        if declared is not None and declared != configured:
            raise ContractError(
                f"Dimension {dimension} is declared as {declared} but the prototype scale sets "
                f"{attribute}={configured}. These are one decision stated twice."
            )


def scale(name: str) -> ScaleSpec:
    """Return a scale by name."""
    for candidate in SCALES:
        if candidate.name == name:
            return candidate
    raise ContractError(
        f"Unknown scale {name!r}. Declared: {', '.join(s.name for s in SCALES)}."
    )
