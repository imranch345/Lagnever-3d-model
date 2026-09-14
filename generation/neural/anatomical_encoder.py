"""AWR to anatomical latent: the encoder that sits between symbols and geometry.

STATUS: the feature-to-subspace routing table is implemented and tested; the
encoder itself is a declared interface.

What this stage is for
----------------------

The AWR is symbolic and exact. The geometry decoder needs something continuous.
This stage is the only place the two meet, and the design rule is that it should
**inject what the AWR already knows rather than make the model rediscover it**.
Entity type, tree depth, laterality and relationships are all available as exact
symbols; spending model capacity on inferring them from data would be spending it
twice.

Routing
-------

Each entity-latent subspace declares which AWR tensors feed it, which is checked
against the AWR batch contract by :func:`validate_routing`. That check exists
because a subspace with no declared source is a subspace that will quietly learn
nothing, and the failure is invisible at training time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from awr.errors import ContractError, NotYetImplementedError
from generation.neural.contracts import CONTRACTS, Contract, DType, PlainArray, TensorSpec
from generation.neural.features import AWR_BATCH_CONTRACT
from generation.neural.graph_encoder import GraphEncoderSpec
from generation.neural.latent import ENTITY_LATENT_LAYOUT

__all__ = [
    "SUBSPACE_ROUTING",
    "validate_routing",
    "AnatomicalEncoderSpec",
    "SCENE_CONTEXT_CONTRACT",
    "AnatomicalReasoningEncoder",
]


SUBSPACE_ROUTING: Mapping[str, tuple[str, ...]] = {
    "identity": ("entity_ids",),
    "type": ("entity_type", "semantic_role", "laterality"),
    "hierarchy": ("parent_index", "depth"),
    "geometry_summary": (),
    "free": (),
}
"""Which AWR tensors feed which entity-latent subspace.

``geometry_summary`` is fed by the geometry tokens, not by the AWR, and ``free``
is deliberately unfed. Both are listed so the absence is a recorded decision
rather than an omission.
"""


def validate_routing() -> None:
    """Check every routed tensor exists in the AWR batch contract."""
    declared = set(AWR_BATCH_CONTRACT.names())
    for subspace, sources in SUBSPACE_ROUTING.items():
        for source in sources:
            if source not in declared:
                raise ContractError(
                    f"Subspace {subspace!r} is routed from {source!r}, which the AWR batch "
                    "contract does not declare."
                )
    known = {subspace.name for subspace in ENTITY_LATENT_LAYOUT.subspaces}
    missing = sorted(known - set(SUBSPACE_ROUTING))
    if missing:
        raise ContractError(f"Entity subspaces with no declared routing: {missing}.")
    extra = sorted(set(SUBSPACE_ROUTING) - known)
    if extra:
        raise ContractError(f"Routing declared for unknown subspaces: {extra}.")


validate_routing()


SCENE_CONTEXT_CONTRACT = CONTRACTS.register(
    Contract(
        name="scene_context",
        purpose="Scene-level conditioning assembled from the AWR and the conditioning path.",
        specs=(
            TensorSpec(
                "scene_lod",
                ("B",),
                DType.INT32,
                "Active level of detail, which selects the geometry token prefix.",
            ),
            TensorSpec(
                "scene_level",
                ("B",),
                DType.INT32,
                "Audience level, which conditions style and how much detail is exposed.",
            ),
            TensorSpec(
                "text_pooled",
                ("B", "D_TEXT"),
                DType.FLOAT32,
                "Conditioning-path text embedding. Influences appearance, never anatomy.",
                normalization="unit L2 norm",
            ),
            TensorSpec(
                "z_scene",
                ("B", "D_SCENE"),
                DType.FLOAT32,
                "Scene latent produced by this stage.",
                normalization="layer-normalised",
            ),
        ),
    )
)
"""Contract for scene-level context."""


@dataclass(frozen=True, slots=True)
class AnatomicalEncoderSpec:
    """Configuration of the proposed anatomical encoder."""

    graph: GraphEncoderSpec = GraphEncoderSpec()
    scene_width: int = 256
    inject_symbolic_features: bool = True
    freeze_identity_subspace: bool = True
    status: str = "PROPOSED"

    def describe(self) -> dict[str, object]:
        """Compact description for reports and configuration checks."""
        return {
            "graph_layers": self.graph.layers,
            "graph_heads": self.graph.heads.total,
            "entity_width": self.graph.entity_width,
            "scene_width": self.scene_width,
            "identity_frozen": self.freeze_identity_subspace,
            "status": self.status,
        }


class AnatomicalReasoningEncoder:
    """Interface for the AWR-to-latent encoder. Declared, not implemented.

    Contract for any implementation:

    * compose entity latents from the routed symbolic features,
    * contextualise them over the three graphs,
    * leave ``z_entity[..., 0:96]``, the identity subspace, bit-identical to its
      input,
    * emit a scene latent that no entity latent duplicates.
    """

    encoder_id = "lagnav-anatomical-encoder-undefined"

    def __init__(self, spec: AnatomicalEncoderSpec | None = None) -> None:
        self.spec = spec or AnatomicalEncoderSpec()
        raise NotYetImplementedError(
            "AnatomicalReasoningEncoder", planned_in="Step 6, curriculum stage S0"
        )

    def encode(
        self, payload: Mapping[str, PlainArray]
    ) -> Mapping[str, PlainArray]:  # pragma: no cover - construction raises
        """Encode an AWR batch into the anatomical latent."""
        raise NotYetImplementedError("AnatomicalReasoningEncoder.encode", planned_in="Step 6")

    @staticmethod
    def invariants() -> Sequence[str]:
        """Properties an implementation must satisfy, checkable by test."""
        return (
            "The identity subspace of z_entity is unchanged by encoding.",
            "Padded entity slots produce zeroed latents and never influence real slots.",
            "Permuting the edge order does not change the output.",
            "Two scenes with identical AWRs produce identical latents.",
        )
