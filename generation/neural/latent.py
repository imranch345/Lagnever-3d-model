"""The proposed Anatomical Latent Representation.

STATUS: PROPOSED, UNVALIDATED. This module declares a design and its contracts;
it trains nothing and asserts nothing about performance.

The design in one sentence: **anatomy is a set of typed entity latents related by
typed edges, with identity held in a write-protected subspace, geometry factorised
per entity, and presentation state kept outside the latent entirely.**

Four tiers
----------

============  ==================================  ==========================================
tier          shape                               what it holds
============  ==================================  ==========================================
scene         ``[B, D_SCENE]``                    organ identity, frame, audience level, style
entity        ``[B, N_ENT, D_ENT]``               one latent per anatomical entity
relation      ``[B, E_REL, D_REL]``               one embedding per typed edge
geometry      ``[B, N_ENT, K_GEO, D_GEO]``        a token block per entity, ordered coarse to fine
============  ==================================  ==========================================

Why four tiers rather than one
------------------------------

A single flat latent is simpler and is the appearance-driven baseline the first
experiment compares against. It is rejected for Lagnav because three of the
system's requirements are *addressing* requirements, not capacity requirements:
"make the left ventricle transparent" needs a place to point at, local editing
needs a slice to rewrite, and correspondence needs an axis to index. A flat latent
has none of these, so every one of them has to be learned and can silently fail.

The cost is real and recorded: more machinery, a padded entity axis, and a
representation whose capacity per entity is fixed in advance.

Identity is structural, not learned
-----------------------------------

``z_entity`` is split into an identity slice and a context slice. The identity
slice is written once from the entity codebook embedding and is **not updated by
the graph encoder**; only the context slice is contextualised. So "the entity id
survived the edit" is a property of the tensor layout, in the same way that Step 4
made it a property of a frozen dataclass. A linear probe for entity identity is
exact by construction, which also means identity probing cannot be used as
evidence that the model learned anything: that is stated here so no experiment
misreads it later.

State is not in the latent
--------------------------

Visibility, opacity, animation phase and level of detail never enter
``z_entity``. They arrive as an explicit 9-channel state vector that modulates
*decoders* (proposed: FiLM-style conditioning). The reason is the invariant from
Step 4: a presentation change must not be able to alter anatomical identity. If
opacity were inside the latent, "make it transparent" and "make it a different
structure" would be the same kind of operation in the same space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from awr.errors import ContractError
from generation.neural.contracts import (
    CONTRACTS,
    Contract,
    DType,
    TensorSpec,
    dim,
)

__all__ = [
    "SubspaceSlice",
    "ENTITY_SUBSPACES",
    "EntityLatentLayout",
    "ENTITY_LATENT_LAYOUT",
    "LATENT_CONTRACT",
    "LatentTier",
    "LATENT_TIERS",
]


@dataclass(frozen=True, slots=True)
class SubspaceSlice:
    """One named span of the entity latent."""

    name: str
    width: int
    source: str
    learned: bool
    write_protected: bool
    rationale: str

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ContractError(f"Subspace {self.name!r} must have positive width.")


ENTITY_SUBSPACES: tuple[SubspaceSlice, ...] = (
    SubspaceSlice(
        name="identity",
        width=96,
        source="learned embedding table indexed by the entity codebook",
        learned=True,
        write_protected=True,
        rationale=(
            "Persistent anatomical identity. Written once from the codebook and never updated "
            "by the encoder, so identity cannot drift during contextualisation or editing."
        ),
    ),
    SubspaceSlice(
        name="type",
        width=48,
        source="embeddings of anatomy_type, semantic_role and laterality, summed",
        learned=True,
        write_protected=False,
        rationale=(
            "Lets the model generalise across entities of the same kind, which is the only "
            "route to transfer when a new ontology adds a chamber or a valve."
        ),
    ),
    SubspaceSlice(
        name="hierarchy",
        width=32,
        source="depth embedding plus a projection of the parent identity embedding",
        learned=True,
        write_protected=False,
        rationale=(
            "Explicit tree position. Available symbolically from the AWR, so it is injected "
            "rather than inferred; the graph encoder should spend capacity on relationships "
            "the tree does not already state."
        ),
    ),
    SubspaceSlice(
        name="geometry_summary",
        width=48,
        source="mean pool of the entity's geometry token block",
        learned=True,
        write_protected=False,
        rationale=(
            "Carries shape information back into anatomical reasoning, so spatial relations "
            "can be checked against geometry rather than asserted independently of it."
        ),
    ),
    SubspaceSlice(
        name="free",
        width=32,
        source="learned, unconstrained",
        learned=True,
        write_protected=False,
        rationale=(
            "Deliberate spare capacity. Without it, every ablation of a named subspace also "
            "removes total capacity, and the ablation measures the wrong thing."
        ),
    ),
)
"""Proposed input composition of an entity latent. Widths are a prototype hypothesis."""


@dataclass(frozen=True, slots=True)
class EntityLatentLayout:
    """The entity latent as a set of named, contiguous subspaces."""

    subspaces: tuple[SubspaceSlice, ...]

    @property
    def width(self) -> int:
        """Total entity latent width."""
        return sum(subspace.width for subspace in self.subspaces)

    def span(self, name: str) -> tuple[int, int]:
        """Half-open ``[start, end)`` index range of a named subspace."""
        offset = 0
        for subspace in self.subspaces:
            if subspace.name == name:
                return offset, offset + subspace.width
            offset += subspace.width
        raise ContractError(
            f"No subspace {name!r}. Declared: "
            f"{', '.join(s.name for s in self.subspaces)}."
        )

    def protected_spans(self) -> tuple[tuple[int, int], ...]:
        """Ranges the graph encoder must not write to."""
        return tuple(
            self.span(subspace.name) for subspace in self.subspaces if subspace.write_protected
        )

    def context_width(self) -> int:
        """Width the encoder is free to contextualise."""
        return self.width - sum(s.width for s in self.subspaces if s.write_protected)

    def validate_against(self, dimension: str = "D_ENT") -> None:
        """Check the layout fills the declared latent width exactly."""
        declared = dim(dimension).size
        if declared is not None and declared != self.width:
            raise ContractError(
                f"Entity subspaces sum to {self.width} but dimension {dimension!r} is declared "
                f"as {declared}. Update whichever is wrong; they are one decision."
            )

    def table(self) -> str:
        """Markdown table of the layout, for the design document."""
        lines = [
            "| subspace | width | span | learned | write-protected | source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for subspace in self.subspaces:
            start, end = self.span(subspace.name)
            lines.append(
                f"| `{subspace.name}` | {subspace.width} | `[{start}, {end})` | "
                f"{'yes' if subspace.learned else 'no'} | "
                f"{'yes' if subspace.write_protected else 'no'} | {subspace.source} |"
            )
        return "\n".join(lines)


ENTITY_LATENT_LAYOUT = EntityLatentLayout(ENTITY_SUBSPACES)
ENTITY_LATENT_LAYOUT.validate_against()


@dataclass(frozen=True, slots=True)
class LatentTier:
    """One tier of the anatomical latent, with the reason it exists separately."""

    name: str
    tensor: str
    dims: tuple[str, ...]
    holds: str
    addressed_by: str
    alternative_if_merged: str


LATENT_TIERS: tuple[LatentTier, ...] = (
    LatentTier(
        name="scene",
        tensor="z_scene",
        dims=("B", "D_SCENE"),
        holds="Organ identity, canonical frame convention, audience level, global style.",
        addressed_by="the batch index",
        alternative_if_merged=(
            "Folding scene context into every entity latent duplicates it N times and makes "
            "a global change require touching every entity."
        ),
    ),
    LatentTier(
        name="entity",
        tensor="z_entity",
        dims=("B", "N_ENT", "D_ENT"),
        holds="Anatomical identity, type, tree position and shape summary per entity.",
        addressed_by="the entity codebook index",
        alternative_if_merged=(
            "A flat scene latent gives commands nothing to point at; part control and local "
            "editing both become learned behaviours that can fail silently."
        ),
    ),
    LatentTier(
        name="relation",
        tensor="z_relation",
        dims=("B", "E_REL", "D_REL"),
        holds="Typed, directed edge embeddings, including graph, flow direction and granularity.",
        addressed_by="the edge index",
        alternative_if_merged=(
            "Encoding relations only as attention connectivity makes 'pumps_to' and "
            "'adjacent_to' indistinguishable, which the specification forbids."
        ),
    ),
    LatentTier(
        name="geometry",
        tensor="z_geometry",
        dims=("B", "N_ENT", "K_GEO", "D_GEO"),
        holds="A coarse-to-fine token block per entity, decoded into that entity's field.",
        addressed_by="the entity index, then the token index",
        alternative_if_merged=(
            "A shared geometry latent forces part identity to be recovered by segmentation "
            "after the fact, which is exactly the failure mode the project exists to avoid."
        ),
    ),
)
"""The four proposed tiers and why each is addressable on its own axis."""


LATENT_CONTRACT = CONTRACTS.register(
    Contract(
        name="anatomical_latent",
        purpose=(
            "The proposed Anatomical Latent Representation: the interface between anatomical "
            "reasoning and every decoder."
        ),
        specs=(
            TensorSpec(
                "z_scene",
                ("B", "D_SCENE"),
                DType.FLOAT32,
                "Scene-level anatomical context.",
                normalization="layer-normalised",
            ),
            TensorSpec(
                "z_entity",
                ("B", "N_ENT", "D_ENT"),
                DType.FLOAT32,
                "Per-entity latent. Slice [0, 96) is the write-protected identity subspace.",
                normalization="layer-normalised on the context slice only",
                mask="entity_mask",
                padding_value=0.0,
            ),
            TensorSpec(
                "entity_mask",
                ("B", "N_ENT"),
                DType.BOOL,
                "True for real entity slots. Carried through from the AWR batch.",
            ),
            TensorSpec(
                "z_relation",
                ("B", "E_REL", "D_REL"),
                DType.FLOAT32,
                "Typed directed edge embedding, built from relation type plus edge features.",
                mask="edge_mask",
                padding_value=0.0,
            ),
            TensorSpec(
                "edge_mask",
                ("B", "E_REL"),
                DType.BOOL,
                "True for real edges. Carried through from the AWR batch.",
            ),
            TensorSpec(
                "z_geometry",
                ("B", "N_ENT", "K_GEO", "D_GEO"),
                DType.FLOAT32,
                "Per-entity geometry tokens, ordered coarse to fine.",
                mask="entity_mask",
                padding_value=0.0,
            ),
            TensorSpec(
                "entity_frame",
                ("B", "N_ENT", "N_FRAME"),
                DType.FLOAT32,
                "Per-entity pose: translation, log-scale and a 6D rotation basis.",
                normalization="translation in the unit cube, log-scale in [-3, 3]",
                mask="entity_mask",
                padding_value=0.0,
                channels=(
                    "tx", "ty", "tz",
                    "log_sx", "log_sy", "log_sz",
                    "r00", "r01", "r02", "r10", "r11", "r12",
                ),
            ),
            TensorSpec(
                "entity_state",
                ("B", "N_ENT", "N_STATE"),
                DType.FLOAT32,
                "Explicit presentation state used to condition decoders, never identity.",
                normalization="per-channel, see features.ENTITY_STATE_LAYOUT",
                mask="entity_mask",
                padding_value=0.0,
                value_range=(-1.0, 1.0),
            ),
        ),
    )
)
"""Contract for the anatomical latent bundle."""


def describe_latent(bindings: Mapping[str, int] | None = None) -> dict[str, Sequence[int]]:
    """Resolved shapes of every latent tensor, for documentation and tests."""
    resolved: dict[str, Sequence[int]] = {}
    for spec in LATENT_CONTRACT:
        try:
            resolved[spec.name] = spec.resolve(bindings)
        except ContractError:
            resolved[spec.name] = ()
    return resolved
