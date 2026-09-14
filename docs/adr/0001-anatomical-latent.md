# ADR 0001: Four-tier anatomical latent with a write-protected identity subspace

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/latent.py`

## Context

The latent has to carry identity, type, hierarchy, three kinds of relationship,
geometry, topology hints, material, animation, audience level, visibility,
correspondence and multiple resolutions. The tempting move is one wide vector per
scene with everything in it.

Three of those requirements are not capacity requirements at all. Addressing an
entity ("the left ventricle"), rewriting one entity's geometry, and mapping a
geometry component back to an anatomical entity are all *indexing* problems. A
flat latent has no index.

## Decision

Four tiers, each addressable on its own axis:

| tier | shape | addressed by |
| --- | --- | --- |
| scene | `[B, D_SCENE]` | batch index |
| entity | `[B, N_ENT, D_ENT]` | entity codebook index |
| relation | `[B, E_REL, D_REL]` | edge index |
| geometry | `[B, N_ENT, K_GEO, D_GEO]` | entity index, then token index |

The entity latent is composed of named subspaces with documented widths, and the
identity subspace `[0, 96)` is **write-protected**: the graph encoder updates the
context slice only, and identity is re-attached unchanged after every layer.

Presentation state (visibility, opacity, animation phase, level of detail) is
**not in the latent**. It arrives as an explicit nine-channel vector that
conditions decoders.

## Alternatives considered

**Single flat scene latent.** Simplest, and it is what the appearance-driven
baseline uses. Rejected because part addressing, local editing and correspondence
all become learned behaviours that can fail silently. Revisit if the ablation
shows the entity axis contributes nothing.

**Entity latents without a protected identity slice.** Fewer constraints, more
capacity for the encoder. Rejected because identity preservation then depends on
training rather than on structure, and the whole Step 4 foundation is built on
identity being structural. Revisit if the protected slice measurably starves the
encoder.

**State inside the entity latent.** Convenient: one vector per entity holds
everything. Rejected because it makes "make this transparent" and "make this a
different structure" the same kind of operation in the same space. Revisit never;
this is close to a first principle for the project.

## Consequences

* Identity probing of `z_entity` is exact by construction, so probe accuracy is
  not evidence of learned anatomy. Any experiment reporting it must say so.
* Per-entity capacity is fixed in advance by `D_ENT` and `K_GEO`.
* The entity axis is padded, so every consumer needs the mask.
* Ablations can zero a named subspace and measure the effect, because the spans
  are declared rather than emergent.

## How this could be wrong

If anatomical structure is better captured by a continuous field than by discrete
entities, the entity axis is an expensive constraint rather than a help, and A1
in the first experiment is where that would show up.
