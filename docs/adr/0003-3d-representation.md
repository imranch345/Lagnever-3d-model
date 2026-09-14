# ADR 0003: Per-entity latent token blocks decoded into per-entity implicit fields

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/three_d_latent.py`

## Context

Nine candidate representations were compared on ten criteria in
`generation/neural/three_d_latent.py`. The weights used are declared there and
favour semantic correspondence and editability over visual quality, because those
are what the research hypothesis is about.

Only the top-ranked candidate and the reasoning behind it should be read from that
table. The ordering among the rejected candidates is an artefact of weighting
several interacting criteria, and it is not meaningful.

## Decision

Each entity owns `K_GEO` geometry tokens, ordered coarse to fine. Each entity's
field is decoded in its own canonical frame, predicted as translation, log-scale
and a 6D rotation basis. The scene field is a composition of entity fields, and a
part-label head attributes every query point to an entity. Meshes are extracted
only at export.

## Alternatives considered

**Single implicit field for the whole organ.** Better inter-part continuity, no
composition seams, simpler training, and strong quality. Rejected because part
identity must then be recovered by segmentation after the fact, and an edit to one
structure rewrites the weights that produce every other. This is ablation A4, so
the decision is tested rather than assumed.

**Triplane.** Excellent quality per unit of compute. Rejected because entities are
entangled in shared planes, so a local edit is a global write; per-entity triplanes
restore locality but multiply memory by the entity count. Revisit at research
scale as a shared coarse context underneath per-entity tokens.

**Dense or sparse voxels.** Part labels are trivial. Rejected on resolution: valve
leaflets need detail that cubic memory will not pay for.

**Point clouds.** Cheap and easy to label. Rejected: no surface and no topology, so
"is this chamber closed" cannot be asked.

**Gaussian splats.** Best appearance and view consistency. Rejected: appearance
first, no surface, no topology. It optimises the quantity this project refuses to
treat as truth. Revisit for a fast preview layer.

**Direct mesh generation.** It is the export format. Rejected for now: discrete
connectivity is hard to generate and harder to train at small data scale.

## Consequences

* Correspondence becomes an index rather than an inference.
* Re-decoding one entity is the natural unit of work, so edits are local by
  construction.
* Composition seams between neighbouring entities are a real risk and are called
  out as such; if they dominate the geometry metrics, A4 is the fallback.
* Decoding quality is expected to trail a well-tuned single field at first.

## How this could be wrong

The synthetic tier-0 corpus is generated per part, which may favour a per-part
representation. A tier-1 replication on real segmented data is required before any
general claim.
