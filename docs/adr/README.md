# Architecture decision records

One file per significant decision. Each records the alternatives that were
considered, why they were not chosen, and the conditions that would make them
worth revisiting.

Step 4 decisions (the deterministic AWR foundation) are in
[../architecture-decisions.md](../architecture-decisions.md).

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-anatomical-latent.md) | Four-tier anatomical latent with a write-protected identity subspace | PROPOSED |
| [0002](0002-graph-encoder.md) | One relational graph transformer with heads partitioned by graph | PROPOSED |
| [0003](0003-3d-representation.md) | Per-entity latent token blocks decoded into per-entity implicit fields | PROPOSED |
| [0004](0004-multimodal-alignment.md) | Prototype-hub alignment anchored on the entity codebook | PROPOSED |
| [0005](0005-editing-strategy.md) | Masked, entity-local re-decoding | PROPOSED |
| [0006](0006-level-of-detail.md) | Nested geometry token prefixes for level of detail | PROPOSED |
| [0007](0007-language-program-interface.md) | Language emits verifiable AWR programs, never geometry | PROPOSED |
| [0008](0008-training-curriculum.md) | Six-stage curriculum whose first two stages need no external data | PROPOSED |
| [0009](0009-no-framework-in-step-5.md) | No deep-learning framework in the design phase | ACCEPTED |
| [0010](0010-dependency-tiers.md) | Two dependency tiers, with PyTorch confined to the second | ACCEPTED |
| [0011](0011-occupancy-and-teacher-forced-frames.md) | Occupancy fields with teacher-forced canonical frames | ACCEPTED |
| [0012](0012-geometry-conditioning.md) | Geometry conditions on anatomy and level of detail only | ACCEPTED |
| [0013](0013-tier0-synthetic-corpus.md) | A tier-0 synthetic corpus as the first testbed | ACCEPTED |
| [0014](0014-strict-relationship-metric.md) | Relationship accuracy counts unplaceable entities as failures | ACCEPTED |

ADRs 0001 to 0009 are Step 5 design proposals. ADRs 0010 to 0014 are Step 6
implementation decisions, taken while building the prototype and marked ACCEPTED
because they describe code that exists and is tested. Neither kind is a validated
research result.
