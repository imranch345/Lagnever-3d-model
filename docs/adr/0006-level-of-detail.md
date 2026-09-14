# ADR 0006: Nested geometry token prefixes for level of detail

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/three_d_latent.py`

## Context

Step 4 established the principle that a level of detail changes what is *visible*,
never what *exists*, and never what something *is*. The neural representation has
to keep that true. The question is whether detail lives in separate per-level
latents or in one latent decoded at different depths.

## Decision

One geometry token block per entity, ordered coarse to fine. A level of detail
reads a **prefix**: 4, 8, 16, 24 and 32 tokens for levels 0 to 4. The schedule is
validated to be non-decreasing and to end at the declared token count.

## Alternatives considered

**Separate latents per level of detail.** Each level can be tuned independently
and quality per level may be higher. Rejected: five latents per entity is five
chances for them to disagree about what the entity is, which is exactly the
failure "show more detail" must not have. This is ablation A5. Revisit if nesting
measurably caps quality at the finest level.

**One latent, one fixed decode resolution, with post-hoc mesh decimation.** Simple
and standard in graphics. Rejected: decimation removes geometry, not anatomical
detail, so a school view would be a blurry medical view rather than a simpler one.

**Learned detail selection.** Let the model decide what to show at each level.
Deferred: it moves an educational decision into the weights, where it cannot be
inspected or corrected by a curriculum designer.

## Consequences

* Identity cannot change with detail, because the prefix is part of the same
  block.
* Coarse decoding is cheaper, which makes low levels fast.
* Token capacity is allocated in advance and the early tokens carry more
  responsibility, which may need nested-dropout style training to work.
* The schedule is configuration, so an educational review can change it.

## How this could be wrong

Nested representations can underuse later tokens, capping detail at the finest
level. If so, the fix is a longer schedule or per-level refinement heads, not
separate latents.
