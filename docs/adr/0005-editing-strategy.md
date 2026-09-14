# ADR 0005: Masked, entity-local re-decoding

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/editing.py`

## Context

"Generate a heart", then "make the left ventricle transparent" must not
regenerate the heart. Step 4 already handles state edits symbolically. The open
question is what happens when an edit actually changes geometry.

## Decision

Edits are classified into four kinds, and the classification decides the work:

| kind | example | geometry recomputed |
| --- | --- | --- |
| state only | "make the left ventricle transparent" | none |
| level of detail | "show more detail" | entities whose token prefix grows |
| structural | remove or replace an entity | the entity and its immediate context |
| geometric | "thicken the ventricle wall" | that entity alone |

For geometric edits: rewrite the affected entity's token block and re-decode that
entity, with neighbouring entities and the scene latent as frozen context. Other
blocks are not written, so they cannot change.

`plan_edit` implements the classification today and is tested against the Step 4
engine, so "this edit recomputes nothing" is a checkable statement rather than a
claim.

## Alternatives considered

**Full regeneration from an amended prompt.** Trivial. Rejected: it produces a
different object, losing identity and every prior edit. It is the behaviour the
project exists to replace.

**Global latent modification.** Simple, one latent to manipulate. Rejected: no
locality, so a locality loss has to approximate what factorisation gives for free.

**Local diffusion or flow over the affected tokens.** Strong generative quality
and natural handling of ambiguity. Deferred: slower at inference, another training
loop before the basics work, and stochastic, so reproducibility needs seed
discipline. Revisit if deterministic re-decoding proves blurry or short on diversity.

**Graph modification alone.** Exactly right for state and structural edits, and
adopted for those. Insufficient for purely geometric changes such as wall
thickness.

## Consequences

* Locality is structural for untouched entities and measurable for touched ones.
* A state edit costs no neural computation at all, which is the strongest
  practical consequence of the whole design.
* Seams at the boundary of an edited entity need managing.
* The edited entity must still fit its neighbours, so context conditioning is
  required.

## How this could be wrong

If context conditioning is too weak, locally re-decoded parts will not fit their
neighbours, and the visible result may be worse than a slightly non-local method
that keeps the scene coherent.
