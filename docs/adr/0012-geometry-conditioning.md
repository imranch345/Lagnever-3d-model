# ADR 0012: Geometry conditions on anatomy and level of detail, never on presentation state

**Status:** ACCEPTED for the prototype
**Date:** Step 6
**Relates to:** `generation/neural/nn/model.py`, `generation/neural/nn/metrics.py`

## Context

Step 5 promised that a presentation edit (hide this, make that transparent) costs no
neural computation. Implementing the model forced the question of what the geometry
path is allowed to see. If the geometry tokens are conditioned on the requested entity
set, then hiding one structure changes the conditioning vector, and every other
structure's geometry moves. The promise would be false in the implementation while
still being true in the document.

## Decision

The geometry path conditions on two things only: the entity latent, which is built
from the AWR's identity, type, hierarchy and typed relations, and a one-hot level of
detail. It never sees visibility, opacity, animation state, or which entities were
requested.

Language still reaches the scene, through the control path, which changes the AWR and
therefore the scene. It does not reach the geometry of a structure that was not named.

`LagnavPrototype.geometry_is_state_invariant` reports this, and a test asserts that a
visibility edit leaves the geometry tokens and the per-entity fields bit-identical.

## Alternatives considered

**Condition geometry on the full request vector.** Lets the model adapt what it draws
to what was asked for, and is what the appearance-driven baseline does. Rejected for
the structured arm: it makes every edit global, which is the failure mode the project
exists to remove. It is retained in the baseline precisely so the difference is
measurable.

**Condition on the request but add a locality loss.** Approximates the same property
by training. Rejected: an approximate guarantee is not a guarantee, and the loss would
have to be tuned against the thing it is supposed to prove.

## Consequences

* Untouched-entity drift is exactly zero for the structured arm, not merely small.
  The metric reports `0.0`, and the token-identity check reports `1.0`.
* The structured arm generates every entity's geometry whether or not it was
  requested; presence selects what is composed. This costs compute on hidden entities
  and is the direct consequence of the AWR holding more than it shows.
* The conditioning path has no influence on geometry in this prototype, because there
  is no appearance to condition. Once materials exist it will condition those.
