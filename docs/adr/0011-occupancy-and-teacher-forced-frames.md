# ADR 0011: Occupancy fields, with teacher-forced canonical frames during training

**Status:** ACCEPTED for the prototype
**Date:** Step 6
**Relates to:** `generation/neural/nn/geometry.py`, `generation/neural/nn/model.py`

## Context

Step 5 recommended per-entity implicit fields in canonical entity frames and left two
things open: whether the field is occupancy or signed distance, and how the frame and
the field are trained together when the frame is itself predicted.

The second question is the awkward one. If the decoder consumes a predicted frame,
early frame error moves every query point into the wrong space, the occupancy signal
becomes noise, and neither component gets a usable gradient.

## Decision

**Occupancy with a binary cross-entropy objective.** The tier-0 primitives give exact
inside and outside labels at any point, while an exact signed distance to a union of
primitives is not available analytically. A signed distance field would also want an
eikonal term, which Step 5 deferred.

**Teacher-forced frames.** During training the field decoder consumes the
ground-truth frame while the frame head is supervised by its own objective. Evaluation
reports both the teacher-forced field quality and the end-to-end result with predicted
frames, so the split is visible in the numbers rather than hidden in the loop.

## Alternatives considered

**Predicted frames throughout training.** Honest end to end, and what deployment
requires. Rejected for the prototype: it couples two unconverged components and makes
a failure impossible to attribute. Revisit by annealing from ground-truth to predicted
frames once both parts train stably on their own.

**Signed distance fields.** Better surfaces, better gradients near the boundary, and
the usual choice in the literature. Deferred: exact supervision is not available on
this corpus without an approximation whose error would be confounded with the model's.

**No canonical frames at all, decoding in scene space.** Simpler. Rejected because it
throws away the separation between where a structure is and what shape it has, which
is the property that makes spatial relations analytically checkable.

## Consequences

* The frame head receives no gradient from the occupancy path, which a test asserts so
  the coupling cannot be reintroduced silently.
* Reported geometry quality is field quality given correct placement. End-to-end
  quality is a separate number and is the one that matters for deployment.
* Occupancy makes the Chamfer metric a point-set distance on shared query samples
  rather than a surface distance. Documented wherever it is reported.
