# ADR 0002: One relational graph transformer with heads partitioned by graph

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/graph_encoder.py`

## Context

The AWR holds three graphs with different semantics. The specification is explicit
that `left_ventricle --pumps_to--> aorta` must not be representable the same way
as `left_ventricle --adjacent_to--> aorta`. Any encoder where the relation only
decides *whether* two entities attend to each other fails that requirement.

Scale matters here: 42 entities and 130 edges. A dense `64 x 64` attention matrix
is negligible, so there is no efficiency argument for sparse message passing, and
there is a real argument for global context.

## Decision

One transformer over entity tokens with attention heads partitioned by graph:
two structure heads, two spatial heads, two functional heads, two unmasked global
heads. Each head adds a typed bias projected from the relation embedding of the
edge it traverses. Direction is preserved because the reverse direction of an edge
uses the inverse relation's embedding, which is a different symbol. A pair
carrying several relations sums their biases.

## Alternatives considered

**Three separate encoders, one per graph, then fused.** Cleanest separation.
Rejected: triples the parameters, and makes cross-graph reasoning ("the septum
adjoins both ventricles *and* separates their flow") a fusion problem rather than
an attention problem. Revisit if the graphs turn out to need very different
depths.

**Relational message passing (R-GCN or RGAT).** The standard choice for typed
graphs, and it scales to far larger graphs than ours. Rejected for the prototype:
global context needs several hops, and at 42 nodes the efficiency it buys is
worth nothing. Revisit when scaling past a single organ.

**One untyped graph attention network.** Cheapest. Rejected outright: it fails
the stated requirement, because relation type would not change the message.

**Cross-attention between per-graph representations.** Expressive. Deferred:
more machinery than the evidence currently justifies.

## Consequences

* Head allocation is a hypothesis in configuration, so ablation A2 and A3 can
  change it without touching code.
* Dense masks cost `N_ENT^2` per head, which is fine at this scale and would need
  revisiting at ten times the entity count.
* The encoder sees connectivity and meaning separately, which makes the typed
  bias directly inspectable during debugging.

## How this could be wrong

If the global heads dominate and the masked heads contribute nothing, the
partition is decoration. The mask builder is implemented and tested precisely so
that this can be measured rather than assumed.
