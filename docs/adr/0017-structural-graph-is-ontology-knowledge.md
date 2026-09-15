# ADR 0017: Structural edges come from the ontology, not from measurement

**Status:** ACCEPTED
**Date:** Step 7
**Relates to:** `generation/neural/nn/whole_organ.py`, `tests/test_whole_organ_corpus.py`

## Context

The Step 7 corpus measures relations from the generated organ rather than asserting them
from the ontology, which is what makes relationship graphs vary between scenes. Spatial
relations (`left_of`, `superior_to`) and functional ones (`opens_into`) are genuine
measurements of a particular organ.

Structural relations are not. Which structures connect to and are continuous with which
does not change when an organ is mirrored; it is ontology knowledge about what kind of
thing a heart is. Nothing supplied them, so the structure graph carried **zero edges**,
and ablation `A2`, nominally "structure graph only", was a no-graph arm that still paid
for a full graph encoder. Its matching the other arms in the first two suites carried no
information.

## Decision

Every scene's relationship graph includes the ontology's structural edges between the
entities that scene contains: 15 edges for the whole-organ entity set. Spatial and
functional edges remain measured per scene and are never taken from the ontology, so the
two kinds of knowledge are not mixed.

A test asserts that all three typed graphs carry edges, and a second test asserts that
the structural edges are identical across scenes.

## Consequences

* All three typed graphs are populated, so the Anatomical World Representation the
  prototype consumes is complete.
* `A2` is a faithful structure-only arm.
* **How `A2` must be read:** its edges are the same for every scene. A constant input can
  act as a prior but cannot distinguish one arrangement from another. `A2` matching a
  no-graph arm on arrangement-sensitive metrics is therefore the expected outcome, and is
  not evidence against typed graphs in general. Only the spatial and functional graphs
  vary per scene, and only they can carry arrangement information.

## Alternatives considered

**Measure structural relations from the organ too.** Rejected: adjacency and containment
measured from a point cloud are already covered by the spatial graph. Deriving
`continuous_with` geometrically would duplicate the spatial measurement under a
structural name and make the two graphs redundant by construction, which is the opposite
of what the ablation is meant to test.

**Leave the structure graph empty and report `A2` as void.** Rejected: it would leave the
partonomy untested, and a Step 7 whose job is to decide which architectural ideas are
necessary should not leave one of the three typed graphs unexercised.
