# ADR-L1-002: Level 1 geometry is reference geometry from real CT, not generated geometry

**Status:** ACCEPTED
**Date:** Level 1, 2026-10-15
**Relates to:** `geometry/representation.py` (`MeshRepresentation`), `docs/LEVEL_1_PLAN.md`

## Context

No step so far produces geometry a person can look at. `SymbolicGeometry` holds component
slots with no payload, and the research prototype decodes occupancy on a synthetic corpus.
Level 1's goal is a working language-controlled heart and lungs. A learned generator is
Level 2, and it needs a GPU and real training data that the project does not have yet.

## Decision

Level 1 fills the existing `MeshRepresentation` slot with meshes extracted from real CT
segmentations, with one mesh per AWR entity. These meshes are **reference geometry**: they
come from one chosen subject, are never presented as generated, and carry their source
subject, label and license.

The founding rules hold:

* The AWR stays the source of truth.
* A mesh is a view, attached by component id, and never decides which entity it belongs to.
* Edits change entity state, never the mesh.

Entities the data cannot show have no mesh and are reported as having none: valves, the
superior vena cava, coronaries, the conduction system, septa and wall layers other than the
myocardium.

## Alternatives considered

**Procedural shapes, such as ellipsoids at entity frames.** Rejected as the deliverable: it
would look like progress and teach nothing about real anatomy. It may still serve as a
placeholder in viewer tests.

**Wait for Level 2's generator.** Rejected: a viewer, a mesh pipeline and the engine wiring are
needed either way, and building them against real meshes first finds integration problems
early.

## Consequences

* Level 2 replaces the mesh source behind the same interface, and nothing above it changes.
* Anatomical claims are bounded by one subject's segmentation. The viewer and README say
  "research CT, not validated".
