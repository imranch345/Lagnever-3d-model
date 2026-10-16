# ADR-L2-001: "Make it larger" resizes within the shape model, recorded in the AWR

**Status:** ACCEPTED
**Date:** Level 2a, 2026-10-16
**Relates to:** `shapes/resize.py`, `language/program.py`, `language/layer.py`,
`viewer/server.py`, [ADR 0007](0007-language-program-interface.md),
[ADR-L1-005](ADR-L1-005-language-layer-claude-programs.md), `docs/LEVEL_2_PLAN.md` §3.7

## Context

Level 2a gives per-entity shape models learned from about 150 real thoraxes. The Level 2 plan
asks for one controlled change through the AWR: `resize`, with an entity and a percentage.
Four constraints shape how:

* **The research vocabulary is frozen.** Adding an intent to `reasoning.parser.Intent` would
  need a matching `ProgramOpType` in `generation/neural/language_encoder.py`, whose enum
  sizes a learned vocabulary. That changes model shapes, and with them the bit-exact
  reproduction the frozen research results depend on.
* **The AWR must record what was asked.** Geometry is a view of it (ADR 0007).
* **The result must be plausible anatomy.** Scaling a mesh is easy, and it is not what an
  enlarged ventricle looks like. Nor may it push into its neighbours.
* **Resize must work without the language model.** The language model path is off by
  default.

## Decision

1. **A product operation, not a research intent.** `resize` joins `language.program.OPERATIONS`
   with exactly one target and a signed percentage of linear size, from -30 to +30.
   `reasoning.parser.Intent` and `ProgramOpType` do not change.
2. **A deterministic rule in the language layer.** It covers forms such as "make the left
   ventricle 20% larger", "shrink the right atrium by 10%" and "enlarge the left lung 15%".
   They resolve through the engine's own entity resolver. Claude can emit the same
   operation.
3. **The AWR records size in a field that already exists.** `EntityState.transform.scale`
   is set through the existing `SetTransform` edit, via the engine's `SceneEditor`, so the
   change is validated, logged and visible in the state. The research code does not read
   `transform`. Size is cumulative: +10% twice is 1.21. It is written only after the
   geometry succeeds.
4. **Geometry is realised by the entity's own shape model.**
   - The target is the current shape scaled about its centroid.
   - It is projected onto the model, with weights clipped at 3 standard deviations, and the
     model's shape is used. The result is the closest shape the population supports, not a
     magnified copy.
   - The reply states the size and volume actually achieved, and how far the shape is from
     plain scaling.
5. **Only the resized entity changes.** Every other entity keeps its current mesh exactly.
6. **Implausible results are refused, not applied.** The resized entity goes through the
   validity checks: folding, overlap deeper than 2 mm with every other entity, and volume
   range. Their limits are calibrated on the training subjects (`validity.json`). On
   failure nothing changes, and the reply says which check failed.
7. **A new geometry resets sizes.** Generating a sample or returning to the real subject
   resets every entity's scale to 1, and the reply says so. The AWR never claims a size the
   geometry does not have.
8. **It needs a model.** Without `--model` the reply says how to enable resize, and nothing
   changes.

## Alternatives considered

**A new parser intent and engine handler.** This was the cleanest route inside the engine.
It was rejected because it changes the research vocabulary (see Context).

**Scaling the mesh in the viewer.** Instant, and shape-blind: a ventricle scaled by 30%
runs into the myocardium. The AWR would claim nothing about it. Rejected.

**Resizing through the joint model.** This would keep every entity consistent. It was
rejected because it changes the other entities, which a request for one structure must not
do. Consistency is instead enforced by the overlap check.

## Consequences

* A sentence changes one structure's shape. The AWR records it, the shape is within what
  real anatomy shows, and it is checked against its neighbours before it appears.
* The base shape must be in model correspondence:
  - a generated sample, from its coefficients;
  - a real subject, from its cached registration.

  A subject without a registration cannot be resized, and the reply says so.
* The research parser, neural vocabulary and frozen results are untouched.

## How this could be wrong

"20% larger" may be meant as volume, not linear size. Linear is used because it is what a
viewer shows; the reply always states both. The overlap limits come from coarse templates
that already overlap a little in real subjects. A resize that stays within those limits can
still touch a neighbour more than real anatomy does.
