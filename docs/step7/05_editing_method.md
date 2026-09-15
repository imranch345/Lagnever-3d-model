# Experiments 8 and 9 — Persistent local geometric editing. Method

Step 6 shipped the editing interfaces and never tested them, so the claim that
per-entity geometry token blocks buy **local** editing was unsupported. This is the test.

## What an edit is here

The generator applies a controlled change to the organ parameters and rebuilds the organ.
Nothing about the result is assumed: the before-organ and the after-organ are evaluated
at the **same points**, and the fraction of each entity's points whose owner changed is
measured. Entities below the threshold are the ones the edit genuinely left alone, and
they are what leakage is scored against.

| Operation | Parameter scaled | Declared target |
| --- | --- | --- |
| `thicken_lining` | `wall_fraction` × 1.45 | endocardium |
| `thin_lining` | `wall_fraction` × 0.68 | endocardium |
| `enlarge_ventricle` | left ventricle × 1.28 | left ventricle |
| `shrink_ventricle` | left ventricle × 0.76 | left ventricle |
| `enlarge_atrium` | left atrium × 1.32 | left atrium |
| `widen_vessel` | aorta × 1.60 | aorta |
| `thicken_valve` | mitral valve × 1.75 | mitral valve |

The two wall operations originally named the myocardium. Measurement showed the
parameter scales the endocardial and epicardial linings, and the myocardium is whatever
tissue those linings do not claim, so it was the third-most-changed entity rather than
the edited one. The declared targets were corrected and a test now requires every
declared target to be an entity its operation actually alters.

## Anatomical structures are coupled, and the metric accounts for it

Measured mean change fraction, six scenes, for `enlarge_ventricle`:

| Entity | Change |
| --- | --- |
| aortic valve | 0.97 |
| mitral valve | 0.81 |
| interventricular septum | 0.74 |
| endocardium | 0.61 |
| left ventricle | 0.50 |

The edited entity is the fifth-most-changed. This is not a defect. Valves sit on the
ventricle's surface and move with it; the septum is defined by the tissue between two
cavities. **A local edit to an organ has genuinely non-local consequences**, and a
metric that counted those as errors would be measuring anatomy, not the model.

The pre-registered metric already handles this: the leakage denominator is "entities the
generator did not alter", so coupled entities are excluded rather than counted as
failures.

## The three arms

| Arm | Edit mechanism | What it tests |
| --- | --- | --- |
| `E-LOCAL` | the head may rewrite **only** the target's geometry token block | locality built into the representation |
| `E-FREE` | same head, same capacity, may rewrite every entity's block | is locality learned, or must it be built in? |
| `E-REGEN` | no head: re-decode the whole scene from edited parameters | the "just regenerate everything" alternative |

The backbone is frozen in all three arms, so the comparison is between edit mechanisms
rather than between differently trained generators. `E-REGEN` is the control that decides
the architecture question: **if regenerating everything is as local as editing one
entity, then per-entity token blocks are not buying persistent local editing** and the
claim should be withdrawn.

`E-REGEN`'s regenerated scene is evaluated at the before-scene's points, so a difference
between arms cannot be an artefact of resampling.

## The head

The head reads entity latents, current geometry tokens and an encoding of the requested
edit, and produces a delta on the token block. Two properties are structural rather than
learned, and are reported as such:

* It is initialised to the exact identity, so an untrained head leaves the scene bit for
  bit unchanged. A measured change cannot be initialisation noise.
* It never writes the entity latent, so the write-protected identity subspace is carried
  through untouched. An edited entity is still the same entity by construction.

## Metrics

Pre-registered, unchanged:

| Metric | Definition |
| --- | --- |
| `edit_target_accuracy` | IoU between the target's predicted post-edit region and the generator's true post-edit region |
| `local_edit_locality` | mean absolute occupancy change inside the edited entity divided by the mean change in entities the generator did not alter |

Declared in amendment A6 before running:

| Metric | Definition |
| --- | --- |
| `unrelated_entity_drift` | the leakage denominator on its own |
| `identity_preservation` | cosine similarity of the identity subspace before and after; 1.0 for the edit head by construction, and not guaranteed for regeneration, which re-derives the latent |

Declared in amendment A7 for the edit → reason → edit loop:

| Metric | Definition |
| --- | --- |
| `edit_persistence` | IoU between the first target's region after the second edit and after the first. An edit forgotten when the next arrives is not persistent. |
| `sequence_consistency` | share of steps where a relation read after the first edit agrees with the generator's true post-edit organ |
| `commutativity_gap` | difference in final occupancy between applying two disjoint edits in either order |

## One implementation choice, disclosed

Occupancy change is measured on the **composed** scene, that is on each entity's
probability of owning a point, not on its independent field. In the target-scoped arm the
untouched entities' independent fields are identically equal before and after by
construction, which would make the locality ratio infinite for structural reasons and
comparable across no arms at all. Composition is what the plan's phrase "occupancy
change" refers to and is what a user would see.

`commutativity_gap` is likewise expected to be exactly zero for both mechanisms: the head
has no cross-entity mixing, and the generator's parameter edits compose commutatively.
Order independence here is a property of the construction, not evidence that anything was
learned.

## What counts as a negative result

* If `E-REGEN` matches `E-LOCAL` on locality, per-entity geometry token blocks are marked
  SIMPLIFY or REMOVE for the purposes of editing.
* If `edit_persistence` is low, persistent editing is reported as **not working**, not as
  partially working.
* If `E-FREE` matches `E-LOCAL`, locality does not need to be built into the
  architecture and the scoping mechanism is unnecessary.
