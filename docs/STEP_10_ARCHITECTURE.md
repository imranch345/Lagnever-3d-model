# Step 10 architecture

Step 10 changes what the frame head is asked to predict, and deliberately changes nothing
about how the result is judged. The head predicts each entity's frame in its **parent's**
coordinates; the model composes those into scene coordinates before anything downstream
sees them; the loss, the metrics, the decoder and the evaluation are Step 9's, unchanged.

That asymmetry is the whole design. Step 10's claim is that a target matching the
hierarchical structure the representation already has will teach the head something the
Step 9 target could not. A claim of that shape is only testable if the ruler does not move
with the target, and a local target has smaller residuals by construction, so scoring
locally would flatter the change for arithmetic reasons.

**This document describes Change 1's design first**, because that is what most of the
machinery was built for. Change 2 asks a different question — whether position, rotation and
scale are learned at all once rotation is genuinely present — and has its own corpus, its own
floors and its own objective; both are covered below. Neither claim survived: Change 1's
parent-relative target placed worse than the global one in all five arms, and Change 2's
rotation did not beat its floor in any. The design is documented here because it is what the
code does, not because it worked.

## What Step 10 changes

| | Step 9 | Step 10 Change 1 |
| --- | --- | --- |
| What the head outputs | a frame in scene coordinates | a frame in its parent's coordinates |
| What the loss sees | that frame | the **composed** global frame |
| What the metric sees | that frame | the **composed** global frame |
| The floor it is read against | 0.1605 identity-only lookup | the same 0.1605 |
| Entities affected | all 20 | the 10 that have a parent |

## Where the change goes, and why there

Composition happens in one place, `LagnavPrototype.predict_frames`. Every caller — the
training loss, `evaluate_step8`, the decoder's `points_to_local`, the editing path — keeps
receiving a global frame and needs no change.

Putting it in the loss instead would have left evaluation scoring local predictions against
global truth. That would not have raised; it would have produced plausible numbers that
were not comparable with anything. Steps 7 to 9 lost time to four defects of exactly that
shape, so the safe location was chosen over the convenient one.

## The hierarchy is the generator's, not AWR's

Change 1's brief gives `left_ventricle -> mitral_valve` as its example. **That tree is not in
AWR.** All 20 whole-organ entities are depth-2 leaves under geometry-free category nodes,
and no containment relation holds between any two of them. Composing along AWR's hierarchy
would compose against parents that have no frame.

The tree that does exist is the generator's construction order: `_build_annuli` derives each
atrioventricular annulus from a chamber pair, `_build_tubes` starts each great artery at its
own valve and branches the pulmonary arteries off the trunk.
`datasets/whole_organ/hierarchy.py` writes that down, and keeps AWR's version beside it as
the control arm. See ADR-STEP10-001.

| | entities with a parent | max depth | depth 1 / 2 / 3 |
| --- | --- | --- | --- |
| `spatial` | 10 of 20 | 3 | 7 / 2 / 1 |
| `taxonomic` (control) | 0 of 20 | 0 | — |

Half the entity set is untouched by Change 1. That bounds what the change can deliver and is
why error is reported by depth as well as in aggregate.

## Composition, and the defect it would otherwise have

Twelve-number frames are **not closed under composition**. `R_p diag(s_p) @ R_c diag(s_c)`
has shear whenever the parent's scale is anisotropic and its rotation is not the identity,
and no `R diag(s)` represents it.

This is invisible on the Step 8/9 corpus because every rotation in it is exactly the
identity. Change 2 makes rotations real, so the defect would have surfaced precisely when
the rotation target became meaningful — and would have read as Change 2 failing.

The fix is that a parent contributes rotation and an **isotropic** scale, while the child
keeps its full anisotropy. Composition is then exact at any rotation. Measured cost of not
doing this, on real frames:

| parent rotation | position error | rotation error | share of the 0.1605 floor |
| --- | --- | --- | --- |
| 0° | 3e-17 | 0.000° | 0.0% |
| 30° | 7.3e-03 | 0.97° | 4.6% |
| 90° | 1.4e-02 | 5.24° | 9.0% |

See ADR-STEP10-002.

## Levels of detail hide parents

A coarse level exposes the aorta while hiding the aortic valve: 8.1% of non-root child
instances have an absent parent. `resolve_parents` walks up to the nearest **present**
ancestor, which removes all of them. It uses only the presence mask — an input the model
already sees — so it leaks nothing about placement.

## The floor moves, and the bar does not

Step 9 made the placement-blind floor a first-class baseline, so Step 10 recomputes it.
Three predictors, all fitted on `train`, all scored on **global** frames with the Step 9
metric:

| split | `global` | `parent_relative` | `parent_relative_oracle` |
| --- | --- | --- | --- |
| test_seen | 0.1605 | 0.1636 (−1.9%) | 0.1213 (+24.4%) |
| test_arrangement | 0.1655 | 0.1680 (−1.5%) | 0.1237 (+25.3%) |
| test_transform | 0.1703 | 0.1725 (−1.3%) | 0.1251 (+26.5%) |
| test_combination | 0.1750 | 0.1774 (−1.4%) | 0.1515 (+13.4%) |

The `global` column reproduces Step 9 exactly, which is what makes the other two readable.

The **bar every arm is read against stays 0.1605**. A floor is a property of the task and
the ruler, not of the arm, and every arm is scored the same way. The parent-relative column
is context — what the new formulation costs a predictor that places parents blindly — and
the oracle column is a diagnostic, not a baseline, because it leaks the true parent.

Read together they made a prediction before anything was trained: the parent-relative target
is worth about 24% of position error **conditional on placing parents well**, and costs
about 2% to a predictor that does not. Step 9 found only A3 reliably clears the floor, so
the expected outcome is that Change 1 helps A3 and does little for the cheaper arms.

**That prediction was not supported.** Change 1 made every arm worse, and A3 worst of all
(+7.8%), losing its floor-clearing margin. The premise is what failed: clearing the floor is
not the same as placing parents well enough to compose against. See
`docs/STEP_10_CHANGE1_REPORT.md`.

## Rotation: the metric and the loss are different functions

`rotation_angle` is the geodesic angle and is what gets reported. `rotation_chordal` is a
monotone, smooth, bounded equivalent and is what gets descended. `arccos` has an unbounded
derivative at zero, so a nearly-correct model would receive an arbitrarily large gradient
from the geodesic form — measured at 1e-3 rad, more than 100× the chordal form's. The same
singularity is why `rotation_angle` returns about 3e-8 rather than 0 for a rotation against
itself in float64.

Change 2 made this live. `Step10Config.rotation_objective` selects the term: `step8_6d_l1`
is Step 8's, the absolute difference of the six stored numbers, and remains the default so
Change 1 stays reproducible; `chordal` is Change 2's. The chordal term is taken **after** the
model's Gram-Schmidt, so it is a distance on rotations and cannot be reduced by inflating the
stored basis vectors — which the six-number term can. Its weight, 0.33, was calibrated on
training data with the term unoptimised. ADR-STEP10-003 records the decision and its cost:
the weight leaves rotation at 7% of the frame loss, because Step 8's scale term takes 64%.

## Change 2: the corpus the rotation target needed

Change 1's corpus stored an identity rotation for every entity of every scene, so a parent
could pass its children a translation and a scale but no orientation, and no rotation result
was obtainable. The generator had been computing those rotations all along —
`WholeOrganScene.frames()` discarded them and wrote the identity.

`step10-rotated-1950-40` is the Change 1 corpus with its frames re-measured: each entity's
rotation declared from the generator's construction, and its extents measured on that
rotation's own axes. Every other stored field is carried over verbatim, so the two corpora are
the same 1,950 organs measured two ways. `derive_rotated_corpus` performs that re-measurement
and fails if a re-measured centroid does not reproduce the stored one.

The rotation is **declared** from construction rather than taken from principal axes: a
chamber is nearly an ellipsoid of revolution, so its second and third principal axes are
decided by sampling noise and the target would be a coin flip. An annulus or a vessel takes
its own construction axis as the frame's third row; everything else takes the organ-to-scene
rotation built from the arrangement's yaw, pitch and roll. Rotation is therefore
hierarchy-conditioned, and it follows the arrangement — which is what the held-out splits hold
out, so the compositional hold-out carries over to rotation rather than being assumed to.

The corpus has its own placement-blind floor, per split and per component. **0.1605 is not
its bar.** Position, rotation and scale each have one, and the four splits' rotation floors —
22.65°, 22.54°, 35.95°, 26.10° — are not interchangeable.

## The integrity gate runs before training

Parent-relative placement fails quietly: a wrong parent, a child composed before its parent,
or a rotation read transposed all still produce frames of the right shape and a falling
loss. So `generation/neural/nn/placement_integrity.py` checks the metadata, and raises,
before a run may start. `Step10Config.resolved` checks the declared tree before the model is
built; `Step10Trainer` runs the full gate on a training batch and again, at evaluation, on
every split. Five checks:

* **source table against the generator** — every declared parent is an entity the
  generator's own code builds that child from (an annulus from one of the two cavities it
  sits between; a vessel from its anchor or from the annulus it starts on, recognised by
  starting on that annulus's axis). Checking the table against a copy of itself would pass
  a shuffled table; checking it against construction does not;
* **slot table** — the model's table addresses the same entities as the declared tree,
  catching the 20-entity versus 64-slot ordering hazard;
* **order** — a permutation with every parent before its children;
* **stored frames** — raw rotation vectors orthonormal and right-handed, read before the
  model's Gram-Schmidt can silently repair them;
* **round trip** — parent-relative targets compose back to the frames, both as stored and
  after giving every slot a distinct real rotation. The second pass is required: every
  rotation in this corpus is the identity, which is its own transpose, so a transposed
  rotation is invisible on the stored frames.

The gate saves and restores the Python, NumPy and torch generators, so it cannot move a
result it guards. It cannot detect a rotation stored transposed *in the data* — that is still
a valid rotation — and on this corpus the question does not arise; a rotated corpus will need
a comparison against the generator's own record.

## What did not change

The AWR source of truth, the entity axis, per-entity geometry tokens, the identity
subspace, geometry correspondence, the lightweight relational mechanism, the whole-organ
dataset, measured relationships, the persistent scene representation, the Step 9 metric, and
the Step 9 translation and scale loss weights.

Change 1 additionally left the corpus untouched. Change 2 derives a new one, and changes two
things in it and nothing else: entity frames carry measured rotations, and extents are
measured on each frame's own axes because an oriented frame with world-axis extents would
describe a box the entity does not occupy. The generator parameters, centroids, relationship
graphs, presence, levels of detail and split assignment are the parent corpus's, verbatim.
