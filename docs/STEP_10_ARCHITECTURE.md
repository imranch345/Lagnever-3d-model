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

## Rotation: the metric and the loss are different functions

`rotation_angle` is the geodesic angle and is what gets reported. `rotation_chordal` is a
monotone, smooth, bounded equivalent and is what should be descended. `arccos` has an
unbounded derivative at zero, so a nearly-correct model would receive an arbitrarily large
gradient from the geodesic form. The same singularity is why `rotation_angle` returns about
3e-8 rather than 0 for a rotation against itself in float64.

## What did not change

The AWR source of truth, the entity axis, per-entity geometry tokens, the identity
subspace, geometry correspondence, the lightweight relational mechanism, the whole-organ
dataset, measured relationships, the persistent scene representation, the Step 9 metric,
the Step 9 loss weights, and the corpus.
