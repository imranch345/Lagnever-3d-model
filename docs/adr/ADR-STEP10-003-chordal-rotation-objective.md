# ADR-STEP10-003: Rotation is descended as a chordal distance and reported as an angle

**Status:** ACCEPTED, with corrections recorded under Amendments after Steps 10 and 14
**Date:** Step 10, Change 2
**Relates to:** `training/step10.py`, `generation/neural/nn/transforms.py`,
`experiments/step10/calibrate_rotation_weight.py`, `experiments/step10/rotation_metrics.py`,
`tests/test_step10_rotation_loss.py`, `training/step8.py`, `generation/neural/nn/losses.py`,
`experiments/step10/run_change2.py`

## Context

Step 8 gave the frame loss three parts, weighted 1.0 translation, 0.5 scale, 0.25 rotation,
and the rotation part was the absolute difference of the six stored numbers. On the Step 8
and Step 9 corpus every rotation was exactly the identity, so that term was identically zero:
a quarter of the declared emphasis was spent on a constant, and no rotation result of any kind
was obtainable.

Change 2's corpus carries real rotations — mean 61.8°, none below 1° — which makes the term
meaningful for the first time and forces two separate questions that had never needed
answering: what should be *descended*, and what should be *reported*.

## Problem

Choose a rotation objective and its coefficient, before any confirmatory run, without tuning
either against a test split.

## Alternatives considered

**Keep Step 8's six-number absolute difference.** Rejected. It is not a distance on
rotations: the stored numbers are a 6D representation that the model Gram-Schmidts into a
matrix, so two different stored pairs can denote the same rotation. Worse, the term is
reducible without changing the rotation at all — scaling the stored basis vectors toward the
target's lowers it while the rotation it denotes is unmoved. `tests/test_step10_rotation_loss`
pins that the chordal term is invariant to exactly that manipulation and the six-number term
is not.

**Descend the geodesic angle.** Rejected, and this is the sharpest of the alternatives
because the geodesic angle is the right thing to report. `arccos` has an unbounded derivative
at 1, so a nearly-correct model receives an arbitrarily large gradient. Measured at 1e-3 rad,
the geodesic form's gradient is more than 100× the chordal form's; the test that pins this
also pins that the chordal gradient stays finite there.

**Regress Euler angles.** Rejected without measurement. The pipeline already emits and
validates the continuous 6D representation, the composition path is built on it, and Euler
angles reintroduce a discontinuous parameterisation the architecture had already avoided.

**Choose the weight by sweeping it against a test split.** Rejected as a protocol violation.

**Choose the weight so rotation is a fixed fraction of the whole frame loss.** Rejected as a
new decision made at the moment it would have been convenient. Step 8 already declared the
emphasis rotation should carry — a quarter of translation — and had simply never been able to
deliver it.

## Decision

**Training loss:** the chordal distance, `||R_pred − R_true||²_F / 8`, taken after the model's
own Gram-Schmidt so it measures the rotation and nothing else. Bounded in [0, 1], monotone in
the geodesic angle, smooth at zero.

**Reporting metric:** the geodesic angle in degrees, clamped. The two are deliberately
different functions, and that is documented wherever either appears.

**Weight:** `lambda_rotation = 0.25 × mean(translation term) / mean(rotation term)`, the rule
fixed before the measurement. Measured on `train` only, over 50 steps for each of the five
arms, **with the rotation term's weight set to zero** so the term was measured rather than
optimised — a calibration that let the term be trained by the weight it was choosing would be
circular. Result: 0.3307, frozen at **0.33**. The runner refuses to start a run whose weight
is not this one, and every run manifest records it.

**Everything else in the Step 8 frame loss is preserved**: translation 1.0, scale 0.5.

## Consequences

The rule was applied as written, and it puts rotation at **7% of the frame loss** — because
Step 8's scale term, an L1 sum over three log axes, takes 64%. That was harmless while
rotation was structurally zero and is not harmless now.

Change 2's result is that no arm beat the rotation floor on any split, best margin −0.07°.
**This ADR's decision is why that result cannot be read as an architectural verdict**: the
experiment cannot separate "the architecture does not learn rotation" from "rotation was 7% of
the objective". The alternative — retuning after seeing that — would have destroyed the
pre-registration that makes the rest of the result trustworthy.

Two consequences follow, both named in the Change 2 report and neither taken here:

* Step 8's frame weights are due for re-derivation now that the rotation term is
  non-degenerate;
* a rotation-weight study must run on `train` and `validation` only, before rotation
  learning is judged.

## Amendments

Added after Step 14. The decision is unchanged. Three statements above no longer describe
the code that ran, so they are corrected here rather than rewritten, which keeps the
original reasoning readable.

### 1. The weight is 3.0, not 0.33

Change 2 ran at the calibrated 0.33, and the study this ADR called for then ran on `train`
and `validation` only. It screened 0.33, 1.0, 3.0 and 10.0 and froze **3.0**
(`SCREENED_ROTATION_WEIGHT` in `experiments/step10/run_change2.py`). Every run from the
Step 10 confirmatory matrix through Step 20 uses 3.0. See
[STEP_10_ROTATION_WEIGHT_STUDY.md](../STEP_10_ROTATION_WEIGHT_STUDY.md).

### 2. There are two frame terms, not one

"Everything else in the Step 8 frame loss is preserved" reads as though the chordal term
became the only rotation penalty. It did not. The chordal term lives in the trainer's frame
loss, `Step10Trainer._frame_loss`, which `Step8Trainer.train_step` adds outside the loss
module. The loss module's own `entity_frame` term (`_frame` in
`generation/neural/nn/losses.py`) stays live at weight 0.5:

```
total = loss-module terms, including
          0.5 * entity_frame     # (L1 on translation and log-scale
                                 #  + 0.5 * L1 on the six rotation numbers) / 6
      + 2.0 * (translation + 0.5 * scale + 3.0 * chordal)    # frame_weight * _frame_loss
      + 0.5 * nested_lod
```

So the six-number absolute difference this ADR rejected is still descended, at a small
weight. Step 14 measured `entity_frame` at 3.1% of the total objective on the frozen A3
checkpoints. Both terms are active in every chordal run, Step 10 Change 2 through Step 20, so
no comparison between those runs is affected. Removing the older term is a candidate for the
next study that changes the loss; it has not been removed.

The docstring on `Step10Trainer._frame_loss`, which says the chordal objective "replaces only
the rotation term", is accurate for that function and is left as it is.

### 3. The effective rotation coefficient is 6.0

λ = 3.0 is the coefficient inside `_frame_loss`, which `train_step` then multiplies by
`frame_weight = 2.0`. Against the other terms of the total objective, rotation therefore
carries 6.0. Every report's "λ = 3.0" is correct as the inner coefficient. The calibration
rule above is unaffected: it is a ratio of terms inside `_frame_loss`, and the outer weight
scales both. Step 14 measured rotation at 16.8% of the total objective at this setting.

Source: [Step 14 report, §3.1](../STEP_14_OBJECTIVE_DIAGNOSTIC_REPORT.md), checked against
`training/step8.py`, `training/step10.py` and `generation/neural/nn/losses.py`.
