# ADR-STEP10-003: Rotation is descended as a chordal distance and reported as an angle

**Status:** ACCEPTED
**Date:** Step 10, Change 2
**Relates to:** `training/step10.py`, `generation/neural/nn/transforms.py`,
`experiments/step10/calibrate_rotation_weight.py`, `experiments/step10/rotation_metrics.py`,
`tests/test_step10_rotation_loss.py`

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
