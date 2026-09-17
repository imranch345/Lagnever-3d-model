# ADR-STEP10-002: A parent contributes rotation and isotropic scale, not full scale

**Status:** ACCEPTED
**Date:** Step 10
**Relates to:** `generation/neural/nn/transforms.py`, `tests/test_step10_transforms.py`,
`tests/test_step10_integrity.py`

## Context

Step 10 defines the placement target by composition: the head predicts a frame in its
parent's coordinates and the global frame is recovered by composing down the hierarchy. The
target is therefore only as correct as the composition.

The frame contract is 12 numbers — 3 translation, 3 per-axis log scale, 6D rotation — which
denotes the similarity transform `p_parent = R @ (exp(s) * p_child) + t`. Composing two of
those multiplies their linear parts:

```
R_p diag(s_p) @ R_c diag(s_c)
```

When `s_p` is anisotropic and `R_p` is not the identity, that product has shear. **No
`R diag(s)` represents it.** The 12-number frame is not closed under composition, and
converting the product back to a frame silently discards the difference.

This is invisible on the Step 8 and Step 9 corpus for one reason: every rotation in it is
exactly the identity, verified across all five splits (`max |R - I| = 0.0`). Diagonal
matrices commute, so the defect never appears. Step 10 Change 2 replaces that constant with
a real rotation — so the defect would have appeared precisely when the rotation target
became meaningful, and would have read as Change 2 failing.

## Problem

Make composition exact for a target that is *defined* by composition, without abandoning the
12-number contract that the decoder, the metrics and every existing checkpoint use.

## Alternatives considered

**Accept the projection error and report it.** Rejected. Measured on `test_seen` with the
corpus's own anisotropic scales and injected rotations, the round trip
`compose(p, relative(p, c))` against `c` costs:

| parent rotation | position error | rotation error | as % of the 0.1605 floor |
| --- | --- | --- | --- |
| 0° | 3e-17 | 0.000° | 0.0% |
| 30° | 7.3e-03 | 0.97° | 4.6% |
| 90° | 1.4e-02 | 5.24° | 9.0% |

Up to 9% of the placement-blind floor and several degrees, sitting under every arm equally
and indistinguishable from model error. A target-definition floor no model could beat.

**Store the local transform as a general 3×3 linear map plus translation.** Composition
becomes exact and the parameter count is unchanged (9 + 3 = 12). Rejected because it
changes what a frame *means* everywhere — the decoder, `points_to_local`, the metrics and
every checkpoint — while Step 10's whole argument depends on the ruler not moving.

**Restrict the corpus to isotropic scales.** Rejected. Anatomical entities are elongated;
removing that would make the corpus easier in a way unrelated to the question.

## Decision

The **parent** contributes rotation and an isotropic scale; the child keeps its full
anisotropic scale. Then

```
R_p s_p @ R_c diag(s_c) = (R_p R_c) (s_p diag(s_c))
```

is again a rotation times a diagonal, and composition is exact at any rotation.
`PARENT_CONVENTIONS` names three settings and `as_parent` applies them:

* `isotropic` — the default. Closed, and a parent's size still reaches its children.
* `rigid` — also closed, but a parent's scale does not propagate at all.
* `full` — the naive choice, retained **only** so its cost can be measured rather than
  argued.

`compose` is consequently not a group operation: the identity is neutral on the left and not
on the right. That asymmetry is what keeps it closed, it is consistent between `relative`
and `compose_hierarchy`, and it is pinned by a test that would otherwise look like a bug
worth "fixing".

`matrix_to_frame` recovers scale as the per-column norms and rotation as the
column-normalised matrix, rather than by QR. For a similarity input the two agree exactly,
and the closed conventions guarantee a similarity input; column norms are additionally
differentiable everywhere, which QR's gradient is not, and this function sits in the loss
path.

## Evidence

* `test_round_trip_is_exact` — exact to `1e-10` in float64 under both closed conventions,
  at unconstrained rotations and anisotropic scales.
* `test_full_convention_is_not_closed` — the same round trip fails at `1e-6` under `full`.
* `test_identity_rotation_hides_the_defect` — with identity rotations, `full` round-trips
  exactly, documenting why Steps 8 and 9 never saw this.
* `test_parent_relative_target_reconstructs_the_global_frames` — on real corpus frames
  through the real hierarchy, at 0°, 30° and 90°, position and linear error below `1e-9`.
* `test_the_unclosed_convention_loses_accuracy_once_rotations_are_real` — the corruption
  test: it fails if `full` ever stops losing anything, so the argument cannot quietly expire.

## Consequences

* A parent's anisotropy does not reach its children. A ventricle that is elongated along one
  axis moves its valve by its *mean* scale, not its per-axis extent. This is a real modelling
  choice, not a neutral one, and `rigid` versus `isotropic` is the arm that tests how much
  it matters.
* The rotation metric and the rotation loss are different functions. `rotation_angle` is the
  geodesic angle and is what is reported; `rotation_chordal` is a monotone, smooth,
  bounded equivalent and is what should be descended, because `arccos` has an unbounded
  derivative at zero and a nearly-correct model would receive an arbitrarily large gradient.
* `rotation_angle` returns about `3e-8` rather than `0` for a rotation against itself in
  float64. That is `arccos` amplifying the square root of the epsilon, it is harmless in a
  metric, and the tolerance in the tests says so rather than hiding it.

## Reversal conditions

Reverse this decision if any of these is shown:

* the corpus stops producing anisotropic scales, in which case all three conventions
  coincide and the reduction is dead weight — the integrity test fails when this happens;
* `rigid` matches `isotropic` within seed spread on every metric, in which case prefer
  `rigid` as the simpler rule;
* the frame contract is deliberately widened to a general affine for other reasons, in
  which case the closure problem disappears and `as_parent` should be deleted rather than
  kept.
