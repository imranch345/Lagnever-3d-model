# ADR-STEP10-004: A parent-relative local frame is not a portable local transform

**Status:** ACCEPTED
**Date:** Step 10, Change 2
**Relates to:** `experiments/step10/change2_diagnostics.py`,
`experiments/step10/inheritance_diagnostic.py`, `generation/neural/nn/placement.py`

This ADR records **how a result must be read**, in the manner of ADR-0018 and ADR-0019. It
does not change any code.

## Context

The case for a parent-relative target rests on an intuition: if a child's frame is predicted
*relative to its parent*, then that prediction is a property of the child alone — its offset,
its orientation, its size within its parent — and the parent's own placement is a separate
problem. Under that reading a local frame is portable: move the parent, and the child follows
correctly.

That intuition is what motivated Change 1, and it is what makes `rigid` and `isotropic`
parent conventions worth distinguishing at all.

## The measurement

Both Step 10 changes ran the same diagnostic: take a trained model, and recompose each child's
predicted local frame onto its parent's **true** frame instead of its predicted one. If local
frames were portable, a correct parent could only help.

**Change 1, position** (identity-rotation corpus, `test_seen`, seed means). At depth 1 the
true parent helps substantially — the child's own local prediction is 25–32% better than the
global target's. At depth 2 it **hurts**: 0.145–0.151 with the true parent against 0.109–0.116
as trained.

**Change 2, rotation** (rotated corpus, `test_seen`, A3). The true parent hurts at *every*
depth, in both parent conventions:

| cell | depth | as trained | with true parent |
| --- | --- | ---: | ---: |
| `T4_rigid` | 1 | 28.54° | 36.72° |
| `T4_rigid` | 2 | 30.12° | 52.78° |
| `T1_spatial` | 1 | 27.55° | 37.72° |
| `T1_spatial` | 2 | 30.41° | 50.54° |

Corroborated independently: the parent-child error correlation is **lower** in the cells that
compose (+0.26 to +0.36) than in `T0_global`, which composes nothing (+0.44, +0.49). Simple
propagation predicts the opposite.

## What it means

Under end-to-end training on composed frames, the loss only ever sees the composition. A child
whose parent is predicted with a systematic error is therefore rewarded for predicting a local
frame that **cancels that error**. The resulting local frame is partly a description of the
child and partly an encoding of its parent's mistakes, and the two cannot be separated after
the fact.

So a "local frame" produced by this training regime is not the parent-independent quantity the
intuition assumes. It is correct only in composition with the parent it was trained against.

## How results must be read

1. **A true-parent oracle is not an upper bound.** It can be, and here usually is, *worse*
   than the model's own composition. Reporting it as "what the model could achieve with a
   perfect parent" is wrong.
2. **A parent-relative local frame must not be reused, transplanted or edited in isolation.**
   Any future local-editing design that moves a parent and expects its children's stored local
   frames to carry over unchanged is relying on a property this measurement contradicts.
3. **Parent-child error correlation is not evidence of propagation.** Composition lowered it
   here. A cell that composes nothing is the control that makes the number readable.
4. **The effect is a property of the training regime, not of the transform algebra.** The
   composition itself is exact — verified to 0.0 against each model's own output — and the
   convention is closed. Nothing here is a defect in the representation.

## What would change this

A stop-gradient through the parent during training would remove the channel through which the
compensation is learned, and is the obvious way to test whether a genuinely portable local
frame is achievable. It is named as an open question in the Change 2 report and has not been
run.
