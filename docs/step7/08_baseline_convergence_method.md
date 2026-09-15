# Experiment 11 — Baseline convergence. Method

## Why this experiment exists

An ablation suite that compares structured arms against a baseline is only meaningful if
the baseline was given a fair chance. A baseline that never converged is not evidence
that structure helps; it is evidence that something was wrong with the run.

This experiment reports, with a stopping criterion fixed in advance, whether each arm's
validation had plateaued by the end of training.

## Stopping criterion, fixed before the runs

Training runs for a fixed 900 steps for every arm. The criterion is used for
**reporting**, not for stopping early, so that every arm gets an identical budget:

> An arm is reported as **converged** if its validation `scene_iou` at the final
> checkpoint is within 0.02 of its best validation value, and its best value did not
> occur at the first validation point. An arm is reported as **not converged** if
> validation was still improving by more than 0.02 over the last third of training, and
> as **failed** if its final validation `scene_iou` is below 0.1.

A **failed** baseline invalidates any comparison drawn against it, and the comparison is
reported as unavailable rather than favourable.

## What the earlier suites showed

In the second suite the appearance baseline `A0` reached a final validation `scene_iou`
of 0.000, 0.000 and 0.133 across three seeds: **failed** under the criterion above.

Investigating that failure is what uncovered the defect that invalidated the whole suite.
`A0`'s training loss sat at roughly 3.7e8, and the term responsible was
`semantic_part_correspondence`, which was scoring points against entity classes the
level-of-detail composition had masked to `-1e9`. Every structured arm was affected
identically, with a median logged value of 3.65e8. Gradient clipping at 1.0 kept the runs
from diverging, which is exactly why the structured arms' other numbers looked plausible
and the defect went unnoticed until the baseline was examined.

See [ADR 0016](../adr/0016-level-aware-part-target.md). The part loss is now 2.4 to 3.1
at every level, and the suite was rerun from scratch.

The lesson is worth recording: **the failing arm was the diagnostic.** Had the baseline
been dismissed as "appearance models do badly here", the defect would have shipped inside
a headline result about graph necessity.
