# ADR 0019: Spatial relation accuracy is reported against a relation-blind floor

**Status:** ACCEPTED
**Date:** Step 7
**Relates to:** `experiments/step7/relation_baseline.py`

## Context

`spatial_relation_accuracy` counts the share of a scene's measured spatial relations that
a prediction reproduces. Read on its own it invites a mistake, because many relations
survive every arrangement: an apex stays inferior to a base however the organ is
mirrored. A model that ignores the relationship graph entirely does not score zero. It
scores whatever the invariant relations are worth.

## Decision

Two reference points are computed from the corpus alone, with no model and no arm:

| Reference | Predictor | Meaning |
| --- | --- | --- |
| oracle | the scene's own measured centroids | must be 1.0, or the scoring code is what is broken |
| blind | the `NORMAL` arrangement of the scene's own family, whatever the scene actually is | exactly what a relation-blind model can achieve |

On the corrected corpus the oracle scores 1.0000 and the blind predictor 0.8878 on the
test split. **The discriminative band of this metric is 0.8878 to 1.0000, a headroom of
0.1122.**

Claims about relational competence are stated as the gap above the blind floor, never as
the raw accuracy.

## Consequences

* A reported accuracy of 0.98 is roughly four fifths of the available headroom, not
  "almost perfect".
* The oracle doubles as a self-check on the scoring code: if it is not 1.0, no result
  from this metric should be believed.
* The floor is corpus-dependent and is recomputed and reported with each corpus.

## Alternatives considered

**Rescale the metric so blind maps to zero.** Rejected: it redefines a pre-registered
metric after seeing results. Reporting the floor alongside the unchanged metric conveys
the same information without that cost.

**Restrict the metric to arrangement-sensitive relations only.** Rejected for the same
reason, and because which relations are sensitive depends on the variant, so the metric
would no longer mean the same thing in every scene.
