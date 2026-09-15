# ADR 0018: Evaluate with placement supplied and with placement inferred

**Status:** ACCEPTED
**Date:** Step 7
**Relates to:** `experiments/step7/metrics.py`, `experiments/step7/run_placement.py`

## Context

The occupancy decoder works in each entity's canonical frame, so it needs that entity's
translation, scale and rotation. `LagnavPrototype.forward` takes those from the batch
unless `use_predicted_frames` is set, and the batch's frames come from the scene.

The whole-organ evaluation never set the flag. It was therefore **handing the model the
true position, scale and rotation of every entity** and asking only what shape to put
there. The frame head, which exists to predict exactly those quantities, was never
exercised at evaluation time.

This resolved three results that did not fit together: a flat profile across the four
arrangements, near-zero counterfactual sensitivity, and a spatial relation accuracy of
0.98 for arms that cannot see spatial edges at all. All three follow if placement is
already given.

Measured on a trained `A3` checkpoint over 32 held-out scenes:

| Condition | Entity ownership IoU | Spatial relation accuracy |
| --- | --- | --- |
| placement given | 0.480 | 1.000 |
| placement inferred | 0.247 | 0.962 |

## Decision

Both conditions are reported for every arm, from the same weights:

* **given** — entity frames supplied. Measures shape and ownership quality when
  placement is known. The original condition, unchanged.
* **inferred** — the model predicts placement from structure, relations and text.

**Decisions about whether the relationship graph is necessary rest on the inferred
condition.** In the given condition the answer a graph would supply has already been
provided through another input, so the condition cannot discriminate.

No metric definition changes. The same metrics are computed under a second condition,
and both are reported.

## Consequences

* Every reported table names its condition. A number without a condition is incomplete.
* The re-evaluation runs from saved checkpoints, so it costs no retraining and the
  weights judged are identical in both conditions.
* Spatial relation accuracy must additionally be read against the relation-blind floor
  (ADR 0019), not against zero.

## Alternatives considered

**Switch to inferred only.** Rejected: the given condition still measures something
real, namely shape quality independent of placement, and discarding a condition after
seeing its results is the practice Step 7 is meant to avoid.

**Retrain with predicted frames fed back during training.** Rejected for Step 7: it is a
training change, not an evaluation fix, and it would confound the defect correction with
a curriculum change. Recorded as future work.
