# ADR-STEP8-002: The model predicts entity placement; evaluation never supplies it

**Status:** ACCEPTED
**Date:** Step 8
**Relates to:** `generation/neural/nn/geometry.py`, `training/step8.py`,
`tests/test_step8_placement.py`

## Context

The occupancy decoder works in each entity's canonical frame, so it needs that entity's
translation, scale and rotation. Through Steps 5, 6 and 7 those came from the batch, which
is to say from the scene. The frame head existed but was a zero-initialised linear probe
of 3,084 parameters, and no evaluation ever asked it for anything.

Step 7 found what that concealed. Scored on the same trained weights:

| Condition | Entity ownership IoU | Spatial relation accuracy |
| --- | --- | --- |
| placement supplied | 0.6339 | 0.9850 |
| placement inferred | 0.3020 | 0.8472 |

Ownership halved. Worse, the supplied condition made the relational question unanswerable:
a model that cannot see spatial edges at all reproduced 0.67 of the arrangement variation
and scored 0.9866 on relation accuracy, because the frames were supplying the arrangement
that the relationship graph was supposed to.

## Problem

A structured anatomical representation whose placement must be handed to it is not doing
the thing it claims. Either the model can infer where structures go from identity and
relations, or the representation is decorative at the point where it matters most.

## Alternatives considered

**Train with predicted frames from step zero.** Rejected. Early frame error corrupts the
occupancy signal before either component works, and neither learns. The Step 7 code
comment claiming this was why frames were teacher-forced was correct about the mechanism
and wrong to make it permanent.

**Keep supplying frames and report the inferred condition alongside.** This is what Step 7
did. Rejected: a model trained under supplied frames is not optimised for the condition it
is judged in, so the inferred number measures a mismatch rather than a capability.

**Predict frames from the pre-relational latent.** Rejected. It would make placement
independent of relations by construction, which is the opposite of the hypothesis under
test.

## Decision

Three changes.

1. **A frame head worth training.** `FramePredictor`, a residual MLP of 135,180
   parameters, reading the latent **after** the relational encoder so that neighbours can
   influence placement. Zero-initialised output with a bias at the canonical frame, so an
   untrained head produces a scene at the origin rather than a degenerate one.
2. **A curriculum.** Teacher forcing is 1.0 for the first 15% of the budget, falls
   linearly to 0.0 by 60%, and stays there. The ratio is logged at every recorded step and
   written into the run manifest.
3. **An evaluation that cannot be fooled.** `use_predicted_frames` is checked **before**
   the teacher-forcing ratio, so a ratio left set by mistake cannot turn a headline result
   into an oracle result. A test asserts that the two paths give identical output.

Supervising placement is not the same as supplying it: the model is told the right answer
during training and must produce it unaided at inference, which is the ordinary
arrangement for any predictive head.

## Evidence

Step 7's two-condition table above is why. The Step 8 evidence that judges the decision is
whether any arm's spatial relation accuracy under inferred placement rises above the
relation-blind floor, recomputed for this corpus. Step 7 found every arm **below** its
floor, meaning a fixed arrangement-blind guess beat every trained model.

## Consequences

* Every reported number names its placement condition. The supplied condition is kept and
  reported as an oracle upper bound, never as a result.
* Headline numbers are lower than Step 7's, because the task is harder and the corpus is
  harder. That is a change in what is being measured, not a regression.
* The frame head is now on the critical path, so frame error is reported in its own right:
  position, scale, rotation, a documented composite, and a breakdown by entity group.

## Reversal conditions

Reverse this decision if:

* predicted placement is shown to be unlearnable at any budget on this corpus, in which
  case the honest response is to state that the representation requires supplied placement
  rather than to quietly resume supplying it;
* a downstream use is found where placement genuinely is given, in which case the supplied
  condition becomes a result for that use and must be labelled as such.

Reverting to supplied placement for the **headline** condition is not an available
outcome. It is the defect Step 7 spent most of its effort discovering.
