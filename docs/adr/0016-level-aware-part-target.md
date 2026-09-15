# ADR 0016: The part-correspondence target names only entities the level exposes

**Status:** ACCEPTED
**Date:** Step 7
**Relates to:** `datasets/whole_organ/sampling.py`, `generation/neural/nn/whole_organ.py`

## Context

The whole-organ generator segments the organ at full detail, then a scene's level of
detail decides which entities are exposed. The part-correspondence target was taken from
the full-detail segmentation, so at coarse levels it named entities the scene does not
expose.

The scene composition masks absent entities to `-1e9`. Cross-entropy against a class
with a `-1e9` logit costs about `1e9` for that point.

| Level | Points naming a hidden entity | Part loss |
| --- | --- | --- |
| 1 | 397 of 1024 (39%) | 3.88e8 |
| 3 | 0 | 3.02 |

Across every structured arm the median logged part-correspondence loss was `3.65e8`,
above `1e6` in 13 of 19 logged steps. Gradient clipping kept the runs from diverging,
which is why the remaining numbers looked plausible.

## Decision

The part target is restricted to the entities the active level exposes. A point whose
owner is hidden at this level reads as background, which is what that level's own
occupancy target already says about it.

This makes the two supervisions consistent: `scene_occupancy` at level L is true exactly
where the part target is not background.

## Consequences

* Part loss is 2.4 to 3.1 at every level.
* The second Step 7 suite is superseded and its arm comparison withdrawn; every arm had
  spent most of its gradient budget on a term none could reduce. It is kept under
  `experiments/runs/step7-whole-organ-v2-broken-part-loss/`.
* Two regression tests guard the invariant: no target may name an entity the level
  hides, and the part target must be background exactly where level occupancy is empty.

## Alternatives considered

**Relabel hidden entities to their nearest visible ancestor.** Attractive, because at
level 1 a valve's points arguably belong to the chamber that contains it. Rejected for
now because it would make the part target disagree with `scene_occupancy`, which treats
those points as empty at that level. Changing both together is a larger design change
than the defect warranted, and is recorded as future work rather than smuggled in as a
bug fix.

**Raise the mask from `-1e9` to a smaller value.** Rejected: it would hide the symptom
and leave the target naming classes the composition cannot produce.
