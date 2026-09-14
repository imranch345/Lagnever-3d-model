# ADR 0014: Relationship accuracy counts unplaceable entities as failures

**Status:** ACCEPTED, and disclosed as a post-hoc metric choice
**Date:** Step 6
**Relates to:** `generation/neural/nn/metrics.py`, `experiments/heart/run_experiment.py`

## Context

The Step 5 pre-registration named "held-out relationship accuracy" without fixing how
to treat a relation whose entities the model cannot locate at all. Two definitions are
defensible:

* **Among checked**: score only relations where both entities were placed. A model
  that finds two structures and gets their relation right scores well even if it
  finds nothing else.
* **Strict**: score every relation whose entities are present in the scene, so failing
  to place an entity fails its relations.

On the first run the two diverged sharply, because the arms differ in how many
entities they can place at all. Reporting only the first would have flattered whichever
arm places fewer entities.

## Decision

Report both. Use the strict form for the pre-registered verdict, on the grounds that a
relation which cannot be recovered from the generated geometry has not been recovered.
Report coverage alongside, so the gap between the two numbers is visible.

## Alternatives considered

**Use the among-checked form.** Closer to a conventional accuracy. Rejected: it can be
maximised by locating two structures well and ignoring the rest, which is the opposite
of what the hypothesis is about.

**Drop the criterion as under-specified.** Defensible, and the most conservative
option. Rejected because both definitions are computed and reported, so a reader can
apply either threshold.

## Consequences

* This choice was made after seeing that the arms differ in coverage. It is therefore
  a post-hoc decision and is disclosed as one here, in the experiment report and in the
  Step 6 write-up.
* Both numbers appear in every report, so the verdict can be recomputed under the other
  definition by anyone who disagrees.
