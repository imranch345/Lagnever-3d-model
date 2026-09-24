# Step 17 — High-resolution escape window: preregistration

**Status:** preregistered. Written while the single instrumented run trained, before any
trajectory inside the window was inspected.
**Question:** between steps 750 and 900, does anything measurable change *before* seed 9 escapes
the identity-only attractor, or does the transition remain effectively abrupt at 5-step
resolution?
**Diagnostic only.** Architecture, objective, optimiser, initialisation, corpus, budget and
protocol are frozen. The only change is recording frequency.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Configuration verification

All seven prior freeze manifests verified **CLEAN**: corpus, Change 1, Step 12, Step 13, Step 14,
Step 15, Step 16. Git `589b925`, clean tree. Seed 9's configuration asserted live before
training: 1200 steps, λ=3.0, `graph_recurrence` 1, `frame_scene_context` False,
`relation_values` False.

**The window is preregistered from Step 16 alone**: Step 16 localised seed 9's escape to
`800 < escape ≤ 850`, and 750–900 brackets it with a margin either side. No test result informed
it.

## 2. The single change

`TrajectoryRecorder` gained an optional `dense_window`. With none, checkpoint selection is
byte-identical to Step 16's (25 checkpoints, verified). With `(750, 900)` it measures the
50-step grid **and** every 5th step inside the window: **31 dense checkpoints — 750, 755, …,
900 — plus 21 coarse ones outside**, 52 in total. Verified by direct enumeration before
launching.

Retaining the coarse grid outside the window is deliberate: the dense run therefore carries
every Step 16 checkpoint, so the two can be compared at shared steps without a second run.

Measurement definitions are **unchanged from Step 16** — graph usage is Step 15's, the
relationally-required subset is Step 14's 652 entities, and the representation, gradient and
objective series are the same code. Nothing was added or renormalised.

## 3. What counts as a precursor

All five conditions must hold (from §9 of the brief):

1. it occurs **before** the escape checkpoint;
2. it is **not merely a persistent offset** already present before step 750;
3. it moves in a direction associated with the subsequent escape;
4. the change is **temporally distinguishable** from the escape at 5-step resolution;
5. it uses an **existing measurement definition**, not a metric invented after seeing the data.

**A quantity that moves at the same checkpoint as graph usage and geometry is not a precursor —
it is part of the transition.**

### 3.1 Detection criterion

Fixed here, before inspecting the window. For each series, the pre-window baseline is its values
at the coarse checkpoints from step 400 to step 750 inclusive. A series is *changing* at the
first dense checkpoint whose value lies more than **3 baseline standard deviations** from the
baseline mean **and** whose next two dense checkpoints also lie outside, i.e. a departure
sustained over 15 steps.

Two-of-three persistence is required because at 5-step spacing single-checkpoint excursions are
common; Step 16 used one-checkpoint persistence at 50-step spacing, and the denser the sampling
the more noise-prone a single point is.

The baseline deliberately **excludes** the initial convergence transient, which is the flaw Step
16 documented in its own preregistered criterion. It is stated here in advance rather than
discovered afterwards.

## 4. Escape definition

**Step 15/16's rule, unmodified:** graph usage ≥ **1.5°**. Reported as
`last checkpoint before < escape ≤ first checkpoint after`.

**A crossing that falls back below the threshold is reported as a transient crossing, not an
escape**, and the permanent escape is the first crossing after which every later checkpoint stays
above.

## 5. Falsification, applied to every candidate

From §14 of the brief. For each candidate precursor:

1. was it already offset before step 750?
2. does the same quantity move in Step 16's non-escaper (seed 5) trajectory?
3. does the movement merely reflect the transition already under way?
4. could measurement noise explain it, given the baseline spread?
5. does the quantity depend on the graph-usage calculation itself?

A candidate failing any check is classified **non-diagnostic**.

## 6. Reproduction check

The dense run must reproduce frozen seed 9 at steps 750, 800, 850, 900 and 1200. Instrumentation
is observational; a difference at any shared checkpoint stops interpretation and becomes the
finding instead.

## 7. Decision rule

Exactly one of:

* **A — precursor resolved.** A variable changes before escape at 5-step resolution and survives
  every falsification check.
* **B — unresolved.** A candidate appears close to escape but temporal separation or measurement
  quality is insufficient.
* **C — effectively abrupt.** No measured quantity changes before escape at 5-step resolution.

**B and C will not be upgraded into a mechanistic explanation.** This is one run: language will
be "preceded", "coincided with", "was consistent with", "was not detected" — not "caused",
"triggered" or "proved".

## 8. Test isolation

**`test_splits_read: []`**. Validation only. The window was fixed from Step 16 before this run.

## 9. Stop conditions

* any frozen checksum changes;
* the dense run does not reproduce frozen seed 9 at a shared checkpoint;
* the escape falls outside 750–900, which would mean the window was chosen wrongly and the
  result would be reported as such rather than the window moved;
* free disk below 2 GiB; no research artifact is deleted to make space.
