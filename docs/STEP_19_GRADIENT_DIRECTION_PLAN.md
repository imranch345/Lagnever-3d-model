# Step 19 — Gradient direction across the transition: preregistration

**Status:** preregistered. Written before the run was launched.
**Authorised as** the next step recommended in
[`STEP_18_ONSET_AND_REPLICATION_REPORT.md`](STEP_18_ONSET_AND_REPLICATION_REPORT.md) §9.
**Question:** does the *direction* of the gradient change before seed 9's transition, in a way
that four steps of magnitude measurement could not see?
**Diagnostic only.** Architecture, objective, optimiser, initialisation, corpus, budget and
protocol frozen. The only change is what is recorded.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why direction

Steps 16, 17 and 18 searched for a precursor at 50-step, 5-step, and corrected-baseline
resolution. The answer was the same each time, and Step 17's one apparent positive dissolved
under a better variance estimate. **Every gradient number in that search was an aggregate L2
norm per module**, which is blind to where the gradient points.

Step 18 established that seed 9's transition occupies about ten steps (subset 810, usage 815,
rotation 820) with no change in any gradient magnitude. A ten-step reorganisation at constant
magnitude is what a change in gradient *direction* looks like. That is the one thing the existing
instrumentation cannot see, and it is why a fourth magnitude study would have been the wrong
experiment.

## 2. The run

One run: **seed 9, frozen configuration, direction window 780–860** at 5-step spacing, 1200
steps, 4 threads (part of the frozen protocol since Step 17). The window brackets Step 18's
810–820 transition with 30 steps of margin on the early side.

Step 16's trajectory recorder runs inside this one unchanged, so every series from Steps 16–18 is
recorded alongside and the comparison stays aligned.

## 3. The two quantities, and what each may claim

**`cosine_successive`** — cos(g_t, g_{t−1}), per pathway and overall. Uses only information
available at step t. **This is the candidate precursor.** If the gradient direction destabilises
or realigns before step 810, it appears here.

**`cosine_escape_axis`** — cos(−g_t, w_final − w_780): the alignment of the descent step with the
displacement the weights actually undergo. **This axis is computed from the end of the run, so it
is not available at training time and cannot be a predictive signal.** It answers a weaker,
descriptive question — was the gradient already pointing along the eventual displacement before
the geometry moved? It will be reported as descriptive and never as a precursor. The artifact
carries `uses_future_information: true` so the distinction cannot be lost downstream.

## 4. Detection criterion

**Step 18's lesson is binding here:** a 3σ rule is only as good as its variance estimate, and two
successive reports were misled by a baseline sampled more coarsely than the measurement. So the
baseline for every series is estimated **at 5-step resolution**, from the window's own pre-
transition span **780–805** (6 checkpoints), and a departure must exceed 3 baseline standard
deviations and persist across three consecutive checkpoints.

Six checkpoints is a thin variance estimate and is stated as a limitation rather than hidden. Where
the pre-transition span is too short to separate signal from noise, the honest report is that the
criterion cannot decide — not a verdict either way.

## 5. Predictions, recorded before the run

1. **`cosine_successive` will be low and noisy throughout**, because minibatch gradients at batch
   8 are dominated by sampling noise. A clean pre-transition shift is unlikely.
2. **No sustained departure in `cosine_successive` before step 810.** Basis: three prior steps
   found no precursor in any quantity, and direction is measured on the same noisy gradients.
3. **`cosine_escape_axis` will be positive throughout the window and will not jump at 810**,
   because the axis is dominated by the large post-transition displacement, and the gradient
   pointed broadly downhill before the transition too. If it does jump at 810 that is the
   transition registering, not a precursor.

The live null is that direction is as uninformative as magnitude, in which case Steps 16–19 have
collectively established that **this instrumentation class cannot resolve the transition**, and
the recommendation will be to change the class of measurement rather than its resolution again.

## 6. Decision rule

* **A — direction precursor found.** `cosine_successive` departs before step 810 and survives the
  falsification checks carried over from Step 17.
* **B — ambiguous.** A shift appears near the transition but the six-checkpoint baseline cannot
  separate it from noise.
* **C — no direction precursor.** No sustained departure before the transition.

**B and C will not be upgraded into a mechanism.** One run; nothing causal.

## 7. Integrity

The direction callback is heavier than Step 16's — it runs extra backward passes and holds
flattened gradients — so neutrality was retested rather than inherited. **The first version failed
that test**: it measured outside the RNG guard, and an instrumented run diverged from its twin.
With the guard in place a 60-step instrumented run now reproduces its uninstrumented twin with
zero differing tensors.

A second bug was caught by the same suite: gradient and weight vectors were flattened over
different parameter sets (parameters with no gradient were dropped on one side), making the
cosines meaningless. Both are now built over one sorted name list with zeros substituted for
missing gradients.

**`test_splits_read: []`.** Validation only for metrics; gradients on the fixed training batches
Steps 14–18 used.

## 8. Stop conditions

* any frozen checksum changes;
* the run does not reproduce frozen seed 9 at the shared coarse checkpoints;
* free disk below 2 GiB; no research artifact is deleted to make space;
* a diagnostic would require changing the model, objective or budget.
