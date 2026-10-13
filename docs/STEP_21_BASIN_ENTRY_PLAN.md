# Step 21 — When is seed 9's escape fixed? Basin entry by branching: preregistration

**Status:** draft preregistration. Written before any Step 21 code or run exists. Binding once
committed; any later change is recorded as an amendment, not an edit.
**Authorised as** the next step recommended in
[`STEP_20_CHECKPOINT_BRANCHING_REPORT.md`](STEP_20_CHECKPOINT_BRANCHING_REPORT.md) §14.
**Question:** what is the earliest training step from which seed 9's escape into the
graph-using mode is unanimous across data orderings?
**Intervention, as in Step 20.** Architecture, objective, optimiser, initialisation, corpus,
budget and protocol frozen. The only thing varied is the training loader's ordering after a
branch point, and the only new thing is where the branch points sit.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why this step

Step 20 held seed 9's full state at step 805 fixed and varied only the subsequent data
ordering. All eight branches escaped, within 10 steps of each other, to within 0.023° of the
same final rotation. The outcome was already fixed at 805.

Step 20's own first limitation is the question here: 805 is 25 steps before the transition, so
the result says nothing about *where* the outcome became fixed. It could be step 700, step 400,
or initialisation. Steps 16–19 could not answer this by observation, because no measured
quantity moved before the transition. Branching can, because it asks the state directly.

One fact makes the answer worth having. At every candidate branch point from 200 to 750, seed 9
looks like a seed that never escapes. Its graph usage there is 0.14–0.38. Seed 5, which never
escapes, reads −0.05 to 0.82 over the same steps (Step 16 and Step 18 trajectories). If branches from such a
point are unanimous, the outcome is fixed in a state that every Step 16–19 measure reads as a
non-escaper.

## 2. A fact checked before writing this plan

The run seed sets both the initial weights and the training order, and Step 15 could not
separate the two. **Initialisation does differ by seed.** Built at the frozen configuration,
seeds 9 and 5 differ in 106 of 141 parameter tensors, with parameter sums 2782.06 and 2245.55.
Seeds 9, 5 and 3 all score 61.74° at step 0, but that coincidence is a property of the metric at
initialisation, not evidence of identical weights.

So a branch at **step 0** keeps seed 9's initial weights and changes only the ordering. That
separates initialisation from data order, which is the comparison Step 15 §8 asked for, at no
extra machinery cost.

## 3. Design

**One capture run.** Seed 9 is trained from scratch to step 805 at the frozen configuration,
saving a full continuable state (Step 20's `save_state`: parameters, AdamW moments, step counter,
stream position, RNG) at **0, 200, 400, 500, 600, 650, 700, 750 and 805**. The 805 state exists
only as a cross-check against Step 20's frozen state (§7). Nine states at ~39 MB is ~350 MB.

**Continuations from each branch point used**, exactly as in Step 20:

* **8 branches**, loader seeds **2000–2007**, the same seeds at every branch point so that
  `branch_k` means the same loader seed everywhere. Declared here and never extended.
* **1 reference**, which restores the original stream position exactly and must reproduce
  frozen seed 9.

Each continuation runs to step 1200, the frozen budget.

**Two stages, the second fixed now and chosen only by the first's result.** "Not unanimous"
below means split or none (§4).

| stage | branch points | runs if (400, 600, 700) |
| --- | --- | --- |
| **1** | **400, 600, 700** | always |
| **2a** | **200 and 0** | all three unanimous |
| **2b** | **500** | 400 not unanimous; 600 and 700 unanimous |
| **2c** | **650** | 400 and 600 not unanimous; 700 unanimous |
| **2d** | **750** | none of the three unanimous |

Stage 1 uses the three points Step 20 §14 named. Stage 2 narrows the bracket that Stage 1 leaves.
If all three are unanimous it runs both 200 and 0 rather than bisecting, because step 0 is the
initialisation-versus-order comparison of §2 and is too informative to leave to a third stage.
Any other Stage 1 pattern is non-monotone, with a later point not unanimous while an earlier
one is, and runs no Stage 2 (§6, outcome C).

## 4. Measurement

All definitions are Steps 15, 16 and 20's, unchanged:

* **Escape:** graph usage at or above **1.5°** (`GRAPH_USING_DEG`) from some checkpoint to the
  end of the run, without dropping back. It must happen within the 1200-step budget.
* **Recorder:** Step 16's `TrajectoryRecorder`, 50-step coarse grid, dense 5-step window
  **805–900** as in Step 20, so continuations from 805-adjacent points stay comparable. An escape
  outside the window is timed to 50 steps, and is reported at that resolution.
* **Validation only.** `test_splits_read: []` in every artifact.

**Primary outcome:** the escape rate at each branch point, out of 8.

* **unanimous**: 8/8
* **split**: 1–7 of 8
* **none**: 0/8

**Secondary outcomes**, reported for every branch point and never used to rescue the primary:

1. the escape-step distribution: mean, spread, minimum and maximum;
2. final graph usage and validation rotation, mean and spread across escaped branches, read
   against Step 15's 0.021° graph-using spread and Step 20's 0.023°;
3. **for each non-escaping branch**, its final usage: **stuck** if below 0.5 (Step 15's
   non-using seeds sit at 0.09–0.38), **in transition** if between 0.5 and 1.5 (as seed 3 was,
   at 1.30, at step 1200). Both count as non-escape for the primary outcome. The split matters
   because "too late for the budget" and "a different attractor" are different findings.

## 5. Hypotheses and predictions, recorded before the run

**H15 — fixed at initialisation.** Seed 9's escape is decided by its initial weights. Data
ordering does not change the outcome from any point. *Predicts* 8/8 at every branch point,
including 0.

**H16 — entered during training.** There is a step s* in (0, 805] before which orderings split
the outcome and from which it is unanimous. *Predicts* a monotone pattern: split early,
unanimous late.

H13 from Step 20 (state-determined from 805) is the common ground of both and is not retested.

**Predictions**, with their basis, written now so they can be wrong:

1. **Every reference reproduces frozen seed 9** at all shared checkpoints, and the capture's
   805 state matches Step 20's fingerprint. High confidence: the code path is Step 20's.
2. **700 and 600: 8/8.** At 805 the escape-timing spread was 10 steps, with about 370 steps of
   budget to spare. Another 100–200 steps of different ordering will probably widen the timing
   without moving any branch past the budget. Moderate confidence.
3. **400: 8/8, weakly.** No better basis than extrapolating prediction 2.
4. **0: split, with the non-escapers "in transition" rather than "stuck".** Low confidence. Step
   15's seeds escaped anywhere from mid-run to past the budget. Some of that spread may be
   ordering, and ordering alone could then push some branches past step 1200. Step 15 varied
   initialisation too, so this basis is weak, and that is why step 0 is worth running.
5. **The escape-step spread grows as the branch point moves earlier.** Moderate confidence.

## 6. Decision rule

* **A — fixed from initialisation.** Step 0 is unanimous, and so is every point run. H15
  supported. For seed 9, ordering does not decide the outcome at any point.
* **B — entered during training.** The pattern is monotone and brackets the entry:
  s* ∈ (latest point not unanimous, earliest unanimous point]. H16 supported. The bracket width
  is reported as the precision: 200 steps if the entry lies before 400, 100 between 400 and 600,
  and about 50 after 600. If the bracket is (700, 805] or narrower, it is reported as **late
  entry**. That case is worth naming because no Step 16–19 quantity moved there.
* **C — non-monotone.** A later point is not unanimous while an earlier one is. No entry step is
  claimed and the pattern is reported as found. It would mean the basin picture is too simple.
* **Invalid.** Any reference fails to reproduce frozen seed 9, or any capture state fails its
  check. Stop and report. No branch result is interpreted.

**What 8/8 can and cannot say.** Eight escapes out of eight is consistent with a true escape rate
anywhere above about 0.69 (one-sided 95%). "Unanimous" therefore means a high escape rate, not
certainty. One non-escape is enough to call a point split. That asymmetry is deliberate and is
stated in the report next to every verdict.

**Nothing beyond what an intervention licenses.** Unanimity at s shows the outcome survived
every ordering tried from s. It does not show that the state at s causes it.

## 7. Integrity

Carried over from Step 20, plus one new cross-check:

1. **All twelve freeze manifests** in `experiments/runs/` are verified CLEAN before and after.
2. **The capture reproduces frozen seed 9** at every shared checkpoint: Step 16's coarse grid,
   and Step 18 run A's dense 600–900 window where it overlaps.
3. **New: the capture's 805 state matches Step 20's frozen `state_seed9_step805.pt`.** Parameter
   sum 2872.0822747919 and moment sum 111893.9867783195. This proves the multi-point capture is
   the same state Step 20 branched from, not a near relative.
4. **One starting fingerprint** across all continuations from a given point. Distinct
   fingerprints across points.
5. **First-batch fingerprints differ** across the 8 branches at every point.
6. **The reference at every branch point** reproduces frozen seed 9 from that point on,
   including the escape at 830 < e ≤ 835.
7. **One determinism replay**: `branch_00` at the earliest Stage 1 point, run twice, identical
   at every checkpoint.
8. **`OMP_NUM_THREADS=4`** for every run, part of the frozen protocol since Step 17. At most
   three runs at once, as in Step 20, which reproduced exactly under that load.

## 8. Code, before any run

* `experiments/step21/basin_entry.py`: Step 20's runner, generalised to a list of branch points
  and a multi-point capture. `experiments/step20/state.py` is reused **unchanged**.
* `tests/test_step21_basin_entry.py`: a state saved at s mid-capture continues bit-identically
  to an uninterrupted run, a branch at step 0 keeps seed 9's initial weights while changing the
  first batch, and stage selection follows §3's table exactly.
* No change to `training/`, `generation/` or any frozen module. The full suite (1024 tests) must
  pass before the capture starts.

## 9. Compute and resources

From Step 20's logs, three concurrent 395-step continuations took about 17 minutes, so
throughput is about 0.86 s per step-run. The capture runs alone, at about 1.2 s per step.

| stage | continuations | step-runs | estimated wall time |
| --- | ---: | ---: | --- |
| capture to 805 | 1 | 805 | ≈ 16 min |
| 1: 400, 600, 700, plus one replay | 28 | 17,900 | ≈ 4.3 h |
| 2a: 200 and 0 | 18 | 19,800 | ≈ 4.7 h |
| 2b / 2c / 2d: one point | 9 | 6,300 / 4,950 / 4,050 | ≈ 1.5 / 1.2 / 1.0 h |

Worst case is about 9.5 hours of machine time across two sittings. Disk use is about 0.4 GB:
nine states plus ~50 KB per trajectory JSON. 37 GB is free now. The machine runs on AC power
throughout.

## 10. Stop conditions

* any frozen checksum changes;
* the capture does not reproduce frozen seed 9, or its 805 state does not match Step 20's;
* any reference continuation fails to reproduce frozen seed 9;
* free disk below 2 GiB. No research artifact is deleted to make space;
* a run would need a change to the model, objective, optimiser or budget.

## 11. What this step does not do

* **No seed 5.** Branching a non-escaper is the mirror experiment, Step 22. Its design depends on
  this step's result. If the answer is A, its decisive point is step 0.
* **No budget extension.** A branch "in transition" at 1200 stays a non-escape here. Whether it
  would escape later is Step 23's question.
* **No architecture, objective or optimiser change**, no test split, no new metric.

## 12. What each outcome would lead to

* **A:** Step 22 branches seed 5 at step 0. If seed 5 is also unanimous, both modes are set by
  initialisation, and the programme's question becomes what distinguishes a seed-9 start from a
  seed-5 start.
* **B:** the branches at the latest point that was not unanimous form a matched pair: identical
  state, different futures. Comparing the escaping and non-escaping branches over the steps right after
  that point is the first contrast in this programme that is controlled by construction. It
  replaces the single-trajectory searches of Steps 16–19.
* **C:** stop bisecting. Report the pattern, and reconsider whether "basin entry" is the right
  description before designing anything else.
