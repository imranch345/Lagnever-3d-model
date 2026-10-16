# Step 21 — When is seed 9's escape fixed? Basin entry by branching: report

**Status: Stage 1 complete and valid; Stage 2 (branch points 200 and 0) running.** The
decision under §6 (A, B or C) is recorded only when Stage 2 completes. This report is
filled in as the preregistration's sections are answered.

**Stage 1: unanimous at 400, 600 and 700.** Each point is 8 of 8 orderings. With Step 20's
805, the pattern from 400 on is monotone, so outcome C is ruled out for 400–805.

**`test_splits_read: []`** in every artifact. Nothing causal is claimed beyond what an
intervention licenses (§6).

**Synthetic research data. Not anatomy, not validated, not clinical.**

Plan: [`STEP_21_BASIN_ENTRY_PLAN.md`](STEP_21_BASIN_ENTRY_PLAN.md), with Amendment 1.
Runs: `experiments/runs/step21/`.

---

## 1. Integrity (§7)

| check | result |
| --- | --- |
| 1. All twelve freeze manifests clean, before and after | **182 of 182** files match at the Stage 1 gate; checked again when Stage 2 ends |
| 2. The capture reproduces frozen seed 9 | **18 of 18** shared checkpoints exact |
| 3. The capture's 805 state equals Step 20's frozen state | **identical**, field for field: parameter sum 2872.0822747918805, moment sum 111893.98677831954, 3,226,647 parameters, stream at epoch 5, batch 60 |
| 4. One starting fingerprint per point, distinct across points | 1 per point at 400, 600 and 700, and the three differ |
| 5. First-batch fingerprints differ across the 8 branches | 8 distinct at every point |
| 6. Each reference reproduces frozen seed 9 from its point on, escape included | **400: 35/35, 600: 31/31, 700: 29/29** checkpoints exact; every reference escapes at 830 < e ≤ 835 |
| 7. Determinism replay (`branch_00` at 400, run twice) | identical at every checkpoint |
| 8. `OMP_NUM_THREADS=4`, at most three runs at once | yes |

The Stage 2 gate (`stage_two_gate`) reported no problems, and Stage 2 started automatically.

## 2. The first Stage 1 run was invalid (Amendment 1)

The first Stage 1 failed check 6 at every point. Only the branch-point checkpoint itself
matched (1/35, 1/31, 1/29).
- **Cause.** During the teacher-forcing ramp (steps 180–719), the model draws a per-entity
  mask with `torch.rand`, which consumes global RNG. `load_state` never restored the saved
  RNG. Step 20's audit was made at step 805, where nothing draws, so it could not see this.
- **Fix.** Restore the RNG.
- **Recorded.** Amendment 1, with tests. The invalid run is kept in
  `experiments/runs/step21/ABANDONED_stage1_no_rng/`. Its branch results were never
  interpreted.

## 3. Stage 1 results

### 3.1 Primary outcome (§4)

| branch point | escaped | verdict |
| --- | --- | --- |
| 400 | 8/8 | **unanimous** |
| 600 | 8/8 | **unanimous** |
| 700 | 8/8 | **unanimous** |
| 805 (Step 20) | 8/8 | unanimous |

**What 8/8 means (§6).** Eight escapes out of eight is consistent with a true escape rate
anywhere above about 0.69 (one-sided 95%). "Unanimous" means a high rate, not certainty.

### 3.2 Every branch

Escape step is the first checkpoint from which graph usage stays at or above 1.5° to the end.
Final values are at step 1200, on validation.

| loader seed | from 400: escape · usage · rotation | from 600 | from 700 |
| --- | --- | --- | --- |
| 2000 | 800 · 2.581° · 20.975° | 835 · 2.558° · 20.984° | 820 · 2.522° · 21.017° |
| 2001 | 750 · 2.613 · 20.974 | 855 · 2.574 · 21.021 | 830 · 2.546 · 20.985 |
| 2002 | 800 · 2.564 · 20.992 | 855 · 2.499 · 21.073 | 855 · 2.510 · 20.990 |
| 2003 | 800 · 2.583 · 20.974 | 855 · 2.571 · 21.012 | 845 · 2.567 · 20.982 |
| 2004 | 800 · 2.546 · 20.963 | 865 · 2.492 · 21.036 | 850 · 2.548 · 20.993 |
| 2005 | 750 · 2.534 · 20.976 | 855 · 2.571 · 20.960 | 820 · 2.594 · 20.955 |
| 2006 | 800 · 2.560 · 20.982 | 865 · 2.598 · 20.991 | 830 · 2.575 · 20.970 |
| 2007 | 800 · 2.557 · 20.963 | 855 · 2.542 · 21.013 | 840 · 2.559 · 20.960 |
| reference | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 |

### 3.3 Secondary outcomes (§4)

| branch point | escape step: mean (SD) · range | final usage (mean ± SD) | final rotation (mean ± SD) |
| --- | --- | --- | --- |
| 400 | 787.5 (21.7) · 750–800 \* | 2.567° ± 0.023 | 20.975° ± 0.009 |
| 600 | 855.0 (8.7) · 835–865 | 2.551° ± 0.035 | 21.011° ± 0.032 |
| 700 | 836.3 (12.4) · 820–855 | 2.553° ± 0.025 | 20.981° ± 0.018 |
| 805 (Step 20) | 830–840 | – | 20.988° ± 0.023 |

\* Escapes before the dense window (805–900) are timed on the 50-step grid (§4). The spread
at 400 is not comparable with the others'.

All 24 branches end in the graph-using mode: final rotation 20.955–21.073°, against
Step 15's graph-using spread of 0.021° and Step 20's 0.023°. No branch is "stuck" or "in
transition".

## 4. Predictions recorded before the run (§5)

| prediction | result |
| --- | --- |
| 1. Every reference reproduces, and the 805 state matches Step 20 | **confirmed**, after Amendment 1 (§2) |
| 2. 700 and 600 unanimous | **confirmed** |
| 3. 400 unanimous, weakly | **confirmed** |
| 4. Step 0 split, with non-escapers "in transition" | *pending Stage 2* |
| 5. Escape-step spread grows as the branch point moves earlier | **not cleanly**. SD is 8.7 steps at 600 and 12.4 at 700, both above 805's; 400 cannot be compared (50-step grid) |

## 5. An observation the plan did not anticipate

**Every branch from 400 escapes before seed 9 itself does**: at 750–800, against 830–835.
From 700 they escape both before and after it (820–855). From 600, at or after it (835–865).
- Data ordering moves *when* the escape happens, by 30–85 steps from 400.
- It has not changed *whether* the escape happens, from any point tested.
- Every escaped branch ends at the same rotation to within 0.12°.

This is reported as found. It is not a preregistered outcome, and no claim rests on it.

## 6. Stage 2 (§3): branch points 200 and 0

*Running.* The preregistered table selected Stage 2a because all three Stage 1 points were
unanimous. Step 0 is the comparison of initialisation against order (§2):
- if step 0 is unanimous, the outcome is A: fixed at initialisation;
- if it is not, the outcome is B: entered during training, bracketed between the latest
  non-unanimous point and the earliest unanimous one.

## 7. Decision (§6)

*Pending Stage 2.*

## 8. Interruptions, and why they do not touch the results

The machine restarted twice during the sweep: at about 03:20 and at about 14:24 on
2026-10-16. Each time the chain was resumed with the same command.
- **What was kept.** Completed runs, those with a result JSON, were skipped.
- **What was rerun.** A run that had been stopped part-way was started again from its saved
  branch-point state. It was not continued from where it stopped.
- **Why the reruns are trustworthy.** Every reported run is therefore one complete
  continuation. The determinism replay (§1, check 7) shows that a rerun from the same state
  is identical.

Point 700's branches 02–04 and point 200's branches 05–07 were rerun this way. The resumes
are marked in `sweep.log` ("=== RESUMED ...").

## 9. Limitations

* One seed (9) and one architecture, on a synthetic corpus.
* Eight orderings per point. Unanimity is a lower bound on the escape rate (§3.1), not a
  certainty.
* The escape-time grid is coarser before step 805. Timing comparisons across points are
  limited by it.
* An intervention on data order only. Weights, optimiser state and initialisation are not
  varied here; that is Step 22's design.
