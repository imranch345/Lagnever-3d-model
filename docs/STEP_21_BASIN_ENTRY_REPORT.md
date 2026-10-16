# Step 21 — When is seed 9's escape fixed? Basin entry by branching: report

**Decision: A — fixed from initialisation.** From every branch point tested (0, 200, 400, 600
and 700), all eight data orderings escape into graph use. So does Step 20's 805. That is
**48 of 48 branches in this step**, from bit-identical states that every reference reproduces
exactly. For seed 9, data ordering does not decide the outcome at any point, including step 0.

**H15 supported. H16 not supported for seed 9.**

**`test_splits_read: []`** in every artifact. Nothing causal is claimed beyond what an
intervention licenses (§6).

**Synthetic research data. Not anatomy, not validated, not clinical.**

Plan: [`STEP_21_BASIN_ENTRY_PLAN.md`](STEP_21_BASIN_ENTRY_PLAN.md), with Amendment 1.
Runs: `experiments/runs/step21/`. Summary: `experiments/runs/step21/summary.json`.

---

## 1. Integrity (§7)

| check | result |
| --- | --- |
| 1. All twelve freeze manifests clean, before and after | **182 of 182** files match at the Stage 1 gate and again at the end |
| 2. The capture reproduces frozen seed 9 | **18 of 18** shared checkpoints exact |
| 3. The capture's 805 state equals Step 20's frozen state | **identical**, field for field: parameter sum 2872.0822747918805, moment sum 111893.98677831954, 3,226,647 parameters, stream at epoch 5, batch 60 |
| 4. One starting fingerprint per point, distinct across points | 1 per point at all five points, and the points differ |
| 5. First-batch fingerprints differ across the 8 branches | 8 distinct at every point |
| 6. Each reference reproduces frozen seed 9 from its point on, escape included | **0: 43/43, 200: 39/39, 400: 35/35, 600: 31/31, 700: 29/29** checkpoints exact. Every reference escapes at 830 < e ≤ 835 |
| 7. Determinism replay (`branch_00` at 400, run twice) | identical at every checkpoint |
| 8. `OMP_NUM_THREADS=4`, at most three runs at once | yes |

No stop condition (§10) was met.

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

## 3. Results

### 3.1 Primary outcome (§4)

Stage 1 ran 400, 600 and 700. All three were unanimous, so the preregistered table selected
Stage 2a: 200 and 0.

| branch point | stage | escaped | verdict |
| --- | --- | --- | --- |
| 0 | 2a | 8/8 | **unanimous** |
| 200 | 2a | 8/8 | **unanimous** |
| 400 | 1 | 8/8 | **unanimous** |
| 600 | 1 | 8/8 | **unanimous** |
| 700 | 1 | 8/8 | **unanimous** |
| 805 (Step 20) | – | 8/8 | unanimous |

**What 8/8 means (§6).** Eight escapes out of eight is consistent with a true escape rate
anywhere above about 0.69 (one-sided 95%), point by point. "Unanimous" means a high rate, not
certainty. One non-escape would have made a point split. None occurred in 48 branches.

### 3.2 Every branch

Escape step is the first checkpoint from which graph usage stays at or above 1.5° to the end.
Final values are at step 1200, on validation. Each cell gives escape step · final usage ·
final rotation.

| loader seed | from 0 | from 200 | from 400 | from 600 | from 700 |
| --- | --- | --- | --- | --- | --- |
| 2000 | 800 · 2.436° · 21.007° | 810 · 2.565 · 20.970 | 800 · 2.581 · 20.975 | 835 · 2.558 · 20.984 | 820 · 2.522 · 21.017 |
| 2001 | 805 · 2.513 · 20.977 | 800 · 2.627 · 20.996 | 750 · 2.613 · 20.974 | 855 · 2.574 · 21.021 | 830 · 2.546 · 20.985 |
| 2002 | 825 · 2.559 · 20.925 | 860 · 2.516 · 21.026 | 800 · 2.564 · 20.992 | 855 · 2.499 · 21.073 | 855 · 2.510 · 20.990 |
| 2003 | 750 · 2.567 · 20.988 | 800 · 2.533 · 20.968 | 800 · 2.583 · 20.974 | 855 · 2.571 · 21.012 | 845 · 2.567 · 20.982 |
| 2004 | 750 · 2.545 · 20.947 | 700 · 2.530 · 20.966 | 800 · 2.546 · 20.963 | 865 · 2.492 · 21.036 | 850 · 2.548 · 20.993 |
| 2005 | 860 · 2.488 · 20.994 | 750 · 2.470 · 20.962 | 750 · 2.534 · 20.976 | 855 · 2.571 · 20.960 | 820 · 2.594 · 20.955 |
| 2006 | 800 · 2.553 · 20.975 | 815 · 2.457 · 21.035 | 800 · 2.560 · 20.982 | 865 · 2.598 · 20.991 | 830 · 2.575 · 20.970 |
| 2007 | 800 · 2.514 · 20.969 | 800 · 2.493 · 21.026 | 800 · 2.557 · 20.963 | 855 · 2.542 · 21.013 | 840 · 2.559 · 20.960 |
| reference | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 | 835 · 2.518 · 20.993 |

### 3.3 Secondary outcomes (§4)

| branch point | escape step: mean (SD) · range | final usage (mean ± SD) | final rotation (mean ± SD) |
| --- | --- | --- | --- |
| 0 | 798.8 (34.0) · 750–860 \* | 2.522° ± 0.041 | 20.973° ± 0.025 |
| 200 | 791.9 (44.6) · 700–860 \* | 2.524° ± 0.051 | 20.994° ± 0.029 |
| 400 | 787.5 (21.7) · 750–800 \* | 2.567° ± 0.023 | 20.975° ± 0.009 |
| 600 | 855.0 (8.7) · 835–865 | 2.551° ± 0.035 | 21.011° ± 0.032 |
| 700 | 836.3 (12.4) · 820–855 | 2.553° ± 0.025 | 20.981° ± 0.018 |
| 805 (Step 20) | 830–840 | – | 20.988° ± 0.023 |

SD is the population standard deviation, as in Step 21's summary.

\* Escapes before the dense window (805–900) are timed on the 50-step grid (§4). Points with
such escapes mix resolutions, and their spreads are upper-bounded by the grid. They are not
comparable with 600 and 700.

- **All 48 branches end in the graph-using mode.** Final rotation is 20.925–21.073°, against
  Step 15's graph-using spread of 0.021° and Step 20's 0.023°.
- **No branch is "stuck" or "in transition"** (§4 secondary outcome 3).

## 4. Predictions recorded before the run (§5)

| prediction | result |
| --- | --- |
| 1. Every reference reproduces, and the 805 state matches Step 20 | **confirmed**, after Amendment 1 (§2) |
| 2. 700 and 600 unanimous | **confirmed** |
| 3. 400 unanimous, weakly | **confirmed** |
| 4. Step 0 split, with non-escapers "in transition" | **wrong.** Step 0 is unanimous. All eight orderings escape, at 750–860, well inside the budget |
| 5. Escape-step spread grows as the branch point moves earlier | **broadly, not cleanly.** Range: 30 steps at 600, 35 at 700, then 110 at 0 and 160 at 200. SD at 600 (8.7) is below 700 (12.4), and before 805 the grid limits the comparison |

## 5. Decision (§6)

**A — fixed from initialisation.**
- Step 0 is unanimous, and so is every other point run: 200, 400, 600 and 700, with Step
  20's 805.
- For seed 9, no data ordering tried, from any point, changed whether the escape happens.
- H15 (fixed at initialisation) is **supported**.
- H16 (entered during training) is **not supported for seed 9**: there is no step before
  which orderings split the outcome.
- Outcome C (non-monotone) did not occur.

**What this licenses.** From seed 9's initial weights, the escape into graph use survives
every one of eight data orderings. Data order changes *when* it happens: from 700 to 865
across all branches, against seed 9's own 830–835. It does not change *whether* it happens,
nor where training ends (20.93–21.07°).

**What it does not license** (§6).
- It does not show that the initial weights *cause* the escape. Eight orderings are a
  sample, and other interventions (weights, optimiser, objective) were not tried.
- It says nothing about other seeds. That seed 9's initialisation lies in the graph-using
  basin does not say why Step 15's seeds split 6 / 3 / 1.

## 6. An observation the plan did not anticipate

**Ordering moves the timing, sometimes by a lot.**
- From 400, every branch escapes before seed 9 itself (750–800 against 830–835).
- From 200, one branch escapes at step 700, 130 steps early. Another escapes at 860, after
  seed 9.

This is reported as found. It is not a preregistered outcome, and no claim rests on it.

## 7. Interruptions, and why they do not touch the results

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

## 8. What follows (§12)

**Outcome A leads to Step 22:** branch **seed 5**, a non-escaper in Step 15, **at step 0**.
- If seed 5 is also unanimous (never escapes, under any ordering), both modes are set by
  initialisation. The programme's question then becomes what distinguishes a seed-9 start
  from a seed-5 start.
- If seed 5 splits, initialisation fixes the outcome for some starts but not others, and
  the matched pairs at its step 0 become the first controlled contrast (§12, B).

Step 22 is preregistered before any run, as every step has been. Nothing in this report
starts it.

## 9. Limitations

* One seed (9) and one architecture, on a synthetic corpus.
* Eight orderings per point. Unanimity is a lower bound on the escape rate (§3.1), not a
  certainty.
* The escape-time grid is coarser before step 805. Timing comparisons across points are
  limited by it.
* An intervention on data order only. Weights, optimiser state and initialisation are not
  varied here; that is Step 22's design.
