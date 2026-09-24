# Step 18 — Usage-rise onset and replication: preregistration

**Status:** preregistered. Written while the two runs trained, before any data inside either new
window was inspected.
**Authorised as** the next step recommended in
[`STEP_17_HIGH_RES_ESCAPE_REPORT.md`](STEP_17_HIGH_RES_ESCAPE_REPORT.md) §17.
**Diagnostic only.** Architecture, objective, optimiser, initialisation, corpus, budget and
protocol frozen. The only change is where the dense recording window sits.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why these two runs

Step 17 resolved seed 9's escape into an ordered process — graph usage departs its baseline at
step 755, geometry follows at 815, the 1.5° threshold is crossed between 830 and 835 — and found
no representation, gradient or objective quantity moving before any of it.

It left exactly one binding limitation, stated in its own §16: **the window opened after the rise
had begun.** Usage at step 750 was 0.37, the largest value in its own 400–750 baseline, and the
coarse value at step 700 was 0.240. So "usage leads geometry by ~60 steps" is a lower bound and
the onset is unresolved.

It also left the ordering resting on a single run.

## 2. The two runs

| run | seed | dense window | purpose |
| --- | ---: | --- | --- |
| **A** | 9 | **600–900** | locate the onset of the usage rise |
| **B** | 3 | **900–1200** | test whether the ordering recurs in a second seed |

Both at the frozen configuration, 1200 steps, λ=3.0, validation only, 5-step spacing inside the
window plus the usual 50-step grid outside.

**Run A extends to 900 rather than stopping at 800**, which Step 17 §17 proposed. The extra
hundred steps cost little and buy a free internal consistency check: run A overlaps Step 17's
window across 750–900, so the two independent runs must agree bit-for-bit there. A disagreement
would be a finding about reproducibility, not about onset.

**Run B's window is 900–1200, not the final 200 steps** as §17 loosely put it. Step 16's coarse
trajectory for seed 3 is usage 0.019 at step 900, 0.234 at 1000, 0.899 at 1100, 1.305 at 1200, so
the onset lies between 900 and 1000. Opening at 900 brackets it — applying the lesson that cost
Step 17 its headline claim, rather than repeating the mistake.

Seed 3 never crosses 1.5° within the budget, so run B measures the *ordering*, not an escape.

## 3. Baselines

Each window needs a baseline that precedes it, taken from the coarse grid:

| run | window | baseline | coarse points |
| --- | --- | --- | ---: |
| A (seed 9) | 600–900 | **300–600** | 7 |
| B (seed 3) | 900–1200 | **400–900** | 11 |

Seven points is thin, and the resulting standard deviation is a weak estimate; that is a stated
limitation rather than a hidden one. Detection criterion is otherwise **unchanged from Step 17**:
the first dense checkpoint more than 3 baseline standard deviations from the baseline mean, with
the departure sustained across three dense checkpoints (15 steps).

## 4. Definitions carried over unchanged

Graph usage is Step 15's. The relationally-required subset is Step 14's 652 entities. Escape is
Step 15's 1.5° threshold. The precursor definition and the five falsification checks are Step
17's, **including its two corrections**: validation rotation and the relationally-required subset
are excluded as candidate precursors because graph usage is
`rotation_without_graph − rotation_intact` and they are its terms; and the standing-offset check
uses the declared baseline rather than every earlier step.

## 5. Predictions, recorded before inspecting either window

1. **Seed 9's usage onset lies before step 750.** Basis: 750 was already its baseline maximum and
   step 700 sat at 0.240.
2. **Usage still departs before geometry in seed 9**, with the gap larger than Step 17's 60 steps
   once the true onset is visible.
3. **No upstream quantity departs before the usage onset** in either run. Basis: Steps 16 and 17
   both found none, twice, at two resolutions.
4. **Seed 3 shows the same ordering** — usage departs its baseline before its geometry does.

Live null outcomes, each of which would be reported plainly:

* the onset is at or after 750, meaning Step 17's window was adequate and prediction 1 was wrong;
* seed 3's geometry moves first, meaning the ordering is seed-specific and prediction 4 fails;
* an upstream quantity departs first once a longer window is available, overturning the Step
  16/17 null.

## 6. Reproduction requirements

* Run A must reproduce Step 17's dense values **bit-identically across 750–900**, and frozen seed
  9 at the shared coarse checkpoints.
* Run B must reproduce frozen seed 3 (Step 16) at its shared coarse checkpoints and at step 1200
  (usage 1.305, rotation 22.17).
* **Both runs use 4 threads**, matching Steps 16 and 17. Step 17 established that torch CPU
  reductions on this machine depend on per-process thread count: a 6-thread run diverged from the
  frozen trajectory after ~150 steps. Thread count is part of the frozen protocol now.

## 7. Decision rule

For the onset question, exactly one of:

* **A — onset located.** The usage departure is found strictly inside 600–900 with a baseline
  interval reported, and the window contains quiet checkpoints before it.
* **B — onset still unresolved.** The departure sits at or adjacent to the window's opening edge,
  meaning the rise again begins before the window.
* **C — no departure in the window**, which would contradict Step 17 and require investigation
  before interpretation.

For the precursor question, Step 17's rule is reused unchanged (A precursor resolved / B
unresolved / C effectively abrupt), and **B or C will not be upgraded into a mechanism.**

For the replication, the ordering either recurs in seed 3 or it does not; with two seeds nothing
is claimed about the population.

## 8. Test isolation

**`test_splits_read: []`.** Validation only. Both windows were chosen from Step 16 and Step 17
artifacts.

## 9. Preflight and stop conditions

Disk 8.3 GiB free at launch — above the 2 GiB guard but **down 5.7 GiB since the Step 17 check
with no corresponding writes from this work**, so it is recorded and watched. Battery 46% and
discharging with about 2:16 estimated; two ~20-minute runs should fit, the runner writes per-run
artifacts atomically, and no research artifact is deleted to make space.

Stop if: any frozen checksum changes; either run fails to reproduce its frozen counterpart;
free disk falls below 2 GiB; or a diagnostic would require changing the model, objective or
budget.
