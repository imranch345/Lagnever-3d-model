# Step 22 — Is the non-escape fixed too? Seed 5 at initialisation, and a crossed swap: report

**Decision: B′ — data order can rescue a non-escaper.**
- From seed 5's initial weights, **1 of 8** orderings escapes into graph use (at step 950,
  ending at 21.06°, a full member of the graph-using mode). One more ends "in transition",
  and six stay stuck.
- From step **400** and from step **800**, **0 of 8** escape.
- Seed 5's non-escape is therefore *not* fixed at initialisation. It is fixed by step 400:
  the bracket is (0, 400].

**H18 supported for seed 5. H17 not supported.** Together with Step 21 the picture is
asymmetric:
- seed 9's escape survives every ordering from step 0;
- seed 5's non-escape does not, though it is the strong tendency (7 of 8), and it is sealed
  within the first 400 steps.

**The crossed swap agrees with the weights.**
- Seed 9's weights with seed 5's batch order escape (step 750).
- Seed 5's weights with seed 9's batch order stay stuck.

**`test_splits_read: []`** in every artifact. Nothing causal is claimed beyond what an
intervention licenses (plan §6).

**Synthetic research data. Not anatomy, not validated, not clinical.**

Plan: [`STEP_22_MIRROR_PLAN.md`](STEP_22_MIRROR_PLAN.md), with its clarifications before any
code. Runs: `experiments/runs/step22/`. Summary: `experiments/runs/step22/summary.json`.

---

## 1. Integrity (§7)

| check | result |
| --- | --- |
| 1. Freeze manifests clean, before and after | **182 of 182** before the start and at the end |
| 2. The capture reproduces frozen seed 5 (Step 16) | **17 of 17** shared checkpoints, 0–800, exact |
| 2. Each reference reproduces frozen seed 5 from its point on | **0: 25/25, 400: 17/17, 800: 9/9**, exact |
| 3. Identity controls equal their references | `ident_5w_5o` = seed 5's reference, **exact**; `ident_9w_9o` = Step 21's step-0 reference, **exact** |
| 4. One starting state per point; seed 5's step 0 differs from seed 9's | 1 at each of 0, 400 and 800; the three capture states differ; seed 5 ≠ seed 9 at step 0 |
| 5. First batches differ across the 8 branches | 8 distinct at 0, 400 and 800 |
| 6. Determinism replay (`branch_00` at step 0, twice) | identical at every checkpoint |
| 7. `OMP_NUM_THREADS=4`, at most three runs at once | yes |

No stop condition (§10) was met.

**Two interruptions, neither touching a result.**
- **A bug at launch.** The runner's declared-run check refused every continuation before it
  started: runs were keyed by name only, so point 800 shadowed point 0. It was fixed and
  tested (`3ffb62f`). No continuation had produced a result, and the capture was unaffected.
- **The machine stopped overnight** at about 02:53, on battery. Branches 05–07 were near step
  830. The chain resumed at 21:19 the next day, only once on AC power. Completed results were
  kept, and stopped runs restarted from their saved state.

Both are marked in `sweep.log` and recorded in the lab notebook.

## 2. Results

### 2.1 Arm 1 — seed 5 from step 0 (primary, §4)

| loader seed | escaped | escape step | final usage | final rotation | class |
| --- | --- | --- | --- | --- | --- |
| 2000 | no | – | 0.064° | 23.511° | stuck |
| 2001 | no | – | 0.127 | 23.485 | stuck |
| 2002 | no | – | 0.225 | 23.468 | stuck |
| 2003 | no | – | 0.050 | 23.545 | stuck |
| 2004 | no | – | 0.190 | 23.432 | stuck |
| 2005 | no | – | 0.138 | 23.510 | stuck |
| **2006** | **yes** | **950** | **2.365** | **21.063** | **escaped** |
| 2007 | no | – | 0.615 | 23.165 | in transition |
| reference | no | – | 0.095 | 23.526 | stuck |

**Verdict: split (1 of 8).**
- The escaping branch is a full escape, not a partial one:
  - its graph usage, 2.37°, is inside Step 15's graph-using band (2.26–3.82°);
  - its rotation, 21.06°, is within the range of Step 21's 48 escaped branches
    (20.93–21.07°) and 2.4° below the non-using band.
- Branch 2007 moved partway (0.615°) and had not finished by the budget.

**What 1 of 8 can say.** The escape rate from seed 5's start is not zero. Its 95% interval is
wide: about 0.003 to 0.53 (two-sided, exact). "Split" is the finding; no rate is claimed.

### 2.2 Stage 2 — seed 5 from steps 400 and 800

| branch point | escaped | verdict | final usage, all 8 (mean, range) |
| --- | --- | --- | --- |
| 400 | 0/8 | **none** | 0.081° (0.062–0.101) |
| 800 | 0/8 | **none** | 0.090° (0.062–0.123) |

All 16 branches are stuck. Ordering 2006, which rescued seed 5 from step 0, does not rescue
it from 400 or 800. **0 of 8 is consistent with an escape rate below about 0.31**
(one-sided 95%) at each point.

**Bracket.** Seed 5's non-escape is entered during training, in **(0, 400]**. The precision
is 400 steps, the plan's spacing.

### 2.3 Arm 2 — the crossed swap (secondary)

| run | weights and global RNG | batch order | escaped | escape step | final usage | final rotation |
| --- | --- | --- | --- | --- | --- | --- |
| `cross_9w_5o` | seed 9 | seed 5's | **yes** | 750 | 2.611° | 20.978° |
| `cross_5w_9o` | seed 5 | seed 9's | **no** | – | 0.201 | 23.510 |
| `ident_9w_9o` | seed 9 | seed 9's | yes | 835 | 2.518 | 20.993 |
| `ident_5w_5o` | seed 5 | seed 5's | no | – | 0.095 | 23.526 |

In this one pair, the outcome followed the weights, not the order:
- seed 9's weights escaped even with seed 5's order, and 85 steps earlier than with their own;
- seed 5's weights stayed stuck with seed 9's order.

The swap is batch order only. The model's own random draws stay with the weights' seed
(plan §2). The identity controls prove the swap changed nothing else.

## 3. Predictions recorded before the run (§5)

| prediction | result |
| --- | --- |
| 1. References reproduce, and the identity controls equal their references | **confirmed**, all exact |
| 2. Arm 1 = none, all eight stuck | **wrong.** One branch escapes and one is in transition |
| 3. Swap: `cross_5w_9o` stays, `cross_9w_5o` escapes | **confirmed** |
| 4. Any seed-5 escape is late (after 900) | **confirmed.** The one escape is at 950, against 750–860 for seed 9's step-0 branches |

## 4. Decision (§6)

**B′ — order can rescue a non-escaper.**
- Arm 1 is split. H18 is supported for seed 5; H17 is not.
- Stage 2 ran, as fixed in the plan, and brackets the moment seed 5's non-escape becomes
  robust to ordering: in **(0, 400]**.

**What this licenses.**
- For seed 5, the initial weights do not by themselves decide the mode: one ordering in eight
  turned the same start into a full graph user.
- After 400 steps of its own training, no ordering tried changes seed 5's outcome.
- In the one crossed pair, the weights decided the outcome under the other seed's order.

**What it does not license.**
- That ordering 2006 *causes* the escape, or what about it does.
- A rate for either seed.
- Anything about other seeds.

**Read together with Step 21.**

| | from step 0 | from 400 | from 805 |
| --- | --- | --- | --- |
| seed 9 (escaper) | 8/8 escape | 8/8 | 8/8 |
| seed 5 (non-escaper) | **1/8 escape** | 0/8 | – (800: 0/8) |

The initial weights bias the outcome strongly in both seeds, but the bias is not absolute for
seed 5. Both seeds' outcomes are sealed early: seed 9's at or before step 0 under the
orderings tried, seed 5's by step 400. The teacher-forcing ramp runs from step 180 to 719,
so the window that matters lies before or near its start.

## 5. An observation the plan did not anticipate

**The ordering that rescues seed 5 is not unusual for seed 9.** Loader seed 2006 escaped from
seed 9's step 0 in Step 21 at step 800, mid-range among seed 9's eight (750–860). From seed
5's step 0 it escapes at 950, and it is the only one of the eight that does.

This is reported as found. No claim rests on it.

## 6. What follows (§12, B′)

**Matched pairs from one state.**
- From seed 5's identical step-0 state, branch 2006 escapes, branch 2007 moves partway, and
  branches 2000–2005 stay stuck.
- This is the first contrast in the programme that is controlled by construction. The
  divergence must lie in steps (0, 400]. That is the first-ever window narrow enough to
  inspect, after Steps 16–19 found no precursor along single trajectories.

**Candidate Step 23**, to be preregistered before any run:
1. Narrow the bracket by branching seed 5 at 100, 200 and 300, with the same eight orderings.
2. Compare the escaping and stuck branches over that window: which batches, which gradients,
   which representation changes.

Nothing in this report starts it.

## 7. Limitations

* Two seeds, one architecture, one synthetic corpus.
* Eight orderings per point. One escape in eight is enough to call a split, and too few to
  estimate a rate.
* The Stage 2 bracket has 400-step precision.
* The swap is one pair in each direction. It agrees with Arm 1, but on its own it shows only
  that this order did not override these weights.
* An intervention on batch order only. Weights, optimiser and objective are not varied.
