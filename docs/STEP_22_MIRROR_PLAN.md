# Step 22 — Is the non-escape fixed too? Seed 5 branched at initialisation, and a crossed swap: preregistration

**Status: PREREGISTERED. No code for this step exists and nothing has run.** Written
2026-10-16, after Step 21's decision A, before any Step 22 code.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why this step

Step 21 found that for **seed 9**, the escape into graph use is fixed from initialisation:
- from steps 0, 200, 400, 600 and 700, all eight data orderings escape (48 of 48);
- ordering changes *when* the escape happens (steps 700–865), never *whether*.

Step 21's §12 named the mirror experiment as the next step: branch a seed that **does not**
escape, at step 0.
- If its non-escape is equally fixed, both of Step 15's modes are set by the starting weights.
- If some orderings rescue it, initialisation decides for some starts but not others.

Step 15 (§"open questions") left a second question that the same machinery answers
directly: **is a seed's outcome its initialisation or its data order?** In Step 15 the seed
controlled both. A crossed swap of the two separates them.

## 2. Facts checked before writing this plan

* **Seed 5 is the deepest non-user** of Step 15's ten seeds:
  - graph usage 0.095° at step 1200, against 0.29° and 0.38° for seeds 4 and 1;
  - validation rotation 23.526°;
  - at the identity floor at steps 400, 800 and 1200 (23.73°, 23.47°, 23.65°).
  `experiments/runs/step15/seed_modes.json`.
* **A frozen 50-step trajectory exists for seed 5**: `experiments/runs/step16/trajectory_seed5.json`,
  25 checkpoints from 0 to 1200, under `STEP16_FROZEN.sha256`. It is the reference the capture
  must reproduce.
* **The data order is the loader's own generator, seeded with the run seed.**
  `WholeOrganLoader` builds `torch.Generator().manual_seed(seed)` with `seed=config.seed`
  (`training/whole_organ.py:104, 158`). A fresh stream from step 0 with loader seed *s* is
  therefore run *s*'s batch order.
* **The swap is batch order only.** The teacher-forcing masks during steps 180–719 draw from
  the *global* torch RNG (Step 21, Amendment 1), which follows the run seed, that is, the
  initialisation. A crossed run swaps which batches arrive in which order. The model's own
  random draws stay with the initialisation's seed. This is stated next to every crossed
  result.
* **Step 20/21's runner is fixed to seed 9** (`SEED = 9` in `experiments/step20/branching.py`).
  Step 22 needs a seed parameter (§8).

## 3. Design

All runs use the frozen configuration of Steps 15–21: A3, global placement target, spatial
hierarchy, chordal rotation objective, λ = 3.0, 1200 steps, CPU, `OMP_NUM_THREADS=4`.

**Arm 1 (primary): seed 5 from step 0.**

* **8 branches** with loader seeds **2000–2007**. These are the same eight orderings that, from
  seed 9's step 0, all escaped in Step 21. So `branch_k` is the same batch order in both steps.
* **1 reference**, restoring seed 5's original stream. It must reproduce frozen seed 5.
* **1 determinism replay**: `branch_00` run twice.

**Arm 2 (secondary): the crossed swap, all from step 0.**

| run | initialisation (weights, global RNG) | batch order | role |
| --- | --- | --- | --- |
| `cross_5w_9o` | seed 5 | seed 9's (loader seed 9) | **swap** |
| `cross_9w_5o` | seed 9 | seed 5's (loader seed 5) | **swap** |
| `ident_5w_5o` | seed 5 | seed 5's (loader seed 5) | identity control: must equal Arm 1's reference |
| `ident_9w_9o` | seed 9 | seed 9's (loader seed 9) | identity control: must equal Step 21's step-0 reference |

The identity controls prove that "loader seed *s* from step 0" reproduces run *s*'s batch
order exactly. Without them a swap could differ for an unnoticed reason.

**Stage 2, fixed now, run only if Arm 1 is not "none".** If any seed-5 branch escapes from
step 0, seed 5 is branched again at **400 and 800**: 8 branches plus a reference at each, with
the same loader seeds. This brackets when its non-escape becomes fixed. If Arm 1 is "none",
Stage 2 does not run.

Each run continues to step 1200, the frozen budget. Nothing is extended.

## 4. Measurement

All definitions are Steps 15, 16, 20 and 21's, unchanged:

* **Escape:** graph usage at or above **1.5°** from some checkpoint to the end of the run,
  without dropping back, within the 1200-step budget.
* **Recorder:** Step 16's `TrajectoryRecorder`, 50-step grid, with Step 20's dense 5-step
  window at 805–900, unchanged for comparability.
* **Non-escape classes:** **stuck** if final usage is below 0.5° (Step 15's non-users sit at
  0.09–0.38°); **in transition** if between 0.5° and 1.5°. Both count as non-escape.
* **Validation only.** `test_splits_read: []` in every artifact.

**Primary outcome:** the number of seed-5 branches out of 8, from step 0, that escape.

* **none**: 0/8 escape, a unanimous non-escape;
* **split**: 1–7 of 8;
* **all**: 8/8.

**Secondary outcomes**, reported and never used to rescue the primary:
1. per branch, final usage and validation rotation, and stuck versus in transition;
2. the crossed swap's two outcomes, escape or not, with their final usage;
3. escape steps of any escaping branch.

## 5. Hypotheses and predictions, recorded before the run

**H17 — both modes are fixed at initialisation.** Seed 5's non-escape survives every ordering.
*Predicts* Arm 1 = none. In the swap, the weights decide: seed 5's weights with seed 9's order
do not escape, and seed 9's weights with seed 5's order do.

**H18 — the non-escape is order-sensitive.** Some orderings rescue seed 5. *Predicts* Arm 1 =
split or all.

**Predictions**, with their basis, written now so they can be wrong:

1. **The reference reproduces frozen seed 5 at all 25 shared checkpoints, and both identity
   controls equal their references exactly.** High confidence. The code path is Step 21's, with
   a seed parameter.
2. **Arm 1 = none, all eight "stuck"** (final usage below 0.5°). Moderate confidence.
   - Basis for: seed 5 is the deepest non-user, and Step 21 found seed 9's outcome fixed
     from step 0.
   - Basis against: in Step 21, ordering spread seed 9's escape over 165 steps (700–865), and Step 15
     showed escapes as late as step 1200.
3. **Swap: `cross_5w_9o` does not escape; `cross_9w_5o` escapes.** Moderate-to-high confidence,
   given prediction 2 and Step 21's eight escapes from seed 9's weights under other orders.
4. **If any seed-5 branch escapes, it escapes late** (after step 900). Low confidence. Basis:
   seed 5 shows no movement toward graph use by step 1200 in its original run.

## 6. Decision rule

* **A′ — both modes fixed from initialisation.** Arm 1 = none. H17 supported. For seeds 5 and
  9, the starting weights decided the mode under every ordering tried.
* **B′ — order can rescue a non-escaper.** Arm 1 = split. H18 supported for seed 5.
  - Stage 2 (400, 800) brackets when the non-escape becomes fixed.
  - The escaping and non-escaping branches from the identical step-0 state become matched
    pairs for a controlled contrast. That contrast is a later step.
* **C′ — the original order was the accident.** Arm 1 = all. Seed 5's non-escape was specific
  to its own batch order, and the "non-using mode" is not an initialisation property for
  seed 5. Reported as found; Stage 2 does not run.
* **Invalid.** The reference fails to reproduce frozen seed 5, or an identity control differs
  from its reference. Stop and report. No branch or swap result is interpreted.

**The swap is secondary.** It is reported under any outcome, and it cannot change the decision.
If it disagrees with Arm 1, both are reported, and the disagreement is the finding.

**What 0/8 can and cannot say.** No escapes in eight is consistent with a true escape rate
anywhere below about 0.31 (one-sided 95%). "None" means a low rate, not zero. The asymmetry
with Step 21's "unanimous" is deliberate and stated next to every verdict.

**Nothing beyond what an intervention licenses.** A unanimous outcome shows survival under the
orderings tried. It does not show that the initial weights cause it.

## 7. Integrity

1. **All freeze manifests** in `experiments/runs/` are verified clean before and after.
2. **The reference reproduces frozen seed 5** (Step 16) at every shared checkpoint, including
   step 1200.
3. **Both identity controls equal their references at every checkpoint.**
   - `ident_9w_9o` against Step 21's `point_0000/reference.json`.
   - `ident_5w_5o` against Arm 1's reference.
4. **One starting fingerprint** across Arm 1. Seed 5's and seed 9's step-0 fingerprints differ.
5. **First-batch fingerprints differ** across the 8 branches.
6. **Determinism replay** of `branch_00` is identical at every checkpoint.
7. **`OMP_NUM_THREADS=4`**, at most three runs at once.

## 8. Code, before any run

* **A seed parameter, without changing Steps 20 and 21.**
  - `experiments/step20/branching.py::_build` and Step 21's capture gain `seed: int = SEED`.
  - With the default, every existing call is unchanged.
  - Step 20's and 21's tests must pass unchanged.
* **`experiments/step22/mirror.py`:**
  - capture of seed 5 at step 0;
  - Arm 1;
  - the four Arm 2 runs, through `continue_point` with an explicit loader seed;
  - the summary, with verdicts for a non-escaper (none / split / all);
  - the identity-control comparison;
  - the Stage 2 gate.
* **`tests/test_step22_mirror.py`:**
  - the seed parameter's default leaves Step 21's capture unchanged;
  - a capture for seed 5 starts from seed 5's initial weights;
  - loader seed *s* from step 0 gives run *s*'s first batches;
  - the verdict mapping is none / split / all;
  - the stage selection follows §3 exactly;
  - an identity mismatch makes the summary invalid.
* The full suite must pass before the capture starts.

## 9. Compute and resources

From Step 21, a 1200-step continuation takes about 45–50 minutes with three running at once.

| part | runs | estimated wall time |
| --- | ---: | --- |
| Arm 1 (8 branches, reference, replay) | 10 | ≈ 3 h |
| Arm 2 (two swaps, two identity controls) | 4 | ≈ 1.2 h |
| Stage 2, only if Arm 1 is not none (two points × 9) | 18 | ≈ 5 h |

Disk: well under 1 GB. No download and no cloud spending. Runs are detached and resumable,
with completed results kept, as in Step 21.

## 10. Stop conditions

* any frozen checksum changes;
* the reference does not reproduce frozen seed 5, or an identity control differs;
* free disk below 2 GiB;
* a run would need a change to the model, objective, optimiser or budget.

## 11. What this step does not do

* **No budget extension.** A branch "in transition" at 1200 stays a non-escape.
* **No interpolation between initialisations.** Locating the boundary between a seed-5 start
  and a seed-9 start is a possible Step 23. Its design depends on this step's result.
* **No other seeds.** Seeds 1, 4 and the intermediate seed 3 are not run.
* **No architecture, objective or optimiser change**, no test split, no new metric.

## 12. What each outcome would lead to

* **A′:** both modes are properties of the starting weights. The programme's question becomes
  *what distinguishes a seed-9 start from a seed-5 start*: initial-weight statistics, early
  representations, or a boundary found by interpolating between them (Step 23).
* **B′:** Stage 2 brackets the fixing point for seed 5. Its matched pairs give the first
  controlled contrast of escaping against non-escaping training from one state.
* **C′:** the non-using mode is not an initialisation property for seed 5. Reconsider whether
  Step 15's modes are attractors of the initialisation at all before designing anything else.
