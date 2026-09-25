# Step 20 — Checkpoint branching / basin sensitivity: report

**Decision: A — state-dominated.** All eight branches from the identical step-805 state escaped
into the relational attractor. **escape_rate = 8/8 = 1.000.** Changing the subsequent data
ordering moved the escape step by at most 10 steps and changed nothing about the outcome.

**H13 supported. H14 weakened.**

**`test_splits_read: []`.** Nothing causal is claimed beyond what an intervention licenses.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Resource preflight (§2)

| item | value |
| --- | --- |
| free disk at launch | **10 GiB** (the 3.6 GiB in the Step 19 report was transient purgeable space, not a leak) |
| Step 20 artifacts | 37 MB full state + 10 trajectory JSONs ≈ 250 MB |
| runner guard | 2.0 GiB, never approached |
| power | AC / ≥70% battery throughout |
| artifacts deleted or archived | **none** |

All nine prior freeze manifests verified CLEAN before and after.

## 2. Frozen checkpoint verification (§3) — and a blocker that had to be built around

**§3 could not be satisfied by any existing artifact.** There was no step-805 checkpoint, and no
checkpoint in this project could have served one: every saved checkpoint holds
`{config, model, parameters}` and carries **no optimiser state, no scheduler state, no RNG state
and no step counter**. A continuation from one would have been a different experiment under the
same name, which §3 explicitly forbids.

So Step 20 began by adding full-state checkpointing and capturing step 805 by re-running seed 9.
What exact continuation actually needs here was established by audit:

* model parameters;
* AdamW `exp_avg`, `exp_avg_sq`, `step` per parameter — 139 entries;
* the step counter (the learning rate is a pure function of the step, so there is no separate
  scheduler state);
* the data stream's position (§4);
* global RNG is **not** required — `train_step` was measured not to consume it. Saved anyway, so
  a future change that starts consuming it cannot silently invalidate a branch.

**The capture reproduces frozen seed 9 at 18/18 shared checkpoints**, including step 805 itself:
usage **0.5122**, rotation **23.4104**, identical to the Step 18 dense trajectory.

## 3. Branch initialisation fingerprints (§11)

| quantity | value |
| --- | --- |
| distinct starting fingerprints across all 10 continuations | **1** |
| step | 805 |
| model tensors | 3,226,647 parameters, sum **2872.0822747919** |
| optimiser | 139 entries, moment sum **111893.9867783195** |
| stream position | epoch 5, 60 batches consumed |

**Every branch started from bit-identical weights and optimiser state.**

## 4. RNG and data-order isolation audit (§6, §7)

Measured, not assumed:

* **`WholeOrganLoader` draws scene order and batch order from its own `torch.Generator`**, not the
  global RNG, so a branch seed cannot reach the weights.
* **`build()` consumes no global RNG** and returns the identical batch under a different global
  seed — batch *contents* are a deterministic function of the scene group.
* **`train_step` consumes no global RNG.**

The intervention was therefore narrowed to reseeding that one generator. **No object is
replaced** — the trainer's own loader is kept — so there is nothing else that could differ between
branches by construction.

**The one subtlety worth recording.** `epoch()` draws two permutations up front to build a whole
epoch's batch list, so a generator state saved mid-epoch has already spent the draws that produced
the current epoch. Restoring it would rebuild a *different* epoch and the reference continuation
would silently not be a continuation. `StreamState` records the generator state as it was **at the
epoch boundary**, plus the epoch index and batches consumed, and two tests check that resumption
is exact both mid-epoch and across an epoch boundary.

## 5. Data-order verification (§16)

First batch after the branch point, by scene position in the training list:

| branch | first batch |
| --- | --- |
| branch_00 | 480, 524, 1175, 948, 816, 184, 660, 1149 |
| branch_01 | 1172, 649, 657, 260, 591, 834, 737, 1074 |
| branch_02 | 920, 1070, 1064, 193, 405, 1032, 107, 532 |
| branch_03 | 203, 95, 434, 889, 755, 212, 637, 116 |
| branch_04 | 1153, 57, 54, 162, 569, 133, 1176, 650 |
| branch_05 | 141, 254, 1121, 379, 238, 918, 524, 894 |
| branch_06 | 740, 1024, 643, 1129, 1158, 687, 41, 562 |
| branch_07 | 433, 921, 244, 1003, 699, 1103, 795, 918 |
| reference | 510, 27, 928, 170, 316, 42, 577, 154 |

**8 of 8 branches received distinct orderings.** No branch merely carried a different nominal seed
while processing identical batches.

## 6. Determinism replay control (§17)

`branch_00` was run twice with the same seed. The two runs are **identical at all 26
checkpoints** — graph usage and rotation alike — and the replay's first batch matches.

## 7. The reference control

Not required by the brief; added because without it eight non-escaping branches would have been
indistinguishable from broken machinery.

Continuing from step 805 with the **original stream position restored exactly** reproduces frozen
seed 9 at **26/26 shared checkpoints from 805 onward**, including the escape at
**830 < escape ≤ 835**. The branching machinery reproduces the phenomenon it is branching.

## 8. Branch escape table (§13)

| branch | seed | escaped | escape interval | final usage | final rotation | relational subset |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| branch_00 | 2000 | **yes** | 825 < escape ≤ 830 | 2.55 | 20.99 | 33.98 |
| branch_01 | 2001 | **yes** | 835 < escape ≤ 840 | 2.56 | 20.95 | 33.93 |
| branch_02 | 2002 | **yes** | 825 < escape ≤ 830 | 2.58 | 20.96 | 33.96 |
| branch_03 | 2003 | **yes** | 825 < escape ≤ 830 | 2.53 | 21.02 | 34.03 |
| branch_04 | 2004 | **yes** | 830 < escape ≤ 835 | 2.55 | 20.99 | 33.98 |
| branch_05 | 2005 | **yes** | 825 < escape ≤ 830 | 2.55 | 21.01 | 33.99 |
| branch_06 | 2006 | **yes** | 830 < escape ≤ 835 | 2.55 | 21.00 | 33.99 |
| branch_07 | 2007 | **yes** | 835 < escape ≤ 840 | 2.56 | 20.97 | 33.98 |
| *reference* | — | *yes* | *830 < escape ≤ 835* | *2.52* | *20.99* | *33.97* |
| *branch_00 replay* | 2000 | *yes* | *825 < escape ≤ 830* | *2.55* | *20.99* | *33.98* |

No transient crossings in any branch.

## 9. Escape-time and outcome distributions (§15)

**escape_rate = 8/8 = 1.000**

Escape steps: **830, 830, 830, 830, 835, 835, 840, 840** — a 10-step spread, all within one dense
checkpoint either side of the reference's 835.

| quantity | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| final graph usage | **2.553** | **0.015** | 2.527 | 2.579 |
| final validation rotation | **20.988** | **0.023** | 20.954 | 21.020 |

The brief cautions against a binary rate alone, and the distribution is the stronger evidence:
**every branch arrives at the same place, and arrives within 10 steps of the same time.** The
final-rotation spread across branches (0.023) is as tight as the spread across the six independent
graph-using *seeds* in Step 15 (0.021) — the same attractor, reached from the same state by eight
different routes.

## 10. H13 and H14 (§12, §13, §22)

**H13 — state-determined basin. Supported.** Its predictions were that branches from the identical
step-805 state mostly converge to the same mode and that data ordering has little effect on whether
escape occurs. Both hold: 8/8, with escape timing varying by ≤10 steps.

**H14 — path-dependent stochastic transition. Weakened.** Its predictions were that branches
diverge in whether or when they escape, and that outcome correlates with ordering. Eight distinct
orderings produced eight escapes with essentially identical endpoints. There is no divergence in
*whether*, and the variation in *when* is small and shows no relation to ordering that eight
samples could distinguish from timing noise.

**Falsification, as §22 requires.** H13 would have been weakened by substantially different
outcomes under different ordering; none occurred. H14 would have been weakened by convergence
despite different ordering; that is what happened. No threshold was chosen to manufacture the
result — the escape criterion is Step 15's unmodified 1.5°, and the outcome was unanimous, so no
threshold choice could have changed it.

## 11. Interpretation, within the §24 restriction

**What the experiment licenses:** from the identical seed-9 state at step 805, the relational
escape occurred under every subsequent data ordering tested. The outcome is **not** sensitive to
stochastic continuation under this protocol.

**What it does not license:** this does not show the state at 805 *causally determined* the
outcome, only that the outcome survived every perturbation this intervention could apply. Nor does
it show that data ordering is irrelevant in general — only that from *this* state, 395 steps from
the end, it does not change the basin.

Read together with Steps 16–19, the picture is consistent: those steps found no precursor because,
by step 805, there was nothing left to precede. The transition was already inevitable, and the
visible movement at 810–820 is its expression rather than its cause.

## 12. Limitations

* **One branch point.** Step 805 is 25 steps before the transition. That the outcome is fixed there
  says nothing about where it became fixed, which could be step 700, step 400, or initialisation.
* **Eight branches.** A rate of 8/8 is consistent with a true rate anywhere above roughly 0.7; a
  rare non-escaping ordering would not be detected.
* **One seed.** Seed 9 is one escaper of six. Nothing here bears on the three seeds that never
  escape — their step-805 states were not branched, and the interesting question of whether *their*
  outcome is equally fixed is untouched.
* **Data ordering is the only stochastic factor varied**, and the audit shows it is the only one
  available in this training path. That is a property of this implementation, not a general claim.
* **The capture is a re-run.** It reproduces frozen seed 9 exactly at all 18 shared checkpoints, but
  it is a fresh execution rather than the original process.
* One corpus, one scale, 1200 steps, synthetic throughout.

## 13. Answer to the required final question (§26)

> **Starting from the identical Seed 9 state at step 805, does changing only the subsequent
> stochastic continuation materially change whether the model escapes into the relationally useful
> attractor?**

**No.** Eight branches with eight verifiably distinct data orderings, from bit-identical weights and
optimiser state, all escaped — within a 10-step window and to within 0.023° of the same final
rotation.

## 14. Where the next investigation should focus

The brief asks this result to decide between the state/basin at the checkpoint and path-dependent
optimisation dynamics. **It decides for the state.** Path-dependence is not the mechanism, at least
not this late.

The question that follows is therefore **when** the basin is entered, and it is now directly
answerable with the machinery this step built. Full-state capture and exact branching exist; the
natural next experiment is to **repeat this design at earlier branch points** — for instance 700,
600, 400 — and find the earliest step from which the escape is still unanimous. That locates basin
entry by bisection rather than by searching trajectories for a signal, which is what Steps 16–19
exhausted.

The second question, equally cheap and arguably more informative, is the mirror image: **branch a
non-escaping seed.** If seed 5's step-805 state also produces 8/8 of its own outcome, the two modes
are both locked in early and the programme's question becomes what distinguishes them at
initialisation. If seed 5's branches split, then non-escape is the fragile state and escape the
stable one, which would be a genuinely new asymmetry.

**Neither is started here**, and no architecture change is proposed, as §26 requires.
