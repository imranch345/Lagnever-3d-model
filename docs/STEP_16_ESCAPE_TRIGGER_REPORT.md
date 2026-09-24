# Step 16 — Escape trigger diagnostic: report

**Preregistered in** [`STEP_16_ESCAPE_TRIGGER_PLAN.md`](STEP_16_ESCAPE_TRIGGER_PLAN.md), written
before any instrumented run was launched.

**Result: C — no detectable precursor.** Nothing in the representation, gradient or objective
diagnostics changes before escape. Rotation, graph usage and the relationally-required subset all
move together in a single 50-step interval.

**`test_splits_read: []`.** Diagnostic only; no intervention.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Configuration verification

All six prior freeze manifests verified **CLEAN** before and after: corpus (7 files), Change 1
(101), Step 12 (8), Step 13 (13), Step 14 (5), Step 15 (10). Git `1987b6c`, clean tree.

Frozen A3 configuration, read live: 1200 steps, batch 8, lr 3e-4, warmup 60, clip 1.0, decay
0.01, chordal objective, λ=3.0, `frame_weight` 2.0, `lod_weight` 0.5, `graph_recurrence` 1,
`frame_scene_context` False, `relation_values` False. Asserted per run before training.

**The instrumentation does not change training, verified twice.** A 60-step instrumented run
reproduces its uninstrumented twin with **zero differing tensors**. More convincingly, all three
full 1200-step instrumented runs reproduce their frozen Step 15 results exactly:

| seed | Step 15 usage / rotation | Step 16 usage / rotation |
| ---: | --- | --- |
| 9 | 2.52° / 20.99° | **2.52° / 20.99°** |
| 5 | 0.09° / 23.53° | **0.09° / 23.53°** |
| 3 | 1.30° / 22.17° | **1.30° / 22.17°** |

### 1.1 A discrepancy in the brief, documented before training

§10 requests checkpoints at steps 1400–2600. **The frozen budget is 1200 steps**, and §3 forbids
changing training duration while §11 forbids intervention. Per §10's own fallback, the trajectory
is reported to step 1200 and steps 1400–2600 are recorded as **outside the frozen budget and not
measured**. Extending the budget is a separate experiment.

## 2. Seeds and rationale

Fixed from the frozen Step 15 report before any trajectory was inspected. Seed 9 (late escaper)
was named by the brief. Seed 5 was chosen as non-escaper for having the lowest graph usage of all
ten seeds (0.09°) and a trajectory that never leaves the floor; seed 1 was rejected for the
highest usage of the three non-users (0.38°) and a slightly declining trajectory. Seed 3 is the
single intermediate.

## 3. Instrumentation

A callback on the frozen loop at every 50 steps, including step 0 (the callback runs *before* the
step it is given, so step 0 observes the initialisation) and once after the final step —
**25 checkpoints per run**. `fit()` gained one optional `on_step` parameter defaulting to `None`.

Python, NumPy and Torch RNG are saved and restored around every measurement (building a batch
samples points); gradients are zeroed (the pathway measurement calls `backward`); training mode is
restored. Cost: 5.2 s per checkpoint.

Graph usage is **Step 15's definition unchanged**. The relationally-required subset is **Step 14's
preregistered selection**, rebuilt from train-only tables and verified to contain exactly **652
entities**. An earlier implementation keyed it on `(relation signature, slot)` and reached 991,
because one qualifying entity enrolled every entity in every other validation scene sharing its
signature; that would have been a broader subset wearing a preregistered name. A second bug in the
same family — keying per-entity errors on the signature — would have silently dropped colliding
entities from every average. Both were found and fixed before the runs.

## 4–6. Raw trajectory: graph usage and rotation

Validation, all three seeds. Full tables are in
`experiments/runs/step16/trajectory_seed{9,5,3}.json`.

| step | seed 9 rot / usage | seed 5 rot / usage | seed 3 rot / usage |
| ---: | --- | --- | --- |
| 0 | 61.74 / +0.00 | 61.74 / +0.00 | 61.74 / +0.00 |
| 100 | 24.65 / +1.13 | 24.95 / +1.31 | — |
| 200 | 23.56 / +0.32 | 23.79 / +0.82 | — |
| 400 | 23.62 / +0.14 | 23.58 / +0.30 | — |
| 600 | 23.86 / +0.29 | 23.46 / +0.15 | — / +0.10 |
| 700 | 23.31 / +0.24 | 23.52 / +0.16 | — / −0.07 |
| 750 | 23.73 / +0.37 | 23.47 / +0.03 | — / +0.01 |
| 800 | 23.42 / +0.53 | 23.25 / +0.19 | 23.59 / +0.06 |
| **850** | **21.87 / +1.95** | 23.50 / +0.11 | 23.39 / +0.04 |
| 900 | 21.12 / +2.55 | 23.70 / −0.00 | 23.52 / +0.02 |
| 1000 | 21.01 / +2.58 | 23.62 / +0.07 | 23.24 / +0.23 |
| 1100 | 20.99 / +2.52 | 23.54 / +0.11 | 22.61 / +0.90 |
| 1200 | 20.99 / +2.52 | 23.53 / +0.09 | 22.17 / +1.30 |

Every run begins with the same rapid descent (61.7 → ~24 by step 100; rotation at step 0 is
identical across seeds because the frame head's output projection is zero-initialised). All three
then sit on the identity-only floor — 23.51° — for hundreds of steps.

**Seed 9's escape is abrupt.** In one 50-step interval: rotation 23.42 → 21.87, usage 0.53 → 1.95,
subset 41.29 → 36.14. It then settles within 0.1° and does not move again.

## 7. Relationally-required subset

The 652 entities where identity does badly and the graph demonstrably helps.

| step | seed 9 | seed 5 | seed 3 |
| ---: | ---: | ---: | ---: |
| 400 | 41.77 | 42.14 | — |
| 800 | 41.29 | 41.27 | — |
| **850** | **36.14** | 41.84 | — |
| 1200 | 33.97 | 41.85 | — |

The subset moves at exactly the same checkpoint as the aggregate, and by more (−5.15° against
−1.55°). It gives no earlier warning.

## 8. Representation diagnostics

| step | context written (9 / 5 / 3) | entity latent std (9 / 5 / 3) |
| ---: | --- | --- |
| 0 | 7.86 / 7.45 / 7.99 | — |
| 200 | 20.56 / 20.59 / 21.21 | — |
| 400 | 22.26 / 21.41 / 21.92 | — |
| 600 | 23.46 / 21.91 / 23.37 | 1.710 / 1.635 / — |
| 800 | 24.11 / 22.82 / 24.08 | 1.740 / 1.680 / — |
| 850 | 23.97 / 22.86 / 24.20 | 1.726 / 1.685 / — |
| 1200 | 24.20 / 23.17 / 24.32 | — |

How much the graph encoder writes into the context slice rises steadily in **all three** runs and
**does not change at escape** — seed 9's value actually dips slightly, 24.11 → 23.97. The two
eventual escapers do sit above the non-escaper from roughly step 250 onward, which looked like a
precursor until it was controlled (§11).

`relation_bias_abs_max` was the one series the preregistered criterion flagged early (step 450),
and it is spurious: it rises monotonically in every run, and the **non-escaper finishes higher
than the escaper** (0.0644 against 0.0606). It is a weight growing under the loss, not an escape
signal.

## 9. Gradient diagnostics

| step | graph encoder (9 / 5) | frame head (9 / 5) | relation params (9 / 5) |
| ---: | --- | --- | --- |
| 600 | 0.710 / 0.590 | 13.70 / 12.43 | 0.0045 / 0.0046 |
| 750 | 0.636 / 0.652 | 12.04 / 14.35 | 0.0052 / 0.0037 |
| 800 | 0.659 / 0.620 | 12.16 / 11.97 | 0.0051 / 0.0045 |
| **850** | **0.775** / 0.608 | **14.25** / 13.34 | 0.0048 / 0.0039 |
| 900 | 0.646 / 0.587 | 11.46 / 12.24 | 0.0051 / 0.0038 |

Gradient magnitudes are noisy and overlapping between the two runs throughout. Seed 9's graph and
frame-head gradients are highest at step 850 — **at the escape, not before it** — and return to
baseline immediately after. Nothing in the gradient series anticipates the transition.

## 10. Objective diagnostics

Every component — total, `frame_chordal`, `frame_rotation`, `frame_translation`, `frame_scale`,
`geometry_reconstruction`, `semantic_part_correspondence`, `nested_lod`, `entity_frame`,
`entity_identity`, `language_anatomy_alignment` — fluctuates within a narrow band in both runs and
shows **no systematic pre-escape movement**. The rotation term falls from 0.0630 to 0.0586 between
steps 800 and 850, which is the escape registering in the loss, not preceding it.

This is consistent with Step 15's free finding that training loss does not distinguish the modes,
now confirmed at 50-step resolution on the component level.

## 11. Escaper versus non-escaper

Three series separate the two escapers from the non-escaper before the first escape, by the
preregistered criterion (between-group gap exceeding within-escaper spread, sustained):

| series | sustained from |
| --- | ---: |
| `entity_latent_std` | step 100 |
| `context_written_norm` | step 250 |
| `context_written_share` | step 400 |

**A control removes them.** With two escapers and one non-escaper, "both escapers on the same
side" occurs by chance two thirds of the time, so every seed was tried as the odd one out:

| odd one out | series separating |
| --- | ---: |
| seed 5 — **the real grouping** | **3** |
| seed 9 — control | **4** |
| seed 3 — control | 1 |

**An arbitrary regrouping separates more series than the real one.** The separation test has no
discriminating power at this sample size, and its three hits cannot support a precursor claim.
(The control grouping's four hits all appear at steps 750–800, which is seed 9 diverging from the
other two *as it escapes* — real, but the escape itself.)

Note also that these three series are persistent **offsets**, not changes: they separate the
groups hundreds of steps before escape and do not move at escape. Even taken at face value they
would describe a predisposition, not a trigger.

## 12. The intermediate seed

**Seed 3 is a slower escaper caught mid-transition, not a third mode.** Its usage is flat near
zero until step ~950, then climbs 0.23 → 0.90 → 1.30 over the final 250 steps and is **still
rising when training stops**; rotation falls 23.24 → 22.61 → 22.17 in the same window. It had not
crossed the 1.5° threshold by step 1200, so it is formally a non-escaper, but its trajectory is
the escaper's shape arriving later.

This confirms the speculation recorded in Step 15 §7 and resolves the ambiguity that report left
open. It also means the base rate of 6/10 graph-using at 1200 steps is a *lower bound*: at least
one further seed was on its way.

## 13. Escape intervals

| seed | interval |
| ---: | --- |
| 9 | **800 < escape ≤ 850** (usage 1.95 at crossing) |
| 5 | never crossed |
| 3 | never crossed by 1200; mid-transition, usage 1.30 and rising |

No interpolation is offered: at 50-step spacing the data does not support a finer estimate.

## 14. Ordering of observed changes

**T1 first graph-usage change:** step 850 (preregistered criterion). A sub-threshold ramp is
visible at 750–800 (0.37, 0.53 against a plateau mean of 0.31 ± 0.10) but does not reach 3σ.

**T2 first representation change:** none detected. No representation series departs its own
plateau at any point, including at escape.

**T3 first gradient change:** none detected before escape. Magnitudes peak at step 850.

**T4 first objective change:** none detected. The rotation term moves at step 850.

**T5 first geometric improvement:** step 850.

**T6 observed ordering:** **T1 = T5 = step 850, with T2, T3 and T4 absent.** Graph usage, the
relationally-required subset and validation rotation move in the same 50-step interval; no
measured representation, gradient or objective quantity moves earlier, or indeed at all by its own
criterion.

### 14.1 The preregistered criterion was poorly suited, and this is reported rather than patched

The plan fixed the baseline window at steps 0–400. That window contains the initial convergence
transient, where rotation falls from 61.7° to 23.6°, giving a baseline standard deviation of about
12° — so three of those exceeds every later movement and the criterion **cannot fire for the
series whose change matters most**. Under it, only `graph_usage_deg` (step 850) and the spurious
`relation_bias_abs_max` (step 450) are flagged.

A second criterion using the **plateau** (steps 150–800) as baseline is therefore computed and
**labelled post hoc everywhere**. It flags rotation, usage and subset at step 850 and nothing
else, which is the §14 result above. The preregistered criterion was not quietly replaced, and the
post-hoc one is not used to claim a mechanism — it only sharpens a null.

## 15. H9–H12

| hypothesis | verdict |
| --- | --- |
| **H9 — representation trigger** | **not supported.** No representation series departs its plateau before or at escape; the early escaper/non-escaper separations fail their own grouping control. |
| **H10 — gradient trigger** | **not supported.** Gradients peak at the escape checkpoint, not before, and overlap between runs throughout. |
| **H11 — objective trigger** | **not supported.** No component moves systematically before escape. |
| **H12 — no detectable precursor** | **supported.** |

## 16. Limitations

* **Three runs.** The grouping control shows this is enough to disprove a separation claim but
  nowhere near enough to establish one. A precursor present in most escapers could easily be
  invisible here.
* **50-step resolution.** A precursor confined to the ~50 steps before the crossing would be
  missed entirely. The escape occupies one interval, so the instrumentation is exactly at the
  edge of its own resolution — this is the single most likely way H12 is wrong.
* **The diagnostics are the ones already built.** Step 14's pathway norms are aggregate L2 norms
  per module; gradient *direction*, per-layer structure, curvature and optimiser state were not
  measured. A precursor in any of those would not appear.
* **Graph usage is one aggregate.** It is the cost of removing all relationships, which Step 13
  showed is the only discriminating version, but it is a single number.
* **Nothing here is causal.** Every statement is about what co-occurs with escape in three runs.
* One corpus, one scale, 1200 steps, synthetic throughout.

## 17. Conclusion

**Decision: C — no detectable precursor.**

Every A3 run begins at the identity-only floor. Some leave it, and when one does, the departure is
abrupt: within a single 50-step interval, graph usage, the relationally-required subset error and
overall validation rotation all move together, and the run then settles into a 0.02°-wide
attractor. Before that interval, nothing measured here is moving — not the graph encoder's output
statistics, not the gradient reaching any pathway, not any component of the objective.

The three quantities that do distinguish eventual escapers from the non-escaper are persistent
offsets rather than changes, and a grouping control shows an arbitrary regrouping of the same
three runs separates more series than the real one. They are reported as failing, not as weak
support.

**This is a null result about precursors, not about the transition.** The transition is real,
sharp, reproducible and now localised to a 50-step interval. What Step 16 establishes is that it
is not preceded by a gradual build-up in any quantity this instrumentation can see.

## 18. Answer to the research question

> **What, if anything, changes first when an A3 training run transitions from the identity-only
> attractor to the relationally useful attractor?**

**Nothing measured changes first.** Within the resolution of this experiment, the model's use of
the relationship graph, its error on the entities that need the graph, and its overall rotation
accuracy all change in the same 50-step interval, with no antecedent movement in representation
statistics, gradient magnitudes, or any objective component.

## 19. Recommended next step

The one limitation that could plausibly overturn the null is resolution: the escape occupies a
single interval, so a precursor lasting a few dozen steps would be invisible. That is a cheap,
well-posed question and it does not require any change to the model, objective or budget.

**Re-run seed 9 alone with 5-step instrumentation over steps 750–900.** It is one run, the
configuration is untouched, the escape window is already known, and it would either reveal a
build-up inside the interval or close the resolution objection. Because the escape step is known
in advance, the window is preregisterable without circularity.

Two further things are worth stating about what *not* to do next.

**Do not add graph supervision on the strength of Step 16.** A null on precursors is not evidence
that a precursor could be manufactured usefully, and Step 14 already showed the objective pays
82% for graph conditioning without being collected.

**The base rate needs revising upward, not re-measuring.** Seed 3 was mid-transition at step 1200,
so 6/10 understates the eventual graph-using fraction. Whether the remaining non-escapers are slow
or genuinely stuck is a budget question — it requires training past 1200 steps, which §3 forbids
here and which Step 15 §8 already flagged as a distinct experiment.

**Not started.** No intervention, no architecture change, no new loss, no budget change, no
Step 17.
