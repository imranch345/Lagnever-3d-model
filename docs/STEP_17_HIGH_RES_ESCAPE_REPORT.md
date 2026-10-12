# Step 17 — High-resolution escape window: report

**Preregistered in** [`STEP_17_HIGH_RES_ESCAPE_PLAN.md`](STEP_17_HIGH_RES_ESCAPE_PLAN.md).

**Result: C — no upstream precursor detected.** No representation, gradient or objective quantity
changed before the transition at 5-step resolution.

**But Step 16's characterisation of the transition as abrupt does not survive.** At 5-step
spacing the escape resolves into an ~80-step process in which graph usage rises first, geometry
follows about 60 steps later, and the classification threshold is crossed later still.

**`test_splits_read: []`.** One run, diagnostic only. Nothing here is causal.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Frozen configuration verification

All seven prior freeze manifests verified **CLEAN** before and after: corpus, Change 1, Step 12,
Step 13, Step 14, Step 15, Step 16. Git `589b925`, clean tree. Seed 9's configuration asserted
live: 1200 steps, λ=3.0, `graph_recurrence` 1, `frame_scene_context` False, `relation_values`
False. Budget unchanged; no test split read.

Checkpoint grid verified by enumeration **before** launching: 31 dense checkpoints
(750, 755, …, 900) plus 21 coarse, 52 total; and with no window declared the selection is
unchanged from Step 16 at 25.

### 1.1 A reproducibility failure, found and corrected before interpreting anything

The first launch used `OMP_NUM_THREADS=6`. It matched frozen seed 9 exactly at steps 0–150 and
then diverged: step 200 gave 23.58 against the frozen 23.564757.

**The cause was thread count, not instrumentation.** Step 16's runs were launched at 4 threads,
and torch CPU reductions on this machine depend on thread count — a property recorded during the
Step 10 work. The run was killed, its log kept as `seed9_dense_6threads_ABANDONED.log` rather
than deleted, and the experiment relaunched at 4 threads.

This is worth recording because §12 asks for a reproduction check and a naive one would have
passed: the divergence appears only after ~150 steps and is small enough to read as noise.

## 2. Seed 9 reproduction

The 4-thread dense run reproduces frozen seed 9 **bit-identically** at every shared checkpoint:

| step | dense | frozen | identical |
| ---: | ---: | ---: | --- |
| 750 | 23.733745 | 23.733745 | **yes** |
| 800 | 23.420001 | 23.420001 | **yes** |
| 850 | 21.865176 | 21.865176 | **yes** |
| 900 | 21.122634 | 21.122634 | **yes** |
| 1200 | 20.993299 | 20.993299 | **yes** |

Graph usage and the relationally-required subset are identical at the same checkpoints. The
instrumentation is observational.

## 3–6. The 750–900 trajectory

Graph usage, overall validation rotation, and the relationally-required subset. Full series in
`experiments/runs/step17/trajectory_seed9_dense.json`.

| step | usage | rotation | subset | | step | usage | rotation | subset |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 750 | +0.37 | 23.73 | 42.01 | | 830 | +1.23 | 22.60 | 38.95 |
| 755 | **+0.50** | 23.66 | 42.01 | | **835** | **+1.61** | 22.26 | 37.80 |
| 760 | +0.51 | 23.54 | 42.00 | | 840 | +2.00 | 21.78 | 36.20 |
| 765 | +0.48 | 23.59 | 42.05 | | 845 | +2.06 | 21.69 | 35.71 |
| 770 | +0.46 | 23.60 | 41.99 | | 850 | +1.95 | 21.87 | 36.14 |
| 775 | +0.52 | 23.63 | 41.92 | | 855 | +1.90 | 21.89 | 36.26 |
| 780 | +0.62 | 23.62 | 41.79 | | 860 | +2.19 | 21.45 | 34.97 |
| 785 | +0.62 | 23.53 | 41.58 | | 870 | +2.35 | 21.28 | 34.55 |
| 790 | +0.59 | 23.40 | 41.39 | | 880 | +2.46 | 21.17 | 34.33 |
| 795 | +0.60 | 23.35 | 41.30 | | 890 | +2.61 | 21.17 | 34.30 |
| 800 | +0.53 | 23.42 | 41.29 | | 900 | +2.55 | 21.12 | 34.23 |
| 805 | +0.51 | 23.41 | 41.13 | | | | | |
| 810 | +0.64 | 23.18 | 40.72 | | | | | |
| 815 | +0.78 | **23.07** | **40.43** | | | | | |
| 820 | +0.87 | 22.94 | 40.05 | | | | | |
| 825 | +0.97 | 22.81 | 39.60 | | | | | |

**The transition is gradual, not a step change.** Graph usage climbs monotonically from 0.51 at
step 805 to 2.06 at step 845 — eight consecutive checkpoints, no reversal. Rotation falls
monotonically 23.41 → 21.69 over the same span. What Step 16 saw as one 50-step jump is a ramp
occupying roughly 40 steps, preceded by a further ~50 steps of elevated usage at flat geometry.

## 7. Representation trajectory

| step | context written | entity latent std | frame rotation output std |
| ---: | ---: | ---: | ---: |
| 750 | 23.887 | 1.7322 | 0.5223 |
| 780 | 24.129 | 1.7406 | 0.5177 |
| 810 | 24.087 | 1.7371 | 0.5152 |
| 840 | 24.112 | 1.7332 | 0.5115 |
| 870 | 23.997 | 1.7287 | 0.5087 |
| 900 | 24.094 | 1.7340 | 0.5060 |

`context_written_norm` baseline (steps 400–750) is 23.273 ± 0.574, threshold 1.721; its largest
departure anywhere in the window is **0.974**. It does not change. `entity_latent_std` is
similarly flat. `frame_rotation_output_std` drifts slowly downward and is flagged at step 815 —
simultaneous with geometry, and it is a statistic of the head's own output, so it is a
restatement of the prediction rather than something upstream of it.

## 8. Gradient trajectory

| step | graph encoder | relation params | frame head |
| ---: | ---: | ---: | ---: |
| 750 | 0.6359 | 0.00522 | 12.044 |
| 780 | 0.6006 | 0.00473 | 11.508 |
| 810 | 0.7688 | 0.00601 | 12.220 |
| 825 | 0.7568 | 0.00545 | 11.655 |
| 840 | 0.7444 | 0.00542 | 12.581 |
| 870 | 0.6447 | 0.00496 | 12.515 |
| 900 | 0.6462 | 0.00508 | 11.463 |

The graph-encoder gradient baseline is 0.638 ± 0.053, threshold 0.159. Its largest departure in
the window is **0.171** — marginally over 3σ at a single checkpoint, but not sustained across
three, so it is **not detected** by the preregistered criterion. The elevation it does show
(≈0.75 across steps 810–840) **coincided with** the ramp rather than preceding it. No gradient
series is detected as changing.

## 9. Objective trajectory

| step | frame rotation term | total |
| ---: | ---: | ---: |
| 750 | 0.06324 | 2.5554 |
| 795 | 0.06433 | 2.5823 |
| 825 | 0.06468 | 2.5816 |
| 840 | 0.05866 | 2.5458 |
| 870 | 0.05918 | 2.5180 |
| 900 | 0.05877 | 2.5067 |

The rotation term's baseline is 0.0670 ± 0.0044, threshold 0.0133; its largest departure is
**0.0087**. Not detected. It falls from ≈0.064 to ≈0.0587 **between steps 825 and 840 — after
the usage ramp is well under way**. No objective component is detected as changing before the
transition, which is consistent with Steps 15 and 16 finding that training loss does not
distinguish the modes.

## 10. Exact escape interval

**830 < escape ≤ 835**, with graph usage 1.606 at the crossing. **No transient crossings**: usage
never touches 1.5° and falls back.

This sits inside Step 16's `800 < escape ≤ 850`, so the coarse localisation was correct and has
been sharpened tenfold.

## 11. Temporal ordering

| quantity | first detectable change | escape-relative timing |
| --- | ---: | ---: |
| **graph usage** | **step 755** | **80 steps before the crossing** |
| **geometry** (subset 815, rotation 820) | **step 815** | **20 steps before the crossing** |
| representation | step 815 (`frame_rotation_output_std`, a head-output statistic) | 20 steps before |
| gradient | **none detected** | — |
| objective | **none detected** | — |

**Observed order: graph usage → geometry → threshold crossing.** No upstream class precedes
graph usage.

Against the brief's patterns this is **Pattern D** — graph usage and geometry moving with no
measurable precursor. It is explicitly **not Pattern E**: the changes do *not* all occur within
one 5-step interval. Usage leads geometry by about 60 steps.

## 12. Candidate precursor analysis

Four series were flagged as changing before the crossing. Only the last is eligible, and it fails.

| candidate | first change | verdict |
| --- | ---: | --- |
| `graph_usage_deg` | 755 | **non-diagnostic** — it is the quantity being explained |
| `relational_subset_deg` | 815 | **non-diagnostic** — a term of the usage calculation |
| `validation_rotation_deg` | 820 | **non-diagnostic** — a term of the usage calculation |
| `frame_rotation_output_std` | 815 | **non-diagnostic** — coincided with geometry, not before the transition; a statistic of the head's own output |

### 12.1 Two corrections to my own analysis, made on definitional grounds

The first version of the analysis reported `validation_rotation_deg` and `relational_subset_deg`
as **surviving precursors**, and concluded Case 1. That was wrong, and the reason is an identity
rather than a judgement call: **graph usage is `rotation_without_graph − rotation_intact`**, so
validation rotation is one of its two terms and the subset is that same term on 652 entities. A
rotation series crossing its own 3σ detection band before usage crosses the separate 1.5°
classification threshold is *one transition seen through two variables*. Falsification check five
exists to catch exactly this, and its exclusion list was incomplete. Both series are now excluded
by definition, and a test pins the list.

The second: the standing-offset check compared each candidate against *every* step before the
window, which pulled the initial convergence transient into the reference and made real
candidates look like pre-existing offsets. It now uses the declared 400–750 baseline. This
correction works *against* the null — it makes candidates easier to accept, not harder — and the
verdict is unchanged.

Both corrections were made before writing this report and neither was chosen to produce a
particular outcome; the first removed a false positive, the second removed a false negative.

## 13. Falsification attempts

Applied to every candidate, per §14 of the brief.

| check | outcome |
| --- | --- |
| already offset before step 750? | `frame_rotation_output_std` passes; the rotation series pass |
| does it move in the non-escaper (seed 5)? | none of the candidates is detected in seed 5's window |
| does it merely reflect the transition? | **this is what disqualifies all four.** The three rotation/usage series are the transition by construction; `frame_rotation_output_std` moves at the same checkpoint as geometry |
| could measurement noise explain it? | the graph-encoder gradient's single 0.171 excursion, yes — it is sub-threshold under three-checkpoint persistence |
| does it depend on the graph-usage calculation? | **yes for three of four**, which is decisive |

## 14. Comparison with Step 16

| | Step 16 (50-step) | Step 17 (5-step) |
| --- | --- | --- |
| escape interval | 800 < escape ≤ 850 | **830 < escape ≤ 835** |
| character of the transition | "abrupt", one interval | **gradual ramp over ~40 steps, preceded by ~50 steps of elevated usage** |
| first usage change | step 850 (the crossing) | **step 755** |
| first geometry change | step 850 | **step 815** |
| ordering | T1 = T5, simultaneous | **usage precedes geometry by ~60 steps** |
| upstream precursor | none detected | **none detected** |
| verdict | C | **C** |

**Step 16's null on precursors holds. Its description of the transition as abrupt does not.** At
50-step spacing, usage rising from 0.53 to 1.95 and rotation from 23.42 to 21.87 looked like one
event; at 5-step spacing they are separable, and usage moves first.

Step 16 did note a sub-threshold usage ramp at steps 750–800 (0.37, 0.53) and could not resolve
it. That ramp is now resolved: it is real, sustained, and begins at or before step 750.

## 15. Conclusion

**Decision: C — the transition has no detectable upstream precursor, but it is not abrupt.**

Between steps 755 and 810, seed 9's graph usage sat at 0.46–0.64° against a baseline of
0.25 ± 0.07 — a sustained elevation of four to five baseline deviations — while validation
rotation stayed flat at 23.4–23.7° and the relationally-required subset barely moved. From step
810 onward both ramped together to the values they hold for the rest of training.

So there is a phase in which **the model's measured reliance on the relationship graph increased
without any accompanying improvement in the geometry it produces.** Whether that phase is a
build-up toward the transition or the earliest visible part of it cannot be settled by one run,
and no causal reading is offered.

What is not there: no gradient magnitude, and no objective component, departed its pre-window
baseline anywhere in the window. The graph-encoder gradient's elevation coincided with the ramp
rather than preceding it, and the objective's rotation term fell only after the ramp had begun.

## 16. Limitations

* **The window may not contain the onset.** Usage at step 750 is 0.37, the highest value in its
  own 400–750 baseline, and the coarse value at step 700 was 0.24. The rise plausibly begins at
  or before 750, so **"usage leads geometry by ~60 steps" is a lower bound** and the onset is
  unresolved. This is the single most important limitation and it is a direct consequence of
  choosing the window from Step 16's coarse interval.
* **One run.** Nothing here is a population claim. Seed 9 is one escaper of six.
* **The diagnostics are those already built.** Gradient measurements are aggregate L2 norms per
  module; direction, per-layer structure, curvature and optimiser state were not measured, and a
  precursor in any of them would not appear.
* **Sub-threshold movement exists.** The graph-encoder gradient reaches 0.171 above baseline
  against a 0.159 threshold at one checkpoint. Under a weaker persistence rule it would be
  flagged. The rule was fixed in advance and is not being relaxed now, but the movement is real
  and small.
* Graph usage is one aggregate — the cost of removing all relationships.
* One corpus, one scale, 1200 steps, synthetic throughout.

## 17. Recommended next step

**Extend the dense window backwards, to step 600–800.** This is the only change that addresses the
binding limitation: it would locate the onset of the usage rise, which the current window cannot
see because it starts after the rise has begun. It is one run at the frozen configuration, the
instrumentation already exists, and the window is justified by this report rather than by any
later result.

Two secondary points, both cheap and neither an intervention:

1. **The 755–810 phase is the interesting object now**, not the crossing. If usage can be elevated
   for 50+ steps without geometric benefit, then what changes at ~810 to make it pay off is a
   sharper question than "what triggers escape". A backwards-extended window would also say
   whether this phase is longer than it currently appears.
2. **Seed 3 is the natural replication.** Step 16 showed it mid-transition at step 1200 with usage
   still climbing, so a dense window over its final 200 steps would test whether the
   usage-leads-geometry ordering recurs. That is a second run, and it would be the first evidence
   that the ordering is more than one seed's history.

**Not started.** No intervention, no architecture change, no new loss, no budget change.

## 18. Answer to the research question

> **Between steps 750 and 900, does anything measurable change before seed 9 escapes the
> identity-only attractor, or does the relational transition remain effectively abrupt even at
> 5-step resolution?**

**The transition is not abrupt at 5-step resolution, but nothing upstream of it was detected
changing first.** Graph usage departed its baseline at step 755 and remained elevated for about
55 steps while geometry stayed flat; geometry began improving at step 815; the classification
threshold was crossed between steps 830 and 835. No representation, gradient or objective
quantity departed its pre-window baseline at any point in the window.
