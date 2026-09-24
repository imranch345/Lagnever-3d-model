# Step 18 — Usage-rise onset and replication: report

**Preregistered in** [`STEP_18_ONSET_AND_REPLICATION_PLAN.md`](STEP_18_ONSET_AND_REPLICATION_PLAN.md).

**Two results, one of which corrects Step 17.**

1. **Step 17's headline ordering claim does not survive.** Its "graph usage leads geometry by ~60
   steps" was an artifact of a baseline computed from 8 sparse coarse points, which understated
   the true variability by nearly half. With a dense pre-window baseline, everything
   transition-related in seed 9 moves inside a **10-step span** (810–820).
2. **The precursor null replicates.** In both seeds, no representation, gradient or objective
   quantity departs consistently before the transition. **Case 3.**

**`test_splits_read: []`.** Diagnostic only; nothing causal.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Configuration and reproduction

All eight prior freeze manifests **CLEAN**. Both runs at the frozen configuration, 4 threads
(now part of the frozen protocol, after Step 17 found a 6-thread run diverged).

| check | result |
| --- | --- |
| seed 9 reproduces Step 16 at steps 800 / 900 / 1200 | **yes** |
| seed 3 reproduces Step 16 at steps 800 / 900 / 1200 | **yes** |
| seed 9 final: usage 2.5183, rotation 20.9933 | matches frozen |
| seed 3 final: usage 1.3046, rotation 22.1679 | matches frozen |
| **Step 17 vs Step 18 across all 31 shared dense checkpoints (750–900)** | **bit-for-bit identical** |

The last row is the internal consistency check run A was extended to 900 to obtain. Two
independent runs of the same seed, instrumented over different windows, agree exactly.

## 2. The correction to Step 17

Step 17 reported that seed 9's graph usage departed its baseline at step **755** and geometry only
at **815**, concluding a ~60-step ordered lead and "a phase in which reliance on the graph
increased without accompanying improvement in geometry".

**Step 18's dense pre-window data shows that phase is ordinary noise.** Usage across 600–750,
measured every 5 steps:

```
600:+0.29 605:+0.15 610:+0.04 615:+0.12 620:+0.27 625:+0.46 630:+0.47 635:+0.33
640:+0.29 645:+0.26 650:+0.20 655:+0.27 660:+0.45 665:+0.52 670:+0.36 675:+0.36
680:+0.38 685:+0.30 690:+0.49 695:+0.44 700:+0.24 705:+0.10 710:+0.22 715:+0.34
720:+0.36 725:+0.39 730:+0.36 735:+0.25 740:+0.25 745:+0.33 750:+0.37
```

Usage oscillates between **0.04 and 0.52** for the entire pre-transition period. The values Step
17 flagged as a sustained departure — 0.50 at 755, 0.51 at 760, 0.48 at 765 — sit inside that
range. Step 625 (0.46), 630 (0.47), 665 (0.52) and 690 (0.49) are equally high, 90–130 steps
earlier, and nothing happened.

**Why Step 17 got it wrong:** its baseline was the 8 *coarse* checkpoints from 400 to 750, whose
standard deviation is **0.0714**. The 31 *dense* checkpoints from 600 to 750 give **0.1159** —
62% larger. The coarse grid happened to sample the quieter part of the oscillation, so the 3σ
band was too narrow and ordinary fluctuation crossed it.

This is the same class of error Step 16 documented in *its* preregistered criterion (a baseline
window containing a transient) and that Step 17 believed it had avoided. It had not: it avoided
the transient and instead under-sampled the variance.

## 3. Seed 9 with a dense baseline (600–750, 31 points)

| series | baseline | 3σ | first sustained departure |
| --- | --- | ---: | ---: |
| `relational_subset_deg` | 41.825 ± 0.248 | 0.744 | **810** |
| `graph_usage_deg` | 0.311 ± 0.116 | 0.348 | **815** |
| `validation_rotation_deg` | 23.551 ± 0.173 | 0.520 | **820** |
| `obj_frame_rotation` | 0.0678 ± 0.0029 | 0.0086 | 835 |
| `frame_rotation_output_std` | 0.5233 ± 0.0043 | 0.0128 | 860 |
| `grad_graph_encoder` | 0.681 ± 0.075 | 0.225 | **none** |
| `context_written_norm` | 23.803 ± 0.172 | 0.516 | **none** |
| `entity_latent_std` | 1.7291 ± 0.0087 | 0.0262 | **none** |

**Everything transition-related falls in 810–820 — three dense checkpoints.** The objective's
rotation term follows at 835, the head's output spread at 860. No gradient and no representation
statistic departs at all.

The escape interval is unchanged from Step 17: **830 < escape ≤ 835**, usage 1.6062 at crossing.

## 4. Seed 3 replication (baseline 900–1000, window 1000–1200)

Seed 3 never crosses 1.5° inside the budget, so this measures ordering, not escape.

| series | first sustained departure |
| --- | ---: |
| `obj_frame_rotation` | **1015** |
| `relational_subset_deg` | **1025** |
| `graph_usage_deg` | **1035** |
| `validation_rotation_deg` | **1055** |
| `entity_latent_std` | 1075 |
| `grad_graph_encoder` | 1100 |
| `context_written_norm` | none |

Its transition is far slower than seed 9's — usage 0.02 → 1.30 over 300 steps against seed 9's
0.3 → 2.0 over 30 — so the departures are correspondingly smeared.

**The within-transition ordering replicates.** In both seeds the relationally-required subset
departs before graph usage, which departs before overall rotation:

| | subset | usage | rotation |
| --- | ---: | ---: | ---: |
| seed 9 | 810 | 815 | 820 |
| seed 3 | 1025 | 1035 | 1055 |

That ordering is consistent, and it is the ordering one would expect mechanically: the subset is
the 652 entities where the graph demonstrably helps, so a gain there is visible before it dilutes
into the 2,316-entity average. It is not evidence of a precursor — all three are the transition.

## 5. Candidate precursors, and why none survives

`obj_frame_rotation` is the only upstream quantity that departs before the transition in either
run — at 1015 in seed 3, 20 steps before its subset. **It fails replication:** in seed 9 the same
term departs at **835**, *after* the transition. A quantity that leads in one seed and lags in the
other is not a precursor, and seed 3's threshold for it is extremely tight (3σ = 0.0028 on a
baseline std of 0.0009), so it is the series most exposed to small drift.

`grad_graph_encoder`: no departure in seed 9, departure at 1100 in seed 3 — 65 steps *after* the
transition began there. Not a precursor.

`context_written_norm`: no departure in either run.

`entity_latent_std`: no departure in seed 9; 1075 in seed 3, after the transition.

**No upstream quantity departs before the transition in both runs. Case 3 — no signal.**

## 6. Predictions, scored

| prediction | outcome |
| --- | --- |
| 1. seed 9's usage onset lies before step 750 | **falsified.** With a dense baseline the onset is 815, and the pre-750 elevation Step 17 saw is noise. |
| 2. usage still leads geometry, by more than 60 steps | **falsified, and reversed.** The subset departs 5 steps *before* usage; the whole transition spans 10 steps. |
| 3. no upstream quantity departs before the usage onset | **supported** in both runs. |
| 4. seed 3 shows the same ordering | **supported** — subset → usage → rotation in both. |

Two of four predictions failed, both in the direction of Step 17 having over-read its own data.
The two that held are the null and the replication.

## 7. Decision

**Onset question: A — onset located.** Seed 9's transition begins at step 810–815, with 40+ quiet
dense checkpoints before it inside the window. The window was wide enough; the earlier apparent
onset was a baseline artifact.

**Precursor question: C — the transition remains effectively abrupt.** It occupies ~10 steps in
seed 9 and ~30 in seed 3, and nothing upstream of the prediction moves first in both.

**Replication: the within-transition ordering recurs**, in two seeds. Nothing is claimed about the
population from two runs.

## 8. Limitations

* **Two runs.** Seed 9 and seed 3 are both eventual graph-users; no non-escaper was densely
  instrumented over a comparable window, so "these series do not move" is not contrasted against
  a run where nothing happens at all.
* **The detection criterion is baseline-sensitive, which is the lesson of this step.** A 3σ rule
  is only as good as its variance estimate, and two successive reports have now been misled by
  one. Any future use should estimate variance at the resolution of the measurement.
* Seed 3's transition is slow and still incomplete at step 1200, so its departure steps are less
  sharply defined than seed 9's.
* The diagnostics remain aggregate module-level norms; direction, per-layer structure, curvature
  and optimiser state were not measured.
* One corpus, one scale, 1200 steps, synthetic throughout.

## 9. Recommended next step

**Stop looking for a precursor with these diagnostics.** Three steps have now searched — Step 16
at 50-step spacing, Step 17 at 5-step spacing, Step 18 with a corrected baseline and a
replication — and the answer has been the same each time, with the one apparent positive
dissolving under a better variance estimate. A fourth search at finer spacing would face the same
instrumentation limit, not a resolution limit.

What has never been measured is *direction*. Every gradient number in Steps 14–18 is an aggregate
L2 norm per module, which is blind to a change in where the gradient points. A transition
occupying 10 steps with no magnitude change is exactly what a change in gradient *direction*
would look like. The concrete diagnostic: record the cosine similarity between successive
gradient steps, and between the gradient and the direction that separates the two attractors, on
the frozen seed 9 configuration over 780–860. That is one run, no intervention, and it tests a
hypothesis the existing instrumentation cannot see.

**Also worth correcting:** Step 17's report states a 60-step usage lead and a phase of graph
reliance without geometric benefit. Both are superseded by §2 above. Step 17's artifacts are
preserved unchanged; this report is the correction, and any future summary should cite this one
for the ordering.

**Not started.** No intervention, no architecture change, no new loss, no budget change.
