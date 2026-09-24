# Step 15 — The seed mode: preregistration

**Status:** preregistered. Written before any new run was launched.
**Question:** given an objective that would pay 82% for graph conditioning, why does training
reach a graph-using solution on some seeds and an identity-only solution on others?
**Nothing about the model, objective, corpus or λ changes.** This is a diagnostic on the frozen
configuration.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why this is the next question

Four candidate explanations have been eliminated by measurement:

| step | hypothesis | outcome |
| --- | --- | --- |
| 12 | relation encoding is too thin | negative: −0.013°, the channel settled at 0.33% of signal |
| 13 | propagation depth (H2) | negative: 22.64 vs 21.80, and the graph became worth *less* |
| 13 | global graph context (H6) | negative: 21.88 vs 21.80, though the context provably carries graph configuration |
| 14 | the objective is insufficiently relational (H7) | negative: graph conditioning cuts the objective's rotation term by **82.2%** |

What survives all four is a **bimodality in training outcome**, now measured three independent
ways on the same three seeds:

* Step 11 — the whole graph is worth **2.59°, 0.38°, 2.51°** to seeds 0, 1, 2;
* Step 13 — the mode is *manipulable*: doubling propagation moved seed 0 from using (2.59°) to
  barely using (1.15°), and its rotation from 20.96° to 23.43°;
* Step 14 — on the relationally-required subset, seed 1 scores **41.56°** against an
  identity-only lookup at 42.22°, i.e. it sits on the identity floor exactly where the graph
  would help most.

## 2. Free findings already obtained, before any new training

Read from the existing run manifests' `train_history` and `validation_history`. These cost
nothing and shape what the sweep needs to measure.

**(a) The mode is decided between step 400 and step 800, and is stable afterwards.**
Validation rotation (the manifest's `rotation_error`, which reads ~0.1–0.15° above the
canonical `rotation.validation.mean_deg` — see Step 14 §1.1 for that metric trap):

| seed | step 400 | step 800 | step 1200 |
| --- | ---: | ---: | ---: |
| 0 | 23.75 | **21.14** | 21.07 |
| 1 | 23.71 | **23.66** | 23.57 |
| 2 | 24.03 | **21.21** | 21.13 |

All three begin at the identity-only floor (23.51°). Two leave it during the second third of
training; one never does.

**(b) The training loss does not distinguish the modes.** Frame-loss trajectories across 25
recorded steps are indistinguishable, and seed 1's is frequently the *lowest* of the three
(step 300: 0.477 against 0.623 and 0.585; step 750: 0.454 against 0.537 and 0.637). At step
1199: 0.510, 0.537, 0.595.

So the two modes are near-degenerate in achieved training loss while differing by ~2.2° in graph
usage and ~1.8° in held-out rotation. **This is the sharpened puzzle:** Step 14 showed the
objective *would* pay 82% for graph conditioning in principle, and (b) shows that at the
solutions training actually reaches, it is not paying differentially at all.

## 3. What Step 15 measures

Only what three seeds cannot establish: **the base rate, and whether the outcome is genuinely
bimodal or a continuum.** Three samples cannot tell a two-mode distribution from a wide
unimodal one.

Seven new runs, seeds 3–9, at the frozen configuration. Ten seeds total with the existing three.

| held identical | value |
| --- | --- |
| arm | A3 |
| corpus | `datasets/processed/step10_rotated` (freeze verified) |
| objective | chordal, λ=3.0, `frame_weight` 2.0 |
| `relation_values` | false |
| `graph_recurrence` | 1 |
| `frame_scene_context` | false |
| steps / batch / optimiser / schedule | 1200 / 8 / AdamW / Step 9 cosine |
| splits scored | **validation only** |

The only thing that varies is the seed. No architecture change, no new loss, no λ change.

## 4. Preregistered classification rule

A seed's **graph usage** is the established Step 11 measure: the rotation cost of removing every
relationship at evaluation (`entities_only` − `intact`), on validation, on frozen weights.

Fixed now, from the three observed values (2.59, 0.38, 2.51) and the 1.0° minimum used
throughout Steps 11–13:

> * **graph-using** if graph usage **≥ 1.5°**
> * **non-using** if graph usage **≤ 0.8°**
> * **intermediate** otherwise (0.8°–1.5°)

The gap between 0.8 and 1.5 is deliberately left unlabelled: it is the region whose population
decides the shape question, and pre-assigning it to either mode would beg it.

### 4.1 What counts as bimodal

> The outcome is called **bimodal** only if the **intermediate** band holds a strict minority of
> seeds — at most 2 of 10 — and both outer bands are occupied.
>
> It is called a **continuum** if the intermediate band holds 3 or more.
>
> If one outer band is empty, the correct description is that the mode is not bimodal but
> **near-deterministic in that direction**, and the three-seed picture was a small-sample
> artefact.

### 4.2 Secondary, reported but not decisive

* validation rotation per seed, canonical `rotation.validation.mean_deg`;
* whether graph usage and validation rotation are monotonically related across ten seeds — if
  they are not, "graph usage" and "accuracy" are separable and that is itself a finding;
* the step-400 / step-800 / step-1200 validation trajectory for each new seed, to test finding
  2(a) on seven fresh samples rather than the three that suggested it.

## 5. Predictions, recorded before the sweep

1. **The distribution is bimodal** with the intermediate band nearly empty. Basis: the three
   observed values cluster tightly at 2.5 and low at 0.4, with nothing between, and Step 13's
   depth arm produced 1.15 — which would fall in the intermediate band, weak evidence against.
2. **The non-using mode is the minority**, roughly 1 in 3. Basis: one of three seeds so far.
3. **Every seed sits at the identity floor at step 400**, and those that leave it do so by step
   800. Basis: finding 2(a), on three seeds.

Prediction 1 and 2 are about the same measurement and could both fail together. The null — a
broad continuum with no modes — is a live outcome and would mean the "mode" framing used since
Step 11 is wrong.

## 6. Test discipline

**`test_splits_read: []`.** The sweep trains and scores on validation only. Ten seeds at the
frozen configuration is not a model-selection exercise — no configuration is being chosen — so
there is nothing for a test measurement to confirm. If a later step proposes an intervention,
its confirmatory test belongs to that step.

## 7. Stop conditions

* any freeze manifest changes;
* the three existing seeds' graph usage does not reproduce (2.59, 0.38, 2.51);
* free disk falls below 2 GiB; research artifacts are never deleted to make space;
* a run cannot be reproduced from its seed;
* the sweep would require changing the model, objective or corpus.

## 8. What Step 15 will not do

No intervention. No architecture change. No new loss. No λ change. Step 15 establishes the
shape of the outcome distribution and when it is decided; it does not attempt to move a seed
from one mode to the other. That would be the next study, and it is not started here.
