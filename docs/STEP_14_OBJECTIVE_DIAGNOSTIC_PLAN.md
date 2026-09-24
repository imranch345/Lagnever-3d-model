# Step 14 — Objective diagnostic: preregistration

**Status:** preregistered. Written before any Step 14 measurement.
**Nothing is trained.** No loss is added, no coefficient changed, no model modified.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Question

Does the current training objective actually force the model to learn relational information
when relational information is necessary to minimise the target error?

**H7 — the objective is insufficiently relational.** It can be minimised substantially through
entity identity alone, so the optimiser has little incentive to encode
relationship-dependent information.

**H8 — the objective already provides sufficient relational supervision.** The remaining
failure has another cause.

Neither is assumed.

## 2. Frozen references, verified from artifacts before planning

| reference | value | source |
| --- | ---: | --- |
| A3 rotation, `test_seen` | 21.38° ± 1.03 (20.79, 22.57, 20.78) | `rotation.test_seen.mean_deg` |
| identity-only lookup, `test_seen` | 22.65° | `step10-change2/graph_conditional_floor.json` |
| identity + graph lookup, `test_seen` | 11.57° | same |
| identity-only lookup, validation | 23.51° | `step11/graph_conditional_validation.json` |
| identity + graph lookup, validation | 12.24° | same |
| A3 rotation, validation | 21.80° | Step 13 evaluation |
| Step 12 | 21.80 → 21.81, gain −0.013° | frozen |
| Step 13 | A3-depth 21.99, A3-global 21.44 | frozen |

All four freeze manifests verified CLEAN before starting.

**A3 recovers 15.2% of the available relational improvement on validation**
((23.51 − 21.80) / (23.51 − 12.24)). This is the gap Step 14 asks about.

## 3. Preregistered rules

### 3.1 The relationally-required subset (brief §12)

Selected on **validation only**, from the lookup tables fitted on **train only**.

> An entity is in `relationally_required_validation` if, for that entity,
> **(a)** the identity-only lookup error is at or above the validation identity-only mean
> (23.51°), **and**
> **(b)** the graph-aware lookup error is at least **10.0°** lower than the identity-only error.

Clause (a) is "identity does badly here". Clause (b) is "the graph demonstrably fixes it", and
10.0° is chosen as slightly under the measured mean improvement (11.27°), so the subset is the
clear cases rather than a tail. Entities whose graph signature is unseen in training are
excluded, because for them the graph-aware lookup is *defined* to equal the identity-only
lookup and the comparison would be vacuous.

The rule is fixed here. It will not be adjusted after seeing A3's score on the subset.

### 3.2 The H7 decision rule (brief §17)

> H7 is supported only if **both**:
> 1. relationship-dependent target variation is present and measurable, **and**
> 2. the current objective provides substantially weaker supervision for that variation than
>    for identity-predictable variation.

**A model simply failing to learn graph information is not evidence that the loss is
inadequate.** Clause 2 is about the objective, and is scored on the objective, not on A3.

Clause 2 is operationalised as: on relationship-varying examples, the objective value of a
**graph-conditioned** prediction must be *materially lower* than that of the best
**identity-only** prediction. "Materially" is fixed as a **≥10% reduction** in the objective's
rotation term. If the objective separates them by less than that, it cannot be said to reward
relational conditioning; if it separates them by more, the supervision exists.

### 3.3 Gradient diagnostic (brief §10)

Reported, not decisive. Gradient magnitude reaching the identity pathway, the graph pathway,
the relation encoder, the entity latent and the frame head, on frozen checkpoints, three seeds.
Used to answer whether *any* meaningful optimisation signal reaches the graph pathway — not to
select anything.

### 3.4 Graph-scrambled objective test (brief §11)

The objective is computed twice on the same batches with the same targets and the same frozen
weights: once with the correct graph, once with a seeded permutation of the relation graph. The
question is whether the **objective** moves, not whether the model scores well.

## 4. Test discipline

**Diagnostic phase: `test_splits_read: []`.** Every measurement in §3 uses train (for fitting
lookups and for gradients) and validation (for subset selection, thresholds and all objective
diagnostics). The test splits are read only for a final confirmatory measurement, after every
diagnostic choice is frozen, and cannot change any decision.

## 5. Stop conditions

* the current objective cannot be reconstructed exactly;
* either lookup baseline cannot be reproduced;
* the relationally-required subset cannot be defined without test leakage;
* counterfactuals cannot be found in existing data without modifying the corpus;
* any diagnostic would require changing the model;
* any frozen checksum changes.

## 6. What Step 14 will not do

No loss is added. No coefficient is changed. A3 is not retrained. If the diagnostics support
H7, the intervention is **designed and written down only** — not implemented, not trained. If
they do not, no loss is added and the next question is defined instead.
