# Step 12 — Relation-conditioned values: preregistration

**Status:** preregistered. Written before any Step 12 model was trained.
**Scope:** one architectural change, one arm, three seeds, validation only.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why this experiment exists

Step 11 measured where relational information is lost between the relationship graph and the
predicted frame, and the answer was specific enough to act on. Destroying relation *type* at
evaluation — same endpoints, types permuted — moved A3's rotation error by **0.0003°**. The
reason was visible in the weights rather than inferred from the behaviour: relation type enters
this architecture only as an additive scalar bias on attention logits, and that bias carries
**0.55%** of the logit scale (std 0.0026 against 0.4663). The channel never left its zero
initialisation in any useful sense.

That is a missing channel, not a shortage of capacity. Step 11 §23 rejected a larger model,
greater depth, and a redesigned head on the evidence. Step 11 §24 named the replacement:

> **The smallest change that gives a relation its own message.** Condition the attention
> *value* on relation type, rather than only the logit — a per-relation additive term on the
> value vector, or an edge-type embedding concatenated before the value projection. One
> change, a handful of parameters per relation, and it isolates exactly the mechanism §11
> identifies.

This is that experiment, run to that specification.

---

## 2. The change

A per-relation additive term on the attended value. For each attention head, the mass that
query *i* sends to each relation type is accumulated from the existing attention weights, and
each relation's own embedding vector is added to the attended output in proportion:

```
attended[b,h,i] += Σ_r mass[b,h,i,r] · V_rel[r,h]
```

where `mass[b,h,i,r]` is the total attention weight query *i* places on keys reached by
relation *r*. The embedding is zero-initialised, so at initialisation the term is exactly zero
and the model is the Step 11 model.

**Parameter delta, measured:** 3,226,647 → 3,231,255, **+4,608 (+0.14%)**. One vector of width
256 per relation type per the encoder's 18-slot vocabulary, minus the heads that are scoped out.

**What is unchanged:** the corpus, the splits, AWR, the entity axis, the protected identity
slice (channels 0–95), the parent-relative frame definition, the rotation target, the chordal
objective, λ=3.0, the optimiser, the schedule, the step count, the batch size, the seeds, and
every other dimension of the encoder. The existing scalar relation bias stays; this adds to it
rather than replacing it.

### 2.1 Isolation — verified, not assumed

A new parameter tensor ordinarily consumes RNG at construction and shifts every weight
initialised after it. That would have made the control differ from the treatment by seed *as
well as* by the channel under test — and with A3's seed spread at ~1.0°, roughly the size of
the effect being measured, that confound would have been fatal and invisible.

The embedding is therefore constructed inside `torch.random.fork_rng(devices=[])`. Verified
before preregistration:

* the entity latent is **bit-identical** between control and treatment at initialisation;
* the set of parameters whose values differ at initialisation is **empty**;
* the only difference in the state dict is the added `graph_encoder.relation_value.weight`;
* permuting the embedding's rows changes the latent, so the term does depend on relation type;
* channels 0–95 (protected identity) are untouched.

---

## 3. Predictions, recorded before the run

Step 11 §24 asked for two predictions to be recorded so that the result can contradict them.
Quoted verbatim from that section:

> Two predictions worth recording before it runs, so the result can contradict them: if H3 is
> the binding constraint, `shuffle_spatial_types` should become *expensive* under the new
> encoding; and the gain should appear in the two seeds that already use connectivity before
> it appears in seed 1.

Made concrete for scoring:

**P1 — the type channel becomes load-bearing.** On Step 12 checkpoints, the
`shuffle_spatial_types` ablation should cost at least **1.0°**, against **0.0003°** in Step 11.
P1 fails if the shuffle still costs under 1.0°, or if the seeds disagree in sign.

**P2 — the gain is ordered by prior graph usage.** Step 11 §12 measured what the graph is
worth to each seed: seed 0 **2.59°**, seed 2 **2.51°**, seed 1 **0.38°**. If the new channel
acts by enriching a mechanism the model already uses, seeds 0 and 2 should improve before seed
1 does. P2 fails if seed 1's improvement matches or exceeds the mean of seeds 0 and 2.

P1 and P2 are scored independently and reported whichever way they fall. **Neither is the
decision rule.** They are the mechanism's claims about itself; the primary outcome below is
what decides the step.

---

## 4. Primary outcome and decision rule

**Metric:** mean geodesic rotation error in degrees on the **validation** split.

**Control:** the three frozen A3 λ=3.0 checkpoints from Step 10 Change 2, re-scored under the
identical evaluation path. Validation `intact` = **21.80°** mean (seed 0 20.96, seed 1 23.43,
seed 2 21.01), as measured by the Step 11 ablation.

**Treatment:** A3 with `relation_values=True`, three seeds (0, 1, 2), λ=3.0, everything else
frozen.

**Rule — fixed now, before any result:**

> The relation-conditioned value encoding is judged to help only if the mean validation
> rotation error improves by at least **1.0°** and all three seeds move in the same direction.

1.0° is the minimum used throughout Step 11, chosen because A3's own seed spread is 1.03° and
a smaller difference cannot be distinguished from it. It is not adjusted here.

**Outcomes:**

* **improves** (mean ≥ 1.0° better, seeds agree) — the missing-channel diagnosis is supported;
  H3 is the binding constraint. A confirmatory test-set run may then be pre-registered
  separately. No test split is read in Step 12.
* **no meaningful change** (< 1.0°, or seeds disagree) — H3 is not the binding constraint. Per
  Step 11 §24, *"If that change moves nothing, H2 and H6 become the live hypotheses and depth
  or global pooling is the next study."* This is a real and reportable outcome; it will be
  reported plainly, not reframed.
* **worse** (mean ≥ 1.0° worse, seeds agree) — the added term is harmful at this λ and
  parameter count; reported as such.

---

## 5. Test-set discipline

**No test split is read in Step 12.** Not for model selection, not for scoring, not for
illustration. The step is scored on validation against a validation control. A test measurement
would require its own preregistration after this result is in.

---

## 6. Stop conditions

Declared in advance. If any of these occurs, the run stops and the report says so:

* the control and treatment are found to differ at initialisation by anything other than the
  new tensor;
* the integrity gate fails any of its six checks on a Step 12 run;
* a run cannot be reproduced bit-exactly from its seed;
* free disk falls below 2 GiB (the runner's guard; research artifacts are never deleted to
  make space);
* the parameter delta exceeds +1% — this is meant to be the *smallest* change that gives a
  relation its own message, and a larger one is a different experiment.

---

## 7. What this experiment cannot establish

* It cannot show that relation-conditioned values are the *best* encoding — only whether the
  cheapest one moves the primary metric.
* It cannot separate H2 (depth) from H3 (encoding) if the result is negative; it can only
  remove H3 from contention.
* A validation improvement is not a test result and will not be described as one.
* The corpus is synthetic. Nothing here is a claim about anatomy.
