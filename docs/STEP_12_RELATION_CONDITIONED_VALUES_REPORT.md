# Step 12 — Relation-conditioned values: report

**Pre-registered in** [`STEP_12_RELATION_CONDITIONED_VALUES_PLAN.md`](STEP_12_RELATION_CONDITIONED_VALUES_PLAN.md),
written before any Step 12 model was trained.

**Status: negative result. The change did not help, and the pre-registered mechanism prediction
failed.** Reported as measured.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Verdict

| | |
| --- | --- |
| **Primary outcome** | no meaningful change |
| **Mean validation rotation error** | control **21.80°** → treatment **21.81°** |
| **Mean gain** | **−0.013°** (the treatment is very slightly worse) |
| **Seeds agree in sign** | **no** (−0.002, −0.040, +0.004) |
| **Decision threshold** | 1.0° |
| **P1 — type channel becomes load-bearing** | **fails** |
| **P2 — gain ordered by prior graph usage** | holds arithmetically, **uninformative** |

The effect is roughly **80× smaller** than the threshold the step was required to clear, and
smaller than the precision at which the two arms could be told apart at all.

**H3 is not the binding constraint.** Giving each relation its own message, rather than only its
own attention weight, changes nothing about how well A3 places rotations.

---

## 2. What was changed

A per-relation additive term on the attended value. For each attention head, the attention mass
query *i* places on keys reached by relation *r* is accumulated, and each relation's learned
vector is added in proportion:

```
attended[b,h,i] += Σ_r mass[b,h,i,r] · V_rel[r,h]
```

| | |
| --- | --- |
| parameters, control | 3,226,647 |
| parameters, treatment | 3,231,255 |
| delta | **+4,608 (+0.14%)** |

Everything else is frozen: corpus, splits, AWR, entity axis, protected identity slice, frame
definition, rotation target, chordal objective, λ=3.0, optimiser, schedule, 1200 steps, batch
size, seeds. The existing scalar relation bias is retained; this adds to it.

### 2.1 The comparison is genuinely paired

A new parameter tensor ordinarily consumes RNG at construction and shifts every weight
initialised after it. That would have made control and treatment differ by **seed as well as by
the channel under test** — and A3's seed spread is ~1.0°, the same size as the effect being
measured. The confound would have been fatal and invisible.

The embedding is therefore constructed inside `torch.random.fork_rng(devices=[])`. Verified and
pinned by test:

* the entity latent is **bit-identical** between arms at initialisation;
* the set of shared parameters whose values differ at initialisation is **empty**;
* the only state-dict difference is the added `graph_encoder.relation_value.weight`;
* permuting the embedding's rows moves the latent, so the term does read relation type;
* channels 0–95 (protected identity) are untouched.

Both arms were then scored **through one evaluation path** — the Step 11 ablation harness, same
split, same batching, same seeded generator. Reading the control's own training-time record
instead would have compared numbers produced by different code, which a 1.0° rule cannot
survive.

---

## 3. Primary result

Validation, mean geodesic rotation error in degrees:

| seed | control | treatment | gain |
| --- | ---: | ---: | ---: |
| 0 | 20.956 | 20.957 | −0.002 |
| 1 | 23.428 | 23.468 | −0.040 |
| 2 | 21.010 | 21.006 | +0.004 |
| **mean** | **21.798** | **21.810** | **−0.013** |

The pre-registered rule required a mean improvement of at least 1.0° with all three seeds moving
the same way. Neither condition is met; the seeds do not even agree in sign.

---

## 4. The change was real, and it did train

A negative result is worth nothing if the change was never applied, so this was checked before
anything was concluded.

**The term is live at inference.** The two arms' outputs differ, at the third and fourth decimal
place. This is not a dead code path.

**The weights left zero.** From zero initialisation, after 1200 steps:

| seed | `relation_value` abs-max | `relation_value` std | `relation_bias` abs-max |
| --- | ---: | ---: | ---: |
| 0 | 0.0828 | 0.00651 | 0.0308 |
| 1 | 0.1017 | 0.00750 | 0.0310 |
| 2 | 0.0994 | 0.00798 | 0.0360 |

**The routing is correct.** Of 4,608 entries, 768 are non-zero — which is exactly right, not a
defect:

* 12 of the 18 vocabulary slots correspond to relations that occur in this corpus; all 12 have
  non-zero rows, and the 6 untouched rows are relations the corpus never contains;
* each relation reaches only the two heads scoped to its graph kind — structure `[0,1]`,
  spatial `[2,3]`, functional `[4,5]` — and the two global heads (`head_graph = −1`) correctly
  carry no relation message;
* 12 relations × 2 heads × 32 channels per head = **768**, matching exactly.

So the channel exists, is reachable, receives gradient, and trained. It still did nothing.

---

## 5. Why it did nothing — the mechanism

Step 11 explained the inert bias channel by measuring it: relation type entered only as a scalar
attention bias holding **0.55% of the attention logit scale**. Step 12's negative result gets the
same treatment, because "the weights trained and nothing happened" is not yet an explanation.

The direct analogue is the norm of the new message against the norm of the ordinary attended
value it joins. Measured on a real validation batch, replicated from the layer's own forward pass
(and tested to agree with it to 1e-5):

| layer | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| 0 | 0.31% | 0.30% | 0.37% |
| 1 | 0.29% | 0.30% | 0.38% |
| 2 | 0.32% | 0.35% | 0.37% |
| 3 | 0.31% | 0.34% | 0.36% |
| **mean** | **0.31%** | **0.32%** | **0.37%** |

**The relation message settled at 0.33% of the attended value — the same order as the 0.55%
scalar bias it was meant to replace.** Flat across all four layers, consistent across all three
seeds.

This is the finding that makes the negative result useful. The model was given a channel with
256 dimensions per relation, free to grow, starting from zero, and the optimiser put
approximately the same negligible fraction of signal into it as it had put into a single scalar.
**The problem is not the channel's *form*.** Changing relation type from a scalar to a vector did
not change how much the model wanted to use relation type.

---

## 6. The two pre-registered predictions

Both were recorded before the run so the result could contradict them. It did.

### P1 — `shuffle_spatial_types` should become expensive. **Fails.**

| | Step 11 | Step 12 |
| --- | ---: | ---: |
| cost of permuting relation types | 0.0003° | **0.0006°** |
| per-seed | +0.0003, +0.0002, +0.0002 | +0.0003, **−0.0002**, +0.0018 |
| threshold | 1.0° | 1.0° |

The cost roughly doubled in a relative sense and remains **three orders of magnitude below the
threshold**, with seed 1 now moving the wrong way. Destroying relation type is still free. P1
predicted this would change if H3 were binding; it did not.

### P2 — the gain should land in seeds 0 and 2 before seed 1. **Holds, but says nothing.**

Seeds 0 and 2 average +0.001°; seed 1 is −0.040°. The ordering is as predicted, so the test
passes as written.

**This is not evidence for anything.** P2 was specified as an ordering rather than a threshold,
and an ordering among three numbers that are all indistinguishable from zero is noise. Reporting
it as support for the mechanism would be overclaiming. It is recorded as passing because it was
pre-registered as passing under exactly these arithmetic conditions, and it is discounted here
for the reason above rather than reinterpreted after the fact.

### Secondary diagnostic — what the graph is worth

Carried over from Step 11 §12 (`entities_only` − `intact`), not part of the decision rule:

| seed | control | treatment |
| --- | ---: | ---: |
| 0 | 2.590° | 2.856° |
| 1 | 0.376° | 0.337° |
| 2 | 2.510° | 2.499° |

The bimodality Step 11 identified survives unchanged: two seeds extract ~2.5° from the
relationship graph, seed 1 extracts almost nothing. The new encoding did not move a seed across
that divide, which is the outcome most directly relevant to whether the encoding matters.

---

## 7. Integrity and discipline

* **No test split was read.** The runs were launched with `--splits validation`; the runner's
  default is the four test splits and was overridden. `protocol.splits == ["validation"]` in
  every run record, and `test_splits_read: []` in every report.
* **The corpus was not touched.** Both freeze manifests verify clean by SHA-256:
  `step10-change2-w3/CORPUS_FROZEN.sha256` (7 files) and
  `step10-placement/CHANGE1_FROZEN.sha256`.
* **No stop condition was triggered.** Initialisation isolation held; the parameter delta is
  +0.14% against a +1% limit; free disk stayed above the 2 GiB guard.
* **Nothing was tuned after seeing results.** The threshold, the control, and both predictions
  are as written in the plan.

---

## 8. What this does and does not establish

**Establishes:** the cheapest form of relation-conditioned messaging does not improve rotation
placement in this architecture at this scale, and the reason is that the optimiser does not
increase its use of relation type when given a richer way to express it.

**Does not establish:**

* that relation-conditioned values are useless in general — only that this form, at this size,
  under this objective, moved nothing;
* that a larger or deeper version would also fail — untested;
* anything about the test splits;
* anything about anatomy. The corpus is synthetic.

It also does not rescue H3. The honest reading is that H3 is removed from contention, and that
the 0.33% measurement suggests the obstacle is upstream of how a relation is encoded.

---

## 9. Where this leaves the programme

Step 11 §24 fixed the consequence in advance:

> If that change moves nothing, H2 and H6 become the live hypotheses and depth or global pooling
> is the next study.

That is the position. H2 (the signal is never propagated far enough) and H6 (there is no global
aggregation to carry scene-level structure) are now the live hypotheses.

One observation from §5 should shape whatever comes next. Two different encodings of relation
type — a scalar bias and a 256-dimensional message — both settled at well under 1% of their
respective signal scales. Two independent parameterisations converging on "barely used" is
weak evidence that the constraint is not in the parameterisation at all, and that a third
encoding would land in the same place. Depth and global pooling are the pre-registered next
candidates; the 0.33% figure is a reason to also ask whether the training objective rewards
relational information through this path at all.

That question is out of scope here and is not started.
