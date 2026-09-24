# Step 19 — Gradient direction across the transition: report

**Preregistered in** [`STEP_19_GRADIENT_DIRECTION_PLAN.md`](STEP_19_GRADIENT_DIRECTION_PLAN.md).

**Result: C — no direction precursor.** The gradient's direction does not change before seed 9's
transition. What does change, changes at step 825 or later — after the transition began at 810.

**`test_splits_read: []`.** One run, diagnostic only, nothing causal.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Why this run

Steps 16–18 searched for a precursor at three resolutions and found none; Step 17's one apparent
positive dissolved under a better variance estimate. Every gradient number in that search was an
aggregate **L2 norm**, which is blind to direction. Step 18 established that the transition
occupies ~10 steps with no magnitude change — which is what a change in *where the gradient
points* would look like. This measured that.

## 2. Reproduction

| check | result |
| --- | --- |
| reproduces frozen seed 9 at steps 800 / 900 / 1200 | **yes** |
| final usage 2.5183, rotation 20.9933 | matches frozen |
| **agrees with Step 18 across all 17 shared dense checkpoints in 780–860** | **bit-for-bit** |

## 3. Successive-gradient direction — the candidate precursor

cos(g_t, g_{t−1}), all parameters:

| step | cos | | step | cos | | step | cos |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 785 | +0.843 | | 815 | +0.732 | | 845 | +0.736 |
| 790 | +0.789 | | 820 | +0.935 | | 850 | +0.826 |
| 795 | +0.829 | | 825 | +0.835 | | 855 | +0.784 |
| 800 | +0.853 | | 830 | +0.948 | | 860 | +0.944 |
| 805 | +0.905 | | 835 | +0.780 | | | |
| 810 | +0.905 | | 840 | +0.860 | | | |

Mean **+0.844**, std 0.069, range [0.732, 0.948].

**Prediction 1 is falsified.** I predicted this would be low and noisy, on the grounds that
minibatch gradients at batch 8 are sampling-noise dominated. They are not: successive gradients
are strongly and stably aligned. That is a fact about this optimisation worth recording on its
own — the descent direction is persistent, not a random walk.

Per pathway, first sustained departure from the 785–805 baseline (3σ, three consecutive
checkpoints):

| pathway | baseline | departure |
| --- | --- | ---: |
| `all_parameters` | +0.844 ± 0.042 | **none** |
| `graph_encoder` | +0.823 ± 0.039 | **none** |
| `frame_head` | +0.844 ± 0.043 | **none** |
| `relation_parameters` | +0.895 ± 0.084 | **none** |
| `alignment` | +0.993 ± 0.004 | **none** |
| `entity_identity` | +0.860 ± 0.026 | 825 |
| `geometry` | +0.747 ± 0.087 | 825 |
| `scene_and_text` | +0.925 ± 0.051 | 825 |

**The three that depart do so at step 825 — after the transition began at 810** (Step 18: subset
810, usage 815, rotation 820). They are part of the transition, not precursors to it. The
graph encoder, the pathway most directly implicated, shows **no departure at all**.

**Prediction 2 is supported:** no sustained departure before step 810 in any pathway.

## 4. Gradient magnitude, re-measured

No pathway's norm departs its baseline except `alignment`, at step 850 — after. This reproduces
Steps 16–18's magnitude null on a fourth, independent measurement.

## 5. Escape-axis alignment — descriptive only

cos(−g_t, w_final − w_780). **This axis uses the end of the run, so it is not available at
training time and is not a predictive signal.** The artifact records
`uses_future_information: true`.

| step | all parameters | graph encoder |
| ---: | ---: | ---: |
| 780 | +0.0048 | +0.069 |
| 800 | −0.0016 | +0.022 |
| 810 | −0.0013 | +0.029 |
| 820 | −0.0023 | +0.042 |
| 830 | +0.0017 | +0.072 |
| 840 | −0.0073 | **−0.065** |
| 860 | −0.0054 | −0.060 |

**The descent direction is very nearly orthogonal to the displacement the weights actually
undergo.** Overall alignment stays within ±0.01; in 3.2 million dimensions, chance alignment is
about 0.0006, so this is roughly ten times chance and still negligible. The graph encoder is
larger (~0.02–0.07, about a hundred times chance) and **flips sign at step 840**, after the
transition.

**Prediction 3 is partly falsified**: alignment is not reliably positive — it hovers at zero —
and there is a sign flip, though it occurs after the transition rather than at it. Either way it
is descriptive and carries no precursor claim.

## 6. Decision

**C — no direction precursor.** No quantity available at training time departs before step 810.

Combined with Steps 16–18 this is the fourth independent search, and the second class of
measurement, to return the same answer.

## 7. Limitations

* **The baseline is six checkpoints (780–805).** That is a thin variance estimate, stated in the
  plan in advance. A departure smaller than ~3× that spread would not be detected, and the
  criterion genuinely cannot decide for effects of that size.
* **One run.** Seed 9 only; no non-escaper was instrumented for direction, so "these cosines do
  not move" is not contrasted against a run where nothing happens.
* **Gradients are measured on four fixed training batches**, which is what makes them comparable
  across checkpoints and with Steps 14–18, but it is not the gradient the optimiser actually took
  at that step. Direction measured on the training stream's own batches could differ.
* The escape axis is one straight line from 780 to the end; the real path is curved, so a small
  cosine does not mean the gradient was unhelpful.
* One corpus, one scale, 1200 steps, synthetic throughout.

## 8. Recommended next step

**Stop searching for a precursor.** Four searches, two measurement classes, three resolutions,
one replication — and every apparent positive has dissolved on inspection. The honest reading is
that **this class of instrumentation cannot resolve what happens at the transition**, and a fifth
variant of it is not the experiment to run.

Two things follow that are worth more than another search.

**First, the finding that is not null.** Successive gradients are aligned at ~0.84 throughout,
and the descent direction is almost orthogonal to the net displacement. Both are stable, both
were unpredicted, and together they say the optimiser is taking consistent steps that
individually point almost nowhere near where the weights end up. That is a description of the
loss landscape, not of the transition, and it is the first thing in Steps 16–19 that was not
already known.

**Second, what would actually settle the question.** Everything so far is observational on a
single trajectory. The question "what causes the escape" is causal, and causal questions need
intervention: the natural design is a **branch test** — take the frozen seed 9 checkpoint at step
805, before the transition, and continue it several times with different data orders. If most
branches escape within ~50 steps, the transition was already determined by the weights at 805 and
the search for a precursor in the *trajectory* was misdirected. If branches diverge, the trigger
is in the data ordering, which is measurable. Either outcome is informative, and neither requires
changing the architecture, objective or budget.

That is an intervention, so it is a new step and is **not started here**.

## 9. Answer to the research question

> **Does the direction of the gradient change before seed 9's transition, in a way that magnitude
> measurement could not see?**

**No.** Successive-gradient alignment is high and stable at +0.844 ± 0.069 through the window, and
no pathway's direction departs its baseline before step 810. The three pathways that do depart do
so at step 825, after the transition had begun, and the graph encoder — the pathway the
hypothesis concerned — does not depart at all.
