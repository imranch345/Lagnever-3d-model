# STEP 10 — ROTATION WEIGHT STUDY

**Lagnav 3D — Settling what Change 2 could not: the weight, or the architecture?**

Screening only, on validation. All data is procedurally generated; nothing here is anatomy or
is medically validated.

---

## 1. Why this ran

Change 2 concluded that no arm beat the rotation floor on any split, and named the reason it
could not call that an architectural verdict: the rotation term was **7% of the frame loss**.
That share followed from the pre-registered calibration rule — rotation gets a quarter of the
translation term, Step 8's declared emphasis — combined with Step 8's inherited scale weight,
which takes 64% because it is an L1 sum over three log axes.

Two explanations remained, and the experiment could not separate them:

* the architecture does not learn rotation;
* rotation was 7% of the objective.

A prior question had to be answered first, and it was, without training anything: **is the
information even in the model's inputs?** The corpus is built so the relationship graph is the
only channel carrying the arrangement. A lookup keyed on entity identity *and* graph — inputs
the model already receives, fitted on `train` — halves the rotation error, 22.65° to 11.57° on
`test_seen`. The headroom is real and reachable, so the weight was worth screening.

---

## 2. Protocol, fixed before the first run

| | |
| --- | --- |
| weights | 0.33 (Change 2's), 1.0, 3.0, 10.0 — spanning 7% to about 70% of the frame loss |
| arms | A3 (full typed graph) and A1 (no graph) |
| seeds | 1 — this is screening, not a confirmatory result |
| scored on | **validation only**; the runner reads no test split |
| everything else | Step 9's protocol, unchanged: 1200 steps, batch 8, CPU, 4 threads |

**The rule.** Choose the **lowest** weight whose validation rotation error is within **0.1°**
of the best, subject to validation position degrading no more than **2%** against 0.33.
Lowest rather than best, because a larger weight buys rotation with position and the tie-break
should favour the least disturbance to a result that already works.

The 0.1° tie tolerance was added after a four-step smoke run showed two indistinguishable
configurations being separated by their far decimals — which would have selected a weight on
noise. It was fixed before any real run, and it is the conservative direction.

**No test split was read.** A weight chosen against test data would make every later number on
this corpus untrustworthy.

---

## 3. Results

Validation floors: position 0.1605, rotation 23.51°, scale 0.1227 (A1) / 0.1194 (A3 sees a
different entity mix at its level of detail; each run records its own).

| arm | weight | position | rotation | scale | beats position floor | beats rotation floor |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A3 | 0.33 | 0.1518 | 24.47° | 0.1194 | yes | no |
| A3 | 1.0 | 0.1531 | **21.36°** | 0.1194 | yes | **yes** |
| A3 | 3.0 | 0.1529 | **20.96°** | 0.1194 | yes | **yes** |
| A3 | 10.0 | 0.1532 | **20.86°** | 0.1193 | yes | **yes** |
| A1 | 0.33 | 0.1673 | 24.48° | 0.1223 | no | no |
| A1 | 1.0 | 0.1672 | 23.73° | 0.1223 | no | no |
| A1 | 3.0 | 0.1679 | 23.51° | 0.1223 | no | no |
| A1 | 10.0 | 0.1679 | 23.48° | 0.1223 | no | marginally |

**Chosen: λ_rotation = 3.0.** Averaged over the two arms, 10.0 is better by 0.06°, inside the
0.1° tie tolerance, so the tie goes to the lower weight. Position at 3.0 is 0.1604 against the
baseline's 0.1595, a 0.56% degradation, well inside the 2% allowance.

---

## 4. What it establishes

**The weight was the binding constraint, and Change 2's rotation conclusion is superseded.**
A3 moves from 0.96° *above* the rotation floor to 2.55° *below* it. The architecture can learn
rotation; at 7% of the objective it was not being asked to.

**Position was not the price.** A3 stays comfortably below its position floor at every weight
— 0.1518 to 0.1532 against a floor of 0.1605 — and scale does not move at all. The trade-off
the 2% allowance was designed to catch did not materialise.

**Only the arm that reads the graph can spend the extra weight.** A1 has no relational
mechanism and never clears the floor by more than a rounding error, even at 10.0, while A3
clears it by 2.5° from 1.0 onward. That is exactly what the graph-conditional bound predicted:
the scene-specific rotation lives in the relationship graph, so an arm that cannot read the
graph has nothing extra to gain from caring more about rotation.

**Returns diminish quickly.** 24.47 → 21.36 → 20.96 → 20.86. Almost all of the gain arrives by
weight 1.0.

---

## 5. What it does not establish

* **Nothing confirmatory.** One seed, two arms, validation only. The confirmatory statement
  required the full matrix at the single frozen weight; §6 is that matrix, and it delivered a
  smaller margin than this screening suggested — 1.27° rather than 2.55°.
* **That 3.0 is optimal.** It is the lowest weight indistinguishable from the best of four
  screened values under a rule fixed in advance. A finer sweep was not run and is not needed
  to answer the question this study asked.
* **That the remaining 21° is irreducible.** The graph-conditional lookup reaches 11.57° on
  `test_seen`, so a model that reads the graph should in principle do better than A3 does
  here. That gap survives the confirmatory matrix and is the next open question.
* **Anything about the parent-relative cells at this weight.** `T4_rigid` and `T1_spatial`
  were screened at 0.33 only. More rotation pressure might change how much a parent's
  orientation is worth, and that has not been tested.

---

## 6. Confirmed on the test splits

The confirmatory matrix ran at the single frozen weight of 3.0: `T0_global`, five arms, three
seeds, scored on all four test splits. Artifacts in `experiments/runs/step10-change2-w3/`.

`test_seen`, rotation floor 22.65°, position floor 0.1605:

| arm | rotation at 3.0 | at 0.33 | margin | position | beats rotation floor |
| --- | ---: | ---: | ---: | ---: | --- |
| A1 | 22.74 ± 0.01 | 23.85 | −0.08 | 0.1674 | no |
| A1M | 22.73 ± 0.01 | 23.60 | −0.08 | 0.1672 | no |
| A3Lite | 22.64 ± 0.04 | 23.74 | +0.01 | 0.1597 | at the floor |
| A3L | 22.69 ± 0.01 | 23.61 | −0.04 | 0.1642 | no |
| **A3** | **21.38 ± 1.03** | 23.75 | **+1.27** | **0.1561** | **yes** |

`test_transform`, the extrapolation split, floor 35.95°: A3 reaches 35.51 (+0.45), A3Lite
35.79 (+0.16), the rest at or below the floor.

### What this confirms

**Rotation is learnable above the floor, and only by the full typed graph.** A3 clears it by
1.27° on `test_seen` and 0.45° on `test_transform`, on held-out data, at a weight chosen
without ever reading a test split. Change 2's rotation conclusion is now superseded by a
confirmatory result rather than a screening one.

**The weight helped every arm; the architecture decided who could use it.** All five improved
by about 1.1° when the rotation term went from 7% to 41% of the frame loss. Four of them
stopped at the floor. Capacity is not the discriminator either — A1M has 3.1M parameters and
lands exactly where A1's 0.56M does. Only the arm that reads the typed relational graph turns
the extra pressure into a margin, which is what the graph-conditional bound predicted: the
scene-specific rotation lives in the relationship graph.

**Position was not the price.** A3 holds 0.1561 against a 0.1605 floor, better than its own
0.1563 at weight 0.33. Scale is unchanged. The trade-off the 2% allowance existed to catch
never materialised on test data either.

### What weakens it, and must be read with it

**A3's seed spread is 1.03°, twenty times any other arm's**, from a bimodal result: two seeds
at 20.78 and 20.79, one at 22.57. All three are below the floor, so the sign is consistent,
but one clears it by only 0.08°. A single seed would have given either "a decisive 1.9° margin"
or "at the floor", and both would have been wrong.

**The margin is half what validation suggested** — 1.27° against 2.55°. Screening on one seed
overstated it, which is the expected direction for a value selected on one split.

**The instability is itself a finding.** Raising the rotation weight from 7% to 41% makes the
full typed graph better on average and much less repeatable. Whether that is an optimisation
problem, a capacity conflict between the objectives, or something about this corpus is not
answered here.

**A3's 21.38° is still far above what a lookup can do.** The graph-conditional bound is 11.57°
on `test_seen`. The model beats a lookup keyed on identity alone and remains well short of one
keyed on identity and graph — the information is there, and most of it is still unused.

---

## 7. What runs next

Three things, in the order I would do them:

1. **Seeds, for A3 at this weight.** The 1.03° spread is the weakest part of the result. More
   seeds would say whether 22.57 is a rare basin or a third of the distribution.
2. **The gap to 11.57°.** A model that reads the graph should approach a lookup that reads the
   graph. It does not. That gap is now the most informative open question about the
   architecture, and it is a question Change 2 could not have asked.
3. **The parent-relative cells at this weight.** `T4_rigid` and `T1_spatial` were only ever run
   at 0.33. More rotation pressure might change what a parent's orientation is worth, and
   ADR-STEP10-004's portability finding was measured under the old weight.

Change 2's runs at 0.33 are untouched in their own directory and remain reproducible: the
rotation weight is an explicit run parameter whose default is still 0.33.

Reproduce:

```
python -m experiments.step10.weight_study --arms A3
python -m experiments.step10.weight_study --arms A1
python -m experiments.step10.weight_study --decide-only
python -m experiments.step10.run_change2 --seeds 0 --cells T0_global --rotation-weight 3.0 \
    --out experiments/runs/step10-change2-w3
```

Artifacts: `experiments/runs/step10-weight-study/` and `experiments/runs/step10-change2-w3/`.
