# Step 15 — The seed mode: report

**Preregistered in** [`STEP_15_SEED_MODE_PLAN.md`](STEP_15_SEED_MODE_PLAN.md), written before any
new run was launched.

**Status: the outcome is bimodal, confirmed on ten seeds.** 6 graph-using, 3 non-using, 1
intermediate. Nothing was intervened on; only the seed varied.

**`test_splits_read: []`.**

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Result

Ten seeds at the frozen A3 configuration, λ=3.0, validation only. Graph usage is the Step 11
measure: the rotation cost of removing every relationship at evaluation.

| seed | validation rotation | rotation with no graph | graph usage | band |
| ---: | ---: | ---: | ---: | --- |
| 0 | 20.96 | 23.55 | **2.59** | graph-using |
| 1 | 23.43 | 23.80 | **0.38** | non-using |
| 2 | 21.01 | 23.52 | **2.51** | graph-using |
| 3 | 22.17 | 23.47 | **1.30** | *intermediate* |
| 4 | 23.56 | 23.85 | **0.29** | non-using |
| 5 | 23.53 | 23.62 | **0.09** | non-using |
| 6 | 20.97 | 24.79 | **3.82** | graph-using |
| 7 | 20.96 | 23.22 | **2.26** | graph-using |
| 8 | 20.98 | 23.66 | **2.67** | graph-using |
| 9 | 20.99 | 23.51 | **2.52** | graph-using |

Sorted graph usage, with the interior gaps:

```
0.09  0.29  0.38  │      1.30      │  2.26  2.51  2.52  2.59  2.67  3.82
      ├──── non-using ────┤        ├──────── graph-using ────────┤
gaps:  0.19  0.09   0.93     0.96    0.25  0.01  0.07  0.08  1.15
```

The two largest interior gaps (0.93 and 0.96) sit immediately either side of the single
intermediate seed. The bands are separated by roughly a degree of empty space on both sides.

**Verdict, by the preregistered rule** (bimodal only if the intermediate band holds at most 2 of
10 and both outer bands are occupied): **BIMODAL**.

## 2. The two modes are sharp attractors

| band | n | validation rotation | std | range |
| --- | ---: | ---: | ---: | --- |
| graph-using | 6 | **20.978°** | **0.021** | 20.96 – 21.01 |
| non-using | 3 | **23.506°** | 0.070 | 23.43 – 23.56 |
| intermediate | 1 | 22.17° | — | — |

Two things stand out.

**The graph-using mode is extraordinarily tight: std 0.021° across six independent seeds.** Six
different initialisations and six different data orders converge to within a fiftieth of a
degree of each other.

**The non-using mode sits exactly on the identity-only floor.** Its mean is 23.506° against the
measured validation identity-only lookup of **23.51°**. To three significant figures, a
non-using seed has learned the identity lookup and nothing else — not approximately, but to the
precision of the measurement.

## 3. Graph usage and accuracy are separable

`usage_and_rotation_monotone: false`, and the reason is not noise.

Within the graph-using mode, graph usage spans **2.26° to 3.82°** — a factor of 1.7 — while
validation rotation spans **20.96° to 21.01°**, a span of 0.05°. Seed 6 relies on the graph 69%
more than seed 7 and is 0.01° worse.

**So once a seed is in the graph-using mode, additional reliance on the graph buys no additional
accuracy.** Whatever the mode consists of, it is reached in an all-or-nothing way and then
saturates. "How much does this model use the graph" and "how accurate is it" are two questions
with different answers, and every earlier step used the first as a proxy for the second.

## 4. What ten seeds change about the earlier numbers

| | 3 seeds (Steps 10–14) | 10 seeds |
| --- | ---: | ---: |
| mean validation rotation | 21.80° | **21.86°** |
| std | 1.41 | **1.20** |
| graph usage, mean | 1.83° | **1.84°** |

The headline figures barely move, which is worth stating plainly: **the three-seed estimates were
not biased.** What three seeds could not deliver was the *shape*, and the shape is what every
interpretation since Step 11 rested on.

The 1.0° decision threshold used throughout Steps 11–14 is also vindicated as a choice — it
happens to fall inside the empty gap between the two modes, so no earlier verdict is sensitive
to it.

## 5. Predictions, scored as they fell

**Prediction 1 — bimodal with a nearly empty middle. Supported.** 1 of 10 intermediate, with
~0.95° of empty space on each side of it.

**Prediction 2 — the non-using mode is a minority, roughly 1 in 3. Supported.** 3 of 10, or 4 of
10 counting the intermediate seed as a partial failure.

**Prediction 3 — every seed sits at the identity floor at step 400, and those that leave do so by
step 800. First half supported, second half FALSIFIED.**

| seed | step 400 | step 800 | step 1200 | escaped by |
| ---: | ---: | ---: | ---: | --- |
| 0 | 23.75 | 21.14 | 21.07 | 800 |
| 1 | 23.71 | 23.66 | 23.57 | never |
| 2 | 24.03 | 21.21 | 21.13 | 800 |
| 3 | 23.89 | 23.72 | 22.27 | **1200** |
| 4 | 23.88 | 23.63 | 23.67 | never |
| 5 | 23.73 | 23.47 | 23.65 | never |
| 6 | 23.86 | 21.07 | 21.08 | 800 |
| 7 | 23.80 | 21.72 | 21.08 | 800 |
| 8 | 23.62 | 21.14 | 21.09 | 800 |
| 9 | 23.84 | **23.54** | 21.11 | **1200** |

All ten seeds begin at the identity floor — that part holds without exception, and it means
**every run starts in the non-using mode and some escape it.** The escape is not a divergence
from a common start; it is a departure from a common floor.

But the departure is *not* confined to steps 400–800. **Seed 9 was still at 23.54° at step 800 —
indistinguishable from the seeds that never escape — and finished at 21.11°, a full member of the
graph-using mode.** Seed 3 also moved late, and incompletely.

This falsifies the free finding from the three-seed manifests that motivated the prediction. The
three-seed sample happened to contain only early escapers. **A seed sitting on the floor at step
800 has not necessarily failed**, which matters directly: any future study that judges the mode
from a mid-training checkpoint would have misclassified 2 of 10 seeds here, and any intervention
evaluated at step 800 would have been credited or blamed wrongly.

## 6. Integrity

* All four prior freeze manifests verified **CLEAN** before and after.
* The survey harness reproduces the three known seeds exactly — 2.59, 0.38, 2.51 — which was a
  preregistered stop condition.
* The loader asserts `graph_recurrence == 1`, `frame_scene_context is False`,
  `relation_values is False` and `rotation_loss_weight == 3.0` for every seed before scoring, so
  the ten runs cannot silently differ in anything but the seed.
* All ten seeds scored through one code path, not read from their own training records.
* Seeds 3–9 trained with `--splits validation`; every run record carries
  `protocol.splits == ["validation"]`.

## 7. Limitations

* Ten seeds establish a shape and a rough base rate, not a precise one: 3 of 10 has a wide
  confidence interval, and the single intermediate seed could be a third mode, a slow escaper
  caught mid-transition, or noise. The trajectory data suggests the second — seed 3 was still
  moving at step 1200.
* Graph usage is measured by removing *all* relationships. Step 13 found no individual typed
  graph matters, so this is the only available aggregate, but it is one number standing in for a
  structured thing.
* Only 3 validation points per run were recorded (steps 400, 800, 1200), which is why §5 can
  bound the escape to "between 800 and 1200" and no more precisely.
* Everything is one corpus, one scale, 1200 steps. Synthetic throughout.
* **No mechanism is offered.** Step 15 establishes that the bimodality is real, sharp, and
  partly late-deciding. It does not explain what distinguishes the two basins.

## 8. Recommended next step

The bimodality is now established rather than inferred, and two of its properties are new
constraints on any explanation:

1. **Every seed starts in the non-using mode** and 60–70% escape. The question is not "why do
   some seeds fail" but "what triggers the escape".
2. **Escape can happen as late as the final third**, so whatever decides it is not a property of
   initialisation alone.
3. **The escape is discrete** — the graph-using mode is a 0.021°-wide attractor — and additional
   graph reliance past the threshold buys nothing.

Together these describe something closer to a phase transition than to a gradual fit, which
points at the training trajectory rather than the architecture, the objective, or the
initialisation. Step 14 already showed the objective would pay for the transition; Step 15 shows
most runs eventually collect, some late, some never.

The cheapest next diagnostic that could identify the trigger is **dense trajectory
instrumentation on the existing configuration**: re-run a small number of seeds — including one
known escaper (9), one known non-escaper (5) and the intermediate (3) — recording graph usage
every 50 steps rather than every 400. That would locate the transition to within a few dozen
steps and show whether it is abrupt or gradual, at the cost of three runs and no change to
anything.

Two things worth ruling out cheaply at the same time, both pure diagnostics:

* **Is the escape data-order or initialisation?** The seed controls both. Running one escaper's
  initialisation with a non-escaper's data order would separate them. This is a configuration
  change to the *runner*, not the model, and needs preregistering as its own comparison.
* **Does the late escaper keep improving?** Seed 9 escaped at the very end and seed 3 was still
  moving. Extending only those two beyond 1200 steps would say whether 1200 is simply too short
  for some basins — but that changes the training budget and so is a different experiment,
  **not** something to fold into a frozen-configuration comparison.

**Not started.** No intervention, no architecture change, no new loss, no budget change.
