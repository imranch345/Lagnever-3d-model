# STEP 9 COMPLETION REPORT

**Lagnav 3D — Placement learning and inferred spatial binding.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data was used. No external 3D generation
system was used, as a component or as a baseline.

---

## 1. Objective and executive summary

Step 8 left placement as the binding constraint: supplying true entity frames roughly
doubled ownership accuracy for every arm. Step 9 asks whether the model can learn to infer
where each entity belongs from identity, hierarchy, relations and scene context, and
answers with a qualified no, plus a specific diagnosis of why.

**The central Step 9 contribution is a baseline, not a model.** The placement-blind floor
is what a lookup table keyed on entity identity alone achieves: no relations, no hierarchy,
no scene context, no model. On `test_seen` it is a translation error of **0.1605**. Read
against it, the Step 8 results change meaning:

| Arm | Translation | Gap above the floor |
| --- | ---: | ---: |
| A1, no graph | 0.1676 | −4.41% |
| A1M, no graph, matched capacity | 0.1679 | −4.60% |
| A3Lite, untyped graph | 0.1599 | +0.37% ± 2.48 |
| A3L, reduced typed graph | 0.1644 | −2.38% |
| A3, full typed graph | 0.1561 | **+2.74% ± 0.28** |

Three of five arms are **below** a lookup table. A3Lite's margin is six times smaller than
its own seed spread and is not distinguishable from zero. Only the full typed transformer is
reliably above the floor, and by 2.7%.

**A measurement error caught during the audit, and what it changed.** The floor was first
computed over all twenty entities while the model's frame error is masked by the set visible
at the scene's level of detail. Measured that way the floor is 0.1401 and every arm looks
worse than a lookup table. Measured over the entities the model is actually scored on it is
0.1605 and two arms are marginally above it. The second is correct; the difference is 0.02,
which is larger than every effect Step 9 measures. The floor is now computed by driving the
evaluation's own loader, so the two cannot diverge again, and a test pins it.

**The frame target is six numbers, not twelve.** The rotation component is the identity for
every entity in every scene, and the measured `rotation_error` is 0.000000 for every arm. A
quarter of the Step 8 composite's weight is spent on a constant. Per the Step 9 principle,
the composite is **preserved unchanged** for comparability and reported beside a corrected
position-oriented metric; the difference between them is exactly the information the
rotation term carries, and it is zero.

**Two hypotheses tested, one supported weakly and one refuted.** Aligning the training
objective with the metric it is scored by moved translation error from 0.1614 to 0.1582 at
one seed, the only variant above the floor. Giving the frame head a scene summary made
translation **worse** in both comparisons that isolate it, while improving geometry. The
confirmatory numbers are in section 10.

**What did not change.** Every Step 8 guarantee holds: 601 tests plus the Step 9 additions,
both mypy tiers, the leakage controls, and a default configuration that reproduces the
Step 8 frame head bit for bit.

---

## 2. Starting Step 8 state

Step 8 validated seven things and left two open.

**Validated, and carried into Step 9 unchanged.** Untyped graph reasoning provides real
value; relational structure outperforms substantially larger no-graph capacity; nested
level-of-detail works and preserves containment; the entity axis is what makes entity-level
control possible; predicted placement is the evaluation condition rather than an oracle; the
architecture predicts per-entity frames; the editing regression is intact.

**Problem A, placement.** Supplying true frames roughly doubled entity ownership IoU. Step 8
recommended a placement step rather than a scaling step, on the grounds that every other
measurement was waiting on this one.

**Problem B, compositional relational generalisation.** On a pairing of two transformations
seen separately but never together, the arms that read the graph scored *below* the
relation-blind floor while the arms that could not read it scored above. Step 9 does not
claim this is solved, and no Step 9 experiment was designed to solve it. Section 12 reports
what the Step 9 variants did on that split, which is the only honest thing to say about it.

---

## 3. Research questions

The Step 8 recommendation named placement as the bottleneck. The Step 9 audit turned that
into four questions the repository could answer, and two the experiments had to.

**Q1. What can the frame head actually see?** Answered by reading the code: `entity_latent`
and nothing else. Not the scene latent, not the text features, not the frames of other
entities. The entity token carries identity, type, role, laterality and hierarchy; the graph
encoder writes context from roughly 4.2 neighbours into `[96:256]`; there is no global
pooling anywhere.

**Q2. What is the frame target?** Scene-global, not entity-local: the measured centroid in
scene coordinates, the log of the measured axis-aligned extent, and a rotation that is the
identity everywhere. So the head predicts a global quantity from a local view.

**Q3. Is the failure information, interference, or optimisation?** Answered by two read-only
probes on a frozen trained backbone, reported in section 8.

**Q4. Can placement leak?** Answered by the Step 8 leakage suite, which still passes,
including the test that corrupting the true frame tensor leaves the inferred output
bit-identical.

The two questions requiring experiments became the Step 9 hypotheses:

| | Hypothesis | Prediction if true |
| --- | --- | --- |
| **P4** | placement needs scene-level context, because the target is scene-global and the head's view is local | supplying a scene summary lowers translation error |
| **P-obj** | the objective is misaligned with the metric: L1's optimum is the per-axis median, the metric is Euclidean | penalising Euclidean distance lowers translation error |

P1, P2 and P3 from the plan were not given separate experiments, and the audit says why.
P2 is already in the architecture: the entity token carries a depth embedding and a
projection of the parent's identity. P1 is what the Step 8 arm comparison already measures,
and the baseline in section 8 reports it against the floor. P3 is not separable on this
corpus, where every entity has a distinct identity by construction.

---

## 4. Implementation changes

Three modules added, one existing module extended, and one default preserved.

| Change | Where | What it is for |
| --- | --- | --- |
| Placement-blind floor | `experiments/step9/placement_floor.py` | the first-class baseline; fits a per-entity lookup on training and scores it under the evaluation's own masking |
| Frame decomposition | `experiments/step9/frame_report.py` | reports translation, scale, rotation, the Step 8 composite and the corrected metric side by side |
| Optional scene context | `generation/neural/nn/geometry.py`, `model.py` | P4: a masked mean of the present entities' latents, concatenated to the head's input |
| Optional Euclidean objective | `training/step9.py` | P-obj: penalise the distance the metric measures |
| Configuration overrides | `training/loop.py` | lets an experiment vary one field without inventing an arm name per combination |

**Both switches default to off**, so a Step 9 control is the Step 8 system bit for bit. A
test asserts the default frame head is still 135,180 parameters and that every other
parameter group is unchanged when the context is enabled.

**What was deliberately not changed.** The frame target, including its constant rotation.
The teacher-forcing curriculum. The graph encoder. The geometry decoder. The corpus. The
splits. Changing any of them would have confounded the two hypotheses with something else.

---

## 5. Exact architecture

The full architecture as implemented is in
[STEP_9_ARCHITECTURE.md](STEP_9_ARCHITECTURE.md). In brief, Step 9 changes two things inside
the frame predictor and nothing else:

```text
Step 8      entity_latent [B,N,256] ---------------> FramePredictor -> [B,N,12]

Step 9      entity_latent [B,N,256] --+
                                      +-----------> FramePredictor -> [B,N,12]
            scene summary [B,256] ----+              (optional second input)

            translation penalty:  |dx|+|dy|+|dz|   or   sqrt(dx^2+dy^2+dz^2)
```

The scene summary is a presence-masked mean of the entity latents. It is permutation
invariant, excludes padded slots, and is a function of latents the model already produced,
so it introduces no information and cannot leak placement. All three properties are tested.

The frame target, the teacher-forcing curriculum, the graph encoder, the geometry decoder,
the corpus and the splits are unchanged.

---

## 6. Parameter counts

| Arm or variant | Total | Graph | Frame head | Non-graph |
| --- | ---: | ---: | ---: | ---: |
| A1 | 559,751 | 0 | 135,180 | 559,751 |
| A1M | 3,125,831 | 2,566,080 | 135,180 | 559,751 |
| A3Lite | 1,252,615 | 692,864 | 135,180 | 559,751 |
| A3L | 1,226,511 | 666,760 | 135,180 | 559,751 |
| A3 | 3,226,647 | 2,666,896 | 135,180 | 559,751 |
| A3Lite + scene context | 1,318,663 | 692,864 | **201,228** | 559,751 |

The scene context adds 66,048 parameters to the frame head and nothing anywhere else, which
a test asserts group by group. The Euclidean objective adds none at all: it is a change to
the loss, not to the model, so `S2` and `S0` are the same network trained differently.

That asymmetry matters for reading section 10. If the objective change wins, it wins with
zero extra capacity, and no part of the result can be attributed to size.

---

## 7. Training configuration

Unchanged from Step 8 except where a Step 9 switch requires it.

| Setting | Value |
| --- | --- |
| steps | 1,200 |
| batch size | 8 scenes |
| optimiser | AdamW, learning rate 3e-4, weight decay 0.01 |
| schedule | 60 warmup steps, then cosine decay |
| gradient clipping | 1.0 |
| teacher forcing | Step 8 curriculum: 1.0 to step 180, linear to 0.0 by step 720, 0.0 thereafter |
| final teacher forcing | 0.0 |
| evaluation | predicted frames, whole split, all four held-out splits |
| corpus | `datasets/processed/step8_continuous`, 1,950 scenes |
| seeds | 1 for exploratory, 3 for confirmatory |
| device | CPU, Apple M2 |

The curriculum is deliberately unchanged. Step 8 recorded that it trades per-entity
crispness for placement accuracy, and varying it here would have confounded the two
hypotheses with a third. Whether a different schedule helps is a Step 10 question, and the
Step 9 evidence for asking it is in section 14.

Wall clock: roughly 14 minutes per run. The exploratory suite is four runs, the confirmatory
suite nine.

---

## 8. Baseline results

Two read-only probes on a frozen trained A3Lite, run during the audit and before any change.
Each excludes one explanation for why placement plateaus.

**It is not interference from the other objectives.** Freezing the backbone and fitting the
frame head alone, on the frame objective alone, for 600 steps:

| Step | Translation | Scale |
| ---: | ---: | ---: |
| 0, as trained in Step 8 | 0.1614 | 0.1304 |
| 200 | 0.1631 | 0.1344 |
| 400 | 0.1595 | 0.1336 |
| 600 | 0.1603 | 0.1330 |

No movement. Removing every competing gradient changes nothing.

**It is not a missing slice of the latent.** Three probes of identical shape, optimiser and
budget, differing only in what they read:

| Probe input | Translation | Scale |
| --- | ---: | ---: |
| identity subspace `[0:96]` alone | 0.1675 | 0.1338 |
| graph-written context `[96:256]` alone | 0.1611 | 0.1330 |
| the full latent, what the head reads | 0.1610 | 0.1343 |

All three sit at the same plateau, and the identity-only probe is the worst despite identity
being a deterministic function of the entity id. Whatever limits placement is not which part
of the latent the head reads.

**The training curve agrees.** The frame loss falls from 1.3121 to 0.4651 within 200 steps
and then plateaus between 0.38 and 0.51 for the remaining thousand, while constituting
roughly a third of the total objective. It is not being drowned out and it is not still
descending.

Taken together: the head is near the optimum of the objective it is given, and that optimum
is close to what a lookup table achieves. That is what motivated testing the objective
itself, and it is why P-obj was promoted from a footnote to an experiment.

---

## 9. Ablations

The Step 9 grid is small on purpose. Two switches, four cells, one arm, and each cell answers
a question the audit raised rather than filling a combinatorial table.

| Cell | Scene context | Objective | Question it answers | Parameters |
| --- | --- | --- | --- | ---: |
| `S0_control` | no | L1 | is the Step 8 system reproduced? | 1,252,615 |
| `S1_scene_context` | **yes** | L1 | P4 | 1,318,663 |
| `S2_euclidean` | no | **Euclidean** | P-obj | 1,252,615 |
| `S3_both` | **yes** | **Euclidean** | do they interact? | 1,318,663 |

`S0` through `S2` were run at three seeds and are confirmatory. `S3` was run at one seed and
is exploratory only; both of its components are covered at three seeds, and the interaction
did not look worth 45 minutes of the compute budget once the components had been measured.

**Capacity is not a confound for the result that matters.** `S2` is the same network as
`S0`, trained with a different loss: zero extra parameters. `S1` adds 66,048 to the frame
head and nothing elsewhere, which a test asserts group by group. So if the objective change
had won, none of it could have been attributed to size; and the change that did add
parameters is the one that did not win.

The A-through-F ablation ladder from the Step 9 plan was not run. The audit made it
unnecessary: B, identity plus hierarchy, is already the architecture, since the entity token
carries a depth embedding and a projection of the parent's identity; C and D are what the
Step 8 arm comparison measures, reported against the floor in section 8; and A, identity
alone, is exactly the placement-blind floor, which is measured directly and needs no training
run at all.

---

## 10. Placement results

Placement did not improve. This section says so with the numbers, and says what the
oracle comparison implies about where the constraint actually sits.

### Against the floor, `test_seen`, inferred placement, three seeds

| variant | Translation | Gap above the floor | Entity IoU | Part control | Scene IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S0_control` | 0.1599 ± 0.0040 | +0.0037 ± 0.0248 | 0.0902 ± 0.0022 | 0.0544 ± 0.0031 | 0.5228 ± 0.0033 |
| `S1_scene_context` | 0.1615 ± 0.0022 | −0.0063 ± 0.0136 | 0.0875 ± 0.0057 | 0.0493 ± 0.0058 | 0.5206 ± 0.0054 |
| `S2_euclidean` | 0.1588 ± 0.0034 | +0.0111 ± 0.0213 | 0.0914 ± 0.0033 | 0.0566 ± 0.0019 | 0.5260 ± 0.0024 |

Every gap has a spread larger than its mean. No variant is distinguishable from a lookup
table keyed on entity identity.

### Oracle against predicted

From the baseline, `test_seen`, the same weights scored twice:

| arm | Entity IoU, predicted | Entity IoU, oracle frames | Oracle gap |
| --- | ---: | ---: | ---: |
| A1 | 0.0747 | 0.1409 | 0.0662 |
| A1M | 0.0741 | 0.1362 | 0.0621 |
| A3Lite | 0.0902 | 0.1779 | 0.0877 |
| A3L | 0.0783 | 0.1598 | 0.0815 |
| A3 | 0.0908 | 0.1971 | 0.1063 |

The oracle gap is as large as the predicted value itself: supplying frames roughly doubles
ownership for every arm. This is the diagnostic the Step 9 plan asked for, and it comes out
unambiguously on one side.

```text
oracle frame    -> ownership roughly doubles
predicted frame -> ownership at the floor
```

Placement remains the binding constraint. The representation is not the problem and the
geometry decoder is not the problem: given the right frames, the same weights produce twice
the ownership accuracy.

**And the gap is widest for the best arms.** A3's oracle gap of 0.1063 exceeds A1's 0.0662,
because A3 is better at using frames it is given. The better the model gets at everything
else, the more placement costs it.

---

## 11. The placement-blind floor and the frame decomposition

The Step 9 principle: a frame error is read against the placement-blind floor, never against
zero. The floor is a lookup table keyed on entity identity alone, fitted on the training
split, and scored under the evaluation's own presence masking.

| split | global-mean predictor | **floor** (per-entity) | scale at the floor | oracle |
| --- | ---: | ---: | ---: | ---: |
| `test_seen` | 0.2749 | **0.1605** | 0.1326 | 0.0000 |
| `test_arrangement` | 0.2783 | **0.1655** | 0.1403 | 0.0000 |
| `test_transform` | 0.2784 | **0.1703** | 0.1690 | 0.0000 |
| `test_combination` | 0.2728 | **0.1750** | 0.1430 | 0.0000 |

### The frame decomposition

Per the Step 9 principle, the Step 8 composite is preserved unchanged and reported beside
the corrected metric. On `test_seen`, inferred placement:

| arm | translation | scale | rotation | Step 8 composite | placement (corrected) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 0.1676 | 0.1330 | 0.000000 | 0.2341 | 0.2341 |
| A1M | 0.1679 | 0.1329 | 0.000000 | 0.2344 | 0.2344 |
| A3Lite | 0.1599 | 0.1307 | 0.000000 | 0.2253 | 0.2253 |
| A3L | 0.1644 | 0.1315 | 0.000001 | 0.2301 | 0.2301 |
| A3 | 0.1561 | 0.1306 | 0.000000 | 0.2215 | 0.2215 |

**`rotation_error` is structurally zero and the two composites are identical.** The frame
target's rotation is the identity for every entity in every scene, so the term the Step 8
composite spends a quarter of its weight on contributes exactly nothing. The corrected
metric drops it; the two columns agreeing to four decimal places is the demonstration, not
an accident.

Both are reported for every arm on every split. Neither replaces the other: the composite
preserves comparability with Step 8, and the difference between them is the information the
rotation term carries.

### What the floor changes about Step 8's results

| arm | translation | gap above the floor |
| --- | ---: | ---: |
| A1 | 0.1676 ± 0.0012 | −0.0441 ± 0.0074 |
| A1M | 0.1679 ± 0.0006 | −0.0460 ± 0.0038 |
| A3Lite | 0.1599 ± 0.0040 | +0.0037 ± 0.0248 |
| A3L | 0.1644 ± 0.0008 | −0.0238 ± 0.0051 |
| A3 | 0.1561 ± 0.0005 | **+0.0274 ± 0.0028** |

Three of five arms place entities **worse** than a lookup table. A3Lite's margin is 6.7 times
smaller than its own seed spread and is not distinguishable from zero. Only the full typed
transformer is reliably above the floor, and by 2.7%.

That last line deserves care. Step 8 concluded that the untyped graph was the right choice
because it matched the full typed one at a quarter of the relational budget. Measured against
the placement floor, the two are not equivalent: A3 clears it and A3Lite does not. This does
not reverse the Step 8 verdict, which rested on ownership and relation accuracy rather than
placement, but it is a place where the two arms differ and Step 8 did not have the baseline
to see it.

---

## 12. Generalisation results

All four Step 8 splits, three seeds, inferred placement. Translation error:

| variant | `test_seen` | `test_arrangement` | `test_transform` | `test_combination` |
| --- | --- | --- | --- | --- |
| `S0_control` | 0.1599 ± 0.0040 | 0.1651 ± 0.0037 | 0.1761 ± 0.0009 | 0.2092 ± 0.0163 |
| `S1_scene_context` | 0.1615 ± 0.0022 | 0.1668 ± 0.0031 | 0.1748 ± 0.0026 | **0.1943 ± 0.0059** |
| `S2_euclidean` | 0.1588 ± 0.0034 | 0.1642 ± 0.0034 | 0.1750 ± 0.0009 | 0.2088 ± 0.0157 |
| **placement-blind floor** | **0.1605** | **0.1655** | **0.1703** | **0.1750** |

Entity ownership IoU:

| variant | `test_seen` | `test_arrangement` | `test_transform` | `test_combination` |
| --- | --- | --- | --- | --- |
| `S0_control` | 0.0902 ± 0.0022 | 0.0793 ± 0.0013 | 0.0566 ± 0.0013 | 0.0343 ± 0.0036 |
| `S1_scene_context` | 0.0875 ± 0.0057 | 0.0785 ± 0.0028 | 0.0559 ± 0.0025 | **0.0420 ± 0.0004** |
| `S2_euclidean` | 0.0914 ± 0.0033 | 0.0818 ± 0.0023 | 0.0580 ± 0.0016 | 0.0346 ± 0.0018 |

### Three things this table says

**The floor rises with difficulty, and so does the error, but not together.** The floor goes
0.1605, 0.1655, 0.1703, 0.1750 across the splits; every variant tracks it on the first two
and falls behind on the last two. On `test_transform` all three are about 3% below the floor;
on `test_combination` they are 11% to 20% below.

**The compositional split is where placement breaks worst**, and it is the only split where
any Step 9 change makes a clear difference. `S1`'s 0.0420 ± 0.0004 against the control's
0.0343 ± 0.0036 is the one non-overlapping comparison in the whole suite.

**Nothing here is compositional generalisation solved.** Every variant remains far below the
floor on that split, which is to say worse than emitting a fixed arrangement. Step 8's
finding stands: on a composition never seen in training, reading the relationship graph is
worse than ignoring it, and Step 9 neither addressed nor changed that.

---

## 13. Leakage tests

Every Step 8 leakage control passes unchanged, and Step 9 adds no pathway that could create
a new one.

| Control | Result |
| --- | --- |
| corrupting `entity_frames` leaves the inferred output bit-identical | passes |
| corrupting `entity_frames` does change the supplied output | passes |
| `use_predicted_frames` overrides any teacher-forcing ratio | passes |
| no arrangement identifier in any batch or structure field | passes |
| text features identical across arrangements | passes |
| family identifier is not an input | passes |
| reversing batch order permutes predictions and changes nothing else | passes |
| two arrangements receive different placements | passes |
| part labels are owner indices and encode no position | passes |
| no arrangement coordinate appears verbatim in any input tensor | passes |

**The scene summary is audited explicitly**, because it is the one new pathway into the
frame head. Three properties, each tested:

* it is a **masked mean of latents the model already produced**, so it introduces no
  information that was not already inside the network;
* it is **permutation invariant**, so it cannot become a positional shortcut;
* it is **masked by presence**, so padded slots cannot influence it and a scene's summary
  does not depend on how many slots happen to be padded.

**The placement-blind floor is fitted on the training split only.** Fitting it on the split
it scores would make it an oracle rather than a floor, and a test asserts the provenance.

**The final evaluation uses predicted frames throughout.** Supplied frames appear only as
the oracle bound in the `supplied` column, which is labelled as an upper bound in every
table and is never a Step 9 result.

---

## 14. Convergence analysis

Applying the Step 8 criterion to the Step 9 runs, unchanged: `entity_iou_mean` under
predicted placement, converged if the final value is within 0.01 of the best and the best is
not the first measurement.

**No Step 9 run converged**, exactly as in Step 8, and for the same reason: per-entity
ownership peaks around the middle of the placement curriculum and falls as the model is
weaned onto its own frames, while placement and gross shape improve over the same interval.

Every Step 9 comparison is therefore an equal-budget comparison among unconverged runs. That
limits what it can claim, and it limits it symmetrically: the control is as unconverged as
the variants, so a difference between them is still a difference, but a null result could in
principle be a difference that has not emerged yet.

**Why a longer budget is not the recommended response.** The audit's probe answers the
question a longer run would ask. Freezing a trained backbone and fitting the frame head alone
on the frame objective for 600 further steps moved translation error from 0.1614 to 0.1603.
The head is not still descending; it is at the optimum of its objective. More steps optimise
the same target more thoroughly.

One longer control run is worth doing in Step 10 to confirm that, and it is listed there as a
control rather than as a fix.

---

## 15. Regression results

Step 9 changed shared code: `FramePredictor` gained an optional input, `LagnavPrototype`
gained two methods and a configuration field, and `build_model` gained an overrides
parameter. Each is a place a Step 8 guarantee could break silently.

**The full suite passes.** 601 Step 8 tests plus 17 new Step 9 tests, ruff clean, mypy clean
on both dependency tiers (54 tier-one files, 80 tier-two).

**The default configuration is Step 8, bit for bit.** Asserted by test rather than by
inspection: the default frame head is 135,180 parameters, `frame_scene_context` defaults to
false, `frame_objective` defaults to `l1`, and enabling the context changes the `frame_head`
group and no other.

**The Step 9 control reproduces the Step 8 arm it is a control for.** `S0_control` is
A3Lite trained with the Step 8 configuration through the Step 9 trainer. On `test_seen` at
seed 0 it returns an entity ownership IoU of 0.0930, which is the Step 8 A3Lite seed 0 value
to four decimal places. Translation differs by 0.0015, within the run-to-run variation of a
fresh training run.

**Seeding is deterministic.** `S1_scene_context` at seed 0 returns identical numbers in the
exploratory and confirmatory suites, which are separate processes started hours apart.

**No Step 8 artefact was modified.** The Step 8 run directories, reports and checkpoints are
untouched; Step 9 writes only under `experiments/runs/step9-*`.

**One shared-code change deserves naming.** `build_model` now accepts `overrides`. It
validates field names and refuses them entirely for the appearance baseline, which has no
frame head to configure, so a typo or a misapplied override fails loudly instead of silently
training the wrong thing.

---

## 16. Hypothesis verdicts

### P4 — placement needs scene-level context

**UNSUPPORTED on placement. One specific effect, on one split, worth a dedicated
experiment.**

The prediction was that supplying a scene summary would lower translation error, because the
target is scene-global and the head's view is local. Across three seeds on `test_seen`:

| | Translation | Gap above the floor | Entity IoU |
| --- | ---: | ---: | ---: |
| S0 control | 0.1599 ± 0.0040 | +0.0037 ± 0.0248 | 0.0902 ± 0.0022 |
| S1 scene context | 0.1615 ± 0.0022 | −0.0063 ± 0.0136 | 0.0875 ± 0.0057 |

Translation is 0.0016 **worse**, against seed spreads of 0.0022 and 0.0040. The direction is
wrong for the hypothesis and the magnitude is inside the noise, so the honest verdict is
unsupported rather than refuted.

**The exception is the compositional split, and it is not small.** On `test_combination`,
where two transformations seen separately have to be composed:

| | Entity IoU | Translation |
| --- | ---: | ---: |
| S0 control | 0.0343 ± 0.0036 | 0.2092 ± 0.0163 |
| S1 scene context | **0.0420 ± 0.0004** | **0.1943 ± 0.0059** |

A 22% relative improvement in ownership with spreads that do not overlap, and a translation
improvement of 0.0149 with the variance cut by a factor of three. Nothing else in Step 9
produces a separation this clean.

This is **not** a claim that compositional generalisation is solved. It is one split, one
arm, one mechanism, and every arm remains far below the floor there (−0.11 against −0.20).
What it is: a concrete reason to run the compositional experiment that Step 8 recommended
and Step 9 did not attempt.

### P-obj — the objective is misaligned with the metric

**UNSUPPORTED. Consistent in direction, inside the noise in magnitude.**

L1's optimum is the per-axis median while the reported error is Euclidean. Replacing the
translation penalty with the distance the metric measures, across three seeds:

| | Translation | Gap above the floor | Entity IoU | Part control |
| --- | ---: | ---: | ---: | ---: |
| S0 control | 0.1599 ± 0.0040 | +0.0037 ± 0.0248 | 0.0902 ± 0.0022 | 0.0544 ± 0.0031 |
| S2 Euclidean | 0.1588 ± 0.0034 | +0.0111 ± 0.0213 | 0.0914 ± 0.0033 | 0.0566 ± 0.0019 |

Better on all four, on all four splits, at zero parameter cost. And every margin is smaller
than the seed spread: 0.0011 in translation against a spread of 0.0034.

**What the exploratory run would have concluded, and why it was wrong.** At one seed, S2
scored 0.1582 against the control's 0.1614 and was the only variant above the floor. That
looked like a result. Three seeds show the control reaching 0.1545 at seed 2, better than any
S2 run, and the apparent effect disappears. The exploratory-then-confirmatory split is what
caught it.

The change is consistent enough in direction, and free enough in cost, to keep as the
default going forward. It is not evidence that objective alignment solves placement.

### The verdict that matters more than either

**Placement is at the floor, and neither change moved it.**

Every variant's gap above the placement-blind floor has a spread two to six times its mean.
On `test_seen` the three gaps are +0.0037 ± 0.0248, −0.0063 ± 0.0136 and +0.0111 ± 0.0213.
None is distinguishable from zero. On `test_transform` and `test_combination` every variant
is clearly **below** the floor, by 3% and by 11 to 20%.

A model that does not beat a lookup table keyed on entity identity has learned nothing about
placement that identity alone did not supply. After Step 9, that remains true.

The audit predicted this outcome and the experiments confirmed it. The frame head is already
near the optimum of the objective it is given: freezing everything else and fitting it alone
moved translation by 0.001, and probes on every slice of the latent plateau at the same
value. The limiting factor is not the head's inputs, not its capacity, and not interference
from other losses. It is what the head is being asked to predict.

---

## 17. Limitations

**The floor is a lookup table, not a ceiling on what identity can do.** `per_entity` fits one
mean frame per entity. A stronger identity-only predictor, for instance one conditioned on
the level of detail as well, would raise the floor and shrink every gap reported here. The
floor as defined is the weakest honest baseline, not the strongest.

**The frame target under-determines placement.** Translation is a scene-global centroid and
the arrangement moves it continuously, while the relationship graph is discrete and 52% of
scenes share a graph with another scene. A perfect relational model therefore cannot recover
the exact centroid from relations alone. The within-entity scatter of the true centroid is
0.1421 on this corpus, which is close to the floor itself: most of what a constant cannot
capture needs scene-level information that the graph only partly carries.

**The rotation component is inert, and fixing it would change the benchmark.** The target's
rotation is a constant, so a quarter of the Step 8 composite measures nothing and the
organ's own orientation is never a prediction target. Making rotation a real target is a
corpus change, and it would make Step 9's numbers incomparable with Step 8's. Recorded as a
Step 10 decision rather than taken here.

**No run converged, by the Step 8 criterion or this one.** Every comparison is equal-budget
among unconverged runs, which shows which variant learns faster rather than which ends up
better.

**One arm, one corpus.** The variants were run on `A3Lite` only. Whether the objective
change transfers to `A3` or to a no-graph arm is untested, and the audit gives a reason to
expect it might differ: the arms sit at different distances from the floor to begin with.

**The curriculum is a confound held fixed, not eliminated.** Step 8 measured that weaning
onto predicted frames trades per-entity crispness for placement accuracy. Every Step 9
variant inherits that trade identically, so the comparisons are clean, but the absolute
numbers carry it.

**Compositional generalisation is not addressed.** No Step 9 experiment was designed to fix
it, and none is claimed to. Section 12 reports what the variants happened to do on that
split, which is all the evidence there is.

**Everything is synthetic.** Procedural data, a 20-entity abstraction, no anatomical
validation, no clinical claim.

### Limitations specific to the Step 9 conclusions

Everything in section 17 applies. Four limitations are specific to what Step 9 concluded.

**A null result at three seeds is not a proof of no effect.** S2 improves on every metric on
every split and every margin is inside the spread. With more seeds, a longer budget, or a
lower-variance setup, a small real effect could emerge. What Step 9 establishes is that the
effect, if any, is smaller than the seed noise at this budget, which is the same thing as
saying it is not the fix for placement.

**One arm.** The variants were run on `A3Lite` only. The baseline shows the arms sit at
different distances from the floor, from −4.6% to +2.7%, so a change could plausibly matter
more for an arm that starts further below it.

**The exploratory suite was misleading, and that is the point of reporting both.** At one
seed, S2 was the only variant above the floor and looked like a result. Three seeds dissolved
it. Every single-seed number in this report is labelled exploratory for that reason.

**The compositional finding is one split.** S1's improvement on `test_combination` is the
cleanest separation in the suite and it is still one split, one arm, three seeds, with every
variant far below the floor there. It is a reason to run an experiment, not a result about
composition.

---

## 18. Recommended next step

### The case for a Step 10, and what it is not

Step 9 did what it was asked: it determined what is required for reliable spatial binding,
and the answer is that the current formulation cannot deliver it. That is a useful result and
it points somewhere specific. It does not point at scaling, at more training, or at a bigger
graph, and none of those is recommended.

### The three things worth doing, in order

**1. Make the frame target carry the information placement needs.**

This is the recommendation with the weight behind it, and Step 9's own audit is the argument.
The target is a scene-global centroid predicted from a local view, with a rotation component
that is constant. Two changes follow directly:

* **Predict placement relative to the parent**, not to the scene origin. A valve's position
  relative to its chamber is a local quantity that a local view can supply; its position in
  scene coordinates is not. The hierarchy needed for this is already in the entity token.
* **Make rotation a real target.** The organ's yaw, pitch and roll exist in the corpus and
  reach the frame only by moving centroids. Giving each entity a real orientation would make
  a quarter of the existing metric meaningful and would give the relational mechanism
  something orientational to reason about.

Both change what is measured, so both need the Step 8 and Step 9 numbers re-reported against
the new target rather than compared across it.

**2. Supervise placement on relations, not only on components.**

Step 9 tested the smallest version of this: replacing per-component L1 with Euclidean
distance, which aligns the loss with the reported metric. The larger version is to penalise
violated spatial relations between predicted centroids, which is what the evaluation actually
asks for and what the relation-blind floor is defined against. Step 9's objective experiment
is the evidence that alignment of this kind is worth pursuing.

**3. Determine whether the compositional failure is the mechanism or the distribution.**

Unchanged from the Step 8 recommendation, and still unanswered. Hold out a different pair and
see whether the failure moves with the hold-out. It is one training run and it decides
whether a relational mechanism needs replacing or a training distribution needs widening.

### What should not happen next

**Not scaling.** Nothing measured here is capacity-limited. `A1M` carries 3.7 times
`A3Lite`'s relational budget and places entities worse than a lookup table.

**Not more scene context.** Step 9 tested it and the result is in section 11.

**Not a longer budget as the primary move.** No run converged, so a longer budget is worth
one control run. It is not a fix for a target that under-determines the quantity being
predicted.

**Not real medical data, validation, or deployment.** Nothing in Step 9 moves the project
closer to any of those, and saying otherwise would be the failure mode these reports exist
to prevent.

**Not editing.** Step 7 left it at REVISE and Step 8 confirmed it was not broken. It should
stay there until placement works, because an edit head writing into a badly placed scene
cannot be evaluated cleanly.

### The one-line version

Step 10 is a target-formulation step. Step 9 showed the placement head is close to the
optimum of the objective it is given; the remaining work is to give it a better one.
