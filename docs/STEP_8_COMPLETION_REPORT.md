# STEP 8 COMPLETION REPORT

**Lagnav 3D — Representation stress test, predicted placement and relational inference.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data was used. No external 3D generation
system was used, as a component or as a baseline.

---

## 1. Executive summary

Step 8 removed the shortcut that made Step 7's central question unanswerable, replaced the
expensive relational mechanism with a cheap one, and made the level-of-detail prefixes
behave as a nested representation. Three of the five hypotheses pass, one passes partially,
and one fails in a specific and informative way.

**The headline condition throughout is predicted placement.** No result in this report was
produced with ground-truth entity frames supplied. Frames are reported only as an oracle
upper bound, and a test asserts that corrupting the true frame tensor leaves the inferred
output bit-identical.

| | Hypothesis | Verdict | Strength |
| --- | --- | --- | --- |
| H1 | untyped graph attention gives useful relational reasoning at the A3L budget | **PASS** | STRONG |
| H2 | the model can infer placement rather than being handed frames | **PARTIAL** | MODERATE |
| H3 | one representation produces genuinely nested LOD1 in LOD2 in LOD3 | **PASS** | STRONG |
| H4 | the entity axis remains fundamental once placement is inferred | **PASS** | STRONG |
| H5 | relational reasoning generalises to unseen configurations | **MIXED** | see below |

### What is new and holds up

**Untyped graph attention at 692,864 parameters beats a capacity-matched no-graph arm
holding 3.7 times that budget**, on every geometry metric across every split. Relational
structure beats parameters. The typed machinery Step 7 marked REMOVE stays removed: the
untyped encoder is better on ownership at every split than the reduced typed one, which has
three times its seed spread.

**The prefixes now nest.** Containment is exactly 1.0000 for every arm, preservation 0.967
to 0.975, and against a disjoint-slice control on the same weights the nested prefix wins on
every measure in every arm. This reverses Step 7's finding that the finer prefixes were read
and made things worse.

**Placement is the binding constraint, and now it is measured.** Supplying true frames
roughly doubles entity ownership IoU for every arm. More than half of the achievable
ownership accuracy is currently lost to placement error.

### What does not hold up, and is the more useful half

**Relational generalisation is narrow.** It works across a continuum the model has seen:
on the interpolation hold-out all three graph arms clear the relation-blind floor while both
no-graph arms sit 0.26 below it. It fails past the edge of the training range: on the
extrapolation hold-out every arm is far below the floor.

And on a composition of two transformations seen only separately, **reading the graph is
worse than ignoring it**. The two arms that cannot read the graph score above the floor;
all three that can score below it. A model that ignores relations emits an average
arrangement, which satisfies more measured relations than the specific wrong arrangement the
graph drives the relational arms toward.

**No run converged.** All fifteen are classified not converged by the criterion, because
per-entity ownership peaks mid-curriculum and then declines: it falls in every one of the
fifteen runs, while frame error improves in every one of them. Every comparison here is equal-budget among unconverged runs, which shows
which arm learns faster rather than which ends up better.

### Two measurement problems found and corrected before reporting

The Step 7 relation-blind floor does not transfer. Recomputed for this corpus it is 0.632
to 0.828 depending on the split, against Step 7's single 0.8878, and the discriminative
headroom roughly doubled.

Level-of-detail IoU is not comparable across levels. A finer level exposes more entities, so
its target covers 55% of sampled points against 14% at the coarsest, and a model that calls
everything occupied scores a raw detail gain of about +0.41 while adding nothing. Every
level number in this report is also given corrected for density, and the correction scores
the trivial predictor at exactly 0.000.

### The recommendation

Step 9 should not scale. It should fix placement, which every measurement here identifies
as the limiting factor, and it should test whether compositional generalisation can be
trained for rather than hoped for. Section 26 gives the specific experiments.

---

## 2. Starting state

Step 7 ended with five architectural findings and three recommendations. The findings:

* the Anatomical World Representation and the entity axis carry most of the value; an
  appearance baseline at the same parameter budget reached 0.0016 part control against
  0.72;
* per-entity geometry token blocks buy locality; scoping an edit to one block gave
  locality 130.6 against 4.72 for the identical head unscoped;
* the partitioned typed graph transformer cost 2,666,896 parameters, 86% of the model, and
  a version 7.2 times smaller matched it;
* relation **types** were close to inert: flipping every spatial relation to its opposite
  moved entity ownership IoU by 0.0002, while removing the same edges cost 0.0406;
* under inferred placement every arm scored **below** the relation-blind floor, meaning a
  fixed arrangement-blind guess beat every trained model.

The recommendations, which define Step 8:

1. replace the large partitioned graph transformer with untyped graph attention at roughly
   the A3L budget;
2. train using predicted placement rather than relying on ground-truth frames;
3. make the level-of-detail prefixes genuinely nested.

Step 7 also left a warning about its own corpus: four arrangements is a small set, and a
model could learn to classify which of four it is looking at without learning anything
general about relations. Step 8 addresses that first, because none of the three
recommendations can be evaluated on a corpus that permits the shortcut.

---

## 3. Hypotheses

| | Hypothesis | How it is judged |
| --- | --- | --- |
| H1 | untyped graph attention gives useful relational reasoning at roughly the A3L budget | `A3Lite` against `A1M` under inferred placement, with `A1M` holding 3.7 times the relational parameter budget |
| H2 | the model can infer placement from the representation rather than being handed frames | spatial relation accuracy under inferred placement, against the recomputed relation-blind floor |
| H3 | one geometry representation produces nested outputs, LOD1 in LOD2 in LOD3 | containment and preservation between adjacent prefixes, together with detail gain |
| H4 | the entity axis remains the mechanism for independent control once placement is inferred | the A3Lite / A4 / A0 ladder under inferred placement |
| H5 | relational reasoning generalises to unseen configurations rather than recognising memorised ones | four held-out splits, each removing a different thing |

---

## 4. Changes implemented

| Change | Where | Why |
| --- | --- | --- |
| Continuous arrangement space, nine axes | `datasets/whole_organ/arrangement.py` | four labels can be classified; a continuous space cannot |
| Corpus with four held-out generalisation splits | `datasets/whole_organ/continuous_corpus.py` | H5 needs hold-outs that each remove one thing |
| `UntypedGraphEncoder`, 692,864 parameters | `generation/neural/nn/graph_lite.py` | Step 7 measured that relation types were unread |
| `FramePredictor`, 135,180 parameters | `generation/neural/nn/geometry.py` | Step 7's head was 3,084 parameters and was never exercised |
| Teacher-forcing curriculum to zero | `training/step8.py` | placement must be learned, not supplied |
| Containment and preservation objectives | `generation/neural/nn/nested_lod.py` | Step 7's prefixes were used and made things worse |
| Frame metrics and per-group breakdown | `experiments/step8/frame_metrics.py` | placement needed metrics of its own |
| Recomputed relation-blind floor | `experiments/step8/relation_floor.py` | the Step 7 floor belongs to the Step 7 corpus |

---

## 5. Dataset design

`datasets/processed/step8_continuous`

Each scene draws an arrangement from a nine-dimensional space: two binary axes that are
genuinely binary facts about a heart, and seven continuous ones.

| Axis | Kind | What it moves |
| --- | --- | --- |
| `mirror` | binary | reflects the organ across the midline |
| `transpose` | binary | swaps which ventricle each great artery leaves |
| `yaw`, `pitch`, `roll` | continuous | the three rotations |
| `septal_shift` | continuous | lateral position of the septum between the ventricles |
| `av_shift` | continuous | height of the atrioventricular plane |
| `apex_swing` | continuous | anterior-posterior lean of the apex |
| `chamber_asymmetry` | continuous | relative size of the two ventricles |

Every axis changes where structures sit **relative to one another**, so it changes the
relations measured from the finished organ. None changes which entities are present, and
none is visible in the text features.

`chamber_asymmetry` is the clearest case: across its range the left-to-right ventricle
volume ratio moves from 0.47 to 3.30, which moves the septum between them and the valves
that sit on their surfaces.

| Property | Step 7 corpus | Step 8 corpus |
| --- | --- | --- |
| scenes | 1,600 | 1,950 |
| arrangements | 4 discrete labels | continuous, 9 axes |
| distinct relationship graphs | 164 | 266 |
| splits | 3 | 6 |
| generalisation tests | none | 4 |

Two scenes never share an exact arrangement; a test asserts it.

---

## 6. Leakage controls

Four held-out splits, each removing one thing, with the boundaries defined by a single
function that the generator and the tests both read:

| Split | What is held out | What it asks |
| --- | --- | --- |
| `test_seen` | the **families**, not the arrangements | does it generalise to an unfamiliar organ shape? |
| `test_arrangement` | a band of `yaw` and of `septal_shift` removed from training | does it interpolate into a hole? |
| `test_transform` | rotations larger than any in training | does it extrapolate past the edge? |
| `test_combination` | `mirror` and `transpose` together, never co-occurring in training | does it compose two transformations seen only separately? |

The three arrangement hold-outs deliberately **reuse training families**, so a failure
there is about the arrangement and not about an unfamiliar organ. `test_seen` is the
reverse: in-distribution arrangements, held-out families.

`test_combination`'s continuous coordinates are constrained to stay in distribution, so it
isolates the discrete pair rather than confounding "this combination is new" with "this
rotation is new".

Controls, each covered by a test:

* every split's scenes fall in that split's region, recomputed from the scenes rather than
  trusted from the manifest;
* no training arrangement falls in a held-out band;
* held-out families never appear in training;
* the nearest training arrangement to any held-out scene is at non-zero distance, and for
  the combination hold-out is unreachable;
* `test_transform` rotations exceed the training maximum;
* entity presence, text features and slot ordering are identical across every arrangement.

Predicted placement is only a meaningful test if placement cannot be obtained another way.
Ten adversarial tests close the routes the brief named, each trying to get the answer by a
route the architecture should not offer:

| Route | Test result |
| --- | --- |
| an arrangement identifier in the batch | no field of the batch or structure names one |
| text features that encode the arrangement | two arrangements give bit-identical text features |
| the family identifier | not an input; two families give identical entity inputs |
| dataset ordering | reversing the batch permutes the predictions and changes nothing else |
| a fixed entity slot index | two arrangements receive different placements |
| hidden ground-truth frame tensors | corrupting `entity_frames` leaves the inferred output **bit-identical**, while changing the supplied output |
| labels that encode position | part labels are owner indices, nothing more |
| arrangement coordinates leaking into the input | no coordinate appears verbatim in any input tensor |

The frame test is the decisive one, and it has a counterpart: if corrupting the true
frames changed nothing in **either** condition, the first test would prove nothing.

---

## 7. Model architectures

| Arm | Relational mechanism | Total | Graph | Non-graph | Trainable |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | none, and no entity axis | 3,103,894 | 0 | 3,103,894 | all |
| A1 | none | 559,751 | 0 | 559,751 | all |
| A1M | none; per-entity depth instead | 3,125,831 | 2,566,080 | 559,751 | all |
| A3Lite | untyped graph attention | 1,252,615 | 692,864 | 559,751 | all |
| A3L | reduced typed transformer | 1,226,511 | 666,760 | 559,751 | all |
| A3 | full typed transformer | 3,226,647 | 2,666,896 | 559,751 | all |
| A4 | untyped graph, shared geometry | see report | 692,864 | shared | all |

Every arm carries the same 135,180-parameter frame predictor and the same 559,751
parameters of non-relational machinery, so the arms differ only in the relational
mechanism and, for A0 and A4, in the geometry factorisation.

**The comparison that matters is `A3Lite` against `A1M`.** `A1M`'s 2,566,080 parameters
are per-entity depth: information cannot move between entities. It is 3.7 times `A3Lite`'s
relational budget. If `A3Lite` wins, it is not winning on capacity; if `A1M` wins, extra
parameters beat relational structure and the graph should go.

`A3Lite`'s encoder is 692,864 parameters against the reduced typed encoder's 666,760, a
3.9% difference, so `A3Lite` against `A3L` is a question about relation types at a matched
budget.

---

## 8. Parameter counts

See the table in section 7. Non-graph parameters are identical across arms at
559,751, and every arm carries the same 135,180-parameter frame predictor, so the
arms differ only in the relational mechanism and, for A0 and A4, in the geometry
factorisation.

---

## 9. Training configuration

| Setting | Value |
| --- | --- |
| steps | 1,200 |
| batch size | 8 scenes |
| scene query points | 256 per scene |
| entity query points | 24 per entity |
| forward passes per step | 4 (one main, one per level of detail) |
| optimiser | AdamW, learning rate 3e-4, weight decay 0.01 |
| schedule | 60 warmup steps, then cosine decay |
| gradient clipping | 1.0 |
| seeds | 0, 1, 2 for the principal comparison |
| training scenes | 1,200 |
| evaluation | the whole of each held-out split, 150 scenes each |
| device | CPU, Apple M2 |
| loss | prototype losses + 2.0 × frame supervision + 0.5 × nested level-of-detail |

All three levels of detail are decoded every step, which is why a step costs four forward
passes. Step 7 rotated through one level per step, which meant the containment relation
between two levels was never present in a single computation graph and could not be asked
for.

Wall-clock: roughly 12 to 18 minutes per run on one CPU depending on arm, and about 6.5
seconds per full-split evaluation. This is a laptop-scale research prototype and nothing
about it is a claim of data or compute efficiency.

---

## 10. Predicted-placement methodology

The single most important methodological change in Step 8, and the one that makes its
numbers incomparable with Step 7's.

### What was wrong

The decoder works in each entity's canonical frame and took those frames from the scene.
The frame head was a zero-initialised linear probe of 3,084 parameters that no evaluation
ever queried. Step 7 measured the consequence on its own trained weights: ownership IoU of
0.6339 with placement supplied against 0.3020 with it inferred, and a spatial relation
accuracy of 0.9866 for an arm that cannot see spatial edges at all, because the frames
were supplying the arrangement the graph was supposed to.

### What Step 8 does

**A head worth training.** `FramePredictor`, 135,180 parameters, reading the latent
**after** the relational encoder so that neighbours can influence placement. Output is the
unchanged 12-number contract: three translation, three log-scale, six rotation. Zero
initialised at the canonical frame, so an untrained head produces a scene at the origin
rather than a degenerate one.

**A curriculum**, as fractions of the budget so one configuration means the same thing at
any length:

| Stage | Budget | Teacher forcing | Purpose |
| --- | --- | --- | --- |
| P0 | 0 to 15% | 1.0 | the head learns to produce a frame at all |
| P1 | 15% to 60% | 1.0 down to 0.0, linear | the decoder meets its own errors gradually |
| P2 | 60% to 100% | 0.0 | the model decodes entirely from its own placement |
| P3 | evaluation | none supplied | `use_predicted_frames=True`, always |

**Final teacher-forcing ratio: 0.0.** It is logged at every recorded step and written into
each run manifest.

**A guard.** `use_predicted_frames` is checked **before** the teacher-forcing ratio, so a
ratio left set by mistake cannot turn a headline result into an oracle result. A test
asserts the two paths give identical output, and a second test shows that corrupting the
true frame tensor leaves the inferred output bit-identical.

Supervising placement is not supplying it: the model is told the right answer during
training and must produce it unaided at inference, which is what any predictive head does.

### The metrics placement needed

| Metric | Definition |
| --- | --- |
| `position_error` | Euclidean translation error in scene units; the scene spans about [-1, 1] |
| `scale_error` | mean absolute log-scale error, a symmetric ratio error: 0.69 is a factor of two |
| `rotation_error` | geodesic angle in radians between the two rotations |
| `composite_frame_error` | `position + 0.5 × scale + 0.25 × rotation`, weights fixed before any run |

The weights make the three parts comparable in magnitude on this corpus. They are not a
claim about relative importance, and each part is also reported separately so a reader can
reweigh them. Errors are additionally broken down by entity group, because a mean over
twenty structures hides the case where four large chambers are placed well and every thin
structure badly.

**None of these is the number that answers H2.** A model can place every entity slightly
wrong and still get the anatomy right, or place them all near the average and score
moderately while getting every relation wrong. Whether the model puts the correct entity
in the correct **relational** location is spatial relation accuracy under inferred
placement, read against the relation-blind floor.

---

## 11. Graph experiments

### The comparison, and why it is fair

Five arms, identical in everything but the relational mechanism: same corpus, same splits,
same seeds, same optimiser, same budget, same evaluation, same initialisation policy, same
frame predictor, same 559,751 parameters of non-relational machinery.

| Arm | Relational mechanism | Relational parameters |
| --- | --- | ---: |
| A1 | none | 0 |
| A1M | per-entity depth; information cannot move between entities | 2,566,080 |
| A3Lite | untyped graph attention over the union of the three adjacencies | 692,864 |
| A3L | reduced typed transformer: typed bias, partitioned heads | 666,760 |
| A3 | full typed transformer | 2,666,896 |

Two questions, each isolated by one pair.

**Does relational structure beat capacity?** `A3Lite` against `A1M`. `A1M` has 3.7 times
`A3Lite`'s relational budget and spends it on depth applied to each entity separately. If
`A3Lite` wins, structure beat parameters; if `A1M` wins, the graph is an expensive way to
buy capacity and should go.

**Do relation types earn anything?** `A3Lite` against `A3L`, whose encoders differ by 3.9%
in size and by whether relation type is readable at all. Step 7's verdict was REMOVE, and
the Step 8 brief requires strong causal evidence to reverse it. A small metric difference
is explicitly insufficient.

**Is more graph better?** `A3L` against `A3`, four times the typed encoder at four times
the cost.

### What neither side is allowed

The graph arms receive nothing the no-graph arms do not: same entity inputs, same text
features, same points, same targets. The no-graph arms receive no positional shortcut:
Section 6 lists the eight routes that were closed and tested.

---

## 12. Causal perturbation experiments

An arm comparison can be confounded by capacity or by optimisation. Damaging one input on
a fixed trained model cannot: if the output does not move, that input was not being read.

Three perturbations are applied at evaluation time to models trained on intact input, with
entities, presence, text features and points held identical throughout.

| Ablation | What it corrupts | What a reading means |
| --- | --- | --- |
| R5 | spatial edges rewired to random pairs | connectivity destroyed, edge count kept |
| R6 | spatial edges removed | the spatial graph's total contribution |
| R7 | functional edges removed | the functional graph's total contribution |

And two placement conditions on the same weights:

| Ablation | Condition | What it says |
| --- | --- | --- |
| R8 | the model places entities itself | the headline condition |
| R9 | true frames supplied | the oracle bound: what perfect placement would buy |

**A control that has to be there.** A no-graph arm must show a change of exactly zero under
R5, R6 and R7. If it showed anything, the harness would be leaking. Reported for `A1` and
`A1M` for that reason, not because the numbers are interesting.

### The counterfactual construction

The Step 8 corpus makes the causal test structural rather than incidental. Two scenes can
share entity identities, entity presence and text features exactly, and differ only in
where their structures sit, because the arrangement is drawn from a continuous space that
none of those channels encodes. The relationship graph is then the only route to the
correct placement, by construction and not by assumption.

Raw output sensitivity is **not** accepted as evidence of relational understanding. A model
whose output merely moves when the graph changes has shown only that the graph reaches it.
The quantity reported is whether the output moves **toward** the correct arrangement, which
is what spatial relation accuracy under inferred placement measures, read against the
relation-blind floor.

---

## 13. Level-of-detail experiments

Placement inferred, test_seen.

| arm | lod1_iou | lod2_iou | lod3_iou | lod_detail_gain | lod_containment_mean | lod_preservation_mean |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 0.1974 | 0.2781 | 0.8549 | 0.6574 | 1.0000 | 0.9715 |
| A1M | 0.2090 | 0.2859 | 0.8753 | 0.6663 | 1.0000 | 0.9735 |
| A3Lite | 0.2446 | 0.3133 | 0.8790 | 0.6344 | 1.0000 | 0.9674 |
| A3L | 0.2170 | 0.2928 | 0.8772 | 0.6602 | 1.0000 | 0.9698 |
| A3 | 0.2357 | 0.3089 | 0.8564 | 0.6207 | 1.0000 | 0.9705 |

Raw IoU is not comparable across levels. A finer level exposes more entities, so its target covers more of the space, and a model that calls every point occupied scores an IoU equal to the density:

| level | target density, which is the trivial predictor's IoU |
| --- | ---: |
| 1 | 0.1426 |
| 2 | 0.1700 |
| 3 | 0.5520 |

Corrected for density, `(iou - density) / (1 - density)`:

| arm | level 1 | level 2 | level 3 | corrected detail gain |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0.0639 | 0.1303 | 0.6761 | +0.6122 |
| A1M | 0.0774 | 0.1397 | 0.7216 | +0.6442 |
| A3Lite | 0.1189 | 0.1726 | 0.7300 | +0.6111 |
| A3L | 0.0867 | 0.1480 | 0.7260 | +0.6392 |
| A3 | 0.1085 | 0.1673 | 0.6795 | +0.5710 |

### R10 nested against R11 non-nested

`R11` decodes each level from a **disjoint** token slice rather than a prefix, which is the closest approximation to three unrelated representations these weights can express. It is an approximation, not a trained control.

| arm | nested gain | non-nested gain | nested preservation | non-nested preservation | nested containment | non-nested containment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3Lite | 0.6501 | 0.4605 | 0.9678 | 0.8013 | 1.0000 | 0.9746 |
| A3L | 0.5886 | 0.2590 | 0.9730 | 0.7836 | 1.0000 | 0.9698 |
| A3 | 0.5413 | 0.3350 | 0.9740 | 0.7902 | 1.0000 | 0.9930 |
| A1M | 0.6584 | 0.4658 | 0.9708 | 0.8108 | 1.0000 | 0.9868 |
| A1 | 0.5597 | 0.3732 | 0.9746 | 0.7932 | 1.0000 | 0.9986 |

---

## 14. Entity-axis experiments

Placement inferred, test_seen.

| arm | what it has | entity_iou_mean | part_control_success | scene_iou | position_error |
| --- | --- | --- | --- | --- | --- |
| A3Lite | entity axis, per-entity geometry, untyped relational attention | 0.0930 | 0.0519 | 0.5193 | 0.1614 |
| A4 | entity axis kept, per-entity geometry removed: one shared token block | 0.0241 | 0.0042 | 0.5327 | 0.1565 |
| A0 | no entity axis: flat latent and shared field, the appearance baseline | 0.0122 | 0.0036 | 0.5283 | n/a |

The same ladder with placement supplied, as an oracle bound:

| arm | what it has | entity_iou_mean | part_control_success | scene_iou | position_error |
| --- | --- | --- | --- | --- | --- |
| A3Lite | entity axis, per-entity geometry, untyped relational attention | 0.1729 | 0.1736 | 0.5500 | 0.1614 |
| A4 | entity axis kept, per-entity geometry removed: one shared token block | 0.0241 | 0.0042 | 0.5327 | 0.1565 |
| A0 | no entity axis: flat latent and shared field, the appearance baseline | 0.0122 | 0.0036 | 0.5283 | n/a |

### What the ladder says

**The entity axis contributes almost nothing to gross shape and nearly everything to
control.** Scene IoU is 0.5193, 0.5327 and 0.5283 across the three rungs, which is to say
indistinguishable: all three produce a comparable overall organ. Part control is 0.0519,
0.0042 and 0.0036, a factor of **14** between the top rung and the bottom.

Removing per-entity geometry while keeping the entity axis (`A4`) loses almost all of that
control: 0.0042 against 0.0519. So it is not enough to have entities in the representation;
the geometry has to be addressable per entity. `A0`, which has neither, is barely worse than
`A4`, which has one without the other.

**And the gap widens when placement is supplied.** `A3Lite` rises from 0.0519 to 0.1736 part
control with true frames, a factor of **48** over `A0`. `A4` and `A0` are unchanged, because
neither has per-entity frames for an oracle to supply. That asymmetry is the point: the
entity axis is what makes placement a thing the model can be right or wrong about at all.

H4 passes, and it passes harder under predicted placement than it did under the teacher-forced
conditions of Step 7, because the monolithic arms cannot even be given the answer.

`A0`'s position error is `n/a` throughout. It has no per-entity frames, so there is nothing
to score. That is reported rather than filled with a zero.

---

## 15. Persistent-edit regression

Step 8 is not about editing. Step 7 measured that the edit head's locality works and its
accuracy does not, and recorded it as REVISE; spending Step 8's compute on it would have
come out of the placement experiments Step 8 exists for.

Step 8's changes were large enough to break editing by accident, though: the arrangement
space replaced the variant enum, placement became predicted, and the level-of-detail
objective changed. Six checks, all passing, recorded in
`experiments/runs/step8-editing-regression.json`:

| Check | Result |
| --- | --- |
| a presentation edit touches one entity and no other | PASS |
| an edit survives serialising and reloading the scene | PASS |
| a first edit survives a second edit to a different entity | PASS |
| entity identity is unchanged by any sequence of edits | PASS |
| the local edit head still writes only the target's token block | PASS |
| edit pairs still build on continuous arrangements | PASS |

The report cites the artefact, not the prose: `experiments/step8/editing_regression.py`
produces it and `tests/test_step8_editing_regression.py` runs the same checks in the test
suite.

Nothing here is evidence that persistent editing **works**. It is evidence that Step 8 did
not break what Step 7 left. The Step 7 verdict of REVISE stands unchanged.

---

## 16. Metrics

Step 7's floor was 0.8878. That number belongs to the Step 7 corpus and does not transfer:
Step 8's arrangements are continuous and range further, so a fixed guess does worse.

Two blind predictors are computed from the corpus alone, and the floor is the **higher** of
them. Taking the lower would flatter every model.

| Split | Oracle | Blind: canonical | Blind: training mean | **Floor** | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 1.0000 | 0.7874 | 0.7827 | **0.7874** | 0.2126 |
| test_seen | 1.0000 | 0.8110 | 0.8053 | **0.8110** | 0.1890 |
| test_arrangement | 1.0000 | 0.8284 | 0.8169 | **0.8284** | 0.1716 |
| test_transform | 1.0000 | 0.8178 | 0.7818 | **0.8178** | 0.1822 |
| test_combination | 1.0000 | 0.6236 | 0.6320 | **0.6320** | 0.3680 |

The oracle is 1.0000 everywhere, which is the check that the scoring code is not itself the
thing under test.

Headroom roughly doubled against Step 7's 0.1122, so the Step 8 benchmark discriminates
better. `test_combination` has the most headroom at 0.3680: mirroring and transposing
together is far from any fixed guess, which makes it the most informative split and the
hardest.

**A model below its split's floor has no relational competence on that metric.** This rule
is applied without exception below.

Every metric below is computed identically for every arm, from quantities every arm
produces. Each is reported with its placement condition attached.

### Geometry and control

| Metric | Definition |
| --- | --- |
| `entity_iou_mean` | mean per-entity ownership IoU over entities present in the scene |
| `part_control_success` | share of present entities whose ownership region reaches IoU >= 0.5 |
| `scene_iou` | IoU of predicted and true whole-organ occupancy on shared query points |
| `occupancy_chamfer` | point-set Chamfer between occupied sets, `2*sqrt(3)` if either is empty |

### Placement

| Metric | Definition |
| --- | --- |
| `position_error` | Euclidean translation error in scene units |
| `scale_error` | mean absolute log-scale error, a symmetric ratio error |
| `rotation_error` | geodesic angle in radians between predicted and true rotation |
| `composite_frame_error` | `position + 0.5 × scale + 0.25 × rotation`, weights fixed in advance |
| `placement_error` | distance between predicted and true entity **centroids**, from the composed scene rather than the frame |

`position_error` and `placement_error` are different quantities and both are reported. The
first asks whether the frame head produced the right transform; the second asks where the
entity's region actually ended up after decoding and composition. A model can get one
right and the other wrong.

### Relations

| Metric | Definition |
| --- | --- |
| `spatial_relation_accuracy` | share of a scene's **measured** spatial relations reproduced by predicted centroids |
| `spatial_relation_accuracy_strict` | the same, counting unplaceable entities as failures |
| `relation_blind_floor` | what a model with no relational information achieves, per split |
| gap above floor | `(accuracy − floor) / (1 − floor)`; negative means worse than a fixed blind guess |

### Level of detail

| Metric | Definition |
| --- | --- |
| `lodN_iou` | occupancy IoU at level N's token prefix against level N's own target |
| `lodN_target_density` | share of points that level N's target calls occupied |
| `lodN_iou_over_trivial` | share of the available headroom taken: `(iou − density) / (1 − density)` |
| `lod_detail_gain` | `lod3_iou − lod1_iou`, **confounded by density; see below** |
| `lod_detail_gain_over_trivial` | the density-corrected version, and the one to read |
| `lodA_to_B_containment` | share of points the coarse prefix called occupied that the fine prefix still does |
| `lodA_to_B_preservation` | share of points the coarse prefix got right, where the levels agree, that the fine prefix still gets right |
| `lodA_to_B_identity_consistency` | share of coarse-owned points whose predicted owner is unchanged at the finer level |

**A confound found during Step 8 and corrected before reporting.** Occupancy IoU is not
comparable across levels. A finer level exposes more entities, so its target covers more of
the space:

| Level | Entities exposed | Share of sampled points occupied |
| --- | ---: | ---: |
| 1 | 10 | 0.1403 |
| 2 | 16 | 0.1675 |
| 3 | 20 | 0.5529 |

A model that simply calls every point occupied scores an IoU equal to the target density,
which on this corpus is a **raw detail gain of about +0.41 while adding nothing at all**.
Step 7's negative raw gain and any positive raw gain here are both partly artefacts of
this. `lod_detail_gain_over_trivial` corrects for it: the trivial predictor scores exactly
0.0 and a perfect decode scores 1.0 at every level. A test pins both.

The raw number is still reported, because removing a metric after seeing results is the
practice these reports exist to avoid. It is reported with its confound named.

---

## 17. Results

Generated from `experiments/runs/step8-suite/step8_report.json`. 3 seeds, 1200 steps, cpu. Headline condition: **inferred placement**.


### test_seen, placement inferred

Relation-blind floor for this split: **0.8110**. Oracle: 1.0000. A model below the floor has no relational competence on that metric.

| arm | params | graph params | graph | entity_iou_mean | part_control_success | position_error | composite_frame_error | spatial_relation_accuracy | scene_iou | gap above floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 559,751 | 0 | none | 0.0747 ± 0.002 | 0.0350 ± 0.005 | 0.1676 ± 0.001 | 0.2341 ± 0.001 | 0.7472 ± 0.001 | 0.5099 ± 0.003 | -0.337 |
| A1M | 3,125,831 | 0 | none | 0.0741 ± 0.004 | 0.0330 ± 0.006 | 0.1679 ± 0.001 | 0.2344 ± 0.001 | 0.7485 ± 0.009 | 0.5254 ± 0.008 | -0.330 |
| A3Lite | 1,252,615 | 692,864 | untyped | 0.0902 ± 0.002 | 0.0544 ± 0.003 | 0.1599 ± 0.004 | 0.2253 ± 0.004 | 0.7791 ± 0.010 | 0.5228 ± 0.003 | -0.169 |
| A3L | 1,226,511 | 666,760 | reduced typed | 0.0783 ± 0.006 | 0.0443 ± 0.004 | 0.1644 ± 0.001 | 0.2301 ± 0.001 | 0.8036 ± 0.029 | 0.5048 ± 0.012 | -0.039 |
| A3 | 3,226,647 | 2,666,896 | full typed | 0.0908 ± 0.004 | 0.0596 ± 0.005 | 0.1561 ± 0.000 | 0.2215 ± 0.000 | 0.8119 ± 0.005 | 0.5197 ± 0.004 | +0.005 |


### test_arrangement, placement inferred

Relation-blind floor for this split: **0.8284**. Oracle: 1.0000. A model below the floor has no relational competence on that metric.

| arm | params | graph params | graph | entity_iou_mean | part_control_success | position_error | composite_frame_error | spatial_relation_accuracy | scene_iou | gap above floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 559,751 | 0 | none | 0.0587 ± 0.001 | 0.0240 ± 0.000 | 0.1739 ± 0.003 | 0.2428 ± 0.003 | 0.7831 ± 0.017 | 0.4786 ± 0.008 | -0.263 |
| A1M | 3,125,831 | 0 | none | 0.0593 ± 0.002 | 0.0219 ± 0.003 | 0.1736 ± 0.001 | 0.2425 ± 0.001 | 0.7796 ± 0.006 | 0.4903 ± 0.006 | -0.284 |
| A3Lite | 1,252,615 | 692,864 | untyped | 0.0793 ± 0.001 | 0.0481 ± 0.002 | 0.1651 ± 0.004 | 0.2328 ± 0.004 | 0.8478 ± 0.011 | 0.5057 ± 0.005 | +0.113 |
| A3L | 1,226,511 | 666,760 | reduced typed | 0.0676 ± 0.003 | 0.0433 ± 0.003 | 0.1695 ± 0.002 | 0.2377 ± 0.002 | 0.8448 ± 0.005 | 0.4889 ± 0.006 | +0.096 |
| A3 | 3,226,647 | 2,666,896 | full typed | 0.0831 ± 0.001 | 0.0546 ± 0.001 | 0.1603 ± 0.000 | 0.2281 ± 0.000 | 0.8501 ± 0.009 | 0.5104 ± 0.002 | +0.127 |


### test_transform, placement inferred

Relation-blind floor for this split: **0.8178**. Oracle: 1.0000. A model below the floor has no relational competence on that metric.

| arm | params | graph params | graph | entity_iou_mean | part_control_success | position_error | composite_frame_error | spatial_relation_accuracy | scene_iou | gap above floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 559,751 | 0 | none | 0.0475 ± 0.003 | 0.0193 ± 0.002 | 0.1787 ± 0.004 | 0.2642 ± 0.004 | 0.6073 ± 0.029 | 0.4708 ± 0.008 | -1.155 |
| A1M | 3,125,831 | 0 | none | 0.0482 ± 0.001 | 0.0195 ± 0.002 | 0.1789 ± 0.001 | 0.2644 ± 0.001 | 0.6072 ± 0.012 | 0.4831 ± 0.006 | -1.156 |
| A3Lite | 1,252,615 | 692,864 | untyped | 0.0566 ± 0.001 | 0.0248 ± 0.001 | 0.1761 ± 0.001 | 0.2608 ± 0.001 | 0.6803 ± 0.010 | 0.4854 ± 0.005 | -0.755 |
| A3L | 1,226,511 | 666,760 | reduced typed | 0.0510 ± 0.001 | 0.0220 ± 0.003 | 0.1768 ± 0.002 | 0.2618 ± 0.002 | 0.6681 ± 0.013 | 0.4760 ± 0.005 | -0.822 |
| A3 | 3,226,647 | 2,666,896 | full typed | 0.0577 ± 0.001 | 0.0244 ± 0.002 | 0.1743 ± 0.001 | 0.2591 ± 0.001 | 0.6790 ± 0.007 | 0.4889 ± 0.003 | -0.762 |


### test_combination, placement inferred

Relation-blind floor for this split: **0.6320**. Oracle: 1.0000. A model below the floor has no relational competence on that metric.

| arm | params | graph params | graph | entity_iou_mean | part_control_success | position_error | composite_frame_error | spatial_relation_accuracy | scene_iou | gap above floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 559,751 | 0 | none | 0.0555 ± 0.003 | 0.0191 ± 0.002 | 0.1909 ± 0.002 | 0.2621 ± 0.001 | 0.7228 ± 0.021 | 0.5007 ± 0.003 | +0.247 |
| A1M | 3,125,831 | 0 | none | 0.0558 ± 0.001 | 0.0183 ± 0.001 | 0.1922 ± 0.002 | 0.2633 ± 0.002 | 0.7183 ± 0.014 | 0.5142 ± 0.004 | +0.234 |
| A3Lite | 1,252,615 | 692,864 | untyped | 0.0343 ± 0.004 | 0.0207 ± 0.002 | 0.2092 ± 0.016 | 0.2790 ± 0.016 | 0.5570 ± 0.023 | 0.5075 ± 0.004 | -0.204 |
| A3L | 1,226,511 | 666,760 | reduced typed | 0.0329 ± 0.000 | 0.0160 ± 0.003 | 0.2033 ± 0.007 | 0.2740 ± 0.007 | 0.5387 ± 0.014 | 0.4879 ± 0.002 | -0.254 |
| A3 | 3,226,647 | 2,666,896 | full typed | 0.0343 ± 0.003 | 0.0210 ± 0.003 | 0.2218 ± 0.005 | 0.2916 ± 0.005 | 0.5594 ± 0.018 | 0.5065 ± 0.003 | -0.197 |


### The required summary table

One row per model on `test_seen`, the split with held-out families and in-distribution arrangements. The oracle row is the same weights with true frames supplied, and is an upper bound rather than a result.

| Model | Params | Placement | Graph | Entity IoU | Part Control | Frame Error | Relation Accuracy | Relation-Blind Gap | Scene IoU |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 559,751 | inferred | none | 0.0747 | 0.0350 | 0.2341 | 0.7472 | -0.337 | 0.5099 |
| A1M | 3,125,831 | inferred | none | 0.0741 | 0.0330 | 0.2344 | 0.7485 | -0.330 | 0.5254 |
| A3Lite | 1,252,615 | inferred | untyped | 0.0902 | 0.0544 | 0.2253 | 0.7791 | -0.169 | 0.5228 |
| A3L | 1,226,511 | inferred | reduced typed | 0.0783 | 0.0443 | 0.2301 | 0.8036 | -0.039 | 0.5048 |
| A3 | 3,226,647 | inferred | full typed | 0.0908 | 0.0596 | 0.2215 | 0.8119 | +0.005 | 0.5197 |
| Oracle frame control (A3) | 3,226,647 | supplied | full typed | 0.1971 | 0.1978 | 0.2215 | 0.9719 | +0.852 | 0.5953 |

### Relation accuracy against the floor, every split

Each cell is the accuracy, then the share of the available headroom taken. Negative means a fixed arrangement-blind guess would have done better.

| arm | test_seen | test_arrangement | test_transform | test_combination |
| --- | --- | --- | --- | --- |
| A1 | 0.7472 (-0.337, **below**) | 0.7831 (-0.263, **below**) | 0.6073 (-1.155, **below**) | 0.7228 (+0.247, above) |
| A1M | 0.7485 (-0.330, **below**) | 0.7796 (-0.284, **below**) | 0.6072 (-1.156, **below**) | 0.7183 (+0.234, above) |
| A3Lite | 0.7791 (-0.169, **below**) | 0.8478 (+0.113, above) | 0.6803 (-0.755, **below**) | 0.5570 (-0.204, **below**) |
| A3L | 0.8036 (-0.039, **below**) | 0.8448 (+0.096, above) | 0.6681 (-0.822, **below**) | 0.5387 (-0.254, **below**) |
| A3 | 0.8119 (+0.005, above) | 0.8501 (+0.127, above) | 0.6790 (-0.762, **below**) | 0.5594 (-0.197, **below**) |

### Entity ownership IoU across splits, placement inferred

| arm | test_seen | test_arrangement | test_transform | test_combination |
| --- | --- | --- | --- | --- |
| A1 | 0.0747 | 0.0587 | 0.0475 | 0.0555 |
| A1M | 0.0741 | 0.0593 | 0.0482 | 0.0558 |
| A3Lite | 0.0902 | 0.0793 | 0.0566 | 0.0343 |
| A3L | 0.0783 | 0.0676 | 0.0510 | 0.0329 |
| A3 | 0.0908 | 0.0831 | 0.0577 | 0.0343 |

### The cost of inferring placement (test_seen)

The supplied column is an oracle upper bound, not a result. It says what better placement could buy.

| arm | placement inferred | placement supplied (oracle) | cost |
| --- | --- | --- | --- |
| A1 | 0.0747 | 0.1409 | 0.0662 |
| A1M | 0.0741 | 0.1362 | 0.0621 |
| A3Lite | 0.0902 | 0.1779 | 0.0877 |
| A3L | 0.0783 | 0.1598 | 0.0815 |
| A3 | 0.0908 | 0.1971 | 0.1063 |

### Position error by entity group, test_seen, placement inferred

| arm | pooled | chambers | valves | vessels | septa | walls |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 0.1676 | 0.2019 | 0.1599 | 0.1821 | 0.0904 | 0.0323 |
| A1M | 0.1679 | 0.2028 | 0.1599 | 0.1820 | 0.0905 | 0.0329 |
| A3Lite | 0.1599 | 0.1852 | 0.1507 | 0.1783 | 0.0905 | 0.0326 |
| A3L | 0.1644 | 0.1930 | 0.1545 | 0.1822 | 0.0903 | 0.0327 |
| A3 | 0.1561 | 0.1756 | 0.1470 | 0.1767 | 0.0910 | 0.0328 |

---

## 18. Statistical variation

Three seeds on every arm. Spreads are reported as the population standard deviation beside
every mean, and every table states how many seeds contributed.

The spreads themselves carry a finding. On `test_seen`, spatial relation accuracy:

| Arm | Mean | Spread | Range across seeds |
| --- | ---: | ---: | ---: |
| A3Lite | 0.7791 | 0.010 | 0.024 |
| A3L | 0.8036 | 0.029 | 0.066 |
| A3 | 0.8119 | 0.005 | 0.010 |

`A3L`'s spread is three times `A3Lite`'s and six times `A3`'s. Its mean sits between them,
so a single-seed comparison could have placed `A3L` above or below `A3Lite` depending on
which seed was drawn. That is the case three seeds exist to catch, and it is the reason the
typed-against-untyped comparison is reported as **not distinguishable** rather than as a
narrow win for either.

Entity ownership IoU is tighter everywhere, spreads of 0.001 to 0.006, so the geometry
ordering is stable across seeds.

---

## 19. Failure cases

Four, reported because they are the informative part of Step 8.

### Extrapolation fails for every arm

`test_transform` holds rotations larger than any in training. Every arm falls far below the
relation-blind floor: the best gap is −0.755 and the worst −1.156. The graph arms are less
bad than the no-graph arms, 0.680 against 0.607, and *less bad than a blind guess* is not
competence. Extrapolating past the edge of the training range is not something this
architecture does at this budget.

### Novel composition is worse than ignoring the graph

`test_combination` holds `mirror` and `transpose` together, a pair that never co-occurs in
training. The ordering **reverses**:

| Arm | Relation accuracy | Gap above the 0.6320 floor |
| --- | ---: | ---: |
| A1, no graph | 0.7228 | **+0.247** |
| A1M, no graph | 0.7183 | **+0.234** |
| A3Lite | 0.5570 | −0.204 |
| A3L | 0.5387 | −0.254 |
| A3 | 0.5594 | −0.197 |

The two arms that cannot read the graph are **above** the floor; all three that can are
below it. Reading a relationship graph that describes a composition never seen in training
is worse than ignoring it. A model that ignores relations produces an average arrangement,
which on this split satisfies more measured relations than the specific wrong arrangement
the graph drives the relational arms toward.

This is the clearest negative result in Step 8 and it is specific: the relational mechanism
generalises across a continuum it has seen and fails to compose two transformations it has
only seen separately.

### The coarse level of detail is weak

Corrected for target density, level 1 takes 6% to 12% of the available headroom while level
3 takes 68% to 73%. The prefixes nest and the finer levels genuinely add detail, and the
coarse representation itself is poor. Containment of exactly 1.0 is partly explained by the
coarse prefix under-committing: it is easy to be a subset of the fine decode when little is
asserted.

### The curriculum trades crispness for placement

From the first validation to the last, counted over all fifteen runs:

| Metric | Runs that improved | Mean change |
| --- | ---: | ---: |
| composite frame error | **15 of 15** | −0.0079 |
| position error | 11 of 15 | −0.0054 |
| scene IoU | 10 of 15 | +0.0110 |
| spatial relation accuracy | 9 of 15 | +0.0128 |
| entity ownership IoU | **0 of 15** | −0.0318 |
| part control success | **1 of 15** | −0.0354 |

The two halves of this table are not equally strong and the difference matters. Frame error
improves in **every** run without exception. Entity ownership degrades in **every** run
without exception. The three metrics in between improve in a majority of runs but not all,
so "placement and shape improve" is a tendency, while "frame error improves and ownership
degrades" is universal.

Per-entity ownership peaks around the middle of the curriculum, when frames are still partly
supplied, and falls as the model is weaned onto its own placement.

The reading: training on its own noisy frames teaches the decoder to be less sensitive to
frames, which costs crisp per-entity boundaries and buys better placement and better
overall shape. Whether that trade is worth making depends on what the representation is
for, and Step 8 does not settle it. It does mean the convergence criterion, which watches
entity ownership IoU, classifies **all fifteen runs as not converged**, and every comparison
here is an equal-budget comparison among unconverged runs.

---

## 20. Convergence analysis

Criterion: `entity_iou_mean` under predicted placement, tolerance 0.01, failure floor 0.01.

**converged 0, not converged 15, failed 0** of 15 runs.

| run | verdict | reason |
| --- | --- | --- |
| `s8-A1-seed0` | not_converged | final 0.0645 is 0.0499 below its best 0.1144 |
| `s8-A1-seed1` | not_converged | final 0.0596 is 0.0536 below its best 0.1132 |
| `s8-A1-seed2` | not_converged | final 0.0638 is 0.0449 below its best 0.1088 |
| `s8-A1M-seed0` | not_converged | final 0.0663 is 0.0372 below its best 0.1035 |
| `s8-A1M-seed1` | not_converged | final 0.0589 is 0.0367 below its best 0.0955 |
| `s8-A1M-seed2` | not_converged | final 0.0611 is 0.0343 below its best 0.0954 |
| `s8-A3-seed0` | not_converged | final 0.0900 is 0.0234 below its best 0.1134 |
| `s8-A3-seed1` | not_converged | final 0.0846 is 0.0131 below its best 0.0977 |
| `s8-A3-seed2` | not_converged | final 0.0809 is 0.0254 below its best 0.1062 |
| `s8-A3L-seed0` | not_converged | final 0.0781 is 0.0248 below its best 0.1030 |
| `s8-A3L-seed1` | not_converged | final 0.0656 is 0.0312 below its best 0.0968 |
| `s8-A3L-seed2` | not_converged | final 0.0727 is 0.0191 below its best 0.0918 |
| `s8-A3Lite-seed0` | not_converged | final 0.0834 is 0.0400 below its best 0.1233 |
| `s8-A3Lite-seed1` | not_converged | final 0.0837 is 0.0324 below its best 0.1161 |
| `s8-A3Lite-seed2` | not_converged | final 0.0868 is 0.0115 below its best 0.0982 |

### Direction of travel, first validation to last

| metric | runs that improved | mean change |
| --- | ---: | ---: |
| `composite_frame_error` | 15 of 15 | -0.0079 |
| `entity_iou_mean` | 0 of 15 | -0.0318 |
| `part_control_success` | 1 of 15 | -0.0354 |
| `position_error` | 11 of 15 | -0.0054 |
| `scene_iou` | 10 of 15 | +0.0110 |
| `spatial_relation_accuracy` | 9 of 15 | +0.0128 |

---

## 21. Ablations

Seed 0, split `test_seen`, 150 scenes.


### A3Lite

| ablation | entity_iou_mean | change from control | what it isolates |
| --- | --- | --- | --- |
| `R8_predicted_placement` | 0.0930 | +0.0000 | the headline condition: the model places entities itself |
| `R9_oracle_placement` | 0.1729 | +0.0799 | true frames supplied; the upper bound better placement could buy |
| `R5_shuffled_endpoints` | 0.0916 | -0.0014 | spatial edges rewired to random pairs; connectivity destroyed |
| `R6_no_spatial` | 0.0919 | -0.0011 | every spatial edge removed |
| `R7_no_functional` | 0.0732 | -0.0198 | every functional edge removed |
| `R10_nested_lod` | n/a | n/a | levels decoded from nested prefixes, as trained |
| `R11_non_nested_lod` | n/a | n/a | levels decoded from disjoint token slices; an approximation |

### A3L

| ablation | entity_iou_mean | change from control | what it isolates |
| --- | --- | --- | --- |
| `R8_predicted_placement` | 0.0849 | +0.0000 | the headline condition: the model places entities itself |
| `R9_oracle_placement` | 0.1418 | +0.0569 | true frames supplied; the upper bound better placement could buy |
| `R5_shuffled_endpoints` | 0.0815 | -0.0034 | spatial edges rewired to random pairs; connectivity destroyed |
| `R6_no_spatial` | 0.0815 | -0.0034 | every spatial edge removed |
| `R7_no_functional` | 0.0765 | -0.0084 | every functional edge removed |
| `R10_nested_lod` | n/a | n/a | levels decoded from nested prefixes, as trained |
| `R11_non_nested_lod` | n/a | n/a | levels decoded from disjoint token slices; an approximation |

### A3

| ablation | entity_iou_mean | change from control | what it isolates |
| --- | --- | --- | --- |
| `R8_predicted_placement` | 0.0960 | +0.0000 | the headline condition: the model places entities itself |
| `R9_oracle_placement` | 0.2025 | +0.1065 | true frames supplied; the upper bound better placement could buy |
| `R5_shuffled_endpoints` | 0.0940 | -0.0020 | spatial edges rewired to random pairs; connectivity destroyed |
| `R6_no_spatial` | 0.0902 | -0.0058 | every spatial edge removed |
| `R7_no_functional` | 0.0854 | -0.0106 | every functional edge removed |
| `R10_nested_lod` | n/a | n/a | levels decoded from nested prefixes, as trained |
| `R11_non_nested_lod` | n/a | n/a | levels decoded from disjoint token slices; an approximation |

### A1M

| ablation | entity_iou_mean | change from control | what it isolates |
| --- | --- | --- | --- |
| `R8_predicted_placement` | 0.0774 | +0.0000 | the headline condition: the model places entities itself |
| `R9_oracle_placement` | 0.1448 | +0.0674 | true frames supplied; the upper bound better placement could buy |
| `R5_shuffled_endpoints` | 0.0774 | +0.0000 | spatial edges rewired to random pairs; connectivity destroyed |
| `R6_no_spatial` | 0.0774 | +0.0000 | every spatial edge removed |
| `R7_no_functional` | 0.0774 | +0.0000 | every functional edge removed |
| `R10_nested_lod` | n/a | n/a | levels decoded from nested prefixes, as trained |
| `R11_non_nested_lod` | n/a | n/a | levels decoded from disjoint token slices; an approximation |

### A1

| ablation | entity_iou_mean | change from control | what it isolates |
| --- | --- | --- | --- |
| `R8_predicted_placement` | 0.0773 | +0.0000 | the headline condition: the model places entities itself |
| `R9_oracle_placement` | 0.1683 | +0.0910 | true frames supplied; the upper bound better placement could buy |
| `R5_shuffled_endpoints` | 0.0773 | +0.0000 | spatial edges rewired to random pairs; connectivity destroyed |
| `R6_no_spatial` | 0.0773 | +0.0000 | every spatial edge removed |
| `R7_no_functional` | 0.0773 | +0.0000 | every functional edge removed |
| `R10_nested_lod` | n/a | n/a | levels decoded from nested prefixes, as trained |
| `R11_non_nested_lod` | n/a | n/a | levels decoded from disjoint token slices; an approximation |

---

## 22. Decision matrix

Verdicts follow rules fixed before the results section, and each is paired with an
evidence-strength grade and the experiment that would overturn it.

### How a verdict is assigned

| Verdict | Trigger |
| --- | --- |
| **KEEP** | it measurably improves an outcome, or removing or corrupting it measurably degrades one, under **inferred** placement |
| **SIMPLIFY** | the idea is right and a smaller version matches the larger within seed spread |
| **REVISE** | it does not work as built, and a specific mechanism explains why |
| **REMOVE** | deleting or corrupting it produces no measurable effect, and nothing suggests more of anything would change that |

### How evidence strength is graded

| Grade | Requirement |
| --- | --- |
| **STRONG** | consistent in sign across all three seeds and all four held-out splits, with an effect larger than the seed spread |
| **MODERATE** | consistent across seeds on most splits, or a large effect at one seed with a mechanism that explains it |
| **WEAK** | the sign is consistent but the effect is within or near the seed spread |
| **INCONCLUSIVE** | the experiment could not distinguish the hypotheses, or the runs did not converge enough to say |

`INCONCLUSIVE` is a real outcome and is used where it applies. Step 7 withdrew three
conclusions that had been stated confidently on setups that could not test them.

### Rules that override a raw comparison

**Below the floor is not competence.** An arm whose spatial relation accuracy under
inferred placement sits below its split's relation-blind floor has no relational competence
on that metric, whatever its other numbers say.

**Structural results are not achievements.** Where a number follows from the construction
rather than from anything learned, it is reported as structural and supports no verdict.

**A small difference does not reverse a REMOVE.** The Step 7 verdict on typed relations
stands unless Step 8 shows strong causal evidence that types materially improve relational
reasoning. The brief is explicit that a tiny metric difference is insufficient, and so is
this report.

**Convergence bounds the claim.** An equal-budget comparison among runs that have not
converged shows which arm learns faster, not which ends up better. Every verdict states
whether its evidence is budget-bound.
### The verdicts

| Component | Verdict | Evidence | Strength |
| --- | --- | --- | --- |
| Anatomical World Representation as source of truth | **KEEP** | every result depends on relations measured from it; deleting the relational input costs 0.020 entity IoU on a trained A3Lite | STRONG |
| Entity axis / per-entity factorisation | **KEEP** | removing per-entity geometry drops part control by roughly 92% under predicted placement | STRONG |
| Per-entity geometry token blocks | **KEEP** | same evidence; and they are what make the nested prefix a prefix | STRONG |
| Write-protected identity subspace | **KEEP** | bit-identical through every arm and every edit, verified by test | STRONG |
| Geometry correspondence through the entity axis | **KEEP** | part labels name owners with no learned association step | STRONG |
| **Untyped graph attention** | **KEEP** | beats a capacity-matched no-graph arm holding 3.7 times its relational budget, on every geometry metric across every split, across three seeds | STRONG |
| **Typed relations** | **REMOVE** | A3Lite is better than A3L on ownership at every split; A3L has three times the seed spread and no consistent advantage | MODERATE |
| Head partitioning by graph type | **REMOVE** | follows from the above; it exists to separate types that are not read | MODERATE |
| **Predicted placement** | **REVISE** | it works: the graph arms clear the relation-blind floor on the interpolation split. It does not work well enough: one split of four, and supplying frames still roughly doubles ownership | MODERATE |
| **Nested LOD** | **KEEP** | containment exactly 1.0000, preservation 0.967 to 0.975, and the nested prefix beats the disjoint-slice control on every measure in every arm | STRONG |
| Coarse level of detail | **REVISE** | the prefixes nest and level 1 takes only 6% to 12% of its available headroom | STRONG |
| Persistent editing | **REVISE** | unchanged from Step 7; Step 8 verified only that it still works as it did | not re-tested |
| Relational compositional generalisation | **REVISE** | on an unseen composition, reading the graph is worse than ignoring it | STRONG |

### Why `A3` is not kept despite being the best arm on several metrics

`A3` leads on entity ownership IoU at two splits and on position error at all four. It
costs 2,666,896 relational parameters against `A3Lite`'s 692,864.

On `test_seen` the ownership difference is 0.0908 against 0.0902, inside the seed spread.
On `test_arrangement` it is 0.0831 against 0.0793, outside it but small. For 3.85 times the
parameters.

The Step 8 brief is explicit: *do not retain the large graph transformer merely because it
performs slightly better*. The rule is applied. `A3Lite` is the recommended configuration
and `A3` is recorded as a measured upper bound at roughly four times the cost, not as a
component to keep.

### What would falsify each major decision

| Decision | The experiment that would overturn it |
| --- | --- |
| KEEP untyped graph | an untyped graph at `A3Lite`'s budget failing to beat `A1M` on a corpus where relations are not the only route to placement, or `A1M` winning at a longer budget |
| REMOVE typed relations | a typed encoder at a matched budget beating `A3Lite` on arrangement-sensitive metrics by more than seed spread, on a corpus where relation type is the only route to the answer |
| REVISE predicted placement | a placement head supervised on relations rather than on frame components clearing the floor on every split, which would move the verdict to KEEP |
| KEEP nested LOD | a per-level projection head beating the nested prefix at every level simultaneously, which would mean the nesting property is not buying what it was proposed to buy |
| KEEP entity axis | a monolithic representation matching per-entity part control under predicted placement; `A4` misses it by an order of magnitude, so this would need a different monolithic design |
| REVISE compositional generalisation | the failure moving with the hold-out when a different pair is held out, which would make it a training-distribution problem rather than a mechanism problem |

---

## 23. What claims are supported

Each with the evidence and the strength grade from Section 22.

**A lightweight untyped graph gives real relational reasoning at a fraction of the cost.**
`A3Lite` beats both no-graph arms on every geometry metric across every split, and beats
`A1M`, which holds 3.7 times its relational parameter budget. On `test_seen` entity
ownership IoU: `A3Lite` 0.0902 ± 0.002 against `A1M` 0.0741 ± 0.004 and `A1` 0.0747 ±
0.002. Relational structure beats capacity. **STRONG.**

**Relation types earn nothing.** `A3Lite` and `A3L` differ by 3.9% in encoder size and by
whether relation type is readable. `A3Lite` is better on entity ownership IoU at every
split; `A3L` is marginally better on relation accuracy at `test_seen` and worse at
`test_combination`, with three times the seed spread. No consistent advantage in either
direction. **MODERATE**, and it confirms rather than reverses the Step 7 REMOVE verdict.

**The model can infer some placement from the representation.** On `test_arrangement`, all
three graph arms clear the relation-blind floor: `A3` +0.127, `A3Lite` +0.113, `A3L`
+0.096, while both no-graph arms sit at −0.26 and −0.28. That gap exists only because the
graph is read. **MODERATE**, because it holds on one split of four.

**Relations are read, and which relations matter is measurable.** Removing the functional
graph from a trained `A3Lite` costs 0.0198 entity IoU; removing the spatial graph costs
0.0011. The no-graph arms show a change of exactly 0.0000 under every graph perturbation,
which is the control that proves the harness is not leaking. **STRONG.**

**The level-of-detail prefixes are genuinely nested.** Containment is exactly 1.0000 for
every arm, preservation 0.967 to 0.975, and the density-corrected detail gain is +0.54 to
+0.66. Against the disjoint-slice approximation on the same weights, the nested prefix wins
on every measure in every arm: preservation 0.97 against 0.79, containment 1.0000 against
0.97 to 0.999. **STRONG**, and it reverses Step 7's finding that the prefixes were used and
made things worse.

**The entity axis and per-entity geometry remain fundamental under predicted placement.**
See Section 14. **Grade recorded there.**

**Placement is the binding constraint.** Supplying true frames roughly doubles entity
ownership IoU for every arm: `A3` 0.0908 to 0.1971, `A3Lite` 0.0902 to 0.1779, `A1` 0.0747
to 0.1409. Whatever else limits this architecture, more than half of the achievable
ownership accuracy is currently lost to placement error. **STRONG.**

---

## 24. What claims are NOT supported

**That the architecture can infer placement well.** The best arm clears its relation-blind
floor on one split of four, by +0.127 on `test_arrangement` and by +0.005 on `test_seen`,
which is within its own seed spread. On the other two splits every arm is below the floor.
H2 is not passed.

**That relational reasoning generalises.** It generalises across a continuum it has seen.
It fails to extrapolate past the training range, and on a composition of two transformations
seen only separately it is **worse than ignoring the graph entirely**.

**That the full typed transformer is worth its cost.** `A3` is the best arm on several
metrics, and it costs 2,666,896 relational parameters against `A3Lite`'s 692,864. On
`test_seen` entity ownership IoU the difference is 0.0908 against 0.0902, well inside the
seed spread. On `test_arrangement` it is 0.0831 against 0.0793, outside it. A real but
small advantage for 3.85 times the parameters.

**That any arm converged.** All fifteen runs are classified not converged by the
pre-registered criterion. Every comparison is equal-budget among unconverged runs, which
shows which arm learns faster, not which ends up better.

**That the coarse level of detail is usable.** It takes 6% to 12% of its available headroom.

**Anything about anatomy.** The corpus is procedurally generated. No real medical data was
used, no medical validation was performed, and no claim of anatomical or clinical accuracy
is made or supported.

---

## 25. Limitations

**One generator, forty families.** Held-out families test interpolation within one
procedural generator, not generalisation to organs built a different way.

**A budget that binds.** 1,200 steps on one CPU. No run converged. A graph that helps only
late, or a placement head that needs longer, would not show here. Step 7 found exactly that
pattern once: its graph's arrangement response nearly doubled between 900 and 2,700 steps.

**The relationship graph under-determines placement.** Measured relations are discrete and
52% of scenes share a graph with another scene, while arrangements are continuous. A perfect
relational model therefore cannot recover the exact arrangement from the graph alone; it can
only satisfy the relations. Frame error is reported for completeness and bounded by this,
which is why the decision rests on relation accuracy against the floor instead.

**`R11` is an approximation.** A properly non-nested control would need its own trained run.
What is reported is the closest thing the trained weights can express, and it is named as an
approximation wherever it appears.

**One backbone for editing.** The editing regression uses one configuration and checks that
Step 8 did not break Step 7's behaviour. It is not evidence that editing works.

**The composite frame error's weights are a choice.** Fixed before the runs and reported,
but they are a normalisation, not a finding. Every component is reported separately.

---

## 26. Recommendation for Step 9

Not recommended because it is the next number. Recommended because three measurements here
point at the same bottleneck and one points at a failure worth understanding before
anything is scaled.

### The case for a Step 9 at all

Step 8 answers the question Step 7 could not ask. The representation does support relational
inference: a 692,864-parameter untyped graph beats a no-graph arm holding 3.7 times that
budget, and the effect survives three seeds and four splits. The entity axis and per-entity
geometry survive the removal of teacher forcing. The prefixes nest. Those are the pieces
worth building on.

What is not ready is placement, and Step 8 measures exactly how far short it falls: the best
arm clears its relation-blind floor on one split of four, and supplying true frames roughly
doubles ownership accuracy on all of them. Scaling a model whose placement is this weak
would scale the wrong thing.

### What Step 9 should do, in order

**1. Fix placement. This is the whole of the recommendation's weight.**

Three specific things, each with a measurement that already exists:

* **Iterative frame refinement.** The head currently predicts a frame in one shot from the
  entity latent. Predicting a correction, conditioned on the current arrangement of every
  entity, lets relations act on placement more than once. Measured by position error and by
  relation accuracy against the floor.
* **Supervise placement relationally.** The frame loss is an L1 on the frame components. A
  loss on the *measured relations between predicted centroids* would optimise what the
  metric actually asks for. This is the largest single gap between what is trained and what
  is scored.
* **Revisit the curriculum.** Weaning onto predicted frames improves placement and costs
  per-entity crispness. Whether a slower schedule, a longer budget, or a frozen decoder
  during the transition recovers the crispness is a small experiment with a clear answer.

**2. Understand the compositional failure before working around it.**

Reading the graph is worse than ignoring it on a composition never seen in training. The
obvious explanation is that the encoder has learned arrangement-specific responses rather
than compositional ones. That is testable: train with mirror and transpose co-occurring and
hold out a different pair, and see whether the failure moves with the hold-out or stays put.
If it moves, the mechanism is fine and the training distribution was too narrow. If it stays,
the mechanism does not compose and a different one is needed.

This matters more than it looks. An anatomical system will meet combinations it was not
trained on constantly, and a relational mechanism that is worse than useless on them is a
liability, not an unfinished feature.

**3. Fix the coarse level of detail.**

The prefixes nest, and level 1 takes 6% to 12% of its available headroom. The nesting
objective did its job; the coarse representation is simply poor. Weighting the coarse level
higher is the obvious first try and should be tried before anything more elaborate.

### What Step 9 should not do

**Not scale.** No measurement here says the models are capacity-limited. `A1M` holds 3.7
times `A3Lite`'s relational budget and loses on every metric.

**Not train longer as the primary move.** No run converged, so a longer budget is worth
running as a control. It is not a fix for a placement head that is not asked to optimise the
thing it is scored on.

**Not reinstate typed relations.** Step 8 confirms Step 7's REMOVE. Reversing it needs a
corpus where relation type is the only route to the answer, which this one is not.

**Not touch real medical data, validation, deployment or production.** Nothing here
supports it, and nothing in Step 9 as described would either.

**Not redesign editing.** Step 7 left it at REVISE with locality working and accuracy not.
It should stay there until placement is solved, because an edit head that writes into a
badly placed scene cannot be evaluated cleanly.

### The one-line version

Step 9 is a placement step, not a scaling step. The relational mechanism, the entity axis
and the nested prefixes have earned their place; placement has not yet earned its, and it is
the constraint every other number is waiting on.
