# STEP 7 COMPLETION REPORT

**Lagnav 3D / Lagnav AI — Representation stress test, graph necessity, whole-organ
benchmark and persistent local editing.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data was used. No external 3D generation
system was used, as a component or as a baseline.

---

## 1. What Step 7 was for

Step 7 is not a scaling step. It asks which of the architectural ideas carried forward
from Steps 4, 5 and 6 are genuinely necessary, and removes or simplifies the ones that
are not. A negative result is a valid outcome and is reported as one.

The central question: the partitioned relational graph transformer costs 2,666,896 of the
prototype's 3,094,551 parameters, 86% of the model. Does it buy anything?

## 2. The headline finding, stated first

**The most important result of Step 7 is methodological, and it is uncomfortable.**

The central question has now been answered three times, and every answer has been
withdrawn on inspection. Not because the measurements were wrong. Each was a correct
measurement of a setup that could not distinguish the hypotheses it was meant to test.

| Attempt | Finding | Defect found afterwards |
| --- | --- | --- |
| Step 6 | graph encoder not earning its parameters | the corpus carried **one** relationship graph across all 2,000 scenes |
| Step 7 suite 1 | `A1` (no graph) matches `A3` (full graph) | variant was aliased onto family: every family appeared in exactly one arrangement |
| Step 7 suite 2 | `A1` matches `A3` | the part-correspondence target named entities the level hides, costing ~1e9 per such point for **every** arm |

Two further conditions were found to invalidate any reading of the numbers:

* the evaluation was supplying **ground-truth entity placement**, which hands the model
  the arrangement a relationship graph would otherwise have to supply;
* the **structure graph was empty**, so the structure-only ablation was a no-graph arm
  that still paid for a graph encoder.

Five defects, each of which would on its own have produced a confident, publishable, and
wrong conclusion about whether a major architectural component is necessary.

The general lesson, which applies beyond this project: **an ablation is only informative
once the input actually differs between the cases the arms are meant to distinguish, and
once the training signal is one the arms can act on.** Neither condition is visible in
the result table. Both have to be checked separately, and all five defects here were
found by checking a number that did not fit rather than by reading the headline.

## 3. What was built

| Component | Location |
| --- | --- |
| Whole-organ generator: organ first, segmented afterwards | `datasets/whole_organ/` |
| Measured relationships, not asserted ones | `datasets/whole_organ/relations.py` |
| Anatomical variants and edit operations | `datasets/whole_organ/parameters.py`, `sampling.py` |
| Per-scene relationship graphs in batches | `generation/neural/nn/whole_organ.py` |
| Level-specific detail supervision | `training/whole_organ.py` |
| Local edit head, two scopes | `generation/neural/nn/editing_head.py` |
| Perturbation harness | `experiments/step7/perturbations.py` |
| Relation-blind floor | `experiments/step7/relation_baseline.py` |
| Convergence criterion | `experiments/step7/convergence.py` |
| Placement-condition evaluation | `experiments/step7/run_placement.py` |
| Result tables rendered from run JSON | `experiments/step7/report_tables.py` |

Every component has tests.

| Check | Result |
| --- | --- |
| `pytest` | **522 passed**, of which **60** are new Step 7 tests |
| `ruff check .` | all checks passed |
| `mypy` tier one, 54 files | no issues |
| `mypy --config-file mypy-prototype.toml` tier two, 53 files | no issues |

Several of those tests exist because a defect got past review and only a test could stop
it recurring: that no family may be locked to one arrangement, that no part target may
name an entity its level hides, that all three typed graphs carry edges, and that
`invert_spatial` is a structural no-op so its zero is never read as a finding.

## 4. The corpus

`datasets/processed/whole_organ_1600_v2`

| Property | Step 6 corpus | Step 7 corpus |
| --- | --- | --- |
| scenes | 2,000 | 1,600 |
| distinct relationship graphs | **1** | **164** |
| generation order | entities built separately, then assembled | organ built whole, then segmented |
| relationships | asserted from the ontology | measured from the organ |
| arrangements | one | four, crossed with every family |
| entities owning volume in every scene | wall layers often empty | all 20, in all 160 test scenes |

The corpus is smaller than Step 6's and carries 164 times more relational variety. That
ratio is the point of Step 7's data work.

## 5. Defect 1 — Step 6's corpus carried one relationship graph

Verified by counting, before any Step 7 model was trained:

```
distinct relation signatures across 2,000 training scenes : 3   (one per level of detail)
distinct relation value sets                              : 1   (all relations true, always)
LOD 1 graph identical to LOD 2 graph                      : yes
```

The relationships came from the ontology, so they were properties of *hearts in general*
rather than of any particular scene. A graph encoder reading a constant input can only
contribute a constant prior. Step 6's conclusion that the graph encoder was not earning
its parameters was therefore **untestable on that corpus**, and is withdrawn as
unsupported rather than confirmed or refuted.

This is what Step 7's whole-organ corpus was built to fix.

## 6. Defect 2 — variant was aliased onto family

The first Step 7 generator used

```python
family_id = index % families            # 40 families
variant   = variants[index % len(variants)]   # 4 variants
```

Four divides forty, so the second expression is a function of the first. **Every family
appeared in exactly one arrangement.** Validation held only `mirrored` and `rotated`
scenes; test held only `normal` and `transposed`.

The whole design rests on showing the same family in different arrangements so that
presence, identity and text are constant and only the graph differs. With the alias,
"different arrangement" always also meant "different family", so any measured difference
could be a family effect. A model ignoring relations was never put under pressure.

Fixed by advancing the variant once per pass over the families. The generator now refuses
to write a corpus where any family covers fewer than every variant, or any non-empty
split is missing one. See [ADR 0015](../adr/0015-crossed-variant-corpus.md).

## 7. Defect 3 — the part-correspondence target named hidden entities

Found while investigating why the appearance baseline never converged. It was not a
baseline problem. Every arm was affected.

The target named each point's true owner among all twenty entities, regardless of which
entities the scene's level of detail exposes. The composition masks absent entities to
`-1e9`, so each such point cost about `1e9`.

| Level | Points naming a hidden entity | Part loss |
| --- | --- | --- |
| 1 | 397 of 1024 (39%) | 3.88e8 |
| 3 | 0 | 3.02 |

Median logged part-correspondence loss across every structured arm: **3.65e8**, above
`1e6` in 13 of 19 logged steps. Gradient clipping at 1.0 kept the runs from diverging,
which is exactly why the other numbers looked plausible and the defect went unnoticed.

A comparison in which every arm spends most of its gradient budget on a term none can
reduce is not a comparison of the arms. The second suite's finding is withdrawn.

Fixed: the part target is restricted to the entities the level exposes, so it agrees with
that level's own occupancy target. Part loss is now 2.4 to 3.1 at every level. See
[ADR 0016](../adr/0016-level-aware-part-target.md).

**The failing arm was the diagnostic.** Had the baseline been dismissed as "appearance
models do badly here", this would have shipped inside a headline result.

## 8. Defect 4 — the evaluation supplied ground-truth placement

The decoder works in each entity's canonical frame. `forward` takes those frames from the
batch unless `use_predicted_frames` is set, and the batch's frames come from the scene.
The evaluation never set it, so the model was handed the true position, scale and
rotation of every entity and asked only what shape to put there. The frame head, whose
job is to predict those quantities, was never exercised at evaluation time.

This resolved three results that did not fit together: a flat profile across the four
arrangements, near-zero counterfactual sensitivity, and a spatial relation accuracy of
0.98 for arms that cannot see spatial edges at all. All three follow if placement is
already given.

| Condition | Entity ownership IoU | Spatial relation accuracy |
| --- | --- | --- |
| placement given | 0.480 | 1.000 |
| placement inferred | 0.247 | 0.962 |

Both conditions are now reported for every arm, from the same weights. Decisions about
graph necessity rest on the **inferred** condition, because in the given condition the
answer a graph would supply has already been provided through another input. See
[ADR 0018](../adr/0018-evaluate-both-placement-conditions.md).

## 9. Defect 5 — the structure graph was empty

The corpus measures spatial and functional relations from the organ. Structural relations
are not measurements: which structures connect to and are continuous with which does not
change when an organ is mirrored. Nothing supplied them, so the structure graph carried
**zero edges**, and ablation `A2` was a no-graph arm paying for a full graph encoder.

Fixed: every scene's graph now includes the ontology's structural edges between the
entities that scene contains, 15 for the whole-organ entity set. All three typed graphs
carry edges, guarded by a test.

**How `A2` must be read even now:** its edges are identical for every scene. A constant
input can act as a prior but cannot distinguish arrangements. `A2` matching a no-graph arm
on arrangement-sensitive metrics is the *expected* outcome and is not evidence against
typed graphs in general. See
[ADR 0017](../adr/0017-structural-graph-is-ontology-knowledge.md).

## 10. Two reading rules that change what the numbers mean

**The relation-blind floor.** `spatial_relation_accuracy` counts the share of measured
spatial relations a prediction reproduces. Many relations survive every arrangement: an
apex stays inferior to a base however the organ is mirrored. A predictor that always emits
the `NORMAL` arrangement, which is exactly what a relation-blind model can do, scores
**0.8878** on the test split. The oracle scores 1.0000.

**The discriminative band is 0.8878 to 1.0000, a headroom of 0.1122.** An accuracy of
0.98 is roughly four fifths of the available headroom, not "almost perfect". Every claim
about relational competence below is stated as the gap above this floor.

**The step budget.** Under the pre-registered convergence criterion, most runs at 900
steps were still improving. An equal-budget comparison at a non-converged point shows
which arm learns *faster*, which is not the same as which ends up *better*. A graph
encoder that helps only late would not show. A longer run on the two decisive arms checks
whether the conclusion holds.

## 11. Experiment 4 and 10 — the corpus, and whether every entity owns volume

**Status: complete. Both succeeded.**

The whole-organ generator builds the organ from global parameters and segments it
afterwards, so no entity is placed independently and no relationship is asserted. The
septum exists because there is tissue between two cavities, not because the ontology says
a heart has one.

Measured ownership over the 160 held-out scenes:

| Entity group | Share of sampled points |
| --- | --- |
| chambers | 38.3% |
| vessels | 31.2% |
| wall layers | 18.3% |
| valves | 10.7% |
| septa | 1.6% |

**Every one of the 20 entities owns volume in all 160 test scenes.** Step 6's problem,
where wall layers owned so few points that per-entity metrics on them were undefined, is
resolved. The fixed priority rule from the pre-registered plan was implemented as written.

The thinnest structure is the interventricular septum at 0.28% of points. It is present
everywhere but small, so its per-entity numbers carry more sampling noise than the rest.

The arrangements are genuine geometry, not labels. Rendered slices under
`experiments/runs/step7-views/variants/` show the mirrored organ as a left-right
reflection of the normal one, and a test asserts that the reflection relationship holds
numerically rather than only to the eye.

## 12. Experiment 10, restated — the metric problem Step 6 left

Step 6 could not report per-entity numbers for the wall layers because they owned almost
no points. Two things were changed:

1. **Ownership rule.** Eight priority levels, fixed in the plan before any run:
   background, pericardium, chamber, valve, vessel, septum, endocardium, epicardium,
   myocardium. Each point goes to the highest-priority class containing it.
2. **Geometry.** Wall layers are derived from the organ envelope and the cavity surfaces
   rather than placed as separate primitives, so they necessarily have thickness.

Result: myocardium 5.9%, pericardium 5.9%, epicardium 4.9%, endocardium 1.6% of sampled
points, each non-zero in every scene. The metric is now defined everywhere it is reported.

## 13. Experiments 5 and 6 — is the relationship graph necessary?

**Status: answered, and the answer is no for the graph transformer and emphatically yes
for the entity axis.**

All five defects fixed. Corpus crossed, part target level-aware, all three typed graphs
populated, full 160-scene held-out evaluation, three seeds on the headline arms.

### Placement supplied, 900 steps

| arm | parameters | entity IoU | part control | relation accuracy | placement error | scene IoU |
| --- | --- | --- | --- | --- | --- | --- |
| A3 full graph | 3,094,551 | 0.6339 ± 0.006 | 0.7218 ± 0.006 | 0.9850 ± 0.001 | 0.0563 | 0.8416 ± 0.003 |
| A3L one layer | 1,094,415 | **0.6383** | **0.7268** | 0.9872 | 0.0567 | 0.8367 |
| A2 structure only | 3,094,551 | 0.6360 | 0.7229 | 0.9833 | 0.0559 | 0.8403 |
| A1 no graph | **427,655** | 0.6304 | 0.7089 | 0.9865 | 0.0562 | 0.8417 |
| A1M no graph, matched | 2,993,735 | 0.6317 ± 0.003 | 0.7173 ± 0.001 | 0.9866 | 0.0563 | 0.8434 ± 0.005 |
| A0 appearance | 3,103,894 | 0.0164 ± 0.005 | 0.0016 ± 0.001 | 0.0000 | 0.2094 | 0.5430 ± 0.005 |

Three-seed arms show the spread; single-seed arms do not.

### Two findings, pointing opposite ways

**The entity axis is essential.** `A0` has the same parameter budget and the same data,
and it reaches 0.54 scene IoU, so it does learn the organ's gross shape. But its
part-level control is 0.0016 against 0.72, a factor of 450. It cannot say which structure
owns which region. Whatever else Step 7 concludes, factorising the scene by entity is
doing the work that makes this a controllable anatomical representation rather than a
shape generator.

**The graph transformer is not earning its parameters.** `A1`, with **427,655
parameters, 7.2 times smaller than `A3`**, matches it: 0.6304 against 0.6339. The gap of
0.0035 is smaller than `A3`'s own seed spread of 0.006. `A1M`, which has `A3`'s capacity
but no way to move information between entities, also matches. `A3L`, with a third of the
graph encoder, scores **highest** of all arms on both entity IoU and part control.

The 2,666,896 parameters of the partitioned relational graph transformer, 86% of the
model, buy a difference indistinguishable from seed noise.

### The per-arrangement profile is flat

Entity IoU by arrangement:

| arm | normal | mirrored | transposed | rotated | spread |
| --- | --- | --- | --- | --- | --- |
| A3 | 0.6535 | 0.6147 | 0.6324 | 0.6767 | 0.0620 |
| A3L | 0.6591 | 0.6186 | 0.6468 | 0.6657 | 0.0471 |
| A2 | 0.6501 | 0.6186 | 0.6411 | 0.6720 | 0.0534 |
| A1 | 0.6604 | 0.6134 | 0.6018 | 0.6742 | 0.0725 |
| A1M | 0.6514 | 0.6191 | 0.6228 | 0.6690 | 0.0499 |

No arm handles the arrangements distinctly better than any other, and the arm with **no
graph at all** has the largest spread. If the graph were carrying the arrangement, the
graph arms should be flatter and higher. They are not.

### The decisive measurement: arrangement response

The four arrangements differ only in their relationship graph. So: how much of the
ground-truth variation between arrangements does each model reproduce?

| Model | graph access | placement given | placement inferred |
| --- | --- | --- | --- |
| A3 | full three-graph | 0.7184 | **0.0359** |
| A3L | one layer, four heads | 0.7213 | **0.0187** |
| A2 | structure only, constant across scenes | 0.7399 | **0.0000** |
| A1 | none | 0.7457 | **0.0000** |
| A1M | none, matched capacity | 0.7313 | **0.0000** |
| A0 | none, no entity axis | 0.0000 | 0.0000 |

With placement supplied every structured arm appears to distinguish the arrangements
well, including the three that cannot see a per-scene graph at all. The entity frames were
supplying them.

With placement inferred the table separates cleanly along exactly the line the
architecture predicts:

* the two arms with access to **per-scene** spatial and functional edges respond: `A3`
  recovers 3.6% of the available arrangement information, `A3L` 1.9%;
* `A2`, whose only edges are the structural ones and are identical in every scene,
  recovers **exactly zero**, as [ADR 0017](../adr/0017-structural-graph-is-ontology-knowledge.md)
  predicted it must;
* `A1` and `A1M`, with no graph, recover exactly zero.

So the graph transformer is doing something no other arm does, and the amount is 3.6% of
what the corpus makes available. That is not what 2,666,896 parameters were spent for.
Halving the response by cutting the encoder to one layer also cut its parameters by two
thirds, which is the shape of a mechanism that is under-delivering rather than one that is
saturated.

This is visible as well as measurable. Under
`experiments/runs/step7-views/predictions/` the normal and mirrored predictions are
reflections of one another. Under `predictions-inferred/` they are the same picture.

### Counterfactual response, the pre-registered metric

| arm | sensitivity | correctness |
| --- | --- | --- |
| A3 | 0.0006 | 0.0340 |
| A3L | 0.0002 | 0.0387 |
| A2 | 0.0000 | 0.0348 |
| A1 | 0.0000 | 0.0282 |
| A1M | 0.0000 | 0.0335 |

Swapping a scene's relationship graph for another arrangement's moves predicted entity
placement by 0.0006 in `A3` and not at all in the arms that cannot see spatial edges.
Correctness sits near 0.03 for every structured arm, which is the rate expected from
entities that happen to lie near the counterfactual position anyway.

`A0`'s correctness of 0.48 is not a result. It places almost nothing, so very few
entities are scored, and the figure is noise on a tiny sample.

Both readings agree: **relations are reaching the model, and the model is very nearly
ignoring them.**

## 14. Experiments 1, 2 and 3 — which parts of the representation are read

**Status: complete, and the most architecturally specific result of Step 7.**

A model trained on intact input, evaluated with one channel damaged at a time. Entities,
presence, text features and points are identical throughout, so any change arrived
through the relationship graph. Placement inferred, since that is the condition where a
graph can matter.

### How much is the relationship graph worth at all?

`A3`, entity ownership IoU:

| Case | IoU | Change |
| --- | --- | --- |
| `A_full` whole representation | 0.3020 | — |
| `C_no_functional` | 0.2856 | −0.0164 |
| `D_no_structure` | 0.2900 | −0.0120 |
| `B_no_spatial` | 0.2614 | −0.0406 |
| `E_entities_only` no relationships at all | 0.2106 | **−0.0914** |

The full graph is worth **0.0914 entity IoU** to a model trained with it, a 30% relative
improvement over the entity axis alone. The spatial graph contributes most, then the
functional graph, then the structural one.

`A1M` and `A1` show a change of exactly 0.0000 for every case, which is the harness
working: arms with no graph encoder cannot be affected by damaging a graph.

### The comparison that matters

`E_entities_only` on `A3` (0.2106) is **worse** than `A1`, which never had a graph
(0.2914). A model trained to depend on relations is damaged by having them removed; a
model that never had them is not. So the honest comparison is full against full:

| Model | entity IoU, placement inferred |
| --- | --- |
| `A3`, full graph, 3,094,551 parameters | 0.3020 |
| `A1`, no graph, 427,655 parameters | 0.2914 |
| `A1M`, no graph, matched capacity | 0.2820 |

The graph transformer buys **+0.011 over a 7.2-times-smaller model** and +0.020 over a
capacity-matched one. Real, consistent in sign, and small.

### Which property of the graph is being read

`A3`, placement inferred, entity ownership IoU against the intact control:

| Perturbation | What it corrupts | IoU | Change | Raw output change |
| --- | --- | --- | --- | --- |
| `intact` | nothing | 0.3020 | — | — |
| `invert_spatial` | nothing the encoder can see | 0.3020 | +0.0000 | 0.000 |
| `invert_spatial_labels` | what the relations say | 0.3022 | **+0.0002** | 0.056 |
| `shuffle_spatial_types` | what the relations say | 0.3020 | **+0.0000** | 0.072 |
| `randomise_spatial_endpoints` | which entities are connected | 0.2655 | **−0.0365** | 20.07 |
| `drop_spatial` | which entities are connected | 0.2614 | **−0.0406** | 20.03 |

`A3L` shows the same pattern at its smaller scale: −0.0110 for `drop_spatial`, −0.0099
for `randomise_spatial_endpoints`, and +0.0000 for both label corruptions.

**Telling the model that every spatial relation is the opposite of the truth changes its
output by 0.0002 entity IoU. Removing the same edges costs 0.0406, two hundred times
more.** On the raw output the ratio is about 350 to 1.

Two corruptions of "what the relations say" and two of "which entities are connected", and
the split is clean in both arms.

The typed relation vocabulary, the relation bias and the head partitioning are the
distinguishing features of the partitioned relational graph transformer over a plain
untyped graph network. **They are close to inert.** The graph is doing useful work as an
adjacency structure and almost none as a typed relational one.

### One perturbation that tests nothing, reported as such

`invert_spatial` replaces each relation with its inverse and swaps the endpoints, which
is the correct way to state an inversion. It produces a bit-identical relation bias,
because the encoder already adds the inverse relation's embedding in the reverse
direction. Its zero is a design property, not a finding, and `invert_spatial_labels` was
added to test what it could not. Exact zeros are worth being suspicious of; this one was
found by being suspicious of two of them.

## 15. Experiments 8 and 9 — persistent local geometric editing

**Status: built and measured. The locality mechanism works. The edit head does not
produce accurate geometry, and that is reported as not working.**

Step 6 shipped the editing interfaces and never tested them. Three mechanisms, one
frozen `A3` backbone, 145 edit cases on held-out scenes:

| Arm | head parameters | target accuracy | locality | drift | persistence | consistency | commutativity gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `E-LOCAL` scoped to the target | 83,777 | 0.1519 | **130.56** | 0.0006 | 0.8691 | 0.9699 | 0.0000 |
| `E-FREE` same head, unscoped | 83,777 | 0.2567 | 4.72 | 0.0057 | 0.8426 | 0.9190 | 0.0002 |
| `E-REGEN` re-decode everything | 0 | **0.4293** | 14.87 | 0.0010 | 0.8298 | 0.9699 | 0.0000 |

### Scoping the edit is what produces locality

`E-LOCAL` and `E-FREE` are the same head with the same 83,777 parameters, trained
identically. The only difference is whether the delta is masked to the edited entity's
geometry token block. Locality differs by a factor of **28** (130.56 against 4.72) and
leakage into untouched entities by a factor of **9.5**.

Locality is therefore a property of the per-entity token block being separately
addressable, not something the head learns on its own. That is direct evidence for the
per-entity geometry design, and it is the first such evidence the project has: Step 6
asserted this property and never measured it.

### Regeneration wins on accuracy, and the reason is not subtle

`E-REGEN` reaches 0.4293 target accuracy against `E-LOCAL`'s 0.1519. Re-decoding the
whole scene from edited parameters produces much more nearly the right geometry, because
the backbone already knows how to render an organ from its description, while the edit
head has to learn to synthesise a shape change from an eight-number instruction against a
frozen decoder.

**So the pre-registered question splits.** `E-REGEN` does *not* match `E-LOCAL` on
locality, so the per-entity mechanism is not marked for removal on that ground. But the
edit head as built cannot make the edit accurately, so **persistent local editing is
reported as not working yet**, not as partially working.

### Two results that are structural, not achievements

* `commutativity_gap` is 0.0000 for `E-LOCAL` and `E-REGEN`. The head has no cross-entity
  mixing and the generator's parameter edits compose commutatively, so order independence
  follows from the construction. `E-FREE`'s 0.0002 is the small exception, which is
  itself consistent: an unscoped head can let one edit influence another.
* `identity_preservation` is exactly 1.0 for all three. The edit head never writes the
  entity latent, so the write-protected identity subspace is carried through untouched by
  construction. It is measured rather than assumed, and it holds.

### Edits persist, mostly

`edit_persistence` is 0.83 to 0.87 across arms: after a second edit to a different entity,
about 85% of the first target's region survives. Not forgotten, not untouched. The
residual is real coupling, since the second edit competes for points at the boundary.

`sequence_consistency` of 0.97 says a spatial relation read off the geometry after the
first edit agrees with the generator's true edited organ almost always.

### What the generator taught us about edits

`shrink_ventricle` was excluded from 31 of 36 training scenes and 23 of 24 evaluation
scenes, because shrinking the left ventricle **deletes a structure**: the interventricular
septum is defined as the tissue between two cavities, and the valve annuli as the regions
between them. Shrink one cavity enough and those structures cease to exist.

This is not a defect in the generator; it is a fact about derived anatomy. It is reported
because a locality measurement compares a before and an after over one entity set, and an
edit that removes an entity has no such comparison. Cases that delete a structure are
counted and excluded, never silently dropped.

## 16. Experiment 7 — level of detail

**Status: complete. The level-specific objective did not fix it. Marked REVISE.**

Step 6 supervised every token prefix against the same full-detail target, so a coarse
prefix was asked for fine geometry it cannot represent. Step 7 supervises each prefix
against that level's own target, which is a request a coarse prefix can satisfy.

Occupancy IoU at each prefix, each against its own level's target:

| arm | level 1 | level 2 | level 3 | detail gain | token delta |
| --- | --- | --- | --- | --- | --- |
| A3 | 0.7314 ± 0.023 | 0.7211 ± 0.031 | 0.5952 ± 0.016 | −0.1362 | 0.1610 |
| A3L | 0.7512 | 0.7154 | 0.5968 | −0.1543 | 0.1677 |
| A2 | 0.7431 | 0.7214 | 0.6136 | −0.1294 | 0.1806 |
| A1 | 0.7024 | 0.7437 | 0.6534 | **−0.0490** | 0.2189 |
| A1M | 0.7261 ± 0.020 | 0.7098 ± 0.027 | 0.6028 ± 0.053 | −0.1233 | 0.1646 |
| A0 | 0.3974 | 0.4450 | 0.4131 | +0.0157 | 0.0034 |

`lod_detail_gain` is negative for every structured arm. Finer token prefixes score
**worse** against their own targets.

### The honest reading, including the part that argues against the conclusion

Level 3 is a harder target than level 1: twenty structures rather than ten. A lower
level-3 IoU is therefore not on its own proof that the extra tokens are wasted, and the
method document said so before the runs.

The quantity that separates the two readings is `lod_token_delta`, which is 0.16 to 0.22
for the structured arms. **The extra tokens do change the output substantially.** They are
not ignored, as they largely were in Step 6. They are used, and the result is worse.

So the level-specific objective fixed the problem it was aimed at, and revealed a
different one: the model uses the finer prefixes and does not use them well.

### Verdict

This is the second objective tried for nested level-of-detail prefixes and the second
that does not deliver improving geometry at finer levels. Marked **REVISE**, with the
specific mechanism recorded: the token prefixes are being read but the model has not
learned a coarse-to-fine decomposition, and the most likely reason is that nothing in the
objective requires the coarse prefix to be a *prefix* of the fine one rather than an
independent summary. That is testable and is recorded as the next thing to try, not
asserted as the answer.

`A0`'s small positive gain is not a counterexample. Its token delta is 0.0034, meaning its
prefixes barely change the output at all; it scores about the same everywhere because it
does about the same thing everywhere.

## 17. Experiment 11 — baseline convergence

**Status: complete, and it is what uncovered the defect that invalidated the previous
suite.**

The criterion was fixed before the runs: converged if the final validation `scene_iou` is
within 0.02 of the best and the best is not the first measurement; not converged if still
improving by more than 0.02 over the last stretch; failed below 0.1.

| Arm | Verdict |
| --- | --- |
| A3 | converged at seeds 1 and 2, still improving at seed 0 |
| A1M | converged at seeds 0 and 1, still improving at seed 2 |
| A1, A2, A3L | converged |
| A0 | **still improving at all three seeds** (0.41 to 0.54) |

Eight of twelve runs converged, against two of twelve in the defective suite. The
structured comparison is therefore made at or near a plateau, which is what makes it
worth reading.

**`A0` has not converged.** Its validation scene IoU was still climbing at 900 steps. The
comparison against it establishes that the entity axis matters for part-level control by a
factor of 450, which is not a margin that more training closes, but the gross-shape
comparison (0.84 against 0.54) is budget-limited and is reported as such rather than as a
settled gap.

### What this experiment was actually worth

In the previous suite `A0` **failed**: final scene IoU of 0.000, 0.000 and 0.133.
Investigating that failure found a part-correspondence loss of 3.65e8 affecting every arm,
which invalidated the whole suite including its headline graph result.

**The failing arm was the diagnostic.** Had the baseline been waved away as "appearance
models do badly on this task", a defect that broke every arm's training would have shipped
inside a confident claim about whether a major architectural component is necessary. An
experiment whose only purpose is to check that the comparison is fair paid for itself
several times over.

## 18. Architecture decisions

Assigned by the rules in [10_architecture_decision.md](10_architecture_decision.md), which
were written before this section. Every verdict names the evidence and the condition it
holds under.

| Component | Verdict | Evidence |
| --- | --- | --- |
| Anatomical World Representation as source of truth | **KEEP** | the appearance baseline with the same parameter budget reaches 0.0016 part control against 0.72; a factor of 450 |
| Entity axis / per-entity factorisation | **KEEP** | same evidence; this is what makes the scene addressable |
| Per-entity geometry token blocks | **KEEP** | scoping an edit to one block gives locality 130.6 against 4.72 for the identical head unscoped, a factor of 28 |
| Write-protected identity subspace | **KEEP** | identity preservation is exactly 1.0 through every edit, by construction and verified |
| Geometry correspondence as a tensor axis | **KEEP** | part control of 0.72 with no learned association step; the winning channel names the owner |
| Spatial relationship graph | **KEEP, reduced** | worth −0.041 entity IoU when deleted; the largest single relational contribution |
| Functional relationship graph | **KEEP, reduced** | worth −0.016 when deleted; small but consistent |
| Structural relationship graph | **KEEP as a prior** | worth −0.012 when deleted, but it is identical for every scene, so it cannot carry arrangement |
| **Partitioned relational graph transformer** | **SIMPLIFY** | see below |
| **Typed relation vocabulary and relation bias** | **REMOVE** | the model is ~350× more sensitive to which entities are connected than to what the relation says; shuffling every relation type moves entity IoU by <0.0001 |
| Head partitioning by graph type | **REMOVE** | follows from the above; the partitioning exists to keep typed relations separate, and the types are inert |
| Nested level-of-detail token prefixes | **REVISE** | detail gain is −0.14 while token delta is 0.16; the prefixes are read and used badly |
| Persistent local editing | **REVISE** | locality works (130.6), accuracy does not (0.15 against regeneration's 0.43) |
| Prototype-hub multimodal alignment | **OPEN** | not isolated by any Step 7 experiment; no evidence either way |

### The graph transformer: SIMPLIFY, with the numbers

It costs 2,666,896 parameters, 86% of the model. What it delivers:

| Measure | With it (A3) | Without it (A1, 7.2× smaller) |
| --- | --- | --- |
| entity IoU, placement given | 0.6339 ± 0.006 | 0.6304 |
| entity IoU, placement inferred | 0.3020 | 0.2914 |
| arrangement response, inferred | 0.0359 | 0.0000 |

Two things are true at once and both must be said.

**It is doing something no other arm does.** Only the arms with access to the per-scene
spatial and functional graphs respond to arrangement at all: `A3` 3.6%, `A3L` 1.9%, and
exactly 0.0% for `A2`, `A1`, `A1M`. Deleting the graph from a model trained with it costs
0.09 entity IoU. The mechanism is not inert.

**What it delivers is far below what it costs.** 3.6% of the available arrangement signal.
A difference of 0.0035 in the given condition, smaller than the seed spread. `A3L`, with
one layer and a third of the parameters, scores highest of all arms on entity IoU and part
control while retaining half the arrangement response.

**Verdict: SIMPLIFY.** Keep a relational mechanism, because the ablations show it
contributes and nothing else recovers arrangement at all. Remove the typed machinery, the
relation bias and the head partitioning, because the perturbation sweep shows the model
reads connectivity and not relation type. Size it like `A3L` or smaller. The specific next
configuration to test is an untyped graph attention over the union of the three
adjacencies, at roughly `A3L`'s budget.

### The convergence check, which was declared in advance and sharpens both sides

Most runs at 900 steps were still improving, so amendment A13 declared a longer run on the
two decisive arms before seeing it. Three times the budget, one seed:

| Arm | 900 steps | 2,700 steps |
| --- | --- | --- |
| A3, full graph, 3,094,551 | 0.6339 | 0.6659 |
| A1M, no graph, 2,993,735 | 0.6317 | **0.6843** |

At the short budget they are tied. At the long budget **`A1M` pulls ahead by 0.018**. The
pre-registered rule for this case reads: "If `A1M` pulls ahead, the finding against the
graph encoder strengthens." It does. On this evidence the graph transformer is not merely
surplus; at equal capacity, spending those parameters on per-entity depth instead produces
a better model, and the advantage grows with training rather than shrinking.

**And the measurement that argues the other way, reported because it does.** The graph's
arrangement response nearly doubles over the same interval:

| Arm | arrangement response at 900 steps | at 2,700 steps |
| --- | --- | --- |
| A3 | 0.0359 | **0.0647** |
| A1M | 0.0000 | 0.0000 |

The relational mechanism is being learned, and it is being learned *slowly*. At three
times the budget it recovers 6.5% of the available arrangement information instead of
3.6%. The trend is upward and Step 7 does not establish where it flattens.

So the SIMPLIFY verdict carries an explicit caveat: it holds at the budgets tested, where
the graph's growing relational competence does not pay for its parameters. A budget large
enough to change that was not run and is not ruled out. What Step 7 does establish is that
nothing at this scale justifies 2.67M parameters, and that the typed machinery inside them
is inert regardless of budget, since relation type is not read at either budget.

This verdict supersedes Step 6's "not earning its parameters", which is withdrawn as
untestable rather than confirmed. The direction is similar; the evidence is different and
this time the comparison could have come out the other way.

## 19. The placement conditions, and the single clearest result in Step 7

Every checkpoint scored twice from the same weights: with entity frames supplied, and
with the model predicting placement itself.

### Entity ownership IoU

| arm | placement given | placement inferred | cost of inferring |
| --- | --- | --- | --- |
| A3 | 0.6339 | **0.3020** | 0.3319 |
| A3L | 0.6383 | 0.2955 | 0.3429 |
| A2 | 0.6360 | 0.2931 | 0.3429 |
| A1 | 0.6304 | 0.2914 | 0.3390 |
| A1M | 0.6317 | 0.2857 | 0.3460 |
| A0 | 0.0164 | 0.0164 | 0.0000 |

Inferring placement costs about **half the ownership accuracy** for every structured arm.
`A0` is unchanged because it has no per-entity frames to be given.

### Spatial relation accuracy, against the floor

| arm | placement given | placement inferred |
| --- | --- | --- |
| A3 | 0.9850 | 0.8472 |
| A3L | 0.9872 | 0.8491 |
| A2 | 0.9833 | 0.8466 |
| A1 | 0.9865 | 0.8430 |
| A1M | 0.9866 | 0.8478 |
| **relation-blind floor** | — | **0.8878** |
| **oracle** | — | **1.0000** |

**Every arm scores below the relation-blind floor when it must infer placement.**

The blind predictor emits the `NORMAL` arrangement of the scene's own family whatever the
scene actually is, which is precisely what a model with no relational information can do.
It scores 0.8878. The best trained model scores 0.8491.

So the honest statement is not that the models have limited relational competence. It is
that **on this measure they have none, and are worse than a fixed arrangement-blind
guess.** Predicting an averaged organ is worse than committing to one specific wrong
arrangement, because the average satisfies fewer of any particular scene's relations than
a clean arrangement does.

Without the floor, 0.847 reads as "pretty good". With it, the number reverses its meaning.
This is why [ADR 0019](../adr/0019-relation-blind-floor.md) exists, and it is the single
most valuable reporting decision in Step 7.

### Why the two conditions differ so much

Entity frames carry translation, log-scale and a 6D rotation per entity. That is the
arrangement, handed over directly. With them supplied, a model that cannot see spatial
edges at all reproduces 0.67 of the arrangement variation and scores 0.9866 on relation
accuracy. Neither number tells you anything about relational reasoning.

Any future report of a relation metric from this codebase must state the placement
condition. A number without it is not interpretable.

## 20. What is withdrawn, and what Step 7 does not establish

### Withdrawn

| Claim | Where it was made | Why it is withdrawn |
| --- | --- | --- |
| "The typed graph encoder is not earning its parameters" | Step 6 experiment report | the corpus carried one relationship graph across all 2,000 scenes; the comparison was untestable, not refuted |
| "`A1` matches `A3`" | Step 7 suite 1 | variant was aliased onto family; the intended contrast was never presented |
| "`A1` matches `A3`" | Step 7 suite 2 | the part target named entities the level hides; every arm spent most of its gradient budget on an unoptimisable term |
| "The appearance baseline fails" | Step 7 suite 2 | it failed because of the same defect; with it fixed, `A0` reaches 0.54 scene IoU |

Step 7's own conclusion about the graph transformer is **similar in direction** to the
withdrawn Step 6 one. That is not a vindication of the earlier claim. The earlier claim
could not have come out any other way; this one could have, and the corpus was built
specifically so that it could.

### Limitations, stated plainly

**The data is procedural.** Forty families from one generator. Held-out families test
interpolation inside that generator, not generalisation to organs built differently.
Nothing here bears on human anatomy.

**Four arrangements is a small set.** A model could learn to classify which of four
arrangements it is looking at without learning anything general about spatial relations.
Nothing in Step 7 distinguishes those two possibilities.

**The step budget binds on the baseline.** `A0` had not converged at 900 steps, so the
gross-shape gap of 0.84 against 0.54 is budget-limited. The part-control gap of 0.72
against 0.0016 is a factor of 450 and is not a margin more training closes, but it is
reported at a fixed budget like everything else.

**The graph verdict is for this configuration.** It says a partitioned typed relational
transformer at this scale, on this data, at this budget, contributes about 3.6% of the
available arrangement signal and reads connectivity rather than relation type. It does not
say typed relations cannot work. A different objective, a corpus where relations are the
only route to the answer, or a curriculum that forces relational inference might all
change it. Those are testable and untested.

**One editing backbone, one seed.** The editing experiments use a single frozen `A3`
checkpoint. The locality contrast between scoped and unscoped is a factor of 28 and is
unlikely to be a seed effect, but it has not been shown across seeds.

**Alignment was never isolated.** No Step 7 experiment separates the prototype-hub
multimodal alignment objective, so it has no verdict.

### What would change these conclusions

Stated now, so that a future result cannot be presented as a surprise:

* If an untyped graph at `A3L`'s budget matches `A3`, the SIMPLIFY verdict is confirmed.
  If it does **not** match, the typed machinery was doing something the perturbation
  sweep failed to detect, and the REMOVE verdict on relation types is withdrawn.
* If training with predicted frames from the start lifts any arm above the relation-blind
  floor, the "no relational competence" finding becomes a training-procedure finding
  rather than an architectural one.
* The longer-budget check was run and is reported in section 18: at 2,700 steps `A1M`
  pulls ahead rather than falling behind, and the graph's arrangement response rises from
  3.6% to 6.5%. If a budget several times larger again reverses the accuracy ordering, the
  SIMPLIFY verdict becomes budget-dependent and is amended. Step 7 does not establish
  where the arrangement-response trend flattens.

## 21. What Step 7 did not do, and what comes next

### Did not do

No real medical data. No medical validation. No clinical claim. No large-scale training.
No production model. No billions of parameters. No multimodal real-data training. No
deployment. No external 3D generation system, as a component or as a baseline. The largest
model here is 3.1M parameters and the longest run is 2,700 steps on a laptop CPU.

Nothing in this repository is anatomy. It is a synthetic testbed built to make
architectural questions answerable, and its only claim is about which parts of a proposed
architecture do measurable work on that testbed.

### The three things worth doing next, in order

1. **Replace the graph transformer with an untyped graph attention at `A3L`'s budget.**
   The perturbation sweep says the model reads connectivity and not relation type, so the
   typed machinery, the relation bias and the head partitioning should come out. This is
   a direct test of the SIMPLIFY verdict and would return roughly 2.5M parameters.
2. **Train with predicted placement.** Every model here was trained with ground-truth
   frames supplied and is barely able to place structures without them, scoring below the
   relation-blind floor. That is the single largest gap Step 7 found, and it is plausibly
   a training-procedure problem rather than an architectural one.
3. **Make the level-of-detail prefixes nest.** The prefixes are read and used badly.
   Nothing in the objective requires the coarse prefix to be a *prefix* of the fine
   decode rather than an independent summary. That is one testable change.

### The lesson worth carrying forward

Five defects, each of which would on its own have produced a confident and wrong
conclusion about whether a major architectural component is necessary. All five were
found by following a number that did not fit, never by reading the headline result:

* a relationship graph that never varied;
* a variant aliased onto family;
* a loss term of 3.65e8 hiding behind gradient clipping;
* an evaluation handing the model the answer through a second input;
* an entire typed graph that was empty.

The headline table looked reasonable in every one of those states. **An ablation is only
informative once the input actually differs between the cases the arms are meant to
distinguish, and once the training signal is one the arms can act on.** Neither condition
is visible in a result table. Both have to be checked on purpose.

---

**STEP 7 COMPLETE.** As instructed, work stops here.
