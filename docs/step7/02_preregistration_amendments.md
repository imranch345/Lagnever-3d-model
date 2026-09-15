# Step 7 — Amendments to the pre-registered plan

Written **after** the first whole-organ suite finished and **before** the corrected
suite and the editing experiments were run. Every amendment is recorded here with its
reason, so that the difference between "planned" and "changed after seeing numbers" is
visible rather than buried.

Nothing in this file changes a metric definition from
[`01_experimental_plan.md`](01_experimental_plan.md). Two amendments fix defects in the
data and the evaluation sampling; two add controls and arms that make the existing,
unchanged metrics interpretable.

---

## A1. Dataset defect: variant and family were confounded

**Found:** after the first suite, while checking why the counterfactual sensitivity was
near zero for every arm.

The corpus generator assigned

```python
family_id = index % families            # 40 families
variant   = variants[index % len(variants)]   # 4 variants
```

Because 4 divides 40, the second line is a function of the first. **Every family
appeared in exactly one arrangement.** The validation split therefore held only
`mirrored` and `rotated` scenes and the test split only `normal` and `transposed`.

Why this invalidates the graph conclusion: the whole design of Experiments 5 and 6 rests
on presenting the *same* organ family in *different* arrangements, so that presence,
entity identity and text are constant and the relationship graph is the only channel
that distinguishes the cases. With the confound, "different arrangement" always also
meant "different family", so any measured difference could be a family effect. A model
that ignores relations entirely was never actually put under pressure.

**Fix:** the variant now advances once per full pass over the families, which removes
the aliasing for any family and variant count:

```python
family_id = index % families
variant   = variants[(index // families) % len(variants)]
```

Every family is now seen in all four arrangements ten times each, and every split holds
all four arrangements. The generator refuses to write a corpus where any family covers
fewer than every variant, or any non-empty split is missing a variant, and it records
the crossing in the manifest.

**Consequence for reporting:** the first suite is kept in
`experiments/runs/step7-whole-organ-v1-confounded/` with a `SUPERSEDED.md` explaining
why. Its numbers are reported in the completion report as a withdrawn measurement. They
are not used for any KEEP / SIMPLIFY / REVISE / REMOVE decision.

## A2. Evaluation sampling defect: the final evaluation was truncated

**Found:** at the same time.

The final evaluation was capped at a batch count, and the evaluation loader walks the
level-of-detail buckets in order without shuffling. The cap therefore stopped partway
through the split, biasing the reported numbers toward whichever levels and
arrangements happened to come first.

**Fix:** the final, reported evaluation now runs over the **entire** held-out split.
The batch cap survives only for the cheap periodic checks during training. The number of
scenes actually evaluated is recorded as `eval_scenes` in every run, so a truncated
evaluation cannot be mistaken for a complete one again.

## A3. New control: the relation-blind floor

**Reason:** `spatial_relation_accuracy` is hard to read on its own, because many
measured relations survive every arrangement. An apex stays inferior to a base however
the organ is mirrored. A model that ignores the relationship graph completely therefore
does not score zero on this metric; it scores whatever the invariant relations are
worth. Reading a high raw accuracy as relational competence would be a mistake.

**What is added:** a reference point computed from the corpus alone, with no model and
no arm involved (`experiments/step7/relation_baseline.py`).

| Reference | Predictor | Meaning |
| --- | --- | --- |
| `oracle` | the scene's own measured centroids | must be 1.0, or the scoring code is what is broken |
| `blind_normal_arrangement` | the `NORMAL` arrangement of the scene's own family, whatever variant the scene is | exactly what a relation-blind model can achieve |

On the corrected corpus the oracle scores 1.0000 and the blind predictor scores 0.8878
on the test split. **The discriminative band of this metric is therefore 0.8878 to
1.0000, a headroom of 0.1122.** Claims about relational competence must be stated as
the gap above the blind floor, never as the raw accuracy.

This does not redefine `spatial_relation_accuracy`. It states which values of the
unchanged metric correspond to using no relational information.

## A4. New breakdown: metrics per arrangement

**Reason:** the direct test of whether a model uses the relationship graph is whether
it handles the four arrangements differently. A model that cannot see relations must do
worse on the arrangements that depart from the commonest one.

**What is added:** the headline metrics are reported per arrangement as well as pooled
(`variant_<name>_<metric>`). A flat profile across arrangements is evidence that the
graph is unused or unnecessary. A sloped profile is evidence it is being used. Both are
reportable results; a flat profile is the negative result and is reported as such.

## A5. Undefined metric values

**Reason:** `placement_error` has no value for an arm that places no entity at all,
which is what the appearance baseline does. Averaging crashed on the resulting NaN.

**Rule, fixed now:** an undefined value is recorded as NaN for that seed, excluded from
the mean and spread, and counted in `defined_n`. A summary computed from two of three
seeds is never presented as though it came from three. Dropping the failing run instead
would flatter the arm that failed.

## A6. Editing arms (Experiment 8)

The two editing metrics, `local_edit_locality` and `edit_target_accuracy`, are unchanged
from the original plan. The arms below are declared now, before the editing runs, so
that the comparison is fixed in advance.

| Arm | Edit mechanism | What it tests |
| --- | --- | --- |
| `E-LOCAL` | the edit head may rewrite **only** the target entity's geometry token block | locality built into the representation |
| `E-FREE` | the same head, same capacity, may rewrite **every** entity's token block | whether locality has to be built in, or is simply learned |
| `E-REGEN` | no edit head: re-decode the whole scene from the edited parameters | the "just regenerate everything" alternative the design is meant to beat |

`E-REGEN` is the control that matters for the architecture decision. If it matches
`E-LOCAL` on locality, then per-entity geometry token blocks are not buying persistent
local editing and the claim should be withdrawn.

Additional measurements, defined now:

| Name | Definition |
| --- | --- |
| `unrelated_entity_drift` | mean absolute predicted-occupancy change over entities the generator's own measurement says the edit did not alter |
| `identity_preservation` | cosine similarity of the write-protected identity subspace `[0, 96)` before and after an edit; must be exactly 1.0 |

## A7. Edit → reason → edit loop (Experiment 9)

Defined now, before running.

The loop applies edit 1, reads a spatial relation off the resulting geometry, then
applies edit 2 to a different entity, and checks that edit 1 survived.

| Name | Definition |
| --- | --- |
| `edit_persistence` | IoU between the first target's predicted occupancy after edit 2 and its predicted occupancy after edit 1. An edit that is forgotten when the next one arrives is not persistent. |
| `sequence_consistency` | share of steps where the relation read after edit 1 agrees with the relation measured from the generator's true post-edit-1 organ |
| `commutativity_gap` | absolute difference in final scene occupancy between applying the two edits in one order and the other, where the two edits touch disjoint entities and the order should not matter |

## Success and failure, restated

A negative result is a result. Specifically:

* If `E-REGEN` matches `E-LOCAL`, the per-entity edit mechanism is reported as
  unnecessary and marked SIMPLIFY or REMOVE.
* If the per-arrangement profile is flat and the counterfactual sensitivity stays near
  zero on the corrected corpus, the graph transformer is reported as not earning its
  parameters and marked SIMPLIFY or REMOVE. It will not be retained on the grounds that
  it ought to help.
* If `edit_persistence` is low, persistent editing is reported as not working, not as
  partially working.

---

## A8. The evaluation was supplying ground-truth placement

**Found:** after the corrected suite had started, while checking why the
per-arrangement profile was flat and the counterfactual sensitivity was near zero on a
corpus where the arrangement genuinely varies.

The model's decode takes each entity's canonical frame, and `forward` uses
`batch.entity_frames` unless `use_predicted_frames` is set. Those frames come from the
scene, so **the evaluation was handing the model the true position, scale and rotation
of every entity** and asking only what shape to put there. The frame head, which exists
to predict exactly those quantities, was never exercised at evaluation time.

This explains the three results that did not fit together:

* the flat profile across arrangements, since the frames supply the arrangement equally
  well for all four;
* the near-zero counterfactual sensitivity, since swapping the relationship graph cannot
  matter much when placement is already given;
* a spatial relation accuracy of 0.98 for arms that cannot see spatial edges at all.

Measured on a trained `A3` checkpoint, 32 held-out scenes:

| Condition | Entity ownership IoU | Spatial relation accuracy |
| --- | --- | --- |
| placement given (`use_predicted_frames=False`) | 0.480 | 1.000 |
| placement inferred (`use_predicted_frames=True`) | 0.247 | 0.962 |

Ownership IoU halves. A conclusion about whether the relationship graph is necessary
cannot be drawn from the *given* condition, because in that condition the answer the
graph would supply has already been provided through another input.

**Fix, and why it is an addition rather than a change:** both conditions are now
reported for every arm.

| Condition | What it measures |
| --- | --- |
| `given` | shape and ownership quality when placement is supplied. The original condition, unchanged. |
| `inferred` | whether the model knows *where* structures go, which is where a relationship graph could earn its parameters. |

No metric definition changes. The same metrics are computed under a second condition,
and the **inferred** condition is the one the graph-necessity decision rests on. The
given condition is retained and reported, not discarded.

This evaluation runs from the saved checkpoints, so it costs no retraining and the
trained weights being judged are identical in both conditions.

## A9. Two editing results that are exact for structural reasons

Observed in the editing smoke runs and reported as structural facts rather than as
measurements that came out well:

* `commutativity_gap` is exactly zero for both `E-LOCAL` and `E-REGEN`. The edit head
  has no cross-entity mixing, so a delta written to one entity's token row cannot affect
  another's, and the generator's parameter edits compose commutatively. Order
  independence here is a property of the construction, not evidence that a model learned
  it.
* `unrelated_entity_drift` would be *identically* zero for `E-LOCAL` if measured on
  independent per-entity fields, because that arm leaves other entities' token blocks
  untouched bit for bit. The locality metric is therefore measured on the **composed**
  scene, where entities compete for each point. That is what the plan's "occupancy
  change" refers to and the only version of the quantity that is comparable across arms.

---

## A10. The part-correspondence target named entities the level hides

**Found:** while checking Experiment 11, why the appearance baseline never converged.
It turned out not to be a baseline problem at all. Every arm was affected.

The part-correspondence target named each point's true owner among all twenty entities,
regardless of which entities the scene's level of detail exposes. The scene composition
masks absent entities to `-1e9`, so every such point cost about `1e9` in cross-entropy.

| Level | Points naming a hidden entity | Part loss |
| --- | --- | --- |
| 1 | 397 of 1024 (39%) | 3.88e8 |
| 2 | some | large |
| 3 | 0 | 3.02 |

Across the logged training steps of every structured arm the median part-correspondence
loss was **3.65e8**, above 1e6 in 13 of 19 logged steps. Gradient clipping at 1.0 stopped
the runs diverging, which is why the other numbers looked plausible, but those steps
contributed a clipped update dominated by an objective no arm could reduce.

**Why this invalidates the second suite:** a comparison in which every arm spends most of
its gradient budget on an unoptimisable term is not a comparison of the arms. The
observation that `A1` matched `A3` is therefore **withdrawn as unsupported**, for the
same reason the Step 6 graph conclusion was withdrawn. So is the appearance baseline's
failure: a baseline that never converged is not evidence that structure helps.

**Fix:** the part target is restricted to the entities the active level exposes; a point
owned by a finer structure reads as background, which is exactly what that level's own
occupancy target already says about it. The part loss is now 2.4 to 3.1 at every level.
Two regression tests guard it.

This was a Step 7 regression, not a Step 6 one. Step 6's sampler derived ownership from
the visible primitives only, so its targets could not name a hidden entity.

## A11. The structural graph was empty

**Found:** while building the Experiment 3 partial-representation cases, by counting the
edges of each typed graph in a batch.

The whole-organ corpus measures spatial and functional relations from the generated
organ. Structural relations are not measurements; which structures connect to and are
continuous with which is ontology knowledge that does not change when an organ is
mirrored. Nothing supplied them, so **the structure graph carried zero edges**, and the
structure-only ablation `A2` was a no-graph arm that still paid for a graph encoder. Its
matching the other arms in the first two suites means nothing.

**Fix:** every scene's graph now includes the ontology's structural edges between the
entities that scene contains, 15 of them for the whole-organ entity set. All three typed
graphs now carry edges, guarded by a test.

**How to read `A2` now:** the structural edges are identical for every scene, by their
nature. A constant input can act as a prior but cannot distinguish one arrangement from
another. `A2` matching a no-graph arm on arrangement-sensitive metrics is therefore the
*expected* outcome and is not evidence against typed graphs in general; only the spatial
and functional graphs vary per scene and only they can carry arrangement information.

## A12. Perturbation categories for Experiments 1, 2 and 3

Declared before running. All are applied at evaluation time to a model trained on intact
input, and all alter relationship tensors only: entities, presence, text features and
sampled points are untouched, so any change in the output arrived through the graph.

| Perturbation | What it does |
| --- | --- |
| `intact` | control |
| `drop_spatial` | every spatial edge removed |
| `invert_spatial` | every spatial relation replaced by its inverse, endpoints swapped with it |
| `shuffle_spatial_types` | spatial relation types permuted among the same edges |
| `randomise_spatial_endpoints` | spatial edges rewired to random entity pairs |
| `drop_functional` | every functional edge removed |
| `drop_structure` | every structural edge removed |

Partial-representation cases for Experiment 3:

| Case | Graphs kept |
| --- | --- |
| `A_full` | structure, spatial, functional |
| `B_no_spatial` | structure, functional |
| `C_no_functional` | structure, spatial |
| `D_no_structure` | spatial, functional |
| `E_entities_only` | none |

**Reading rule, fixed now:** a channel the model depends on must degrade the output when
corrupted. A channel that can be deleted with no measurable effect is not being used,
whatever role the architecture assigns it, and its component is marked SIMPLIFY or
REMOVE.

---

## A13. 900 steps is not convergence, and the comparison must say so

**Found:** by applying the Experiment 11 criterion mechanically to the second suite's
manifests, before the corrected suite finished.

Under the pre-registered criterion, most structured runs at 900 steps are **not
converged**: validation `scene_iou` was still improving by more than the 0.02 tolerance
between step 600 and step 900. Only two of twelve runs classified as converged.

This does not invalidate the comparison, because every arm received an identical budget
and the checkpoint-selection rule was fixed in advance. It does limit what the comparison
can claim. An equal-budget comparison at a non-converged point shows which arm learns
**faster**, which is not the same as which arm ends up **better**. A graph encoder that
helps only late would not show here.

**What is added, declared now:** after the six-arm suite, a focused convergence check
trains the two decisive arms, `A3` and `A1M`, for three times the budget at one seed, and
reports whether the gap between them changes. The rule for reading it is fixed now:

* If `A3` and `A1M` are still within noise of each other at the longer budget, the
  graph-necessity conclusion holds and is reported as holding at convergence.
* If `A3` pulls ahead, the conclusion is reported as **budget-dependent**: the graph
  encoder helps, but only with more training than the equal-budget comparison allowed,
  and the headline finding is amended accordingly.
* If `A1M` pulls ahead, the finding against the graph encoder strengthens.

Every headline table states the step budget and the convergence verdict alongside the
numbers. A result reported without its convergence verdict is incomplete.

---

## A14. Arrangement response, a diagnostic added after the fact

**Disclosed as post-hoc.** This was written after the placement-condition finding (A8),
not before the runs. It is reported **alongside** the pre-registered counterfactual
metrics, never in place of them.

### Why it was added

`counterfactual_relation_sensitivity` measures how far predicted entity *centroids* move
when the relationship graph is swapped. It is a real measurement, but a small number is
hard to interpret: it could mean the model ignores relations, or that centroids are a
coarse summary of a change that did happen.

The arrangement response measures the same thing on the composed scene and normalises it
by how much the truth itself varies, which makes it readable as a fraction.

### Definition

Take one organ family. Render it in all four arrangements. Presence, entity identity and
text features are identical across the four by construction, so the relationship graph is
the only input that differs.

* **truth disagreement**: mean pairwise share of points that two arrangements' true
  ownership labels differ on.
* **prediction disagreement**: the same quantity computed on the model's predicted
  ownership.
* **response ratio**: prediction disagreement divided by truth disagreement.

A ratio near 1 means the model distinguishes arrangements about as well as the truth
does. A ratio near 0 means it produces one average organ regardless of what the graph
says.

### Why it settles the placement question visually as well as numerically

On a trained `A3` checkpoint, one family, at 96 by 96 slice resolution:

| Source | Mean pairwise disagreement between arrangements |
| --- | --- |
| ground truth | 0.1026 |
| `A3` prediction, placement given | 0.0894 |
| `A3` prediction, placement inferred | 0.0067 |

With placement supplied the model appears to distinguish the arrangements. It does not:
the entity frames were supplying them. With placement inferred, the four predictions are
nearly identical while the four truths are not.

Rendered slices of this are under `experiments/runs/step7-views/predictions/` and
`predictions-inferred/`. In the first the normal and mirrored predictions are visibly
reflections of each other. In the second they are visibly the same picture.

---

## A15. `invert_spatial` is a structural no-op, and what replaced it

**Found:** while checking why `invert_spatial` and `shuffle_spatial_types` both produced
a change of exactly 0.0000 in the perturbation sweep. Exact zeros deserve suspicion.

`invert_spatial` replaces each spatial relation with its inverse **and swaps the
endpoints**, because the inverse of a relation is the same edge read backwards. That is
the right way to state an inversion, and it is invisible to this encoder: the graph
encoder already adds the inverse relation's embedding in the reverse direction, so
`(a, left_of, b)` and `(b, right_of, a)` produce a **bit-identical** relation bias.

Measured on a trained `A3` checkpoint: bias change `0.000000`, output change `0.000000`.

A null result from this perturbation therefore says nothing about whether the model reads
relation types. It says the encoder treats a relation and its inverse as the same fact,
which is a design property and arguably a correct one.

**Added:** `invert_spatial_labels` flips the relation and leaves the endpoints in place,
so the graph now asserts the opposite of what the organ shows. This is visible to the
encoder and is the perturbation that tests relation sensitivity. It states something the
ontology forbids, which is the point of a corruption test.

Both are now reported, with `invert_spatial` labelled as a structural no-op so its zero
is not read as a finding. A test asserts the no-op property directly, so that it is
documented behaviour rather than a surprise.

### What the corrected sweep shows

Output change on a trained `A3` checkpoint, held-out scenes, placement inferred:

| Perturbation | What it corrupts | Output change |
| --- | --- | --- |
| `invert_spatial` | nothing the encoder can see | 0.000 |
| `invert_spatial_labels` | what the relations say | 0.056 |
| `shuffle_spatial_types` | what the relations say | 0.072 |
| `randomise_spatial_endpoints` | which entities are connected | 20.07 |
| `drop_spatial` | which entities are connected | 20.03 |

**The model is about 350 times more sensitive to which entities are connected than to
what the relation between them says.** The typed relation vocabulary and the relation
bias, which are the distinguishing feature of the partitioned relational graph
transformer, are close to inert. The graph is functioning as an untyped adjacency
structure.

---

## A13 result — the convergence check, reported against the rule declared in A13

Run as declared: the two decisive arms, three times the budget, one seed.

| Arm | 900 steps | 2,700 steps |
| --- | --- | --- |
| `A3` full graph | 0.6339 | 0.6659 |
| `A1M` no graph, matched capacity | 0.6317 | **0.6843** |

A13 fixed three readings in advance. The third applies: **`A1M` pulls ahead, so the
finding against the graph encoder strengthens.**

The arrangement response over the same interval moves the other way and is reported
because it does:

| Arm | 900 steps | 2,700 steps |
| --- | --- | --- |
| `A3` | 0.0359 | 0.0647 |
| `A1M` | 0.0000 | 0.0000 |

The relational mechanism is being learned, slowly, and Step 7 does not establish where
that trend flattens. The SIMPLIFY verdict therefore carries the caveat that it holds at
the budgets tested. The REMOVE verdict on the typed relation machinery does not need that
caveat, because relation type is unread at both budgets.
