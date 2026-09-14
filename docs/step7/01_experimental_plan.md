# Step 7 experimental plan

**Written before any Step 7 experiment was run. Metric definitions and ownership rules
below are fixed in advance and are not to be changed after seeing results.**

---

## 0. The audit finding that reframes Step 7

Step 6 concluded that the typed graph encoder was worth about one point of part control
and asked whether it should be removed. The audit found why that comparison could not
have gone any other way:

```
distinct relation signatures across 2,000 training scenes : 3   (one per level of detail)
distinct relation value sets                              : 1   (all relations true, always)
AWR edge tensors identical for every scene at a given LOD : yes
LOD 1 graph identical to LOD 2 graph                      : yes
```

**Every scene in the Step 6 corpus carried the same relationship graph.** It came from
the ontology, not from the scene. A graph encoder reading a constant input can only
contribute a constant prior, so A1 (no graph) matching A3 (full graph) was guaranteed by
the experimental design rather than being evidence about relational reasoning.

Step 6's conclusion "the graph encoder is not earning its parameters" is therefore
**withdrawn as unsupported**. It was neither confirmed nor refuted; it was untestable on
that corpus.

Step 7's first job is to build data where relations carry scene-specific information, so
the question becomes answerable.

## 1. Design principle for Step 7 data

An experiment can only show that relations matter if **entity identity alone is
insufficient**. The Step 7 corpus therefore contains anatomical **variants** in which the
same entity set appears in different arrangements:

| variant | what changes | what the relations say |
| --- | --- | --- |
| `normal` | reference arrangement | left chambers on the anatomical left, and so on |
| `mirrored` | the organ is reflected across the midline | `left_of` / `right_of` invert |
| `transposed` | the great arteries swap their ventricles | aorta connects to the right ventricle |
| `rotated` | the organ is rotated about the vertical axis | `anterior_to` / `posterior_to` change |

Presence is identical across variants. Text features are identical. The **only** channel
carrying the variant is the relationship graph. A model that cannot read the graph must
produce the average arrangement and will be measurably wrong on three variants out of
four. That is the causal test the Step 6 corpus could not provide.

## 2. Whole-organ generation, and why it is different

Step 6 generated each entity independently and assembled them, which biases the
comparison toward an entity-factorised representation. Step 7 reverses the order:

1. build the **whole-organ field** from global parameters;
2. carve **cavities** out of it in the organ's own frame;
3. **derive** valves as the annulus regions between adjacent cavities;
4. **derive** septa as the myocardium that lies between two cavities;
5. attach vessels at openings computed from the cavities;
6. assign every point an owner by the rule in section 3;
7. **measure** the relationships from the resulting labelled field.

No entity is placed independently, and no relationship is asserted. Septa and valves in
particular exist only as residuals of the whole, so an entity-factorised model has to
discover them rather than be handed them.

## 3. Ownership rule (fixed in advance, Experiment 10)

Step 6's wall layers scored near zero because its ownership rule gave every interior
point to whatever a shell enclosed. This rule replaces it. Each point gets exactly one
owner, resolved in this order:

1. Outside the pericardial sac: **background**.
2. Inside the sac but outside the organ envelope: **pericardium**.
3. Inside a cavity: the **chamber** that cavity belongs to.
4. Within the valve annulus of a cavity pair: that **valve**.
5. Inside a vessel lumen or wall: that **vessel**.
6. Myocardial tissue, resolved in this sub-order:
   a. within `septum_band` of two different cavities: the **septum** separating them;
   b. within `endocardium_thickness` of any cavity surface: **endocardium**;
   c. within `epicardium_thickness` of the outer organ surface: **epicardium**;
   d. otherwise: **myocardium**.

Consequences, stated before running: the wall layers now own real volume, so their IoU
is measurable rather than structurally zero. Every reported ownership metric uses this
rule and only this rule.

**Geometry quality and semantic ownership are reported separately** and never mixed:
occupancy reconstruction of the whole organ is one number; per-entity ownership IoU is
another.

## 4. Metric definitions (fixed in advance)

| metric | definition |
| --- | --- |
| `scene_iou` | intersection over union of predicted and true whole-organ occupancy on shared query points |
| `occupancy_chamfer` | point-set Chamfer between predicted-occupied and true-occupied query points, `2*sqrt(3)` if either set is empty |
| `part_control_success` | share of present entities whose predicted ownership region reaches IoU >= 0.5 against the true region |
| `entity_iou_mean` | mean per-entity ownership IoU over present entities |
| `spatial_relation_accuracy` | share of measured spatial relations reproduced by the predicted entity centroids |
| `structure_relation_accuracy` | share of measured structural (adjacency) relations reproduced by predicted regions |
| `placement_error` | mean distance between predicted and true entity centroids, in scene units |
| `counterfactual_relation_sensitivity` | mean displacement of predicted entity placement when the relation graph is swapped to another variant's, with entities and presence held fixed |
| `counterfactual_correctness` | share of entities whose placement moves **toward** the counterfactual variant's true placement rather than merely moving |
| `invariant_preservation` | displacement of entities whose true placement is unchanged by the perturbation; lower is better |
| `local_edit_locality` | mean absolute occupancy change inside the edited entity divided by the mean absolute change in entities the generator did not alter |
| `edit_target_accuracy` | IoU between the edited entity's predicted geometry and the generator's true edited geometry |
| `lod_detail_gain` | improvement in entity IoU from the coarsest to the finest token prefix, against level-specific targets |

`counterfactual_correctness` is the one that matters. A model can be sensitive to a
relation change without being **correctly** sensitive; only movement toward the
counterfactual ground truth counts.

## 5. Model variants

| arm | description | parameter control |
| --- | --- | --- |
| A0 | appearance baseline, no entity axis | matched to A3 |
| A1-small | entity axis, no graph encoder | not matched, the Step 6 configuration |
| A1-matched | entity axis, no cross-entity information, capacity restored by per-entity depth | matched to A3 within 5 percent |
| A2 | structure graph only | same as A3 |
| A3 | full three-graph encoder | reference |
| A3-lite | one layer, four heads | smaller on purpose |

A1-matched is the arm that decides Question C. It has A3's capacity and A3's inputs but
cannot move information between entities.

## 6. Controls

* Three seeds for every headline comparison; one seed for secondary ablations, labelled
  as such.
* Identical data, steps, optimiser, learning rate, batch size and evaluation protocol
  across arms.
* Family-level splits; no procedural family appears in two splits.
* Checkpoint selection rule, fixed now: **the final checkpoint**, with the convergence
  check in Experiment 11 reporting whether validation had plateaued.
* Any input available to one arm is available to all. The variant label is never an
  input to any arm; it is only in the relations.

## 7. What would count as a negative result

A negative result is a result. Specifically:

* If A1-matched matches A3 on the whole-organ corpus **and** on the counterfactual
  tests, the graph encoder should be simplified or removed.
* If the spatial graph alone explains the counterfactual behaviour, the functional
  graph should be deferred.
* If higher levels of detail do not improve geometry against level-specific targets,
  the nested level-of-detail design should be revised or removed.
* If local editing cannot change a target without disturbing its neighbours, persistent
  geometric editing is not ready and should be reported as not working.

None of these outcomes is to be avoided by changing a definition after the fact.
