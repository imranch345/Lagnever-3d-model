# STEP 10 — CHANGE 2 CORPUS REPORT

**Lagnav 3D — A rotated corpus, its rotation target, and its own placement-blind floor.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data, meshes or external 3D systems were used.
No model was trained in this pass, and none may be until the Change 2 training formulation is
approved.

---

## 1. Why a new corpus is necessary

Change 1 asked whether predicting each entity's frame relative to its spatial parent places
better than predicting global frames, and answered no, in all five arms, on every seed. That
result stands and is frozen.

It could not, however, test the thing a parent-relative frame is most obviously *for*. The
Change 1 corpus stores an identity rotation for every entity of every scene. A parent can
therefore pass its children only a translation and a scale; orientation — a vessel's direction
relative to the valve it leaves through — carries no information at all, because every
orientation in the corpus is the same one. Three consequences, each measured rather than
argued during Change 1:

* a quarter of the Step 8 composite's weight was spent on a term that was identically zero;
* the convention's closure under rotation was untestable on the data, so a transposed-rotation
  defect could pass a round trip unseen — the Change 1 integrity gate had to inject synthetic
  probe rotations to detect it at all;
* no rotation result of any kind could be obtained, so the question of whether the transform
  representation *preserves* position, rotation and scale through composition stayed open.

Change 2 asks that question: **when the corpus contains genuine rotational variation, does
the chosen transformation representation and composition mechanism correctly learn and
preserve position, rotation and scale?** This pass builds and validates the data that
question needs. It is a new controlled experiment on a limitation found during Step 10's
foundation work, not an attempt to rescue Change 1.

### Change 1 is frozen

`experiments/runs/step10-placement/` holds all 45 runs, their checkpoints, the assembled
report, the generated tables, the depth analysis, the integrity and leakage results, the
inheritance diagnostic, the worker logs and the code snapshot. `FROZEN.md` states the terms
and `CHANGE1_FROZEN.sha256` covers all 101 artifacts plus
`docs/STEP_10_CHANGE1_REPORT.md` and `experiments/step10/placement_floor.json`. Verified
after this pass: **101 of 101 checksums match**, so nothing in Change 1 was modified.

The Change 1 conclusion and its post-hoc finding are preserved verbatim in `FROZEN.md`.

---

## 2. The exact rotation-generation procedure

**Rotation is ground truth, recovered from the generator, not an augmentation.** The generator
already computed every rotation this corpus records; `WholeOrganScene.frames()` simply
discarded it and wrote the identity. Nothing about the geometry changed in this pass — what
changed is that the frame now reports the orientation the geometry already had.

**Declared from construction, not from principal axes.** Each entity's rotation is read from
how the generator builds it:

| entity group | count | rotation |
| --- | --- | --- |
| chambers, septa, endocardium, myocardium, epicardium, pericardium | 10 | the organ-to-scene rotation, built from the arrangement's yaw, pitch and roll |
| the four annuli | 4 | the disc's own axis as the frame's third row, on top of the organ rotation |
| the six vessels | 6 | the vessel's direction as the frame's third row, on top of the organ rotation |

Principal axes were the obvious alternative and are rejected on purpose: a chamber is nearly
an ellipsoid of revolution, so its second and third principal axes are decided by sampling
noise, and the target would be a coin flip no model could learn. Declaring the rotation from
construction is exact, reproducible from the scene's own parameters, and stable.

**The two axes across a vessel are a convention.** A circular tube is symmetric about its own
axis, so nothing in the geometry distinguishes one spin from another. `_basis_from_axis` fixes
them against a reference direction, swapping the reference when it is nearly parallel to the
axis. That switch is a discontinuity, and a target that jumped across it between scenes would
be unlearnable, so the margin is measured rather than assumed: across all four arrangement
regions no construction axis comes within **0.05** of the switch, and the one entity that sits
on the far side of it (`pulmonary_arteries`, |x| = 0.958) sits there in every scene.
`test_no_entity_sits_near_the_reference_switch` pins this.

**Applied before measurement.** The geometry is rotated, then measured: the sampled points are
already in scene coordinates when centroids and extents are taken from them.

**Rotations are hierarchy-conditioned, not independent.** Every entity in a scene shares that
scene's organ rotation; an annulus or tube composes its construction axis on top of it. A
parent and its children are therefore rotationally related, which is the whole point — an
independent per-entity rotation would destroy the structure the hierarchy is supposed to
exploit.

**One consequence, recorded as a change.** An oriented frame must measure its extents on its
own axes; a frame that claimed a rotated basis while reporting world-axis extents would
describe a box the entity does not occupy. So extents moved, for 38,999 of 39,000 entity
frames. That is the only other difference from the parent corpus, and §7 treats it as such.

**Maximum magnitude is not capped.** The distribution is whatever the generator's construction
and the existing arrangement ranges produce; no bound was chosen, and no distribution was
selected to make training easier. The arrangement ranges are the Change 1 corpus's, unchanged,
because they define the hold-out.

---

## 3. Rotation statistics

See the generated tables in §8. In summary, over the 29,920 frames the evaluation loader
delivers: mean **61.8°**, median 67.8°, minimum **0.74°**, maximum **179.98°**, **none** below
1°, 85% above 10°, 59% above 30°, 55% above 60°.

`max |R − I| > 0` holds with a wide margin: the largest stored rotation is 1.9984 away from
the identity in the raw basis vectors, and the largest angle is 179.98°.

**A corpus where predicting the identity is nearly free would be rejected.** Here, predicting
the identity would score about 62° of mean geodesic error, so it is not nearly free.

**But the honest bar is much lower than 62°, and §6 is where that is measured.** The
per-entity table shows why: each entity's rotation varies by only 4–13° about its own mean, so
most of the 62° is predictable from entity identity alone. This is the same lesson Step 9
learned about position, and it is why the floor — not the raw target magnitude — is the
reference for every future rotation result.

---

## 4. Mathematical composition validation

Run **before** the corpus was generated, as `tests/test_step10_rotated_composition.py`, 116
tests, all passing. The convention is exercised at parent rotations of **0°, 30°, 90° and
180°** crossed with child rotations at the same angles, about three different axes, with
anisotropic child scales and non-unit parent scales.

| claim | result |
| --- | --- |
| `compose(parent, relative(parent, child)) == child` | exact to < 1e-12, both closed conventions, every angle pair |
| the composed frame represents the composed affine map exactly | points mapped through the composition equal points mapped through both frames, < 1e-12 |
| the composed linear part carries no shear | `similarity_residual` < 1e-10 |
| a depth-3 chain, every level rotated and scaled | exact to < 1e-11 |
| `full` (the unclosed convention) shears and loses it | residual > 1e-3, round trip fails by > 1e-6 |
| identity rotations hide that defect | residual < 1e-12, round trip exact — which is why Change 1's corpus could not have caught it |

The second row is the one that matters most. A representation that silently projected a
sheared matrix back onto rotation-times-diagonal would pass a round trip built from the same
projection; mapping points through both paths and comparing is what rules that out. **No
composition defect remains, so corpus generation was allowed to proceed.**

No invalid composed transform is projected back into the representation anywhere in the
closed conventions: under `isotropic` and `rigid` the product is always exactly
rotation-times-diagonal, which is what closure means.

---

## 5. Transform representation

Unchanged from Change 1, and now tested under real rotation:

* **parent contributes** rotation and a single isotropic scale;
* **child contributes** rotation and its full anisotropic scale.

The old non-closed 12-number composition is not used and is retained only as the `full`
convention that the tests above demonstrate to be lossy.

**Rotation metrics.** Reporting uses the geodesic angle, and §4 confirms it is well defined at
the identity (finite, < 1e-6 against itself) and recovers small angles correctly (1e-4°, 1e-2°,
0.5°, 30°, 179.9°). For optimisation, the geodesic form is **not** to be used as the loss: its
gradient is measured here to be more than 100× the chordal form's at 1e-3 rad, because
`arccos` has an unbounded derivative at zero. `rotation_chordal` is monotone in the angle
across 0–180° and is the form to descend. The rotation target is not degenerate: no entity's
rotation is constant across scenes, and none is the identity.

---

## 6. New placement-blind floors

Computed fresh on this corpus, fitted on `train` only, from an identity-only lookup with no
relations, no hierarchy and no scene context — Step 9's conception, applied to new data. The
numbers are in §8's generated tables.

**Position: 0.1605 / 0.1655 / 0.1703 / 0.1750.** Identical to Change 1's, and that is a
*measurement*, not a reuse: the corpus was derived from the Change 1 corpus, so every centroid
is the same number, and a blind predictor of position therefore lands in the same place. §7 is
explicit about what that does and does not license.

**Rotation, which is new: 22.65° on `test_seen`**, 22.54° on `test_arrangement`, **35.95° on
`test_transform`** and 26.10° on `test_combination`.

Two things to read from those:

* **The bar for rotation is ~23°, not 62°.** A model that reports 40° of rotation error is
  worse than a lookup table, however impressive 40° looks beside the target's 62° mean.
* **`test_transform` is much harder (35.95°)**, and that is the hold-out working: it holds out
  |yaw| beyond the training edge, so a lookup fitted on training rotations cannot extrapolate
  to it. The compositional structure Step 9 built now bites on rotation as well as position.

**The floor estimator.** Position and log scale are arithmetic means; the rotation is the
**chordal mean** — the average matrix projected back onto SO(3). Averaging the six stored
numbers and re-orthonormalising would not be the mean of the rotations and can cancel to near
zero when rotations are widely spread, at which point Gram-Schmidt returns an arbitrary frame.
A floor that is weaker than it needs to be flatters every model measured against it. On an
identity-rotation corpus the two estimators coincide, and the frozen Change 1 floor is
confirmed bit-identical after the change.

**Determinism (§11).** The floor was computed twice on the same corpus with the same
configuration; the two runs are **identical**. The floor uses no random source: the evaluation
loader runs with `shuffle` off and seed 0. Both runs are kept —
`rotated_placement_floor.json` and `rotated_placement_floor_repeat.json`.

**A parent-relative floor, for Change 2's own question.** On this corpus a blind
parent-relative predictor is *worse* than a global one for position (0.1806 vs 0.1605), as it
was in Change 1. With the true parent supplied it is better (0.1402, 12.7% below the global
floor, against 24.4% on the identity corpus). For **rotation**, the parent-relative target is
worse even with a perfect parent (25.06° vs 22.65°): a child's rotation relative to its parent
varies more across scenes than its absolute rotation does. That is a floor measurement, not a
model result, and it is on record before any Change 2 training.

---

## 7. Old corpus versus new corpus

These are **separate experimental tracks**.

| | Change 1 | Change 2 |
| --- | --- | --- |
| corpus | `step8-continuous-1950-40` | `step10-rotated-1950-40` |
| rotations | identity everywhere | measured, mean 61.8° |
| position floor, `test_seen` | 0.1605 | 0.1605 |
| rotation floor, `test_seen` | 0° by construction | 22.65° |
| scale floor, `test_seen` | 0.1326 | 0.1120 |

The scale floor moved in the other direction, from 0.1326 to 0.1120: extents measured on an
entity's own axes are *more* predictable from identity alone than world-axis extents were, so
the blind predictor does better on scale here. That is a property of the new target, and it is
another reason the two corpora's numbers are not interchangeable.

**No sentence of the form "Change 2 improved translation from 0.1605 to X" may be written.**
The old number is historical context for the old corpus. Even though the position floors
coincide — because the position targets are literally the same numbers — a model trained on
this corpus is solving a different problem: it must also predict real rotations, and its scale
target is measured on different axes. Whether its position error is comparable with Change 1's
is a question to be argued explicitly, not assumed from the floors matching.

Any cross-corpus statement must go through a normalised diagnostic defined for the purpose —
for example each arm's margin over its **own** corpus's floor — and must say so.

---

## 8. Generated tables

Every number below is produced by `experiments/step10/change2_tables.py` from the audit, the
two floor runs and the validation report. None is typed by hand.

<!-- BEGIN GENERATED TABLES -->
<!-- generated by experiments.step10.change2_tables -->

### Corpus manifest

| field | value |
| --- | --- |
| `corpus_id` | step10-rotated-1950-40 |
| `parent_corpus_id` | step8-continuous-1950-40 |
| `generator_version` | whole-organ-generator-2 |
| `generator_commit` | 66606a04de7dad0682f8457f6e12b8d1195d57f6 |
| `random_seed` | 20250915 |
| `rotation_enabled` | True |
| `format_version` | lagnav-whole-organ-1 |
| `scenes` | 1,950 |
| `arrangement_count` | 1,950 |
| `family_count` | 40 |
| `entity_count` | 20 |
| `distinct_relation_graphs` | 266 |

| split | scenes | arrangement region |
| --- | --- | --- |
| train | 1200 | in_distribution |
| validation | 150 | in_distribution |
| test_seen | 150 | in_distribution |
| test_arrangement | 150 | test_arrangement |
| test_transform | 150 | test_transform |
| test_combination | 150 | test_combination |

**Changes from the parent corpus**

* entity frames carry measured rotations instead of the identity
* extents are measured on each entity's own frame axes, not the world axes, because an oriented frame with world-axis extents describes a different box

### Rotation distribution

Geodesic angle from the identity, in degrees, over the entities the evaluation loader delivers. `<1 deg` is the share a model could get right by predicting no rotation at all.

| split | mean | median | std | min | max | <1 deg | >10 deg | >30 deg | >60 deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 61.13 | 67.66 | 47.01 | 0.74 | 179.96 | 0.000 | 0.84 | 0.57 | 0.55 |
| validation | 61.74 | 68.17 | 46.64 | 3.10 | 179.82 | 0.000 | 0.86 | 0.56 | 0.54 |
| test_seen | 60.66 | 66.33 | 47.35 | 0.75 | 178.85 | 0.003 | 0.81 | 0.57 | 0.54 |
| test_arrangement | 61.12 | 67.51 | 46.92 | 2.72 | 178.57 | 0.000 | 0.93 | 0.57 | 0.55 |
| test_transform | 69.50 | 71.13 | 41.25 | 26.65 | 179.98 | 0.000 | 1.00 | 0.83 | 0.54 |
| test_combination | 61.03 | 68.18 | 48.32 | 1.69 | 179.79 | 0.000 | 0.79 | 0.56 | 0.55 |

All splits together: 29,920 frames, mean 61.77 deg, max 179.98 deg, 0.000 below 1 deg.

### Rotation per entity

Per entity, all splits together. The ten entities with no distinguished construction axis carry the organ's own rotation; the ten annuli and tubes carry their axis as well. A small `std` means an identity-only lookup can predict most of that entity's rotation, which is what the floor measures.

| entity | mean | std | min | max |
| --- | --- | --- | --- | --- |
| left_atrium | 16.24 | 8.01 | 0.74 | 35.79 |
| left_ventricle | 16.24 | 8.01 | 0.74 | 35.79 |
| right_atrium | 16.24 | 8.01 | 0.74 | 35.79 |
| right_ventricle | 16.24 | 8.01 | 0.74 | 35.79 |
| interatrial_septum | 16.43 | 7.99 | 0.75 | 34.49 |
| interventricular_septum | 16.43 | 7.99 | 0.75 | 34.49 |
| endocardium | 16.58 | 7.98 | 1.63 | 34.49 |
| epicardium | 16.58 | 7.98 | 1.63 | 34.49 |
| myocardium | 16.58 | 7.98 | 1.63 | 34.49 |
| pericardium | 16.58 | 7.98 | 1.63 | 34.49 |
| pulmonary_valve | 66.99 | 6.74 | 54.39 | 86.20 |
| pulmonary_trunk | 67.03 | 6.75 | 54.39 | 88.86 |
| tricuspid_valve | 80.14 | 6.77 | 56.92 | 95.33 |
| aorta | 81.08 | 6.05 | 69.17 | 99.03 |
| aortic_valve | 81.08 | 6.16 | 69.22 | 98.73 |
| mitral_valve | 81.92 | 6.33 | 61.11 | 96.76 |
| superior_vena_cava | 96.91 | 4.11 | 88.90 | 108.20 |
| inferior_vena_cava | 96.99 | 4.13 | 88.89 | 107.09 |
| pulmonary_arteries | 117.92 | 13.09 | 89.60 | 151.49 |
| pulmonary_veins | 171.79 | 4.00 | 162.85 | 179.98 |

### New placement-blind floor

Fitted on `train` only, identity-only lookup, no relations, no hierarchy, no scene context. Computed twice: the two runs are identical.

| split | old floor (Change 1) | new position floor | new rotation floor (deg) | new scale floor | new composite | directly comparable? |
| --- | --- | --- | --- | --- | --- | --- |
| test_seen | 0.1605 | 0.1605 | 22.65 | 0.1120 | 0.3154 | No |
| test_arrangement | 0.1655 | 0.1655 | 22.54 | 0.1199 | 0.3238 | No |
| test_transform | 0.1703 | 0.1703 | 35.95 | 0.1248 | 0.3895 | No |
| test_combination | 0.1750 | 0.1750 | 26.10 | 0.1231 | 0.3505 | No |

**The position column is identical to Change 1's, and that is a measurement, not a reuse.** The rotated corpus was derived from the Change 1 corpus, so every centroid is the same number; a floor fitted on identity alone therefore lands in the same place. The rotation and scale floors are new, the composite is new, and a model trained here faces a different task, so its results still belong to a separate track from Change 1's.

### Parent-relative floors on the rotated corpus

The same identity-only lookup under a parent-relative target, now with real rotations. `blind` composes its own guesses; `true parent` leaks the parent frame and is a diagnostic, never a bar.

| split | global | parent-relative, blind | with true parent | rotation, global (deg) | rotation, true parent (deg) |
| --- | --- | --- | --- | --- | --- |
| test_seen | 0.1605 | 0.1806 | 0.1402 | 22.65 | 25.06 |
| test_arrangement | 0.1655 | 0.1834 | 0.1421 | 22.54 | 25.20 |
| test_transform | 0.1703 | 0.1866 | 0.1500 | 35.95 | 40.23 |
| test_combination | 0.1750 | 0.1910 | 0.1709 | 26.10 | 26.46 |

### Transformation hold-out

Rotation follows the arrangement, so the band a split occupies in the arrangement space is the band it occupies in rotation. `hole` is the held-out yaw interval, `beyond edge` the held-out extrapolation region, and `mirror+transpose` the held-out combination.

| split | abs yaw min | abs yaw max | in hole | beyond edge | mirrored | transposed | mean rotation (deg) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.000 | 0.450 | 0.00 | 0.00 | 0.48 | 0.25 | 61.13 |
| validation | 0.006 | 0.450 | 0.00 | 0.00 | 0.55 | 0.25 | 61.74 |
| test_seen | 0.009 | 0.448 | 0.00 | 0.00 | 0.50 | 0.22 | 60.66 |
| test_arrangement | 0.002 | 0.450 | 0.48 | 0.00 | 0.44 | 0.30 | 61.12 |
| test_transform | 0.453 | 0.599 | 0.00 | 1.00 | 0.41 | 0.21 | 69.50 |
| test_combination | 0.000 | 0.450 | 0.00 | 0.00 | 1.00 | 1.00 | 61.03 |

Held out: |yaw| in [0.15, 0.25] and a septal band (`test_arrangement`); |yaw| above 0.45 (`test_transform`); mirror and transpose together (`test_combination`). `test_seen` holds out families instead, at in-distribution arrangements.

### Corpus validation

| check | result |
| --- | --- |
| scenes paired with the parent corpus | 1,950 |
| worst centroid deviation from the parent | 0.0e+00 |
| relationship graphs, point counts, presence, levels of detail | identical |
| entity frames whose extent moved | 38,999 (the intended consequence) |
| rotations compared against the generator | 39,000 |
| worst rotation deviation from the generator | 5.0e-09 |
| placement gate, stored-frame round trip | 2.2e-16 |
| placement gate, probe-rotated round trip | 3.3e-16 |
| largest stored rotation away from identity | 1.9984 |
| parent table against the generator's construction | matches generator construction |

<!-- END GENERATED TABLES -->

---

## 9. Integrity tests

The Step 10 integrity framework is reused and extended. `validate_corpus` runs every check and
raises on the first failure; reaching the end is the pass. Results are in §8's validation
table: 1,950 scenes paired, 39,000 rotations compared against the generator, worst deviation
5.0e-9 (the corpus's own 8-decimal rounding), round trips at 2.2e-16 and 3.3e-16.

**A sixth check was added to the gate.** `check_corpus_rotations` compares every stored
rotation against the one the generator's construction implies, rebuilt from the scene's own
parameters. The Change 1 report named this as a gap that a rotated corpus would have to
close, and it is closed: a rotation replaced by the identity or stored transposed is
orthonormal, right-handed and exactly composable, so no round trip and no well-formedness
test can see it. Only an independent record can.

**The deliberate corruptions** (`tests/test_step10_rotated_corpus.py`, 19 tests, all passing):

| corruption | expected | result |
| --- | --- | --- |
| A. a real rotation replaced by the identity | validation failure | refused by the generator comparison; passes every local check, as predicted |
| B. a rotation transposed | validation failure | refused by the generator comparison; still a valid rotation, so nothing else sees it |
| C. a non-orthonormal basis | validation failure | refused twice: by the raw-basis check and by the generator comparison |
| D. a broken parent/child composition order | validation failure | refused by the order check |
| a corpus with no rotations offered as a rotated one | validation failure | refused |
| the correct corpus | pass | passes all of the above |

**Two findings from writing these.**

* *A rotation near 180° is nearly its own transpose.* Transposing the largest rotation in a
  scene changes it by less than 0.1, because a 180° rotation is symmetric. The corruption
  therefore has to be applied near 90°, where transposition bites hardest, and the test says
  so rather than quietly picking a weak victim.
* *A pre-existing generator defect was found and fixed.* `measure_relations` drew its
  adjacency point clouds while iterating a **set of entity ids**, so the draws consumed the
  shared random generator in hash order and the resulting adjacency edges depended on which
  process generated the corpus. A corpus was therefore not reproducible from its seed.
  Iteration is now sorted, and identical relation signatures are produced under
  `PYTHONHASHSEED` 0, 1 and 42. The frozen Change 1 corpus was written before the fix and is
  left untouched, which is why this corpus is **derived** rather than regenerated (§13).

---

## 10. Leakage audit

| question | finding | how |
| --- | --- | --- |
| Is the held-out arrangement leaked? | No | every scene classifies into its split's region by `classify_arrangement`, the same function the hold-out is defined with; `train` contains nothing from the interpolation hole or beyond the extrapolation edge |
| Is the held-out transformation leaked? | No | `test_transform` is entirely beyond the yaw edge; `test_combination` is entirely mirror-and-transpose |
| Are global target frames supplied as input? | No | frames are targets; the model's inputs are structure, relations and text, and Change 1's leakage tests (perturbing every true frame leaves inferred output bit-identical) still pass |
| Is the parent oracle reachable? | No | `parent_slots(transpose=True)` still has no production caller, and the parent table remains a module-level constant with no scene input |
| Is future evaluation information present? | No | the floor is fitted on `train` only; the audit reads splits independently and shares nothing between them |
| Does presence carry the arrangement? | No | every scene carries all twenty entities, checked for all 1,950 |
| Are the families held out where they should be? | Yes | `test_seen`'s 8 families are disjoint from `train`'s 32; the other three test splits reuse training families so a failure there is about the arrangement, not the family |

The oracle floor (`parent_relative_oracle`) leaks the true parent by construction. It is
reported as a diagnostic, is kept in a separate code path from the blind scoring, and is never
a bar. Any future oracle experiment stays separate from the main experiment.

---

## 11. Transformation-pair hold-out

Preserved exactly, because the split definition was not touched: the regions come from
`classify_arrangement`, and the corpus was derived scene for scene from a corpus built with
it. §8's hold-out table shows the bands.

**What is held out**

* `test_arrangement` — |yaw| inside [0.15, 0.25], or `septal_shift` inside its own hole:
  interpolation into a gap the training set does not cover. 48% of its scenes are in the yaw
  hole, the rest in the septal hole.
* `test_transform` — |yaw| above 0.45: extrapolation past the training edge. 100% of its
  scenes, with |yaw| from 0.453 to 0.599 against training's 0.000–0.450.
* `test_combination` — mirror **and** transpose together: 100% mirrored and 100% transposed,
  against 48% and 25% in training, which contain each alone but never both.
* `test_seen` — families held out instead, at in-distribution arrangements.

**Why this carries over to rotation.** A scene's organ rotation is built from its yaw, pitch
and roll, so holding out a yaw band holds out a band of rotations, and holding out
mirror-and-transpose holds out the construction axes that combination produces. The rotation
target is therefore *not* exposed equally across train and test, and the rotation floor shows
it: 22.5–22.7° on the in-distribution and interpolation splits against **35.95°** on the
extrapolation split.

---

## 12. Test results

The full suite passes: **841 tests**, up from 706 at the end of Change 1, so **135 are new in
this pass**. Ruff is clean. Both mypy tiers are clean — 54 tier-one files, 97 tier-two.

| suite | tests | what it covers |
| --- | --- | --- |
| `test_step10_rotated_composition.py` | 116 | §4, §5: the convention at 0/30/90/180 degrees, shear, projection loss, the metric at identity, the gradient argument |
| `test_step10_rotated_corpus.py` | 19 | §14: the four corruptions, the frame convention's stability, the audit reading what the model reads |
| all Step 10 suites | 223 | the 88 Change 1 tests remain green, unchanged |
| whole repository | 841 | every earlier step's guarantees |

Every Change 1 test still passes without modification, and the frozen Change 1 floor was
re-derived after the floor estimator changed and came back bit-identical.

---

## 13. Reproducibility

**The corpus is derived, not regenerated.** `derive_rotated_corpus` reads the frozen Change 1
corpus and re-measures each scene's frames, carrying over every other stored field: the
generator parameters, the centroids, the relationship graph, the point counts, presence, the
level of detail, and the split and family assignment. Two reasons:

* the relationship graph is the channel that carries the arrangement to the model, and holding
  it *exactly* constant is the tightest control available on a change of target — the derived
  corpus has 266 distinct relation graphs, the parent's 266, the same ones;
* the parent corpus was generated before the hash-order defect in §9 was fixed, so a freshly
  generated corpus does not reproduce its adjacency edges. A fresh generation in this pass
  produced 293 distinct graphs instead of 266 — a difference that has nothing to do with
  rotation. Deriving avoids attributing it to rotation.

`--fresh` still generates from scratch, now deterministically, and is the path a future corpus
with different arrangements would take.

**Verification built into the derivation.** Every re-measured centroid must reproduce the
stored one; the worst deviation across 39,000 entity frames is **0.0**. A derivation that
measured a different scene would fail rather than proceed.

**Provenance in the manifest** (§8): `corpus_id`, `parent_corpus_id`, `generator_version`,
`generator_commit`, `random_seed`, `rotation_enabled`, `rotation_distribution`,
`arrangement_count`, `family_count`, `entity_count`, `split_definition` and
`changes_from_parent`. The parent manifest predates the `random_seed` field, so the recorded
seed is the generating CLI's default, declared rather than read back, and the manifest says
so. Each scene also carries its own seed, and re-measurement uses `scene.seed + 11`, the same
generator state the parent measured with.

**Commands**

```
python -m datasets.whole_organ.step10_cli --out datasets/processed/step10_rotated
python -m experiments.step10.validate_corpus --corpus datasets/processed/step10_rotated
python -m experiments.step10.rotation_audit --corpus datasets/processed/step10_rotated
python -m experiments.step10.placement_floor --corpus datasets/processed/step10_rotated \
    --out experiments/runs/step10-change2/rotated_placement_floor.json
python -m experiments.step10.change2_tables --insert-into docs/STEP_10_CHANGE2_CORPUS_REPORT.md
```

**Artifacts** in `experiments/runs/step10-change2/`: `corpus_generation.log`,
`corpus_validation.json`, `rotation_audit.json`, `rotated_placement_floor.json`,
`rotated_placement_floor_repeat.json`.

**Code state.** Commit `66606a0` plus this pass's uncommitted changes. The corpus manifest
records that commit as its `generator_commit`; nothing in this pass has been committed.

**Corpus size is unchanged** (§17): 1,950 scenes, 40 families, the same split counts as the
parent. No model size or compute was changed, because nothing was trained.

---

## 14. The exact next step

**Nothing may be trained until the Change 2 training formulation is approved.** When it is,
the order is:

1. **Decide the rotation loss.** The geodesic form must not be the training objective (§5).
   `rotation_chordal` exists, is monotone and is smooth at zero; the weight it carries against
   position and scale has to be fixed *before* any confirmatory run, as does whether the Step 8
   composite weights (1, 0.5, 0.25) are kept for comparability or replaced now that the
   rotation term is no longer identically zero.
2. **Re-pre-register the cells.** Change 1's matrix was global versus parent-relative versus
   taxonomic. On this corpus `T4_rigid` becomes meaningful for the first time — `rigid`
   propagates a parent's orientation without its size, and orientation now carries
   information — so the cell set should be reconsidered rather than copied. Whatever is chosen
   is written down before it runs.
3. **State the decision rule and the bar.** The bar is this corpus's floor: 0.1605 position
   and **22.65°** rotation on `test_seen`, per split as tabulated. Not 0.1605 alone, and never
   the raw 62° target mean.
4. **Then train**, at Step 9's protocol unless a change is justified in writing, with the
   per-run artifacts, resume and disk guard the Change 1 runner already provides.

Open questions this corpus now makes answerable, and which Change 1 could not touch: whether a
parent-relative frame pays for *orientation* even where it did not pay for position; whether
`rigid` beats `isotropic` once a parent's orientation matters; and whether the rotation floor's
extrapolation gap on `test_transform` is something a relational model can close.

---

```text
CHANGE 2 CORPUS STATUS
----------------------
Corpus generation: COMPLETE
Rotation target: VALID
Composition: PASS
Placement floor: COMPLETE
Integrity: PASS
Leakage audit: PASS
Tests: PASS

New rotated-corpus floor:
test_seen        = 0.1605 position, 22.65 deg rotation, 0.1120 scale
test_arrangement = 0.1655 position, 22.54 deg rotation, 0.1199 scale
test_transform   = 0.1703 position, 35.95 deg rotation, 0.1248 scale
test_combination = 0.1750 position, 26.10 deg rotation, 0.1231 scale

Training:
NOT STARTED
```
