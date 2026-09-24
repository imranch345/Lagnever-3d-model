# STEP 10 — CHANGE 1 COMPLETION REPORT

**Lagnav 3D — Parent-relative placement on the Step 8/9 corpus.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data, meshes or external 3D systems were used.
The corpus was not regenerated; every rotation in it is the identity. Change 2 was not
started.

---

## 1. Summary

**Parent-relative placement made global placement worse in every arm.** On `test_seen`, with
the target the only thing changed, translation error rose in all five arms — by 2.9% (A3L)
to 7.8% (A3) — and every seed moved the same way. By the pre-registered rule the loss
exceeds seed variability in four arms; in A3Lite it is consistent across seeds but falls
short of the threshold (t = 3.99 against 4.303). The same direction holds on
`test_arrangement` and `test_transform` in all ten cells. Only `test_combination`, the split
on which every arm is already far below the floor, is mixed.

**The prediction on record is not supported.** It said gains would concentrate in A3, the one
arm reliably above the Step 9 floor. A3 lost the most: from 0.0044 above the floor to 0.0078
below it, with every seed on the losing side.

**The effect comes from the parented entities, but in the wrong direction.** Depth 0 did not
move in four of five arms. Depth 1 — the valves and great veins, seven of the ten parented
entities — carries most of the loss, and depth 2 most of the rest. The one exception points
the other way: depth 3, the pulmonary arteries, placed *better* under parent-relative in all
three graph arms, significantly in two.

**Why, measured rather than guessed (post hoc).** Recomposing the saved models' children onto
their parents' *true* frames shows that the depth-1 local predictions are 25–32% better
than the global target's — and that composing them onto the *predicted* chambers throws
that away and more. Parent-relative placement does not solve placement; it hands it to the
chambers they are parented to, and no arm places the chambers well enough. The isotropic scale cascade is measurable
but secondary after training.

**Every control held.** The taxonomic control is bit-identical to the global control in
15 of 15 arm-seeds on every split. The global control reproduces the Step 9 baseline bit for
bit in 15 of 15. All 45 runs passed the §27 integrity gate before training and on all four
evaluated splits, and no leakage path was found.

```text
CHANGE 1 STATUS
----------------
Training: COMPLETE
Integrity: PASS
Depth analysis: COMPLETE
Taxonomic control: COMPLETE

Parent-relative placement:
NOT SUPPORTED

Prediction:
NOT SUPPORTED

Change 2:
NOT STARTED
```

---

## 2. What was tested

**The question.** Does asking the frame head to predict each entity's frame *relative to its
spatial parent* — and composing those local frames into scene coordinates inside the model —
improve global placement over predicting global frames directly? And if it does, does the
improvement come from the entities that actually have a parent?

**The cells.** Each differs from the control in the target alone.

| cell | target | hierarchy | role |
| --- | --- | --- | --- |
| `T0_global` | global frame | — | Step 9's target; the control |
| `T1_spatial` | parent-relative | spatial (generator construction) | Change 1 |
| `T2_taxonomic` | parent-relative | taxonomic (AWR's own) | taxonomic control |

Five arms (A1, A1M, A3Lite, A3L, A3) × three cells × three seeds = 45 runs, at Step 9's
protocol: 1200 steps, batch 8, CPU, 4 torch threads, the Step 9 loss weights unchanged.
Nothing was tuned on preliminary results; no seed was selected or dropped.

**The metric is Step 9's, unchanged.** The model composes local frames into global frames in
`predict_frames`, before the loss, the decoder or any metric sees them, so every cell is
scored with Step 9's global-frame translation error against the same placement-blind floor
(0.1605 on `test_seen`). No local-frame metric replaces it.

**The spatial hierarchy is not AWR's.** AWR nests every whole-organ entity under a
geometry-free category node, which leaves all twenty entities as roots. The spatial tree is
taken from how the generator builds the organ — an annulus between two cavities, a vessel
from its anchor or its valve — and is an experimental geometric hierarchy, not an AWR
`contains` relation. No relationship absent from the ontology was added to it. Ten of the
twenty entities have a parent; depths run 0–3.

**Decision rule, fixed before any confirmatory result.** `T1 − T0` is paired by seed. A
difference exceeds seed variability only if all three seeds agree in sign **and** the paired
|t| > 4.303 (two-sided 5%, df = 2). The same rule is applied per depth. Recorded in
`experiments/step10/README.md` and coded as `T_CRITICAL_DF2` before the first run finished.

**The prediction on record** (`experiments/step10/README.md`, from the floor computed before
anything was trained): the parent-relative target is worth about a quarter of position error
to a predictor that already places parents well and costs about 2% to one that does not, so
gains should concentrate in A3 — the only arm reliably above the floor in Step 9 — with
little or no useful gain in arms that do not clear it.

---

## 3. Disk and artifact record

At the start of this pass the data volume had 10 GiB free (95% used); an earlier session had
reached 250 MiB and an `ENOSPC` had silently dropped a file edit. The earlier exploratory
sweep had died on 2026-09-16 at 20:54 with no traceback and no report, because the runner
wrote its report only at the end.

| artifact | size | referenced by | action |
| --- | --- | --- | --- |
| `experiments/runs/step7-whole-organ-v1-confounded` | 124 MiB | ADR-0015; `docs/step7/02_preregistration_amendments.md`; `docs/step7/04_graph_necessity_method.md` | **kept** |
| `experiments/runs/step7-whole-organ-v2-broken-part-loss` | 124 MiB | ADR-0016; `docs/step7/04_graph_necessity_method.md` | **kept** |
| `experiments/runs/heart-001` | 363 MiB | `configs/neural/*.yaml`, `experiments/heart/*`, Step 6 docs, `README.md` | **kept** |

**Nothing was deleted.** The two Step 7 directories are superseded but are cited by path as
preserved evidence, so they do not meet the "genuinely unreferenced" condition, and with
10 GiB free there was no space reason to override that. Only scratch files outside the
repository were removed. The runner now refuses to start a run below 2 GiB free, writes each
run file atomically as soon as that run finishes, and resumes rather than retraining.

At completion: 8.5 GiB free; the whole Step 10 run directory, 45 checkpoints included, is
331 MiB.

**An operational note that affects one column.** The machine was on battery with a
one-minute idle-sleep setting and slept repeatedly from 15:09 until `caffeinate` was started
at 19:00; it later ran on battery again. Sleep does not change results — every `T2` and
`T0` run is bit-identical to its reference regardless — but the `train s` column of the
compute table is wall-clock and includes time asleep (A1 `T2_taxonomic` 11,500 s, A1M
`T0_global` 5,679 s). Compute equality rests on identical steps, batch and threads, not on
elapsed time.

---

## 4. Dataset integrity (§27)

A gate, `generation/neural/nn/placement_integrity.py`, now runs before training can start
(`Step10Config.resolved` checks the declared tree before the model is built;
`Step10Trainer` runs the full gate on a training batch) and again on every evaluated split.
Any failure raises `PlacementIntegrityError`. It restores every random generator it touches,
so it cannot move the results it guards.

| check | what it catches |
| --- | --- |
| source table vs generator | a declared parent the generator does not build that child from |
| slot table | a table that addresses the wrong entities (20-entity vs 64-slot ordering) |
| order | a child composed before its parent; an order that skips or repeats a slot |
| stored frames | a non-orthonormal or left-handed stored basis, read before Gram-Schmidt |
| round trip, stored and probe-rotated | a transposed rotation, a wrong order, an unclosed convention |

**The deliberate-corruption tests** (`tests/test_step10_corruption.py`, 22 tests, all pass):

| corruption | expected | result |
| --- | --- | --- |
| A. parent table shuffled in slot space | refused | refused by slot check and full gate |
| A. table built in the wrong slot ordering | refused | refused |
| A. source table shuffled | refused | refused against the generator |
| A. one plausible wrong edge (aorta → RV) | refused | refused, names the edge |
| A. trainer started on a shuffled table | refused | trainer raises before the model is built |
| B. rotation transposed (the historic `frame_to_matrix` bug) | refused | refused on the probe-rotated pass |
| B. non-orthonormal stored basis | refused | refused |
| C. composition order reversed | refused | refused by order check and full gate |
| C. order skipping a slot | refused | refused |
| correct spatial, alternative and taxonomic trees | pass | pass |
| correct placement on real frames | pass | pass, round trip ≤ 3e-16 |

**Two findings from building the gate.**

* *On this corpus a transposed rotation cannot be detected from the stored frames, and in
  the data it cannot occur at all.* Every stored rotation is the identity, which is its own
  transpose. `test_the_stored_frames_alone_could_not_have_caught_it` shows the historic
  transpose bug passing a round trip on stored frames; it is caught only because the gate
  repeats the round trip after giving each slot a distinct real rotation. A rotated corpus
  will additionally need the stored rotation compared against the generator's own record,
  which no round trip can do.
* *An earlier test was mislabelled.* `test_transposing_a_rotation_is_detected` swapped the
  two stored basis vectors, which on an identity rotation is a 180° turn about (1, 1, 0), not
  a transpose. It is renamed `test_swapping_the_stored_basis_vectors_is_detected` and the
  real transpose is tested separately.

The gate's first draft also accepted the inferior vena cava as a child of the tricuspid
annulus, because the IVC happens to start within two annulus thicknesses of it. A tube is
now accepted as built from an annulus only if it starts on that annulus's axis.

**Across the matrix** (generated table, section 6): 45 of 45 runs passed the gate before
training; 180 split-level passes at evaluation; worst round-trip error 5.0e-16 across
stored and probe-rotated frames; depth buckets reconciled with the headline to exactly
0.0 in every run; largest stored rotation deviation from identity exactly 0.0.

---

## 5. Leakage audit (§9)

| question | finding | evidence |
| --- | --- | --- |
| Is held-out arrangement information supplied? | No | the parent table is a module-level constant keyed on entity ids; it has no scene or arrangement input |
| Does the parent table encode the test arrangement? | No | transposed scenes keep the non-transposed pairing (aortic valve under the LV); trainers built on `train` and on `test_arrangement` hold identical tables |
| Is the arrangement oracle reachable? | No | `parent_slots(transpose=True)` has no production caller; the default table differs from the oracle, and the run uses the default |
| Is ground-truth global placement supplied at inference? | No | replacing every true frame in a batch with noise leaves inferred frames **and** part logits bit-identical |
| Does `to_global()` use anything but predicted frames? | No | `predict_frames` composes the head's own output with `parents_for(present)`; presence is a structural input the model already receives |
| Does anything convert true frames to local targets in training? | No | `HierarchicalPlacement.to_local` is called only by the floor script and tests; the loss is on composed global frames |
| Is the floor fitted on test data? | No | `placement_floor.py` fits on `train` only |

The one place true frames reach the model is `_select_frames(use_predicted_frames=False)`,
the "supplied" oracle condition, which is reported beside the headline and never as it.
Tests: `TestNoPlacementLeakage` in `tests/test_step10_corruption.py`.

---

## 6. Results (§29)

Every number in this section is generated by `experiments/step10/report_tables.py` from
`experiments/runs/step10-placement/placement_report.json`. None is typed by hand.

<!-- BEGIN GENERATED TABLES -->
<!-- generated from experiments/runs/step10-placement/placement_report.json by experiments.step10.report_tables -->

### Placement-blind floor

| split | floor (global) | blind parent-relative | with true parent (diagnostic) | reproduces Step 9 |
| --- | --- | --- | --- | --- |
| test_seen | 0.1605 | 0.1636 (-1.9%) | 0.1213 (+24.4%) | yes |
| test_arrangement | 0.1655 | 0.1680 (-1.5%) | 0.1237 (+25.3%) | yes |
| test_transform | 0.1703 | 0.1725 (-1.3%) | 0.1251 (+26.5%) | yes |
| test_combination | 0.1750 | 0.1774 (-1.4%) | 0.1515 (+13.4%) | yes |

### §29 results — test_seen

Split `test_seen`, inferred placement, global-frame translation error (lower is better). Floor = 0.1605. Margin = floor − mean; positive beats the floor. Depth columns are position error by depth in the spatial tree, seed mean.

| arm | cell | target | hierarchy | seeds | mean translation | seed std | floor | floor margin | depth 0 | depth 1 | depth 2 | depth 3 | parameters | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | global | — | 3 | 0.1676 | 0.0015 | 0.1605 | -0.0071 | 0.1460 | 0.2000 | 0.1092 | 0.1698 | 559,751 | 1200 |
| A1 | T1_spatial | parent-relative | spatial | 3 | 0.1774 | 0.0024 | 0.1605 | -0.0169 | 0.1465 | 0.2226 | 0.1158 | 0.1718 | 559,751 | 1200 |
| A1 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1676 | 0.0015 | 0.1605 | -0.0071 | 0.1460 | 0.2000 | 0.1092 | 0.1698 | 559,751 | 1200 |
| A1M | T0_global | global | — | 3 | 0.1679 | 0.0008 | 0.1605 | -0.0074 | 0.1466 | 0.2001 | 0.1086 | 0.1696 | 3,125,831 | 1200 |
| A1M | T1_spatial | parent-relative | spatial | 3 | 0.1735 | 0.0018 | 0.1605 | -0.0130 | 0.1451 | 0.2148 | 0.1146 | 0.1709 | 3,125,831 | 1200 |
| A1M | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1679 | 0.0008 | 0.1605 | -0.0074 | 0.1466 | 0.2001 | 0.1086 | 0.1696 | 3,125,831 | 1200 |
| A3Lite | T0_global | global | — | 3 | 0.1599 | 0.0049 | 0.1605 | +0.0006 | 0.1356 | 0.1953 | 0.0971 | 0.1699 | 1,252,615 | 1200 |
| A3Lite | T1_spatial | parent-relative | spatial | 3 | 0.1651 | 0.0067 | 0.1605 | -0.0045 | 0.1378 | 0.2060 | 0.1098 | 0.1527 | 1,252,615 | 1200 |
| A3Lite | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1599 | 0.0049 | 0.1605 | +0.0006 | 0.1356 | 0.1953 | 0.0971 | 0.1699 | 1,252,615 | 1200 |
| A3L | T0_global | global | — | 3 | 0.1644 | 0.0010 | 0.1605 | -0.0038 | 0.1403 | 0.1973 | 0.1091 | 0.1699 | 1,226,511 | 1200 |
| A3L | T1_spatial | parent-relative | spatial | 3 | 0.1691 | 0.0015 | 0.1605 | -0.0086 | 0.1417 | 0.2119 | 0.1090 | 0.1537 | 1,226,511 | 1200 |
| A3L | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1644 | 0.0010 | 0.1605 | -0.0038 | 0.1403 | 0.1973 | 0.1091 | 0.1699 | 1,226,511 | 1200 |
| A3 | T0_global | global | — | 3 | 0.1561 | 0.0006 | 0.1605 | +0.0044 | 0.1298 | 0.1931 | 0.0964 | 0.1642 | 3,226,647 | 1200 |
| A3 | T1_spatial | parent-relative | spatial | 3 | 0.1684 | 0.0049 | 0.1605 | -0.0078 | 0.1380 | 0.2138 | 0.1108 | 0.1520 | 3,226,647 | 1200 |
| A3 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1561 | 0.0006 | 0.1605 | +0.0044 | 0.1298 | 0.1931 | 0.0964 | 0.1642 | 3,226,647 | 1200 |

### §29 results — test_arrangement

Split `test_arrangement`, inferred placement, global-frame translation error (lower is better). Floor = 0.1655. Margin = floor − mean; positive beats the floor. Depth columns are position error by depth in the spatial tree, seed mean.

| arm | cell | target | hierarchy | seeds | mean translation | seed std | floor | floor margin | depth 0 | depth 1 | depth 2 | depth 3 | parameters | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | global | — | 3 | 0.1739 | 0.0035 | 0.1655 | -0.0084 | 0.1508 | 0.2018 | 0.1243 | 0.1855 | 559,751 | 1200 |
| A1 | T1_spatial | parent-relative | spatial | 3 | 0.1835 | 0.0026 | 0.1655 | -0.0180 | 0.1508 | 0.2235 | 0.1325 | 0.1873 | 559,751 | 1200 |
| A1 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1739 | 0.0035 | 0.1655 | -0.0084 | 0.1508 | 0.2018 | 0.1243 | 0.1855 | 559,751 | 1200 |
| A1M | T0_global | global | — | 3 | 0.1736 | 0.0007 | 0.1655 | -0.0081 | 0.1506 | 0.2016 | 0.1243 | 0.1852 | 3,125,831 | 1200 |
| A1M | T1_spatial | parent-relative | spatial | 3 | 0.1795 | 0.0015 | 0.1655 | -0.0140 | 0.1486 | 0.2167 | 0.1306 | 0.1865 | 3,125,831 | 1200 |
| A1M | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1736 | 0.0007 | 0.1655 | -0.0081 | 0.1506 | 0.2016 | 0.1243 | 0.1852 | 3,125,831 | 1200 |
| A3Lite | T0_global | global | — | 3 | 0.1651 | 0.0046 | 0.1655 | +0.0005 | 0.1398 | 0.1964 | 0.1115 | 0.1854 | 1,252,615 | 1200 |
| A3Lite | T1_spatial | parent-relative | spatial | 3 | 0.1702 | 0.0072 | 0.1655 | -0.0047 | 0.1409 | 0.2071 | 0.1235 | 0.1649 | 1,252,615 | 1200 |
| A3Lite | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1651 | 0.0046 | 0.1655 | +0.0005 | 0.1398 | 0.1964 | 0.1115 | 0.1854 | 1,252,615 | 1200 |
| A3L | T0_global | global | — | 3 | 0.1695 | 0.0029 | 0.1655 | -0.0040 | 0.1440 | 0.1985 | 0.1246 | 0.1853 | 1,226,511 | 1200 |
| A3L | T1_spatial | parent-relative | spatial | 3 | 0.1744 | 0.0021 | 0.1655 | -0.0089 | 0.1451 | 0.2135 | 0.1226 | 0.1661 | 1,226,511 | 1200 |
| A3L | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1695 | 0.0029 | 0.1655 | -0.0040 | 0.1440 | 0.1985 | 0.1246 | 0.1853 | 1,226,511 | 1200 |
| A3 | T0_global | global | — | 3 | 0.1603 | 0.0005 | 0.1655 | +0.0053 | 0.1327 | 0.1934 | 0.1112 | 0.1806 | 3,226,647 | 1200 |
| A3 | T1_spatial | parent-relative | spatial | 3 | 0.1723 | 0.0056 | 0.1655 | -0.0068 | 0.1404 | 0.2140 | 0.1244 | 0.1628 | 3,226,647 | 1200 |
| A3 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1603 | 0.0005 | 0.1655 | +0.0053 | 0.1327 | 0.1934 | 0.1112 | 0.1806 | 3,226,647 | 1200 |

### §29 results — test_transform

Split `test_transform`, inferred placement, global-frame translation error (lower is better). Floor = 0.1703. Margin = floor − mean; positive beats the floor. Depth columns are position error by depth in the spatial tree, seed mean.

| arm | cell | target | hierarchy | seeds | mean translation | seed std | floor | floor margin | depth 0 | depth 1 | depth 2 | depth 3 | parameters | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | global | — | 3 | 0.1787 | 0.0043 | 0.1703 | -0.0084 | 0.1517 | 0.2076 | 0.1331 | 0.2169 | 559,751 | 1200 |
| A1 | T1_spatial | parent-relative | spatial | 3 | 0.1882 | 0.0027 | 0.1703 | -0.0179 | 0.1520 | 0.2277 | 0.1407 | 0.2190 | 559,751 | 1200 |
| A1 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1787 | 0.0043 | 0.1703 | -0.0084 | 0.1517 | 0.2076 | 0.1331 | 0.2169 | 559,751 | 1200 |
| A1M | T0_global | global | — | 3 | 0.1789 | 0.0010 | 0.1703 | -0.0086 | 0.1521 | 0.2078 | 0.1328 | 0.2164 | 3,125,831 | 1200 |
| A1M | T1_spatial | parent-relative | spatial | 3 | 0.1832 | 0.0014 | 0.1703 | -0.0129 | 0.1493 | 0.2202 | 0.1388 | 0.2178 | 3,125,831 | 1200 |
| A1M | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1789 | 0.0010 | 0.1703 | -0.0086 | 0.1521 | 0.2078 | 0.1328 | 0.2164 | 3,125,831 | 1200 |
| A3Lite | T0_global | global | — | 3 | 0.1761 | 0.0011 | 0.1703 | -0.0059 | 0.1486 | 0.2053 | 0.1266 | 0.2168 | 1,252,615 | 1200 |
| A3Lite | T1_spatial | parent-relative | spatial | 3 | 0.1794 | 0.0042 | 0.1703 | -0.0092 | 0.1463 | 0.2169 | 0.1352 | 0.2095 | 1,252,615 | 1200 |
| A3Lite | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1761 | 0.0011 | 0.1703 | -0.0059 | 0.1486 | 0.2053 | 0.1266 | 0.2168 | 1,252,615 | 1200 |
| A3L | T0_global | global | — | 3 | 0.1768 | 0.0021 | 0.1703 | -0.0065 | 0.1477 | 0.2065 | 0.1331 | 0.2167 | 1,226,511 | 1200 |
| A3L | T1_spatial | parent-relative | spatial | 3 | 0.1811 | 0.0018 | 0.1703 | -0.0108 | 0.1476 | 0.2195 | 0.1352 | 0.2103 | 1,226,511 | 1200 |
| A3L | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1768 | 0.0021 | 0.1703 | -0.0065 | 0.1477 | 0.2065 | 0.1331 | 0.2167 | 1,226,511 | 1200 |
| A3 | T0_global | global | — | 3 | 0.1743 | 0.0016 | 0.1703 | -0.0040 | 0.1454 | 0.2046 | 0.1260 | 0.2147 | 3,226,647 | 1200 |
| A3 | T1_spatial | parent-relative | spatial | 3 | 0.1813 | 0.0029 | 0.1703 | -0.0110 | 0.1452 | 0.2228 | 0.1358 | 0.2092 | 3,226,647 | 1200 |
| A3 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1743 | 0.0016 | 0.1703 | -0.0040 | 0.1454 | 0.2046 | 0.1260 | 0.2147 | 3,226,647 | 1200 |

### §29 results — test_combination

Split `test_combination`, inferred placement, global-frame translation error (lower is better). Floor = 0.1750. Margin = floor − mean; positive beats the floor. Depth columns are position error by depth in the spatial tree, seed mean.

| arm | cell | target | hierarchy | seeds | mean translation | seed std | floor | floor margin | depth 0 | depth 1 | depth 2 | depth 3 | parameters | steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | global | — | 3 | 0.1909 | 0.0018 | 0.1750 | -0.0159 | 0.1517 | 0.2305 | 0.1359 | 0.2520 | 559,751 | 1200 |
| A1 | T1_spatial | parent-relative | spatial | 3 | 0.1951 | 0.0030 | 0.1750 | -0.0202 | 0.1514 | 0.2358 | 0.1408 | 0.2624 | 559,751 | 1200 |
| A1 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1909 | 0.0018 | 0.1750 | -0.0159 | 0.1517 | 0.2305 | 0.1359 | 0.2520 | 559,751 | 1200 |
| A1M | T0_global | global | — | 3 | 0.1922 | 0.0020 | 0.1750 | -0.0172 | 0.1507 | 0.2356 | 0.1383 | 0.2481 | 3,125,831 | 1200 |
| A1M | T1_spatial | parent-relative | spatial | 3 | 0.1869 | 0.0002 | 0.1750 | -0.0119 | 0.1466 | 0.2231 | 0.1383 | 0.2547 | 3,125,831 | 1200 |
| A1M | T2_taxonomic | parent-relative | taxonomic | 3 | 0.1922 | 0.0020 | 0.1750 | -0.0172 | 0.1507 | 0.2356 | 0.1383 | 0.2481 | 3,125,831 | 1200 |
| A3Lite | T0_global | global | — | 3 | 0.2092 | 0.0200 | 0.1750 | -0.0342 | 0.1932 | 0.2418 | 0.0992 | 0.2511 | 1,252,615 | 1200 |
| A3Lite | T1_spatial | parent-relative | spatial | 3 | 0.2143 | 0.0315 | 0.1750 | -0.0393 | 0.1792 | 0.2503 | 0.1154 | 0.3137 | 1,252,615 | 1200 |
| A3Lite | T2_taxonomic | parent-relative | taxonomic | 3 | 0.2092 | 0.0200 | 0.1750 | -0.0342 | 0.1932 | 0.2418 | 0.0992 | 0.2511 | 1,252,615 | 1200 |
| A3L | T0_global | global | — | 3 | 0.2033 | 0.0087 | 0.1750 | -0.0284 | 0.1741 | 0.2346 | 0.1383 | 0.2512 | 1,226,511 | 1200 |
| A3L | T1_spatial | parent-relative | spatial | 3 | 0.1958 | 0.0023 | 0.1750 | -0.0208 | 0.1633 | 0.2273 | 0.1106 | 0.3021 | 1,226,511 | 1200 |
| A3L | T2_taxonomic | parent-relative | taxonomic | 3 | 0.2033 | 0.0087 | 0.1750 | -0.0284 | 0.1741 | 0.2346 | 0.1383 | 0.2512 | 1,226,511 | 1200 |
| A3 | T0_global | global | — | 3 | 0.2218 | 0.0064 | 0.1750 | -0.0468 | 0.2105 | 0.2557 | 0.0985 | 0.2458 | 3,226,647 | 1200 |
| A3 | T1_spatial | parent-relative | spatial | 3 | 0.2163 | 0.0224 | 0.1750 | -0.0413 | 0.1787 | 0.2556 | 0.1153 | 0.3198 | 3,226,647 | 1200 |
| A3 | T2_taxonomic | parent-relative | taxonomic | 3 | 0.2218 | 0.0064 | 0.1750 | -0.0468 | 0.2105 | 0.2557 | 0.0985 | 0.2458 | 3,226,647 | 1200 |

### Paired: spatial vs global — test_seen

Split `test_seen`: `T1_spatial` − `T0_global`, paired by seed. Negative = parent-relative placed better. Rule fixed before results: exceeds seed variability only if every seed agrees in sign AND |t| > 4.303 (df = 2).

| arm | scope | seed 0 | seed 1 | seed 2 | mean | std | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | all entities | +0.0108 | +0.0114 | +0.0073 | +0.0098 | 0.0022 | +7.64 | yes | YES |
| A1 | depth 0 | -0.0020 | +0.0029 | +0.0006 | +0.0005 | 0.0025 | +0.35 | no | no |
| A1 | depth 1 | +0.0277 | +0.0240 | +0.0159 | +0.0226 | 0.0060 | +6.49 | yes | YES |
| A1 | depth 2 | +0.0067 | +0.0067 | +0.0064 | +0.0066 | 0.0001 | +81.42 | yes | YES |
| A1 | depth 3 | +0.0020 | +0.0009 | +0.0029 | +0.0020 | 0.0010 | +3.33 | yes | no |
| A1M | all entities | +0.0043 | +0.0069 | +0.0056 | +0.0056 | 0.0013 | +7.66 | yes | YES |
| A1M | depth 0 | -0.0004 | -0.0045 | +0.0004 | -0.0015 | 0.0026 | -1.01 | no | no |
| A1M | depth 1 | +0.0106 | +0.0206 | +0.0127 | +0.0146 | 0.0053 | +4.80 | yes | YES |
| A1M | depth 2 | +0.0052 | +0.0075 | +0.0055 | +0.0061 | 0.0012 | +8.50 | yes | YES |
| A1M | depth 3 | +0.0017 | +0.0005 | +0.0016 | +0.0013 | 0.0007 | +3.18 | yes | no |
| A3Lite | all entities | +0.0039 | +0.0077 | +0.0038 | +0.0051 | 0.0022 | +3.99 | yes | no |
| A3Lite | depth 0 | +0.0005 | +0.0015 | +0.0047 | +0.0022 | 0.0022 | +1.76 | yes | no |
| A3Lite | depth 1 | +0.0097 | +0.0183 | +0.0042 | +0.0107 | 0.0071 | +2.61 | yes | no |
| A3Lite | depth 2 | +0.0121 | +0.0085 | +0.0176 | +0.0128 | 0.0046 | +4.81 | yes | YES |
| A3Lite | depth 3 | -0.0186 | -0.0137 | -0.0192 | -0.0172 | 0.0030 | -9.80 | yes | YES |
| A3L | all entities | +0.0051 | +0.0039 | +0.0052 | +0.0047 | 0.0007 | +11.67 | yes | YES |
| A3L | depth 0 | -0.0015 | +0.0028 | +0.0030 | +0.0014 | 0.0026 | +0.96 | no | no |
| A3L | depth 1 | +0.0183 | +0.0111 | +0.0144 | +0.0146 | 0.0036 | +7.07 | yes | YES |
| A3L | depth 2 | -0.0000 | -0.0001 | -0.0004 | -0.0002 | 0.0002 | -1.42 | yes | no |
| A3L | depth 3 | -0.0149 | -0.0171 | -0.0164 | -0.0162 | 0.0011 | -24.70 | yes | YES |
| A3 | all entities | +0.0067 | +0.0151 | +0.0149 | +0.0122 | 0.0048 | +4.43 | yes | YES |
| A3 | depth 0 | +0.0057 | +0.0095 | +0.0095 | +0.0082 | 0.0022 | +6.45 | yes | YES |
| A3 | depth 1 | +0.0101 | +0.0272 | +0.0249 | +0.0207 | 0.0093 | +3.88 | yes | no |
| A3 | depth 2 | +0.0164 | +0.0142 | +0.0127 | +0.0145 | 0.0019 | +13.39 | yes | YES |
| A3 | depth 3 | -0.0149 | -0.0160 | -0.0057 | -0.0122 | 0.0057 | -3.73 | yes | no |

### Paired: spatial vs global — test_arrangement

Split `test_arrangement`: `T1_spatial` − `T0_global`, paired by seed. Negative = parent-relative placed better. Rule fixed before results: exceeds seed variability only if every seed agrees in sign AND |t| > 4.303 (df = 2).

| arm | scope | seed 0 | seed 1 | seed 2 | mean | std | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | all entities | +0.0092 | +0.0128 | +0.0069 | +0.0096 | 0.0030 | +5.60 | yes | YES |
| A1 | depth 0 | -0.0044 | +0.0049 | -0.0003 | +0.0000 | 0.0047 | +0.01 | no | no |
| A1 | depth 1 | +0.0265 | +0.0232 | +0.0154 | +0.0217 | 0.0057 | +6.60 | yes | YES |
| A1 | depth 2 | +0.0089 | +0.0074 | +0.0085 | +0.0083 | 0.0008 | +18.69 | yes | YES |
| A1 | depth 3 | +0.0023 | +0.0007 | +0.0022 | +0.0017 | 0.0009 | +3.44 | yes | no |
| A1M | all entities | +0.0042 | +0.0079 | +0.0058 | +0.0059 | 0.0019 | +5.51 | yes | YES |
| A1M | depth 0 | -0.0019 | -0.0038 | -0.0004 | -0.0020 | 0.0017 | -2.05 | yes | no |
| A1M | depth 1 | +0.0111 | +0.0211 | +0.0129 | +0.0150 | 0.0053 | +4.89 | yes | YES |
| A1M | depth 2 | +0.0060 | +0.0066 | +0.0065 | +0.0064 | 0.0003 | +36.11 | yes | YES |
| A1M | depth 3 | +0.0012 | +0.0011 | +0.0015 | +0.0013 | 0.0002 | +12.41 | yes | YES |
| A3Lite | all entities | +0.0050 | +0.0078 | +0.0026 | +0.0051 | 0.0026 | +3.42 | yes | no |
| A3Lite | depth 0 | -0.0001 | +0.0003 | +0.0033 | +0.0011 | 0.0018 | +1.07 | no | no |
| A3Lite | depth 1 | +0.0114 | +0.0188 | +0.0022 | +0.0108 | 0.0083 | +2.25 | yes | no |
| A3Lite | depth 2 | +0.0114 | +0.0092 | +0.0153 | +0.0120 | 0.0031 | +6.67 | yes | YES |
| A3Lite | depth 3 | -0.0231 | -0.0157 | -0.0227 | -0.0205 | 0.0042 | -8.49 | yes | YES |
| A3L | all entities | +0.0041 | +0.0056 | +0.0050 | +0.0049 | 0.0008 | +10.88 | yes | YES |
| A3L | depth 0 | -0.0032 | +0.0041 | +0.0024 | +0.0011 | 0.0038 | +0.51 | no | no |
| A3L | depth 1 | +0.0182 | +0.0122 | +0.0146 | +0.0150 | 0.0030 | +8.59 | yes | YES |
| A3L | depth 2 | -0.0016 | -0.0015 | -0.0028 | -0.0020 | 0.0007 | -4.80 | yes | YES |
| A3L | depth 3 | -0.0177 | -0.0208 | -0.0191 | -0.0192 | 0.0016 | -21.38 | yes | YES |
| A3 | all entities | +0.0053 | +0.0157 | +0.0151 | +0.0120 | 0.0059 | +3.56 | yes | no |
| A3 | depth 0 | +0.0044 | +0.0092 | +0.0095 | +0.0077 | 0.0029 | +4.62 | yes | YES |
| A3 | depth 1 | +0.0085 | +0.0281 | +0.0251 | +0.0206 | 0.0105 | +3.38 | yes | no |
| A3 | depth 2 | +0.0145 | +0.0131 | +0.0119 | +0.0132 | 0.0013 | +17.10 | yes | YES |
| A3 | depth 3 | -0.0228 | -0.0199 | -0.0109 | -0.0178 | 0.0062 | -4.99 | yes | YES |

### Paired: spatial vs global — test_transform

Split `test_transform`: `T1_spatial` − `T0_global`, paired by seed. Negative = parent-relative placed better. Rule fixed before results: exceeds seed variability only if every seed agrees in sign AND |t| > 4.303 (df = 2).

| arm | scope | seed 0 | seed 1 | seed 2 | mean | std | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | all entities | +0.0083 | +0.0133 | +0.0067 | +0.0095 | 0.0034 | +4.80 | yes | YES |
| A1 | depth 0 | -0.0045 | +0.0055 | +0.0000 | +0.0003 | 0.0050 | +0.12 | no | no |
| A1 | depth 1 | +0.0245 | +0.0222 | +0.0138 | +0.0202 | 0.0056 | +6.20 | yes | YES |
| A1 | depth 2 | +0.0076 | +0.0074 | +0.0080 | +0.0077 | 0.0003 | +45.28 | yes | YES |
| A1 | depth 3 | +0.0024 | +0.0006 | +0.0032 | +0.0021 | 0.0013 | +2.74 | yes | no |
| A1M | all entities | +0.0022 | +0.0061 | +0.0045 | +0.0043 | 0.0020 | +3.78 | yes | no |
| A1M | depth 0 | -0.0018 | -0.0064 | -0.0002 | -0.0028 | 0.0032 | -1.52 | yes | no |
| A1M | depth 1 | +0.0073 | +0.0197 | +0.0102 | +0.0124 | 0.0065 | +3.30 | yes | no |
| A1M | depth 2 | +0.0056 | +0.0068 | +0.0058 | +0.0061 | 0.0007 | +15.82 | yes | YES |
| A1M | depth 3 | +0.0019 | +0.0003 | +0.0021 | +0.0014 | 0.0010 | +2.51 | yes | no |
| A3Lite | all entities | +0.0022 | +0.0068 | +0.0009 | +0.0033 | 0.0031 | +1.85 | yes | no |
| A3Lite | depth 0 | -0.0025 | -0.0004 | -0.0042 | -0.0024 | 0.0019 | -2.10 | yes | no |
| A3Lite | depth 1 | +0.0094 | +0.0169 | +0.0085 | +0.0116 | 0.0046 | +4.36 | yes | YES |
| A3Lite | depth 2 | +0.0077 | +0.0074 | +0.0110 | +0.0087 | 0.0020 | +7.58 | yes | YES |
| A3Lite | depth 3 | -0.0086 | -0.0043 | -0.0091 | -0.0073 | 0.0026 | -4.89 | yes | YES |
| A3L | all entities | +0.0041 | +0.0047 | +0.0041 | +0.0043 | 0.0003 | +21.70 | yes | YES |
| A3L | depth 0 | -0.0026 | +0.0021 | +0.0003 | -0.0001 | 0.0024 | -0.05 | no | no |
| A3L | depth 1 | +0.0155 | +0.0106 | +0.0129 | +0.0130 | 0.0025 | +9.14 | yes | YES |
| A3L | depth 2 | +0.0028 | +0.0021 | +0.0014 | +0.0021 | 0.0007 | +5.00 | yes | YES |
| A3L | depth 3 | -0.0050 | -0.0072 | -0.0069 | -0.0064 | 0.0012 | -9.26 | yes | YES |
| A3 | all entities | +0.0022 | +0.0083 | +0.0107 | +0.0070 | 0.0044 | +2.78 | yes | no |
| A3 | depth 0 | -0.0022 | -0.0015 | +0.0032 | -0.0001 | 0.0029 | -0.09 | no | no |
| A3 | depth 1 | +0.0092 | +0.0233 | +0.0223 | +0.0182 | 0.0079 | +4.02 | yes | no |
| A3 | depth 2 | +0.0106 | +0.0098 | +0.0089 | +0.0098 | 0.0009 | +19.91 | yes | YES |
| A3 | depth 3 | -0.0069 | -0.0063 | -0.0033 | -0.0055 | 0.0019 | -4.94 | yes | YES |

### Paired: spatial vs global — test_combination

Split `test_combination`: `T1_spatial` − `T0_global`, paired by seed. Negative = parent-relative placed better. Rule fixed before results: exceeds seed variability only if every seed agrees in sign AND |t| > 4.303 (df = 2).

| arm | scope | seed 0 | seed 1 | seed 2 | mean | std | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | all entities | +0.0056 | +0.0042 | +0.0031 | +0.0043 | 0.0013 | +5.81 | yes | YES |
| A1 | depth 0 | -0.0005 | -0.0014 | +0.0010 | -0.0003 | 0.0012 | -0.38 | no | no |
| A1 | depth 1 | +0.0082 | +0.0079 | -0.0002 | +0.0053 | 0.0047 | +1.93 | no | no |
| A1 | depth 2 | +0.0060 | +0.0044 | +0.0042 | +0.0049 | 0.0010 | +8.42 | yes | YES |
| A1 | depth 3 | +0.0102 | +0.0026 | +0.0185 | +0.0104 | 0.0080 | +2.26 | yes | no |
| A1M | all entities | -0.0075 | -0.0038 | -0.0045 | -0.0053 | 0.0020 | -4.65 | yes | YES |
| A1M | depth 0 | -0.0028 | -0.0056 | -0.0037 | -0.0040 | 0.0014 | -4.88 | yes | YES |
| A1M | depth 1 | -0.0201 | -0.0063 | -0.0111 | -0.0125 | 0.0070 | -3.10 | yes | no |
| A1M | depth 2 | -0.0016 | +0.0008 | +0.0006 | -0.0001 | 0.0013 | -0.10 | no | no |
| A1M | depth 3 | +0.0134 | -0.0050 | +0.0115 | +0.0066 | 0.0101 | +1.14 | no | no |
| A3Lite | all entities | -0.0098 | +0.0045 | +0.0206 | +0.0051 | 0.0152 | +0.58 | no | no |
| A3Lite | depth 0 | -0.0247 | +0.0025 | -0.0198 | -0.0140 | 0.0145 | -1.67 | no | no |
| A3Lite | depth 1 | -0.0167 | -0.0048 | +0.0469 | +0.0085 | 0.0338 | +0.44 | no | no |
| A3Lite | depth 2 | +0.0180 | +0.0102 | +0.0204 | +0.0162 | 0.0053 | +5.27 | yes | YES |
| A3Lite | depth 3 | +0.0721 | +0.0346 | +0.0813 | +0.0627 | 0.0247 | +4.39 | yes | YES |
| A3L | all entities | -0.0004 | -0.0144 | -0.0078 | -0.0075 | 0.0070 | -1.87 | yes | no |
| A3L | depth 0 | +0.0026 | -0.0230 | -0.0119 | -0.0108 | 0.0128 | -1.45 | no | no |
| A3L | depth 1 | -0.0057 | -0.0120 | -0.0042 | -0.0073 | 0.0041 | -3.06 | yes | no |
| A3L | depth 2 | -0.0281 | -0.0241 | -0.0311 | -0.0278 | 0.0035 | -13.62 | yes | YES |
| A3L | depth 3 | +0.0515 | +0.0533 | +0.0479 | +0.0509 | 0.0027 | +32.13 | yes | YES |
| A3 | all entities | +0.0170 | -0.0240 | -0.0093 | -0.0055 | 0.0208 | -0.46 | no | no |
| A3 | depth 0 | -0.0127 | -0.0494 | -0.0333 | -0.0318 | 0.0184 | -2.99 | yes | no |
| A3 | depth 1 | +0.0283 | -0.0249 | -0.0035 | -0.0001 | 0.0268 | -0.00 | no | no |
| A3 | depth 2 | +0.0177 | +0.0169 | +0.0158 | +0.0168 | 0.0010 | +30.54 | yes | YES |
| A3 | depth 3 | +0.0988 | +0.0607 | +0.0627 | +0.0741 | 0.0214 | +5.99 | yes | YES |

### Depth analysis

Split `test_seen`. Depth from the spatial tree for every cell. Position error, then scale error, each seed mean ± std.

| arm | cell | depth | entities | position | scale |
| --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 0 | 1000 | 0.1460 ± 0.0032 | 0.1152 ± 0.0004 |
| A1 | T0_global | 1 | 850 | 0.2000 ± 0.0001 | 0.1540 ± 0.0002 |
| A1 | T0_global | 2 | 300 | 0.1092 ± 0.0014 | 0.1228 ± 0.0002 |
| A1 | T0_global | 3 | 150 | 0.1698 ± 0.0003 | 0.1523 ± 0.0001 |
| A1 | T1_spatial | 0 | 1000 | 0.1465 ± 0.0008 | 0.1148 ± 0.0001 |
| A1 | T1_spatial | 1 | 850 | 0.2226 ± 0.0061 | 0.1575 ± 0.0001 |
| A1 | T1_spatial | 2 | 300 | 0.1158 ± 0.0013 | 0.1474 ± 0.0013 |
| A1 | T1_spatial | 3 | 150 | 0.1718 ± 0.0007 | 0.1615 ± 0.0005 |
| A1M | T0_global | 0 | 1000 | 0.1466 ± 0.0019 | 0.1155 ± 0.0004 |
| A1M | T0_global | 1 | 850 | 0.2001 ± 0.0005 | 0.1536 ± 0.0000 |
| A1M | T0_global | 2 | 300 | 0.1086 ± 0.0012 | 0.1228 ± 0.0002 |
| A1M | T0_global | 3 | 150 | 0.1696 ± 0.0003 | 0.1521 ± 0.0003 |
| A1M | T1_spatial | 0 | 1000 | 0.1451 ± 0.0008 | 0.1150 ± 0.0002 |
| A1M | T1_spatial | 1 | 850 | 0.2148 ± 0.0050 | 0.1578 ± 0.0006 |
| A1M | T1_spatial | 2 | 300 | 0.1146 ± 0.0001 | 0.1468 ± 0.0014 |
| A1M | T1_spatial | 3 | 150 | 0.1709 ± 0.0005 | 0.1619 ± 0.0008 |
| A3Lite | T0_global | 0 | 1000 | 0.1356 ± 0.0082 | 0.1107 ± 0.0005 |
| A3Lite | T0_global | 1 | 850 | 0.1953 ± 0.0029 | 0.1531 ± 0.0001 |
| A3Lite | T0_global | 2 | 300 | 0.0971 ± 0.0018 | 0.1201 ± 0.0002 |
| A3Lite | T0_global | 3 | 150 | 0.1699 ± 0.0006 | 0.1523 ± 0.0003 |
| A3Lite | T1_spatial | 0 | 1000 | 0.1378 ± 0.0065 | 0.1107 ± 0.0001 |
| A3Lite | T1_spatial | 1 | 850 | 0.2060 ± 0.0097 | 0.1577 ± 0.0004 |
| A3Lite | T1_spatial | 2 | 300 | 0.1098 ± 0.0029 | 0.1476 ± 0.0020 |
| A3Lite | T1_spatial | 3 | 150 | 0.1527 ± 0.0036 | 0.1626 ± 0.0009 |
| A3L | T0_global | 0 | 1000 | 0.1403 ± 0.0028 | 0.1125 ± 0.0012 |
| A3L | T0_global | 1 | 850 | 0.1973 ± 0.0002 | 0.1529 ± 0.0000 |
| A3L | T0_global | 2 | 300 | 0.1091 ± 0.0003 | 0.1229 ± 0.0003 |
| A3L | T0_global | 3 | 150 | 0.1699 ± 0.0004 | 0.1521 ± 0.0003 |
| A3L | T1_spatial | 0 | 1000 | 0.1417 ± 0.0005 | 0.1110 ± 0.0004 |
| A3L | T1_spatial | 1 | 850 | 0.2119 ± 0.0036 | 0.1574 ± 0.0007 |
| A3L | T1_spatial | 2 | 300 | 0.1090 ± 0.0001 | 0.1482 ± 0.0005 |
| A3L | T1_spatial | 3 | 150 | 0.1537 ± 0.0008 | 0.1628 ± 0.0002 |
| A3 | T0_global | 0 | 1000 | 0.1298 ± 0.0009 | 0.1107 ± 0.0001 |
| A3 | T0_global | 1 | 850 | 0.1931 ± 0.0018 | 0.1529 ± 0.0003 |
| A3 | T0_global | 2 | 300 | 0.0964 ± 0.0003 | 0.1200 ± 0.0001 |
| A3 | T0_global | 3 | 150 | 0.1642 ± 0.0056 | 0.1518 ± 0.0002 |
| A3 | T1_spatial | 0 | 1000 | 0.1380 ± 0.0029 | 0.1106 ± 0.0002 |
| A3 | T1_spatial | 1 | 850 | 0.2138 ± 0.0092 | 0.1576 ± 0.0003 |
| A3 | T1_spatial | 2 | 300 | 0.1108 ± 0.0018 | 0.1478 ± 0.0014 |
| A3 | T1_spatial | 3 | 150 | 0.1520 ± 0.0004 | 0.1627 ± 0.0008 |

### Taxonomic control vs global control

| arm / seed | max abs difference, all splits | bit-identical |
| --- | --- | --- |
| A1/seed0 | 0.000e+00 | yes |
| A1/seed1 | 0.000e+00 | yes |
| A1/seed2 | 0.000e+00 | yes |
| A1M/seed0 | 0.000e+00 | yes |
| A1M/seed1 | 0.000e+00 | yes |
| A1M/seed2 | 0.000e+00 | yes |
| A3/seed0 | 0.000e+00 | yes |
| A3/seed1 | 0.000e+00 | yes |
| A3/seed2 | 0.000e+00 | yes |
| A3L/seed0 | 0.000e+00 | yes |
| A3L/seed1 | 0.000e+00 | yes |
| A3L/seed2 | 0.000e+00 | yes |
| A3Lite/seed0 | 0.000e+00 | yes |
| A3Lite/seed1 | 0.000e+00 | yes |
| A3Lite/seed2 | 0.000e+00 | yes |

### Step 9 reproduction by the global control

| arm / seed | Step 10 T0_global | Step 9 baseline | abs difference | identical |
| --- | --- | --- | --- | --- |
| A1/seed0 | 0.169034 | 0.169034 | 0.000e+00 | yes |
| A1/seed1 | 0.166109 | 0.166109 | 0.000e+00 | yes |
| A1/seed2 | 0.167716 | 0.167716 | 0.000e+00 | yes |
| A1M/seed0 | 0.167852 | 0.167852 | 0.000e+00 | yes |
| A1M/seed1 | 0.168701 | 0.168701 | 0.000e+00 | yes |
| A1M/seed2 | 0.167197 | 0.167197 | 0.000e+00 | yes |
| A3/seed0 | 0.156015 | 0.156015 | 0.000e+00 | yes |
| A3/seed1 | 0.155644 | 0.155644 | 0.000e+00 | yes |
| A3/seed2 | 0.156730 | 0.156730 | 0.000e+00 | yes |
| A3L/seed0 | 0.165497 | 0.165497 | 0.000e+00 | yes |
| A3L/seed1 | 0.163588 | 0.163588 | 0.000e+00 | yes |
| A3L/seed2 | 0.164003 | 0.164003 | 0.000e+00 | yes |
| A3Lite/seed0 | 0.161387 | 0.161387 | 0.000e+00 | yes |
| A3Lite/seed1 | 0.163930 | 0.163930 | 0.000e+00 | yes |
| A3Lite/seed2 | 0.154521 | 0.154521 | 0.000e+00 | yes |

### Compute and capacity

| arm | cell | parameters | steps | batch | torch threads | seeds | train s (mean) | eval s (mean) | depth s (mean) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 559,751 | 1200 | 8 | 4 | 0,1,2 | 1742 | 67 | 13 |
| A1 | T1_spatial | 559,751 | 1200 | 8 | 4 | 0,1,2 | 1468 | 67 | 14 |
| A1 | T2_taxonomic | 559,751 | 1200 | 8 | 4 | 0,1,2 | 11500 | 67 | 14 |
| A1M | T0_global | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 5679 | 74 | 14 |
| A1M | T1_spatial | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 1862 | 95 | 19 |
| A1M | T2_taxonomic | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 1702 | 76 | 15 |
| A3Lite | T0_global | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 1907 | 99 | 20 |
| A3Lite | T1_spatial | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 1842 | 90 | 17 |
| A3Lite | T2_taxonomic | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 1793 | 84 | 16 |
| A3L | T0_global | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1675 | 76 | 15 |
| A3L | T1_spatial | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1668 | 82 | 20 |
| A3L | T2_taxonomic | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1682 | 74 | 15 |
| A3 | T0_global | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1779 | 87 | 16 |
| A3 | T1_spatial | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1858 | 89 | 18 |
| A3 | T2_taxonomic | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1734 | 87 | 15 |

### Integrity

| check | result |
| --- | --- |
| runs that passed the gate before training | 45 of 45 |
| split-level gate passes | 180 |
| runs composing a hierarchy | 30 |
| worst round-trip error, stored and probe-rotated | 5.00e-16 |
| worst depth-vs-headline reconciliation | 0.00e+00 |
| largest stored rotation deviation from identity | 0.00e+00 |

### Inheritance diagnostic (post hoc)

**Post hoc, exploratory** — added after the confirmatory results were known. Split `test_seen`, translation error of parented children, seed mean. `true parent`: the child's predicted local frame composed onto its parent's true frame. `true parent scale`: onto its predicted parent with the scale replaced by the truth.

| arm | depth | T0_global | T1 as trained | T1, true parent | T1, true parent scale |
| --- | --- | --- | --- | --- | --- |
| A1M | 1 | 0.2001 | 0.2148 | 0.1365 | 0.2148 |
| A1M | 2 | 0.1086 | 0.1146 | 0.1467 | 0.1127 |
| A1M | 3 | 0.1696 | 0.1709 | 0.1141 | 0.1686 |
| A1 | 1 | 0.2000 | 0.2226 | 0.1505 | 0.2226 |
| A1 | 2 | 0.1092 | 0.1158 | 0.1513 | 0.1135 |
| A1 | 3 | 0.1698 | 0.1718 | 0.1148 | 0.1698 |
| A3L | 1 | 0.1973 | 0.2119 | 0.1354 | 0.2120 |
| A3L | 2 | 0.1091 | 0.1090 | 0.1472 | 0.1054 |
| A3L | 3 | 0.1699 | 0.1537 | 0.1138 | 0.1522 |
| A3Lite | 1 | 0.1953 | 0.2060 | 0.1334 | 0.2062 |
| A3Lite | 2 | 0.0971 | 0.1098 | 0.1453 | 0.1063 |
| A3Lite | 3 | 0.1699 | 0.1527 | 0.1141 | 0.1512 |
| A3 | 1 | 0.1931 | 0.2138 | 0.1409 | 0.2141 |
| A3 | 2 | 0.0964 | 0.1108 | 0.1455 | 0.1068 |
| A3 | 3 | 0.1642 | 0.1520 | 0.1146 | 0.1504 |

<!-- END GENERATED TABLES -->

---

## 7. Seed-spread analysis

Read from the generated `§29` and paired tables.

* **The direction is not a seed effect.** On `test_seen` every one of the fifteen paired
  seed differences is positive: all three seeds of all five arms placed worse under
  parent-relative.
* **The size clears the rule in four arms of five.** Paired t: A1 7.64, A1M 7.66,
  A3L 11.67, A3 4.43; A3Lite 3.99, below 4.303. A3Lite's result is recorded as *not
  established*, not as absent — its three differences are +0.0039, +0.0077, +0.0038.
* **Parent-relative makes training less repeatable.** Across-seed standard deviation rises
  in every arm: A1 0.0015 → 0.0024, A1M 0.0008 → 0.0018, A3Lite 0.0049 → 0.0067,
  A3L 0.0010 → 0.0015, A3 0.0006 → 0.0049. A3 goes from the most repeatable arm to one of
  the least, which is what an error inherited through a chain of predictions would do.
* **The effect sizes dwarf the controls' noise.** The taxonomic control and the Step 9
  reproduction differ from their references by exactly zero, so there is no measurement
  noise between cells to confuse with the effect; the only variation is seed variation,
  and the rule is applied against that.

---

## 8. Depth analysis (§20 / §30)

Depth is read from the spatial tree for every cell, so each bucket names the same entities
in the global and parent-relative runs. Depth 0 is ten roots (1,000 entity instances on
`test_seen`); depth 1 is the four valves and three great veins (850); depth 2 the aorta and
pulmonary trunk (300); depth 3 the pulmonary arteries (150).

| question | answer on `test_seen` (paired table) |
| --- | --- |
| Does depth 0 stay approximately unchanged? | **Yes in four arms, no in A3.** A1, A1M, A3Lite and A3L move by −0.0015 to +0.0022, none passing the rule. A3's roots got worse by +0.0082 (t = 6.45). |
| Does depth 1 improve? | **No — it carries the loss.** Worse in all five arms, +0.0107 to +0.0226; passes the rule in A1, A1M and A3L. |
| Does depth 2 improve? | **No.** Worse in four arms (+0.0061 to +0.0145, all passing); flat in A3L (−0.0002). |
| Does depth 3 improve? | **Yes, in the graph arms only.** A3Lite −0.0172 (t = −9.80), A3L −0.0162 (t = −24.70), A3 −0.0122 (t = −3.73, same sign, short of the rule). The no-graph arms are marginally worse. |

**What that means for the mechanism.** The brief's test was that an aggregate change the
depth-≥1 entities do not share is evidence against the mechanism. The opposite holds here:
the aggregate change *is* the depth-≥1 change, and depth 0 is stable in four arms. The
mechanism acts where it should. It acts harmfully at depths 1–2 and helpfully at depth 3.

**The A3 exception.** A3's roots degraded under a target change that does not touch their
target. That can only happen through shared parameters: training the head on
parent-relative targets for the other ten entities made the same network place the roots
worse. It means A3's aggregate loss is partly not the mechanism — and because roots are
the depth-1 entities' parents, it also feeds the depth-1 loss.

**Post hoc: where the depth-1 and depth-2 errors come from** (generated inheritance table).
Recomposing each parented child from its saved checkpoint:

* *Depth 1.* Composed onto its parent's **true** frame, the child's own local prediction
  scores 0.133–0.151 — well below both the parent-relative result as trained (0.206–0.223)
  and the global target (0.193–0.200). A3: 0.141 against 0.193. Fixing only the parent's
  scale moves the result by at most 0.0002. So the whole depth-1 loss is inherited parent
  *position* error, and the local prediction itself is the best of the three.
* *Depth 2.* The true parent makes the child **worse** (0.145–0.151 against 0.109–0.116 as
  trained). Trained end to end on composed frames, the child's local frame has learned to
  cancel its parent's systematic error, so the "local" frame is not independent of the
  parent's mistakes. Fixing the parent's scale removes 0.002–0.004 — the size of the scale
  cascade after training.
* *Depth 3.* Parent-relative already beats global in the graph arms, and the true parent
  would help further (≈0.114).

This is the floor analysis, reproduced by trained models: the parent-relative target is
worth about a quarter of the error to a predictor whose parents are right, and costs to
one whose parents are not. No arm's roots are right enough.

---

## 9. Taxonomic control

`T2_taxonomic` composes over AWR's own hierarchy, in which every whole-organ entity is a root.
It runs the parent-relative code path end to end — the `parent_relative` target, the
composition, the gate — with no parents to compose over.

**It reproduces nothing of the effect, because it reproduces the global control exactly.**
All fifteen arm-seeds are bit-identical to `T0_global` on every split (maximum absolute
difference 0.0); for A1 seed 0 every recorded metric on every split was checked, not only
translation. Two conclusions follow:

* the parent-relative machinery is exactly neutral when it has nothing to compose, so the
  `T1_spatial` effect is attributable to the spatial parent assignments, not to the code
  path, the gate or the target mode;
* merely imposing AWR's hierarchy delivers nothing — not better, not worse — because AWR's
  hierarchy has no geometric parents to offer. No `left_ventricle → mitral_valve`-style
  relationship exists in the ontology, and none was invented.

---

## 10. The scale cascade (§11)

The parent convention is isotropic: a parent passes its rotation and a single isotropic scale
to its children; the child keeps its own anisotropic scale. That convention is closed — the
composed map is always a similarity-times-diagonal, which the round trip confirms to 5e-16
with real rotations — but it has a consequence the rigid convention does not.

* **Isotropic:** a parent's scale error multiplies into every descendant, undiminished.
  `test_isotropic_propagates_a_parents_scale_error`.
* **Rigid:** a parent's scale error stays on the entity that made it.
  `test_rigid_confines_a_scale_error_to_the_entity_that_made_it`.

Both tests pass. The convention was not changed during this experiment; no mathematical
defect was found that would justify changing it.

**In the trained models** (generated depth table), the cascade is far smaller than at
initialisation — where parent-relative scale error ran 1.37 / 2.92 / 4.61 at depths 1–3 on
an untrained model — but it is still visible. Under `T1`, A3's scale error is unchanged at
depth 0 and higher at every parented depth, most at depth 2 (0.1200 → 0.1478). Its effect
on *translation*, measured post hoc by giving each parent its true scale, is
0.0020–0.0040 at depth 2, 0.0015–0.0023 at depth 3 and nil at depth 1 — secondary to inherited position error. A
rigid parent would remove that term; this is an estimate from the diagnostic, not a
result, because `T4_rigid` was not part of the pre-registered matrix and was not run.

---

## 11. Answers

**Q1. Does parent-relative training improve global placement compared with global-target
training?** No. It is worse in all five arms on `test_seen` (+0.0047 to +0.0122), and worse
in all fifteen arm × split cells on `test_seen`, `test_arrangement` and `test_transform`. On
`test_combination` it is mixed: worse for A1 and A3Lite, better on average for A1M, A3L and
A3.

**Q2. Does that improvement survive across three seeds?** There is no improvement on
`test_seen` to survive. What survives is the loss: all three seeds, all five arms, the same
sign.

**Q3. Does the improvement exceed seed variability?** The *loss* does, in A1, A1M, A3L and A3
on `test_seen`; A3Lite's does not reach the rule. Of the twenty arm × split comparisons one
*improvement* passes the rule — A1M on `test_combination` (−0.0053, t = −4.65). One pass in
twenty is about what chance gives at this threshold, it is on the split where every arm is
far below the floor, and it is not claimed.

**Q4. Does the improvement occur specifically at depth ≥ 1?** The *change* does. Depth 0 is
stable in four arms of five; depths 1 and 2 carry a loss and depth 3 a gain in the graph arms.
The single exception is A3, whose roots also got worse — an indirect effect through shared
parameters, and so a part of A3's loss that is not the mechanism.

**Q5. Does the taxonomic control reproduce the effect?** No. It is bit-identical to the
global control everywhere, so the effect belongs to the spatial parent assignments.

**Q6. Which arms, if any, beat the placement-blind floor?** On `test_seen` under the global
target, A3 does (+0.0044, all three seeds below 0.1605); A3Lite's mean is +0.0006 but only
one of its three seeds is below the floor, so it does not reliably. Under parent-relative,
no arm's mean beats the floor; A3Lite seed 2 (0.1583) is the only single run that does. On
the other splits, only A3 and A3Lite under the global target on `test_arrangement` are above
it; nothing is on `test_transform` or `test_combination`.

**Q7. Does the result support or weaken the pre-registered prediction?** It weakens it — the
prediction is **not supported**. Its second half holds in a narrow sense: the arms below the
floor gained nothing useful, and lost 2.9–5.9% where the blind floor analysis suggested
about 2%. Its first half fails outright: A3 did not gain; it lost the most (7.8%) and fell
below the floor. The premise underneath it — that clearing the floor means placing parents
well enough to benefit — is what the inheritance diagnostic refutes: A3's chambers are still
among roots placed at 0.130 on average under the global target (0.138 under parent-relative),
and at depth 1 that costs more than the target change saves.

---

## 12. Scientific conclusion

Everything below is about **this formulation** — isotropic parent convention, the
construction-derived spatial tree, the Step 9 loss and protocol — **on this corpus**, whose
rotations are all the identity. It does not show that hierarchical placement fails in
general, any more than a positive result would have shown it works in general.

### Established findings

1. On this corpus, parent-relative placement under the isotropic convention places worse than
   the global target in all five arms, on every seed; the loss exceeds seed variability in
   four arms on `test_seen` and appears in all fifteen cells of the three non-combination
   splits.
2. It removes the one floor-clearing result Step 9 had: A3 moves from above the floor to
   below it.
3. The effect is carried by the parented entities — depth 0 is stable in four arms — with a
   loss at depths 1–2 and a gain at depth 3 in the graph arms.
4. Imposing AWR's taxonomic hierarchy has no effect whatever: bit-identical to the global
   control. The parent-relative code path is exactly neutral without parents.
5. Parent-relative training makes results less repeatable across seeds in every arm.
6. The global control reproduces Step 9 bit for bit, so every comparison here is against
   the same baseline Step 9 reported.
7. On this corpus a transposed rotation cannot be detected from stored frames and cannot
   occur in the data; the integrity gate detects the transform-level defect only through
   its probe rotation.
8. *(Post hoc)* The depth-1 loss is inherited parent position error: the children's own
   local predictions are better than the global target's, and composing them onto
   predicted parents costs more than that gain.
9. *(Post hoc)* End-to-end training makes a child's local frame absorb its parent's
   systematic error, so parent-relative local frames are not parent-independent.

### Unsupported hypotheses

* That parent-relative placement helps, or fails, hierarchical placement in general.
* That the rigid convention would reverse the result. The diagnostic puts the scale term at
  no more than 0.004 of translation — too small to reverse a 0.005–0.012 loss on its own — but
  `T4_rigid` was not run.
* That parenting the AV annuli to the atria instead would help. `T3_alternative` was not run.
* That the fixed table's non-transposed pairing costs anything in transposed scenes.
  `test_arrangement` behaves like `test_seen`, which is no evidence of a large effect, but
  the factor was not isolated.
* That the depth-3 gain reflects a general benefit of long chains. It is two significant arms
  of three at one depth with one entity type.
* That A1M's `test_combination` improvement is real. It is one pass in twenty.

### Architectural consequences

1. **Keep the global target as the default.** Parent-relative placement is not adopted on
   this corpus. The implementation, the integrity gate and the diagnostics stay; the switch
   stays off.
2. **Placement quality at the roots is the prerequisite.** Parent-relative placement relocates
   the placement problem to the roots instead of solving it. Revisit a hierarchical target
   only when root placement has improved materially; the inheritance diagnostic gives the
   test — at depth 1 the true-parent result must be approached by the as-trained one.
3. **A parent-relative local frame is not a clean, parent-independent quantity.** Under end-to-end
   training it encodes compensation for the parent's error. Any later local-edit design that
   moves a parent and expects its children's local frames to carry over unchanged must
   account for this.
4. **The isotropic convention stays.** No mathematical defect was found; its scale cascade is
   real and documented but secondary after training.
5. **The integrity gate is permanent, and it must grow for Change 2** (section 13).
6. **Per-run artifacts, resume and the disk guard are the runner's contract from now on.**
   The first sweep lost everything to one kill; this one lost nothing to a day of sleep and
   battery.

### Open questions

* **Rotation.** On this corpus a parent can pass only translation and scale to its children.
  Orientation — a vessel's direction relative to its valve — is where a parent-relative
  frame should matter most, and it cannot be tested here. That is the central reason this
  negative result does not settle the question, and it needs Change 2's corpus.
* Does parent-relative placement pay once root placement improves?
* Would a selective hierarchy — parenting only where the parent is reliably placed, or only
  along long chains — keep the depth-3 gain without the depth-1 loss?
* Rigid against isotropic, once rotations are real.
* What local-edit semantics survive the co-adaptation in finding 9?

---

## 13. The Change 2 boundary

Change 2 was not started. No corpus was generated, no non-identity rotation was introduced,
no rotation target was trained, the floor was not changed, and no real data, meshes, dataset
scaling or local-edit experiments were added.

Change 2 needs a new corpus, because the question it asks — does predicting rotation help —
has no answer on data whose every rotation is the identity. The order is fixed:

```text
NEW ROTATED CORPUS
        ↓
NEW PLACEMENT-BLIND FLOOR
        ↓
ROTATION TARGET VALIDATION
        ↓
CHANGE 2 TRAINING
```

* **The new corpus gets its own floor.** 0.1605 is the Step 9 / current-corpus reference
  and is not a floor for any rotated corpus. No sentence of the form "Step 10 improved
  translation from 0.1605 to X" may be written with X from a rotated corpus; any
  cross-corpus comparison must go through a normalised diagnostic defined for the purpose,
  such as each arm's margin over its own corpus's floor.
* **Rotation target validation** must include a check this report shows the current gate
  cannot make: the stored rotation compared against the generator's own record, because a
  transposed stored rotation is still a valid rotation and survives every round trip.
* **The gate's probe rotation stays**, and the stored-frame round trip becomes informative
  once rotations are real.

---

## 14. Reproducibility

The runs executed from commit `66606a0` plus uncommitted changes, which are snapshotted —
diff, new module, file times and SHA-256 — in
`experiments/runs/step10-placement/code_snapshot/`. Every runtime file predates the worker
launch (2026-09-19 14:56:12).

```
python -m experiments.step10.run_placement --seeds 0 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --seeds 1 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --seeds 2 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --assemble-only --out experiments/runs/step10-placement
python -m experiments.step10.report_tables --all-splits --insert-into docs/STEP_10_CHANGE1_REPORT.md
```

**Regression.** The full suite passes — 706 tests, of which 30 are new in this pass (22
corruption and leakage, 8 analysis) — ruff is clean, and mypy is clean on both dependency
tiers (54 tier-one files, 92 tier-two).

Three workers ran concurrently, one per seed, each at Step 9's 4 torch threads. CPU results
depend on thread count, not on load, and every `T0_global` run is checked against the
Step 9 baseline for its arm and seed.

---

## 15. Completion checklist

- [x] Disk usage is safe — 10 GiB free at start, 8.5 GiB at completion, 2 GiB guard in the runner.
- [x] Required old artifacts preserved; nothing deleted (section 3).
- [x] All pre-registered arm cells run — 5 arms × 3 cells.
- [x] Three seeds completed for each required arm and target — 45 of 45 runs.
- [x] Taxonomic control completed — 15 runs, bit-identical to the global control.
- [x] Depth 0/1/2/3 analysis complete, reconciled to the headline in every run.
- [x] Parent-table corruption tests pass.
- [x] Rotation corruption tests pass.
- [x] Topological-order corruption tests pass.
- [x] Correct dataset passes integrity validation — 45 runs, 225 gate passes.
- [x] No placement leakage found (section 5).
- [x] Scale-cascade tests pass.
- [x] §29 table generated directly from artifacts (section 6).
- [x] Results reported with seed spread.
- [x] Floor margins reported.
- [x] Change 1 scientific conclusion written (section 12).
- [x] No new corpus generated.
- [x] Change 2 untouched.
