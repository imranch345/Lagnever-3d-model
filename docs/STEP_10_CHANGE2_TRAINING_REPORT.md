# STEP 10 — CHANGE 2 TRAINING REPORT

**Lagnav 3D — Learning position, rotation and scale on the rotated corpus.**

All data is procedurally generated. Nothing here is anatomy, is medically validated, or
supports any clinical claim. No real medical data, meshes or external 3D systems were used.
The corpus was not modified, regenerated or scaled, and neither was the model.

---

## 1. The frozen corpus

| | |
| --- | --- |
| corpus | `step10-rotated-1950-40` |
| derived from | `step8-continuous-1950-40` (Change 1), scene for scene |
| rotations | measured, mean 61.8°, max 179.98°, none below 1° |
| checksums | `experiments/runs/step10-change2/CORPUS_FROZEN.sha256`, 7 files |

Every run verifies all seven corpus files against those checksums **before training starts**
and refuses to run if any has changed (§25). The verification is recorded in each run file.

**The floors are this corpus's own**, per split, per component:

| split | position | rotation | scale |
| --- | ---: | ---: | ---: |
| test_seen | 0.1605 | 22.65° | 0.1120 |
| test_arrangement | 0.1655 | 22.54° | 0.1199 |
| test_transform | 0.1703 | 35.95° | 0.1248 |
| test_combination | 0.1750 | 26.10° | 0.1231 |

The position floor coincides with Change 1's because the corpus is derived and the centroids
are the same numbers — a measured property, not a reuse. Change 2 results are read against
**this** table, on the matching split, per component. A Change 2 model's raw score is never
set beside a Change 1 model's as though they shared a benchmark.

---

## 2. The loss formulation

The frame term keeps Step 8's structure and replaces one part of it:

```text
frame_term = mean over present entities of
      1.0  * | t_pred - t_true |_1                 (translation, unchanged)
    + 0.5  * | log s_pred - log s_true |_1          (scale, unchanged)
    + 0.33 * || R_pred - R_true ||_F^2 / 8          (rotation, NEW)

L_total = step8_losses + 2.0 * frame_term + lod_weight * lod_term
```

The rotation term was previously the absolute difference of the six stored numbers. That was
a defensible term only while the target's rotation was constant — on the Change 1 corpus it
was identically zero — and it has two defects once rotation is real. It is not a distance on
rotations, and it can be reduced by shrinking the stored basis vectors towards the target's
without changing the rotation they represent. The chordal term is taken **after** the model's
own Gram-Schmidt, so it measures the rotation and nothing else; a test pins that it is
invariant to inflating the basis while the old term is not.

Every other Step 8 term is preserved unchanged (§6). Nothing was made mathematically invalid
by the new rotation term.

---

## 3. The rotation representation

Unchanged, and already validated: the head emits the continuous **6D** representation, which
`frame_rotation` orthonormalises by Gram-Schmidt into a proper rotation matrix **before** any
composition. No Euler angles are regressed anywhere in the pipeline. The transform convention
is the frozen one — parent contributes rotation and isotropic scale, child contributes
rotation and full anisotropic scale — and the Change 2 corpus pass established that it is
closed, with no shear and no projection loss, at 0°, 30°, 90° and 180°.

**Training loss and reporting metric are deliberately different functions.** The chordal
distance is descended because it is smooth at zero; the geodesic angle is reported because it
is interpretable. Measured, not asserted: at 1e-3 rad the geodesic form's gradient is more
than 100× the chordal form's, which is the singularity that makes it unusable as a loss.

---

## 4. The calibration procedure

The rule was fixed **before** the measurement was taken:

```text
lambda_rotation = 0.25 * mean(translation term) / mean(rotation term)
```

so that the rotation term contributes a quarter of what translation does — the emphasis
Step 8 declared for rotation and could never realise on a degenerate target.

Two properties keep it honest. It ran on `train` **only**; no test split was read to produce
it, and none was read until the weight was frozen. And it ran with the rotation term's weight
set to **zero**, so the term was measured rather than optimised — a calibration that let the
term be trained by the weight it was choosing would be circular. The terms are read from the
trainer's own `_frame_loss_parts`, so what was calibrated is exactly what is optimised.

Measured over 50 training steps for each of the five arms at seed 0:

| arm | translation | scale | rotation (chordal) |
| --- | ---: | ---: | ---: |
| A1 | 0.3976 | 1.8064 | 0.3037 |
| A1M | 0.4017 | 1.7716 | 0.2960 |
| A3Lite | 0.3948 | 1.7648 | 0.3013 |
| A3L | 0.3983 | 1.7813 | 0.3022 |
| A3 | 0.4054 | 1.7745 | 0.3070 |
| **mean** | **0.3996** | **1.7797** | **0.3020** |

`lambda_rotation = 0.25 × 0.3996 / 0.3020 = 0.3307`, **frozen at 0.33** (two significant
figures). Recorded in `rotation_weight_calibration.json` and in every run's manifest; the
runner refuses to start if a run's weight is not this one.

---

## 5. The final weights, and a limitation to read with them

| term | weight | share of the frame loss at calibration |
| --- | ---: | ---: |
| translation | 1.0 | 29% |
| scale | 0.5 | 64% |
| rotation | 0.33 | **7%** |

**Rotation is 7% of the frame loss.** That follows from the pre-registered rule plus Step 8's
inherited scale weight, which turns out to dominate because the scale term is an L1 sum over
three log axes. The rule was applied as written rather than retuned after seeing this, which
is the point of pre-registering it.

The consequence must be read with every rotation result below: **if rotation learning is weak,
under-weighting is a live alternative explanation to an architectural one.** The experiment
cannot separate those two, and §21 does not pretend otherwise. A weight sweep is the obvious
follow-up and is named as such rather than run here, because running it now would mean
selecting a weight against test results.

---

## 6. The arm matrix

Five arms — A1, A1M, A3Lite, A3L, A3 — at Step 9's protocol (1200 steps, batch 8, CPU,
4 torch threads), three seeds each, no new architectures, no hyperparameter selection.

| cell | target | convention | role |
| --- | --- | --- | --- |
| `T0_global` | global | — | **primary**: the Change 2 question |
| `T4_rigid` | parent-relative | rigid | **pre-registered control** (§10) |
| `T1_spatial` | parent-relative | isotropic | **secondary replication** (§12) |

45 runs. The decision rule is Change 1's, unchanged and fixed before results: a difference
exceeds seed variability only if all three seeds agree in sign **and** |t| > 4.303.

---

## 7. T4_rigid, and what it controls for

Under `rigid` a parent contributes its rotation but **not** its scale. It therefore cannot
propagate a scale error into its descendants, while still orienting them.

On the Change 1 corpus that was close to a no-op: with every rotation the identity, a parent
had no orientation to pass, so `rigid` reduced to translation-only reparenting. Change 2 is
the first experiment in which the cell means anything, which is why it is pre-registered here
and was excluded there.

It is a **control**, not a candidate. It is not used to tune the primary cell, and no result
below is selected on it.

---

## 8. Results

Every number is generated by `experiments/step10/change2_training_tables.py` from the run
artifacts. None is typed by hand.

<!-- BEGIN GENERATED TABLES -->
<!-- generated by experiments.step10.change2_training_tables -->

### Per seed — test_seen

Split `test_seen`, inferred placement. Margin = floor − value: **positive beats the floor**. Floors: position 0.1605, rotation 22.65°, scale 0.1120.

| arm | cell | seed | position | rotation° | scale | position margin | rotation margin° | scale margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 0 | 0.1678 | 23.63 | 0.1126 | -0.0073 | -0.98 | -0.0006 |
| A1 | T0_global | 1 | 0.1672 | 23.90 | 0.1126 | -0.0066 | -1.25 | -0.0006 |
| A1 | T0_global | 2 | 0.1672 | 24.01 | 0.1128 | -0.0067 | -1.36 | -0.0008 |
| A1 | T1_spatial | 0 | 0.1789 | 27.67 | 0.1251 | -0.0183 | -5.02 | -0.0130 |
| A1 | T1_spatial | 1 | 0.1778 | 27.35 | 0.1251 | -0.0172 | -4.70 | -0.0130 |
| A1 | T1_spatial | 2 | 0.1776 | 27.19 | 0.1250 | -0.0170 | -4.54 | -0.0130 |
| A1 | T4_rigid | 0 | 0.1715 | 26.68 | 0.1127 | -0.0110 | -4.02 | -0.0007 |
| A1 | T4_rigid | 1 | 0.1724 | 26.76 | 0.1128 | -0.0119 | -4.11 | -0.0007 |
| A1 | T4_rigid | 2 | 0.1722 | 26.67 | 0.1128 | -0.0116 | -4.01 | -0.0008 |
| A1M | T0_global | 0 | 0.1671 | 23.48 | 0.1120 | -0.0065 | -0.83 | +0.0001 |
| A1M | T0_global | 1 | 0.1687 | 23.67 | 0.1124 | -0.0082 | -1.02 | -0.0003 |
| A1M | T0_global | 2 | 0.1664 | 23.66 | 0.1124 | -0.0059 | -1.01 | -0.0004 |
| A1M | T1_spatial | 0 | 0.1706 | 27.39 | 0.1247 | -0.0101 | -4.74 | -0.0126 |
| A1M | T1_spatial | 1 | 0.1766 | 27.58 | 0.1249 | -0.0161 | -4.93 | -0.0129 |
| A1M | T1_spatial | 2 | 0.1759 | 27.55 | 0.1247 | -0.0154 | -4.89 | -0.0127 |
| A1M | T4_rigid | 0 | 0.1710 | 26.47 | 0.1123 | -0.0105 | -3.82 | -0.0003 |
| A1M | T4_rigid | 1 | 0.1716 | 26.55 | 0.1125 | -0.0110 | -3.90 | -0.0005 |
| A1M | T4_rigid | 2 | 0.1711 | 26.75 | 0.1128 | -0.0105 | -4.10 | -0.0008 |
| A3 | T0_global | 0 | 0.1556 | 23.63 | 0.1106 | +0.0049 | -0.98 | +0.0014 |
| A3 | T0_global | 1 | 0.1560 | 23.82 | 0.1107 | +0.0045 | -1.17 | +0.0014 |
| A3 | T0_global | 2 | 0.1572 | 23.80 | 0.1109 | +0.0033 | -1.15 | +0.0011 |
| A3 | T1_spatial | 0 | 0.1723 | 28.28 | 0.1234 | -0.0117 | -5.62 | -0.0114 |
| A3 | T1_spatial | 1 | 0.1669 | 27.99 | 0.1234 | -0.0064 | -5.34 | -0.0113 |
| A3 | T1_spatial | 2 | 0.1689 | 27.69 | 0.1235 | -0.0084 | -5.03 | -0.0115 |
| A3 | T4_rigid | 0 | 0.1593 | 27.25 | 0.1107 | +0.0012 | -4.60 | +0.0013 |
| A3 | T4_rigid | 1 | 0.1625 | 27.29 | 0.1109 | -0.0020 | -4.64 | +0.0011 |
| A3 | T4_rigid | 2 | 0.1542 | 27.19 | 0.1109 | +0.0063 | -4.53 | +0.0011 |
| A3L | T0_global | 0 | 0.1651 | 23.80 | 0.1111 | -0.0046 | -1.15 | +0.0010 |
| A3L | T0_global | 1 | 0.1633 | 23.54 | 0.1111 | -0.0028 | -0.89 | +0.0010 |
| A3L | T0_global | 2 | 0.1630 | 23.47 | 0.1113 | -0.0025 | -0.82 | +0.0007 |
| A3L | T1_spatial | 0 | 0.1682 | 28.23 | 0.1239 | -0.0077 | -5.58 | -0.0119 |
| A3L | T1_spatial | 1 | 0.1684 | 27.74 | 0.1236 | -0.0079 | -5.09 | -0.0116 |
| A3L | T1_spatial | 2 | 0.1704 | 27.53 | 0.1237 | -0.0099 | -4.88 | -0.0117 |
| A3L | T4_rigid | 0 | 0.1651 | 27.52 | 0.1111 | -0.0045 | -4.87 | +0.0009 |
| A3L | T4_rigid | 1 | 0.1641 | 27.02 | 0.1114 | -0.0035 | -4.37 | +0.0006 |
| A3L | T4_rigid | 2 | 0.1652 | 27.28 | 0.1114 | -0.0047 | -4.63 | +0.0006 |
| A3Lite | T0_global | 0 | 0.1611 | 23.69 | 0.1107 | -0.0006 | -1.04 | +0.0014 |
| A3Lite | T0_global | 1 | 0.1633 | 23.81 | 0.1105 | -0.0028 | -1.16 | +0.0015 |
| A3Lite | T0_global | 2 | 0.1536 | 23.71 | 0.1108 | +0.0069 | -1.06 | +0.0013 |
| A3Lite | T1_spatial | 0 | 0.1669 | 27.68 | 0.1235 | -0.0064 | -5.03 | -0.0115 |
| A3Lite | T1_spatial | 1 | 0.1699 | 27.86 | 0.1234 | -0.0093 | -5.21 | -0.0114 |
| A3Lite | T1_spatial | 2 | 0.1602 | 27.59 | 0.1235 | +0.0003 | -4.94 | -0.0114 |
| A3Lite | T4_rigid | 0 | 0.1559 | 26.91 | 0.1110 | +0.0046 | -4.26 | +0.0010 |
| A3Lite | T4_rigid | 1 | 0.1639 | 27.06 | 0.1108 | -0.0034 | -4.41 | +0.0012 |
| A3Lite | T4_rigid | 2 | 0.1542 | 27.17 | 0.1110 | +0.0063 | -4.52 | +0.0010 |

### Aggregated by arm and cell

Split `test_seen`, three seeds each. Floors: position 0.1605, rotation 22.65°, scale 0.1120.

| arm | cell | role | mean position | position std | mean rotation° | rotation std | mean scale | scale std | beats position floor | beats rotation floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | primary | 0.1674 | 0.0004 | 23.85 | 0.19 | 0.1127 | 0.0001 | no | no |
| A1 | T4_rigid | pre-registered control | 0.1720 | 0.0005 | 26.70 | 0.05 | 0.1128 | 0.0001 | no | no |
| A1 | T1_spatial | secondary replication | 0.1781 | 0.0007 | 27.40 | 0.24 | 0.1250 | 0.0000 | no | no |
| A1M | T0_global | primary | 0.1674 | 0.0012 | 23.60 | 0.11 | 0.1122 | 0.0002 | no | no |
| A1M | T4_rigid | pre-registered control | 0.1712 | 0.0003 | 26.59 | 0.14 | 0.1125 | 0.0002 | no | no |
| A1M | T1_spatial | secondary replication | 0.1744 | 0.0033 | 27.51 | 0.10 | 0.1248 | 0.0001 | no | no |
| A3Lite | T0_global | primary | 0.1594 | 0.0051 | 23.74 | 0.06 | 0.1106 | 0.0001 | yes | no |
| A3Lite | T4_rigid | pre-registered control | 0.1580 | 0.0052 | 27.05 | 0.13 | 0.1109 | 0.0001 | yes | no |
| A3Lite | T1_spatial | secondary replication | 0.1657 | 0.0049 | 27.71 | 0.14 | 0.1235 | 0.0001 | no | no |
| A3L | T0_global | primary | 0.1638 | 0.0011 | 23.61 | 0.17 | 0.1111 | 0.0001 | no | no |
| A3L | T4_rigid | pre-registered control | 0.1648 | 0.0006 | 27.27 | 0.25 | 0.1113 | 0.0002 | no | no |
| A3L | T1_spatial | secondary replication | 0.1690 | 0.0012 | 27.83 | 0.36 | 0.1237 | 0.0002 | no | no |
| A3 | T0_global | primary | 0.1563 | 0.0008 | 23.75 | 0.10 | 0.1107 | 0.0002 | yes | no |
| A3 | T4_rigid | pre-registered control | 0.1587 | 0.0042 | 27.24 | 0.06 | 0.1108 | 0.0001 | yes | no |
| A3 | T1_spatial | secondary replication | 0.1694 | 0.0027 | 27.98 | 0.29 | 0.1234 | 0.0001 | no | no |

### Rotation in detail

Split `test_seen`, seed mean of each statistic. `below floor` is the share of individual entities under the 22.65° floor — a model can sit on the floor on average while being better on most entities and far worse on a few.

| arm | cell | mean° | median° | std° | within 5° | within 10° | within 20° | below floor | floor margin° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 23.85 | 20.08 | 22.03 | 0.05 | 0.23 | 0.50 | 0.59 | -1.20 |
| A1 | T4_rigid | 26.70 | 20.67 | 26.79 | 0.04 | 0.21 | 0.48 | 0.57 | -4.05 |
| A1 | T1_spatial | 27.40 | 21.86 | 26.96 | 0.03 | 0.17 | 0.43 | 0.53 | -4.75 |
| A1M | T0_global | 23.60 | 19.87 | 22.87 | 0.04 | 0.23 | 0.50 | 0.60 | -0.95 |
| A1M | T4_rigid | 26.59 | 20.50 | 27.13 | 0.04 | 0.22 | 0.48 | 0.57 | -3.94 |
| A1M | T1_spatial | 27.51 | 21.72 | 27.32 | 0.03 | 0.17 | 0.44 | 0.53 | -4.85 |
| A3Lite | T0_global | 23.74 | 20.30 | 22.84 | 0.04 | 0.23 | 0.49 | 0.59 | -1.09 |
| A3Lite | T4_rigid | 27.05 | 20.86 | 27.02 | 0.04 | 0.21 | 0.47 | 0.56 | -4.40 |
| A3Lite | T1_spatial | 27.71 | 21.47 | 28.21 | 0.03 | 0.16 | 0.45 | 0.54 | -5.06 |
| A3L | T0_global | 23.61 | 20.05 | 21.65 | 0.04 | 0.23 | 0.50 | 0.59 | -0.95 |
| A3L | T4_rigid | 27.27 | 20.80 | 27.63 | 0.04 | 0.20 | 0.47 | 0.56 | -4.62 |
| A3L | T1_spatial | 27.83 | 21.55 | 27.84 | 0.03 | 0.17 | 0.45 | 0.54 | -5.18 |
| A3 | T0_global | 23.75 | 20.37 | 20.56 | 0.04 | 0.22 | 0.49 | 0.59 | -1.10 |
| A3 | T4_rigid | 27.24 | 20.99 | 26.81 | 0.04 | 0.19 | 0.47 | 0.55 | -4.59 |
| A3 | T1_spatial | 27.98 | 21.86 | 27.96 | 0.03 | 0.16 | 0.44 | 0.52 | -5.33 |

### Aggregated — test_arrangement

Split `test_arrangement`, three seeds each. Floors: position 0.1655, rotation 22.54°, scale 0.1199.

| arm | cell | role | mean position | position std | mean rotation° | rotation std | mean scale | scale std | beats position floor | beats rotation floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | primary | 0.1742 | 0.0014 | 23.50 | 0.17 | 0.1167 | 0.0002 | no | no |
| A1 | T4_rigid | pre-registered control | 0.1779 | 0.0005 | 26.21 | 0.04 | 0.1168 | 0.0001 | no | no |
| A1 | T1_spatial | secondary replication | 0.1847 | 0.0012 | 26.88 | 0.25 | 0.1295 | 0.0003 | no | no |
| A1M | T0_global | primary | 0.1730 | 0.0004 | 23.20 | 0.06 | 0.1166 | 0.0000 | no | no |
| A1M | T4_rigid | pre-registered control | 0.1766 | 0.0005 | 26.07 | 0.17 | 0.1167 | 0.0001 | no | no |
| A1M | T1_spatial | secondary replication | 0.1813 | 0.0032 | 26.95 | 0.11 | 0.1293 | 0.0002 | no | no |
| A3Lite | T0_global | primary | 0.1654 | 0.0043 | 23.32 | 0.09 | 0.1147 | 0.0000 | yes | no |
| A3Lite | T4_rigid | pre-registered control | 0.1627 | 0.0049 | 26.55 | 0.08 | 0.1148 | 0.0001 | yes | no |
| A3Lite | T1_spatial | secondary replication | 0.1713 | 0.0059 | 27.16 | 0.17 | 0.1278 | 0.0002 | no | no |
| A3L | T0_global | primary | 0.1690 | 0.0027 | 23.26 | 0.18 | 0.1149 | 0.0001 | no | no |
| A3L | T4_rigid | pre-registered control | 0.1689 | 0.0015 | 26.73 | 0.23 | 0.1149 | 0.0003 | no | no |
| A3L | T1_spatial | secondary replication | 0.1751 | 0.0010 | 27.31 | 0.36 | 0.1277 | 0.0001 | no | no |
| A3 | T0_global | primary | 0.1611 | 0.0012 | 23.46 | 0.16 | 0.1148 | 0.0001 | yes | no |
| A3 | T4_rigid | pre-registered control | 0.1626 | 0.0039 | 26.75 | 0.10 | 0.1148 | 0.0001 | yes | no |
| A3 | T1_spatial | secondary replication | 0.1752 | 0.0032 | 27.44 | 0.27 | 0.1275 | 0.0001 | no | no |

### Aggregated — test_transform

Split `test_transform`, three seeds each. Floors: position 0.1703, rotation 35.95°, scale 0.1248.

| arm | cell | role | mean position | position std | mean rotation° | rotation std | mean scale | scale std | beats position floor | beats rotation floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | primary | 0.1789 | 0.0015 | 36.29 | 0.15 | 0.1229 | 0.0002 | no | no |
| A1 | T4_rigid | pre-registered control | 0.1816 | 0.0006 | 38.43 | 0.06 | 0.1230 | 0.0002 | no | no |
| A1 | T1_spatial | secondary replication | 0.1887 | 0.0015 | 38.71 | 0.11 | 0.1343 | 0.0002 | no | no |
| A1M | T0_global | primary | 0.1776 | 0.0006 | 36.03 | 0.21 | 0.1225 | 0.0001 | no | no |
| A1M | T4_rigid | pre-registered control | 0.1799 | 0.0007 | 38.43 | 0.12 | 0.1227 | 0.0002 | no | no |
| A1M | T1_spatial | secondary replication | 0.1846 | 0.0037 | 38.78 | 0.16 | 0.1344 | 0.0002 | no | no |
| A3Lite | T0_global | primary | 0.1758 | 0.0006 | 36.06 | 0.18 | 0.1208 | 0.0001 | no | no |
| A3Lite | T4_rigid | pre-registered control | 0.1798 | 0.0005 | 38.78 | 0.26 | 0.1210 | 0.0001 | no | no |
| A3Lite | T1_spatial | secondary replication | 0.1804 | 0.0022 | 38.93 | 0.19 | 0.1333 | 0.0002 | no | no |
| A3L | T0_global | primary | 0.1763 | 0.0016 | 36.07 | 0.21 | 0.1213 | 0.0002 | no | no |
| A3L | T4_rigid | pre-registered control | 0.1783 | 0.0003 | 38.87 | 0.20 | 0.1213 | 0.0002 | no | no |
| A3L | T1_spatial | secondary replication | 0.1814 | 0.0009 | 39.00 | 0.26 | 0.1333 | 0.0003 | no | no |
| A3 | T0_global | primary | 0.1747 | 0.0003 | 36.15 | 0.21 | 0.1209 | 0.0003 | no | no |
| A3 | T4_rigid | pre-registered control | 0.1783 | 0.0008 | 38.86 | 0.13 | 0.1210 | 0.0001 | no | no |
| A3 | T1_spatial | secondary replication | 0.1823 | 0.0032 | 39.07 | 0.33 | 0.1331 | 0.0001 | no | no |

### Aggregated — test_combination

Split `test_combination`, three seeds each. Floors: position 0.1750, rotation 26.10°, scale 0.1231.

| arm | cell | role | mean position | position std | mean rotation° | rotation std | mean scale | scale std | beats position floor | beats rotation floor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | primary | 0.1892 | 0.0012 | 27.94 | 0.69 | 0.1223 | 0.0002 | no | no |
| A1 | T4_rigid | pre-registered control | 0.1894 | 0.0008 | 31.27 | 0.16 | 0.1224 | 0.0002 | no | no |
| A1 | T1_spatial | secondary replication | 0.1913 | 0.0030 | 32.26 | 0.11 | 0.1311 | 0.0001 | no | no |
| A1M | T0_global | primary | 0.1890 | 0.0013 | 28.15 | 0.58 | 0.1220 | 0.0002 | no | no |
| A1M | T4_rigid | pre-registered control | 0.1879 | 0.0014 | 31.39 | 0.25 | 0.1222 | 0.0002 | no | no |
| A1M | T1_spatial | secondary replication | 0.1858 | 0.0042 | 32.47 | 0.41 | 0.1307 | 0.0000 | no | no |
| A3Lite | T0_global | primary | 0.2063 | 0.0206 | 27.56 | 0.44 | 0.1198 | 0.0002 | no | no |
| A3Lite | T4_rigid | pre-registered control | 0.2550 | 0.0333 | 32.37 | 0.42 | 0.1201 | 0.0001 | no | no |
| A3Lite | T1_spatial | secondary replication | 0.2141 | 0.0309 | 33.09 | 0.63 | 0.1296 | 0.0001 | no | no |
| A3L | T0_global | primary | 0.2030 | 0.0091 | 27.43 | 0.16 | 0.1206 | 0.0002 | no | no |
| A3L | T4_rigid | pre-registered control | 0.2107 | 0.0064 | 32.70 | 0.49 | 0.1207 | 0.0000 | no | no |
| A3L | T1_spatial | secondary replication | 0.1926 | 0.0062 | 32.56 | 0.39 | 0.1295 | 0.0002 | no | no |
| A3 | T0_global | primary | 0.2190 | 0.0081 | 26.72 | 0.01 | 0.1201 | 0.0001 | no | no |
| A3 | T4_rigid | pre-registered control | 0.2518 | 0.0216 | 32.41 | 0.01 | 0.1203 | 0.0005 | no | no |
| A3 | T1_spatial | secondary replication | 0.1977 | 0.0027 | 33.00 | 0.39 | 0.1296 | 0.0001 | no | no |

### Depth analysis

Split `test_seen`, seed mean. Depth is read from the spatial tree for every cell, so depth 0 names the same ten root entities everywhere. Rotation is in degrees.

| arm | cell | depth | entities | position | rotation° | scale |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 0 | 1000 | 0.1462 | 14.75 | 0.1120 |
| A1 | T0_global | 1 | 850 | 0.1999 | 22.51 | 0.1289 |
| A1 | T0_global | 2 | 300 | 0.1078 | 26.83 | 0.0805 |
| A1 | T0_global | 3 | 150 | 0.1698 | 86.14 | 0.0800 |
| A1 | T4_rigid | 0 | 1000 | 0.1441 | 15.77 | 0.1123 |
| A1 | T4_rigid | 1 | 850 | 0.2074 | 27.57 | 0.1288 |
| A1 | T4_rigid | 2 | 300 | 0.1225 | 30.51 | 0.0803 |
| A1 | T4_rigid | 3 | 150 | 0.1748 | 86.99 | 0.0806 |
| A1 | T1_spatial | 0 | 1000 | 0.1453 | 18.18 | 0.1148 |
| A1 | T1_spatial | 1 | 850 | 0.2263 | 26.71 | 0.1384 |
| A1 | T1_spatial | 2 | 300 | 0.1138 | 29.59 | 0.1181 |
| A1 | T1_spatial | 3 | 150 | 0.1705 | 88.40 | 0.1022 |
| A1M | T0_global | 0 | 1000 | 0.1459 | 14.80 | 0.1119 |
| A1M | T0_global | 1 | 850 | 0.2000 | 22.05 | 0.1283 |
| A1M | T0_global | 2 | 300 | 0.1076 | 26.50 | 0.0801 |
| A1M | T0_global | 3 | 150 | 0.1695 | 85.31 | 0.0789 |
| A1M | T4_rigid | 0 | 1000 | 0.1435 | 15.60 | 0.1124 |
| A1M | T4_rigid | 1 | 850 | 0.2074 | 27.97 | 0.1284 |
| A1M | T4_rigid | 2 | 300 | 0.1196 | 29.33 | 0.0803 |
| A1M | T4_rigid | 3 | 150 | 0.1732 | 86.61 | 0.0797 |
| A1M | T1_spatial | 0 | 1000 | 0.1454 | 18.19 | 0.1146 |
| A1M | T1_spatial | 1 | 850 | 0.2172 | 26.91 | 0.1380 |
| A1M | T1_spatial | 2 | 300 | 0.1138 | 29.89 | 0.1177 |
| A1M | T1_spatial | 3 | 150 | 0.1695 | 88.25 | 0.1023 |
| A3Lite | T0_global | 0 | 1000 | 0.1358 | 14.75 | 0.1081 |
| A3Lite | T0_global | 1 | 850 | 0.1944 | 21.69 | 0.1278 |
| A3Lite | T0_global | 2 | 300 | 0.0956 | 28.47 | 0.0791 |
| A3Lite | T0_global | 3 | 150 | 0.1699 | 85.77 | 0.0796 |
| A3Lite | T4_rigid | 0 | 1000 | 0.1299 | 16.09 | 0.1086 |
| A3Lite | T4_rigid | 1 | 850 | 0.1903 | 28.28 | 0.1280 |
| A3Lite | T4_rigid | 2 | 300 | 0.1195 | 29.97 | 0.0796 |
| A3Lite | T4_rigid | 3 | 150 | 0.1575 | 87.31 | 0.0794 |
| A3Lite | T1_spatial | 0 | 1000 | 0.1370 | 17.62 | 0.1110 |
| A3Lite | T1_spatial | 1 | 850 | 0.2093 | 27.43 | 0.1379 |
| A3Lite | T1_spatial | 2 | 300 | 0.1096 | 30.04 | 0.1184 |
| A3Lite | T1_spatial | 3 | 150 | 0.1507 | 91.94 | 0.1029 |
| A3L | T0_global | 0 | 1000 | 0.1397 | 14.77 | 0.1090 |
| A3L | T0_global | 1 | 850 | 0.1969 | 21.92 | 0.1279 |
| A3L | T0_global | 2 | 300 | 0.1082 | 26.83 | 0.0804 |
| A3L | T0_global | 3 | 150 | 0.1699 | 85.59 | 0.0799 |
| A3L | T4_rigid | 0 | 1000 | 0.1360 | 15.99 | 0.1093 |
| A3L | T4_rigid | 1 | 850 | 0.2002 | 28.51 | 0.1280 |
| A3L | T4_rigid | 2 | 300 | 0.1212 | 30.38 | 0.0803 |
| A3L | T4_rigid | 3 | 150 | 0.1566 | 89.33 | 0.0795 |
| A3L | T1_spatial | 0 | 1000 | 0.1416 | 17.68 | 0.1119 |
| A3L | T1_spatial | 1 | 850 | 0.2126 | 27.55 | 0.1376 |
| A3L | T1_spatial | 2 | 300 | 0.1089 | 30.98 | 0.1183 |
| A3L | T1_spatial | 3 | 150 | 0.1520 | 90.83 | 0.1025 |
| A3 | T0_global | 0 | 1000 | 0.1302 | 14.83 | 0.1084 |
| A3 | T0_global | 1 | 850 | 0.1923 | 21.76 | 0.1277 |
| A3 | T0_global | 2 | 300 | 0.0962 | 28.56 | 0.0792 |
| A3 | T0_global | 3 | 150 | 0.1682 | 84.87 | 0.0790 |
| A3 | T4_rigid | 0 | 1000 | 0.1304 | 16.30 | 0.1083 |
| A3 | T4_rigid | 1 | 850 | 0.1905 | 28.54 | 0.1280 |
| A3 | T4_rigid | 2 | 300 | 0.1200 | 30.12 | 0.0796 |
| A3 | T4_rigid | 3 | 150 | 0.1564 | 87.18 | 0.0791 |
| A3 | T1_spatial | 0 | 1000 | 0.1401 | 18.02 | 0.1111 |
| A3 | T1_spatial | 1 | 850 | 0.2155 | 27.55 | 0.1382 |
| A3 | T1_spatial | 2 | 300 | 0.1080 | 30.41 | 0.1175 |
| A3 | T1_spatial | 3 | 150 | 0.1508 | 91.97 | 0.1026 |

### T4_rigid against the global cell

Split `test_seen`: `T4_rigid` − `T0_global`, paired by seed. **Negative means the treatment was better.** Rule fixed before results: every seed agrees in sign AND |t| > 4.303.

| arm | component | seed 0 | seed 1 | seed 2 | mean | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | position | +0.0037 | +0.0053 | +0.0050 | +0.0046 | +9.67 | yes | YES |
| A1 | rotation° | +3.04 | +2.86 | +2.65 | +2.85 | +25.31 | yes | YES |
| A1 | scale | +0.0001 | +0.0001 | +0.0000 | +0.0001 | +2.02 | yes | no |
| A1M | position | +0.0040 | +0.0028 | +0.0047 | +0.0038 | +7.20 | yes | YES |
| A1M | rotation° | +2.99 | +2.88 | +3.09 | +2.99 | +50.15 | yes | YES |
| A1M | scale | +0.0003 | +0.0001 | +0.0004 | +0.0003 | +4.04 | yes | no |
| A3Lite | position | -0.0052 | +0.0006 | +0.0006 | -0.0014 | -0.70 | no | no |
| A3Lite | rotation° | +3.22 | +3.25 | +3.46 | +3.31 | +43.99 | yes | YES |
| A3Lite | scale | +0.0004 | +0.0003 | +0.0003 | +0.0003 | +7.77 | yes | YES |
| A3L | position | -0.0000 | +0.0008 | +0.0022 | +0.0010 | +1.50 | no | no |
| A3L | rotation° | +3.72 | +3.47 | +3.81 | +3.67 | +36.57 | yes | YES |
| A3L | scale | +0.0000 | +0.0003 | +0.0001 | +0.0002 | +1.72 | yes | no |
| A3 | position | +0.0037 | +0.0065 | -0.0030 | +0.0024 | +0.85 | no | no |
| A3 | rotation° | +3.63 | +3.47 | +3.39 | +3.50 | +50.09 | yes | YES |
| A3 | scale | +0.0001 | +0.0002 | +0.0000 | +0.0001 | +1.91 | yes | no |

### T1_spatial against the global cell (replication of Change 1)

Split `test_seen`: `T1_spatial` − `T0_global`, paired by seed. **Negative means the treatment was better.** Rule fixed before results: every seed agrees in sign AND |t| > 4.303.

| arm | component | seed 0 | seed 1 | seed 2 | mean | t | same sign | exceeds seed variability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | position | +0.0110 | +0.0106 | +0.0104 | +0.0107 | +54.72 | yes | YES |
| A1 | rotation° | +4.03 | +3.45 | +3.18 | +3.55 | +14.09 | yes | YES |
| A1 | scale | +0.0124 | +0.0125 | +0.0122 | +0.0124 | +140.21 | yes | YES |
| A1M | position | +0.0035 | +0.0079 | +0.0095 | +0.0070 | +3.92 | yes | no |
| A1M | rotation° | +3.91 | +3.92 | +3.88 | +3.90 | +397.25 | yes | YES |
| A1M | scale | +0.0127 | +0.0126 | +0.0123 | +0.0125 | +109.34 | yes | YES |
| A3Lite | position | +0.0058 | +0.0066 | +0.0066 | +0.0063 | +22.78 | yes | YES |
| A3Lite | rotation° | +3.99 | +4.06 | +3.88 | +3.97 | +77.45 | yes | YES |
| A3Lite | scale | +0.0129 | +0.0129 | +0.0127 | +0.0128 | +231.23 | yes | YES |
| A3L | position | +0.0031 | +0.0051 | +0.0074 | +0.0052 | +4.25 | yes | no |
| A3L | rotation° | +4.43 | +4.19 | +4.06 | +4.23 | +39.43 | yes | YES |
| A3L | scale | +0.0129 | +0.0125 | +0.0124 | +0.0126 | +88.07 | yes | YES |
| A3 | position | +0.0166 | +0.0109 | +0.0117 | +0.0131 | +7.24 | yes | YES |
| A3 | rotation° | +4.65 | +4.17 | +3.89 | +4.23 | +19.12 | yes | YES |
| A3 | scale | +0.0128 | +0.0127 | +0.0126 | +0.0127 | +251.12 | yes | YES |

### Parent-rotation error propagation

Split `test_seen`, seed mean, degrees. `T0_global` composes nothing, so its correlation is the shared-representation baseline; anything above it in the parent-relative cells is what composition added.

| cell | arm | depth | parent rotation° | child rotation° | correlation |
| --- | --- | --- | --- | --- | --- |
| T0_global | A1M | 1 | 14.91 | 22.05 | +0.43 |
| T0_global | A1M | 2 | 24.04 | 26.50 | +0.53 |
| T0_global | A1M | 3 | 24.01 | 85.31 | -0.55 |
| T0_global | A1 | 1 | 14.91 | 22.51 | +0.42 |
| T0_global | A1 | 2 | 24.55 | 26.83 | +0.52 |
| T0_global | A1 | 3 | 23.95 | 86.14 | -0.56 |
| T0_global | A3L | 1 | 14.94 | 21.92 | +0.44 |
| T0_global | A3L | 2 | 23.27 | 26.83 | +0.52 |
| T0_global | A3L | 3 | 24.01 | 85.59 | -0.55 |
| T0_global | A3Lite | 1 | 14.91 | 21.69 | +0.44 |
| T0_global | A3Lite | 2 | 23.01 | 28.47 | +0.51 |
| T0_global | A3Lite | 3 | 24.89 | 85.77 | -0.60 |
| T0_global | A3 | 1 | 15.04 | 21.76 | +0.44 |
| T0_global | A3 | 2 | 23.22 | 28.56 | +0.49 |
| T0_global | A3 | 3 | 25.10 | 84.87 | -0.52 |
| T1_spatial | A1M | 1 | 19.75 | 26.91 | +0.39 |
| T1_spatial | A1M | 2 | 41.20 | 29.89 | +0.35 |
| T1_spatial | A1M | 3 | 24.21 | 88.25 | -0.58 |
| T1_spatial | A1 | 1 | 19.81 | 26.71 | +0.37 |
| T1_spatial | A1 | 2 | 40.67 | 29.59 | +0.36 |
| T1_spatial | A1 | 3 | 24.21 | 88.40 | -0.58 |
| T1_spatial | A3L | 1 | 19.11 | 27.55 | +0.34 |
| T1_spatial | A3L | 2 | 42.72 | 30.98 | +0.31 |
| T1_spatial | A3L | 3 | 25.48 | 90.83 | -0.49 |
| T1_spatial | A3Lite | 1 | 19.01 | 27.43 | +0.34 |
| T1_spatial | A3Lite | 2 | 42.43 | 30.04 | +0.32 |
| T1_spatial | A3Lite | 3 | 24.51 | 91.94 | -0.48 |
| T1_spatial | A3 | 1 | 19.51 | 27.55 | +0.36 |
| T1_spatial | A3 | 2 | 42.76 | 30.41 | +0.31 |
| T1_spatial | A3 | 3 | 24.46 | 91.97 | -0.51 |
| T4_rigid | A1M | 1 | 16.13 | 27.97 | +0.29 |
| T4_rigid | A1M | 2 | 42.49 | 29.33 | +0.29 |
| T4_rigid | A1M | 3 | 24.74 | 86.61 | -0.56 |
| T4_rigid | A1 | 1 | 16.49 | 27.57 | +0.30 |
| T4_rigid | A1 | 2 | 41.82 | 30.51 | +0.33 |
| T4_rigid | A1 | 3 | 24.95 | 86.99 | -0.57 |
| T4_rigid | A3L | 1 | 16.81 | 28.51 | +0.30 |
| T4_rigid | A3L | 2 | 44.34 | 30.38 | +0.29 |
| T4_rigid | A3L | 3 | 25.03 | 89.33 | -0.45 |
| T4_rigid | A3Lite | 1 | 17.01 | 28.28 | +0.31 |
| T4_rigid | A3Lite | 2 | 44.35 | 29.97 | +0.27 |
| T4_rigid | A3Lite | 3 | 24.82 | 87.31 | -0.60 |
| T4_rigid | A3 | 1 | 17.28 | 28.54 | +0.32 |
| T4_rigid | A3 | 2 | 44.64 | 30.12 | +0.26 |
| T4_rigid | A3 | 3 | 24.78 | 87.18 | -0.64 |

### True-parent oracle

Split `test_seen`, seed mean. Parent-relative cells only — the global cell composes nothing, so there is no parent to replace. `lost to predicted parent` is as-trained minus oracle: what composing onto a predicted parent costs.

| cell | arm | depth | rotation, as trained° | rotation, true parent° | rotation lost° | position, as trained | position, true parent | position lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_spatial | A1M | 1 | 26.91 | 37.30 | -10.40 | 0.2172 | 0.1733 | +0.0439 |
| T1_spatial | A1M | 2 | 29.89 | 48.73 | -18.84 | 0.1138 | 0.2526 | -0.1388 |
| T1_spatial | A1M | 3 | 88.25 | 96.74 | -8.50 | 0.1695 | 0.1095 | +0.0600 |
| T1_spatial | A1 | 1 | 26.71 | 36.79 | -10.08 | 0.2263 | 0.1834 | +0.0429 |
| T1_spatial | A1 | 2 | 29.59 | 48.39 | -18.80 | 0.1138 | 0.2549 | -0.1411 |
| T1_spatial | A1 | 3 | 88.40 | 96.50 | -8.10 | 0.1705 | 0.1139 | +0.0566 |
| T1_spatial | A3L | 1 | 27.55 | 37.32 | -9.76 | 0.2126 | 0.1682 | +0.0444 |
| T1_spatial | A3L | 2 | 30.98 | 51.43 | -20.45 | 0.1089 | 0.2547 | -0.1458 |
| T1_spatial | A3L | 3 | 90.83 | 97.14 | -6.31 | 0.1520 | 0.1173 | +0.0346 |
| T1_spatial | A3Lite | 1 | 27.43 | 36.87 | -9.44 | 0.2093 | 0.1664 | +0.0428 |
| T1_spatial | A3Lite | 2 | 30.04 | 50.15 | -20.11 | 0.1096 | 0.2538 | -0.1443 |
| T1_spatial | A3Lite | 3 | 91.94 | 98.87 | -6.92 | 0.1507 | 0.1129 | +0.0378 |
| T1_spatial | A3 | 1 | 27.55 | 37.72 | -10.17 | 0.2155 | 0.1719 | +0.0436 |
| T1_spatial | A3 | 2 | 30.41 | 50.54 | -20.13 | 0.1080 | 0.2545 | -0.1465 |
| T1_spatial | A3 | 3 | 91.97 | 99.23 | -7.26 | 0.1508 | 0.1143 | +0.0365 |
| T4_rigid | A1M | 1 | 27.97 | 34.87 | -6.90 | 0.2074 | 0.1392 | +0.0682 |
| T4_rigid | A1M | 2 | 29.33 | 50.98 | -21.65 | 0.1196 | 0.2876 | -0.1680 |
| T4_rigid | A1M | 3 | 86.61 | 93.96 | -7.34 | 0.1732 | 0.1146 | +0.0587 |
| T4_rigid | A1 | 1 | 27.57 | 34.90 | -7.32 | 0.2074 | 0.1439 | +0.0635 |
| T4_rigid | A1 | 2 | 30.51 | 51.93 | -21.42 | 0.1225 | 0.2880 | -0.1655 |
| T4_rigid | A1 | 3 | 86.99 | 94.51 | -7.53 | 0.1748 | 0.1194 | +0.0553 |
| T4_rigid | A3L | 1 | 28.51 | 36.21 | -7.70 | 0.2002 | 0.1341 | +0.0661 |
| T4_rigid | A3L | 2 | 30.38 | 53.93 | -23.55 | 0.1212 | 0.2815 | -0.1602 |
| T4_rigid | A3L | 3 | 89.33 | 95.30 | -5.97 | 0.1566 | 0.1192 | +0.0374 |
| T4_rigid | A3Lite | 1 | 28.28 | 36.17 | -7.89 | 0.1903 | 0.1306 | +0.0597 |
| T4_rigid | A3Lite | 2 | 29.97 | 52.72 | -22.75 | 0.1195 | 0.2854 | -0.1660 |
| T4_rigid | A3Lite | 3 | 87.31 | 94.56 | -7.25 | 0.1575 | 0.1245 | +0.0330 |
| T4_rigid | A3 | 1 | 28.54 | 36.72 | -8.18 | 0.1905 | 0.1296 | +0.0608 |
| T4_rigid | A3 | 2 | 30.12 | 52.78 | -22.66 | 0.1200 | 0.2856 | -0.1656 |
| T4_rigid | A3 | 3 | 87.18 | 94.60 | -7.43 | 0.1564 | 0.1227 | +0.0337 |

### Compute and capacity

Objective: translation 1.0, scale 0.5, rotation 0.33 (chordal), all inside a frame term weighted 2.0. Every run used these; the runner refuses to start otherwise.

| arm | cell | parameters | steps | batch | threads | seeds | train s | eval s | depth s | rotation weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | T0_global | 559,751 | 1200 | 8 | 4 | 0,1,2 | 6088 | 115 | 18 | 0.33 |
| A1 | T4_rigid | 559,751 | 1200 | 8 | 4 | 0,1,2 | 1538 | 79 | 12 | 0.33 |
| A1 | T1_spatial | 559,751 | 1200 | 8 | 4 | 0,1,2 | 1743 | 99 | 16 | 0.33 |
| A1M | T0_global | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 1808 | 94 | 15 | 0.33 |
| A1M | T4_rigid | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 1776 | 126 | 19 | 0.33 |
| A1M | T1_spatial | 3,125,831 | 1200 | 8 | 4 | 0,1,2 | 2194 | 94 | 15 | 0.33 |
| A3Lite | T0_global | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 5100 | 118 | 17 | 0.33 |
| A3Lite | T4_rigid | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 1721 | 95 | 16 | 0.33 |
| A3Lite | T1_spatial | 1,252,615 | 1200 | 8 | 4 | 0,1,2 | 1746 | 101 | 16 | 0.33 |
| A3L | T0_global | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1446 | 83 | 14 | 0.33 |
| A3L | T4_rigid | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1524 | 84 | 16 | 0.33 |
| A3L | T1_spatial | 1,226,511 | 1200 | 8 | 4 | 0,1,2 | 1412 | 78 | 13 | 0.33 |
| A3 | T0_global | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1648 | 88 | 15 | 0.33 |
| A3 | T4_rigid | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1777 | 96 | 16 | 0.33 |
| A3 | T1_spatial | 3,226,647 | 1200 | 8 | 4 | 0,1,2 | 1780 | 97 | 14 | 0.33 |

### Integrity

| check | result |
| --- | --- |
| runs that passed the gate before training | 45 of 45 |
| corpus files verified against the frozen checksums, per run | {7} |
| split-level gate passes | 180 |
| worst round-trip error | 5.00e-16 |
| worst depth-vs-headline reconciliation | 0.00e+00 |

<!-- END GENERATED TABLES -->

---

## 9. Position results

**A3 clears the position floor on every seed; the no-graph arms clear it on none.** On
`test_seen`, `T0_global`: A3 0.1563 (floor 0.1605), A3Lite 0.1594, A3L 0.1638, A1 0.1674,
A1M 0.1674. The ordering is Step 9's and Change 1's, unchanged.

**Learning rotation cost nothing in position.** Change 1's A3 sat 0.0044 below the position
floor; Change 2's A3 sits 0.0042 below it. Read as margins over a floor that is numerically
identical on targets that are numerically identical, that is the same result — the model
acquired a real rotation objective without giving up any position accuracy. This is the one
cross-corpus statement this report makes, it is made as a margin rather than a raw score, and
it is confined to position, where the targets genuinely are the same numbers.

**The held-out splits are where position fails.** No arm beats the floor on `test_transform`
(best A3 0.1747 against 0.1703) or `test_combination`, and on the combination split the graph
arms are the *worst*: A3 0.2190 against a floor of 0.1750, A3Lite 0.2063, against A1's 0.1892.
That reproduces Step 8's compositional finding — the arms that read the relationship graph
fail hardest when a transformation pair is held out — and rotation has not changed it.

---

## 10. Rotation results

> **AMENDED 2026-09-22 — this section's conclusion is superseded.** A validation-only weight
> study afterwards showed the rotation term's weight was the binding constraint, not the
> architecture. At λ=3.0 instead of 0.33, A3 scores 20.96° against a validation floor of
> 23.51° — 2.55° *below* it — with position unchanged and still under its own floor.
>
> **Confirmed on the test splits since.** At λ=3.0, A3 scores 21.38° ± 1.03 against the
> 22.65° floor on `test_seen` — 1.27° below it — and 35.51° against 35.95° on
> `test_transform`, with position better than its own λ=0.33 result. The other four arms
> improve by about 1.1° and stop at the floor.
>
> Everything measured below stands **as measured at λ=0.33**, and the §5 limitation names
> exactly why that mattered. What no longer follows is any statement about the architecture's
> capacity to learn rotation. See `docs/STEP_10_ROTATION_WEIGHT_STUDY.md`.


**No arm, in any cell, on any split, beat the rotation floor.** That is the central result.

| split | best arm and cell | rotation | floor | margin |
| --- | --- | ---: | ---: | ---: |
| test_seen | A1M `T0_global` | 23.60° | 22.65° | **−0.95°** |
| test_arrangement | A1M `T0_global` | 23.20° | 22.54° | −0.66° |
| test_transform | A1M `T0_global` | 36.03° | 35.95° | −0.07° |
| test_combination | A3 `T0_global` | 26.72° | 26.10° | −0.62° |

**It is not a near miss that better seeds would close.** Rotation's across-seed standard
deviation is 0.05–0.36°, an order of magnitude smaller than the gap to the floor, and all
fifteen `T0_global` runs are above it.

**Nor is it an artefact of comparing a mean with a tail.** The model's error distribution is
skewed — median 19.9–20.4° against a mean of 23.6–23.9° — which at first looks like it might
beat a floor quoted as a mean. It does not: the floor's own median is **18.92°**, its share
within 10° is **28%** against the model's 22–23%, and within 20° **55%** against 49–50%. The
lookup wins on the mean, the median and every threshold. (`below floor` in the generated table
compares individual entities against the floor's *mean* and therefore flatters the model; it is
reported for completeness and should be read with this paragraph.)

**On `test_transform` the model does not extrapolate at all.** The floor rises to 35.95° there
because a lookup fitted on training yaw cannot reach the held-out band; every arm lands at
36.03–36.29°, which is the lookup's behaviour and nothing more.

**What the model did learn is the per-entity mean rotation.** That is what an identity-keyed
lookup is, which is why it ties with one. Each entity's rotation varies only 4–13° about its
own mean on this corpus, so the scene-specific part — the part that needs the arrangement, the
relations, or the hierarchy — is small, and none of it was captured.

---

## 11. Scale results

**The graph arms beat the scale floor; the no-graph arms do not.** On `test_seen`,
`T0_global`: A3Lite 0.1106, A3 0.1107, A3L 0.1111, all below the 0.1120 floor; A1M 0.1122 and
A1 0.1127 above it.

**Scale is where `T4_rigid` separates from `T1_spatial`, exactly as designed.** Against
`T0_global`, the isotropic parent costs +0.0124 to +0.0128 of scale error in all five arms,
every one passing the decision rule with t from 88 to 251. The rigid parent costs +0.0001 to
+0.0003, and passes the rule in only one arm. A parent that does not propagate its scale does
not propagate its scale error: the Change 1 scale-cascade tests, confirmed in trained models.

---

## 12. Placement-floor comparisons

Per split, `T0_global`, seed means, against this corpus's own position floor:

| arm | test_seen | test_arrangement | test_transform | test_combination |
| --- | --- | --- | --- | --- |
| A1 | −0.0069 | −0.0086 | −0.0086 | −0.0142 |
| A1M | −0.0069 | −0.0075 | −0.0073 | −0.0140 |
| A3Lite | **+0.0012** | **+0.0001** | −0.0056 | −0.0314 |
| A3L | −0.0033 | −0.0035 | −0.0060 | −0.0280 |
| A3 | **+0.0042** | **+0.0045** | −0.0044 | −0.0440 |

Positive beats the floor. Two arms clear it on `test_seen`, two on
`test_arrangement` (A3 clearly, A3Lite by 0.0001), none anywhere else, and the deficit on `test_combination` grows with how
much of the relationship graph an arm reads.

---

## 13. Rotation-floor comparisons

The same table for rotation, in degrees, positive meaning better than the floor:

| arm | test_seen | test_arrangement | test_transform | test_combination |
| --- | --- | --- | --- | --- |
| A1 | −1.20 | −0.96 | −0.34 | −1.83 |
| A1M | −0.95 | −0.66 | −0.07 | −2.05 |
| A3Lite | −1.09 | −0.78 | −0.11 | −1.46 |
| A3L | −0.95 | −0.72 | −0.11 | −1.33 |
| A3 | −1.10 | −0.92 | −0.20 | **−0.62** |

Every cell of every arm on every split is negative. The four splits' floors are not
interchangeable — 22.65°, 22.54°, 35.95° and 26.10° — and each column is read against its own.

The parent-relative cells are worse again: `T4_rigid` adds 2.85–3.67° of rotation error and
`T1_spatial` 3.56–4.23°, in all five arms, every one passing the decision rule (t from 14 to
397). **Composing a child's rotation onto a predicted parent is worse than predicting the
child's rotation directly**, which is Change 1's position finding reproduced on a component
Change 1 could not measure.

---

## 14. Depth analysis

Rotation error grows steeply with depth — A3 `T0_global`: 14.8° at depth 0, 21.8° at depth 1,
28.6° at depth 2, **84.9°** at depth 3.

**That growth is not propagation, and the control proves it.** `T0_global` composes nothing;
each entity's frame is predicted directly. The profile is therefore entity difficulty: the
corpus's depth-3 entity, the pulmonary arteries, has the largest and most variable rotation of
any entity (mean 117.9°, the only one above 150° at its maximum). Any experiment that read the
depth profile of a composing cell alone would have mistaken this for a cascade.

Measured against that control, composition adds a roughly constant offset rather than a
depth-amplified one: A3 `T4_rigid` runs 16.3° / 28.5° / 30.1° / 87.2° by depth against `T0_global`'s 14.8° /
21.8° / 28.6° / 84.9°. The largest excess is at depth 1, not depth 3.

Position and scale by depth are in §8's table and follow Change 1's pattern: depth 0 is stable
across cells, and the isotropic scale cascade appears at depth ≥ 1 while rigid's does not.

---

## 15. Parent-rotation error propagation

For every parented entity, its parent's rotation error was recorded beside its own (A3,
`test_seen`, seed means):

| cell | depth | parent rotation° | child rotation° | correlation |
| --- | --- | ---: | ---: | ---: |
| `T0_global` (composes nothing) | 1 | 15.04 | 21.76 | +0.44 |
| | 2 | 23.22 | 28.56 | +0.49 |
| | 3 | 25.10 | 84.87 | −0.52 |
| `T4_rigid` | 1 | 17.28 | 28.54 | +0.32 |
| | 2 | 44.64 | 30.12 | +0.26 |
| `T1_spatial` | 1 | 19.51 | 27.55 | +0.36 |
| | 2 | 42.76 | 30.41 | +0.31 |

**Composition did not increase the parent-child error correlation — it lowered it.** The
baseline, where nothing is composed, is +0.44 and +0.49; the composing cells sit at +0.26 to
+0.36. A simple propagation story predicts the opposite.

The explanation is in §16: the child's predicted local rotation is compensating for its
parent's error, which is exactly what would *reduce* the correlation between the two.

---

## 16. True-parent oracle

Each child's predicted local frame was recomposed onto its parent's **true** frame. The
expectation, from Change 1's position result, was that this would improve the child.

**It makes rotation worse, at every depth, in every parent-relative cell** (A3, `test_seen`):

| cell | depth | as trained° | with true parent° | change |
| --- | --- | ---: | ---: | ---: |
| `T4_rigid` | 1 | 28.54 | 36.72 | **+8.18 worse** |
| | 2 | 30.12 | 52.78 | **+22.66 worse** |
| | 3 | 87.18 | 94.60 | +7.43 worse |
| `T1_spatial` | 1 | 27.55 | 37.72 | +10.17 worse |
| | 2 | 30.41 | 50.54 | +20.13 worse |
| | 3 | 91.97 | 99.23 | +7.26 worse |

**So recursive parent error is not the dominant remaining error — the opposite holds.** The
child's local prediction has learned to cancel its parent's systematic rotation error, and
supplying a correct parent breaks that cancellation. Under end-to-end training on composed
frames, a parent-relative "local rotation" is not a parent-independent quantity at all; it is
partly an encoding of the parent's mistakes.

Change 1 saw the beginning of this for position at depth 2. On rotation it is larger, appears
at every depth, and is the main thing the diagnostic found.

The oracle leaks the true parent by construction. It is a diagnostic, it is computed only
here, and it is not a benchmark.

---

## 17. Leakage audit

| question | finding | evidence |
| --- | --- | --- |
| Are true global frames supplied at inference? | No | replacing every true frame with noise leaves inferred frames **and** part logits bit-identical, in all three cells |
| Could a *valid* alternative rotation leak? | No | swapping the two stored basis vectors of every true frame — a well-formed rotation, not noise — also leaves the output bit-identical |
| Is the oracle condition distinguishable? | Yes | the supplied condition does change the output, so the tests above are not vacuous |
| Are true parent frames supplied? | No | `to_global` composes the head's own output; `to_local` is never called in training or evaluation |
| Is held-out arrangement information supplied? | No | the parent table is built from the ontology, identical on every split, and is not the transposed oracle table |
| Is test transformation information supplied? | No | splits are read independently; the floor is fitted on `train` only |

Tests: `tests/test_step10_change2_leakage.py`, 7 tests, all passing. The true-parent oracle in
§16 leaks the parent **by construction**, is computed only in the post-hoc diagnostic, is
never a benchmark, and cannot be reached from any scoring path.

---

## 18. Integrity results

The §27 gate runs before every training run and again on every evaluated split: the parent
table against the generator's construction, the slot table, the composition order, the stored
bases, and the round trip — informative on stored frames for the first time, because the
rotations are real. In addition every run verifies the corpus checksums before training.

Results are in §8's integrity table.

---

## 19. Compute accounting

Recorded per run and reported in §8: arm, seed, cell, target formulation, hierarchy,
transform convention, parameter count, training steps, batch size, torch threads, training
time, evaluation time, depth-analysis time, the rotation objective and its weight, and every
final metric. All cells share one protocol, and no arm received extra tuning — no tuning was
performed at all.

---

## 20. Scientific conclusion

Everything below concerns **this formulation** — the chordal rotation term at 7% of the frame
loss, the frozen transform convention, Step 9's protocol — on **this synthetic corpus**. None
of it says the Lagnav 3D architecture is solved, and none of it is a claim about anatomy.

### Established

1. **Rotation was not learned above the identity-lookup floor _at this weight_.** No arm, no
   cell, no split, on mean, median or any threshold; the best margin anywhere is −0.07°, on
   the split whose floor is highest. **Superseded as a general claim**: at λ=3.0 the same
   architecture clears the floor by 2.55° on validation. The qualifier "at this weight" was
   not in the original text and is the correction.
2. **The failure is consistent, not noisy.** Rotation's across-seed spread is 0.05–0.36°
   against gaps to the floor of 0.6–2.1°.
3. **Learning rotation cost nothing in position.** A3's margin over the position floor is
   +0.0042 here against +0.0044 in Change 1, on identical position targets.
4. **The graph arms clear the position and scale floors on `test_seen`; the no-graph arms clear
   neither.** The Step 9 ordering survives the new objective.
5. **Composing onto a predicted parent hurts rotation**, by 2.85–4.23°, in all five arms, both
   conventions, passing the decision rule everywhere.
6. **`T4_rigid` behaves differently from `isotropic`, as pre-registered**: it costs +0.0001 to
   +0.0003 of scale where isotropic costs +0.0124 to +0.0128. A parent that propagates no scale
   propagates no scale error.
7. **Rotation error grows with hierarchy depth even where nothing is composed**, so the growth
   is entity difficulty, not a cascade.
8. **A parent-relative local rotation is not parent-independent.** Supplying the true parent
   makes rotation worse by 7–23°, because the child's prediction is compensating for its
   parent's error.
9. **The transform representation is not the problem.** It is closed under real rotation, loses
   nothing in composition, and reproduces each model's own frames to 0.0 in the diagnostics.

### Unsupported

* ~~That the architecture **cannot** learn rotation.~~ **Resolved against this report.** The
  rotation term was 7% of the frame loss (§5), and raising it to 19% or more clears the floor.
  Under-weighting was the explanation; this report was right to refuse the stronger claim.
* That the result would survive a different rotation weight, a longer schedule, or a
  rotation-only objective. None was run, because each would have meant choosing against test
  results.
* That parent-relative placement fails *in general*. Both cells here are worse than global, in
  two experiments now, on one synthetic corpus with one hierarchy.
* That the depth-3 rotation failure is about depth. It is confounded with a single entity.
* That the compensation effect in §16 generalises beyond this training regime.

### Architectural consequences

1. **Keep the global target.** Parent-relative placement is now worse on position (Change 1)
   and on rotation (here). It stays off.
2. **The rotation objective needs its own weight study before rotation is judged.** 7% of the
   frame loss is the first thing to rule out, and it must be decided on `train` or
   `validation`, never on test.
3. **Step 8's frame weights are due for revision.** The scale term takes 64% of the frame loss
   because it is an L1 sum over three log axes. That was harmless when rotation was
   structurally zero; it is not now.
4. **A parent-relative frame cannot be treated as a portable local transform.** §16 shows it
   encodes the parent's error. Any future local-edit design that moves a parent and expects
   children to follow must account for this.
5. **The entity-difficulty confound must be controlled in every future depth analysis.** The
   non-composing cell is what makes a depth profile readable.

### Open questions

* Does rotation learning appear at a larger rotation weight, and at what cost to position?
* ~~Is the scene-specific part of rotation learnable at all here?~~ **Answered after the fact —
  yes, and by a wide margin.** See the addendum below.
* Why does no arm extrapolate on `test_transform`, where the floor itself degrades to 35.95°?
* Would a rotation-only head, or a separate rotation loss schedule, break the tie with the
  lookup?
* Does the §16 compensation persist with a stop-gradient through the parent?

---

## 20a. Addendum: the information is in the inputs, and the models did not use it

**Added after the conclusion above was written**, and it changes how §10 should be read. It
trains nothing: it is a lookup table, fitted on `train`, using only inputs the model already
receives.

The corpus is built so that the relationship graph is the **only** channel carrying the
arrangement — presence, text features and entity ordering are identical across arrangements.
The question is therefore whether that channel carries enough to predict rotation. Keying the
same chordal-mean predictor on **entity identity and relation graph** instead of entity
identity alone:

| split | identity only | identity + graph | gain | training graph seen |
| --- | ---: | ---: | ---: | ---: |
| test_seen | 22.65° | **11.57°** | 11.08° | 92% |
| test_arrangement | 22.54° | **11.33°** | 11.21° | 91% |
| test_transform | 35.95° | **19.26°** | 16.69° | 81% |
| test_combination | 26.10° | 26.10° | 0.00° | 0% |

**Roughly half the rotation error is recoverable by a lookup table.** The trained models
scored 23.6–23.9° on `test_seen` — worse than that lookup by a factor of two, and barely
distinguishable from the predictor that ignores the graph entirely.

Three consequences.

1. **The result in §10 is not explained by missing information.** The arrangement reaches the
   model, and the part of rotation that depends on it is substantially predictable. The models
   did not use it.
2. **The headroom is large and now quantified.** Future rotation work has a second, much
   harder bar: 11.57° on `test_seen`, not 22.65°. A model that merely beat the identity-only
   floor would still be well short of a lookup.
3. **`test_combination` is the exception that confirms the mechanism.** Its graphs are never
   seen in training — the mirror-and-transpose combination is held out — so the graph channel
   supplies nothing there, exactly as the compositional hold-out intends.

This strengthens the §5 limitation rather than replacing it: with real, reachable headroom
left untouched, "rotation was 7% of the objective" becomes the leading explanation, and the
weight study in §21 becomes the first thing to run.

Produced by `experiments/step10/graph_conditional_floor.py`; artifact
`graph_conditional_floor.json`.

---

## 21. Next step

**Done, and it changed the conclusion.** The study ran; see
`docs/STEP_10_ROTATION_WEIGHT_STUDY.md` and the amendment at §10. What follows is the
recommendation as it was written before it ran:

**Recommended: a rotation-weight study on `train` and `validation` only, before any further
architectural work** — and §20a makes it considerably more urgent, because it shows the
headroom the study would be chasing is real and large. This experiment cannot distinguish "the architecture does not learn
rotation" from "rotation was 7% of the objective", and that distinction decides everything
after it. The study should fix its weights and its stopping rule in advance, report on
`validation`, and touch no test split until a single weight is frozen.

Two smaller pieces of work follow from §16 and §5 and can run alongside: re-deriving the frame
weights now that the rotation term is non-degenerate, and testing whether a stop-gradient
through the parent removes the compensation effect.

**Not recommended yet: real anatomical data, production geometry or scaling.** Change 2 shows
the controlled synthetic benchmark still has an unanswered question at its centre. Per §23,
that finishes first.

---

## Test results

The full suite passes: **866 tests**, up from 841 at the end of the corpus pass, so **25 are
new in this pass**. Ruff is clean. Both mypy tiers are clean — 54 tier-one files, 103
tier-two. Every Step 10 suite is green: 248 tests.

| suite | tests | what it covers |
| --- | --- | --- |
| `test_step10_rotation_loss.py` | 18 | §3–§5: the chordal objective, that it cannot be gamed by inflating the stored basis, that its gradient is finite where the geodesic diverges, and that the default still reproduces Change 1 bit for bit |
| `test_step10_change2_leakage.py` | 7 | §17: noise replacement and a valid-alternative-rotation probe, in all three cells |
| all Step 10 suites | 248 | Change 1's and the corpus pass's tests remain green, unmodified |
| whole repository | 866 | every earlier step's guarantees |

**Both freezes verified after the experiment**: Change 1's 101 artifacts and the corpus's
7 files all match their recorded checksums.

---

## Reproducibility

```
python -m experiments.step10.calibrate_rotation_weight
python -m experiments.step10.run_change2 --seeds 0
python -m experiments.step10.run_change2 --seeds 1
python -m experiments.step10.run_change2 --seeds 2
python -m experiments.step10.run_change2 --assemble-only
python -m experiments.step10.change2_diagnostics
python -m experiments.step10.change2_training_tables --insert-into docs/STEP_10_CHANGE2_TRAINING_REPORT.md
```

Artifacts in `experiments/runs/step10-change2/`: `runs/*.json` (one per run),
`checkpoints/`, `change2_report.json`, `change2_diagnostics.json`,
`rotation_weight_calibration.json`, `rotated_placement_floor.json`, `rotation_audit.json`,
`corpus_validation.json`, `CORPUS_FROZEN.sha256`, `logs/`.

Code state: committed on branch `step10-placement-and-rotation`. The corpus manifest records
`66606a0` as its `generator_commit`, which is the commit the corpus was generated from.

---

## Status at completion

```text
CHANGE 2 STATUS
----------------
Corpus: FROZEN
Training: COMPLETE
Integrity: PASS
Leakage: PASS
Rotation target: VALID
Rotation learning: NOT SUPPORTED at lambda = 0.33
                   SUPPORTED at lambda = 3.0, A3 only, confirmed on the test splits
                   (+1.27 deg on test_seen, seed spread 1.03; see the weight study)
Position preservation: SUPPORTED
Scale behavior: SUPPORTED
T4_rigid control: COMPLETE

Rotation floor (test splits, lambda = 0.33 runs):
test_seen = 22.65 deg
test_arrangement = 22.54 deg
test_transform = 35.95 deg
test_combination = 26.10 deg

Change 3:
NOT STARTED
```

The two rotation-learning lines are the amendment of 2026-09-22. The first is what this
experiment measured; the second is what the weight study found afterwards, on validation, and
it is not yet a confirmatory result — the matrix at the frozen weight is what makes it one.
