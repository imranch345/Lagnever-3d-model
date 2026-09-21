# Step 10 experiments

Target formulation, hierarchical placement and relation-grounded supervision.

## What is here

| file | what it answers |
| --- | --- |
| `placement_floor.py` | what an identity-only predictor scores under each target formulation |
| `run_placement.py` | Change 1: does a parent-relative target place better than a global one |
| `depth_analysis.py` | where in the hierarchy any difference actually occurs |
| `report_tables.py` | every §29 table, generated from `placement_report.json` |
| `inheritance_diagnostic.py` | post hoc: how much of a child's error it inherits from its parent |

The §27 integrity gate the trainer runs before training lives in
`generation/neural/nn/placement_integrity.py`.

## Run order

The floor first, always. It is computed from the corpus alone — no model, no arm, no
training — and it is what every arm is read against.

```
python -m experiments.step10.placement_floor \
    --corpus datasets/processed/step8_continuous \
    --out experiments/step10/placement_floor.json
```

The confirmatory matrix is five arms × three pre-registered cells × three seeds, at Step 9's
protocol (1200 steps, batch 8, CPU, 4 torch threads). It runs as one worker per seed; each
finished run is written to `runs/<arm>__<cell>__seed<n>.json` at once, and a restarted
worker reuses finished runs rather than retraining them:

```
python -m experiments.step10.run_placement --seeds 0 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --seeds 1 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --seeds 2 --out experiments/runs/step10-placement
python -m experiments.step10.run_placement --assemble-only --out experiments/runs/step10-placement
python -m experiments.step10.report_tables --all-splits
```

`T3_alternative` and `T4_rigid` are defined but are **not** part of this matrix: the parent
convention stays fixed for the duration of the Change 1 experiment.

## Decision rule, fixed before the confirmatory results existed

The comparison is `T1_spatial − T0_global`, paired by seed (the seed fixes initialisation and
data order, so pairing removes the variation the two cells share). A difference **exceeds
seed variability** only if

1. all three paired seed differences have the same sign, **and**
2. the paired t-statistic exceeds 4.303 in magnitude (two-sided 5%, 2 degrees of freedom).

The same rule is applied per depth. It is coded as `T_CRITICAL_DF2` in `run_placement.py`
and was written here before any confirmatory run finished. With three seeds it is a strict
rule; a result that fails it is reported as not established, not as absent.

## The prediction, recorded before anything was trained

`placement_floor.json`, computed first:

| split | `global` | `parent_relative` | `parent_relative_oracle` |
| --- | --- | --- | --- |
| test_seen | 0.1605 | 0.1636 (−1.9%) | 0.1213 (+24.4%) |
| test_combination | 0.1750 | 0.1774 (−1.4%) | 0.1515 (+13.4%) |

The parent-relative target is worth about a quarter of the position error **to a predictor
that already places parents well**, and costs about two percent to one that does not. Step 9
found only `A3` reliably clears the floor, `A3Lite`'s margin was indistinguishable from zero
and three of five arms placed worse than a lookup table. So the expected result is that
Change 1 helps `A3`, does little for `A3Lite`, and actively hurts `A1`.

Confirming that is a result. Contradicting it is also a result. It is written down here
first so that neither can be decided after the fact.

**Outcome (2026-09-20): not supported.** Parent-relative placement placed worse than the
global target in all five arms, and A3 — the arm the prediction expected to gain — lost the
most and fell below the floor. The full analysis, including a post-hoc diagnostic of where
the loss comes from, is in `docs/STEP_10_CHANGE1_REPORT.md`.

## Reading the numbers

**The bar is 0.1605, for every cell.** Every arm is scored on global frames with the Step 9
metric, so the floor is a property of the task and the ruler, not of the arm. The
`parent_relative` floor is context for what the new formulation costs a blind predictor; the
oracle column leaks the true parent and is a diagnostic, never a baseline.

**`T2_taxonomic` must equal `T0_global` exactly.** AWR's hierarchy leaves every whole-organ
entity a root, so composing over it is the identity. `run_placement._control_identity`
checks it rather than assuming it; a non-zero difference means the control has quietly
become a third treatment and the comparison is invalid.

**Half the entities cannot move directly.** Ten of twenty have a parent, so an aggregate
difference is consistent with the hierarchy working and equally consistent with the run
drifting. `depth_analysis.py` separates those: depth 0 is a built-in control, because those
entities are roots whose target is the same in both cells. Depth is always read from the
spatial tree, whatever the cell, and each run reconciles its depth buckets against the
headline number to 1e-6.

**Translation, scale and rotation are reported separately.** Rotation error is structurally
zero on this corpus — the frame target's rotation is a constant identity in every scene —
and is reported as measured rather than quietly dropped.

## What would invalidate a result here

* an aggregate improvement that does not appear at depth ≥ 1: depth-0 entities are roots
  whose target is identical in both cells, so Change 1 can reach them only indirectly,
  through shared parameters — an effect concentrated there is not the mechanism claimed;
* `T2_taxonomic` differing from `T0_global` in any arm or seed;
* a `T0_global` run failing to reproduce the Step 9 baseline for its arm and seed;
* the `global` floor not reproducing Step 9's 0.1605 / 0.1655 / 0.1703 / 0.1750;
* the depth buckets failing to reconcile with the headline number (the runner stops);
* the §27 integrity gate failing (the trainer refuses to start).

Each is checked in code, reported in `placement_report.json`, or asserted in
`tests/test_step10_integrity.py` and `tests/test_step10_corruption.py`.

## Change 2: the rotated corpus

Change 1 is frozen; Change 2 is a separate track with its own corpus and its own floor. This
pass built and validated the data only — **nothing is trained**.

| file | what it does |
| --- | --- |
| `validate_corpus.py` | every corpus check: manifest, pairing with the parent, rotations, the placement gate, levels of detail, splits, hold-out |
| `rotation_audit.py` | §9: rotation statistics per split and per entity, read through the evaluation loader |
| `change2_tables.py` | every Change 2 report table, generated from the artifacts |

```
python -m datasets.whole_organ.step10_cli --out datasets/processed/step10_rotated
python -m experiments.step10.validate_corpus
python -m experiments.step10.rotation_audit
python -m experiments.step10.placement_floor --corpus datasets/processed/step10_rotated \
    --out experiments/runs/step10-change2/rotated_placement_floor.json
```

**The rotated corpus has its own floor, and 0.1605 is not it.** On `test_seen` the new floor is
0.1605 for position — the same number, because the corpus is derived from Change 1's and the
centroids are identical — and **22.65 degrees for rotation**, which is the number that matters
for Change 2. The raw target mean of 61.8 degrees is not the bar.

The full account is in `docs/STEP_10_CHANGE2_CORPUS_REPORT.md`.

## Change 2 training: does it learn rotation?

| file | what it does |
| --- | --- |
| `calibrate_rotation_weight.py` | §5: picks the rotation loss weight on `train` only, with the term unoptimised |
| `run_change2.py` | the 45-run matrix: 5 arms × {global, rigid, isotropic} × 3 seeds |
| `rotation_metrics.py` | §8: geodesic error in degrees, against this corpus's rotation floor |
| `floor_distribution.py` | the floor's own distribution, so a model's median is read against a median |
| `change2_diagnostics.py` | §15–§16: error propagation with depth, and the true-parent oracle |
| `change2_training_tables.py` | every §20 table, generated from the run artifacts |

**Outcome (2026-09-21): rotation NOT learned above the floor.** No arm, no cell, no split, on
mean, median or any threshold; best margin −0.07°. Position was unaffected — A3 keeps the same
+0.004 margin over the position floor it had in Change 1 — and `T4_rigid` separated from
`isotropic` on scale exactly as pre-registered. The rotation term was 7% of the frame loss, so
under-weighting is not ruled out; that is the recommended next study.

The full account is in `docs/STEP_10_CHANGE2_TRAINING_REPORT.md`.

## Data

Synthetic research data. Not anatomy, not validated, not clinical.
