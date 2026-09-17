# Step 10 experiments

Target formulation, hierarchical placement and relation-grounded supervision.

## What is here

| file | what it answers |
| --- | --- |
| `placement_floor.py` | what an identity-only predictor scores under each target formulation |
| `run_placement.py` | Change 1: does a parent-relative target place better than a global one |
| `depth_analysis.py` | where in the hierarchy any difference actually occurs |

## Run order

The floor first, always. It is computed from the corpus alone — no model, no arm, no
training — and it is what every arm is read against.

```
python -m experiments.step10.placement_floor \
    --corpus datasets/processed/step8_continuous \
    --out experiments/step10/placement_floor.json

python -m experiments.step10.run_placement \
    --corpus datasets/processed/step8_continuous \
    --arms A1 A3Lite A3 --cells T0_global T1_spatial \
    --seeds 0 --steps 1200 --out experiments/runs/step10-placement
```

Exploratory at one seed first. Only a cell that moves is worth three seeds, and the
distinction is recorded in the report's `status` field rather than left to the reader.

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

## Reading the numbers

**The bar is 0.1605, for every cell.** Every arm is scored on global frames with the Step 9
metric, so the floor is a property of the task and the ruler, not of the arm. The
`parent_relative` floor is context for what the new formulation costs a blind predictor; the
oracle column leaks the true parent and is a diagnostic, never a baseline.

**`T2_taxonomic` must equal `T0_global` exactly.** AWR's hierarchy leaves every whole-organ
entity a root, so composing over it is the identity. `run_placement._control_identity`
checks it rather than assuming it; a non-zero difference means the control has quietly
become a third treatment and the comparison is invalid.

**Half the entities cannot move.** Ten of twenty have a parent, so an aggregate difference is
consistent with the hierarchy working and equally consistent with the run drifting.
`depth_analysis.py` separates those: depth 0 is a built-in control, because those entities
are roots and are predicted identically in both cells.

**Translation, scale and rotation are reported separately.** Rotation error is structurally
zero on this corpus — the frame target's rotation is a constant identity in every scene —
and is reported as measured rather than quietly dropped.

## What would invalidate a result here

* a difference at depth 0 between two cells: that is not the hierarchy, because those
  entities were predicted the same way;
* `T2_taxonomic` differing from `T0_global`;
* the `global` floor not reproducing Step 9's 0.1605 / 0.1655 / 0.1703 / 0.1750, which would
  mean the pipeline has drifted and nothing else is comparable.

All three are asserted in code or in `tests/test_step10_integrity.py`.

## Data

Synthetic research data. Not anatomy, not validated, not clinical.
