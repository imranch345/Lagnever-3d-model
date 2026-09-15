# Step 7 — Representation stress test, graph necessity, whole-organ benchmark, editing

Step 7 is not a scaling step. Its purpose is to find out **which architectural ideas are
genuinely necessary**, and to remove or simplify the ones that are not.

All data here is procedurally generated. Not anatomy, not validated, not clinical.

## Read in this order

| Document | What it covers |
| --- | --- |
| [01 Experimental plan](01_experimental_plan.md) | Written before any run. Metric definitions, ownership rule, arms, controls, and what would count as a negative result. |
| [02 Pre-registration amendments](02_preregistration_amendments.md) | Every change made after the plan, with its reason and the date it was made relative to the runs. **Read this before any result table.** |
| [03 Whole-organ dataset](03_whole_organ_dataset.md) | Experiments 4 and 10. How the corpus is built and why the generation order was reversed. |
| [04 Graph necessity, method](04_graph_necessity_method.md) | Experiments 5 and 6. The arms, the controls, and the three prior answers that were withdrawn. |
| [05 Editing, method](05_editing_method.md) | Experiments 8 and 9. The three edit mechanisms and what would show the design does not work. |
| [06 Perturbation, method](06_perturbation_method.md) | Experiments 1, 2 and 3. Damaging one channel at a time to see which are read. |
| [07 Level of detail, method](07_level_of_detail_method.md) | Experiment 7. The level-specific objective, and why Step 6's was probably at fault. |
| [08 Baseline convergence, method](08_baseline_convergence_method.md) | Experiment 11. The stopping criterion, fixed in advance. |
| Completion report | Results, decisions, and what is withdrawn. |

## The one thing to know before reading any number

Step 7's central question has been answered three times, and every answer was withdrawn
on inspection. Not because the measurements were wrong, but because each setup could not
distinguish the hypotheses it was supposed to test.

| Attempt | Finding | Why withdrawn |
| --- | --- | --- |
| Step 6 | graph encoder not earning its parameters | one relationship graph across all 2,000 scenes |
| Step 7 suite 1 | `A1` matches `A3` | variant aliased onto family, so the contrast was never presented |
| Step 7 suite 2 | `A1` matches `A3` | part target named entities the level hides, costing ~1e9 per point for every arm |

Superseded runs are kept with their own notes under `experiments/runs/`, not deleted.

Three further conditions turned out to matter, and every result table names them:

* **Placement condition.** Supplying ground-truth entity frames hands the model the
  arrangement a relationship graph would otherwise supply. Decisions rest on the
  inferred condition. ([ADR 0018](../adr/0018-evaluate-both-placement-conditions.md))
* **The relation-blind floor.** A model ignoring relations scores 0.8878 on spatial
  relation accuracy, not zero. ([ADR 0019](../adr/0019-relation-blind-floor.md))
* **The step budget.** Most runs at 900 steps were still improving, so an equal-budget
  comparison shows which arm learns faster, not which ends up better. A longer run on the
  decisive arms checks whether the conclusion holds.

## Architecture decision records from Step 7

| ADR | Subject |
| --- | --- |
| [0015](../adr/0015-crossed-variant-corpus.md) | Variant and family must be crossed |
| [0016](../adr/0016-level-aware-part-target.md) | The part target names only entities the level exposes |
| [0017](../adr/0017-structural-graph-is-ontology-knowledge.md) | Structural edges come from the ontology, not measurement |
| [0018](../adr/0018-evaluate-both-placement-conditions.md) | Evaluate with placement supplied and inferred |
| [0019](../adr/0019-relation-blind-floor.md) | Relation accuracy is reported against a blind floor |

## Reproducing

```bash
# Corpus
python -m datasets.whole_organ.cli --out datasets/processed/whole_organ_1600_v2 \
    --scenes 1600 --families 40

# Six-arm suite (Experiments 4, 5, 6, 7)
python -m experiments.step7.run_whole_organ \
    --corpus datasets/processed/whole_organ_1600_v2 --steps 900 \
    --seeds 0 1 2 --arms A3 A1M A0 --single-seed-arms A1 A2 A3L \
    --out experiments/runs/step7-whole-organ

# Experiment 11
python -m experiments.step7.convergence \
    --checkpoints experiments/runs/step7-whole-organ/checkpoints

# Experiments 1, 2, 3
python -m experiments.step7.run_perturbation \
    --corpus datasets/processed/whole_organ_1600_v2 \
    --checkpoints experiments/runs/step7-whole-organ/checkpoints

# Experiments 8, 9
python -m experiments.step7.run_editing \
    --corpus datasets/processed/whole_organ_1600_v2 \
    --checkpoint experiments/runs/step7-whole-organ/checkpoints/wo-A3-seed0.pt

# Placement conditions
python -m experiments.step7.run_placement \
    --corpus datasets/processed/whole_organ_1600_v2 \
    --checkpoints experiments/runs/step7-whole-organ/checkpoints

# Result tables, rendered from the run JSON
python -m experiments.step7.report_tables
```

## What Step 7 does not do

No real medical data. No medical validation. No large-scale training. No production
model. No claim of anatomical or clinical accuracy. External 3D systems appear nowhere,
as baselines or otherwise.
