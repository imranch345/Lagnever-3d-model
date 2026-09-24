# Step 16 — Escape trigger: preregistration

**Status:** preregistered. Written before any instrumented run was launched.
**Question:** what, if anything, changes first when an A3 run transitions from the identity-only
attractor to the relationally useful one?
**Diagnostic only.** Architecture, objective, λ, corpus, splits, budget, optimiser and
data-generation protocol are all frozen. The only addition is instrumentation.
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Configuration verification

Performed before writing this plan.

| check | result |
| --- | --- |
| corpus freeze (7 files) | **CLEAN** |
| Change 1 freeze (101 files) | **CLEAN** |
| Step 12 freeze (8 files) | **CLEAN** |
| Step 13 freeze (13 files) | **CLEAN** |
| Step 14 freeze (5 files) | **CLEAN** |
| Step 15 freeze (10 files) | **CLEAN** |
| git | `1987b6c`, clean working tree |

Frozen A3 configuration, read live from `default_config("A3")`:

```
steps 1200   batch_size 8   learning_rate 3e-4   warmup_steps 60   gradient_clip 1.0
weight_decay 0.01   rotation_objective chordal   rotation_loss_weight 3.0
frame_weight 2.0   lod_weight 0.5   frame_objective l1   placement_target global
graph_recurrence 1   frame_scene_context False   relation_values False
```

`record_seed` asserts `steps == 1200`, `graph_recurrence == 1`,
`frame_scene_context is False`, `relation_values is False` and `rotation_loss_weight == 3.0`
before training, so a drifted configuration stops the run rather than producing a tidy table.

### 1.1 A discrepancy in the brief, documented before training

§10 asks for checkpoint comparisons at steps 1400 through 2600. **The frozen budget is 1200
steps.** §3 forbids changing training duration and §11 forbids any intervention, so those
checkpoints cannot exist without violating the same brief that asks for them.

Resolution, per §10's own instruction to "use the nearest available checkpoint and document it":
Step 16 reports the dense trajectory to step 1200 and records steps 1400–2600 as **outside the
frozen budget and not measured**. Extending the budget is a different experiment; §8 of the Step
15 report already flagged it as such.

## 2. Seeds and rationale

Selected from the frozen Step 15 report, before any new trajectory was inspected.

| role | seed | rationale from Step 15 |
| --- | ---: | --- |
| **late escaper** | **9** | 23.54° at step 800 — indistinguishable from a permanent non-escaper — finishing at 21.11° with graph usage 2.52°. Named by the brief. |
| **non-escaper** | **5** | graph usage **0.09°**, the lowest of all ten seeds, and rotation 23.53° against a 23.51° identity-only floor. The clearest persistent identity-floor behaviour available, and its trajectory (23.73 → 23.47 → 23.65) never leaves the floor. Seed 1 was rejected because its usage (0.38°) is the highest of the three non-users and its trajectory declines slightly; seed 4 (0.29°) is second-clearest. |
| **intermediate** | **3** | the single seed in the unlabelled 0.8–1.5° band: usage 1.30°, rotation 22.17°, and still moving at step 1200 (23.89 → 23.72 → 22.27). |

No seed will be replaced or re-selected after inspecting a trajectory.

## 3. Instrumentation

A callback on the frozen training loop, invoked at every multiple of **50 steps** — including
step 0, which observes the initialisation because the callback runs *before* the step it is
given — and once more after the final step. Twenty-five checkpoints per run.

`fit()` gained one optional parameter, `on_step`, defaulting to `None`. With no callback
installed the loop is byte-for-byte the one every run since Step 8 used.

**The instrumentation must not change the run.** Three restorations around every measurement:
Python, NumPy and Torch RNG state (building a batch samples points); gradients zeroed (the
pathway measurement calls `backward`); and training mode restored. **Verified by test: a
60-step instrumented run reproduces its uninstrumented twin's weights exactly, with zero
differing tensors.**

### 3.1 Quantities recorded

**Graph usage** — Step 15's definition, unchanged: validation rotation with every relationship
removed, minus validation rotation with the graph intact. No new metric is created.

**Rotation** — intact, no-relationship, and the relationally-required subset. The identity-only
reference (23.51°) is recorded as a constant from the frozen artifact rather than recomputed.

**The relationally-required subset** is Step 14's preregistered selection, rebuilt from
train-only tables and **verified to contain exactly 652 entities**, matching Step 14. An earlier
implementation keyed it on `(relation signature, slot)` and reached 991 entities, because one
qualifying entity enrolled every entity in every other validation scene sharing its signature;
that would have been a broader subset wearing the preregistered name.

**Representation** — entity latent spread before and after the graph encoder, the norm and share
of what the encoder writes into the context slice, relation-bias scale, and the frame head's
rotation and translation output spread.

**Objective** — Step 14's decomposition, including the frame term's translation/scale/rotation
split, on a fixed set of training batches so checkpoints are comparable.

**Gradients** — Step 14's pathway norms, on the same fixed batches.

## 4. Escape, defined before inspecting anything

**Escape uses Step 15's existing threshold with no modification:** a checkpoint is in the
graph-using band when graph usage ≥ **1.5°**.

At 50-step resolution the crossing step is not the escape step. Step 16 reports the **interval**:

> previous checkpoint < escape ≤ first crossing checkpoint

**No interpolation of an exact escape step** unless the trajectory is dense enough within an
interval to support it, which at 50-step spacing it will not be.

## 5. Analysis plan

For each seed, locate: T1 first graph-usage departure, T2 first representation change, T3 first
gradient change, T4 first objective change, T5 first rotation improvement. Then report the
**observed** ordering. No ordering is assumed.

"First change" needs a criterion fixed in advance. For each scalar series, the first checkpoint
at which the value departs from its own step-0-to-400 baseline by more than **three baseline
standard deviations**, and stays departed at the next checkpoint. The two-checkpoint persistence
requirement is there because a single-checkpoint excursion at this resolution is not
distinguishable from noise.

## 6. Hypotheses

* **H9 — representation trigger.** A graph-conditioned representation becomes informative before
  downstream geometric improvement.
* **H10 — gradient trigger.** A change in gradient magnitude precedes the representation or
  geometry transition.
* **H11 — objective trigger.** One objective component changes in a way that predicts escape
  before graph usage rises.
* **H12 — no detectable precursor.** The transition occurs without a measurable precursor in
  these diagnostics.

**H12 is a live outcome and will not be dressed as a mechanism.**

## 7. Decision rule

Exactly one of:

* **A — clear precursor.** A variable changes before escape *and* separates the escaper from the
  non-escaper reproducibly.
* **B — weak or ambiguous precursor.** A candidate changes near escape but timing or separation
  is insufficient.
* **C — no detectable precursor.**

**B and C will not be converted into mechanistic claims.** Correlation will not be reported as
causality, and graph usage will not be treated as the mechanism — it is the thing being
explained.

## 8. Test discipline

**`test_splits_read: []`** for the whole diagnostic. Training uses train; every trajectory
measurement uses validation. No checkpoint is selected on test, and no mechanism is identified
from test.

## 9. Preflight

Disk 15 GiB free, well above the 2 GiB guard. Battery 100% and the machine is attended;
`caffeinate` holds it awake. Measured cost: **5.2 s per diagnostic call**, so 25 checkpoints ×
3 seeds ≈ 6.5 minutes of instrumentation on top of roughly 12 minutes of training per run.
Artifacts are three JSON trajectories plus three checkpoints (~12 MB each). No prior artifact is
deleted, overwritten or mutated; Step 16 writes only under `experiments/runs/step16`.

## 10. Stop conditions

* any frozen checksum changes;
* the instrumented run cannot be shown to reproduce an uninstrumented one;
* the relationally-required subset does not reproduce Step 14's 652 entities;
* a diagnostic would require changing the model, objective or budget;
* free disk falls below 2 GiB; research artifacts are never deleted to make space.
