# Experiment heart-001: structured anatomical representation versus appearance

**Status: run, measured, reported. Synthetic data only. No medical claim.**

Pre-registered in Step 5, executed in Step 6, thresholds unchanged. Everything below
was measured on a corpus this project generated, stamped
`SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED`. An ellipsoid is not a ventricle,
and no number here is evidence about anatomy.

---

## 1. Hypothesis

> **H1.** A structured anatomical latent representation, jointly encoding semantic
> identity, anatomical relationships, spatial structure and 3D geometry, provides
> better anatomical consistency, controllability, semantic editing and persistent scene
> modification than a purely appearance-driven representation.

Decomposed, as registered: **H1a** part control, **H1b** held-out relationship
accuracy, **H1c** identity and edit locality across a session, **H1d** advantage grows
as data shrinks, **H0** no difference beyond seed variance.

## 2. Implementation

Described in [step-6-prototype.md](step-6-prototype.md). In one line: the Step 5
architecture built in PyTorch, trained on a generated tier-0 corpus, against a
parameter-matched appearance-driven baseline.

## 3. Setup

| | |
| --- | --- |
| corpus | tier-0 synthetic, 2,000 scenes, 40 procedural families, levels of detail 1 to 3 |
| splits | 1,600 train / 200 validation / 200 test, by family, no family in two splits |
| steps | 1,200 per run, batch 8, AdamW, learning rate 3e-4, warmup 50, cosine decay |
| sampling | 256 scene points and 32 points per entity per scene |
| seeds | 0, 1, 2 for the two main arms; seed 0 for each ablation |
| device | CPU, Apple M2, 4 torch threads |
| runtime | 3,131 s for 11 runs; A3 about 400 s per run, A0 about 72 s |
| parameters | A3 3,094,551 and A0 3,103,894, matched to 0.3 percent |

## 4. Results: structured versus appearance

Mean over three seeds, with the per-seed range.

| metric | A3 structured | A0 appearance | direction |
| --- | --- | --- | --- |
| part-control success | **0.931** [0.929, 0.931] | 0.313 [0.289, 0.359] | higher better |
| part-control, named entities | **0.988** | 0.381 | higher better |
| part IoU (mean per entity) | **0.803** | 0.323 | higher better |
| relationship accuracy, strict | **0.900** | 0.700 | higher better |
| relationship accuracy, among placed | 0.945 | 0.910 | higher better |
| relationship coverage | **0.950** | 0.764 | higher better |
| untouched-entity drift | **0.000000** | 0.001303 | lower better |
| geometry tokens identical after edit | **1.00** | 0.00 | higher better |
| occupancy-set Chamfer | **0.0137** | 0.0478 | lower better |
| scene IoU | **0.891** | 0.670 | higher better |
| level-of-detail part agreement | 0.9992 | 0.9992 | higher better |
| level-of-detail centroid shift | 0.0003 | 0.0004 | lower better |

Per-seed part-control success:

| seed | A3 | A0 |
| --- | --- | --- |
| 0 | 0.9309 | 0.3592 |
| 1 | 0.9295 | 0.2891 |
| 2 | 0.9312 | 0.2908 |

The structured arm's three seeds span 0.002; the baseline's span 0.070. The ranges do
not overlap.

## 5. Pre-registered criteria

| hypothesis | metric | threshold | result | verdict |
| --- | --- | --- | --- | --- |
| H1a | part-control success | gap >= 20 points, non-overlapping seed ranges | **+61.7 points**, ranges [0.929, 0.931] vs [0.289, 0.359] | **PASS** |
| H1b | relationship accuracy (strict) | gap >= 15 points | **+20.0 points** | **PASS** |
| H1c | untouched-entity drift | structured near zero where baseline drifts | structured max **0.0**, baseline mean 1.3e-3 | **PASS** |
| guard | occupancy Chamfer | no more than 20 percent worse | **71.4 percent better** | **PASS** |
| H1d | sample efficiency | advantage grows as data shrinks | gap flat: +61.8 points at full data, +62.7 at one tenth | **NOT SUPPORTED** |

Four of the five criteria pass. **H1d does not**, and the reason is instructive: the
corpus is too easy to be data-limited, so sample efficiency cannot be measured on it.

**H0 is not supported** on this corpus: every primary metric separates the arms by far
more than seed variance.

Two caveats attach to this table before anything is concluded from it, and they are in
sections 9 and 10.

## 6. Ablations

Seed 0, same corpus, same budget. A0 is the appearance baseline; A3 is the full
proposal.

| arm | change | part control | relationship (strict) | scene IoU | Chamfer | parameters |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | no structure at all | 0.313 | 0.700 | 0.670 | 0.0478 | 3,103,894 |
| A1 | entity axis, **no graph encoder** | 0.920 | 0.896 | 0.881 | 0.0142 | **427,655** |
| A2 | structure graph only | 0.930 | 0.900 | 0.891 | 0.0137 | 3,094,551 |
| A3 | all three graphs (full) | 0.931 | 0.900 | 0.891 | 0.0137 | 3,094,551 |
| A4 | structured encoder, **shared field** | 0.523 | 0.843 | 0.696 | 0.0376 | 3,182,402 |
| A5 | per-level latents instead of nesting | 0.933 | 0.898 | 0.893 | 0.0136 | 3,102,743 |
| A6 | pairwise alignment instead of prototypes | 0.926 | 0.900 | 0.897 | 0.0142 | 3,094,551 |

### What the ablations say

**Most of the effect is factorisation, not relational reasoning.** A1 removes the
graph encoder entirely and keeps 98.8 percent of A3's part control with 14 percent of
the parameters. The typed graphs are worth about **1.1 points** of part control and
**0.4 points** of relationship accuracy here. That is a genuine and uncomfortable
result for the design: the entity axis and per-entity fields carry the claim, and the
graph encoder, which is 86 percent of the model, contributes little on this corpus.

**Per-entity fields are what matter.** A4 keeps the entity latents and all three graphs
but decodes one shared field with a segmentation head, and part control falls from
0.931 to 0.523. Per-entity geometry is the single most load-bearing decision in the
architecture.

**Nesting is not required for these metrics.** A5 gives each level its own latent block
and matches A3 (0.933 versus 0.931). Nesting buys identity stability across levels,
which both arms achieve here anyway; it does not buy accuracy.

**Prototype anchoring is not decisive on this corpus.** A6 swaps the prototype hub for
pairwise contrastive alignment and loses 0.5 points. With one seed that is noise.

A1's parameter count is **not matched**, because removing the graph encoder removes
most of the model. A1 is therefore a test of "what does the graph encoder add", not a
budget-matched comparison, and the difference could partly reflect the different
optimisation problem a smaller model faces.

## 7. Sample efficiency (H1d)

Both arms re-run with the training split cut to 160 scenes, one tenth of the data, at
the same step budget and seed.

| metric | A3 full (1,600) | A0 full | gap | A3 tenth (160) | A0 tenth | gap |
| --- | --- | --- | --- | --- | --- | --- |
| part-control success | 0.9305 | 0.3130 | **+0.618** | 0.9280 | 0.3012 | **+0.627** |
| relationship accuracy (strict) | 0.8998 | 0.6999 | +0.200 | 0.9023 | 0.7349 | +0.168 |
| scene IoU | 0.8906 | 0.6702 | +0.220 | 0.8861 | 0.6746 | +0.212 |
| occupancy Chamfer | 0.0137 | 0.0478 | -0.034 | 0.0141 | 0.0439 | -0.030 |

**Verdict: H1d NOT SUPPORTED.** The advantage is flat, not growing. The part-control
gap moves from 61.8 to 62.7 points, which is within the seed variation already seen,
and the relationship gap actually shrinks.

The more informative result is what happened to both arms: **neither degraded**. A
tenfold data cut cost the structured arm 0.3 points of part control and the baseline
1.2 points. The tier-0 corpus is not data-limited at 2,000 scenes, so it cannot
discriminate sample efficiency at all. That is a property of the corpus, not evidence
against the hypothesis, and it is a concrete instruction for the next experiment:
fewer procedural families, wider parameter ranges, or a harder task.

One seed per arm at the reduced size, so no variance estimate. The machine-generated
criteria file records H1d as `NOT MEASURED` because this run was a separate invocation;
this section is the measurement.

## 8. Diagnostics

Run on the trained checkpoints, on held-out scenes.

| diagnostic | A3 (three seeds) | A0 | reading |
| --- | --- | --- | --- |
| sibling cosine similarity | 0.061 to 0.104 | not applicable | the four chambers have **not** collapsed; sibling similarity is at or below random-pair similarity |
| relation sensitivity, subject context delta | 0.036 to 0.043 | not applicable | adding one typed edge does move the latent |
| relation sensitivity, identity delta | **0.0** | not applicable | the encoder never writes the identity slice, as designed |
| level-of-detail token usage, 4 to 32 tokens | 0.0006 to 0.0019 | 0.0003 to 0.0011 | **later tokens contribute almost nothing** |
| untouched drift | **0.0** | 0.0008 to 0.0010 | locality is structural, not learned |

Two diagnostics deserve attention.

**A2's relation sensitivity is exactly 0.0** for an added *spatial* edge, because A2
masks every non-global head to the structure graph. The diagnostic detects the
ablation, which is a check on the diagnostic as much as on the model.

**The nested token block is barely used.** Moving from a 4-token prefix to the full
32-token block changes occupancy probability by about 0.001. The level-of-detail design
preserves identity, as claimed, but the extra capacity at finer levels is not being
exploited. Step 5 recorded this as an open question and warned against assuming nesting
works; the measurement says it does not, yet.

## 9. Failure cases

**Wall layers are not learned at all.** Endocardium, myocardium, epicardium and
pericardium score IoU 0.000 to 0.012 in the structured arm, against 0.94 to 0.98 for
septa and valves. Classification: **data and metric failure, not representation
failure.** These entities are thin shells that enclose everything else, and the corpus
ownership rule gives every interior point to the smaller structure inside. Almost no
query point is *owned* by a wall layer, so there is nothing to learn and nearly nothing
to score. The fix is in the corpus and the metric, not the model: sample points on the
shells themselves and score ownership per surface rather than per volume.

**The baseline had not converged.** A0's part control was still climbing at the end of
training (0.232 at step 600, 0.359 at step 1200) while A3 had plateaued. The 61.7-point
gap is therefore an upper bound on the true gap at this budget. Classification:
**optimisation, not representation.** A longer baseline run is the first thing to do
before repeating this claim.

**Mid-training and final evaluations are not comparable.** The final evaluation uses
twice as many batches and therefore a different mix of levels of detail. A3 reads 0.993
at step 600 and 0.931 at step 1200 for this reason, not because it regressed.

**The corpus generator needed a structural fix.** The first version placed
atrioventricular valves at the midpoint between chamber centres, which for thin-atrium
families left the valve too far from the atrium surface to satisfy the ontology's
adjacency relation; rejection sampling could not fix a systematic error. Classification:
**data failure.** The generator now derives anchors from the sampled radii so the
relation holds by construction.

**The baseline's part head was initially unconditioned.** It read the raw point encoding
with no access to the scene latent, which would have crippled the arm the hypothesis is
tested against. Classification: **implementation bug**, found before the run and fixed;
the part head now reads the same token-conditioned features as the occupancy head, and
both arms mask their part logits to the requested entity set.

## 10. Threats to validity

1. **The corpus is generated per entity, and the winning representation is per entity.**
   This is the sharpest threat. A4 partly addresses it by showing that the entity axis
   without per-entity fields is much weaker, but only real segmented data can settle it.
2. **Part control is structurally advantaged for the structured arm.** Its part labels
   come from composing per-entity fields; the baseline must learn a 43-way point
   classifier. That difference *is* the hypothesis, but it means the metric is not
   neutral between the arms and should not be read as a general quality score.
3. **The structured arm trains on two objectives the baseline cannot use**, entity
   identity and entity frames, because the baseline has no entity axis or frames. No
   equivalent was invented for it. The gap therefore reflects objectives as well as
   architecture.
4. **n = 3 seeds**, and one seed for each ablation. Ranges are reported; no confidence
   interval is claimed.
5. **One organ, 42 entities, one corpus, one budget.** Nothing here generalises to
   other anatomy, to real data, or to larger models.
6. **The relationship metric definition was chosen after seeing the coverage gap**
   between arms. Both definitions are reported; see
   [ADR 0014](adr/0014-strict-relationship-metric.md).

## 11. What was falsified and what was supported

**Supported on this corpus:** that a structured anatomical representation gives
substantially better part-level control, better recovery of held-out anatomical
relationships, and exactly zero drift on untouched entities, at equal parameter count
and without sacrificing geometry quality.

**Not supported:** H1d, that the advantage grows as data shrinks. The gap is flat, and
the corpus turns out not to be data-limited.

**Falsified, or at least not supported:** the implicit assumption running through Step 5
that the **typed graph encoder** carries much of the benefit. It does not, here. A1
reaches within 1.1 points of the full model with 14 percent of the parameters. Step 5's
ADR 0002 argued the head partition matters; this corpus does not show it mattering.

**Also not supported:** that nested level-of-detail prefixes are used. Identity is
preserved across levels, but the later tokens barely change the field, and per-level
latents (A5) perform identically.

**Unmeasured:** everything about real anatomy, image grounding, materials, animation,
and local geometric editing.

## 12. Reproducing this

```bash
pip install -e ".[dev,prototype]"
python -m datasets.synthetic.cli --scenes 2000 --families 40 \
    --out datasets/processed/heart_tier0_2000
python -m experiments.heart.run_experiment \
    --corpus datasets/processed/heart_tier0_2000 \
    --steps 1200 --seeds 0 1 2 --ablations A1 A2 A4 A5 A6 \
    --eval-limit 96 --out experiments/runs/heart-001
```

Environment recorded in every manifest: Python 3.14.6, PyTorch 2.14.0, NumPy 2.5.3,
macOS arm64, CPU, 4 threads. Commit: **unavailable, this is not a git repository**,
which is a reproducibility gap worth closing with `git init`.

Artefacts: `experiments/runs/heart-001/experiment_report.json` (full results and every
run manifest), `diagnostics.json`, `checkpoints/`, and rendered views under `views_A3`
and `views_A0`.

## 13. Recommended next step

See the Step 7 recommendation at the end of the final report. In short: the evidence
points at the **per-entity factorisation**, not at the graph encoder, so the next
experiment should test the graph encoder on a task that actually requires relational
reasoning, and replicate the main result on data that was not generated per entity.

---

## Appendix A: per-entity part IoU, structured arm, seed 0

Best and worst, from `experiments/runs/heart-001/experiment_report.json`.

| entity | IoU (A3) | IoU (A0) |
| --- | --- | --- |
| trabeculae carneae | 0.977 | 0.000 |
| interventricular septum | 0.968 | 0.281 |
| interatrial septum | 0.959 | 0.180 |
| mitral valve | 0.954 | 0.170 |
| pulmonary valve | 0.943 | 0.147 |
| left ventricle | 0.916 | 0.537 |
| moderator band | 0.637 | 0.000 |
| myocardium | 0.012 | 0.000 |
| endocardium | 0.000 | 0.000 |
| epicardium | 0.000 | 0.000 |
| pericardium | 0.000 | 0.000 |

The four wall layers fail in both arms for the reason given in section 9: the corpus's
ownership rule gives their interior points to whatever they enclose, so almost nothing
is labelled as belonging to them.

## Appendix B: artefacts

| path | contents |
| --- | --- |
| `experiments/runs/heart-001/experiment_report.json` | every run manifest, aggregates and criteria |
| `experiments/runs/heart-001/diagnostics.json` | sibling separation, relation sensitivity, token usage, locality |
| `experiments/runs/heart-001/checkpoints/` | 11 checkpoints with their manifests |
| `experiments/runs/heart-001/views_A3/`, `views_A0/` | rendered occupancy, ownership and edit-drift slices |
| `experiments/runs/heart-001-small/` | the reduced-data runs behind section 7 |
| `datasets/processed/heart_tier0_2000/` | the corpus, as parameters plus a manifest |

Rendered views are orthographic slices at `z = 0`, 160 by 160, coloured by entity. Side
by side, the structured arm's ownership map reproduces the ground-truth layout
including the vessels; the baseline's is blobby, loses the vessel tubes and leaves
stray fragments.
