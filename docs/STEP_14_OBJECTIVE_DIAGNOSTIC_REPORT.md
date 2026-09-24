# Step 14 — Objective and relational-supervision diagnostic: report

**Preregistered in** [`STEP_14_OBJECTIVE_DIAGNOSTIC_PLAN.md`](STEP_14_OBJECTIVE_DIAGNOSTIC_PLAN.md).

**Status: H7 is NOT supported. The current objective is relationally identifiable.**
No loss was added. No coefficient was changed. Nothing was trained.

**`test_splits_read: []`** — the entire diagnostic ran on train and validation.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Research question

Does the current training objective actually force the model to learn relational information
when relational information is necessary to minimise the target error?

* **H7** — the objective is insufficiently relational: it can be minimised substantially
  through entity identity alone.
* **H8** — the objective already provides sufficient relational supervision, and the remaining
  failure has another cause.

## 2. Frozen Step 10–13 evidence

Verified from artifacts before any measurement. All four freeze manifests **CLEAN**.

| reference | value |
| --- | ---: |
| A3 rotation, `test_seen` | 21.38° ± 1.03 (20.79, 22.57, 20.78) |
| identity-only lookup, `test_seen` | 22.65° |
| identity + graph lookup, `test_seen` | 11.57° |
| A3 rotation, validation | 21.80° |
| identity-only lookup, validation | 23.51° |
| identity + graph lookup, validation | 12.24° |
| Step 12 relation-conditioned values | 21.80 → 21.81, gain −0.013° |
| Step 13 A3-depth / A3-global | 21.99° / 21.44° |

A3 recovers **15.2%** of the available relational improvement on validation.

## 3. Exact objective audit

Read from code. The total is **not** the loss module's breakdown: `Step8Trainer.train_step`
assembles

```
total = breakdown.total                          # the loss module's weighted terms
      + frame_weight (2.0) * _frame_loss(...)     # added outside the module
      + lod_weight   (0.5) * nested_lod(...)      # added outside the module
```

and `_frame_loss` is itself `translation + 0.5·scale + rotation_loss_weight·chordal`.

| Loss component | Weight | Normalisation | Identity | Graph | Geometry |
| --- | ---: | --- | --- | --- | --- |
| `entity_frame_chordal` (trainer) | **2.0** | mean over present entities | yes | indirect | yes |
| ↳ translation | ×1.0 | L1 over 3 components | yes | indirect | yes |
| ↳ scale | ×0.5 | L1 over 3 log-scale components | yes | indirect | yes |
| ↳ rotation | **×3.0** | `rotation_chordal`, bounded [0,1] | yes | indirect | yes |
| `entity_frame` (module) | **0.5** | (L1 pose + 0.5·L1 rot)/6, masked mean | yes | indirect | yes |
| `geometry_reconstruction` | 1.0 | per-point, masked mean | yes | no | yes |
| `semantic_part_correspondence` | 1.0 | masked mean | yes | no | yes |
| `nested_lod` (trainer) | 0.5 | BCE over levels | yes | no | yes |
| `language_anatomy_alignment` | 1.0 | contrastive | yes | no | no |
| `entity_identity` | 1.0 | prototype alignment | yes | no | no |
| `lod_consistency` | 0.5 | inactive (no prefix output) | — | — | — |
| `edit_consistency` | 1.0 | inactive (no edited output) | — | — | — |

**No loss term depends on the graph directly.** Every term is a function of the prediction, and
the graph reaches the objective only through the prediction. There is no relation loss, no graph
regulariser and no identity-specific penalty beyond `entity_identity`.

### 3.1 Two discrepancies worth recording

**(a) There are two frame terms, not one.** ADR-STEP10-003 describes the chordal objective as
replacing "only the rotation term". In implementation the chordal term is *added* by the trainer
at `frame_weight = 2.0`, while the older L1-on-six-numbers term (`entity_frame`, weight 0.5,
including its own 0.5·L1 rotation) remains live in the loss module. Both are active in every
Step 10–13 run.

**(b) The effective rotation coefficient is 6.0, not 3.0.** λ=3.0 is the coefficient *inside*
`_frame_loss`, which is then multiplied by `frame_weight = 2.0`. Every report's "λ=3.0" is
accurate as written but describes an inner coefficient. Not changed here, as required.

## 4. Loss decomposition

Frozen A3 checkpoints, train split, measured at step 1199 (teacher forcing off — the regime
training mostly runs in). Seed 0; all three seeds agree within 1%.

| component | raw | weight | weighted | share of total |
| --- | ---: | ---: | ---: | ---: |
| `entity_frame_chordal` | 0.6102 | 2.00 | 1.2205 | **49.9%** |
| `geometry_reconstruction` | 0.5490 | 1.00 | 0.5490 | 22.4% |
| `semantic_part_correspondence` | 0.4406 | 1.00 | 0.4406 | 18.0% |
| `nested_lod` | 0.3144 | 0.50 | 0.1572 | 6.4% |
| `entity_frame` | 0.1538 | 0.50 | 0.0769 | 3.1% |
| `language_anatomy_alignment` | 0.0014 | 1.00 | 0.0014 | 0.1% |
| `entity_identity` | 0.0014 | 1.00 | 0.0014 | 0.1% |
| `lod_consistency` / `edit_consistency` | 0.0000 | — | 0.0000 | 0.0% |
| **total** | | | **2.4470** | 100% |

Inside the chordal frame term:

| part | raw | coefficient | share of total objective |
| --- | ---: | ---: | ---: |
| translation | 0.2424 | 1.0 | 19.8% |
| scale | 0.3238 | 0.5 | 13.2% |
| **rotation** | 0.0686 | 3.0 | **16.8%** (16.8 / 17.6 / 16.7 across seeds) |

**Rotation is 16.8% of the total objective.** It is not a marginal term, and the objective is
not dominated by terms that ignore placement — frame supervision is 53% of the total.

## 5. Identity-only analysis

| split | identity-only lookup |
| --- | ---: |
| validation | 23.51° |
| `test_seen` | 22.65° |

Fitted on train only: the chordal mean rotation per entity slot. This is what identity alone can
achieve.

## 6. Graph-aware lookup analysis

| split | identity-only | identity + graph | improvement | coverage |
| --- | ---: | ---: | ---: | ---: |
| validation | 23.51° | **12.24°** | 11.27° | 89.8% |
| `test_seen` | 22.65° | **11.57°** | 11.08° | 91.9% |

Read from existing artifacts; no test split was recomputed. Coverage is the share of entities
whose scene's relation signature appeared at least twice in training.

## 7. Relational residual — clause 1

**Is relationship-dependent target variation present and measurable? Yes, overwhelmingly.**

For each entity slot, the spread of its rotation target is decomposed into what remains after
conditioning on the scene's relation signature and what the signature explains. Measured in the
objective's own units (mean chordal distance from the conditional mean), on train, 20 slots:

| quantity | value |
| --- | ---: |
| total spread | 0.05294 |
| within-graph spread | 0.00635 |
| **explained by the graph** | **0.04659 (88.0%)** |

**88% of the target rotation variation is graph-determined.** Conditioning on identity alone
leaves almost all of it unexplained. Clause 1 is satisfied.

## 8. Counterfactual identifiability test

The corpus **does** contain the required counterfactuals, and they are the normal case rather
than a rare one. The same entity slot appears across many scenes with different relation
signatures, and its target rotation changes accordingly — which is precisely what §7 measures:
within a signature the rotation is nearly fixed (0.00635), across signatures it varies eight
times as much (0.05294).

Concretely, on validation, 2,080 of 2,316 entities (89.8%) have a signature seen in training, so
for them identity and graph-conditioned predictions are both defined and differ. No corpus
modification was needed; no new target representation was introduced.

## 9. Objective identifiability — clause 2

**This is the measurement that decides H7, and it is model-independent by construction.** Two
predictors are built from training data alone and scored with **the objective's own rotation
term** (`rotation_chordal`), on validation:

| prediction | objective's rotation term | same, as geodesic degrees |
| --- | ---: | ---: |
| **A** — identity-only (best per-slot rotation) | 0.06238 | 23.54° |
| **B** — graph-conditioned (best per-signature-and-slot) | **0.01109** | **11.00°** |
| reduction | **82.2%** | 12.54° |

Preregistered threshold: a ≥10% reduction counts as the objective rewarding relational
conditioning. The measured reduction is **82.2%**, eight times the threshold.

**The objective massively prefers the relationship-conditioned prediction.** A model that
learned to condition on the graph would be rewarded with a large, immediate loss reduction. The
supervision exists. Clause 2 **fails**, and with it H7.

## 10. Gradient diagnostics

Frozen checkpoints, train split, three seeds, eight batches each. Gradients computed and
discarded; no optimiser step ran.

| pathway | parameters | gradient norm | per-parameter RMS | share of total norm |
| --- | ---: | ---: | ---: | ---: |
| `frame_head` | 135,180 | 8.2017 | **0.022308** | 99.3% |
| `geometry` | 173,313 | 0.8022 | 0.001927 | 9.7% |
| `entity_identity` | 13,904 | 0.0779 | 0.000660 | 0.9% |
| `graph_encoder` | 2,666,896 | 0.4777 | **0.000293** | 5.8% |
| `relation_parameters` | 144 | 0.0035 | 0.000289 | 0.0% |
| `scene_and_text` | 160,810 | 0.0417 | 0.000104 | 0.5% |
| `alignment` | 76,544 | 0.0083 | 0.000030 | 0.1% |
| all parameters | | 8.2587 | | 100% |

**Meaningful signal does reach the graph pathway — it is 76× weaker per parameter than the
frame head's.** The graph encoder is not starved of gradient (0.478, 5.8% of the total norm),
but the frame head, with one twentieth the parameters, absorbs almost all of it.

Reported, not decisive, per the plan. An earlier version of this table reported exactly 0.00000
for `entity_identity`; that was a misspelled parameter prefix, not a finding. A guard now
refuses to measure unless every prefix matches a real parameter and every parameter belongs to a
pathway.

## 11. Graph-scrambled objective test

Same batches, same targets, same frozen weights, relation graph permuted:

| quantity | mean | per seed |
| --- | ---: | --- |
| objective, correct graph | 2.4516 | — |
| objective, scrambled graph | 2.4802 | — |
| relative change in the total | **+1.17%** | +1.18, +1.28, +1.05 |
| relative change in the rotation term | +0.52% | **−0.19, +1.01, +0.75** |

The objective barely moves, and the rotation term's change **disagrees in sign across seeds**,
so that figure is noise.

**This does not show the objective is insensitive to the graph, and must not be read that way.**
The test measures the objective *through a trained model*. A model that already ignores the
graph produces nearly identical predictions under a scrambled graph, hence a nearly identical
loss — the insensitivity measured is the model's, not the objective's. §9 measures the objective
directly, without a model, and finds an 82.2% separation. The two results are consistent: the
loss would reward graph use, and this model is not using it.

## 12. Relationally-required validation subset

Selected by the rule preregistered in the plan, on validation, from tables fitted on train:
identity-only error ≥ 23.51° **and** graph-aware error ≥ 10.0° lower **and** signature seen at
least twice in training.

| | |
| --- | ---: |
| entities | **652** (28.2% of validation) |
| slots represented | 20 |
| identity-only error | 42.22° |
| graph-aware error | **10.20°** |
| available improvement | 32.02° |

Concentrated in the higher slot indices (18: 136 entities, 19: 83, 20: 64, 17: 58).

## 13. A3 on that subset — the training-objective gap

| band | n | identity-only | **A3** | graph-aware | A3 recovers |
| --- | ---: | ---: | ---: | ---: | ---: |
| all validation | 2316 | 23.51° | **21.80°** | 12.24° | 15.2% |
| relationally required | 652 | 42.22° | **36.44°** | 10.20° | **18.0%** |
| remainder | 1664 | 16.18° | **16.06°** | 13.04° | 3.8% |

A3 per-seed on the subset: **33.92, 41.56, 33.84** — seed 1, the seed Step 11 identified as
barely using the graph, sits at 41.56° against the identity-only 42.22°, i.e. essentially at the
identity floor.

**Does A3 fail specifically where relational information is required? Yes, in absolute terms.**
On the subset A3 leaves **26.24°** on the table against the graph-aware lookup; on the remainder
it leaves 3.02°. But the *share* recovered is slightly higher on the subset (18.0%) than overall
(15.2%) — A3 is not ignoring the graph where it matters most, it is capturing a modest constant
fraction of it everywhere. The failure is uniform under-exploitation, not a targeted blind spot.

## 14. H7 decision

> H7 is supported only if **both** (1) relationship-dependent target variation is present and
> measurable, and (2) the current objective provides substantially weaker supervision for that
> variation than for identity-predictable variation.

| clause | verdict | evidence |
| --- | --- | --- |
| 1. relational variation present | **satisfied** | 88.0% of target rotation spread is graph-explained |
| 2. objective supervises it weakly | **fails** | graph-conditioning reduces the objective's rotation term by **82.2%**, against a 10% threshold |

**H7 is NOT supported. H8 is supported.**

> **The current objective is relationally identifiable; the failure must arise elsewhere.**

The brief's warning applies directly and was the trap this study was built to avoid: A3 failing
to learn graph information is *not* evidence that the loss is inadequate. A3 does fail — it
leaves 26° on the table where the graph would help — and the loss would nonetheless have paid it
handsomely for succeeding.

## 15. Is an objective intervention justified?

**No.** Per §16 of the brief, no relation loss, contrastive term, auxiliary graph prediction,
pairwise loss or ranking loss is added, and
`STEP_14_OBJECTIVE_INTERVENTION_PLAN.md` is deliberately **not** created.

Adding relational supervision to an objective that already separates the two hypotheses by 82%
would be adding a signal the model is already declining to follow. It would very likely produce
another negative result, and a less interpretable one.

## 16. Limitations

* The lookup predictors condition on an **exact** relation signature. That is the right upper
  reference for available information but a generous one: it memorises 183 training graphs and
  abstains (10.2% of validation) where the signature is unseen. A model must generalise across
  signatures; the lookup need not.
* Clause 1 is measured on train, where the conditional tables are fitted. The 88% figure
  describes how much graph-conditional structure exists, not how much transfers.
* The variance decomposition uses chordal spread about a chordal mean, not a true ANOVA; it is
  the objective's own metric, which is what clause 2 needs, but the two components are not
  guaranteed to be orthogonal.
* Gradient norms are a snapshot at step 1199 on frozen weights. They describe the signal a
  converged model receives, not the whole training trajectory.
* The scrambled-graph test is confounded by the model's own graph insensitivity (§11) and is
  reported for completeness rather than as evidence.
* Everything is one corpus, one scale, three seeds, synthetic throughout.

## 17. Recommended next step

The objective is not the bottleneck. Four candidate explanations have now been eliminated:
relation encoding (Step 12), propagation depth and global context (Step 13), and the training
signal (Step 14). What remains unexplained is sharper than when Step 14 began, and it is not an
architecture question.

**The seed dichotomy is now the largest unexplained effect in the programme, and it is
measurable.** Step 11 found the graph is worth 2.59° and 2.51° to seeds 0 and 2 but 0.38° to
seed 1. Step 13 showed the mode is *manipulable*: doubling propagation moved seed 0 from the
using mode to the non-using mode. Step 14 now shows seed 1 sits at the identity-only floor
(41.56° against 42.22°) exactly on the entities where the graph would help most. Three
independent measurements of the same bimodality.

The natural next study is therefore **optimisation, not architecture or objective**: why, given
an objective that pays 82% for graph conditioning, does training reach a graph-using solution on
some seeds and an identity-only solution on others? Concretely, and all cheap:

1. **Widen the seed sample.** Three seeds cannot characterise a bimodal outcome. Ten seeds at
   the frozen configuration would establish the mode's base rate and whether it is genuinely
   bimodal or a continuum, with no architecture change and no new loss.
2. **Find when the mode is decided.** The checkpoints exist; the graph-usage measure (the
   `entities_only` ablation) can be applied to intermediate states if any were kept, or to a
   re-run instrumented to record it. If the mode is fixed early, it is an initialisation or
   warmup phenomenon, which is a different fix from a late-training one.
3. **Test whether the good mode is reachable from the bad one.** Continue a non-using seed with
   a different data order or a warmup restart, and see whether it can cross.

These are diagnostics on the existing configuration, not interventions.

**A note on §3.1(a).** The audit found two live frame terms where the ADR describes one. That
is a documentation defect rather than a bug — both terms were present in every run from Step 10
onward, so no comparison is invalidated — but the ADR should be corrected, and the redundant
L1 rotation term (3.1% of the objective) is a candidate for removal in whatever study next
touches the loss. It was deliberately **not** changed here.

**Not started.** No Step 15. No new loss. No architecture change. No real medical data.
