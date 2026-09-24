# Step 13 — Graph propagation depth vs global graph context: report

**Preregistered in** [`STEP_13_GRAPH_DEPTH_GLOBAL_CONTEXT_PLAN.md`](STEP_13_GRAPH_DEPTH_GLOBAL_CONTEXT_PLAN.md),
written before any Step 13 model was trained.

**Status: both hypotheses unsupported.** Neither increased propagation depth nor global graph
context explains the remaining relational-information gap. Reported as measured.

**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Research question

Is the missing relational information caused by insufficient graph propagation depth (**H2**), or
because the entity representation lacks access to global scene-level graph context (**H6**)?

Step 13 is diagnostic. It began with neither hypothesis assumed.

## 2. Frozen Step 11 evidence

Removing all relationships costs A3 about **1.83°**. Relation-type destruction was negligible
(0.0003°); endpoint identity contributed very little; latent probing showed limited relational
information entering the entity latent; the geometry head was not the bottleneck; the
relation-type attention bias was nearly inert at 0.55% of the logit scale. H3 was supported, H1
was not, and **H2 and H6 were left untested**.

Step 11 also found A3's seed variance is *bimodal*, and the mode is graph usage: the whole graph
is worth 2.59°, 0.38° and 2.51° to seeds 0, 1 and 2.

## 3. Frozen Step 12 negative result

Relation-conditioned values did not help: control 21.80° → treatment 21.81° on validation, mean
gain −0.013°, seeds disagreeing in sign, against a 1.0° threshold. The treatment was implemented
correctly, its parameters trained off zero, and the relation message settled at 0.33% of the
attended value — the same order as the 0.55% scalar it replaced. Not reopened here.

## 4. Repository audit

Read from code. Two traps were found; neither is a code/documentation contradiction, and both
would have corrupted this study silently.

**Trap 1 — two rotation metrics that differ by half the decision threshold.** Every run record
carries both:

| field | `test_seen` per seed | mean |
| --- | --- | ---: |
| `rotation.test_seen.mean_deg` | 20.79, 22.57, 20.78 | **21.38** |
| `splits.test_seen.inferred.rotation_error` (×180/π) | 21.29, 23.14, 21.28 | 21.90 |

The frozen references, Step 11 and Step 12 all use the **`rotation` block**. Step 13 does too.

**Trap 2 — global pooling already exists, and is switched off.** See §7.

**No discrepancies** were found between code and documentation in the Step 11 probe, the Step 12
relation-value implementation, the ablation utilities, the runner, or the integrity gate.

## 5. Exact A3 architecture

```
entity ids ─► entity_composer ─► entity_latent_in                    [B, N, 256]
                                        │
                     PartitionedGraphEncoder, graph_layers = 4
                       per layer: LayerNorm ─► 8-head masked attention
                                  (+ additive scalar relation bias)
                                  ─► out proj ─► residual into context
                                  ─► LayerNorm ─► MLP(×4) ─► residual into context
                       identity slice (channels 0–95) copied through, never written
                                        │
                                  entity_latent                      [B, N, 256]
                                        │
                     FramePredictor (2 layers, hidden 256)
                                        │
                       12 = 3 translation + 3 log-scale + 6D rotation
```

* **Four message-passing layers already.** A signal travels at most four graph hops.
* Heads scoped by graph kind: `head_graph = [0,0,1,1,2,2,−1,−1]`.
* Pre-norm residuals; both sublayers write only to the context slice (96–255).
* **No graph readout. No pooling anywhere inside the encoder.**
* `relation_values = False` throughout, as required.

## 6. H2 definition

The four-layer stack runs **twice, sharing weights**: `graph_recurrence = 2`, propagation steps
**4 → 8**, **+0 parameters**.

Weight tying rather than added layers, because two untied layers cost **+1,333,448 (+41.3%)**
against H6's +2.05%. Under §10 of the brief that asymmetry forces capacity matching and a stop;
worse, a gain would be uninterpretable, since A3's seed spread (~1.0°) is the size of the effect
being measured. Weight tying makes propagation depth the only variable.

**Limitation, stated in advance:** recurrence reuses one transformation, so it tests propagation
*reach*, not additional distinct transformations.

## 7. H6 definition

`frame_scene_context = True`. The frame head additionally reads a presence-masked mean of the
post-graph entity latents. **+66,048 parameters (+2.05%)**, all inside `frame_head`.

**This pathway already existed and was disabled.** `scene_summary()` is implemented and wired,
gated on `frame_scene_context`, which the Step 10 config sets false — so A3 has had no global
pooling in Steps 10–12. It was added in **Step 9 as hypothesis P4** with nearly identical
reasoning, and Step 9's verdict was **UNSUPPORTED on placement** (translation 0.1615 vs 0.1599),
with one clean exception on `test_combination` (IoU 0.0343 → 0.0420).

**Why this is still a valid test:** Step 9 ran on the *unrotated* corpus, where every rotation
target was the identity. Step 9 could not measure rotation at all. Enabling an existing, tested
pathway is also strictly smaller than writing a new one.

## 8. Preregistered predictions

**H2:** rotation improves ≥1.0°; relational information in the entity latent increases; graph
ablations become more discriminative than the control's 1.83°.

**H6:** rotation improves ≥1.0°; the global context contains graph-configuration information
above chance; entity representations become more sensitive to scene-level relational change.

**Null:** neither improves meaningfully — recorded in advance as a live and unremarkable outcome.

## 9. Experiment matrix

| arm | `graph_recurrence` | `frame_scene_context` | seeds |
| --- | ---: | --- | --- |
| A3 (frozen control) | 1 | false | 0,1,2 |
| A3-depth | **2** | false | 0,1,2 |
| A3-global | 1 | **true** | 0,1,2 |

Six runs trained; the control is the frozen Step 10 Change 2 λ=3.0 set, rescored here.

## 10. Parameter counts

| arm | total | delta | location |
| --- | ---: | ---: | --- |
| A3 | 3,226,391 | — | — |
| A3-depth | 3,226,391 | **+0 (+0.00%)** | none; weights reused for a second pass |
| A3-global | 3,292,439 | **+66,048 (+2.05%)** | `frame_head.trunk.0.weight` (+65,536), `frame_head.context_norm.*` (+512) |

The two treatments differ greatly in parameter count. This is recorded, not corrected; no
artificial capacity matching was introduced. Since **neither** treatment improved, the asymmetry
does not affect the conclusion.

## 11. Training configuration

Identical across arms and identical to the control's original run: rotated corpus
`datasets/processed/step10_rotated`, chordal objective, λ=3.0, 1200 steps, batch 8, AdamW with
the Step 9 schedule, CPU, three seeds. No hyperparameter tuned per arm; no arm trained longer.

**Initialisation isolation.** `fork_rng` is insufficient here because H6 *reshapes*
`trunk.0.weight`; without correction, **37 shared tensors differ**, including the geometry
tokeniser. `generation/neural/nn/init_alignment.py` builds both models at the same seed and
copies every shared name-and-shape tensor from control into treatment, reading the control only.

| arm | copied | shared still differing | intended differences |
| --- | ---: | ---: | --- |
| A3-depth | 141 | **0** | none |
| A3-global | 140 | **0** | exactly the three `frame_head` tensors |

## 12. Validation results

Primary metric, mean geodesic rotation error:

| arm | rotation | seed 0 | seed 1 | seed 2 | position | scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A3 | **21.80** ±1.41 | 20.96 | 23.43 | 21.01 | 0.1522 | 0.1195 |
| A3-depth | **22.64** ±1.42 | 23.43 | 23.49 | 21.00 | 0.1503 | 0.1196 |
| A3-global | **21.88** ±1.48 | 21.01 | 23.58 | 21.03 | 0.1579 | 0.1199 |

| arm | per-seed gain | mean gain | seeds agree | supported |
| --- | --- | ---: | --- | --- |
| A3-depth | −2.47, −0.06, +0.01 | **−0.84** | no | **no** |
| A3-global | −0.06, −0.15, −0.02 | **−0.08** | yes | **no** |

Both are *worse*. A3-depth's deficit is one seed: seed 0 fell from 20.96° to 23.43°, essentially
onto the 23.51° validation floor.

## 13. Confirmatory test results

One evaluation, after the matrix was frozen and validation recorded. **The control reproduces
the frozen reference exactly: 21.38 ± 1.03 on `test_seen` (20.79, 22.57, 20.78).**

| split | A3 | A3-depth | A3-global |
| --- | ---: | ---: | ---: |
| `test_seen` | **21.38** ±1.03 | 21.99 ±1.06 (−0.61, seeds disagree) | 21.44 ±1.07 (−0.06) |
| `test_arrangement` | 21.13 ±1.10 | 21.78 ±1.13 (−0.65, disagree) | 21.21 ±1.16 (−0.09) |
| `test_transform` | 35.51 ±0.23 | 35.64 ±0.28 (−0.14, disagree) | 35.59 ±0.23 (−0.09) |
| `test_combination` | 30.43 ±2.62 | 28.98 ±2.69 (**+1.45**, disagree) | 29.76 ±2.78 (+0.67) |

**No arm is supported on any split.** A3-depth exceeds 1.0° on `test_combination` but the seeds
disagree (+4.50, −0.08, −0.07) — the entire effect is seed 0 again, and the rule requires both
conditions. Test results changed nothing; they are confirmatory only.

## 14. Rotation results

Against the frozen floors — identity-only **22.65°**, graph-aware lookup **11.57°** — on
`test_seen`: A3 21.38°, A3-depth 21.99°, A3-global 21.44°. All three beat the identity-only
floor; all three remain about 10° above the graph-aware lookup. **Step 13 closed none of that
gap.**

## 15. Position results

| split | floor | A3 | A3-depth | A3-global |
| --- | ---: | ---: | ---: | ---: |
| `test_seen` | 0.1605 | 0.1561 | **0.1549** | 0.1600 |
| `test_combination` | — | 0.2175 | 0.2252 | **0.1978** |

A3-global is consistently *worse* on position on the ordinary splits (validation 0.1579 vs
0.1522; `test_seen` 0.1600 vs 0.1561). **This replicates Step 9's P4 result** — scene context
made translation worse there too (0.1615 vs 0.1599) — now on the rotated corpus.

**And Step 9's exception replicates as well.** On `test_combination`, A3-global improves position
from 0.2175 to **0.1978**, with all three seeds better (0.2113→0.2098, 0.2272→0.1964,
0.2140→0.1871). Step 9 saw the same direction (0.2092 → 0.1943). This is a secondary metric on
one split and is **not** a Step 13 finding about rotation; it is recorded because it is the
second independent replication of the same effect.

## 16. Scale results

Essentially unmoved: `test_seen` floor 0.1120, A3 0.1106, A3-depth 0.1107, A3-global 0.1110.
Neither treatment changes scale.

## 17. Depth latent-probe results

Step 11's probe methodology exactly, fitted on train, scored on validation. Lower degrees means
the rotation is more recoverable from the tap.

| arm | `entity_in` (control tap) | `entity_latent` | graph encoder gain |
| --- | ---: | ---: | ---: |
| A3 | 23.89 | 22.17 | **+1.71** |
| A3-depth | 23.89 | 22.77 | **+1.12** |

**Doubling propagation put less relational information into the latent, not more: −0.59°.** The
H2 prediction fails in the wrong direction. There is no latent-information-rises-but-geometry-
does-not dissociation to record, because the latent information did not rise.

## 18. Global-context probe results

Relative error against a mean predictor (1.0 = learned nothing), with a permutation control —
the same probe on scene summaries shuffled across scenes. `entity_rotation` is in degrees.

| target | real | permuted | beats control, all seeds |
| --- | ---: | ---: | --- |
| graph arrangement | **0.555** | 1.172 | yes |
| relationship configuration | **0.724** | 1.216 | yes |
| scene transformation | 0.945 | 1.050 | yes, marginally |
| entity placement | 0.993 | 1.045 | barely |
| entity rotation | 61.68° | 61.97° | **no** |

**The global context does contain the graph's configuration.** Arrangement and relation
composition are both well clear of the permutation control. **It contains almost nothing about
the geometry:** transformation and placement sit at the mean-predictor baseline, and per-entity
rotation is no better than shuffled — expected, since one vector per scene must predict the same
rotation for every entity in it.

So H6's information clause holds while its outcome clause fails. **A probe measures presence,
never use:** this shows the context carries graph information, not that the model uses it.

## 19. Graph ablations

Validation, cost in degrees of damaging the graph at evaluation on frozen weights.

| condition | A3 | A3-depth | A3-global |
| --- | ---: | ---: | ---: |
| intact | 21.80 | 22.64 | 21.88 |
| entities_only | 23.62 (**+1.83**) | 24.29 (**+1.66**) | 23.51 (**+1.64**) |
| drop_spatial | 21.76 (−0.04) | 22.65 (+0.02) | 21.88 (+0.00) |
| drop_structure | 21.79 (−0.00) | 22.69 (+0.05) | 21.86 (−0.01) |
| drop_functional | 21.95 (+0.15) | 22.92 (+0.29) | 22.03 (+0.15) |

The control reproduces Step 11's **+1.83°** exactly. **H2's third prediction fails:** the whole
graph is worth *less* under deeper propagation (1.66°), not more. H6's sensitivity clause fails
too (1.64°).

**An incidental structural finding:** no individual typed graph matters in any arm — dropping
spatial, structure or functional costs between −0.04° and +0.29°. Only removing *all*
relationships costs anything. Whatever the graph contributes is redundant across its three kinds.

## 20. Seed analysis

Per-seed cost of removing the whole graph — Step 11's "is this seed using the graph at all":

| arm | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| A3 | +2.59 | +0.38 | +2.51 |
| A3-depth | **+1.15** | +0.59 | +3.22 |
| A3-global | +2.49 | +0.12 | +2.29 |

**Neither treatment reduced A3's seed instability.** Spreads are unchanged or slightly wider
(validation: 1.41 → 1.42 depth, 1.48 global).

The clearest single result in Step 13 is what depth did to **seed 0**: rotation 20.96° → 23.43°
while its graph usage fell 2.59° → 1.15°. Step 11 identified seed 0 as one of the two
graph-using seeds; doubling propagation moved it toward the non-using mode. Extra propagation did
not add relational information — it cost a graph-using seed its graph usage. Seeds 1 and 2 were
substantially unchanged (−0.06, +0.01).

## 21. Integrity results

* Corpus freeze: **7/7 files OK**; Change 1 freeze **OK**; Step 12 freeze **OK** — before and
  after.
* The integrity gate (`validate_placement`, six checks) ran on every training run and every
  scored split, with all RNG state saved and restored.
* Frozen A3 checkpoints load `strict=True` into the Step 13 architecture and reproduce 20.9558°
  on validation and 21.38° on `test_seen` — the Step 13 code is inert when switched off.
* Initialisation alignment verified per run and recorded in each run manifest.
* **Recurrence verified to take effect at evaluation.** Because recurrence adds no parameters, a
  `strict` load would succeed at the wrong setting. Scored explicitly: the depth seed-0
  checkpoint gives 23.4286° at `recurrence=2` (matching the harness) and 23.6965° at
  `recurrence=1`. The correct setting was used, and the depth model is worse either way.

## 22. Leakage results

* **No test split was read during training or selection.** All six runs used
  `--splits validation`; every treatment run record carries `protocol.splits == ["validation"]`.
  The test splits were read once, after the matrix was frozen and validation was recorded.
* Probes fitted on **train** only, scored on validation; no probe touched test.
* All scoring uses `use_predicted_frames=True`; no true frame reaches a prediction.
* **Tested for both new pathways:** replacing `entity_frames` with noise leaves each treatment's
  predicted frames bit-identical, and leaves the scene summary bit-identical. The summary also
  ignores padded slots, so it cannot vary with batch packing.

## 23. Comparison with the identity-only lookup

Floor **22.65°** on `test_seen`. A3 beats it by 1.27°, A3-depth by 0.66°, A3-global by 1.21°.
Every arm still beats a lookup keyed on identity alone; the depth treatment beats it by half as
much as the control does.

## 24. Comparison with the graph-aware lookup

Reference **11.57°** on `test_seen`. The gap from A3 is 9.81°; from A3-depth 10.42°; from
A3-global 9.87°. **Step 13 closed none of it, and the depth arm widened it.** The lookup remains
a diagnostic upper reference for available relational information, not a target.

## 25. Decision-rule application

> A treatment is supported only if (a) rotation improves by ≥1.0°, (b) all three seeds move the
> same way, (c) its mechanism shows measurable evidence, and (d) integrity and leakage pass.

| | A3-depth | A3-global |
| --- | --- | --- |
| (a) ≥1.0° improvement | **no** — −0.84 validation, −0.61 `test_seen` | **no** — −0.08 validation, −0.06 `test_seen` |
| (b) seeds agree | **no** | yes (all three slightly worse) |
| (c) mechanism evidence | **no** — latent information fell 0.59°, graph worth fell to 1.66° | **partly** — context carries graph configuration, but sensitivity fell to 1.64° |
| (d) integrity and leakage | yes | yes |

**H2: not supported.** All three of its predictions fail, two in the wrong direction.

**H6: not supported.** Its information clause holds; its outcome and sensitivity clauses fail.

Both treatments are worse than the control, so no ambiguity needs resolving between them.

## 26. Interpretation

Clearly separated by kind:

**Measured.** Neither treatment improves rotation on validation or on any test split. Depth is
0.84° worse on validation, driven by one seed. Global context is 0.08° worse, consistently.
Depth reduces recoverable relational information in the latent by 0.59° and reduces what the
whole graph is worth from 1.83° to 1.66°.

**Diagnostic evidence.** The scene summary demonstrably contains the graph's configuration
(arrangement 0.555, relation composition 0.724, both clear of permutation) and demonstrably does
not contain the geometry (transformation 0.945, placement 0.993, per-entity rotation no better
than shuffled). No single typed graph matters in any arm; only removing all relationships costs
anything.

**Inference.** H6's failure is now informative in a way Step 12's was not: the information was
*present and available at the prediction site*, and the model still gained nothing. That is a
different failure from "the channel was never written" (Step 11) or "a richer channel stayed
tiny" (Step 12). Here the channel is wide, populated, and adjacent to the head — and it does not
help. Depth's failure is different again: more propagation of a signal that carries little
relational content mostly destabilised the seeds that had found some.

**Architectural interpretation, offered as interpretation and not as result.** Three independent
interventions — a richer relation encoding, more propagation, and global context — have now each
failed to move rotation, while a lookup with the same inputs reaches 11.57°. The common factor is
not any one pathway's design. The parsimonious reading is that the *training signal* does not
reward relational information through any of these routes, rather than that the routes are
individually inadequate. Step 12 already noted this after two parameterisations of relation type
converged on under 1% of their signal scales; Step 13 adds two more routes to the same pattern.
**This is a hypothesis for a later step, not a Step 13 finding, and nothing here tests it.**

## 27. Limitations

* **Recurrence is not untied depth.** It tests propagation reach with one shared transformation.
  Eight untied layers might behave differently, at +41% parameters and a capacity confound.
* **One depth value and one context mechanism.** No sweep was run, by design.
* The parameter counts are asymmetric (0 vs 66k). Since neither helped, this does not affect the
  conclusion, but a *positive* H6 could not have been separated from capacity by this matrix.
* Probes measure presence, never use. The global context containing graph configuration is not
  evidence the model uses it.
* `test_combination` effects are on a secondary metric, on one split, with wide seed spreads
  (±2.6–2.8° on rotation); they are reported, not claimed.
* Everything is one corpus, one scale, 1200 steps, three seeds. Synthetic throughout.

## 28. Recommended next step

**Stop and reassess rather than add complexity**, as §22 of the brief requires when neither
treatment is supported.

The evidence does not point at another architectural pathway. Three have now failed. What has
never been varied across Steps 11–13 is the **objective**: whether the chordal rotation loss at
λ=3.0, applied per entity, rewards a model for using its neighbours at all. The graph-aware
lookup shows the information is sufficient; four experiments show the architecture can carry it;
none shows the training signal asks for it.

The natural next diagnostic is therefore a **training-signal study, not an architecture study** —
for example, whether a model trained with a relationally-structured target (or with entities
scored conditional on neighbours) uses the graph more, measured with the Step 11 ablation. That
would test the common factor rather than a fourth pathway.

Two cheaper things are also worth doing before any of that, and neither expands the matrix:

1. **The seed-0 instability is now reproducible and manipulable.** Depth reliably moved one
   graph-using seed into the non-using mode. Whatever distinguishes the two modes is the largest
   single effect anywhere in Steps 11–13 (2.5° versus 0.4°), and it is currently unexplained.
2. **The redundancy across typed graphs** (§19) means the three-graph partition may be buying
   nothing. That is a simplification question, cheap to test.

**Not started.** No Step 14, no architecture change, no real medical data, no editing.
