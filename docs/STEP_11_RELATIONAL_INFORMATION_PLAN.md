# STEP 11 — RELATIONAL INFORMATION PLAN

**Preregistered before any Step 11 measurement was taken.** Diagnostic study; no architecture
is changed in this pass, and no model is trained unless a listed experiment requires it.

All data is procedurally generated. Nothing here is anatomy or is medically validated.

---

## 1. The question

A lookup keyed on entity identity **and the relationship graph** reaches 11.57° of rotation
error on `test_seen`. The trained A3 model reaches 21.38°. Both read the same inputs. Why is
there a 9.80° gap?

This pass locates where relational information is lost along:

```
relationship graph -> graph representation -> message passing -> entity latent -> frame head
```

It does **not** try to close the gap, and it does not make the model larger.

---

## 2. Phase 0 audit: what the code actually does

Read from the implementation, not the documentation.

### The A3 computation path

```
structure (per scene)
  -> EntityLatentComposer          identity, anatomy type, semantic role, laterality
  -> entity_in            [B, N, 256]
  -> PartitionedGraphEncoder       4 layers, graph_scope="all"
  -> entity_latent        [B, N, 256]
  -> FramePredictor                LayerNorm, 2x(Linear 256->256 + GELU), Linear 256->12
  -> frame                [B, N, 12]   3 translation, 3 log scale, 6D rotation
```

| audit item | finding |
| --- | --- |
| graph representation | per-scene adjacency `[B, 3 kinds, N, N]` plus edge lists: source, target, relation id, graph kind |
| which edges reach the model | the ontology's STRUCTURE edges **and** the scene's measured edges (spatial, functional), concatenated per scene |
| message-passing depth | **4** attention layers |
| relation encoding | `nn.Embedding(18, heads.total)` giving an **additive scalar bias per head** on the attention logits; the reverse direction uses the inverse relation's embedding |
| entity conditioning | identity slice, the first **96** of 256 channels, is passed through every graph layer **unchanged**; attention writes only to the 160-wide context slice |
| geometry-head conditioning | reads the full 256-wide latent; `frame_scene_context` is **off** |
| where identity enters | the composer, before any graph layer |
| where relations enter | the attention **mask** (who may attend to whom) and the relation **bias** (how strongly) |
| can graph information reach every predicted entity? | yes, through the context slice; nothing masks it out for A3 (`graph_scope="all"`; the structure-only masking belongs to the A2 ablation) |
| is graph information discarded before the head? | no |

### Two discrepancies, recorded rather than reconciled

1. **`relation_width` is dead in the typed encoder.** `GraphEncoderConfig.relation_width = 64`
   is never read by `graph.py`, and `generation/neural/scales.py` bills
   `relation_width * graph_heads` for a "relation to attention-bias projection" that the code
   does not contain — it uses a direct `Embedding(vocabulary, heads)`. Expressiveness is
   equivalent, so no result is affected, but the parameter model and the code disagree.
2. **A relation cannot change what a neighbour says, only how loudly.** Values are not
   conditioned on relation type. A relation contributes `heads.total` scalars to attention
   logits and nothing else. The model can route relation types to different heads; it cannot
   give a relation its own message.

Finding 2 is not a defect — it is the design — but it is the sharpest structural candidate
for the gap, and H3 below is written against it.

---

## 3. Frozen references, reproduced from artifacts

Verified before any new measurement, without retraining or regenerating anything:

| quantity | value |
| --- | --- |
| A3 rotation, `test_seen`, λ=3.0 | 21.38° (seeds 20.78, 20.79, 22.57; std 1.03) |
| identity-only rotation floor | 22.65° |
| identity + graph lookup | 11.57° (92% of test entities have their graph seen in training) |
| the gap | 9.80° |

All four match the Step 10 record exactly.

---

## 4. Hypotheses

Stated before measurement. None is assumed.

* **H1 — the information reaches the latent but the head does not use it.** The entity latent
  after the graph encoder carries the rotation-relevant relational signal, and the frame head
  fails to extract it.
* **H2 — insufficient propagation.** Four layers of local attention do not carry enough of the
  scene's graph to each entity.
* **H3 — relation type is weakly used.** The model benefits from knowing *which entities* are
  connected but cannot exploit *which relation* connects them, because a relation is a scalar
  attention bias and never a message.
* **H4 — identity dominates.** The model places entities mostly by identity and uses the graph
  only weakly; the protected 96-channel identity slice is a candidate mechanism.
* **H5 — optimisation variance.** Part of the gap is the 1.03° seed instability rather than a
  representational limit.
* **H6 — the target is scene-global and the encoder has no global pooling.** Ten of twenty
  entities share the organ's rotation exactly, which is a property of the whole scene. The
  graph encoder has no global pooling anywhere, and `frame_scene_context` is off. The lookup
  gets the scene identity for free by keying on the whole graph signature. Step 9 tested the
  scene-context flag and rejected it — but it did so on the identity-rotation corpus, where
  the rotation target was structurally zero and the flag could only add noise.

H6 was added during the Phase 0 audit, before any Step 11 measurement, and is recorded here
as a Step 11 hypothesis rather than presented later as a finding.

---

## 5. The experiment matrix

The smallest set that separates the hypotheses. **Neither experiment retrains the model**;
both read the three frozen A3 checkpoints at λ=3.0.

### E1 — graph ablation at evaluation (Step 7 methodology)

Perturb the relationship graph in the batch, re-evaluate, measure the change in rotation
error. This measures what the trained model **uses**, not what it could use.

| condition | what it removes | brief's label |
| --- | --- | --- |
| `intact` | nothing | C |
| `drop_spatial` | the measured spatial relations | — |
| `drop_functional` | the functional relations | — |
| `drop_structure` | the ontology's structural edges | — |
| `shuffle_spatial_types` | relation **type**, endpoints kept | B and the Phase 6 type swap |
| `randomise_spatial_endpoints` | endpoint **identity**, types kept | D |
| `entities_only` | the whole graph | A |

`randomise_spatial_endpoints` doubles as the Phase 4 E permutation control: it preserves the
tensor structure and destroys the semantic correspondence.

### E2 — representation probe

Train a small probe on **frozen** latents to predict the entity's true rotation, and compare
two taps on the same forward pass:

* `entity_in` — before the graph encoder. Identity only. The control.
* `entity_latent` — after the graph encoder. What the frame head actually reads.

If the graph encoder writes relational information into the latent, the second probe must beat
the first. If the information is there and the head is not using it, the second probe must
beat A3's own 21.38°.

---

## 6. Probe methodology

The repository has no representation-probe precedent, so the method is fixed here.

* **Inputs** are taken from a frozen checkpoint in `eval()` with `torch.no_grad()`; no
  gradient reaches the model, and no model weight changes.
* **Target** is the entity's true rotation, as the 6D representation, scored by the same
  geodesic angle the reports use.
* **Probe** is the smallest thing that can answer the question: a linear map, and a 2-layer
  MLP of width 256 as the non-linear counterpart. Both are reported, because a linear probe
  failing means "not linearly available", not "absent".
* **Fitted on `train`, reported on `validation`.** The test split is not read.
* **Trained** with AdamW, 2000 steps, batch 64, learning rate 1e-3, one probe per tap per
  seed, using the same three seeds as the checkpoints.
* A probe result is **evidence of presence, never of use.** A high probe score with a low
  model score is exactly the H1 signature, and is reported as diagnostic evidence.

---

## 7. Decision rule

Fixed before results are inspected.

* **Primary metric:** mean geodesic rotation error in degrees, on **validation**, over the
  three frozen seeds.
* **Baselines:** A3 as trained; the identity-only floor; the graph-aware lookup as a
  diagnostic reference, *not* a target A3 must reach.
* **Minimum meaningful difference: 1.0°.** Chosen because A3's own seed spread is 1.03°, so
  anything smaller cannot be distinguished from the instability already on record.
* **Seed consistency:** all three seeds must agree in sign. With three seeds and no
  repetition, sign agreement is the statistic; the Change 1 rule (|t| > 4.303) is applied
  where a paired comparison is available.
* **Inconclusive** is a real outcome and will be reported as such: a difference under 1.0°, or
  seeds disagreeing in sign, supports nothing.

### What each result would mean

| observation | supports |
| --- | --- |
| destroying the graph barely moves rotation error | H4: the model is not using the graph |
| dropping relation **type** barely moves it, dropping endpoints does | H3 |
| probe on `entity_latent` ≈ probe on `entity_in` | H2 or H3: the graph is not writing into the latent |
| probe on `entity_latent` beats A3's own error by ≥1.0° | H1: present but unused |
| probes and ablations both flat, seeds inconsistent | H5, or the information is not recoverable at all |

---

## 8. Test discipline

`test_seen`, `test_arrangement`, `test_transform` and `test_combination` are not read in this
pass. Every Step 11 number is on `validation`, except the frozen Step 10 references quoted
above, which are already public record. No architecture is selected, no depth is chosen and no
hyperparameter is tuned against test data.

## 9. Stop conditions

Stop and report rather than continuing if the frozen numbers fail to reproduce, if leakage is
found, if an experiment would require touching the corpus, λ=3.0, the rotation target or the
objective — or **if the smallest diagnostic already answers the question**, which the brief
names explicitly and which E1 may well do on its own.
