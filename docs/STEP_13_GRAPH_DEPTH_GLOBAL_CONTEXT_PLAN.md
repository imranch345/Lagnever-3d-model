# Step 13 — Graph propagation depth vs global graph context: preregistration

**Status:** preregistered. Written before any Step 13 model was trained.
**Question:** is the remaining relational-information gap caused by insufficient graph
propagation depth (H2), or by the entity representation lacking global graph context (H6)?
**Synthetic research data. Not anatomy, not validated, not clinical.**

---

## 1. Frozen references, reproduced from artifacts

Verified by re-reading the run records, not copied from prose:

| reference | value | source |
| --- | ---: | --- |
| A3 rotation, `test_seen`, λ=3.0 | **21.38° ± 1.03** (20.79, 22.57, 20.78) | `rotation.test_seen.mean_deg` |
| identity-only rotation floor | **22.65°** | `rotation.test_seen.floor_deg` |
| identity + graph lookup | **11.57°** | Step 10 Change 2 report §the lookup table |
| A3 rotation, validation, `intact` | **21.80°** (20.96, 23.43, 21.01) | Step 11 ablation |

**Integrity, checked before planning:** corpus freeze 7/7 OK, Change 1 freeze OK, Step 12 freeze
OK, working tree clean, no background processes.

**The control reproduces exactly.** A frozen A3 seed-0 checkpoint loads `strict=True` into the
architecture as modified for Step 13 and scores **20.9558°** on validation, matching Step 11's
20.96. The Step 13 code changes are inert when switched off.

### 1.1 A metric trap, documented

Each run record carries **two** rotation numbers and they disagree by about 0.5°:

| field | seed 0/1/2 on `test_seen` | mean |
| --- | --- | ---: |
| `rotation.test_seen.mean_deg` | 20.79, 22.57, 20.78 | **21.38** |
| `splits.test_seen.inferred.rotation_error` (×180/π) | 21.29, 23.14, 21.28 | 21.90 |

The frozen references, Step 11 and Step 12 all use the **`rotation` block**, which scores
geodesic degrees over the 2,300 present entities. The `splits.…inferred` field is a different
aggregate. Step 13 uses the `rotation` block throughout. This is recorded because reading the
wrong field would silently shift every number by ~0.5° — half the decision threshold.

---

## 2. Repository audit — the actual computation path

Read from code, not documentation.

```
entity ids ─► entity_composer ─► entity_latent_in          [B, N, 256]
                                      │
                                      ▼
           PartitionedGraphEncoder (graph_layers = 4)
             per layer:  LayerNorm ─► 8-head masked attention (+ relation bias)
                         ─► out proj ─► residual into context slice
                         ─► LayerNorm ─► MLP(×4) ─► residual into context slice
             identity slice (channels 0–95) copied through, never written
                                      │
                                      ▼
                                entity_latent               [B, N, 256]
                                      │
                                      ▼
                 FramePredictor (deep_frame_head = True, 2 layers, hidden 256)
                                      │
                                      ▼
                          12 = 3 translation + 3 log-scale + 6D rotation
```

Audited facts, each verified live:

* **`graph_layers = 4`** — A3 already has four message-passing layers. Every signal can travel
  at most four graph hops.
* **Heads are scoped by graph kind**: `head_graph = [0,0,1,1,2,2,−1,−1]` — two structure, two
  spatial, two functional, two global (unscoped).
* **Relation type enters as an additive scalar attention bias only.** Step 12 added an optional
  value pathway; it is **off** for Step 13 (`relation_values = False`), as required.
* **Residuals and normalisation are pre-norm**, and both sublayers write only to the context
  slice (channels 96–255). The identity slice is protected by construction.
* **There is no graph readout.** No pooling of any kind happens inside the graph encoder.
* **`frame_scene_context = False` for A3.** This is the finding that shapes H6 — see §2.1.
* Parameter groups (control): entity_composer 13,904; graph_encoder 2,666,896; scene_encoder
  160,554; frame_head 135,180; geometry_tokens 101,888; field_decoder 71,425; alignment 76,544.

### 2.1 Global pooling already exists in the code, and is switched off

`PrototypeModel.scene_summary()` computes a presence-masked mean of the post-graph entity
latents, and `predict_frames()` feeds it to the frame head — **but only when
`frame_scene_context` is true, and the Step 10 config sets it false.** So A3 has no global
pooling anywhere, and the pathway is dead code for every run in Steps 10–12.

**It has been tested before.** It was added in Step 9 as hypothesis **P4**, with reasoning
almost identical to H6's — the comment in `geometry.py` reads: *"the frame target is a
scene-global centroid, while the head reads a latent whose graph attention reaches roughly four
neighbours and has no global pooling anywhere."* Step 9's verdict:

| Step 9, `test_seen` | translation | entity IoU |
| --- | ---: | ---: |
| S0 control | 0.1599 ± 0.0040 | 0.0902 |
| S1 scene context | 0.1615 ± 0.0022 | 0.0875 |

**P4 was UNSUPPORTED on placement** — slightly worse, inside the noise. It did produce a clean
gain on `test_combination` (IoU 0.0343 → 0.0420, non-overlapping spreads).

**Why this is still a valid H6 test and not a repeat.** Step 9 ran on the *unrotated* corpus:
every rotation target was the identity, so Step 9 could not measure rotation at all. Step 13
asks about **rotation**, on the rotated corpus, under the chordal objective at λ=3.0. That
question has never been asked. What Step 9 does give us is a prior: this mechanism did not help
translation, so a translation gain here would be the surprise, not a rotation gain.

**Consequence for the design:** H6 needs no new architecture. Enabling an existing, already
tested pathway is strictly smaller than writing a new one, which is what §9 of the brief asks
for.

### 2.2 No code/documentation discrepancies found

The Step 11 probe (`experiments/step11/representation_probe.py`), the Step 12 relation-value
implementation, the graph ablation utilities and the integrity gate all match their
documentation. The two items worth recording are the metric trap (§1.1) and the dormant
scene-context pathway (§2.1); neither is a contradiction, both are traps.

---

## 3. Control

The **frozen A3 λ=3.0 runs from Step 10 Change 2** (`experiments/runs/step10-change2-w3`),
re-scored through the same evaluation path as the treatments. Not retrained: retraining the same
configuration to the same seeds would produce the same weights and consume three runs to prove
determinism that is already pinned by test.

Held identical across all arms: corpus, splits, three seeds (0, 1, 2), λ=3.0, rotation target,
chordal objective, optimiser, LR schedule, batch size, 1200 steps, checkpoint policy, geometry
head, entity representation, protected identity slice, `relation_values=False`.

---

## 4. H2 — the depth treatment

**Change:** the four-layer graph stack runs **twice**, sharing weights.
`graph_recurrence = 2`, so propagation steps go **4 → 8**.

**Parameter delta: +0 (+0.00%).**

**Why weight-tied rather than more layers.** Adding two untied layers costs **+1,333,448
parameters (+41.3%)** against H6's +2.05%. §10 of the brief requires that capacity increases not
be hidden and that capacity matching, if needed, triggers a stop before expanding the matrix. A
+41% arm compared against a +2% arm would need exactly that. Worse, a positive result would be
uninterpretable: A3's seed spread (~1.0°) is the size of the effect being measured, so "depth
helped" and "capacity helped" could not be separated. Weight tying makes propagation depth the
*only* variable, which is what §8 asks for.

**What this cannot establish:** that eight *untied* layers would also fail. Recurrence reuses one
transformation, so it tests propagation reach, not additional distinct transformations. If H2 is
supported, an untied follow-up is required — and belongs to a later step, not this one.

---

## 5. H6 — the global context treatment

**Change:** `frame_scene_context = True`. The frame head additionally reads a presence-masked
mean of the post-graph entity latents (`scene_summary`), concatenated to its input.

**Parameter delta: +66,048 (+2.05%)** — 3,226,391 → 3,292,439, all inside `frame_head`
(135,180 → 201,228): `trunk.0.weight` widens from 256 to 512 inputs (+65,536) and a
`context_norm` LayerNorm is added (+512).

This is the smallest available global-context pathway: one masked mean, no new module, no
attention, no graph-encoder change. Relation type is deliberately **not** part of it, so this is
not another relation encoding (§23).

---

## 6. Parameter accounting

| arm | total | delta vs A3 | where |
| --- | ---: | ---: | --- |
| A3 (control) | 3,226,391 | — | — |
| A3-depth | 3,226,391 | **+0 (+0.00%)** | none; weights reused for a second pass |
| A3-global | 3,292,439 | **+66,048 (+2.05%)** | `frame_head.trunk.0.weight`, `frame_head.context_norm.*` |

The two treatments have very different parameter counts (0 vs 66k). This is recorded rather than
corrected: no artificial capacity matching is introduced, because the asymmetry runs *against*
H6 being favoured by capacity only if H6 wins, and the honest reading in that case is stated in
§11.

---

## 7. RNG isolation

Step 12's discipline, extended because `fork_rng` is insufficient here.

* **A3-depth** adds no parameters and consumes no RNG: measured, **0 of 141 shared tensors
  differ** from the control at initialisation. Isolation is free.
* **A3-global reshapes** `frame_head.trunk.0.weight`, so no stream management can align the two
  constructions, and every module built after the frame head shifts: measured, **37 shared
  tensors differ**, including the geometry tokeniser.

New utility `generation/neural/nn/init_alignment.py` builds both models at the same seed and
copies every tensor they share by name *and* shape from control into treatment. The control is
read only, so its initialisation is unchanged and the frozen references stay reproducible.

Measured after alignment:

| arm | tensors copied | shared tensors still differing | intended differences |
| --- | ---: | ---: | --- |
| A3-depth | 141 | **0** | none |
| A3-global | 140 | **0** | `frame_head.context_norm.bias`, `frame_head.context_norm.weight`, `frame_head.trunk.0.weight` |

Every run records its alignment report. A run whose intended differences are not exactly the
expected set is a stop condition.

---

## 8. Experiment matrix

| arm | `graph_recurrence` | `frame_scene_context` | seeds | purpose |
| --- | ---: | --- | --- | --- |
| A3 | 1 | false | 0,1,2 | frozen control |
| A3-depth | **2** | false | 0,1,2 | H2 |
| A3-global | 1 | **true** | 0,1,2 | H6 |

Six runs to train. No other variant is added before these are analysed.

---

## 9. Training and validation protocol

Identical for both treatments, and identical to the control's original run: rotated corpus
`datasets/processed/step10_rotated`, chordal objective, λ=3.0, 1200 steps, batch 8, AdamW with
the Step 9 schedule, CPU, three seeds, per-run JSON written atomically with resume, 2 GiB free
disk guard.

**No hyperparameter is tuned per arm. No arm trains longer.**

Selection and all diagnostics run on **validation**. Probes are fitted on **train** only.

## 10. Confirmatory test protocol

One test evaluation, on `test_seen` plus the three shift splits, **after** both treatments are
trained and the validation result is recorded. Test is never used to choose between H2 and H6,
to pick a recurrence value, or to alter anything. If a treatment fails on validation it still
gets its single test measurement, reported for completeness, and that measurement changes
nothing.

---

## 11. Primary metric and decision rule

**Primary metric:** held-out rotation geodesic error in degrees (`rotation.<split>.mean_deg`).

**Decision rule, fixed now:**

> A treatment is **supported** only if (a) its mean rotation error improves on A3 by at least
> **1.0°**, (b) all three seeds move in the same direction, (c) its intended mechanism shows
> measurable evidence, and (d) integrity and leakage checks pass.

1.0° is the threshold used throughout Steps 11 and 12, chosen because A3's own seed spread is
1.03° and a smaller difference cannot be distinguished from it. It is not adjusted.

Neither treatment is required to approach 11.57°. The graph-aware lookup is an upper reference
for available relational information, not a target.

**If both are supported:** report whether the mechanism evidence distinguishes them. If it does
not, the result is **ambiguous** — a winner is not declared on a numerical margin alone.

**If H6 alone is supported:** note that it also carries +2.05% capacity while H2 carries none,
so a capacity explanation cannot be excluded by this matrix; that would be the follow-up.

**If neither is supported:** record that neither increased propagation depth nor global graph
context explains the gap, and **stop** rather than adding complexity.

---

## 12. Predictions, recorded before training

**H2:** if depth is the bottleneck, A3-depth improves rotation by ≥1.0°, relational information
in the entity latent increases (Step 11 probe on `entity_latent`), and the graph ablations
become more discriminative (`entities_only` costs more than the control's 1.83°).

**H6:** if global context is the bottleneck, A3-global improves rotation by ≥1.0°, the global
context probes predict graph configuration above chance, and entity representations become more
sensitive to scene-level relational change.

**Null:** neither improves meaningfully. Given Step 11 (relation content never enters the
latent), Step 12 (a richer relation channel settled at 0.33% of signal), and Step 9's P4 result
(this exact pooling was unsupported on translation), the null is a live and unremarkable
outcome. It will not be dressed as an architectural discovery.

**Dissociation to watch for:** latent relational information rising while rotation does not
improve. Per §16 of the brief that is evidence of a downstream bottleneck, to be recorded, not
acted on by editing the geometry head.

---

## 13. Stop conditions

Declared in advance:

* frozen A3 references cannot be reproduced — **checked, they can (20.9558° vs 20.96°)**;
* Step 12 or corpus checksums differ — **checked, all clean**;
* alignment leaves any unintended shared-parameter difference;
* H2 or H6 would require changing unrelated architecture;
* capacity matching becomes necessary to interpret the result;
* any test split is read during training or selection;
* free disk falls below 2 GiB; research artifacts are never deleted to make space;
* the implementation is found to differ materially from the audited path in §2.
