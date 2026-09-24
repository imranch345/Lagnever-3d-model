# STEP 11 — RELATIONAL INFORMATION REPORT

**Lagnav 3D — Where the relational signal is lost between the graph and the frame.**

Diagnostic study. No architecture was changed, no model was trained, no corpus was touched,
and no test split was read. All data is procedurally generated; nothing here is anatomy or is
medically validated.

---

## 1. Research question

A lookup keyed on entity identity **and the relationship graph** reaches 11.57° of rotation
error on `test_seen`. The trained A3 model, reading the same inputs, reaches 21.38°. Where
along `graph → representation → message passing → entity latent → frame head` is the
difference lost?

## 2. Frozen evidence from Step 10

Reproduced from artifacts before any Step 11 measurement, without retraining:

| quantity | value |
| --- | --- |
| A3 rotation, `test_seen`, λ=3.0 | 21.38° (20.78, 20.79, 22.57; std 1.03) |
| identity-only rotation floor | 22.65° |
| identity + graph lookup | 11.57° |
| the gap | 9.80° |

All four match the Step 10 record exactly. Change 1 and the corpus remain frozen and
checksummed.

## 3. Repository audit

Read from the code, not the documentation.

| audit item | finding |
| --- | --- |
| graph representation | per-scene adjacency `[B, 3, N, N]` plus edge lists: source, target, relation id, graph kind |
| which edges reach the model | the ontology's STRUCTURE edges **and** each scene's measured spatial and functional edges |
| message-passing depth | 4 attention layers |
| relation encoding | `Embedding(18, heads)` → an **additive scalar bias per head** on attention logits; the reverse direction uses the inverse relation |
| entity conditioning | the identity slice, channels 0–95 of 256, passes through every graph layer **unchanged** |
| what attention may write | only the 160-wide context slice |
| geometry head | reads the full 256-wide latent; `frame_scene_context` off |
| can graph information reach every entity? | yes; nothing masks it for A3 (`graph_scope="all"`) |
| is graph information discarded before the head? | no |

### Two discrepancies, recorded not reconciled

1. **`relation_width` is dead.** `GraphEncoderConfig.relation_width = 64` is never read by
   `graph.py`, and `generation/neural/scales.py` bills `relation_width × graph_heads` for a
   "relation to attention-bias projection" the code does not contain. Expressiveness is
   equivalent; the parameter model and the code disagree. No result depends on it.
2. **A relation can change how loudly a neighbour speaks, never what it says.** Values are not
   conditioned on relation type. This is the design, and §11 shows it is where the signal dies.

## 4. The exact A3 path

```
structure → EntityLatentComposer → entity_in [B,N,256]
          → PartitionedGraphEncoder (4 layers, identity slice frozen, context slice writable)
          → entity_latent [B,N,256]
          → FramePredictor (LayerNorm, 2×(Linear 256→256 + GELU), Linear 256→12)
          → frame [B,N,12]
```

## 5. Baseline reproduction

Reproduced exactly (§2). Nothing was regenerated. On `validation`, where every Step 11 number
below is measured: identity-only floor **23.51°**, identity + graph lookup **12.24°** (89.8%
of validation entities have their graph seen in training), A3 as trained **21.80°**.

## 6. Hypotheses

Preregistered in `STEP_11_RELATIONAL_INFORMATION_PLAN.md` before any measurement: H1 the head
ignores what reaches it; H2 insufficient propagation; H3 relation type weakly used; H4 identity
dominates; H5 optimisation variance; H6 the target is scene-global and the encoder has no
global pooling.

## 7. Preregistered experiment matrix

Two experiments, neither of which retrains anything. **E1** perturbs the relationship graph at
evaluation on the three frozen A3 checkpoints and measures what the model notices. **E2** fits
probes to frozen latents and measures what is present. Minimum meaningful difference **1.0°**,
chosen because A3's own seed spread is 1.03°; all three seeds must agree in sign.

## 8. Probe methodology

Frozen checkpoints in `eval()` under `no_grad`; no weight moves. Target is the 6D rotation,
scored as geodesic degrees. Linear and 2-layer-MLP probes, AdamW, 2000 steps, batch 64, lr
1e-3, one probe per tap per seed. Fitted on `train`, reported on `validation`. **A probe
measures presence, never use.**

## 9. Graph ablations (E1)

Validation, mean over three frozen seeds. Δ is against `intact`.

| condition | what it destroys | seed 0 | seed 1 | seed 2 | mean | Δ | meaningful |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `intact` | — | 20.96 | 23.43 | 21.01 | 21.80 | — | — |
| `drop_spatial` | every spatial edge | 20.89 | 23.45 | 20.95 | 21.76 | −0.04 | no |
| `drop_functional` | every functional edge | 21.04 | 23.52 | 21.28 | 21.95 | +0.15 | no |
| `drop_structure` | every structural edge | 20.98 | 23.42 | 20.98 | 21.79 | −0.00 | no |
| `shuffle_spatial_types` | relation **type**, endpoints kept | 20.96 | 23.43 | 21.01 | 21.80 | **+0.0003** | no |
| `randomise_spatial_endpoints` | endpoint **identity**, types kept | 20.92 | 23.43 | 21.02 | 21.79 | −0.01 | no |
| `entities_only` | every relationship | 23.55 | 23.80 | 23.52 | 23.62 | **+1.83** | **yes** |

Every perturbation was verified to change the tensors it claims to: `drop_spatial` removes 155
of 317 live edges, `shuffle_spatial_types` changes relation ids while leaving adjacency
identical, `randomise_spatial_endpoints` rewires the adjacency. A silent no-op would have
produced this same table for the wrong reason, so it was checked rather than assumed.

**Only the total removal of the graph matters.** Destroying *which* entities are related, or
*which* relation relates them, or any one typed graph, changes the answer by less than a
fifth of a degree.

## 10. Message passing

Not run. The evidence above makes depth the wrong first suspect: propagating a signal the
encoder does not encode would propagate the same generic connectivity further. Phase 5 is
therefore deferred, and this is a decision about sequencing, not a finding about depth. H2
remains untested.

## 11. Relation type — the mechanism

`shuffle_spatial_types` moves the answer by **0.0003°**. The reason is visible in the weights.

Relation type enters only as an additive bias on attention logits. Measured on a real
validation batch through the trained seed-0 encoder:

| quantity | value |
| --- | --- |
| attention logits on live pairs | std **0.4663**, range [−5.49, 4.88] |
| relation bias on live pairs | std **0.0026**, range [−0.049, 0.034] |
| bias as a share of the logit scale | **0.55%** |

Across seeds the trained bias has |mean| 0.0014–0.0019 and max |w| 0.031–0.034, against a
zero initialisation. **The relation-type channel never left its initialisation in any useful
sense.** It is not that the model weighs relation type lightly; it is that this channel
carries 0.55% of the signal that decides attention, and the model placed nothing in it.

## 12. Seed results

A3's instability is not noise around a single behaviour. It is *whether the model learned to
use the graph at all*:

| seed | intact | with the graph destroyed | what the graph is worth |
| --- | ---: | ---: | ---: |
| 0 | 20.96 | 23.55 | **2.59°** |
| 1 | 23.43 | 23.80 | **0.38°** |
| 2 | 21.01 | 23.52 | **2.51°** |

Seed 1 — the outlier on test as well, at 22.57° against 20.78 and 20.79 — sits essentially at
the identity floor and gains almost nothing from the relationship graph. The two good seeds
extract ~2.5°. H5 is supported and sharpened: the variance is bimodal, and the mode is
graph usage.

## 13–15. Position, rotation and scale

Rotation is the subject of this study and is reported throughout. Position follows the same
pattern, seed means on validation: **0.1490** intact, 0.1490–0.1558 across every content
ablation, and **0.1651** under `entities_only`. The only perturbation that costs position is
the one that costs rotation — removing the graph entirely — and `shuffle_spatial_types` leaves
position identical to intact at 0.1490, exactly as it leaves rotation identical. Scale is not
re-measured; no ablation here changes the scale target and Step 10's scale results stand.

## 16–17. Integrity and leakage

No training ran, so no integrity gate was invoked; the Step 10 gate's results stand unchanged.
Both freezes verified after this pass: Change 1 **101/101**, the corpus **7/7**. No test split
was read by any Step 11 experiment — `graph_ablation.json` and `representation_probe.json` both
record `"test_splits_read": []`, and probes were fitted on `train` and reported on
`validation`.

## 18. Against the identity-only lookup

On validation the identity-only lookup scores 23.51°. A3 scores 21.80°, beating it by 1.71°.
With the graph destroyed A3 scores 23.62° — **level with, in fact marginally worse than, the
lookup.** A3's entire advantage over an identity lookup comes from the relationship graph, and
all of that advantage is generic connectivity.

## 19. Against the graph-aware lookup

The identity + graph lookup scores 12.24° on validation. The relational information available
is therefore 23.51 − 12.24 = **11.27°**. A3 captures 1.71° of it, or **15%**.

## 20. Gap decomposition

Validation, rotation in degrees:

| stage | error | what it tells us |
| --- | ---: | --- |
| identity-only lookup | 23.51 | no relational information at all |
| A3 with the graph destroyed | 23.62 | A3 without relations is an identity lookup |
| probe on `entity_in` (MLP) | 23.65 | the pre-graph latent is identity, as designed |
| **A3 as trained** | **21.80** | generic connectivity is worth 1.7–1.8° |
| best probe on `entity_latent` | 21.07 | the latent holds **0.73°** more than the head takes — below the 1.0° threshold |
| identity + graph lookup | 12.24 | 11.27° of relational information exists |

The graph encoder's contribution to the representation is 2.58° (probe on `entity_in` 23.65 →
`entity_latent` 21.07). The head extracts nearly all of it. **Roughly 9° of available
relational information never enters the entity latent.**

## 21. Interpretation

**Direct measurement.** Destroying relation type changes A3's rotation by 0.0003°; destroying
endpoint identity by 0.01°; destroying every relationship by 1.83°. The trained relation bias
is 0.55% of the attention logit scale. A probe on the latent the head reads beats the head by
0.73°, below the preregistered threshold.

**Diagnostic evidence.** The graph encoder writes about 2.6° of value into the entity latent,
and all of it survives to the head. The information the encoder does *not* write is worth
about 9° more, and it is exactly the information the ablations show A3 ignoring: which
entities are related, and by which relation.

**Inference.** The loss is located at **graph representation → entity latent**, and within
that at the relation-conditioning mechanism. A relation can only reweight an attention
edge; it can never contribute a message of its own. The model consequently learns "these
entities are connected, mix them" and cannot learn "this entity is *above* that one, so the
organ is tilted this way".

**Architectural interpretation.** The typed graph is typed in name at this point in training.
Its type channel has a capacity of `heads` scalars per relation and, in these checkpoints,
an effective amplitude of half a percent. This is consistent with the Step 10 finding that only
A3 — the arm with the most attention heads and the full typed graph — extracted any rotation
margin at all: more heads means a slightly wider type channel, and A3's margin over the other
arms was about a degree.

### Hypothesis verdicts

| | verdict | basis |
| --- | --- | --- |
| H1 head ignores what reaches it | **not supported** | best probe beats the head by 0.73°, under the 1.0° threshold |
| H2 insufficient propagation | **untested** | deferred; see §10 |
| H3 relation type weakly used | **supported, directly** | 0.0003° effect; bias at 0.55% of logit scale |
| H4 identity dominates | **supported** | graph destroyed ⇒ A3 = identity lookup; its whole margin is generic connectivity |
| H5 optimisation variance | **supported and sharpened** | bimodal; seed 1 extracts 0.38° from the graph, the others 2.5° |
| H6 scene-global target, no pooling | **untested** | consistent with the evidence, not established |

## 22. Limitations

* **Validation only, three seeds.** Nothing here is a confirmatory test-split result, by design.
* **A probe bounds what is linearly or shallowly decodable, not what is present.** A stronger
  probe might find more in the latent; 21.07° is an upper bound on the information, not a proof
  of its absence.
* **The ablations measure one trained model's sensitivity**, not the architecture's capacity. A
  differently trained A3 might use relation type; these three did not.
* **H2 and H6 are untested.** Depth and global pooling remain live, and the diagnosis above
  does not exclude them as contributing causes.
* **The 11.27° lookup headroom is itself a lower bound on what is recoverable**, and it depends
  on 90% graph coverage between train and validation. It is a diagnostic reference, not a target.

## 23. Decision

**The evidence supports changing the relation encoding.** It does not support any of the
alternatives:

* *retaining the current graph architecture* — no: the type channel is measurably inert;
* *simplifying it* — no: the graph is contributing 1.7°, and removing it costs that;
* *increasing message-passing depth* — not yet: propagating an unencoded signal further will
  not create it, and H2 is untested rather than refuted;
* *modifying the geometry head* — no: H1 is not supported, the head extracts nearly everything
  the latent holds;
* *a larger model* — no, and nothing here argues for one. The failure is a missing channel, not
  a shortage of capacity. A1M already demonstrated that capacity without relational structure
  buys nothing.

## 24. Recommended next experiment

**The smallest change that gives a relation its own message.** Condition the attention *value*
on relation type, rather than only the logit — a per-relation additive term on the value
vector, or an edge-type embedding concatenated before the value projection. One change, a
handful of parameters per relation, and it isolates exactly the mechanism §11 identifies.

Run it as: one arm (A3), three seeds, λ=3.0, everything else frozen, scored on validation,
with the current typed encoder as the paired control and the parameter delta reported
explicitly. The decision rule should be the 1.0° minimum used here.

Two predictions worth recording before it runs, so the result can contradict them: if H3 is
the binding constraint, `shuffle_spatial_types` should become *expensive* under the new
encoding; and the gain should appear in the two seeds that already use connectivity before it
appears in seed 1.

If that change moves nothing, H2 and H6 become the live hypotheses and depth or global pooling
is the next study.
