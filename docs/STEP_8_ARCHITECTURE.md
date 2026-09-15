# Step 8 architecture

The architecture **as implemented**, not as proposed. Every shape below is taken from the
code and checked by a test. Where Step 8 changed something, the Step 7 behaviour is shown
beside it so the difference is visible.

Synthetic research data throughout. Nothing here is anatomy, is medically validated, or
supports any clinical claim.

## The pipeline

```text
TEXT
 |
 |  deterministic command engine (Step 4), used as an oracle and labelled as one
 v
AWR                                   entities, state, three typed graphs, scene memory
 |
 v
ENTITY TOKENS                         [B, N, 256]   composed from identity, type,
 |                                                  role, laterality, hierarchy
 v
RELATIONAL REASONING                  [B, N, 256]   writes only [96:256]
 |                                                  identity subspace [0:96] untouched
 v
PREDICTED FRAMES                      [B, N, 12]    3 translation, 3 log-scale, 6D rotation
 |
 v
GEOMETRY TOKENS                       [B, N, 32, 64]
 |
 v
NESTED LOD DECODER                    tokens[:, :, :K]  with K in (8, 16, 24)
 |
 v
PER-ENTITY IMPLICIT GEOMETRY          [B, N, P]     occupancy logits in entity-local frames
 |
 v
SCENE COMPOSITION                     [B, P] scene, [B, P, N+1] part labels
```

`N` is the padded entity axis, 64 slots for a 42-entity ontology of which 20 are used by
the whole-organ corpus. `P` is the number of sampled query points, 256 in training.

## What Step 8 changed

| Component | Step 7 | Step 8 |
| --- | --- | --- |
| Relational mechanism | partitioned typed graph transformer, 2,666,896 parameters | untyped graph attention, 692,864 |
| Relation types | embedding plus per-head typed bias | **removed**; the encoder cannot read `edge_relation` |
| Head partitioning | heads split by graph kind | **removed**; one mask, shared across heads |
| Frame head | linear probe, 3,084 parameters, never exercised | residual MLP, 135,180 parameters, supervised and used |
| Placement at inference | ground-truth frames supplied | predicted |
| Level-of-detail objective | each prefix against its own target | plus containment and preservation between prefixes |
| Arrangements in the data | four discrete labels | a continuous nine-dimensional space |

## Relational reasoning: `UntypedGraphEncoder`

`generation/neural/nn/graph_lite.py`

```text
input    entity_latent  [B, N, 256]
         structure.graph_adjacency  [B, 3, N, N]  or  [3, N, N]

mask     union over the graph axis, symmetrised, plus the diagonal
         [B, N, N]      one mask, shared by every head

layer    LayerNorm -> multi-head attention (4 heads x 64) -> Linear(256 -> 160)
         LayerNorm -> MLP(256 -> 256 -> 160)
         both sublayers write only latent[..., 96:]

output   [B, N, 256]    latent[..., :96] bit-identical to the input
```

Two layers, 692,864 parameters, within 4% of the reduced typed encoder's graph budget of
666,760. Matching that budget is what makes `A3Lite` against `A1M` a question about
relational structure rather than about capacity.

**Why the types are gone.** Step 7 measured that the typed encoder responded to which
entities are connected and almost never to what the relation says: flipping every spatial
relation to its opposite moved entity ownership IoU by 0.0002, while removing the same
edges cost 0.0406. This encoder therefore never reads `edge_relation`. That is a testable
claim and `tests/test_step8_graph.py` checks it: corrupting relation labels leaves the
output **bit-identical**, while dropping or rewiring edges changes it.

**What it must still distinguish.** "A connects to B" from "A connects to C". The mask is
built from endpoints, so moving an edge's endpoint changes the output; a test covers it.
The encoder is also permutation-equivariant, so entity identity comes from the latent and
never from the slot index.

## Placement: `FramePredictor`

`generation/neural/nn/geometry.py`

```text
input    entity_latent  [B, N, 256]      after the relational encoder
         LayerNorm -> Linear(256 -> 256) -> GELU -> Linear(256 -> 256) -> GELU
         Linear(256 -> 12)
output   [B, N, 12]     translation[0:3], log-scale[3:6], 6D rotation[6:12]
```

135,180 parameters against Step 7's 3,084. The output projection is zero-initialised with
a bias at the canonical frame, so an untrained head produces a sane scene at the origin
and training moves away from it rather than climbing out of a numerical hole.

The head reads the latent **after** the relational encoder. That ordering is the whole
experiment: relations are the route by which an entity's neighbours can influence where it
goes, and a no-graph arm's latent carries no neighbour information, so it can only place
entities by identity.

### The frame selection rule

```python
if use_predicted_frames:      # evaluation: never sees a true frame
    return predicted
if teacher_forcing >= 1.0:
    return truth
if teacher_forcing <= 0.0:
    return predicted
keep = rand(...) < teacher_forcing
return where(keep, truth, predicted)   # per entity
```

`use_predicted_frames` is checked **first**, so a teacher-forcing ratio left set by
mistake cannot turn a headline result into an oracle result. That ordering is the guard
against the Step 7 defect returning by accident, and a test asserts it.

## Nested level of detail

`generation/neural/nn/nested_lod.py`

The mechanism was already a literal prefix. `OccupancyFieldDecoder` reads
`tokens[:, :, :K]`, and the corpus's targets nest as well: the 10 entities exposed at
level 1 are a subset of the 16 at level 2, which are a subset of the 20 at level 3. So the
truth satisfies

```text
occupancy(L1)  subset-of  occupancy(L2)  subset-of  occupancy(L3)
```

What was missing was an objective asking the representation to do the same. Three terms:

| Term | What it asks |
| --- | --- |
| `reconstruction` | each prefix matches **its own** level's target |
| `containment` | a point the coarse prefix calls occupied stays occupied when tokens are added |
| `preservation` | where the coarse prefix is already correct **and the levels agree**, the fine prefix agrees with it |

The second mask on `preservation` matters. The finer level legitimately adds structure, so
restricting to points where the two levels' targets agree is what stops the term punishing
refinement itself. An earlier version took one target and scored a correct refinement as a
failure; a test now pins the corrected behaviour.

The coarse side of both terms is detached, so the gradient moves the fine decode toward
the coarse one rather than dragging the coarse decode down to meet it. Without that, the
cheapest way to satisfy nesting is for both levels to predict nothing.

**Deliberately absent:** any per-level projection head. Separate heads would let each level
learn an unrelated output while the prefix structure became decorative, which is the
failure this module exists to prevent. `GeometryConfig.per_lod_blocks` stays false and a
test asserts it.

## Training

`training/step8.py`

```text
per step:
  main forward at the batch's own level          teacher_forcing = ratio(step)
  three forwards, one per level prefix            same ratio
  loss = prototype losses
       + 2.0 * frame supervision
       + 0.5 * (1.0 * reconstruction + 0.5 * containment + 0.5 * preservation)
```

All three levels are decoded every step. Step 7 rotated through one level per step, which
meant the containment relation between two levels was never present in a single graph and
could not be asked for.

### The placement curriculum

| Stage | Fraction of budget | Teacher forcing | What it is for |
| --- | --- | --- | --- |
| P0 | 0.00 to 0.15 | 1.0 | the frame head learns to predict a frame at all |
| P1 | 0.15 to 0.60 | 1.0 down to 0.0, linear | the decoder meets its own placement errors gradually |
| P2 | 0.60 to 1.00 | 0.0 | the model decodes entirely from its own placement |
| P3 | evaluation | not applicable | `use_predicted_frames=True`, always |

The ratio is a fraction of the budget rather than a step count, so one configuration means
the same thing at any budget. It is logged at every recorded step and written into the run
manifest, so a reader can see what the model was decoding from at any point.

## Arms

| Arm | Relational mechanism | Total | Graph | Non-graph |
| --- | --- | ---: | ---: | ---: |
| A0 | none, and no entity axis | 3,103,894 | 0 | 3,103,894 |
| A1 | none | 559,751 | 0 | 559,751 |
| A1M | none; capacity restored by per-entity depth | 3,125,831 | 2,566,080 | 559,751 |
| A3Lite | untyped graph attention | 1,252,615 | 692,864 | 559,751 |
| A3L | reduced typed transformer | 1,226,511 | 666,760 | 559,751 |
| A3 | full typed transformer | 3,226,647 | 2,666,896 | 559,751 |

`A1M`'s 2,566,080 parameters are per-entity depth, not a graph: information cannot move
between entities. It is the arm that separates "relations matter" from "capacity matters",
and it has **3.7 times** `A3Lite`'s relational budget, so if `A3Lite` wins it is not
winning on size.

## What did not change

The Anatomical World Representation is still the semantic source of truth, not a metadata
layer over a mesh. Persistent entity identifiers, the entity hierarchy, the three typed
graphs, entity-local geometry, geometry correspondence through the entity axis, the
write-protected identity subspace, scene memory, the deterministic command path, the
explicit geometry decoder, and mesh as an export rather than the core representation are
all as Step 4 and Step 5 left them.

The distinction between **anatomical state** and **presentation state** is unchanged, and
still enforced by `scene_conditioning="lod_only"`: geometry conditions on anatomy and level
of detail, never on visibility or opacity, which is what makes a presentation edit provably
free.
