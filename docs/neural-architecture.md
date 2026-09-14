# Lagnav neural architecture

**Step 5 design document. PROPOSED and UNVALIDATED throughout.**

Nothing in this document has been trained, measured or demonstrated. It describes
an architecture we intend to build and an experiment intended to test whether it
is worth building. Where a choice is open, it says so. Where a claim could be
wrong, it says how we would find out.

Code is the source of truth for every contract and every table here. The document
explains the reasoning; `generation/neural/` holds the declarations, and the tests
check that the two agree.

---

## 1. Research hypothesis

> **H1.** A structured anatomical latent representation, which jointly encodes
> semantic identity, anatomical relationships, spatial structure and 3D geometry,
> provides better anatomical consistency, controllability, semantic editing and
> persistent scene modification than a purely appearance-driven text-to-3D
> representation.

Stated that way, H1 is not testable. It is decomposed into four claims that are,
plus a null hypothesis that we expect to have to take seriously:

| | claim | measured by |
| --- | --- | --- |
| **H1a** | Higher part-level control success. | `part_control_success` |
| **H1b** | Generated geometry satisfies more held-out anatomical relationships. | `structural_relationship_accuracy`, `spatial_relationship_accuracy` |
| **H1c** | Identity and edit locality survive a multi-turn session. | `identity_persistence`, `edit_locality` |
| **H1d** | The advantage grows as the training set shrinks. | `sample_efficiency` |
| **H0** | No difference beyond run-to-run variance. | all of the above |

Two things follow from taking H0 seriously. First, both arms of the comparison are
trained with the same data, steps, seeds and parameter count, so a difference is
attributable to the representation. Second, the objectives the hypothesis is
judged on are **held out of training**, because a model trained to satisfy
relationship constraints that then satisfies relationship constraints has
demonstrated only that the objective worked.

---

## 2. Design principles

1. **Anatomy is the truth; geometry is a view of it.** Carried over from Step 4.
   Every decision below is downstream of it.
2. **Identity is structural, not learned.** Entity identity lives in a
   write-protected subspace and on a tensor axis. A model cannot be trained into
   or out of preserving it.
3. **Symbols where symbols are exact.** The ontology already states type,
   hierarchy, laterality and relationships. Those are injected, not inferred. Model
   capacity should go to what the ontology cannot state, which is shape.
4. **Address before you learn.** Part control, local editing and correspondence
   are indexing problems. They are solved with axes and indices, not with losses.
5. **The action space is closed.** Language selects from the ontology's entities
   and a fixed operation vocabulary. Anatomical hallucination becomes structurally
   impossible rather than statistically unlikely.
6. **Presentation state never touches identity.** Visibility, opacity, level of
   detail and animation condition decoders and never enter the latent.
7. **Cheap enough to repeat.** The first experiment targets one GPU and hours per
   run. A hypothesis that takes a week per run gets tested once.
8. **Nothing pretends to work.** Every unimplemented component raises. No
   placeholder returns a plausible number.

---

## 3. Full architecture

```
                    text
                     |
      +--------------+--------------+
      |                             |
  CONTROL PATH                 CONDITIONING PATH
  text -> AWR program          text -> pooled embedding
  (discrete, verifiable)       (continuous, appearance only)
      |                             |
      v                             |
  Step 4 AWR engine                 |
  validate + apply                  |
      |                             |
      v                             |
  ANATOMICAL WORLD REPRESENTATION   |
  entities, hierarchy,              |
  structure/spatial/functional      |
      |                             |
      v                             |
  AWR feature extraction  <---------+
  (implemented today)
      |
      v
  GRAPH ENCODER
  heads partitioned by graph, typed relation bias
      |
      v
  ANATOMICAL LATENT
  z_scene | z_entity | z_relation | z_geometry | entity_frame
      |
      +-----------------+------------------+
      v                 v                  v
  GEOMETRY          MATERIAL           ANIMATION
  per-entity        decoder            decoder
  fields            (declared)         (declared)
      |
      v
  STRUCTURED 3D SCENE  ->  meshes at export
      |
      v
  SCENE MEMORY (Step 4, persistent)
```

Editing re-enters at the AWR: an instruction becomes a program, the program
mutates the scene, and only the affected entities are re-decoded. See section 11.

The pipeline is declared in `generation/neural/model.py` and its composition is
checked by test: every stage's inputs are produced by an earlier stage or supplied
with the batch. A shape trace runs it end to end from a real heart scene.

| stage | status |
| --- | --- |
| AWR feature extraction | implemented |
| language control path | declared |
| language conditioning path | declared |
| graph encoder | declared |
| anatomical encoder | declared |
| geometry tokeniser | declared |
| per-entity field decoder | declared |
| alignment heads | declared |

---

## 4. AWR integration

The AWR is not an input the model happens to receive. It is a **discrete
bottleneck** that every instruction passes through, and it is the only thing
authorised to say what anatomy exists.

`generation/neural/features.py` is the bridge and is implemented today. It turns a
validated scene into index arrays, per-graph adjacency and explicit state
channels, with three properties that matter downstream:

* **Slot order is codebook order.** Slot *i* is the same entity in every tensor and
  in every scene. Nothing has to learn which slot is the left ventricle.
* **Vocabularies are closed and versioned.** Anatomy types, roles, laterality,
  relations, graphs, flow semantics and granularity all map to integers
  deterministically, and the codebook records the ontology version it came from.
  A mismatch raises rather than silently misaligning.
* **State is separated at the boundary.** Visibility, opacity and animation are
  packed into a documented nine-channel vector that decoders consume and the
  identity path never sees.

The AWR batch contract is `awr_batch` in `generation/neural/features.py`: 19
tensors, every one documented with dtype, normalisation, mask and padding value.

---

## 5. The entity latent

`z_entity` is `[B, N_ENT, D_ENT]` with `D_ENT = 256`, composed of named,
contiguous subspaces:

| subspace | width | span | source | write-protected |
| --- | --- | --- | --- | --- |
| identity | 96 | `[0, 96)` | learned embedding indexed by the codebook | **yes** |
| type | 48 | `[96, 144)` | anatomy type, semantic role, laterality | no |
| hierarchy | 32 | `[144, 176)` | depth plus a projection of the parent identity | no |
| geometry summary | 48 | `[176, 224)` | pooled geometry tokens | no |
| free | 32 | `[224, 256)` | learned, unconstrained | no |

**What is learned, what is explicit, what is graph-derived.** This distinction is
the one the brief asks for, and it drives the whole design:

| information | treatment | why |
| --- | --- | --- |
| entity identity | learned embedding, discrete index, protected | must be stable and addressable |
| entity type | learned embedding of a symbolic id | enables transfer across similar entities |
| hierarchy | explicit, injected | the ontology states it exactly |
| structural relations | graph-derived, via attention | relational, not per-entity |
| spatial relations | graph-derived plus predicted frames | needed both as constraint and as output |
| functional relations | graph-derived | relational |
| geometry | learned, continuous tokens | the ontology cannot state shape |
| topology | emergent from the field, measured not encoded | a property of the decoded surface |
| material | decoder output, conditioned on state | presentation |
| animation state | explicit conditioning | presentation |
| educational level | explicit conditioning | presentation |
| visibility | explicit, scene-side only | presentation, and must never touch identity |
| correspondence | structural, the entity axis itself | must not be inferable-but-wrong |
| multi-resolution | nested token prefix | detail must not change identity |

The **free** subspace exists for a methodological reason: without it, ablating a
named subspace also removes capacity, and the ablation measures the wrong thing.

`z_entity` is **not** a concatenation of unrelated vectors. The subspaces are the
*input* composition; the graph encoder then contextualises the non-protected part,
so after encoding the context slice is a mixture and only the identity slice
remains interpretable by construction.

---

## 6. Graph encoding

The requirement is that `pumps_to` and `adjacent_to` cannot be the same edge. That
rules out any encoder where the relation only gates connectivity.

**Proposed:** one relational graph transformer over entity tokens, heads
partitioned by graph.

| head role | heads | attends along |
| --- | --- | --- |
| structure | 2 | structure edges |
| spatial | 2 | spatial edges |
| functional | 2 | functional edges |
| global | 2 | all entity pairs |

Every head adds a typed bias projected from the relation embedding of the edge it
traverses. Direction survives because the reverse direction uses the inverse
relation's embedding, a different symbol. A pair carrying several relations sums
their biases: two structures that both adjoin and connect are more strongly
related than two that only adjoin.

The trade-offs are recorded in [ADR 0002](adr/0002-graph-encoder.md). In short:
separate per-graph encoders triple the parameters and turn cross-graph reasoning
into a fusion problem; relational message passing buys efficiency we do not need at
42 entities and costs global context we do want; an untyped attention network fails
the requirement outright.

The mask builder is implemented and tested: a functional head sees
`left_ventricle -> aorta` and not `left_ventricle -> interventricular_septum`, and
the spatial head is the exact reverse. Padded slots attend to nothing.

---

## 7. Multimodal alignment

The goal: "left ventricle", an illustration of it, a mesh of it, a mask of it and
the node `heart.left_ventricle` all land in the same place.

**Proposed: a hub, not a web.** The entity identity embedding table is the hub.
Each modality is a spoke that projects into a shared space and is trained to land
on the correct entity prototype, with hard negatives drawn from the ontology:
contralateral structure first, then siblings, then same-type entities, then spatial
neighbours. The sampler is implemented; for the left ventricle it returns the right
ventricle, the left atrium, the interventricular septum and the mitral and aortic
valves, which is a far harder set than random sampling would give.

Why not pairwise contrastive learning, the default answer: it needs large paired
corpora we do not have, it aligns appearance rather than anatomy, and it spends
data discovering a correspondence the ontology already states. It is ablation A6,
so the choice is tested rather than asserted. Full reasoning in
[ADR 0004](adr/0004-multimodal-alignment.md).

**Training is multimodal; inference is text-only.** Nothing in the inference path
requires an image or a mesh: text produces a program, the program mutates the AWR,
the AWR conditions decoding. Images and multi-view data are training-time signals
that shape the shared space, and they are curriculum stage S5, which is optional.

**Risks we are not assuming away.** Prototypes average out within-entity variation;
they can collapse between siblings; and they absorb dataset bias, so if every
training left ventricle is a textbook cutaway, the prototype encodes cutaways.
Identity probes will also score near-perfectly by construction and must never be
reported as evidence of learned anatomy.

---

## 8. Candidate 3D representations

Nine candidates scored on ten criteria in `generation/neural/three_d_latent.py`.
The scores are engineering judgement recorded as data, not measurement, and the
weights are declared: semantic correspondence and editability are weighted above
visual quality, because that is what the hypothesis is about.

| candidate | verdict | the deciding reason |
| --- | --- | --- |
| dense voxels | rejected | cubic memory at the resolution valve leaflets need |
| sparse voxels | deferred | implementation cost now, useful when scaling |
| point clouds | rejected | no surface, no topology |
| single implicit field | strong alternative (A4) | part identity only after the fact; edits are global writes |
| triplane | deferred | entities entangled in shared planes |
| gaussian splats | rejected | appearance-first, no surface, no topology |
| direct mesh generation | deferred | discrete connectivity is hard at small data scale |
| **per-entity latent fields** | **recommended** | correspondence becomes an index |

Only the top candidate and the reasoning should be read from the ranking. The
ordering among rejected candidates is an artefact of weighting interacting
criteria and means little. A test asserts that changing the weights to favour
visual quality changes the recommendation, which is the honest statement of what
the ranking is: a consequence of stated priorities.

---

## 9. Recommended 3D representation

**Per-entity latent token blocks decoded into per-entity implicit fields in
canonical entity frames, composed into a scene, meshed only at export.**

* `z_geometry` is `[B, N_ENT, K_GEO, D_GEO]` = `[B, 64, 32, 64]` at prototype scale.
* Each entity's field is decoded in its own canonical frame, predicted as
  translation, log-scale and a 6D rotation basis (`entity_frame`, 12 channels).
* The scene field composes entity fields; a part-label head attributes every query
  point to an entity or to background.

**Why canonical frames.** Separating *where a structure is* from *what shape it
has* makes shape decoding a smaller and more sample-efficient problem, makes
spatial relations computable analytically from predicted frames and therefore
directly measurable, and means moving a structure does not require re-decoding its
shape.

**The honest costs.** Composition seams between neighbouring entities are a real
risk. Per-entity capacity is fixed in advance by `K_GEO`. Decoding quality is
expected to trail a well-tuned single field at first. If seams dominate the
geometry metrics, ablation A4 is the fallback and the decision reverses.

---

## 10. Geometry correspondence

This is the architectural differentiator, and the design goal is that it should
not be a learned association at all.

**The entity axis is the correspondence.** `z_geometry[b, n]` is entity *n*'s
geometry. `entity_occupancy_logits[b, n, p]` is entity *n*'s field at query point
*p*. Slicing by entity index is the whole mechanism. There is no step at which a
mesh exists without knowing what it is.

Three layers reinforce it:

1. **Structural.** The token block and the field output are indexed by entity.
2. **Point-level.** The part-label head predicts an owner for every query point,
   giving correspondence anywhere in space, plus a directly supervisable target.
3. **Persistent.** The extracted component inherits the entity id and is recorded
   in Step 4's `GeometryReference`, which was built for this in the previous step.

So "make the left ventricle transparent" resolves by index, not by search. The
contrast with the baseline arm is exactly this: it recovers part identity with a
segmentation head afterwards, and can therefore be wrong.

---

## 11. Persistent editing

Edits are classified, and the classification decides the work:

| kind | example | geometry recomputed |
| --- | --- | --- |
| state only | "make the left ventricle transparent" | **none** |
| level of detail | "show more detail" | entities whose token prefix grows |
| structural | remove or replace an entity | the entity and its immediate context |
| geometric | "thicken the ventricle wall" | that entity alone |

`plan_edit` implements this today and is tested against the Step 4 engine. For
every state-only instruction in the command vocabulary it returns zero entities to
recompute and all 36 renderable entities frozen. That is the central editing claim
in a form that can be checked rather than believed.

For geometric edits the proposal is **masked, entity-local re-decoding**: rewrite
the affected entity's token block, re-decode that entity with neighbours and the
scene latent as frozen context, and leave every other block untouched. Locality is
then structural for untouched entities and measurable for touched ones.
Alternatives, including local diffusion, are in
[ADR 0005](adr/0005-editing-strategy.md).

Lazy decoding is handled honestly: if an entity has never been decoded, showing it
does require decoding it, and the plan says so rather than claiming a free edit.

---

## 12. Level of detail

**One block, read to different depths.** Token order is coarse to fine, and a
level of detail reads a prefix: 4, 8, 16, 24, 32 tokens for levels 0 to 4. The
schedule is configuration, validated to be non-decreasing and to end at `K_GEO`.

The principle inherited from Step 4 is that a level of detail changes what is
visible, never what exists and never what something *is*. Nesting makes that
structural: there is one geometry latent per entity at every level, so "show more
detail" cannot return a different structure. Separate per-level latents, the main
alternative, give five chances per entity to disagree about identity. That is
ablation A5; see [ADR 0006](adr/0006-level-of-detail.md).

The internal representation stays richer than the visible output at every level,
exactly as in Step 4: all 42 entities exist at LOD 0, where one is drawn.

---

## 13. Training objectives

Fifteen candidate objectives are declared in `generation/neural/losses.py`, each
with what it teaches, what data it needs, what failure it prevents, and its role in
the first experiment. Summarised:

| objective | teaches | needs | prevents | first experiment |
| --- | --- | --- | --- | --- |
| language-anatomy alignment | text to program | Step 4-generated pairs | acting on the wrong structure | **train** (1.0) |
| entity identity | prototypes are distinct | entity labels | sibling collapse | **train** (1.0) |
| structural relationship | composition | ontology only | parts that do not belong together | evaluate only |
| spatial relationship | arrangement | ontology plus frames | impossible layouts | evaluate only |
| functional relationship | flow structure | ontology, later geometry | a heart that does not connect | evaluate only |
| geometry reconstruction | shape | part-labelled 3D | a representation that decodes to nothing | **train** (1.0) |
| semantic part correspondence | which part is which | part labels | geometry that loses identity | **train** (1.0) |
| entity frame | pose | per-entity poses | shape entangled with placement | **train** (0.5) |
| topology | closed surfaces | geometry | holes and shells at export | deferred |
| multi-view consistency | view agreement | renderer, views | view-specific artefacts | deferred |
| cross-modal alignment | images near prototypes | licensed images | ungroundable text-only model | deferred |
| level-of-detail consistency | prefixes agree | geometry only | detail changing identity | **train** (0.5) |
| edit consistency | edits stay local | before/after pairs | edits leaking | **train** (1.0) |
| functional animation | cycle preserves the circuit | temporal 3D | motion that violates physiology | deferred |
| style invariance | anatomy is not appearance | style-varied renders | confounding the two | deferred |

**The initial strategy** is the seven objectives marked *train*: equal weight on
the two that define the deliverable (reconstruct the shape, know which part it is),
half weight on those that shape the representation, and **nothing** on the three
the hypothesis is judged by. `validate_strategy` refuses a configuration that
trains a held-out objective, so the rule is enforced rather than remembered.

Topology regularisers are deferred deliberately: they are easy to weight badly and
their effect is hard to attribute while reconstruction quality is still moving.

---

## 14. Curriculum

Six stages, revised from the nine in the brief and explained in
[ADR 0008](adr/0008-training-curriculum.md):

| stage | goal | external data | exits when |
| --- | --- | --- | --- |
| **S0** | Ontology self-supervised pretraining | **none** | held-out edge prediction beats frequency on all three graphs |
| **S1** | Language to AWR program | **none** | program accuracy on held-out paraphrases exceeds the rule parser's coverage |
| S2 | Geometry autoencoding with part supervision | yes | reconstruction good enough not to dominate generation error |
| S3 | AWR to geometry generation | yes | held-out relationship checks beat the baseline |
| S4 | Persistent editing | yes | untouched token blocks are bit-identical |
| S5 | Optional modalities: images, multi-view, animation | yes | image retrieval accurate on held-out styles |

The change that matters most is not a modelling change: **S0 and S1 need no
external data at all**, because the ontology is the data and Step 4 generates the
supervision. Work can begin before any dataset or licensing decision is settled,
which de-risks the schedule more than any architectural choice in this document.

Stage dependencies, objectives and freeze schedules are declared in
`generation/neural/curriculum.py` and cross-checked against the loss registry by
test: an objective used by a stage it does not declare is a load-time error. That
check has already caught one inconsistency in this design.

---

## 15. Ablations

| key | change | isolates |
| --- | --- | --- |
| **A0** | no entity axis, no graphs, no per-entity geometry | the value of the structured representation as a whole |
| A1 | entity tokens, graph encoder removed | how much comes from factorisation alone |
| A2 | structure graph only | whether hierarchy accounts for the effect |
| A3 | all three graphs | what spatial and functional relations add |
| A4 | shared field with a part head | whether per-entity fields matter |
| A5 | independent per-level latents | whether nesting keeps identity stable |
| A6 | pairwise contrastive alignment | whether ontology anchoring earns its labels |

A0 is the hypothesis test. A1 through A6 attribute any effect to a specific design
decision, and each corresponds to a decision record that could be overturned by
its result.

---

## 16. Baselines

Three kinds, and conflating them would be a reporting error:

| baseline | kind | what it tells us |
| --- | --- | --- |
| Step 4 deterministic AWR | reference ceiling | perfect semantic scores by construction; proves nothing about a learned model |
| appearance-driven ablation | **controlled comparison** | the hypothesis test, matched on data, steps, seeds and parameters |
| external systems (TRELLIS, Hunyuan3D, others) | reference point | where the field is on geometry quality, not evidence about H1 |

External systems are **not integrated and not wrapped**, and none is part of the
Lagnav architecture at any stage. Their licences have not been reviewed, and review
is required before any use, including evaluation.

**Reporting rule, enforced by the capability matrix:** an axis a system has no
interface for is reported as *not supported*, never as a score of zero. A system
that regenerates from a new prompt is not scoring badly at persistent editing, it
is not doing the task. Parameter count, training data and training steps are
reported alongside every number.

---

## 17. Compute assumptions

All figures are **estimates with wide error bars**. Parameter counts are arithmetic
from declared widths; compute is engineering judgement. No cluster is assumed.

| scale | parameters (estimated) | GPUs | wall clock | scenes | storage |
| --- | --- | --- | --- | --- | --- |
| prototype | ~14M | 1, 16-24 GB | 2-12 hours per run | 2k-10k | 20-100 GB |
| research | ~200M | 4-8, 40-80 GB | 2 days-2 weeks | 50k-500k | 1-10 TB |
| production | ~2B | 32-128 | weeks | 1M-10M | 10-100 TB |

The estimator's breakdown produces one immediately useful finding: at prototype
scale, **the language stack dominates**. Of roughly 14M parameters, about 10.3M are
language (mostly the token embedding), 3.2M are the graph encoder, and under 0.5M
are geometry. Two consequences for Step 6: a small frozen pretrained text encoder
is probably the right first move, and the anatomical machinery this document
spends its length on is cheap to train and cheap to ablate.

A rough forward-and-backward estimate for the first experiment is on the order of
1e17 FLOPs, which is single-GPU-hours territory. Accurate to perhaps a factor of
three, which is enough to confirm the experiment is affordable and not enough for
anything else.

---

## 18. Open research questions

1. **Composition seams.** Do per-entity fields produce visible artefacts where
   structures meet, and if so does a shared field with a part head do better? (A4)
2. **Nested token capacity.** Do later tokens in a nested block get used, or does
   nesting cap detail at the finest level? (A5)
3. **Is the graph partition doing work?** If global heads dominate, the partition
   is decoration. (A2, A3)
4. **Prototype collapse.** Do ontology-derived hard negatives keep siblings apart,
   or do prototypes converge anyway?
5. **What is the right edit primitive?** Deterministic re-decoding, token deltas,
   or local diffusion. The interface is fixed; the method is not.
6. **How is style separated from anatomy** without style-varied training data?
7. **Label mapping for real data.** Clinical segmentation conventions do not match
   ontology entities. The mapping is a source of error nobody has measured.
8. **Does the control and conditioning split hold?** The boundary between style and
   content may be blurrier than the design assumes.
9. **Topology supervision.** Whether regularisers help or just make tuning harder.
10. **Animation representation.** Deformation fields, per-phase latents, or
    physical parameters. Blocked on data that does not exist.
11. **Does structure substitute for data?** H1d is the most economically important
    claim and the least certain.

---

## 19. Risks

| risk | mitigation |
| --- | --- |
| The synthetic corpus is generated per part and may favour a per-part model. | A4 plus a tier-1 replication on real data before any general claim. |
| Part-control metrics are easier for the structured arm by construction. | The metric definition must not assume the part attribution the arm provides; defined in `evaluation/neural_metrics.py` before any model exists. |
| A semantic win bought with worse geometry. | A pre-registered guard: more than 20 percent worse Chamfer distance invalidates the win. |
| Real cardiac data may not be licensable on acceptable terms. | S0 and S1 need none; the synthetic tier answers the representation question without it. |
| Fixed per-entity capacity may cap quality. | Measured as reconstruction error in S2 before generation is attempted. |
| Contracts validated symbolically may be impractical in a framework. | First Step 6 task is to build the prototype and find out. |
| The design is elaborate and could be over-engineered for the evidence. | A1 measures whether the entity axis alone accounts for the effect; if it does, most of this document is unnecessary. |
| **A null result.** | Report it. H0 is stated first for a reason. |

---

## 20. Proposed next experiment

**heart-001-structured-vs-appearance**, pre-registered in
`experiments/heart/experiment_001_structured_vs_appearance.py` and not run.

* **Question.** Does a structured anatomical latent improve semantic control and
  anatomical consistency against an appearance-driven representation trained on the
  same data with the same budget?
* **Arms.** Lagnav structured (`configs/neural/prototype.yaml`) against the
  appearance-driven baseline (`configs/neural/baseline.yaml`), matched on
  parameters within 5 percent, data, steps and seeds, three seeds each.
* **Data.** Tier-0 synthetic: 2,000 to 10,000 procedurally generated part-labelled
  hearts, exact ground truth, no licensing exposure, explicitly **not anatomically
  realistic**. Tier-1 real segmented data is a follow-up and is **not acquired**.
* **Success, pre-registered.** H1a needs a 20-point gap in part-control success with
  non-overlapping intervals; H1b needs 15 points on held-out relationship accuracy;
  H1c needs near-zero drift on untouched entities where the baseline drifts
  measurably; the geometry guard caps acceptable Chamfer regression at 20 percent.
  Each criterion states what would refute it.
* **Cost.** Single GPU, hours per run, roughly 40 runs including ablations and
  seeds.

**What this experiment cannot tell us:** anything about clinical accuracy, anything
about photorealism, and anything about how the architecture behaves beyond one
organ at 42 entities.

---

## Status summary

| | |
| --- | --- |
| Implemented | tensor contracts, AWR feature extraction, graph masks and relation bias indices, AWR programs and the Step 4 transcoder, level-of-detail token schedule, ontology hard negatives, edit planner, loss and curriculum registries, scale and compute estimators, pipeline composition and shape trace |
| Declared only | every learned component: language encoder and parser, graph encoder, anatomical encoder, geometry tokeniser, field decoder, material decoder, animation decoder, modality encoders, local editor, baseline adapters, all metrics that need a model |
| Trained | Step 6 trained the prototype: see [step-6-prototype.md](step-6-prototype.md) |
| Measured | Step 6 ran experiment heart-001: see [step-6-experiment-report.md](step-6-experiment-report.md) |
| Claimed | Only what that report measures, on synthetic data, with its stated threats to validity |

**Step 6 outcome, in one line.** Every pre-registered criterion passed on the synthetic
corpus, and the ablations showed that the per-entity factorisation, not the typed graph
encoder, carries the effect. ADR 0002's argument for the head partition is not
supported by this evidence.
