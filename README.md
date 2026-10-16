# Lagnav 3D

**R&D prototype. Step 4 built the deterministic foundation, Step 5 designed the neural
architecture, Step 6 implemented it and ran the first controlled experiment, Step 7
stress-tested the representation to find out which parts are actually necessary,
Step 8 removed the teacher-forced placement that made the relational question unanswerable,
Steps 9 and 10 found what limits placement and showed rotation can be learned above its
floor, and Steps 11 to 20 traced why the model uses so little of the relational
information it is given. Results are on synthetic data only and make no medical claim.**

Lagnav 3D is a research project aiming at a *language-controlled 3D anatomical
intelligence system*: you describe anatomy in words, and a structured, editable,
anatomically meaningful 3D scene responds.

This repository is **not** that system. It contains:

1. **A working deterministic foundation** (Step 4): the Anatomical World
   Representation, the heart ontology, the command engine, persistent scene
   memory. It runs, and it is testable without any neural network.
2. **A designed neural architecture** (Step 5): tensor contracts, typed
   interfaces, a pre-registered experiment and the reasoning behind every choice.
3. **A working prototype and a first experiment** (Step 6): a PyTorch implementation
   of that architecture, a generated synthetic corpus, a parameter-matched
   appearance-driven baseline, and measured results.
4. **A representation stress test** (Step 7): a whole-organ corpus in which
   relationships are measured rather than asserted, ablations with matched parameters,
   perturbation experiments that damage one channel at a time, and persistent local
   editing. Step 7 also withdrew three earlier conclusions that turned out to rest on
   setups which could not test them.
5. **Placement and rotation** (Steps 8–10): predicted rather than teacher-forced
   placement, a placement-blind floor, a rotated corpus with floors of its own, and a
   rotation objective that clears them.
6. **A relational diagnostic programme** (Steps 11–20): preregistered studies of where
   relational information is lost, which ruled out four architectural and objective
   explanations and isolated a seed-dependent transition in training.

Nothing here calls a text-to-3D model, and no external model is part of the
architecture. Where a research decision has not been made, the code contains a typed
interface and a `TODO`, not an invented answer.

* Neural design: **[docs/neural-architecture.md](docs/neural-architecture.md)**
* Prototype: **[docs/step-6-prototype.md](docs/step-6-prototype.md)**
* First experiment: **[docs/step-6-experiment-report.md](docs/step-6-experiment-report.md)**
* Representation stress test: **[docs/step7/](docs/step7/README.md)**
* Predicted placement and relational inference: **[docs/STEP_8_COMPLETION_REPORT.md](docs/STEP_8_COMPLETION_REPORT.md)**
* Step 8 architecture: **[docs/STEP_8_ARCHITECTURE.md](docs/STEP_8_ARCHITECTURE.md)**
* Placement learning: **[docs/STEP_9_COMPLETION_REPORT.md](docs/STEP_9_COMPLETION_REPORT.md)**
* Step 9 architecture: **[docs/STEP_9_ARCHITECTURE.md](docs/STEP_9_ARCHITECTURE.md)**
* Step 10 architecture: **[docs/STEP_10_ARCHITECTURE.md](docs/STEP_10_ARCHITECTURE.md)**
* Parent-relative placement: **[docs/STEP_10_CHANGE1_REPORT.md](docs/STEP_10_CHANGE1_REPORT.md)**
* Rotated corpus and rotation learning: **[corpus](docs/STEP_10_CHANGE2_CORPUS_REPORT.md)**,
  **[training](docs/STEP_10_CHANGE2_TRAINING_REPORT.md)**,
  **[rotation weight study](docs/STEP_10_ROTATION_WEIGHT_STUDY.md)**
* Relational diagnostics, Steps 11–20: one report per step, linked from the table below
* Decision records: [docs/adr/](docs/adr/)

### Level 1: a heart and lungs you can talk to

Alongside the research steps there is now a product track:
**[docs/LEVEL_1_PLAN.md](docs/LEVEL_1_PLAN.md)**. Its goal is a language-controlled 3D heart and
lungs, built from real CT, that runs locally in a browser. What exists so far:

* a **thorax domain**. It includes Heart Ontology v0.1 unchanged and adds both lungs, five
  lobes and the trachea (ADR-L1-001);
* a **label map** from the TotalSegmentator v1 dataset (CC BY 4.0) to AWR entities, and a
  **mesh pipeline** that writes one GLB node per entity, with a manifest that carries source,
  license and quality (ADR-L1-002 to -004);
* a **byte-range fetcher** that takes only the needed label files from the dataset's 28 GB
  zip, and a **quality report** that picks a usable subject;
* a **local viewer**. The `CommandEngine` stays the only writer of scene state, and the page
  draws what `/state` says.

Real CT has not been downloaded yet. Until it is, the viewer runs on a synthetic **phantom**:
simple shapes placed where the structures sit, which is not anatomy.

```bash
pip install -e ".[dev,assets]"
python -m assets.phantom --out datasets/raw/phantom
python -m assets.meshes --subject datasets/raw/phantom/phantom --out datasets/processed/level1/phantom
python -m viewer.server --scene datasets/processed/level1/phantom --port 8791
```

Then open `http://127.0.0.1:8791/` and type, for example, "show only the lungs" or
"make the left ventricle transparent".

### Level 2a: generating a thorax

**[docs/LEVEL_2_PLAN.md](docs/LEVEL_2_PLAN.md)**. A statistical shape model, built from many
real subjects, generates new, plausible hearts and lungs. It comes before any neural model, and
it runs on a CPU. The code is in `shapes/`:

* subjects are aligned by the heart;
* one template per entity is fitted to every subject by non-rigid registration;
* a joint PCA model is fitted over the entities;
* the model is evaluated on held-out subjects against a mean-shape baseline;
* sampled scenes are marked as generated.

The viewer can draw from a model with `--model`, and it keeps the scene's state when the
geometry changes. Generated anatomy is not a measurement of anyone.

Results on 131 real CTs are in **[docs/LEVEL_2A_REPORT.md](docs/LEVEL_2A_REPORT.md)**: on held-out
subjects, the models beat the mean shape by 3.2–4.6×. One structure can be resized by a
sentence, such as "make the left ventricle 20% larger", within its own shape model
(ADR-L2-001).

### The dashboard: everything in one place

```bash
python -m dashboard.server --scene datasets/processed/level1/real/s0029 \
    --model datasets/processed/level2a/models/joint.npz --research ~/code/lagnav-3d
```

It runs on http://127.0.0.1:8795/, local only, and reads live from this machine every 15 s. It
has six tabs:

* **Overview:** jobs running now and headline results;
* **Research:** Step 21's branch map and escape timing, live;
* **Generated anatomy:** data funnel, held-out errors, validity;
* **3D viewer:** the viewer itself, with generate, resize and commands;
* **Evidence:** captioned screenshots;
* **Notebook & docs:** the lab notebook, results, reports and ADRs.

The code is in `dashboard/` and the tests in `tests/test_dashboard.py`.

### What Step 7 changed about how to read Step 6

Step 6 reported that the typed graph encoder was not earning its parameters. **That
conclusion is withdrawn.** The Step 6 corpus carried a single relationship graph across
all 2,000 scenes, so the graph encoder read a constant input and the comparison could
not have come out any other way. It was untestable, not refuted.

Step 7 rebuilt the data so the question is answerable, and found four further defects of
the same kind before any conclusion could be drawn. The details are in
[docs/step7/02_preregistration_amendments.md](docs/step7/02_preregistration_amendments.md).

### Steps 11 to 20, in one table

Step 10 left a gap. A lookup keyed on entity identity **and** the relationship graph reaches
11.57° of rotation error on `test_seen`; A3, reading the same inputs, reaches 21.38°. Steps
11 to 20 asked where that relational information is lost.

| step | question | answer |
| --- | --- | --- |
| [11](docs/STEP_11_RELATIONAL_INFORMATION_REPORT.md) | Where between graph and frame is the signal lost? | Only removing the whole graph matters (+1.83°). Shuffling relation types moves rotation by 0.0003°: the typed graph is typed in name only. |
| [12](docs/STEP_12_RELATION_CONDITIONED_VALUES_REPORT.md) | Does giving each relation its own message help? | No. 21.80° → 21.81° on validation, against a 1.0° threshold. |
| [13](docs/STEP_13_GRAPH_DEPTH_GLOBAL_CONTEXT_REPORT.md) | Is it propagation depth, or missing global context? | Neither. `test_seen`: A3 21.38°, deeper 21.99°, global context 21.44°. |
| [14](docs/STEP_14_OBJECTIVE_DIAGNOSTIC_REPORT.md) | Does the objective ask for relations? | Yes. Graph conditioning reduces its rotation term by 82.2%, so the objective is not the bottleneck. |
| [15](docs/STEP_15_SEED_MODE_REPORT.md) | Is the seed variance real? | Bimodal on ten seeds: 6 use the graph (~21°), 3 do not (~23.5°), 1 is intermediate. |
| [16](docs/STEP_16_ESCAPE_TRIGGER_REPORT.md) | What precedes a seed's escape into graph use? | Nothing measurable at 50-step resolution; the escape falls inside one interval. |
| [17](docs/STEP_17_HIGH_RES_ESCAPE_REPORT.md) | Is there a precursor at 5-step resolution? | No upstream precursor. Its reported 60-step usage lead is withdrawn by Step 18. |
| [18](docs/STEP_18_ONSET_AND_REPLICATION_REPORT.md) | Does that lead survive a dense baseline and a second seed? | No. The transition spans steps 810–820, and the null replicates on seed 3. |
| [19](docs/STEP_19_GRADIENT_DIRECTION_REPORT.md) | Does the gradient's direction change first? | No. Successive gradients stay aligned at +0.844 and nearly orthogonal to the net displacement. |
| [20](docs/STEP_20_CHECKPOINT_BRANCHING_REPORT.md) | From seed 9's step-805 state, does data ordering decide the outcome? | No. All 8 branches escape, at steps 830–840, to a final rotation of 20.988° ± 0.023°. |

Two things follow. Every architectural and objective explanation tested has been ruled out,
and the escape is already fixed in the model's state before any trajectory measurement sees
it coming. And escape is not the whole gap: on validation a graph-using seed still sits near
21° against the lookup's 12.24°, and A3, averaged over three seeds, captures 15.2% of the
available relational improvement.

**Next, not started:** Step 20's full-state checkpointing makes two experiments cheap.
Branching seed 9 earlier (700, 600, 400) would find when the outcome is fixed, and branching
a non-escaping seed would show whether that outcome is fixed too.

### Step 10, in one table

Step 10 changed what is predicted. **Change 1** predicted each entity's frame relative to its
parent and placed worse than the global target in all five arms. **Change 2** built a rotated
corpus (mean rotation 61.8°) with floors of its own, and at first no arm learned rotation
above them. A weight study then showed why: the rotation term was 7% of the frame loss.
At λ = 3.0, chosen on validation only, `test_seen`, rotation floor 22.65°:

| arm | rotation | margin over floor | position (floor 0.1605) |
| --- | --- | --- | --- |
| A3 full typed graph | 21.38 ± 1.03 | +1.27 | 0.1561 |
| A3Lite untyped graph | 22.64 ± 0.04 | +0.01 | 0.1597 |
| A1 no graph | 22.74 ± 0.01 | −0.08 | 0.1674 |

Rotation is learnable above the floor, and only A3 does it. The inputs carry far more: a
lookup keyed on entity identity and the relationship graph reaches **11.57°**. That gap is
what Steps 11 to 20 investigate.

### Step 9, in one line

A lookup table keyed on entity identity alone places structures with a translation error of
0.1605. Three of the five Step 8 arms do worse than that, and the two Step 9 changes tested
moved it by less than the seed spread. Placement is not a tuning problem: the frame target is
a scene-global centroid predicted from a local view, with a rotation component that is
constant. Step 10 should change what is predicted, not how hard it is optimised.

### Step 8, in one table

Five arms on a corpus of continuous arrangements, three seeds, **predicted placement**.
`test_seen` holds out whole organ families:

| metric | A3Lite untyped graph, 1.25M | A1M no graph, 3.13M | A3 full typed graph, 3.23M |
| --- | --- | --- | --- |
| entity ownership IoU | 0.0902 | 0.0741 | 0.0908 |
| part-control success | 0.0544 | 0.0330 | 0.0596 |
| relation accuracy | 0.7791 | 0.7485 | 0.8119 |
| relation-blind floor | 0.8110 | 0.8110 | 0.8110 |

`A1M` holds 3.7 times `A3Lite`'s relational parameter budget and loses on every metric, so
relational structure beats capacity. `A3` costs 3.85 times `A3Lite`'s relational budget for
a difference inside the seed spread.

Two findings cut the other way and matter more. Under **predicted** placement only one arm
clears the relation-blind floor, and only on one split of four. And on a composition of two
transformations seen separately but never together, the arms that read the graph score
**below** the floor while the arms that cannot read it score above: reading relations you
have not learned to compose is worse than ignoring them.

### Step 7, in one table

Six arms on a whole-organ corpus carrying 164 distinct relationship graphs, three seeds on
the headline arms, full 160-scene held-out evaluation:

| metric | A3 full graph, 3.09M | A1 no graph, 0.43M | A0 appearance, 3.10M |
| --- | --- | --- | --- |
| part-control success | 0.7218 | 0.7089 | 0.0016 |
| entity ownership IoU | 0.6339 | 0.6304 | 0.0164 |
| ownership IoU, placement inferred | 0.3020 | 0.2914 | 0.0164 |
| arrangement response, inferred | 0.0359 | 0.0000 | 0.0000 |

Two findings point in opposite directions and both hold. **The entity axis is essential**:
an appearance model with the same parameter budget is 450 times worse at saying which
structure owns which region. **The graph transformer is not**: it costs 86% of the model
and a version 7.2 times smaller matches it, while recovering only 3.6% of the arrangement
information the corpus makes available.

Reading either number requires knowing two things the tables above state and most such
tables do not: whether the model was handed ground-truth placement, and that a
relation-blind predictor scores 0.8878 on the relation metric rather than zero. Under
inferred placement, **every arm scores below that floor.**

### First experiment, in one table

Structured representation against an appearance-driven baseline at matched parameters
(3.09M versus 3.10M), same data, same steps, three seeds, on a **synthetic** corpus:

| metric | structured | appearance |
| --- | --- | --- |
| part-control success | 0.931 | 0.313 |
| held-out relationship accuracy | 0.900 | 0.700 |
| untouched-entity drift after an edit | 0.000000 | 0.001303 |
| occupancy Chamfer (lower is better) | 0.0137 | 0.0478 |

All four pre-registered criteria passed. The ablations then complicate the story: most
of the gain comes from per-entity factorisation, not from the typed graph encoder,
which is 86 percent of the parameters and worth about one point. Details and the six
threats to validity are in the experiment report.

---

## The research hypothesis

> A language-controlled anatomy system becomes possible when the system's source
> of truth is a **semantic anatomical representation**, not a mesh. Geometry
> should be one *view* of that representation, generated from it and addressable
> through it.

The corollary the prototype is built to test: if the representation is right,
persistent multi-turn editing ("show the chambers", then "make the left
ventricle transparent", then "animate blood flow") falls out of it, and does so
before any model is trained.

## Why mesh-only generation is insufficient

Current text-to-3D systems produce a mesh and stop. For anatomy that fails on
five counts:

1. **No identity.** A mesh has vertices, not a left ventricle. There is nothing
   stable for the next instruction to refer to.
2. **No persistence.** The usual fix for an edit is to regenerate from a new
   prompt, which produces a different object. "Make the left ventricle
   transparent" must change one property of one entity that already exists.
3. **No structure.** Chambers, valves and vessels are related: they contain,
   adjoin and feed one another. A surface cannot express that a valve sits
   between two chambers.
4. **No function.** Blood flow, the cardiac cycle and electrical conduction are
   properties of the anatomy, not of the triangles.
5. **No control of depth.** A school diagram and a medical illustration differ in
   what is *shown*, not in what exists. A mesh cannot hold what it does not
   display.

Lagnav's answer is to keep anatomy and geometry separate, and to make anatomy the
authority.

## What the AWR is

The **Anatomical World Representation** is the persistent, typed core of the
system. It holds:

- **Entities** with permanent identity (`heart.left_ventricle`), canonical names,
  synonyms, semantic role, LOD policy and state (visibility, opacity, material,
  transform, animation, geometry reference).
- **A hierarchy**: a single-parent tree used for navigation.
- **Three typed graphs** (below).
- **Scene memory**: current state, a versioned history of every change, and JSON
  persistence.

An entity id is assigned once and never changes. Identity lives on a frozen
object that editing code cannot write to, so "the entity id survived the edit" is
a structural guarantee rather than a convention.

Geometry hangs off entities **by reference**. Each renderable entity owns a
geometry component id (`geometry_part_0042`) from the moment it is created, long
before any geometry exists. Deleting the geometry would not delete the anatomy.

*AWR is a working name and may change.*

## The three graphs

Relationships are typed, explicit and kept in separate graphs rather than in one
undifferentiated edge list. Each relation type declares its graph, whether it is
symmetric, its inverse, and whether it carries physiological flow.

| Graph          | Relations                                                                                             | Example                                                |
| -------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Structure**  | `part_of`, `contains`, `connects_to`, `continuous_with`                                                 | `heart.chambers` → contains → `heart.left_ventricle`    |
| **Spatial**    | `adjacent_to`, `inside`, `surrounds`, `anterior_to`, `posterior_to`, `superior_to`, `inferior_to`, `left_of`, `right_of` | `heart.left_ventricle` → adjacent_to → `heart.interventricular_septum` |
| **Functional** | `receives_from`, `pumps_to`, `opens_into`, `exits_into`, `conducts_to`                                   | `heart.left_ventricle` → pumps_to → `heart.aorta`       |

Adding a fourth graph (developmental, pathological) means adding a graph kind and
relation specs; no query code changes.

The graphs are load-bearing, not decoration. Blood-flow animation is *derived* by
traversing the functional graph, and entity explanations are generated by
verbalising all three.

## Why the heart first

The heart is small enough to model completely and hard enough to be a real test.
It has a clear part hierarchy, non-trivial spatial relationships, genuine
function (flow and conduction), a natural detail ladder from "one organ" to
"conduction system", and it is the structure a learner is most likely to ask
about. If the representation cannot support the heart, it cannot support anatomy.

The rest of the body is explicitly out of scope.

---

## Current status

| Area                                                  | State                                                |
| ----------------------------------------------------- | ---------------------------------------------------- |
| Heart Ontology v0.1 (42 entities, 130 typed edges)     | Implemented. **Draft, not medically validated.**     |
| Typed AWR: entities, state, three graphs, scene memory | Implemented                                          |
| Persistent scene with versioned history and JSON save  | Implemented                                          |
| Entity resolver (synonyms, abbreviations, groups)      | Implemented                                          |
| Deterministic command engine                           | Implemented                                          |
| Editing operations                                     | Implemented for visibility, opacity, isolation, LOD, material, transform, animation binding |
| Level-of-detail system (0–4)                           | Implemented, provisional ladder                      |
| Geometry correspondence                                | Implemented as reserved component slots              |
| Semantic blood-flow and conduction model               | Implemented, derived from the functional graph       |
| Deterministic explanation                              | Implemented, graph-derived                           |
| Validation and evaluation framework                    | Implemented for ontology, identity, edits, persistence |
| Dataset schema and licensing policy                    | Implemented, synthetic example only                  |
| Neural architecture interfaces                         | Declared only                                        |
| **Step 5** tensor contracts and named dimensions       | Implemented and tested                               |
| **Step 5** AWR-to-tensor feature extraction            | Implemented and tested                               |
| **Step 5** anatomical latent, graph encoder, 3D representation, alignment, editing designs | **Proposed and documented; unvalidated** |
| **Step 5** training objectives, curriculum, scales      | Declared as data; nothing trained                    |
| **Step 5** first experiment                            | Pre-registered, and now run                          |
| **Step 6** PyTorch prototype of the full pipeline      | Implemented, trained, measured                       |
| **Step 6** tier-0 synthetic corpus generator           | Implemented; 2,000 scenes, exact ground truth        |
| **Step 6** training loop, checkpoints, manifests       | Implemented                                          |
| **Step 6** seven first-wave losses                     | Implemented; relationship objectives held out        |
| **Step 6** metrics, diagnostics, exporters and views   | Implemented                                          |
| **Step 6** experiment heart-001 with five ablations    | Run; **graph conclusion withdrawn as untestable**    |
| **Step 7** whole-organ generator, relations measured   | Implemented; 1,600 scenes, 164 relationship graphs   |
| **Step 7** anatomical variants crossed with families   | Implemented and guarded by the generator             |
| **Step 7** perturbation harness, partial-AWR cases     | Implemented and tested                               |
| **Step 7** local edit head, two scopes, plus regeneration control | Implemented, trained, measured            |
| **Step 7** relation-blind floor and placement conditions | Implemented; both required to read any relation number |
| **Step 7** convergence criterion, fixed in advance     | Implemented; the baseline's failure found a defect in every arm |
| **Step 7** six-arm suite, three seeds, 160 held-out scenes | Run; results on synthetic data only              |
| **Step 7** architecture verdicts                       | Entity axis KEEP; graph transformer SIMPLIFY; typed relations REMOVE; level of detail and editing REVISE |
| **Step 8** continuous arrangement corpus               | Implemented; 1,950 scenes, 9 arrangement axes, 4 generalisation splits |
| **Step 8** untyped graph attention (A3Lite)            | Implemented; 692,864 parameters, provably relation-type blind |
| **Step 8** predicted placement with a curriculum       | Implemented; final teacher forcing 0.0, evaluation never sees a true frame |
| **Step 8** nested level-of-detail objectives           | Implemented; containment exactly 1.0, preservation 0.97 |
| **Step 8** five-arm suite, three seeds, four splits    | Run; results on synthetic data only |
| **Step 8** architecture verdicts                       | Untyped graph KEEP; typed relations REMOVE; entity axis KEEP; nested LOD KEEP; predicted placement REVISE |
| **Step 9** placement-blind floor                       | Implemented; a lookup table on entity identity, 0.1605 to 0.1750 by split |
| **Step 9** frame decomposition                         | Implemented; rotation shown to be structurally zero under the current target |
| **Step 9** scene context and objective alignment       | Implemented and tested; neither moved placement beyond seed noise |
| **Step 9** verdict                                     | Placement is at the floor. The frame target, not the head, is the limiting factor |
| **Step 10** parent-relative placement (Change 1)       | Implemented; placed worse than the global target in all five arms |
| **Step 10** rotated corpus with its own floors (Change 2) | Implemented; 1,950 scenes, mean rotation 61.8°, checksummed and frozen |
| **Step 10** chordal rotation objective and weight study | Implemented; λ = 3.0, chosen on validation only |
| **Step 10** verdict                                    | Rotation beats its floor on held-out data, in A3 only (+1.27° on `test_seen`) |
| **Step 11** graph ablations and latent probes          | Run; only removing the whole graph matters, relation types are inert |
| **Steps 12–14** relation-conditioned values, depth, global context, objective audit | Run; none explains the relational gap |
| **Step 15** ten-seed study                             | Run; the outcome is bimodal, 6 of 10 seeds use the graph |
| **Steps 16–19** escape-trigger instrumentation         | Implemented; no precursor at any resolution, nor in gradient direction |
| **Step 20** full-state checkpointing and exact branching | Implemented and tested; model, AdamW state, step counter and stream position |
| **Step 20** verdict                                    | State-dominated: 8 of 8 branches from step 805 escape |

### Deliberately not implemented

Each of these is declared in typed code and raises
`awr.errors.NotYetImplementedError` if called, so nothing can depend on a
pretend implementation.

- **A learned language model.** The control path is the Step 4 deterministic engine,
  used as an oracle and labelled as one. Only the language-anatomy alignment head is
  trained from text features.
- **Material and animation decoders**, image and multi-view modalities, and cross-modal
  alignment. Local geometric editing was implemented in Step 7 and is measured there:
  the locality mechanism works and the edit head's accuracy does not. Only presentation
  and level-of-detail edits
  are implemented.
- **Any real data.** The corpus is generated by this project. No dataset has been
  downloaded, and `datasets/raw/` is empty.
- **Any medical claim.** The synthetic corpus is not anatomy, has had no clinical
  review, and nothing measured on it is evidence about real hearts.
- **Any geometry.** No mesh, no implicit field, no renderer. The only
  representation implemented is `SymbolicGeometry`, which holds component slots
  with **no geometric payload**. There is no random mesh generator and no
  external text-to-3D call anywhere in the repository.
- **Geometry-dependent editing**: cross-section, explode, segmentation, labels,
  geometry edits, component replacement and per-component regeneration.
- **Geometry and perceptual metrics**: watertightness, manifoldness, surface
  distance, multi-view consistency, text compliance, educational
  appropriateness.
- **Medical validation.** The ontology is a hand-authored v0.1 draft. It has had
  no clinical review, and the code says so in its metadata, its validation output
  and its API.
- **Real data.** No dataset is bundled, downloaded or scraped. The one sample in
  `datasets/annotations/` is synthetic metadata with no assets.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"           # AWR core: PyYAML is the only runtime dependency
pip install -e ".[prototype]"     # adds PyTorch and NumPy for the Step 6 prototype

pytest                            # run the test suite
python -m experiments.heart_session_demo          # Step 4 deterministic session

# Step 6 prototype
python -m experiments.heart.run_experiment --smoke
python -m datasets.synthetic.cli --scenes 2000 --out datasets/processed/heart_tier0_2000
python -m experiments.heart.run_experiment --corpus datasets/processed/heart_tier0_2000 \
    --steps 1200 --seeds 0 1 2 --ablations A1 A2 A4 A5 A6
```

A session, in code:

```python
from reasoning.command_engine import CommandEngine

engine = CommandEngine()
engine.execute("Generate a human heart")
engine.execute("Hide everything except the chambers")
engine.execute("Make the left ventricle transparent")
engine.execute("Show the valves")
engine.execute("Switch to medical level")
print(engine.execute("Explain what is happening").message)

lv = engine.scene.get("heart.left_ventricle")
lv.entity_id          # 'heart.left_ventricle'  - unchanged by every edit above
lv.opacity            # 0.3
lv.geometry_reference.component_id   # 'geometry_part_0004'
```

### Supported commands

The deterministic parser understands a controlled vocabulary:

| Intent           | Examples                                                                   |
| ---------------- | -------------------------------------------------------------------------- |
| Generate         | "generate a human heart", "create a heart"                                  |
| Show / hide      | "show the four chambers", "hide the valves", "show the left ventricle", "show everything" |
| Isolate          | "hide everything except the chambers", "show only the valves", "isolate the mitral valve" |
| Opacity          | "make the left ventricle transparent", "make the left ventricle opaque", "set the left ventricle opacity to 0.45" |
| Level of detail  | "show more detail", "show less detail", "set lod 3"                         |
| Audience level   | "switch to school level", "switch to medical level"                         |
| Animation        | "animate blood flow", "stop the animation"                                  |
| Explanation      | "explain what is happening", "explain the left ventricle"                   |
| Reset            | "reset the scene"                                                           |

Names resolve through the ontology, so "left ventricle", "Left Ventricle", "LV",
"left ventricular chamber" and `heart.left_ventricle` are the same entity.
Ambiguous references ("ventricle") are refused with the candidates listed rather
than guessed.

---

## Layout

```
awr/          Anatomical World Representation: schema, entities, relationships,
              ontology loader, LOD policy, scene memory, validation, config
reasoning/    Parser, entity resolver, explanation, command engine
editing/      Typed scene operations and the editor that commits history
geometry/     3D representation, entity-to-geometry correspondence, decoder interfaces
generation/   Language encoder, anatomical reasoner, scene generator, neural interfaces
animation/    Semantic cardiac flow and conduction model
evaluation/   Anatomy coverage, geometry checks, consistency, benchmarks
datasets/     Dataset schema, licensing policy, synthetic example, synthetic and
              whole-organ corpus generators
ontology/     Versioned ontology data (Heart Ontology v0.1; Thorax Ontology v0.1 includes it)
assets/       Level 1: label map, byte-range fetcher, mesh pipeline, quality report, phantom
viewer/       Level 1: local server around the CommandEngine, and the three.js page
language/     Level 1: free text to a closed AWR program (rule parser first, Claude if enabled)
shapes/       Level 2a: alignment, correspondence, shape models, validity, resize
dashboard/    One local page over everything: research, Level 1/2a, viewer, evidence, docs
configs/      Domain configuration, the (empty) model configuration, neural experiment configs
experiments/  Runnable demonstrations, the pre-registered heart experiment, and the
              runners for Steps 7 to 20
docs/         Architecture decisions, the Step 5 neural design document, and the
              reports for Steps 6 to 20
training/     Training loop, run manifests, command line
visualization/ Dependency-free exporters and research views
tests/        1024 tests
```

Dependency direction is one-way: `awr` depends on nothing above it, and the test
suite enforces that along with the "standard library plus PyYAML" rule.

## Running tests

```bash
pytest                       # 1024 tests with the prototype extra
pytest tests/test_command_engine.py -v
pytest -k resolver
pytest -m "not slow"         # skip the end-to-end training smoke test

ruff check .
mypy awr reasoning editing geometry generation animation evaluation datasets
mypy --config-file mypy-prototype.toml generation/neural/nn datasets/synthetic \
     training visualization experiments
```

Two type-checking passes because NumPy's shipped stubs use Python 3.12 type syntax:
tier one is checked at 3.11, tier two at 3.12. See
[ADR 0010](docs/adr/0010-dependency-tiers.md).

The suite covers ontology loading and corruption detection, entity identity,
hierarchy and relationship integrity, alias resolution, every specified command
scenario, scene persistence across commands, edit consistency, LOD behaviour,
geometry correspondence, and the refusal of every unimplemented component to
pretend otherwise.

## How this evolves toward the neural model

The pipeline is already wired in its final order; each stage currently has a
deterministic implementation behind the interface the learned component will
take over.

```
text
  → LanguageEncoder            lexical stand-in  →  neural encoder        (Step 5)
  → AnatomicalReasoner         ontology-driven   →  learned reasoner      (Step 5)
  → AWR (structure + spatial + functional graphs)          ← stays as-is
  → ThreeDLatentModel          not implemented   →  the open question     (Step 5)
  → GeometryDecoder            symbolic slots    →  geometry decoder      (Step 5)
    MaterialDecoder            semantic state    →  material decoder      (Step 5)
    AnimationDecoder           semantic schedule →  animation decoder     (Step 5)
  → structured 3D anatomy + scene memory                   ← stays as-is
```

Two invariants constrain every future model:

1. **Per-entity decoding.** The latent must be factorised so one component can be
   regenerated without disturbing the others. Interfaces declare this
   requirement (`supports_per_entity_decode`, `supports_partial_decode`).
2. **Correspondence is permanent.** A decoder may fill a component slot; it may
   never change which entity a component belongs to.

Replacing the rule parser with a language model changes one constructor argument
and touches nothing below it. That is the property the whole design exists to
protect.

**Step 6 built and measured it.** The declared stages are implemented in PyTorch, the
tier-0 corpus is generated, and experiment `heart-001` has been run with five
ablations. The architecture's editing claim holds exactly: a presentation edit leaves
every geometry token bit-identical, so untouched-entity drift is 0.000000 rather than
merely small.

The ablations also falsified part of the Step 5 reasoning. The typed graph encoder,
86 percent of the model's parameters, is worth about one point of part control on this
corpus; the entity axis and per-entity fields carry the effect. That is written up in
the experiment report rather than smoothed over.

**Steps 7 to 20 followed**, and the summaries at the top of this file record what each
found. The open question now is why a model handed the relationship graph captures so
little of what it carries; the next experiments are listed under Steps 11 to 20 above.

## Status of the anatomy

Heart Ontology v0.1 is a **hand-authored engineering draft**. It was written to
exercise the representation, not to be clinically authoritative. It contains
documented simplifications: the pulmonary veins and arteries are single entities,
the aorta is not divided into root, arch and descending segments, the pericardium
is filed with the wall layers, and individual papillary muscles are not
enumerated. No clinical review has taken place, no external ontology (FMA,
UBERON, SNOMED CT) has been aligned, and the ontology metadata, the validation
output and the API all say so. Expansion and medical validation are a separate
step.
