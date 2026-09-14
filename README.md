# Lagnav 3D

**R&D prototype. Step 4 built the deterministic foundation, Step 5 designed the neural
architecture, and Step 6 implemented it, generated a synthetic corpus and ran the first
controlled experiment. Results are on synthetic data only and make no medical claim.**

Lagnav 3D is a research project aiming at a *language-controlled 3D anatomical
intelligence system*: you describe anatomy in words, and a structured, editable,
anatomically meaningful 3D scene responds.

This repository is **not** that system. It contains two things:

1. **A working deterministic foundation** (Step 4): the Anatomical World
   Representation, the heart ontology, the command engine, persistent scene
   memory. It runs, and it is testable without any neural network.
2. **A designed neural architecture** (Step 5): tensor contracts, typed
   interfaces, a pre-registered experiment and the reasoning behind every choice.
3. **A working prototype and a first experiment** (Step 6): a PyTorch implementation
   of that architecture, a generated synthetic corpus, a parameter-matched
   appearance-driven baseline, and measured results.

Nothing here calls a text-to-3D model, and no external model is part of the
architecture. Where a research decision has not been made, the code contains a typed
interface and a `TODO`, not an invented answer.

* Neural design: **[docs/neural-architecture.md](docs/neural-architecture.md)**
* Prototype: **[docs/step-6-prototype.md](docs/step-6-prototype.md)**
* First experiment: **[docs/step-6-experiment-report.md](docs/step-6-experiment-report.md)**
* Decision records: [docs/adr/](docs/adr/)

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

# Lagnever-3d-model

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
| **Step 6** experiment heart-001 with five ablations    | Run; results on synthetic data only                  |

### Deliberately not implemented

Each of these is declared in typed code and raises
`awr.errors.NotYetImplementedError` if called, so nothing can depend on a
pretend implementation.

- **A learned language model.** The control path is the Step 4 deterministic engine,
  used as an oracle and labelled as one. Only the language-anatomy alignment head is
  trained from text features.
- **Material and animation decoders**, image and multi-view modalities, cross-modal
  alignment, and local geometric editing. Only presentation and level-of-detail edits
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
datasets/     Dataset schema, licensing policy, synthetic example
ontology/     Versioned ontology data (Heart Ontology v0.1)
configs/      Domain configuration, the (empty) model configuration, neural experiment configs
experiments/  Runnable demonstrations and the pre-registered heart experiment
docs/         Architecture decisions and the Step 5 neural design document
training/     Training loop, run manifests, command line
visualization/ Dependency-free exporters and research views
tests/        462 tests
```

Dependency direction is one-way: `awr` depends on nothing above it, and the test
suite enforces that along with the "standard library plus PyYAML" rule.

## Running tests

```bash
pytest                       # 462 tests with the prototype extra, 375 without
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

**Next: Step 7.** The evidence points at per-entity factorisation rather than at
relational reasoning, so the next experiment should test the graph encoder on a task
that genuinely needs relations, and replicate the main result on data that was not
generated per entity.

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
