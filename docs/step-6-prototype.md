# Step 6: the executable prototype

**Status: implemented, trained on synthetic data, measured. Not validated anatomy.**

Step 5 designed an architecture and stopped before training. Step 6 turns that design
into code that runs, generates the data it needs, trains, and is measured against a
controlled baseline. This document describes what was built. The results are in
[step-6-experiment-report.md](step-6-experiment-report.md).

Nothing here is a medical claim. The corpus is synthetic, stamped
`SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED`, and an ellipsoid is not a
ventricle.

---

## 1. What runs end to end

```
"generate a human heart"                     Step 4 deterministic command engine
        |                                     (labelled the oracle language path)
        v
AWR program  ->  AWR scene                    typed operations over a closed codebook
        |
        v
AWR feature tensors                           generation/neural/nn/tensors.py
  entity ids, types, hierarchy, per-graph
  adjacency, masks, state channels
        |
        v
entity latents                                generation/neural/nn/entity.py
  identity | type | hierarchy | geometry | free
        |
        v
partitioned graph encoder                     generation/neural/nn/graph.py
  structure / spatial / functional / global heads, typed relation bias
        |
        v
anatomical latent  (scene, entity, geometry)  generation/neural/nn/model.py
        |
        v
per-entity geometry tokens                    coarse to fine, prefix per level of detail
        |
        v
per-entity occupancy fields in canonical frames
        |
        v
scene composition + part attribution          the correspondence mechanism
        |
        v
exported mesh / slices                        visualization/export.py
```

The editing path re-enters at the AWR, and for a presentation edit stops there:

```
"make the left ventricle transparent"
    -> AWR program -> scene state change -> zero neural computation
       geometry tokens bit-identical, untouched-entity drift exactly 0.0
```

## 2. Dependency tiers

PyTorch was introduced, and confined. Tier one (`awr`, `reasoning`, `editing`,
`animation`, `geometry`, `evaluation`, `datasets`, `generation`, `generation.neural`)
still imports only the standard library and PyYAML, enforced by a test that is
stricter than the one it replaced. Tier two (`generation.neural.nn`,
`datasets.synthetic`, `training`, `visualization`, `experiments`) may import PyTorch
and NumPy. See [ADR 0010](adr/0010-dependency-tiers.md).

Consequence: **375 tests still pass on Python 3.11 with only PyYAML and pytest
installed**, verified by running the suite in a PyTorch-free environment. With the
prototype extra the suite is 462 tests. The AWR remains a one-dependency library.

## 3. The model

| component | shape | notes |
| --- | --- | --- |
| entity latent | `[B, 64, 256]` | identity slice `[0, 96)` is never written by the encoder |
| relation bias | `[B, 8, 64, 64]` | accumulated per edge; reverse direction uses the inverse relation |
| scene latent | `[B, 256]` | from a one-hot level of detail only, see ADR 0012 |
| geometry tokens | `[B, 64, 32, 64]` | per entity, coarse to fine |
| entity frames | `[B, 64, 12]` | translation, log-scale, 6D rotation |
| per-entity field | `[B, N_visible, P]` | occupancy logits in canonical frames |
| part logits | `[B, P, 65]` | 64 entity slots plus background |

Parameter counts, measured:

| arm | total | notes |
| --- | --- | --- |
| A3 structured | 3,094,551 | composer 13.9k, graph encoder 2.67M, tokens 102k, field 71k |
| A0 appearance baseline | 3,103,894 | matched to within 0.3 percent before training |

The baseline is sized by search against the structured arm's count, capped at 128
latent tokens so the match cannot be bought with an absurd attention cost. An
unmatchable target raises rather than running an unmatched comparison.

### What makes the arms differ

| | A3 structured | A0 appearance |
| --- | --- | --- |
| entity axis | yes | no |
| typed graphs | three, head-partitioned | none |
| geometry | per entity, per-entity fields | one shared latent, one field |
| part identity | an index into the tensor | a segmentation head, learned |
| canonical frames | predicted per entity | none |
| geometry conditioning | anatomy + level of detail | the full request vector |

Both receive the same information about which entities were requested, and both mask
their part logits to the requested set. The difference is how that information is
represented.

## 4. The language path

The control path is the **Step 4 deterministic engine**, used as an oracle and labelled
as one everywhere it appears. It parses an utterance, resolves entity names through the
ontology and emits a typed AWR program. Step 6 does not train a language model, per the
brief: the research question is the anatomical representation.

What *is* learned from text is the language-anatomy alignment objective: a head that
predicts the requested entity set from the deterministic text features. That is a real
trained component with a real loss, and it is not a language model.

## 5. The synthetic corpus

`datasets/synthetic/` generates the tier-0 testbed described in
[ADR 0013](adr/0013-tier0-synthetic-corpus.md).

* One analytic primitive per renderable entity: 36 of them, covering chambers, valves,
  great vessels, septa, wall layers, internal structures, coronary vessels and the
  conduction system.
* Anchors derived from the sampled chamber radii, so 26 required ontology relationships
  hold by construction; a scene that violates one is rejected and resampled.
* Exact ground truth for identity, part ownership, canonical frames, occupancy and
  relationships.
* Scenes stored as parameters, so 2,000 scenes are 3.7 MB and every label is recomputed
  analytically.
* Splits by procedural family, so no family appears in two splits.

Generated corpus used for the experiment: 2,000 scenes, 40 families, levels of detail
1, 2 and 3, split 1,600 / 200 / 200.

## 6. Losses

The seven first-wave objectives from Step 5, and only those:

| objective | applies to | form |
| --- | --- | --- |
| language-anatomy alignment | both arms | binary cross-entropy over the requested entity set |
| entity identity | structured only | prototype contrastive with ontology hard negatives |
| geometry reconstruction | both arms | occupancy binary cross-entropy |
| semantic part correspondence | both arms | cross-entropy over part labels plus background |
| entity frame | structured only | L1 on translation, log-scale and rotation rows |
| level-of-detail consistency | both arms | coarse prefix matched to the detached fine decode |
| edit consistency | baseline only in practice | occupancy change at untouched entities |

Two of them have no counterpart in the baseline, because it has no entity axis and no
frames. The loss reports them as skipped rather than inventing an equivalent, and the
asymmetry is recorded as a threat to validity in the experiment report.

The three relationship objectives remain **held out of training**. A configuration that
tries to train them raises, as does one naming an objective that is declared but not
implemented.

## 7. Reproducibility

Every run writes a manifest with the commit, configuration, seed, Python, PyTorch and
NumPy versions, platform, device, parameter counts per component, dataset identity,
training and validation history, results and runtime. Two runs with the same seed and
configuration produce identical losses, which a test asserts.

One gap, stated plainly: **this repository is not a git repository**, so the manifest
records `unavailable: not a git repository` where the commit should be. Running
`git init` and committing would close it.

## 8. Commands

```bash
pip install -e ".[dev,prototype]"

python -m datasets.synthetic.cli --scenes 2000 --families 40 \
    --out datasets/processed/heart_tier0_2000

python -m experiments.heart.run_experiment --smoke

python -m training.cli --arm A3 --seed 0 --steps 1200 \
    --corpus datasets/processed/heart_tier0_2000

python -m experiments.heart.run_experiment \
    --corpus datasets/processed/heart_tier0_2000 \
    --steps 1200 --seeds 0 1 2 --ablations A1 A2 A4 A5 A6

python -m visualization.render --corpus datasets/processed/heart_tier0_2000 \
    --checkpoint experiments/runs/heart-001/checkpoints/A3-seed0.pt
```

## 9. Code quality

```
pytest                                                        all tests
ruff check .                                                  lint
mypy awr reasoning editing geometry generation animation evaluation datasets
mypy --config-file mypy-prototype.toml generation/neural/nn datasets/synthetic \
     training visualization experiments
```

Two type-checking passes because NumPy's shipped stubs use Python 3.12 type syntax.
Tier one is checked at 3.11, the project's minimum runtime; tier two at 3.12. No
`# type: ignore` was added to make either pass.

## 10. What is implemented versus declared

A component counts as implemented only if code exists, a test exists, it executes, it
participates in the prototype and its behaviour is measured.

**Implemented and measured:** AWR feature extraction, entity latent composition with a
protected identity slice, the partitioned graph encoder with typed relation bias,
per-entity geometry tokens with nested level-of-detail prefixes, canonical frames, the
per-entity occupancy decoder, scene composition and part attribution, the appearance
baseline, seven losses, the training loop, checkpointing, the synthetic corpus
generator, seven evaluation metrics, four diagnostics, the exporters and views, and the
experiment runner with pre-registered criteria.

**Still declared only:** every learned language component (the control path is the Step
4 oracle), the material and animation decoders, image and multi-view modalities,
cross-modal alignment, local geometric editing (only presentation and level-of-detail
edits are implemented), topology objectives, and external baseline adapters.
