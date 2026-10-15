# Level 2 — generating plausible hearts and lungs, controlled through the AWR

**Status:** plan. Written 2026-10-16, after Level 1 worked end to end on real CT.
**Track:** product. Builds on Level 1 (`docs/LEVEL_1_PLAN.md`).
**Not medical.** Generated anatomy is a model's output, not a measurement of anyone. It must
never be presented as a patient, a diagnosis or a reference value.

---

## 1. Goal

Level 1 shows one real subject. Level 2 makes new anatomy:

> "Show me a heart and lungs" → a plausible, new thorax.
> "Make the left ventricle larger" → the same scene, with only that structure changed.

Done means:

1. **Generation.** A model produces a complete thorax: every Level 1 entity, as one closed
   surface each. It passes the Level 1 quality report's hard checks (sides, connectedness,
   closure) at a measured rate, on samples it was not trained on.
2. **Per-entity control.** A request changes one entity's shape through an explicit parameter,
   and the other entities do not move. This is the same identity rule Level 1 holds for
   visibility and opacity.
3. **Measured fidelity.** On held-out real subjects the model's reconstructions are scored
   (surface distance, volume error per entity) against a simple baseline. The scores are
   reported, never asserted.
4. **Same interfaces.** The generator fills `MeshRepresentation` behind the same manifest and
   viewer. Nothing above the geometry layer changes (ADR-L1-002).

## 2. Two stages, the cheaper first

| | 2a: statistical shape model | 2b: neural implicit fields |
| --- | --- | --- |
| What | Per-entity point correspondence, then PCA over the population | Per-entity latent codes with a shared decoder (an auto-decoder, as ADR 0003 proposed) |
| Generates | Mean ± a few interpretable modes (size, elongation, …) | Richer variation; can learn shape detail PCA cannot |
| Control | Mode coefficients: "larger" is a direction with a name | Latent directions, which must be found and validated |
| Compute | CPU, minutes to hours on this Mac | GPU, hours per training run |
| Data | 50–200 subjects | 200–1000 subjects |
| Risk | Correspondence errors become shape errors | Harder to train, to control, and to explain |

**Recommendation: build 2a first.** It answers the product question ("can we generate and
control plausible anatomy?") without a GPU bill. It gives named, testable controls, and it
becomes the baseline 2b has to beat. 2b starts only if 2a's measured fidelity or variety is not
enough.

## 3. Stage 2a in detail

1. **More subjects.** Fetch labels for about 200 thorax subjects with the existing fetcher. At
   Level 1's measured 1.6 MB per subject that is about **330 MB**, which needs the user's
   permission. The quality report keeps the usable ones; Level 1's rate was 15 of 20.
2. **Alignment.** Remove position and orientation per subject: the heart-centred viewer frame
   plus rigid alignment to a reference. Size is kept; it is one of the things to model.
3. **Correspondence.** For each entity, deform one template mesh onto each subject's surface
   with non-rigid registration. Every subject then has the same vertices in the same
   anatomical places. Registration quality is checked by surface distance; subjects that fail
   are dropped, and the drop is counted and reported.
4. **Model.** PCA per entity, plus one joint model of entity centroids, so that structures
   keep plausible relative positions. Keep enough modes for about 95% of variance, and report
   how many that took.
5. **Validity.** Generated samples go through the Level 1 quality report. The sampling range
   (in standard deviations per mode) is the largest at which the hard checks still pass at a
   stated rate.
6. **Evaluation**, on held-out subjects, never seen in fitting:
   - generality: reconstruction error of unseen shapes;
   - specificity: distance from generated samples to the nearest real shape;
   - compactness: variance explained per mode.

   The baseline is the mean shape.
7. **Control in the AWR.** A new operation, for example `resize` with an entity and a
   percentage, maps to the size mode of that entity's model. It needs:
   - a vocabulary change in `language.program` (ADR 0007: new capabilities are deliberate);
   - a matching engine operation;
   - a test that the other entities are unchanged.

## 4. Stage 2b, only if needed

Per-entity latent codes and a shared decoder that maps a 3D point and a code to a signed
distance. It is trained on samples from the 2a-registered meshes. It can be tried on this Mac's
GPU (Metal) first. A cloud GPU is only for runs too slow locally.

**Cost, as a range, not a quote:** one training run on a single data-centre GPU is a few hours.
At typical on-demand prices that is roughly $10–50 per run, and a development cycle of 10–20
runs is roughly $100–1,000. **No cloud spending happens without the user's decision.**

## 5. Milestones

| weeks | milestone | needs the user? |
| --- | --- | --- |
| 1 | Fetch ~200 subjects (~330 MB); quality report; alignment | **yes: download** |
| 2 | Template correspondence per entity, with measured registration error | no |
| 3 | PCA models, sampling, validity rate through the quality report | no |
| 4 | Held-out evaluation against the mean-shape baseline; written report | no |
| 5 | `resize` operation end to end: vocabulary, engine, viewer, tests | no |
| 6+ | Stage 2b, only if the evaluation says 2a is not enough | **yes: GPU budget** |

## 6. Risks

* **Correspondence is the hard part.** Lungs have fissures, and vessels branch differently.
  Some entities may need landmarks or a different template per variant. Failures are counted
  and reported, not hidden in an average.
* **Population bias.** The dataset comes from one hospital system, so generated anatomy
  inherits its population. This is stated wherever samples are shown.
* **Plausible is not correct.** A sample can pass every check and still be anatomically odd.
  The checks bound failures; they do not certify anatomy.
* **License.** TotalSegmentator v1 is CC BY 4.0. A model trained on it and its samples are
  derived works. Attribution travels with them, as in ADR-L1-004.
