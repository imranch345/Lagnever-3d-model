# Level 2a report: statistical shape models of the heart and lungs, from real CT

**Status:** complete for Stage 2a: registration, models, validity and resize. Amendments to
the resize design are recorded in ADR-L2-001.
**Date:** 2026-10-16
**Plan:** [`LEVEL_2_PLAN.md`](LEVEL_2_PLAN.md) §3
**Decisions:** [ADR-L2-001](adr/ADR-L2-001-resize-through-the-shape-model.md)
**Day-by-day record:** [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md)

Not medically validated, and not for clinical use. "Generated" means drawn from a statistical
model. It is not a measurement of anyone.

## 1. Summary

* **Data.** 200 subjects listed as covering the thorax were fetched from TotalSegmentator v1.0: 278 MB out of
  the 28.4 GB archive, by HTTP byte range. The quality gate kept **131** and stated a reason
  for each of the 69 it rejected.
* **Correspondence.** One template per entity was deformed onto every usable subject.
  Median surface fit is **0.24–0.40 mm** for the four chambers and the myocardium, and
  **1.1 mm** for the lungs. 16 lung registrations exceeded the 2 mm threshold; they were
  excluded and listed.
* **Models.** On 26 subjects never seen in fitting, every model beats predicting the mean
  shape by **3.2–4.6×**:
  - chambers and myocardium reconstruct to **1.1–2.1 mm**;
  - lungs to **5.1–6.3 mm**, about 2% of their size;
  - the joint model of all seven entities to **4.3 mm**.
* **Generation.** New thoraxes are drawn from the joint model and appear in the viewer under
  the same AWR and language control as real scans.
* **Validity.**
  - At 0.5 SD every sample passes checks calibrated on real anatomy (50 of 50).
  - At 1 SD, 78% pass, against 90.5% of real held-out subjects. The gap is mostly the left
    atrium pushed into the right lung.
  - The viewer samples at 0.5 SD.
* **Control.** A sentence resizes one structure within its own model. On a real heart,
  shrinking works across the range: LV −15% is applied, at 0.6 mm from plain scaling.
  Growing is refused beyond 2–6%, because neighbours do not give way in a linear model; the
  reply then offers the largest change that passes.

## 2. Question

Can a model learned from real CT generate hearts and lungs that look like real anatomy? And
can one structure be changed by a sentence while everything else stays put? The plan chose
the cheapest method that could answer this first: a linear statistical shape model on a
laptop CPU. Neural models (Stage 2b) are only considered if this one is not enough.

## 3. Data

| | |
| --- | --- |
| source | TotalSegmentator dataset v1.0, Wasserthal et al., Zenodo 10.5281/zenodo.6802614, **CC BY 4.0** |
| fetched | 200 subjects whose metadata lists the thorax among the scanned regions; 14 label masks each; 278 MB on disk |
| access | HTTP byte ranges into the 28.4 GB zip; the archive was never downloaded |
| meshes | marching cubes in the RAS+ frame, Taubin smoothing, then the viewer frame; quality is measured before decimation |
| quality gate | **131 of 200 usable**; the reference subject chosen by the same rule as Level 1 is again **s0029** |

**Why 69 were rejected.** A subject can have several reasons.

| reason | subjects |
| --- | --- |
| a lung or the heart is cut by the scan's field of view | 57 |
| structures missing, mostly the trachea, pulmonary trunk and right upper lobe (scans that stop below the upper chest) | 45 |
| a structure in separate pieces, mostly the aorta, IVC and right upper lobe | 22 |

TotalSegmentator is not a thorax dataset. Many of its scans were taken for other regions. The
gate rejects them for the reason that makes them unusable here, not for their quality.

## 4. Method

1. **Alignment.** Each subject's heart shell is aligned rigidly to the reference's: ICP between
   surface samples, then point-to-surface ICP. There is no scaling, because size is part of
   the shape. The same transform moves every entity of that subject.
2. **Templates.** These come from s0029, decimated to 2000 faces (chambers, myocardium) or 4000
   (lungs), keeping the largest connected piece. Topology is recorded:
   - closed: every template;
   - Euler number 2 (sphere-like): all except the left ventricle (1, a pinch) and the right
     lung (0, one handle). Both are inherited from the reference surface.
3. **Correspondence.**
   - **Placement:** the template is first laid over the subject's entity by a per-axis scale
     and shift matching robust extents.
   - **Deformation:** Amberg's optimal-step non-rigid ICP. The chambers use 4 stages. The
     lungs use 2 more, looser stages, without normals.
   - **Threshold:** a registration is kept if its mean distance from target surface to fit is
     at most **2 mm**.
   - The lung settings come from a probe on 3 subjects before the build
     ([evidence](evidence/2026-10-16_lung_registration_probe.md)). Without them, 4 of 6 lungs
     failed the threshold.
4. **Split.** Subjects are split once, with seed 0: 80% training (105), 20% held out (26).
   Models never see held-out subjects.
5. **Models.** PCA per entity, keeping 95% of the variance, with weights in sample standard
   deviations. A joint model over all seven entities' concatenated vertices keeps the
   population's correlations and relative positions. Only subjects with all seven registered
   contribute to it (98 training).
6. **Evaluation**, on held-out subjects:
   - **generality:** mean vertex error when an unseen subject is reconstructed with the model,
     against the error of predicting the mean shape for everyone;
   - **specificity:** mean distance from 50 random samples (weights N(0, 1), clipped at 3) to
     the nearest training shape;
   - **compactness:** modes kept and the variance they explain.

## 5. Results

### 5.1 Registration

| entity | median | 90th pct | max | excluded (> 2 mm) |
| --- | --- | --- | --- | --- |
| left ventricle | 0.24 mm | 0.28 | 0.61 | 0 |
| left atrium | 0.25 | 0.30 | 1.51 | 0 |
| right atrium | 0.30 | 0.36 | 1.76 | 0 |
| myocardium | 0.31 | 0.40 | 1.30 | 0 |
| right ventricle | 0.40 | 0.57 | 1.57 | 0 |
| left lung | 1.13 | 1.70 | 3.54 | 8 |
| right lung | 1.07 | 1.69 | 3.48 | 8 |

Four subjects failed both lungs (s0436, s0494, s0553, s0629).

### 5.2 Models: compactness, generality, specificity

| entity | train | modes (95%) | first mode | held-out error | mean-shape baseline | improvement | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| left ventricle | 105 | 9 | 42% | **1.13 mm** | 4.28 mm | 3.8× | 2.77 mm |
| myocardium | 105 | 9 | 43% | **1.15** | 4.43 | 3.8× | 2.70 |
| right ventricle | 105 | 11 | 49% | **1.49** | 6.24 | 4.2× | 3.44 |
| left atrium | 105 | 5 | 76% | **1.92** | 8.79 | 4.6× | 3.48 |
| right atrium | 105 | 6 | 68% | **2.08** | 9.58 | 4.6× | 4.26 |
| left lung | 100 | 10 | 34% | **5.12** | 20.76 | 4.1× | 12.04 |
| right lung | 101 | 7 | 57% | **6.34** | 26.15 | 4.1× | 11.77 |
| **joint (all seven)** | 98 | 17 | 45% | **4.31** | 13.71 | 3.2× | 8.61 |

**Reading.**

* The chambers reconstruct unseen hearts to 1–2 mm with 5–11 modes. The atria are the most
  compact: one mode explains 68–76%.
* The lungs reach 5–6 mm. On organs about 250 mm tall, that is roughly 2% of their size.
  The worst held-out right lungs are 13.6 mm (s0587) and 13.0 mm (s0645).
* For the lungs, specificity is about twice generality. With about 100 shapes in a 7–10
  dimensional space, a random sample sits farther from any one training lung than an unseen
  real lung does. Whether such samples are still plausible anatomy is §6's question.

## 6. Validity of generated thoraxes

**Checks that can fail** (`shapes.validity`). A deformed template is closed and in one piece
by construction, so the Level 1 hard checks pass almost every sample. Three checks are added,
each calibrated on the 98 joint-model training subjects as "no worse than the worst of them":

* **folding:** the share of faces whose normal points against the mean shape's. Limit: the
  worst training subject plus one percentage point.
* **overlap:** for every ordered pair of entities, the share of one's vertices more than 2 mm
  inside the other. Limit: the same rule.
* **volume:** within the training minimum and maximum, with no margin.

**Control.** The same checks run on the 21 held-out real subjects that have all seven
entities registered. **19 of 21 pass (90.5%).** Both failures are real lungs larger than any
training lung:
- s0115: left lung 3956 mL against a maximum of 3870;
- s0657: lungs 4057 and 4795 mL, plus 12% of its myocardium inside its left lung.

So the checks reject about 1 real subject in 10. That is their false-reject rate, mostly from
the strict volume range.

**Sweep.** 50 samples per spread, the same seeds at every spread, weights clipped at 3 SD:

| spread (SD) | pass every check | fail folding | fail overlap | fail volume |
| --- | --- | --- | --- | --- |
| 0.5 | **100%** | 0% | 0% | 0% |
| 1.0 | 78% | 0% | 14% | 10% |
| 1.5 | 40% | 12% | 38% | 32% |
| 2.0 | 18% | 40% | 50% | 54% |
| 2.5 | 10% | 50% | 58% | 70% |
| 3.0 | 6% | 58% | 58% | 78% |

**Usable spread** (pass rate at least 95%): **0.5 SD**. The viewer now samples there, and
every generated scene records its own validity verdict.

**Reading.**

* **At 1 SD**, a spread at which samples are distributed like the training subjects, 78% pass
  against 90.5% of real held-out subjects. The gap of about 12 points is overlap. The commonest
  failure is the left atrium pushed 2–4% into the right lung, and sometimes into the left
  ventricle. Independent linear modes produce combinations that real anatomy does not.
* **Lung folding** appears from 1.5 SD: 3–8% of faces turned against the mean. Large lung
  modes fold thin regions.
* **Variety at 0.5 SD is limited.** Each weight is half a standard deviation. Generated
  thoraxes are plausible, but closer to average than real people are.

## 7. Control: resizing one structure

**Design** ([ADR-L2-001](adr/ADR-L2-001-resize-through-the-shape-model.md), with Amendment 1).
- "Make the left ventricle 20% larger" is read by a rule; the language model can emit the
  same operation.
- The structure's current shape is scaled about its centre and projected onto its own shape
  model, with weights clipped at 3 SD.
- The result is checked against every other modelled structure with the validity limits,
  judged as "no worse than real anatomy, nor than this subject already is".
- If it passes, only that structure's mesh changes, and the AWR records the size.
- If it fails, nothing changes, and the reply gives the reason and the largest change that
  does pass.

**On a real heart (s0029):**

| request | result | largest that passes |
| --- | --- | --- |
| left ventricle −15% | **applied**: −15.0% size, −38.7% volume, 0.6 mm from plain scaling | – |
| right lung −10% | **applied** | – |
| left ventricle +15% | refused: through the myocardium and left atrium | +2% |
| right ventricle +10% | refused: into the right atrium | +2% |
| left atrium +10% | refused: into the left ventricle | +5% |
| right atrium +20% | refused: into the right ventricle | +5% |
| left lung +10% | refused: volume above the training maximum | +6% |
| myocardium +5% | refused: into the right ventricle | none |

Evidence: [before](evidence/2026-10-16_resize_s0029_lv_before.jpg),
[after −15%](evidence/2026-10-16_resize_s0029_lv_minus15.jpg),
[+15% refused with the offer](evidence/2026-10-16_resize_s0029_lv_plus15_refused.jpg).

**Two alternatives were tested for growth, and both fail.**
- **The joint model, conditioned on the target size.** It does not grow the LV alone. Its
  first mode is overall size, so the whole thorax grows about 11%, and it still collides
  with the lungs.
- **The LV and the myocardium together.** At +5% they collide with the left atrium and right
  ventricle.

**Finding.** A linear statistical shape model, edited one structure at a time, can shrink a
cardiac structure across a wide range. It can grow one by only a few percent, because the
chambers are packed against each other and against the lungs. Real enlargement deforms the
neighbours, and these models do not represent that. This is the clearest argument so far for
a deformation-aware model (Stage 2b).

## 8. Problems found, and what was done

Each was found before or during the build, and each is in the notebook with its evidence.

| when | problem | consequence if missed | fix |
| --- | --- | --- | --- |
| code review | one exception would end a build of hundreds of subjects, and a failed alignment could register entities unaligned | hours lost, or silently wrong shapes | per-subject exclusion with the error; a scene counts as loaded only once aligned |
| probe, 3 subjects | the chambers' settings missed 4 of 6 lungs (up to 17.9 mm) | most lungs excluded | placement, finer lung templates, looser stages |
| template check | the left ventricle's template carried a two-face sliver | every generated LV "not watertight, two pieces" | keep the largest piece; record topology |
| validity design | touching neighbours counted as overlap (58% of the LV "inside" the myocardium) | a meaningless overlap check | count only vertices more than 2 mm inside |
| machine restart | the build stopped at 114 of 200 | two hours redone | `--skip-built` resume |
| first look in the viewer | generated lungs invisible at the default level of detail | "it works" claimed on a broken view | the `drawn` fallback between shell and parts |
| first look in the viewer | generated scenes written under the real subjects | samples counted as scans by the next quality run | written beside the model instead |
| validity sweep | the viewer sampled at 1 SD, where 22% of samples fail the validity checks | implausible thoraxes shown as generated anatomy | sample at the usable spread (0.5 SD), and record each sample's own verdict |
| resize on real anatomy | every first request refused; checks ran on raw meshes against limits calibrated on correspondence | a feature that says no without saying what would work | checks on correspondence; "no worse than the subject already is"; refusals offer the largest change that passes |
| demo in the viewer | a typed "reset the scene" kept resized geometry while the Reset button undid it | a reset view over a resized heart, and confusing refusals | a typed reset undoes resizes too |

## 9. Limitations

* **Linear models.** They vary shape as the training population did, along at most 17 joint
  directions. Rare anatomy, pathology and children are outside them. TotalSegmentator's
  subjects are clinical CTs from one hospital network.
* **Seven entities.** Vessels, lobes and the trachea are not modelled. A generated scene has
  no geometry for them, and says so.
* **Lung templates** miss thin wedges and recesses (3–8% of the surface beyond 5 mm). The
  right-lung template has a handle inherited from its reference.
* **Surface fit is not anatomical correspondence.** It is the measurable part. Landmark-based
  checks were not made.
* **One split, one seed.** No cross-validation yet. The numbers above carry that uncertainty.
* **Generated variety is limited.** Generation is validated only at 0.5 SD. At 1 SD, one in
  five samples has structures running into each other.
* **The volume check is strict.** It rejects about 1 real held-out subject in 10, both times
  for lungs larger than any training lung.
* **Resizing can grow a structure only a little.** On s0029, growth is limited to +2 to +6%.
  A deformation-aware model is needed for more.

## 10. Reproduce

```bash
python -m assets.fetch --out datasets/raw/totalsegmentator_v1 --subjects 200 --max-mb 450
python -m assets.quality --build-from datasets/raw/totalsegmentator_v1 --scenes datasets/processed/level1/real
python -m shapes.build --scenes datasets/processed/level1/real --out datasets/processed/level2a
python -m shapes.validity --models datasets/processed/level2a
python -m viewer.server --scene datasets/processed/level1/real/s0029 --model datasets/processed/level2a/models/joint.npz
```

Code: branch `level2a-shapes-and-docs` (PR #7). Data and models are local. They are derived
from CC BY 4.0 data, and the attribution travels in every manifest (ADR-L1-004).
