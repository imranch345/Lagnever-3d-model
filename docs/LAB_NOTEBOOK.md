# Lab notebook

A dated, append-only record of what was done, what it showed, and where the evidence is. It
exists so that results can later back research papers and investor material without anyone
reconstructing them from chat or memory.

**Rules.**

* Newest entries go at the bottom.
* Every number links to the report, commit or artifact it comes from.
* Negative results, invalidated runs and corrections are recorded as plainly as successes.
* "Synthetic" means the procedurally generated research corpora. "Real CT" means the
  TotalSegmentator v1.0 dataset (CC BY 4.0). Nothing here is medically validated.

Entries before 2026-10-12 are reconstructed from the reports and commit history, with their
original dates.

---

## 2026-09-13 to 2026-09-17 — foundation, first experiment, stress test (Steps 4–9)

**Done.**
- **Step 4:** the Anatomical World Representation (AWR): a typed entity catalogue, three
  relationship graphs and persistent scene memory.
- **Step 5:** a neural architecture designed around the AWR.
- **Step 6:** a PyTorch prototype and the first preregistered experiment.
- **Step 7:** a representation stress test.
- **Steps 8–9:** predicted placement, and a placement-blind floor.

**Results.**
- **First experiment (Step 6), synthetic.** Structured model (3.09M parameters) against an
  appearance baseline (3.10M), three seeds. All four preregistered criteria passed:

  | metric | structured | appearance |
  | --- | --- | --- |
  | part-control success | 0.931 | 0.313 |
  | held-out relationship accuracy | 0.900 | 0.700 |
  | untouched-entity drift after an edit | 0.000000 | 0.001303 |
  | occupancy Chamfer | 0.0137 | 0.0478 |

  Ablations attributed most of the gain to per-entity factorisation, not to the graph encoder.
- **Step 7.** The entity axis is essential: an appearance model of equal size is about 450×
  worse at ownership (IoU 0.6339 against 0.0164). The graph transformer is 86% of parameters,
  yet a version 7.2× smaller matches it. Step 7 also **withdrew** Step 6's graph conclusion:
  the Step 6 corpus had one constant graph, so the comparison could not have come out
  otherwise.
- **Step 8.** Under predicted placement, only one arm clears the relation-blind floor
  (0.8110), on one split of four.
- **Step 9.** A lookup keyed on entity identity places structures at translation error
  0.1605, and three of five Step 8 arms do worse.

**Evidence.** `docs/step-6-experiment-report.md`, `docs/step7/`,
`docs/STEP_8_COMPLETION_REPORT.md`, `docs/STEP_9_COMPLETION_REPORT.md`; commits `0530245`,
`c8c1612`, `66606a0`.

## 2026-09-20 to 2026-09-22 — rotation (Step 10)

**Results, synthetic.**
- Parent-relative placement made placement worse in all 5 arms (+2.9% to +7.8% error).
- On a rotated corpus (mean rotation 61.8°), rotation was not learned above its floor at
  first. A weight study then found the cause: the rotation term was 7% of the frame loss.
- At the frozen weight λ = 3.0, A3 reaches **21.38° ± 1.03 against a floor of 22.65°**
  (+1.27°, test_seen); only the full typed graph does it.
- A lookup on identity plus graph reaches 11.57°. That gap is what Steps 11–20 studied.

**Evidence.** `docs/STEP_10_*.md`; commits `d445e89` to `2aea521`.

## 2026-09-22 to 2026-09-23 — where the relational signal is lost (Steps 11–14)

**Results, synthetic. All four architectural and objective explanations were ruled out.**

| step | question | finding |
| --- | --- | --- |
| 11 | what matters in the graph | Only removing the whole graph matters (+1.83°). Shuffling relation types changes rotation by 0.0003°. |
| 12 | per-relation messages | No help: 21.80° → 21.81°. |
| 13 | depth or global context | Neither: 21.99° and 21.44° against 21.38°. |
| 14 | the objective | Graph conditioning reduces the objective by 82.2%, so the objective asks for relations; A3 captures 15.2% of the available improvement. |

**Evidence.** `docs/STEP_11_*.md` to `docs/STEP_14_*.md`.

## 2026-09-24 to 2026-09-25 — the seed "escape" (Steps 15–20)

**Results, synthetic.**
- **Step 15.** Over 10 seeds the outcome is bimodal: 6 seeds use the graph, 3 do not, 1 is
  intermediate.
- **Steps 16–19.** Nothing measurable precedes a seed's escape into graph use: not at 50- or
  5-step resolution, not with a corrected baseline, not in gradient direction. Successive
  gradients stay aligned at cosine +0.844. The transition spans steps 810–820.
- **Step 20, an intervention.** From seed 9's full state at step 805, **8 of 8 branches with
  different data orderings escape**, at steps 830–840, to a final rotation of 20.988° ± 0.023.
  The outcome is already fixed at 805.

**Corrections recorded.** Step 18 withdrew Step 17's "60-step usage lead" as an artifact of a
sparse baseline. Step 17 found that CPU thread count changes results, so 4 threads became part
of the frozen protocol.

**Evidence.** `docs/STEP_15_*.md` to `docs/STEP_20_CHECKPOINT_BRANCHING_REPORT.md`; commits
`1987b6c` to `9624a11`.

## 2026-10-12 — housekeeping and a verified move

- **README** brought up to Step 20 (`a909492`).
- **ADR-STEP10-003 amended** (`6bf8a99`). The rotation weight is 3.0, not 0.33; two frame terms
  are live; the effective rotation coefficient is 6.0.
- **Repository moved out of iCloud**, where git had stopped working, to `~/code/lagnav-3d`. The
  move was verified:
  - 818 of 818 run files copied;
  - 182 of 182 frozen checksums match across 12 manifests;
  - **1024 of 1024 tests pass** in a rebuilt environment with the original package versions
    (torch 2.14.0, numpy 2.5.3).

## 2026-10-13 to 2026-10-15 — Step 21: when does the escape become fixed?

- **Preregistered** before any code (`eaf5a40`).
- **Runner and tests** (`1995768`): 42 new tests, 1066 in all.
- **Capture valid.** It reproduces frozen seed 9 at 18 of 18 shared checkpoints, and its step-805
  state is bit-identical to Step 20's.
- **The first Stage 1 run was invalid.** Every reference continuation failed to reproduce frozen
  seed 9 after the branch point (1/35, 1/31, 1/29 checkpoints exact).
- **Cause found and fixed** (`0459a9a`, Amendment 1). During the teacher-forcing ramp (steps
  180–719) the model draws with `torch.rand`; saved RNG was never restored on load. Step 20's
  audit had been made at step 805, where nothing draws.
- **Fix verified.** After it, the step-400 reference reproduces frozen seed 9 at **35 of 35
  checkpoints**, escape included. The invalid run is kept, not deleted.
- **Status.** Stage 1 resumed on 2026-10-16 after the session ended; 400 is complete and
  600–700 are running. The verdict is recorded only when Stage 1 completes, per the plan.

## 2026-10-15 — Level 1: a language-controlled 3D heart and lungs

**Built.**
- a thorax domain that includes the heart ontology unchanged;
- the TotalSegmentator label map and the mesh pipeline;
- a byte-range fetcher and the quality report;
- a local viewer driven by the command engine;
- engine fixes, pulmonary circulation, and cardiac-cycle playback.

**Plan:** `docs/LEVEL_1_PLAN.md`. **ADRs:** L1-001 to L1-004.

**Real data.**
- 20 subjects, **32.6 MB fetched out of a 28.4 GB archive** in 102 requests.
- The first quality pass accepted **0 of 20**. The reasons showed two errors in the checks,
  not in the data:
  - chambers were judged against the heart's centre, but the heart is rotated;
  - watertightness was measured after decimation.

  After the fixes, **15 of 20** pass. The five rejections are genuine: field-of-view cuts and
  missing structures.
- **Subject s0029 chosen.** 17 closed surfaces, 11 of 11 side checks pass. Measured volumes,
  not clinical values: left lung 3160.6 mL, right lung 3654.3 mL, heart shell 634.4 mL.
- **Evidence:** `docs/evidence/` (first real-CT scene, lungs hidden, LOD 0, phone layout).

## 2026-10-16 — language layer, Level 2 plan, shape-model code

- **Language layer** (`21161fa`, ADR-L1-005).
  - The rule parser is tried first. Otherwise Claude produces a program over a closed enum
    schema of operations and entity ids; it is validated, then applied through the engine.
  - Off by default. Tested with a fake client; no API calls made yet.
- **Level 2 plan** (`e5a825f`). A statistical shape model first, on CPU; neural models only if
  needed.
- **Shape-model code** (`c1da4b6`, `4ca278f`).
  - Probe on 4 real left ventricles: non-rigid registration fits the surface at **0.15–0.19 mm
    mean** (0.20–0.29 mm from the target side, max 1.2–3.1 mm), in 5–9 s per registration.
  - A finding on alignment: point-to-point ICP between random samples recovers only about 8 of
    10 degrees on a test ellipsoid. The two-stage alignment now used recovers it to within
    0.5 mm.
- **200-subject fetch** for Level 2a, approved by the user.
  - Zenodo answered one HTTP 500 after 161 subjects, so the fetcher now retries transient
    errors with backoff and reuses subjects already on disk (`02fd2a3`, `15440f7`). It resumed.
- **Process correction.** PR #6 had been merged on 2026-10-15, and 12 commits pushed to its
  branch afterwards never reached `main`. They were moved to PR #7, and #6 was annotated.

## 2026-10-16 — Level 2a data in hand

- **Fetch complete:**
  - **200 subjects** with all 14 required masks: 2800 NIfTI files, **278 MB on disk** out of
    the 28.4 GB archive (about 1%);
  - within the approved ~330 MB;
  - the resumed run fetched the remaining 39 subjects (46.6 MB, 209 requests) and reused the
    161 already on disk.
- **Pipeline queued, all niced and detached:**
  - mesh build and quality ranking of all 200 (`assets.quality --build-from`, about 70 s per
    subject on the shared CPU);
  - then shape-model build (`shapes.build`, log `datasets/processed/level2a/build.log`).
- **Step 21:**
  - Stage 1 points 400 and 600 are complete (reference plus 8 branches each); 700 is
    running.
  - Validity: each reference continuation reproduces frozen seed 9 exactly: 400 at 35 of
    35 checkpoints, 600 at 31 of 31.
  - Branch outcomes are not read until Stage 1 completes, per the preregistration.
- **Shape build hardened before the 200-subject run.** A review found two problems that would
  have cost hours:
  - one subject raising during loading, alignment or registration would have ended the whole
    build;
  - a failed alignment after a successful load would have let that subject's remaining
    entities register **unaligned**.

  Such subjects are now excluded with the error, and not cached, so a rebuild retries them.
  Each registration also records its alignment cost, for later diagnosis. Covered by a new test.
- **Lung registration probe, before the build** (`docs/evidence/2026-10-16_lung_registration_probe.md`).
  - **Negative first.** With the settings validated on left ventricles, **4 of 6 real lungs
    missed** the 2 mm fit, one at 17.9 mm. Non-rigid ICP reaches only 15 mm, and lungs vary
    more than that.
  - **Fix.** Per-axis placement before non-rigid ICP, plus a finer template (4000 faces) and
    looser final stages for lungs only. Now **0 of 6 miss** (0.84–1.76 mm). Chambers are
    unchanged at 0.21–0.50 mm.
  - **Known limit.** 3–8% of each lung's surface (thin wedges and recesses) stays more than
    5 mm from the fit.
  - **Covered by a test.** A target 40 mm beyond reach fails without placement and fits with
    it.
- **Template check, before the build.**
  - **Problem.** Decimating the real left ventricle left a two-face sliver beside its surface.
    Every generated LV would then have failed the quality report as "not watertight, 2
    pieces".
  - **Fix.** Templates now keep only their largest piece. All seven are closed surfaces.
  - **Topology is reported, not hidden.** The build report records each template's faces,
    closedness and Euler number. The LV has a pinch (Euler 1) and the right lung one handle
    (Euler 0); a sphere-like surface would have 2. Both come from the reference subject's
    surface, and both are listed as limits.
- **Honest validity needs checks that can fail.** A deformed template is closed and in one
  piece by construction, so the Level 1 hard checks would pass almost every sample and say
  little. The validity sweep therefore adds checks that can fail:
  - folding;
  - overlap between entities;
  - volume against the training range.
- **Machine restart at about 03:20.** All detached jobs stopped. Quality build was at 114 of
  200; Step 21 point 700 at 3 of 9.
  - Both resumed at 10:14. Finished results are kept; unfinished ones rerun from their saved
    state.
  - The quality build gained `--skip-built` (`f1bdcd4`). A subject counts as finished when
    its manifest, the last file written, exists. All 114 had been built by the current code.
- **Validity sweep written** (`shapes/validity.py`), ready for the built model.
  - Three checks that can fail, each calibrated as "no worse than the worst real training
    subject":
    - folding;
    - overlap deeper than 2 mm;
    - volume range.
  - Held-out real subjects run through the same checks as a control.
  - **Correction during design.** Counting any vertex inside a neighbour as overlap flagged
    touching structures: on the reference, 58% of the left ventricle counted as "inside" the
    myocardium it lies against. Requiring 2 mm depth brings that to 0.
  - **Remaining overlap on the real reference.** 10.2% of the right atrium lies more than 2 mm
    inside the coarse right-lung template, and 7.4% of the myocardium inside the left lung.
    Calibration accounts for this level, which exists in real data.
  - About 12 s per thorax.
- **`resize` end to end, ready for the real model** (ADR-L2-001).
  - **Request.** "Make the left ventricle 20% larger" is read by a rule, so it works without
    the language model. The language model can emit the same operation.
  - **Geometry.** The current shape is scaled and projected onto the entity's own shape model
    (weights clipped at 3 SD). Only that entity's mesh changes. The result is checked against
    the calibrated validity limits, and refused with the reason if it fails.
  - **AWR.** Only after the geometry succeeds is the size written to the AWR, as the existing
    `Transform.scale`.
  - **Viewer.** Reset undoes resizes; a new geometry resets sizes and says so.
  - **Design constraint, recorded.** A new parser intent would have changed the research
    neural vocabulary (`ProgramOpType`) and so the frozen results. `resize` is therefore a
    product operation instead.
  - **Tests (sphere populations).**
    - +5% is reached within 1%, and every other mesh is bit-identical.
    - +30% beyond the training volumes is refused, and nothing is written.
    - A neighbour moved against the ventricle blocks it.
    - Groups, unknown names and changes over 30% are refused before any geometry work.

## 2026-10-16 (11:02) — 200-subject quality report

- **131 of 200 subjects usable** (66%). The same rule as Level 1 again chooses **s0029** as
  the reference, out of 200 rather than 20.
- **69 rejected.** A subject can have several reasons:

  | reason | subjects | notes |
  | --- | --- | --- |
  | field of view cuts a lung or the heart | 57 | most often the right lung (54) and its lower lobe (53) |
  | missing structures | 45 | mostly the trachea, pulmonary trunk and right upper lobe: scans that stop below the upper chest |
  | in separate pieces | 22 | mostly the aorta (7), IVC (6) and right upper lobe (5) |

  14 subjects fail on field of view alone and 30 on field of view plus missing structures.
  The CT scans are not bad; they were acquired for other regions. TotalSegmentator is not a
  thorax dataset, and the gate does what it should.
- **Shape build started 11:02**, niced, on the 131 usable subjects. It uses the hardened
  build, lung placement and single-piece templates from earlier today.
- **Evidence:** `datasets/processed/level1/real/quality_report.json` (local data, not in git).

## 2026-10-16 (11:44) — Step 21 Stage 1 complete: unanimous at 400, 600 and 700

**Validity, all clean.**
- Each reference reproduces frozen seed 9 exactly: 400 at 35/35, 600 at 31/31, 700 at 29/29
  checkpoints.
- One starting fingerprint per point, and 8 distinct orderings at every point.
- Determinism replay (400, branch_00) identical at every checkpoint.
- All 12 freeze manifests clean: 182 of 182 files.
- The Stage 2 gate reports no problems.

**Primary outcome** (escapes out of 8, with Step 20's 805 for comparison):

| branch point | escaped | verdict | escape step, min–max (mean) | final validation rotation |
| --- | --- | --- | --- | --- |
| 400 | 8/8 | unanimous | 750–800 (787.5), timed to 50 steps* | 20.975° ± 0.009 |
| 600 | 8/8 | unanimous | 835–865 (855.0) | 21.011° ± 0.032 |
| 700 | 8/8 | unanimous | 820–855 (836.3) | 20.981° ± 0.018 |
| 805 (Step 20) | 8/8 | unanimous | 830–840 | 20.988° ± 0.023 |

\* These escapes fall before the dense 5-step window (805–900), so per §4 they are timed on
the 50-step grid. Their spread is not comparable with the others'.

**Predictions recorded before the run.**

| prediction | result |
| --- | --- |
| 1. references reproduce | **confirmed** |
| 2. 600 and 700 unanimous | **confirmed** |
| 3. 400 unanimous, weakly | **confirmed**. 8/8 is consistent with any escape rate above about 0.69 (one-sided 95%), so "unanimous" means a high rate, not certainty |
| 5. escape-step spread grows as the branch point moves earlier | **not cleanly**. Standard deviation is 8.7 steps at 600 and 12.4 at 700, both above 805's. 400's escapes are at 50-step resolution and cannot be compared |

**Not anticipated in the plan.** Every branch from 400 escapes **earlier** than frozen seed 9
itself (750–800 against 830–835). Ordering moves *when* the escape happens by 50 steps or more.
It has not changed *whether* it happens from any point tested. Every escaped branch ends at
the same rotation to within 0.03°.

**What this licenses.**
- Seed 9's escape survives every ordering tried from step 400 on.
- The pattern including 805 is monotone, so outcome C (non-monotone) is ruled out for
  400–805.
- Per §6 this does **not** show that the state at 400 causes the escape.

**Next, chosen by the preregistered table.** Stage 2a, branch points 200 and 0, 18 runs,
started 11:44. Step 0 decides between A (fixed at initialisation) and B (entered during
training).

**Evidence:** `experiments/runs/step21/point_0{400,600,700}/` and `sweep.log` (PR #5 branch);
the summary was computed with `experiments.step21.basin_entry.summarise`.

## 2026-10-16 (12:59) — Level 2a shape models built on 131 real thoraxes

**Data.** 131 usable subjects, split once with seed 0: **105 training, 26 held out.**
Registrations are cached (917 = 131 × 7).

**Registration (template → subject), surface fit `mean_from_target`:**

| entity | median | 90th pct | max | excluded (> 2 mm) |
| --- | --- | --- | --- | --- |
| left ventricle | 0.24 mm | 0.28 | 0.61 | 0 |
| left atrium | 0.25 | 0.30 | 1.51 | 0 |
| right atrium | 0.30 | 0.36 | 1.76 | 0 |
| myocardium | 0.31 | 0.40 | 1.30 | 0 |
| right ventricle | 0.40 | 0.57 | 1.57 | 0 |
| left lung | 1.13 | 1.70 | 3.54 | 8 |
| right lung | 1.07 | 1.69 | 3.48 | 8 |

Four subjects fail both lungs (s0436, s0494, s0553, s0629). Every exclusion is listed with its
error in `report.json`; none was dropped silently.

**Models (95% variance kept). Held-out generality against the mean-shape baseline:**

| entity | train | modes | 1st mode | held-out error | mean baseline | ratio | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| left ventricle | 105 | 9 | 42% | **1.13 mm** | 4.28 mm | 3.8× | 2.77 mm |
| myocardium | 105 | 9 | 43% | **1.15** | 4.43 | 3.8× | 2.70 |
| right ventricle | 105 | 11 | 49% | **1.49** | 6.24 | 4.2× | 3.44 |
| left atrium | 105 | 5 | 76% | **1.92** | 8.79 | 4.6× | 3.48 |
| right atrium | 105 | 6 | 68% | **2.08** | 9.58 | 4.6× | 4.26 |
| left lung | 100 | 10 | 34% | **5.12** | 20.76 | 4.1× | 12.04 |
| right lung | 101 | 7 | 57% | **6.34** | 26.15 | 4.1× | 11.77 |
| **joint, all 7** | 98 | 17 | 45% | **4.31** | 13.71 | 3.2× | 8.61 |

Generality is the mean vertex error reconstructing an unseen subject. Specificity is the mean
distance from 50 random samples to the nearest training shape.

**Reading.**
- Every model beats predicting the mean by 3.2–4.6× on subjects it never saw.
- The chambers reconstruct unseen hearts to about 1–2 mm.
- Lungs are harder: errors are 5–6 mm on organs about 250 mm tall, so roughly 2% of size.
  The worst held-out right lungs are s0587 (13.6 mm) and s0645 (13.0 mm).
- Specificity is about twice generality for the lungs. With 100 shapes in a 7–10-dimensional
  space, samples sit farther from any one training lung than an unseen real lung does. Whether
  that means implausible is what the validity sweep measures next.

**Not yet shown.** Validity (the sweep is running), resize on real anatomy, and screenshots.

**Evidence:** `datasets/processed/level2a/report.json`, `build.log` and `models/` (local
data); code at `8fc2d04` + `f1bdcd4`.

## 2026-10-16 (13:05) — First generated thorax in the viewer

- **Seed 1** from the joint model passes the quality checks and appears with its banner and
  credits (`docs/evidence/2026-10-16_generated_thorax_seed1.jpg`). "Hide the lungs" works on
  generated anatomy as on real anatomy.
- **Problem found by looking.** On the first try the generated lungs were invisible. At the
  default level of detail the engine shows a lung as its lobes, and a generated scene has
  lung shells but no lobes. The same happens at LOD 0 for the heart, which has no generated
  shell.
- **Fix.** The viewer's state now carries `drawn`. When every visible member of an organ's
  chosen representation lacks geometry, the other representation is drawn. The AWR's
  `visible` is unchanged, and real scenes, which have both, draw exactly as before. This is
  covered by a test.
- **Second fix.** The viewer wrote generated scenes to `real/samples/`, where the next quality
  run would have counted them as scans. They now go to `level2a/viewer_scenes/`.

## 2026-10-16 (13:40) — A dashboard over the whole project

- **Asked for by the user:** "a frontend dashboard where everything is connected".
  `python -m dashboard.server` serves it on 127.0.0.1:8795, local only.
- **Six tabs, read live every 15 s from files on this machine:**
  - Overview: jobs and headline results;
  - Research: Step 21's branch map, escape timing and table;
  - Generated anatomy: funnel, held-out errors, validity, samples;
  - 3D viewer: the viewer itself, embedded;
  - Evidence;
  - Notebook & docs.
- **Nothing is computed twice.** Step 21 is read from each run's JSON, the models from
  `report.json`, and the headline results from `RESULTS.md`. The research code is not
  imported.
- **Problems found while building it:**
  - A job that writes its log only at the end looked "stopped". Job state now also checks
    for the process.
  - Escapes at the same step overlapped in the strip plot. It is now a proper dot plot.
  - On a phone, charts first overflowed the page, then shrank to unreadable text. They now
    scroll inside their card at a legible size.
- **Charts:** the reference palette's first two slots, validated with the dataviz checker
  (all checks pass, light and dark). Each chart has a legend for two series, tooltips, and a
  table view.
- **Tests:** 7 in `tests/test_dashboard.py`. They cover job states, verdicts, whitelists, path
  traversal and every route.
- **Evidence:** `docs/evidence/2026-10-16_dashboard_step21_live.jpg`.
