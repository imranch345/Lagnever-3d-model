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

## 2026-10-16 (13:36) — A dashboard over the whole project

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

## 2026-10-16 (14:17) — Validity of generated thoraxes

- **Control.** 19 of 21 held-out real subjects pass the validity checks (90.5%). Both
  failures are lungs larger than any training lung, so the strict volume range is the checks'
  false-reject source.
- **Samples, 50 per spread:**

  | spread | pass rate |
  | --- | --- |
  | 0.5 SD | **100%** |
  | 1.0 SD | 78% |
  | 1.5 SD | 40% |
  | 2.0 SD | 18% |
  | 3.0 SD | 6% |

  **Usable spread: 0.5 SD.**
- **Why samples fail.**
  - At 1 SD, mostly overlap: the left atrium pushed 2–4% into the right lung.
  - From 1.5 SD, lung folding.
  - Independent linear modes make combinations real anatomy does not.
- **Action.** The viewer now samples at 0.5 SD, and each sample records its own validity
  verdict. Seed 11 at 0.5 SD passes both the quality and validity checks.

## 2026-10-16 (14:57–15:25) — Resize on a real heart: a negative result, then a working design

- **Second machine restart at about 14:24.** Step 21 resumed at 14:57; point 200's branches
  05–07 rerun from their saved state.
- **First real test, s0029.** Every request was refused: LV ±15%, RA +20%, left lung +10%.
  "62% of the LV inside the myocardium."
- **Alternatives tested, and both failed:**
  - joint-model conditioning grows the whole thorax by about 11% and still collides;
  - LV with myocardium collides with the LA and RV at +5%.
- **Finding.** A linear shape model edited one structure at a time can shrink a structure
  across the range, but grow it only a few percent.
- **Design amended** (ADR-L2-001, Amendment 1):
  - checks on correspondence meshes;
  - "no worse than the subject already is";
  - refusals offer the largest passing change;
  - a typed reset undoes resizes.
- **Result on s0029:**
  - applied: LV −15% (−15.0% size, −38.7% volume, 0.6 mm from plain scaling) and right
    lung −10%;
  - largest increases that pass: LV +2%, RV +2%, LA +5%, RA +5%, left lung +6%;
  - myocardium: no change passes.
- **Evidence:** before and after screenshots and the refusal with its offer, in `docs/evidence/`.
- **Tests:** a refusal's offer passes, and one percent more does not. An existing overlap
  does not block a change that eases it. A typed reset matches the button. A sample carries
  its validity verdict.

## 2026-10-16 (16:14) — Full test suite on PR #7

- **Two environments** run the suite, so that every test has its dependencies. No new package
  was downloaded, and the frozen research environment was not modified:
  - **Product venv** (assets, shapes, language, viewer, dashboard, core): **607 passed**.
  - **Research venv** (torch 2.14; Steps 7–20, whole-organ corpus), on this branch's code:
    **556 passed** at first. The other 6, all Step 14, need a frozen Step 10 checkpoint that
    lives only in the research checkout's run folder. With that one folder linked
    read-only, **all 16 Step 14 tests pass**.
  - **Total: 1169 tests, all passing.**
- **Two real failures found and fixed** (`fed0003`).
  - `test_datasets` still asserted the Step 4 state: nothing ever downloaded, and
    processed/ text only. Level 1's approved CC BY 4.0 data broke both.
  - The tests now enforce what the dataset rule actually requires:
    - every raw source has a recorded licence, URL and attribution;
    - binary files appear only in the derived level1/ and level2a/ folders, which carry
      their provenance.

## 2026-10-16 (16:37) — Which sampler: plausibility against novelty

- **Question.** Can generated thoraxes be both plausible and as varied as real people? So
  far, 0.5 SD passed every check but sat close to the training set, and 1 SD failed one
  sample in five.
- **Novelty measured** as the mean vertex distance to the nearest training subject, median
  over 50 samples:
  - a new real person (held out): 9.1 mm;
  - Gaussian 0.5 SD: 6.7 mm;
  - Gaussian 1 SD: 9.05 mm;
  - kernel density around training subjects: 4.5 mm at h = 0.25, 6.9 mm at h = 0.5.
- **Negative result: kernel density.**
  - At h = 0.25, 88% pass, but each sample is a near-copy of a training subject (4.5 mm).
  - At h = 0.5 it is no more novel than Gaussian 0.5 SD, and passes less often (86%).
  - Rejected.
- **Gaussian 1 SD, filtered by validity.**
  - 40 of 50 draws kept.
  - Kept samples are 8.8 mm from training, about a new person's 9.1 mm.
  - |b| of kept samples is 4.03, against 3.96 for real training subjects.
- **Decision** (ADR-L2-002). `write_valid_sample_scene`:
  - draws at 1 SD from the seed's own sequence and keeps the first valid draw;
  - after 6 rejected draws, falls back to 0.5 SD;
  - records every rejected draw in the manifest.
  The viewer uses it. Tested: every shown sample passes, a seed repeats exactly, and the
  fallback is recorded.
- **Also.** Resize now builds neighbour meshes once per request (`c6ac22e`). A profile puts
  98% of check time in trimesh's pure-Python ray test. A faster inside test would change the
  overlap measure and need recalibration, so it is deferred.

## 2026-10-16 (18:22) — Step 21 complete: decision A, fixed from initialisation

- **Stage 2a** (branch points 200 and 0) finished at 18:22. Every point is unanimous:
  **0, 200, 400, 600 and 700 each 8/8**, and Step 20's 805 is 8/8 too. That is 48 of 48
  branches in this step.
- **Integrity, all clean:**
  - every reference reproduces frozen seed 9 exactly: 0 at 43/43, 200 at 39/39, 400 at
    35/35, 600 at 31/31, 700 at 29/29;
  - one starting state per point, and 8 distinct orderings at every point;
  - the replay is identical;
  - **182/182** frozen checksums at the end;
  - `test_splits_read: []`.
- **Decision under the preregistered rule: A, fixed from initialisation.** H15 is supported.
  H16 is not supported for seed 9.
- **Predictions:**
  - 1–3 confirmed;
  - **4 was wrong**: step 0 was expected to split, and it is unanimous;
  - 5 holds broadly, not cleanly: escape timing spreads from a 30-step range at 600 to 160
    at 200.
- **Seen, not claimed.** Ordering moves *when* the escape happens, between steps 700 and
  865 across all branches, against seed 9's own 830–835. It never moves *whether* it
  happens, nor where training ends (20.93–21.07°).
- **Next, by §12:** Step 22 branches seed 5, a non-escaper, at step 0. It is preregistered
  first; nothing has started.
- **Evidence:** `docs/STEP_21_BASIN_ENTRY_REPORT.md` and
  `experiments/runs/step21/summary.json`. The README research table and RESULTS R7 are
  updated.

## 2026-10-16 (23:50) to 2026-10-17 (00:10) — Step 22 preregistered, coded, tested and started

- **Question.** Step 21 fixed seed 9's escape at initialisation. Is the *non*-escape of seed 5
  fixed too? And is a seed's outcome its weights or its data order, the question Step 15 left
  open?
- **Preregistration** (`docs/STEP_22_MIRROR_PLAN.md`, PR #10), written before any code:
  - **Arm 1:** seed 5 from step 0, under the eight orderings that all escaped from seed 9's
    step 0;
  - **Arm 2:** a crossed swap (seed 5's weights with seed 9's batch order, and the reverse),
    with identity controls;
  - **decision rule:** A′ (both modes fixed), B′ (order rescues seed 5), C′ (the original
    order was the accident) or invalid;
  - **four predictions**, including "Arm 1 = none, all stuck".
- **Checked before code.** Step 16 ran at 4 threads, so seed 5's frozen trajectory can be
  reproduced bit-exactly. The data loader is seeded with the run seed, so loader seed *s*
  from step 0 is run *s*'s batch order. The model's own random draws follow the weights'
  seed: the swap is batch order only.
- **Code.** Steps 20 and 21 gained a seed parameter that defaults to 9, so nothing earlier
  changed. Their tests pass unchanged. `experiments/step22/mirror.py` and 24 tests, including
  a test that the swap changes the first batch to the other seed's and keeps the weights.
- **Full suite: 1232 passed** (5 skipped: asset tests without their extra). The frozen
  checksums were 182/182 before the start.
- **Started at 00:10:** capture of seed 5 to step 800, then 14 continuations, then the
  Stage 2 gate.
- **Housekeeping.** Free disk is down to 8.7 GB, from 40+ GB on 2026-10-12. Step 22 needs
  about 0.2 GB, so it is safe against its 2 GiB stop condition, but the machine is getting
  full.
- **00:20, capture valid.** Seed 5 reproduces its frozen Step 16 trajectory exactly at **17 of
  17** shared checkpoints (0–800), and its step-0 state differs from seed 9's.
- **00:20, first launch failed, then fixed.**
  - **Symptom.** Every continuation exited at once, refused by the runner's own
    declared-run check.
  - **Cause.** The check keyed runs by name alone, and `branch_05` exists at points 0, 400 and
    800, so the point-800 entry shadowed point 0.
  - **Why the tests missed it.** They had checked that a wrong run is refused, not that every
    right run is accepted.
  - **Fix** (`3ffb62f`). Whole runs are compared, with tests for all 32 declared runs.
  - **Impact.** No continuation produced a result, and the capture is unaffected. Recorded in
    `sweep.log` ("=== STOPPED ...", "=== RESUMED ...").
- **00:21, continuations running.** 14 at most 3 at a time, about 4 h.
- **2026-10-17, interruption.**
  - **What stopped.** Runs stopped at about 02:53. Branches 05–07 were near step 830 of 1200.
  - **Restart.** The machine restarted at about 15:14 and was on battery: 7% at 21:20, with
    an "early" battery warning.
  - **Kept.** Six results completed before the stop: the reference and branches 00–04.
  - **Resume.** A watcher resumes the chain only once the machine is on AC power. Interrupted
    runs restart from the saved step-0 state, as in Step 21, and completed results are kept.

## 2026-10-18 — Step 22 complete: decision B′, order can rescue a non-escaper

- **Run.** The chain resumed on AC power at 21:19 on 2026-10-17. Arms 1 and 2 finished, the
  gate found Arm 1 split, so Stage 2 (400 and 800) ran as fixed in the plan.
  `CHAIN EXITED` came before 05:42 on 2026-10-18.
- **Integrity, all clean:**
  - capture 17/17, and references 25/25, 17/17 and 9/9 against frozen seed 5;
  - both identity controls exactly equal their references;
  - one starting state and 8 distinct orderings at 0, 400 and 800;
  - the replay is identical;
  - **182/182** frozen checksums before and after.
- **Arm 1** (seed 5 from step 0): **1/8 escape**. Loader 2006 escapes at step 950 and ends
  at 21.06° with usage 2.37°, a full escape. Loader 2007 is in transition (0.615°). Six are
  stuck. Verdict: **split**.
- **Stage 2:** from 400, 0/8; from 800, 0/8. Every branch is stuck, including ordering
  2006. **Seed 5's non-escape is sealed in (0, 400].**
- **Swap:**
  - seed 9's weights with seed 5's order escape at 750, 85 steps earlier than with their
    own order;
  - seed 5's weights with seed 9's order stay stuck.
  The outcome follows the weights in both directions.
- **Decision B′.** H18 is supported for seed 5; H17 is not.
- **Predictions:**
  - 1 confirmed;
  - **2 was wrong**: "none" was expected and the result is split;
  - 3 confirmed;
  - 4 confirmed: the one escape is late, at 950.
- **Meaning.** The initial weights bias the outcome strongly in both seeds, but the bias is
  absolute only for seed 9, under the orderings tried. Seed 5's outcome is sealed within its
  first 400 steps.
- **Next.** For the first time, branches that share a start and differ in outcome exist. A
  candidate Step 23 narrows (0, 400] and compares them. It is to be preregistered first.
- **Evidence:** `docs/STEP_22_MIRROR_REPORT.md`, `experiments/runs/step22/summary.json`,
  README research table, RESULTS R8.

## 2026-10-18 — Level 2b begins: the user's decisions, and the plan

- **The user's decisions:**
  - **USD 700** of cloud GPU budget;
  - the Claude API is available;
  - investors later: make the product stronger first.
- **Hardware here:** Apple M2 with an 8-core GPU (PyTorch MPS works), 16 GB RAM, 21 GB free
  disk. Development runs on this Mac at no cost, and the GPU budget is spent only on full
  training runs.
- **Plan** (`docs/LEVEL_2B_PLAN.md`). A neural field labels every 3D point with exactly one
  structure. Structures then cannot overlap, by construction, so a growing structure must
  push its neighbours aside.
  - **Measured against Level 2a** on the same held-out 26 subjects:
    - held-out fit no worse;
    - at least 90% of samples valid, without filtering, at real-person novelty;
    - LV growth of at least +15% on s0029, against 2a's +2%.
  - **Milestones with hard caps:** M1 $0 (Mac), M2 ≤ $50, M3 ≤ $100, M4 ≤ $300, and a $250
    reserve. Every session goes in `docs/LEVEL_2B_SPEND.md`.
- **The Claude key is kept out of chat and out of the repository.** The viewer and dashboard
  read `~/.lagnav/anthropic.env`, which the user creates. The loader:
  - refuses a file others can read;
  - sets only the key and the language switch;
  - lets the shell override the file;
  - reports variable names, never values.
  Tested.
- **First live test of the Claude language path** (`claude-opus-5-5`, effort low). The key was
  read from the user's private file and never printed. Eleven requests, 2–4 s each:
  - **understood correctly:**
    - "let me see inside the heart": isolate the heart, walls at 30% opacity, show the
      internal structures;
    - "only the right side of the heart": RA, RV and tricuspid valve;
    - "make the lungs see-through…";
    - "reset everything";
  - **Hinglish:**
    - "dil ke left ventricle ko 10% chhota karo" became `resize heart.left_ventricle -10`;
    - "phephde chhupa do aur sirf dil dikhao" became hide the lungs, isolate the heart;
  - **descriptions:** "the chamber that pumps blood to the whole body" became the left
    ventricle;
  - **refusals:** "show me the spleen" became "The spleen is not one of the structures in
    this scene…".
- **Bug found by the live test, and fixed.**
  - **Symptom.** "Show me X" sentences never reached Claude when X was a description or
    absent anatomy. The rule parser recognised the form, failed to resolve the name, and
    stopped there.
  - **Fix.** When the rules fail only on an unresolved name and the language model is
    enabled, the request goes to the model. A failed rule command changes nothing, so this is
    safe.
  - **Tests:** the fallback; a refusal for absent anatomy; and rules-only behaviour without a
    model.
- **Level 2b M1, on this Mac, $0 spent:**
  - **Environment.** PyTorch 2.14 is installed in the product environment, with the user's
    approval. The M2 GPU works through MPS.
  - **Data** (`fields.data`). 200k labelled points per subject for all 131 usable subjects,
    60% near label boundaries, in Level 2a's aligned frame. Verified on two real subjects:
    94–96% of points labelled k lie inside entity k's Level 1 mesh.
  - **Model and training** (`fields.model`, `fields.train`):
    - a label field with one code per subject;
    - class-weighted cross-entropy;
    - atomic checkpoints every 500 steps, with exact resume. A test proves a run stopped and
      resumed is bit-identical to an uninterrupted one. This was added after the first full
      run started on a 35% battery.
  - **Evaluation** (`fields.evaluate`). Held-out code fitting, grid decoding and marching
    cubes. Level 2b and Level 2a are both scored by surface distance to the real Level 1
    meshes, the same rule for both.
  - **Bugs found by tests.** Decoded meshes came out inside-out, with negative volume. Fixed
    as Level 1 does.
  - **First full prototype.** 105 training subjects, 4000 steps, started 08:16, about 1.6
    steps/s on MPS. Evaluation on the 26 held-out subjects follows automatically.

## 2026-10-18 (09:10) — Level 2b, first full prototype: behind Level 2a, and why

- **Run.** 105 training subjects, 4000 steps on the Mac's M2 GPU in 43 minutes. Probe
  accuracy plateaued near 69% from about step 2000.
- **Held-out (26 subjects).** Symmetric mean surface distance to the real meshes, mm, 2b
  against 2a scored by the same rule:

  | structure | 2b | 2a |
  | --- | --- | --- |
  | LV | 1.42 | 0.89 |
  | myocardium | 1.35 | 0.94 |
  | LA | 2.10 | 1.39 |
  | RA | 2.36 | 1.60 |
  | RV | 2.28 | 1.24 |
  | left lung | 6.87 | 3.80 |
  | right lung | 7.36 | 4.71 |

  **The first prototype is 1.5–2× worse than Level 2a.**
- **Diagnosis, measured:**
  - **Underfitting is the cause.** The model reproduces some *training* subjects badly, for
    example s0024 with RV 9.1 mm and lungs around 10 mm. Each subject was seen only about 300
    times.
  - **Held-out fitting is not the bottleneck.** 1500 fitting steps instead of 300 change
    little.
  - **Fragments do not drive the distance.** The largest piece alone scores about the same.
    Decoded lungs do come in many pieces, though (up to 57), which generation must clean.
- **Consequence.** The recipe is right, and the scale is not. This is M2's purpose: far longer
  training on a rented GPU. On the Mac, 50k steps would take about 9 hours; on a rented RTX
  4090, roughly 30–45 minutes. The user is asked to set up the GPU account.

## 2026-10-18 (10:20) — Level 2b M2: first cloud GPU, benchmark, long run started

- **Code-LR sweep on the Mac,** 2000 steps each, otherwise the prototype's settings:
  - codes LR 1e-3: loss 0.474, probe accuracy 0.694;
  - codes LR 1e-2: loss 0.452, probe accuracy 0.707.

  Adopted 1e-2.
- **Cloud account set up by the user.** Google Cloud project `lagnever-ai-app-n`, Compute
  Engine API enabled, and a GPU quota of 1 (`GPUS_ALL_REGIONS`) requested and approved.
- **The machine.** The user said yes to the plan and its estimate. VM `lagnav-2b-m2`:
  - g2-standard-8 with 1× NVIDIA L4 (23 GB), spot, us-central1-a;
  - Google's Deep Learning image with PyTorch 2.9.1, CUDA 12.9 and driver 580.
- **Safeguards:**
  - it deletes itself after 6 hours;
  - no service account or API scopes, because the project also hosts production Firebase
    data;
  - a dedicated SSH key in instance metadata only;
  - the Mac pulls checkpoints every 4 minutes, so preemption costs at most one interval.

  Ledger: [`LEVEL_2B_SPEND.md`](LEVEL_2B_SPEND.md).
- **Code change for the GPU** (`fields.train`). The whole point set (~0.4 GB) now lives on the
  device, and a step indexes only the drawn points. Before, every step copied the drawn
  subjects' full 200k-point arrays on the CPU. An optional `--tf32` flag allows
  TensorFloat-32 matmuls on NVIDIA, and the checkpoint records it. All 10 Level 2b tests pass,
  including bit-exact resume.
- **Benchmark on the L4,** one training step, 105 subject codes:

  | hidden | batch (subjects × points) | FP32 ms/step | TF32 ms/step | peak memory |
  | --- | --- | --- | --- | --- |
  | 256 | 8 × 8192 | 40.8 | 30.3 | 1.1 GB |
  | 512 | 8 × 8192 | 111.7 | 67.0 | 2.2 GB |
  | 512 | 16 × 16384 | – | 277.9 | 8.8 GB |
  | 768 | 16 × 16384 | – | 521.0 | 13.1 GB |

  The prototype's shape takes 30 ms per step here, against about 650 ms on the Mac's MPS:
  **about 21× faster.**
- **Run `m2-h512` started at 04:50 UTC:**
  - hidden 512, 8 layers;
  - 40,000 steps, 10× the prototype's 4000;
  - codes LR 1e-2, TF32;
  - 74 s per 1000 steps, so about 49 minutes in all;
  - at step 1000, probe accuracy 0.660.

## 2026-10-18 (12:10) — Level 2b M2, run 1: the lungs pass Level 2a, the heart does not

- **Run `m2-h512` finished.** 40,000 steps in 48 minutes on the L4. Final probe accuracy is
  0.930 against the prototype's 0.69, with training loss 0.132.
- **Two measurement checks before reading the numbers:**
  - **Smooth meshing does not help.** Re-scoring the prototype with surfaces at the margin's
    zero level gives 0.02–0.1 mm *larger* distances than the binary meshes. The voxel
    staircase was not the gap. Binary remains the headline.
  - **The yardstick's floor is low.** A perfect field (the true CT labels, decoded on the same
    2 mm grid and meshed the same way) scores 0.35–0.47 mm against the Level 1 meshes. Every
    millimetre above that is the model's.
- **A failure mode found and fixed in the evaluation: stray islands.**
  - **Symptom.** Run 1's *mean* surface distances came out worse than an interim look at step
    12,000 (LV 3.15 against 1.42 mm), while medians were fine.
  - **Cause.** The fitted field leaves small islands of a structure elsewhere in the subject's
    data region, around 1000 voxels at most. For s0494, one LV island made 17.10 mm out of
    what is 1.44 mm with the largest piece only. The islands sit inside the data box, not in
    empty grid.
  - **Fix.** The evaluation now also scores each structure after `keep_largest` (each modelled
    structure is one piece in a real thorax), reports piece counts, and summarises by mean
    and median. Raw scores stay reported. Generation will clean the same way.
- **Held-out, 26 subjects,** surface distance in mm (mean / median). Prototype figures are raw;
  2b's are after cleanup.

  | structure | prototype | M2 run 1 | Level 2a |
  | --- | --- | --- | --- |
  | left lung | 6.87 / 6.27 | **3.61 / 2.88** | 3.80 / 3.58 |
  | right lung | 7.36 / 6.48 | **3.93 / 2.74** | 4.71 / 4.05 |
  | LV | 1.42 / 1.35 | 1.50 / 1.45 | 0.89 / 0.82 |
  | myocardium | 1.35 / 1.27 | 1.51 / 1.44 | 0.94 / 0.90 |
  | RV | 2.28 / 2.19 | 2.04 / 1.87 | 1.24 / 1.20 |
  | LA | 2.10 / 2.08 | 2.48 / 1.97 | 1.39 / 1.25 |
  | RA | 2.36 / 2.09 | 2.42 / 2.06 | 1.60 / 1.43 |

  - **The lungs now beat Level 2a** on both mean and median, for both lungs. This is Level
    2b's first structure-level win.
  - **The heart did not move.** Training accuracy rose from 0.69 to 0.93, yet held-out heart
    distances stayed at the prototype's level.
  - **Reading.** For the heart, the limit is now generalization to a new person, not fitting
    the training set: the network learns 105 hearts rather than "a heart".
- **Heart structures are under-represented in training.** The five heart structures hold 8.7%
  of the points, each chamber 1–2%. The lungs and "nothing" hold the remaining 91%.
- **Next runs, one change each, on the same GPU:**
  - **Run 2 (`m2-h512-focus`).** Half of every step's points come from the box around the
    heart (± 10 mm). New option `--focus-fraction`.
  - **Run 3 (`m2-h512-focus-prior`).** Run 2 plus a code prior 100× stronger (1e-2), to keep
    codes in a compact, smooth region that a new person's fit can reach. New flag
    `--code-prior`.

## 2026-10-18 (13:05) — Level 2b M2: heart focus helps; fit trade-off found; part fields; paused

- **Run 2 (`m2-h512-focus`): half of every step's points from the box around the heart.**
  It beats run 1 on every heart structure and loses nothing on the lungs. Held-out, 26
  subjects, after cleanup, mean / median mm:

  | structure | run 1 | run 2 (focus) | Level 2a |
  | --- | --- | --- | --- |
  | LV | 1.50 / 1.45 | **1.32 / 1.22** | 0.89 / 0.82 |
  | myocardium | 1.51 / 1.44 | **1.33 / 1.22** | 0.94 / 0.90 |
  | RV | 2.04 / 1.87 | **1.83 / 1.67** | 1.24 / 1.20 |
  | LA | 2.48 / 1.97 | **2.10 / 1.89** | 1.39 / 1.25 |
  | RA | 2.42 / 2.06 | **2.14 / 1.90** | 1.60 / 1.43 |
  | left lung | 3.61 / 2.88 | **3.25 / 2.48** | 3.80 / 3.58 |
  | right lung | 3.93 / 2.74 | 3.44 / 2.91 | 4.71 / 4.05 |

  The lungs stay ahead of 2a. The heart improves by 10–15%, but is still behind 2a.
- **Weighted held-out fitting reveals a trade-off.** Run 1 was refitted with training's class
  weights (`--fit-weighted`):
  - the heart improves: LV 1.50 → 1.14, myocardium 1.51 → 1.22, RV 2.04 → 1.95 (means);
  - the lungs get worse: 3.61 → 4.09 and 3.93 → 4.61.

  One code cannot fit a new person's heart and lungs at once. Whatever the fit stresses
  improves, at the other's cost. The 105 training thoraxes do not teach the single,
  entangled code new combinations of heart and lungs.
- **Response: part fields** (`fields.model.PartField`, `--parts heart. respiratory.`):
  - each part has its own code and trunk;
  - the trunks combine as a product of experts over one softmax, so non-overlap still holds
    by construction;
  - a test proves a part's code cannot move the other parts' decisions between their labels
    and "nothing";
  - for the product, "make the heart bigger" can then leave the lungs untouched.

  19 tests pass.
- **Runs in flight when the user paused:**
  - **Run 3 (`m2-h512-focus-prior`, code prior 1e-2):** finished. Training accuracy 0.900
    with code norm 0.98, against run 2's 5.4. Its held-out evaluation was running on the
    Mac at the pause.
  - **Run 4 (`m2-h512-focus-z128`, run 2 with a 128-number code):** stopped, resumable
    from step 6000.
  - **Run 5 (`m2-parts-h384`: parts, 2 × 64-number codes, hidden 384 per trunk, focus
    0.5):** stopped, resumable from step 2000. It reached probe accuracy 0.672 at step
    1000, the best of any run at that point.

  Runs 4 and 5 have the same total code size, so together they separate "more code" from
  "code split by part".
- **Paused, $0 running.** VM `lagnav-2b-m2` ran 2.83 h and was deleted at 07:34 UTC after
  every run was pulled to the Mac. The estimated cost is about $1 (see the ledger).
- **To resume:**
  1. New VM, same flags; upload code and data.
  2. Copy back `runs/gpu/m2-h512-focus-z128/state.pt` and `runs/gpu/m2-parts-h384/state.pt`.
  3. Start the same commands. Training continues from the saved step exactly.
