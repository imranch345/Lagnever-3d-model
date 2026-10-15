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
