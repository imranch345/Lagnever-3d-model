# Results

The curated summary of what Lagnav 3D has shown so far, for papers, grant applications and
investor material. Every number points to where it comes from. The dated, complete record,
including runs that failed, is [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md). This page is updated
whenever a headline result changes.

*Last updated: 2026-10-16.*

## What the project is

Lagnav 3D is research into **language-controlled 3D anatomy**. A structured world model, the
AWR (typed entities, three relationship graphs, persistent scene memory), is the source of
truth. Geometry is a view of it. A user's sentence ("hide the lungs", "show the left ventricle
at higher detail", "animate blood flow") becomes a validated program over that model rather
than a direct edit to pixels or meshes. Because of that, every change is inspectable,
reversible and explainable.

There are two tracks:

* **Research.** Does a model built around the AWR learn the anatomy's structure? Controlled
  experiments on synthetic corpora.
* **Product.** A working system on real CT data: Level 1 (a controllable heart and lungs),
  then Level 2 (generated anatomy from learned shape models).

Nothing here is medically validated or intended for clinical use.

## Headline results

### Research (synthetic corpora; preregistered)

| # | result | numbers | source |
| --- | --- | --- | --- |
| R1 | Building the model around entities gives part-level control that an equally sized appearance model lacks. | part control 0.931 vs 0.313; untouched-entity drift after an edit 0.000000 vs 0.001303; 3.09M vs 3.10M parameters, 3 seeds | Step 6 report |
| R2 | The entity axis is essential, but the large graph module is not what delivers it. | ownership IoU 0.6339 vs 0.0164 (~450×); a 7.2× smaller graph module matches the 86%-of-parameters one | Step 7 |
| R3 | Rotation is learnable from structure only once its loss is weighted correctly; then only the full typed graph beats the floor. | 21.38° ± 1.03 vs floor 22.65° at λ = 3.0; the rotation term had been 7% of the frame loss | Step 10 |
| R4 | The gap to an identity-and-graph lookup (11.57°) is **not** explained by relation types, per-relation messages, depth, global context or the objective. | shuffling relation types changes rotation by 0.0003°; the objective rewards relations (−82.2%) but the model captures 15.2% of it | Steps 11–14 |
| R5 | Whether training uses the graph is bimodal across seeds, with no measurable precursor. | 6 of 10 seeds use it, 3 do not, 1 is intermediate; no warning signal at 5-step resolution or in gradient direction | Steps 15–19 |
| R6 | **By step 805 the outcome is fixed**: changing the data order does not stop a seed that is about to escape. | 8 of 8 branches escape, at steps 830–840, final rotation 20.988° ± 0.023 | Step 20 |
| R7 | *In progress:* how early is the outcome fixed? | Stage 1 at steps 400, 600 and 700, 8 branches each; no verdict until Stage 1 completes | Step 21 plan |

R5–R7 make a result about training dynamics: in this setting, using relational structure is a
discrete event decided by the optimisation state, and is not visible beforehand in standard
signals.

### Product (real CT)

| # | milestone | numbers | evidence |
| --- | --- | --- | --- |
| P1 | A controllable 3D heart and lungs built from real CT, end to end. | 17 closed surfaces for subject s0029; 11 of 11 left/right checks pass | `evidence/2026-10-15_real_ct_s0029_heart_and_lungs.jpg` |
| P2 | Plain-language control of that scene. | "hide the lungs", level-of-detail changes, isolation, opacity, cardiac-cycle playback, each through validated engine commands | `evidence/2026-10-15_real_ct_s0029_lungs_hidden.jpg`, `evidence/2026-10-16_real_ct_s0029_lod0_organ_shells.png` |
| P3 | Data is efficient to acquire. | 20 subjects for 32.6 MB of a 28.4 GB archive (HTTP byte ranges); 200 subjects later fetched the same way | `assets/fetch.py`, notebook 2026-10-15 and 2026-10-16 |
| P4 | Automated quality gate. | 15 of 20 subjects accepted; each of the 5 rejections has a stated reason (field-of-view cut, missing structure) | `datasets/processed/level1/real/quality_report.json` |
| P5 | An open-vocabulary language layer with a safety boundary. | Rules first; otherwise Claude emits a program restricted by a closed schema of operations and entity ids, validated before anything runs. Off by default; tested without network calls | ADR-L1-005, `language/` |
| P6 | *In progress:* generated anatomy (Level 2a statistical shape models). | Registration on real left ventricles fits at 0.15–0.19 mm mean, in 5–9 s each; model build on ~200 subjects is next | `docs/LEVEL_2_PLAN.md`, `shapes/` |

## How the results are protected

These practices are part of the claim. They are why the numbers can be trusted.

* **Preregistration.** Each experiment's questions, metrics, thresholds and stopping rules are
  committed before its code runs. Changes are recorded as dated amendments, never silent
  edits. Example: Step 21 Amendment 1.
* **Frozen results.** 12 manifests with 182 SHA-256 checksums cover the runs that the reports
  rely on. They were verified after the repository moved machines: 182 of 182 matched.
* **Bit-exact reproduction.** Training is deterministic at a fixed thread count (4). Step 21's
  capture reproduces the frozen seed exactly at 18 of 18 checkpoints. After the RNG fix, its
  continuation reproduces it at 35 of 35.
* **Retractions are published.** Step 7 withdrew a Step 6 conclusion. Step 18 withdrew a Step
  17 finding. Step 21's first Stage 1 was declared invalid, its cause found, and the invalid
  run kept for inspection.
* **Tests.** 1024 automated tests on `main` for the research code, 1076 with Step 21's (PR #5). The product code has its own
  suite: ontology, meshes, fetcher, quality gate, viewer, language and shapes.
* **Provenance on data.** Every real-CT asset records its source subject, label map, dataset
  version and licence (TotalSegmentator v1.0, CC BY 4.0). Generated scenes are marked as
  generated.

## Resources used

* **Compute.** Everything so far has run on one laptop CPU. No GPU, no cloud spend.
* **Data.** Public, openly licensed CT segmentations. No patient data was collected.
* **Paid services.** None so far. The Claude language path needs an API key and has not been
  run against the live API.

## Limitations (stated plainly)

* The research results are on synthetic corpora. Whether they transfer to real anatomy is
  untested.
* The real-CT scenes are segmentation surfaces from one dataset. Volumes and shapes are
  measurements, not clinical values, and have not been reviewed by a clinician.
* The cardiac-cycle animation is semantic, a sequence of named phases. It is not a
  physiological simulation.
* Level 2 (generated anatomy) is not yet demonstrated. Its plan and code exist and are tested
  on synthetic shapes, but the real-data model has not been built.

## Evidence index

| what | where |
| --- | --- |
| Dated record of all work | [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) |
| Screenshots with captions and reproduce commands | [`evidence/`](evidence/README.md) |
| Level 1 plan and status | [`LEVEL_1_PLAN.md`](LEVEL_1_PLAN.md) |
| Level 2 plan | [`LEVEL_2_PLAN.md`](LEVEL_2_PLAN.md) |
| Architecture decisions | [`adr/`](adr/README.md) |
| Research reports, Steps 6–20 | [`step-6-experiment-report.md`](step-6-experiment-report.md), [`step7/`](step7/), `STEP_8_*` to `STEP_20_*` in this folder |
| Step 21 preregistration and amendment | `docs/STEP_21_BASIN_ENTRY_PLAN.md` (PR #5) |
