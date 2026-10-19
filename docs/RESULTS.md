# Results

The curated summary of what Lagnav 3D has shown so far, for papers, grant applications and
investor material. Every number points to where it comes from. The dated, complete record,
including runs that failed, is [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md). This page is updated
whenever a headline result changes.

*Last updated: 2026-10-18.*

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
| R7 | **The outcome is fixed from initialisation.** Seed 9's escape survives every data ordering tried, from every branch point, including step 0. Ordering shifts only *when* it happens. | 48 of 48 branches escape (8 each at steps 0, 200, 400, 600 and 700) plus Step 20's 8 at 805. Escapes fall at steps 700–865 against seed 9's own 830–835, and all end at 20.93–21.07°. Every reference reproduces exactly; 182/182 frozen checksums clean. Prediction 4 (step 0 would split) was wrong | `STEP_21_BASIN_ENTRY_REPORT.md` |
| R8 | **But not for a non-escaper.** From seed 5's initial weights, 1 of 8 orderings escapes. From step 400 none do: its non-escape is sealed early in training, in (0, 400]. Swapping batch orders between seeds 5 and 9 leaves each outcome with its weights. | Seed 5 from step 0: 1/8 escape (step 950, ends 21.06°), 1 in transition, 6 stuck; from 400 and 800: 0/8 each. Swap: seed 9's weights with seed 5's order escape (step 750); seed 5's weights with seed 9's order stay stuck. All references and identity controls exact; 182/182 checksums clean. Prediction 2 ("none") was wrong | `STEP_22_MIRROR_REPORT.md` |

R5–R8 make a result about training dynamics. In this setting, whether a model uses relational
structure is a discrete outcome, settled early.
- For seed 9 it is settled by the initial weights: data order changes when the escape
  happens, never whether.
- For seed 5 the weights make non-use very likely but not certain. One ordering in eight
  rescues it from step 0, and none from step 400.
- In both seeds the outcome is not visible beforehand in standard training signals.

Step 22 gives the first pair of runs that share a starting state and differ in outcome.

### Product (real CT)

| # | milestone | numbers | evidence |
| --- | --- | --- | --- |
| P1 | A controllable 3D heart and lungs built from real CT, end to end. | 17 closed surfaces for subject s0029; 11 of 11 left/right checks pass | `evidence/2026-10-15_real_ct_s0029_heart_and_lungs.jpg` |
| P2 | Plain-language control of that scene. | "hide the lungs", level-of-detail changes, isolation, opacity, cardiac-cycle playback, each through validated engine commands | `evidence/2026-10-15_real_ct_s0029_lungs_hidden.jpg`, `evidence/2026-10-16_real_ct_s0029_lod0_organ_shells.png` |
| P3 | Data is efficient to acquire. | 20 subjects for 32.6 MB of a 28.4 GB archive (HTTP byte ranges); 200 subjects later fetched the same way | `assets/fetch.py`, notebook 2026-10-15 and 2026-10-16 |
| P4 | Automated quality gate. | 15 of 20 subjects accepted; each of the 5 rejections has a stated reason (field-of-view cut, missing structure) | `datasets/processed/level1/real/quality_report.json` |
| P5 | An open-vocabulary language layer with a safety boundary. | Rules first; otherwise Claude emits a program restricted by a closed schema of operations and entity ids, validated before anything runs. Off by default; tested without network calls | ADR-L1-005, `language/` |
| P6 | **Generated anatomy from learned shape models** (Level 2a): statistical models of the four chambers, the myocardium and both lungs, learned from 131 real CTs, draw new thoraxes. | Held-out reconstruction error: 1.1–2.1 mm for the chambers, 5.1–6.3 mm for the lungs, 3.2–4.6× better than the mean shape on 26 unseen subjects. **Validity:** 50 of 50 samples at 0.5 SD pass checks calibrated on real anatomy; at 1 SD, 78% pass, against 90.5% of real held-out subjects. Shown samples are drawn at 1 SD and kept only if valid. They are as novel as a new real person: 8.8 mm against 9.1 mm to the nearest training subject | `LEVEL_2A_REPORT.md` §5–6, `evidence/2026-10-16_generated_thorax_seed1.jpg` |
| P7 | **One structure resized by a sentence**, within its own shape model, with an honest refusal. | On a real heart, "make the left ventricle 15% smaller" is applied: −15.0% size, −38.7% volume, 0.6 mm from plain scaling, every other mesh unchanged. Growth beyond 2–6% is refused because neighbours would be penetrated, and the reply names the largest change that passes | `LEVEL_2A_REPORT.md` §7, ADR-L2-001, `evidence/2026-10-16_resize_s0029_*` |
| P8 | **A neural model of the whole thorax** (Level 2b): one network labels every point of a thorax, with one code for the heart and one for the lungs. Two structures can never overlap, by construction. Trained on rented GPUs for about $2. | **Fit to 26 unseen people**, with Level 2a scored by the same rule (median surface distance): LV 0.57 mm against 2a's 0.82; myocardium 0.62 against 0.90; RV 0.80 against 1.20; LA 0.89 against 1.25; RA 1.10 against 1.43; lungs 2.56 and 2.86 against 3.58 and 4.05. **Better than Level 2a on every structure**, and on 21–26 of the 26 people per structure. The yardstick's own floor is 0.35–0.47 mm. **Generation:** 88% of 1 SD draws pass checks calibrated on real anatomy, unfiltered (Level 2a: 78%). No point is claimed twice in any sample. Samples are 94% as novel as a new real person. The viewer shows only valid draws. **Resize:** heart edits leave the lungs untouched. Under real-anatomy limits the LV grows by +2%, as in Level 2a, because the stretched myocardium breaks first; work continues. | `LAB_NOTEBOOK.md` 2026-10-18/19, `LEVEL_2B_SPEND.md`, `evidence/2026-10-18_level2b_*` |

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

* **Compute.**
  - Research and Levels 1–2a ran on one laptop.
  - Level 2b trains on rented NVIDIA L4 GPUs (Google Cloud, spot). Spend so far is about
    $1.65 of a $700 budget, every session logged in
    [`LEVEL_2B_SPEND.md`](LEVEL_2B_SPEND.md).
  - A GPU step runs about 21× faster than on the laptop.
* **Data.** Public, openly licensed CT segmentations. No patient data was collected.
* **Paid services.** The Claude language path (`claude-opus-5-5`) runs with the user's own
  API key, kept in a private file and never printed. Its first live test gave 11 of 11
  requests the intended result, including Hinglish ("dil ke left ventricle ko 10% chhota
  karo") and refusals for absent anatomy (notebook, 2026-10-18).

## Limitations (stated plainly)

* The research results are on synthetic corpora. Whether they transfer to real anatomy is
  untested.
* The real-CT scenes are segmentation surfaces from one dataset. Volumes and shapes are
  measurements, not clinical values, and have not been reviewed by a clinician.
* The cardiac-cycle animation is semantic, a sequence of named phases. It is not a
  physiological simulation.
* Generation keeps only samples that pass the validity checks: about 80% of Level 2a's 1 SD
  draws and 88% of Level 2b's. The shown distribution is therefore truncated.
* Growing one chamber is still limited to a few percent in both models. Level 2b's field can
  make the LV 15% larger, but the stretched myocardium breaks before it gets there, and the
  checks refuse that.
* Level 2a shape models are linear: they vary shape the way the training population did.
  They cover five heart structures and the two lungs, not the vessels or the lobes. Lung
  templates miss thin wedges and recesses (3–8% of the surface). The right-lung template
  carries a topological handle from its reference subject.

## Evidence index

| what | where |
| --- | --- |
| Roadmap to "any body part on request" | [`ROADMAP.md`](ROADMAP.md) |
| Dated record of all work | [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) |
| Screenshots with captions and reproduce commands | [`evidence/`](evidence/README.md) |
| Level 1 plan and status | [`LEVEL_1_PLAN.md`](LEVEL_1_PLAN.md) |
| Level 2 plan | [`LEVEL_2_PLAN.md`](LEVEL_2_PLAN.md) |
| Architecture decisions | [`adr/`](adr/README.md) |
| Research reports, Steps 6–20 | [`step-6-experiment-report.md`](step-6-experiment-report.md), [`step7/`](step7/), `STEP_8_*` to `STEP_20_*` in this folder |
| Step 21 preregistration and amendment | `docs/STEP_21_BASIN_ENTRY_PLAN.md` (PR #5) |
