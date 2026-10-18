# Lagnav 3D roadmap: from a heart and lungs to "any part of the body, on request"

**The goal.** A person names any part of the body, in their own words and in any language
Lagnav supports. Lagnav 3D then generates it as realistic, controllable 3D anatomy: shown in
place, resizable, explainable. Every shape is new, drawn from what real anatomy looks like,
and checked for plausibility before it is shown. It is educational and research anatomy,
not a medical device.

This roadmap is written for the team and for investors. Its estimates are honest ones:
- **Durations** are working time, assuming steady sessions.
- **Costs** are cloud GPU spend, logged session by session in
  [`LEVEL_2B_SPEND.md`](LEVEL_2B_SPEND.md).
- **Every stage ends in measured results,** recorded in [`RESULTS.md`](RESULTS.md) and
  [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md), whether they meet the target or not.

## Where we are (2026-10-19)

| capability | state |
| --- | --- |
| Real anatomy from CT, shown in 3D (Level 1) | 17 heart and lung structures from public CT (TotalSegmentator v1.0, CC BY 4.0), with quality gates and provenance |
| Language control | Rules first, then Claude, restricted to a closed set of validated operations. Hinglish works ("dil ka LV 10% chhota karo"), and so do descriptions and refusals for absent anatomy |
| Generated anatomy, statistical (Level 2a) | Shape models of 7 structures; 78% of draws plausible unfiltered |
| **Generated anatomy, neural (Level 2b)** | **One network for the heart and lungs. On 26 unseen people it fits better than Level 2a on all 7 structures** (heart 25–30% closer). **88% of draws are plausible unfiltered**, and no point ever belongs to two structures. Trained for about $2 of GPU time |
| Known gap | Growing one chamber far (+15%) breaks thin walls first; the limit today is +2% |

## The data already supports the goal

Lagnav learns from the TotalSegmentator dataset: 1204 CT scans with **104 labelled
structures**. They include:
- organs: liver, kidneys, spleen, stomach, pancreas, bowel, bladder and others;
- bones: spine, ribs, pelvis, shoulders, femurs;
- major muscles and major vessels;
- the brain.

So far we use 14 of the 104 labels, for the heart and lungs, fetched selectively: 32.6 MB
for 20 subjects out of a 28.4 GB archive. The other labels come from the same archive, the
same way. Adding a body part to the model is a data and training step, not a new
invention: the architecture gives each part of the body its own code, and the language
layer already maps names to parts.

## Stages

| stage | what a user gets | structures | effort | GPU cost | status |
| --- | --- | --- | --- | --- | --- |
| **1. Heart and lungs** | generated heart and lungs, better than the statistical model; language control | 7 | done | ~$2 | **done** (M1–M2; most of M3) |
| **2. Complete chest** | great vessels (aorta, pulmonary artery, IVC), trachea, the five lung lobes; chamber resizing that holds up | ~14 | 1–2 weeks | ~$10–30 | **next** |
| **3. Torso** | abdominal organs, ribs, spine, abdominal vessels | ~60 | 3–5 weeks | ~$50–150 | planned |
| **4. Whole body, organ level** ⭐ | **name any of the 104 structures and get it generated**, in place, resizable, explained | 104 | ~2–3 months from now | within the $700 budget | **target** |
| **5. Fine anatomy** | hands, eyes, ears, nerves, regions inside the brain | hundreds | 6+ months | new budget | needs new data |

### Stage 2: complete chest (next)

- **Data.** Already on disk: 200 subjects with all 14 chest labels. No download is needed.
- **Model.** The same part-field architecture with more parts: heart, great vessels, lungs
  by lobe, airway. Sampling and decoding will be finer near thin walls.
- **Targets:**
  - every new structure fits unseen people as well as Level 1's own meshes allow;
  - generation stays at 90% or more plausible, unfiltered;
  - one chamber grows by +15% with its walls intact.

### Stage 3: torso

- **Data.** About 45 more labels per subject from the same archive, roughly 2–4 GB for
  200 subjects. The exact size is stated, and approved, before any download.
- **New problem: partial scans.** Most CTs cover the chest or the abdomen, rarely both.
  The model must learn that "not in this scan" is not "not in this body".

### Stage 4: whole body at organ level (the target)

- **Data.** All 104 labels, and the subjects whose scans reach the limbs and the head.
- **Language.** The ontology grows from the thorax to the whole body. Claude already maps
  everyday words (and Hinglish) to anatomy, constrained to known parts.
- **Product.** "Show me the kidneys", "make the liver larger", "hide the ribs": each one is
  generated, checked and shown.

### Stage 5: fine anatomy

CT does not resolve nerves, small hand and face muscles, or the inner ear, and its scans
rarely include the hands or feet. These need other sources: MRI datasets and anatomical
atlases, each with its own licence review. Two routes are possible:
- **show:** a single reference atlas, fast but with no variation;
- **generate:** population data, slower, but every shape new.

## Risks, stated plainly

- **Thin structures** (walls, small vessels) need resolution. Resize already met this
  limit, and stage 2 addresses it first.
- **Partial scans** (stage 3 onward) are a modelling problem, not just more data.
- **Small datasets per structure.** Rare structures, or ones seldom inside the scanned
  region, will be learned from fewer people. Plausibility checks keep weak samples from
  being shown.
- **Estimates move with results.** The Level 2b resize result already moved one: it is
  reported as it came out.

## How progress is proven

Every stage keeps the rules the project already follows:
- **held-out testing:** people the model never saw, scored by one rule for every model;
- **plausibility checks** calibrated on real anatomy, not on the model itself;
- **logged spend,** every GPU session and its cost;
- **evidence:** screenshots of the running product, with commands to reproduce them.
