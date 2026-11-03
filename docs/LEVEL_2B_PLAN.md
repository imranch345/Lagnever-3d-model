# Level 2b — A neural thorax in which structures cannot run into each other

**Status: PLAN.** Written 2026-10-18, after the Level 2a report, before any Level 2b code.
**Budget: USD 700 of cloud GPU**, approved by the user on 2026-10-18. It is spent only in the
milestones below, each with its own cap.
**Relates to:** [`LEVEL_2_PLAN.md`](LEVEL_2_PLAN.md) §4, [`LEVEL_2A_REPORT.md`](LEVEL_2A_REPORT.md),
ADR-L2-001, ADR-L2-002.

Not medically validated, and not for clinical use. Generated anatomy is not a measurement of
anyone.

## 1. Why Level 2b

Level 2a works, and its report names three limits. All three come from one cause: a linear
shape model per structure does not know where its neighbours are.

| limit (Level 2a report) | measured |
| --- | --- |
| Growth collides with neighbours | On s0029, LV +15% is refused; the most that passes is +2% (RV +2%, atria +5%, left lung +6%) |
| Variety trades against plausibility | 1 SD Gaussian samples pass 78% of the validity checks; real held-out subjects pass 90.5%. Shown samples are filtered |
| Seven structures only | No vessels, lobes or trachea in generated scenes |

The user's goal is a model that designs plausible hearts and lungs and lets one structure be
changed by a sentence. For that, a structure that grows must push its neighbours aside, as it
does in a body.

## Status (2026-10-19)

| milestone | state | key numbers |
| --- | --- | --- |
| **M1** labelled points, model, training, evaluation | done | the pipeline is tested; the first prototype was 1.5–2× worse than Level 2a |
| **M2** long GPU training | **done; fit target met** | run 6 (parts with compact codes) beats Level 2a on all seven structures: heart medians 0.57–1.10 mm against 0.82–1.43; lungs 2.56 and 2.86 against 3.58 and 4.05. About $2.15 spent |
| **M3** generation, resize, viewer | in progress | generation 88% unfiltered at 1 SD under real-anatomy limits (target 90%; Level 2a 78%), novelty 94% of a real person's, 0 overlap; the viewer draws valid Level 2b thoraxes. Resize: lungs untouched by heart edits, but LV growth is +2% (target +15%): thin walls break first |
| **M4** more structures, higher resolution | not started | resolution is the lever for resize |

## 2. Approach: one field that labels every point

A neural network f(x, z) takes a 3D point x and a subject code z. It returns, for that point,
a probability for each of K structures and for "nothing".

* **No overlap, by construction.** Each point belongs to exactly one structure, so two
  structures can never occupy the same place. Level 2a's overlap check becomes unnecessary
  rather than merely passed.
* **Neighbours adapt together.** One code z describes the whole thorax. Changing it to grow
  the left ventricle moves the myocardium, atria and lungs in the way the training data says
  they move together, and never through each other.
* **Learned straight from the label masks.** It learns from TotalSegmentator's label volumes,
  the same data Level 1 meshes. No template correspondence is needed, so the lungs' thin
  wedges, which templates could not reach, are learned too.
* **Same interfaces.** Decoded scenes are meshed per structure by marching cubes and written
  as Level 1 scenes, with the same manifest and the "generated" banner. The viewer, AWR and
  language layer work unchanged.

**Model.** An auto-decoder (DeepSDF-style, but with a label per point rather than a distance):
- one learned code per training subject;
- an MLP with Fourier features of the point, conditioned on the code, outputting K + 1 logits.

**Training data.** Points are sampled per subject: dense near the boundaries between labels,
sparse elsewhere. Each point carries its label. The loss is cross-entropy plus a small prior
on the codes.

**Generation.** The training codes are fitted with a Gaussian, full covariance first and a
small normalising flow if needed. A thorax is decoded on a grid.

**Resize.**
- The structure's soft volume is computed from the decoded field; it is differentiable.
- z is optimised to reach the target volume, with a prior keeping it near the start.
- Neighbours change only as much as the field couples them.
- The result goes through the same validity checks as Level 2a: volumes, one piece per
  structure, and sides.

**Structures.** Start with Level 2a's seven, so the comparison is direct. Then add the aorta,
pulmonary artery, IVC, trachea and the five lobes (milestone M4).

**Frame.** Level 2a's rigid alignment by the heart, onto s0029.

## 3. How we will know it is better (measured as Level 2a was)

The same 131 usable subjects and the same held-out 26 subjects (split seed 0) are used, so
every number compares directly with Level 2a.

| question | measure | Level 2a | Level 2b target |
| --- | --- | --- | --- |
| Does it fit unseen people? | held-out per-structure surface distance (mm) and Dice, codes fitted to the held-out masks | chambers 1.1–2.1 mm, lungs 5.1–6.3 mm | chambers ≤ 2a; lungs better than 2a |
| Are samples plausible **and** new? | validity pass rate at real-person novelty (≈ 9 mm from the nearest training subject) | 78% unfiltered | ≥ 90%, unfiltered |
| Can one structure grow? | the largest LV growth on s0029 that passes every check | +2% | **≥ +15%**, neighbours adapting |
| Is it honest? | overlap between structures | checked | **0 by construction**, verified on every sample |

If 2b does not beat 2a on fit and plausibility, the report says so and 2a remains the
product's generator.

## 4. Milestones and budget

| | milestone | where | GPU cap |
| --- | --- | --- | --- |
| **M1** | Data pipeline (label volumes to labelled point samples). Model, training loop, evaluation. A small prototype on a few subjects. | this Mac, Apple M2 GPU (MPS) | **$0** |
| **M2** | First full training on the 105 training subjects; evaluation against 2a | cloud GPU | **≤ $50** |
| **M3** | Resize by code optimisation; generation; viewer and language integration | Mac + short GPU runs | **≤ $100** |
| **M4** | More structures (vessels, trachea, lobes), higher resolution; tuning | cloud GPU | **≤ $300** |
| — | Reserve, released only by the user | — | $250 |

**Costs, as ranges, not quotes.**
- A rented RTX 4090 costs about $0.35–0.70 an hour; an A100 80 GB about $1.2–2.0.
- One full training run is expected to take 3–8 GPU hours, about $2–15 on a 4090.
- So M2 is a few runs, and the whole plan is well inside $700.

**Rules for spending.**
- **A ledger.** `docs/LEVEL_2B_SPEND.md` records every session: date, provider, GPU, hours,
  cost and purpose. It is updated the same day.
- **Caps are hard.** At a milestone's cap work stops, and the next milestone needs the user's
  go-ahead.
- **No idle machines.** Every instance is started with an automatic shutdown, and is stopped
  when its job ends.
- **Accounts, payment and keys are the user's.** The user creates the GPU account and adds
  payment. Claude never handles passwords, API keys or card details, only access the user
  sets up (for example SSH with the user's own key).

## 5. Data

- The 200 subjects already fetched (278 MB) and the 131 the quality gate accepts are enough
  for M1–M3.
- More subjects would help a neural model. Fetching more from the TotalSegmentator archive is
  a separate download decision, with its size stated first.

## 6. Risks

* **131 subjects is small for a neural model.** The codes may memorise subjects. Novelty
  against the nearest training subject is therefore measured on every sample set, as in 2a.
* **Thin structures need resolution.** Vessels and lung edges cost points and grid size, and
  M4 is where that cost goes.
* **MPS differs from CUDA in numerics.** The Mac is for development. Reported numbers come
  from the cloud runs, with their seeds and versions recorded.
* **A label field can create small detached pieces.** These are checked as in Level 1 (one
  piece per structure), and are counted, never hidden.

## 7. What the user is asked for

1. **A GPU provider account**, before M2. RunPod, Lambda or Vast.ai are all fine; RunPod is
   the simplest to start with. Access is given as SSH with the user's own key.
2. **The Claude API key, kept out of chat.** The user puts it in
   `~/.lagnav/anthropic.env` (`ANTHROPIC_API_KEY=...`, readable only by the user). The viewer
   reads it from there, and it never enters the repository.
3. **PyTorch on this Mac's product environment** (about 80 MB from PyPI, the version the
   research environment already uses), for M1.
