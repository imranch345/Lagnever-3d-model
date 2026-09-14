# ADR 0008: Six-stage curriculum whose first two stages need no external data

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/curriculum.py`

## Context

The Step 5 brief proposed nine stages: ontology understanding, text alignment,
image alignment, 3D alignment, anatomy-geometry correspondence, text to
representation, representation to 3D, editing, animation.

The largest risk to the schedule is not modelling, it is data. Real segmented
cardiac data needs licence review, and image data needs more.

## Decision

Six stages:

| stage | content | external data |
| --- | --- | --- |
| S0 | Ontology self-supervised pretraining | none |
| S1 | Language to AWR program | none |
| S2 | Geometry autoencoding with part supervision | yes |
| S3 | AWR to geometry generation | yes |
| S4 | Persistent editing | yes |
| S5 | Optional modalities: images, multi-view, animation | yes |

Three changes from the brief:

1. Text-to-anatomy merges with text-anatomy alignment: same data, same model, and
   splitting them creates a checkpoint nobody uses.
2. Anatomy-geometry correspondence merges into geometry autoencoding: training the
   part head separately invites a representation that reconstructs well and labels
   badly, which is the failure the project is defined against.
3. Images move to the end and become optional: they are the hardest modality to
   licence and are not needed for text-only inference.

## Alternatives considered

**Keep all nine stages.** More granular checkpoints. Rejected: two of the splits
produce artefacts nobody consumes, and more stages means more interfaces to keep
correct.

**Train end to end from the start.** Fewer moving parts, and it sometimes wins.
Rejected for this project: with a small dataset and an unproven representation,
end-to-end training makes attribution impossible when something fails.

**Images before geometry.** Images are more plentiful than labelled 3D. Rejected:
it optimises the modality the product does not need at inference, before the one
it does.

## Consequences

* Work can start immediately: S0 and S1 need no dataset and carry no licensing
  exposure.
* Each stage produces an independently evaluable checkpoint.
* Freeze schedules are explicit per stage, so a regression can be attributed.
* Animation is last and may not be reached in the first cycle.

## How this could be wrong

Staged training can lock in a poor early representation that end-to-end training
would have escaped. If S3 plateaus well below S2's reconstruction quality, that is
the signal to unfreeze and fine-tune jointly.
