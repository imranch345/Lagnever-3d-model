# ADR 0013: A tier-0 synthetic corpus as the first testbed

**Status:** ACCEPTED
**Date:** Step 6
**Relates to:** `datasets/synthetic/`

## Context

The first experiment asks whether a structured representation gives better semantic
control and relationship consistency than an appearance-driven one. Answering it needs
data with exact per-entity labels, exact frames and exact relationships. Real
segmented cardiac data has none of those exactly, needs a licence review, and would
confound the representation question with a data-quality question.

## Decision

Generate the corpus. Each renderable entity of Heart Ontology v0.1 gets one analytic
primitive whose parameters are randomised within ordered ranges. Anchors are derived
from the chamber radii so the ontology's spatial relationships hold by construction,
and a scene that still violates a required relationship is rejected and resampled.

Scenes are stored as parameters, not geometry: 2,000 scenes are a few megabytes and
every label is recomputed analytically, so a run is reproducible down to the query
points.

Everything is stamped `SYNTHETIC_RESEARCH_DATA / NOT_MEDICALLY_VALIDATED`.

## Alternatives considered

**Licensed real cardiac data.** The eventual requirement. Deferred: the licence review
is not done, the labels use clinical rather than ontological conventions, and starting
there would mean debugging the representation and the data at the same time.

**Random blobs.** Trivial to generate. Rejected: with no relational structure there is
nothing for a relational representation to exploit and nothing for the relationship
metric to measure, so the experiment would be uninformative either way.

**A single canonical heart with noise.** Rejected: a model can memorise one shape, and
the generalisation split would mean nothing.

## Consequences

* Exact ground truth for identity, ownership, frames, occupancy and relationships.
* No licensing exposure, and generation takes about five seconds for 2,000 scenes.
* **A structured generator may favour a structured model.** This is the sharpest
  threat to the experiment's validity. Ablation A4 partly addresses it, and a tier-1
  replication on real data is required before any claim leaves the synthetic setting.
* An ellipsoid is not a ventricle. No result measured here is evidence about anatomy.
