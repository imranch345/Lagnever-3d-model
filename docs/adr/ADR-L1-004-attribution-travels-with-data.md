# ADR-L1-004: Attribution and license travel with every asset

**Status:** ACCEPTED
**Date:** Level 1, 2026-10-15
**Relates to:** `assets/`, the asset manifest, the viewer's credits panel

## Context

Level 1 uses the TotalSegmentator dataset v1.0 (Zenodo 6802614), licensed CC BY 4.0, which
requires attribution. Meshes derived from it are adaptations, and they will be copied,
exported and shown far from the pipeline that made them.

## Decision

Every manifest entry records:

* the dataset, version and DOI;
* the license;
* the subject id and source label;
* that the mesh was derived (marching cubes, smoothing and decimation, with their
  parameters).

The viewer has a credits panel built from the manifest, and every exported file carries the
same fields in its metadata. Raw data and derived meshes live under the git-ignored
`datasets/` tree and are never committed.

## Consequences

* A mesh without a manifest entry is an error, not a warning.
* Switching to a dataset with a different license changes data, not code.
