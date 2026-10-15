# ADR-L1-003: Asset tooling is a third dependency tier

**Status:** ACCEPTED
**Date:** Level 1, 2026-10-15
**Relates to:** `pyproject.toml`, [ADR 0010](0010-dependency-tiers.md)

## Context

ADR 0010 keeps the AWR core on the standard library plus PyYAML, and confines PyTorch and
NumPy to a prototype tier. Turning segmentations into meshes needs NIfTI reading, marching
cubes, mesh repair and decimation, and GLB export. Those are not research-model dependencies,
and the core must not import them.

## Decision

Add an `assets` optional extra for the asset pipeline: `nibabel`, `scikit-image` and `trimesh`,
version-pinned. They may be imported only by the `assets/` package and its tests. The label map
itself is plain JSON read with the standard library, so the core and the viewer server can use
it without the extra.

## Consequences

* `pip install -e ".[assets]"` is needed to build meshes, but not to run the viewer on meshes
  that already exist.
* The test suite skips asset-pipeline tests when the extra is absent, as it does for prototype
  tests.
