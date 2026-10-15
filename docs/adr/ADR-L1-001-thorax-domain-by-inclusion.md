# ADR-L1-001: The thorax domain includes Heart Ontology v0.1 unchanged

**Status:** ACCEPTED
**Date:** Level 1, 2026-10-15
**Relates to:** `awr/ontology.py`, `ontology/thorax/`, `configs/thorax.yaml`,
`tests/test_thorax_domain.py`

## Context

Level 1 puts the heart and the lungs in one scene. Until now an ontology has been exactly one
directory, and a domain has loaded exactly one ontology. The heart ontology is also a
research dependency. Steps 6–21 load `ontology/heart` at version 0.1.0, and
`generation/neural/features.py` sizes a neural input from the `AnatomyType` enum. Any change
to either would alter the research models and break bit-exact reproduction of frozen results.

## Decision

1. **Composition by inclusion.** `anatomy.json` may declare `includes`, each pinned by
   `ontology_id`, `version` and a path relative to the including directory. The loader
   validates each include on its own first, then merges its entities, tree nodes and edges
   before the includer's. The ordinary validators then run once over the result. Duplicate
   ids across the boundary, version or id mismatches, and include cycles all raise.
2. **A `thorax` domain.** `ontology/thorax/` adds a `thorax` root, ten respiratory entities and
   fourteen edges, and includes `lagnav.heart@0.1.0`. `configs/thorax.yaml` mirrors
   `configs/heart.yaml` except for the domain block and LOD descriptions.
3. **`AnatomyType` is frozen while research steps run.** Lungs, lobes and the trachea use
   `organ`. `semantic_role` carries `lung`, `lung_lobe` or `airway`. A test pins the enum at
   14 members.
4. **No heart–lung functional edges in v0.1.** The flow animation derives its paths from the
   functional graph. Pulmonary circulation is added later, together with tests on that
   animation.

## Alternatives considered

**Copy the heart entities into a thorax file.** Rejected: two copies of the heart vocabulary
would drift, and a fix to one would silently not reach the other.

**Add lungs to the heart ontology.** Rejected: it changes the research steps' input, the
ontology's id would stop meaning "heart", and v0.1.0 is pinned by every frozen run.

**Load several ontologies side by side in the scene.** Rejected for now. Cross-organ edges,
such as the trachea lying superior to the heart, need one graph to live in, and the engine
resolves names against one ontology.

## Consequences

* The existing `CommandEngine` drives the lungs with no code change: show, hide, isolate,
  opacity, LOD and explain all resolve through the composite.
* Three engine behaviours written for a single organ were fixed in Level 1's engine work
  (`CommandEngine._drawable_targets` and the generate vocabulary):
  - an organ with an LOD-0 shell is drawn as its shell or as its parts, never both;
  - "hide the heart" above LOD 0 hides its parts;
  - "generate the heart and lungs" (or "the thorax", "the chest") builds the thorax scene, and
    is refused with a reason in a domain that has no lungs.
* Extending `AnatomyType` waits until the research track closes, or until the feature
  extractor stops depending on the enum's size.
