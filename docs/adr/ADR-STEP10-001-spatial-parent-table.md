# ADR-STEP10-001: Derive the spatial hierarchy from construction, not from AWR

**Status:** ACCEPTED
**Date:** Step 10
**Relates to:** `datasets/whole_organ/hierarchy.py`, `generation/neural/nn/placement.py`,
`tests/test_step10_hierarchy.py`

## Context

Step 10 Change 1 asks for placement predicted relative to a parent, and gives
`left_ventricle -> mitral_valve` as the example of the tree to compose over.

That tree is not in AWR. Checked against Heart Ontology v0.1:

| Question | Answer |
| --- | --- |
| Whole-organ entities with a whole-organ-entity parent | 0 of 20 |
| `parent_of("heart.mitral_valve")` | `"heart.valves"` |
| Does `heart.valves` carry geometry? | No — not in `WHOLE_ORGAN_ENTITIES` |
| `contains` / `inside` relations among the 20 | none |

AWR's hierarchy is **taxonomic**: every whole-organ entity is a depth-2 leaf under a
category node (`chambers`, `valves`, `great_vessels`, `septa`, `wall_layers`), and those
category nodes have no frame. Composing placement along it would compose each entity
against a parent that does not exist in the scene.

The hierarchy the brief describes does exist, but in the generator's construction order.
`field._build_annuli` derives each atrioventricular annulus from a chamber pair;
`field._build_tubes` starts each great artery at its own valve's annulus, starts each great
vein at its atrium's surface, and branches the pulmonary arteries off the pulmonary trunk.
Those are real geometric dependencies: move the parent and the child moves with it.

## Problem

Supply a hierarchy that composition can actually use, without asserting that AWR has a
spatial hierarchy it does not have, and without making the experiment unable to tell the
two apart.

## Alternatives considered

**Use the taxonomic hierarchy as-is.** Rejected as the treatment, kept as the control. With
no geometry-bearing parents, every entity is a root and parent-relative placement is
numerically identical to the Step 9 global target. That is worth measuring — it is what
"just use AWR's hierarchy" would deliver — but it is not a hierarchy experiment.

**Add `contains` relations to the ontology so AWR has the tree.** Rejected. It would change
the ontology to make a Step 10 result come out, and every later step would inherit an
anatomical claim introduced for an experiment's convenience. The Step 10 brief's §26 lists
the hierarchical entity structure among the things not to change.

**Derive the parent per scene from the arrangement.** Rejected as the default. The
construction parent of the aortic and pulmonary valves follows `transpose`, which is one of
the arrangement axes the held-out splits generalise over. A per-scene table hands the model
the arrangement before it has predicted anything — the same class of defect as the Step 8
ground-truth placement leak. Retained as `spatial_oracle`, labelled as leaking, for
diagnosis only.

## Decision

Three tables, in `datasets/whole_organ/hierarchy.py`:

* `spatial` — the construction dependency, fixed in the non-transposed convention. The
  default. Wrong for transposed scenes in a way the model cannot exploit.
* `taxonomic` — AWR's, which leaves all 20 entities roots. The control arm.
* `spatial_alternative` — the atrioventricular annuli parented to the atrium rather than
  the ventricle, so the one genuinely arbitrary choice is testable rather than assumed.

Parent slots are resolved against the **batch builder's** slot map, which is keyed on
`ontology.ids()` (42 entities, padded to 64), not on `WHOLE_ORGAN_ENTITIES` (20). The two
orderings differ, and the first version of this table used the second: it addressed entirely
different entities and reported a plausible-looking 78% orphan rate rather than raising.
`parent_slots` now requires the slot map and raises `KeyError` on an entity it cannot place.

## Evidence

Measured on `test_seen` through the evaluation loader:

| Property | Value |
| --- | --- |
| Entities with a parent | 10 of 20 |
| Maximum depth | 3 |
| Depth histogram (1 / 2 / 3) | 7 / 2 / 1 |
| Children orphaned by level of detail | 8.1% before reparenting, 0% after |

Half the entity set is untouched by Change 1, which bounds what the change can deliver and
is why `experiments/step10/depth_analysis.py` reports error by depth rather than only in
aggregate.

## Consequences

* The hierarchy is a property of the **corpus generator**, not of the ontology. A different
  generator needs a different table, and `hierarchy.py` says so.
* A control arm exists that is provably identical to the Step 9 target, so any Change 1
  effect is attributable to the hierarchy rather than to the rest of Step 10. The identity
  is asserted in `run_placement._control_identity`, not assumed.
* Transposed scenes are composed against the wrong arterial parent by design. That is a
  known, recorded inaccuracy, preferred to a leak.

## Reversal conditions

Reverse this decision if any of these is shown:

* AWR gains geometry-bearing containment among the whole-organ entities, in which case the
  ontology becomes the source of truth and this table should be deleted rather than kept
  alongside it — `test_awr_has_no_spatial_parent_among_the_whole_organ_entities` fails
  loudly when that happens;
* the `spatial_alternative` convention beats `spatial` by more than seed spread, which would
  mean the ventricle-as-parent choice is doing work and deserves to be derived rather than
  chosen;
* the corpus changes so that fewer than a handful of entities have parents, at which point
  the hierarchy is too thin to carry an experiment.
