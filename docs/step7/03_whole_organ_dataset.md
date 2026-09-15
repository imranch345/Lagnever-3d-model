# Experiment 4 — The whole-organ dataset

**Status: complete.** The corpus exists, is crossed, and every entity owns volume in
every scene.

Synthetic research data throughout. Not anatomy, not validated, not clinical.

## Why the generation order was reversed

Step 6 generated each entity independently and assembled the scene from them. That order
builds the answer into the question: a corpus made of separately-placed parts will
naturally favour a representation that factorises into separately-placed parts, so
finding that an entity-factorised model does well says little.

Step 7 reverses it:

1. build the whole-organ field from global parameters;
2. carve cavities out of it in the organ's own frame;
3. derive valves as the annulus regions between adjacent cavities;
4. derive septa as the myocardium lying between two cavities;
5. attach vessels at openings computed from the cavities;
6. assign every point an owner by a fixed priority rule;
7. **measure** the relationships from the finished organ.

No entity is placed independently, and no relationship is asserted. The septum exists
because there is tissue between two cavities, not because the ontology says a heart has
one.

## What varies, and what deliberately does not

| Varies per scene | Constant across scenes |
| --- | --- |
| global shape parameters, 20 of them, per family | the entity set: the same 20 entities |
| arrangement: one of four anatomical variants | presence and text features |
| level of detail: 1, 2 or 3 | the structural graph, which is ontology knowledge |
| measured spatial and functional relations | |

The four arrangements are the instrument. Presence is identical across them and text
features are identical, enforced by construction, so **the only channel carrying the
arrangement is the relationship graph.** A model that cannot read the graph has no way
to tell a mirrored heart from a normal one and must produce the average of the two.

| Variant | What changes | What the relations say |
| --- | --- | --- |
| `normal` | reference | left chambers on the anatomical left |
| `mirrored` | reflected across the midline | `left_of` and `right_of` invert |
| `transposed` | great arteries swap ventricles | the aorta leaves the right ventricle |
| `rotated` | rotated about the vertical axis | `anterior_to` and `posterior_to` change |

## The corpus

`datasets/processed/whole_organ_1600_v2`

| Property | Value |
| --- | --- |
| scenes | 1,600 |
| families | 40 |
| entities per scene | 20 |
| distinct relationship graphs | 164 |
| splits | 1,280 train / 160 validation / 160 test, disjoint by family |
| arrangements | 400 of each, in every split |
| arrangements per family | 4 of 4, ten scenes each |

Compare Step 6: 2,000 scenes carrying **one** relationship graph between them. The
corpus is smaller and carries 164 times more relational variety.

## The defect that had to be fixed first

The first generator derived the variant from the scene index modulo the variant count,
and the family from the index modulo the family count. Four divides forty, so the variant
was a function of the family: every family appeared in exactly one arrangement, and the
validation and test splits held disjoint pairs of arrangements. The contrast the corpus
was built to present was never presented. See [ADR 0015](../adr/0015-crossed-variant-corpus.md).

The generator now refuses to write a corpus with that property, and records the crossing
in its manifest.

## Ownership, and Experiment 10

Step 6's wall layers owned so few points that per-entity metrics on them were undefined.
The pre-registered ownership rule assigns each point to the highest-priority class
containing it: background, pericardium, chamber, valve, vessel, septum, endocardium,
epicardium, myocardium.

Measured over the 160 test scenes:

| Entity group | Share of sampled points |
| --- | --- |
| chambers | 38.3% |
| vessels | 31.2% |
| wall layers | 18.3% |
| valves | 10.7% |
| septa | 1.6% |

**Every one of the 20 entities owns volume in all 160 test scenes.** The Step 6 metric
problem is resolved: no per-entity metric is undefined for want of a region to score.

The interventricular septum is the thinnest structure at 0.28% of points. It is present
everywhere but small, so its per-entity numbers carry more sampling noise than the rest,
and that should be remembered when reading them.

## What this corpus still cannot support

* It is procedurally generated and bears no validated relationship to human anatomy.
* Four arrangements is a small set. A model could in principle learn to classify the
  arrangement from the graph without learning anything general about relations.
* Families are procedural, so held-out families test interpolation within one generator,
  not generalisation to organs built a different way.
