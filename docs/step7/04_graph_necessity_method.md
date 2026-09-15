# Experiments 5 and 6 — Is the relationship graph necessary? Method

This document fixes the method. Results are in the completion report, produced from the
run JSON by `experiments/step7/report_tables.py`.

## The question

Step 5 proposed a partitioned relational graph transformer as the mechanism by which
language-level anatomical relationships reach the geometry. It costs 2.67M of the
prototype's 3.09M parameters, 86% of the model. Step 7 asks whether that expenditure
buys anything, and is prepared to answer no.

## Three prior answers, all withdrawn

The honest starting point is that this question has been answered three times and every
answer has been withdrawn on inspection.

| Attempt | Finding | Why it was withdrawn |
| --- | --- | --- |
| Step 6 | graph encoder not earning its parameters | the corpus carried **one** relationship graph across all 2,000 scenes, so the graph input was constant and could only contribute a constant prior |
| Step 7 suite 1 | `A1` matches `A3` | variant was aliased onto family, so every family appeared in one arrangement only and the intended contrast was never presented ([ADR 0015](../adr/0015-crossed-variant-corpus.md)) |
| Step 7 suite 2 | `A1` matches `A3` | the part-correspondence target named entities the level hides, costing ~1e9 per such point; every arm spent most of its gradient budget on an objective none could reduce ([ADR 0016](../adr/0016-level-aware-part-target.md)) |

None of these was a wrong measurement of the right thing. Each was a correct measurement
of a setup that could not distinguish the hypotheses. They are kept under
`experiments/runs/step7-whole-organ-v1-confounded/` and
`experiments/runs/step7-whole-organ-v2-broken-part-loss/` with their own notes.

The pattern is worth stating plainly: **a comparison between arms is only informative
once the input actually differs between the cases the arms are supposed to distinguish,
and once the training signal is one the arms can act on.** Three separate defects each
broke one of those conditions.

## The arms

| Arm | Description | Parameters | Role |
| --- | --- | --- | --- |
| `A3` | full three-graph partitioned encoder | 3,094,551 | the proposal |
| `A3L` | one layer, four heads | 1,094,415 | is depth needed? |
| `A2` | structure graph only | 3,094,551 | is the partonomy enough? |
| `A1` | no graph encoder at all | 427,655 | the entity axis alone |
| `A1M` | no graph, capacity restored by per-entity depth | 2,993,735 | **the decisive arm** |
| `A0` | appearance baseline, no entity axis | 3,103,894 | is any structure needed? |

`A1M` is the one that decides the question. It has `A3`'s capacity within 3.3% and
`A3`'s inputs, but no mechanism for moving information between entities. If `A1M`
matches `A3`, the graph transformer is buying nothing that per-entity depth does not,
and the parameters are better spent elsewhere.

`A1` is the small arm: 7.2 times smaller than `A3`. If `A1` also matches, the finding is
stronger still.

## Controls

* Three seeds for `A3`, `A1M` and `A0`; one seed for `A3L`, `A2` and `A1`, labelled as
  such in every table.
* Identical corpus, steps, optimiser, learning rate, batch size and evaluation protocol.
* Family-disjoint splits; no procedural family appears in two splits.
* Checkpoint selection fixed in advance: the final checkpoint.
* The variant label is never an input to any arm. It exists only in the relations.
* Any input available to one arm is available to all.

## How the result must be read

Three reading rules, all fixed before the runs.

**1. Placement condition.** Supplying ground-truth entity frames hands the model the
arrangement a relationship graph would otherwise have to supply. Under that condition
the comparison cannot discriminate, and the decision rests on the inferred condition
where the model predicts placement itself. See
[ADR 0018](../adr/0018-evaluate-both-placement-conditions.md).

**2. The relation-blind floor.** A model ignoring relations entirely scores 0.8878 on
`spatial_relation_accuracy`, because many relations survive every arrangement. Only the
0.1122 above that floor is evidence of relational competence. See
[ADR 0019](../adr/0019-relation-blind-floor.md).

**3. The per-arrangement profile.** A model that reads the graph should handle the four
arrangements differently from one that cannot. A flat profile is evidence the graph is
unused.

**4. How to read `A2`.** Structural edges are ontology knowledge and identical for every
scene. A constant input can act as a prior but cannot distinguish arrangements, so `A2`
matching a no-graph arm on arrangement-sensitive metrics is the *expected* outcome and is
not evidence against typed graphs in general. See
[ADR 0017](../adr/0017-structural-graph-is-ontology-knowledge.md).

## What counts as a negative result

Stated before the runs, and to be reported without softening:

* If `A1M` matches `A3` in the inferred condition **and** on the counterfactual tests,
  the graph transformer is marked SIMPLIFY or REMOVE.
* If `A1` also matches, the entity axis alone is the recommendation and the 2.67M
  parameters are reported as unjustified.
* If the counterfactual sensitivity stays near zero for every arm, the conclusion is
  that no arm uses relations, and the mechanism is reported as not working rather than
  as working weakly.

No component is retained on the grounds that it ought to help.
