# ADR-STEP8-001: Replace the typed graph transformer with untyped graph attention

**Status:** ACCEPTED, pending the Step 8 evidence recorded below
**Date:** Step 8
**Relates to:** `generation/neural/nn/graph_lite.py`, `tests/test_step8_graph.py`

## Context

Step 5 proposed a partitioned relational graph transformer: heads split by graph kind,
a relation-type embedding, and a typed attention bias. It cost 2,666,896 parameters, 86%
of the Step 7 model.

Step 7 measured what it actually read. On a trained checkpoint, held-out scenes:

| Perturbation | What it corrupts | Entity IoU change |
| --- | --- | --- |
| `invert_spatial_labels` | what the relations say | +0.0002 |
| `shuffle_spatial_types` | what the relations say | +0.0000 |
| `randomise_spatial_endpoints` | which entities are connected | −0.0365 |
| `drop_spatial` | which entities are connected | −0.0406 |

Telling the model every spatial relation is the opposite of the truth changed its output
by 0.0002. Removing the same edges cost 0.0406, two hundred times more. The typed
machinery, which is the whole difference between this and a plain graph network, was close
to inert.

## Problem

Keep a relational mechanism, because Step 7 also found it was the only thing that
responded to arrangement at all, while removing the part that is not being read and the
parameters it consumes.

## Alternatives considered

**Keep the typed encoder and train it longer.** Step 7's convergence check found the
graph's arrangement response nearly doubled from 900 to 2,700 steps, so the typed
machinery might be learned slowly rather than never. Rejected as the default because
relation type was unread at **both** budgets, so the growth was in connectivity use, not
type use. Recorded as a reversal condition below rather than dismissed.

**Delete the graph entirely.** Rejected. Step 7 found only the arms with per-scene spatial
and functional edges responded to arrangement at all, `A3` at 3.6% and `A3L` at 1.9%,
against exactly 0.0% for every arm without. Deleting it would remove the one mechanism
that showed any relational effect, on the grounds that the effect was small.

**Keep types but shrink the encoder.** Rejected. It changes two things at once, so a null
result would not say which mattered.

## Decision

`UntypedGraphEncoder`: masked multi-head attention over the **union** of the three
adjacencies, with no relation-type embedding, no typed bias and no head partitioning. Two
layers, four heads, 692,864 parameters, within 4% of the reduced typed encoder's graph
budget of 666,760.

The encoder never reads `edge_relation`. That is asserted as a property and checked: a
test requires the output to be **bit-identical** under relation-label corruption, and to
change under edge dropping and rewiring.

Endpoint identity is preserved, so "A connects to B" differs from "A connects to C", and
the encoder is permutation-equivariant, so identity comes from the latent and not from the
slot index. Both are tested.

## Evidence

The Step 7 perturbation sweep above motivated the change. The Step 8 evidence that judges
it is the `A3Lite` against `A1M` comparison under predicted placement, reported in the
completion report. `A1M` has 2,566,080 parameters of per-entity depth against `A3Lite`'s
692,864 of graph, 3.7 times more, so a win for `A3Lite` cannot be a capacity effect.

## Consequences

* The relational budget falls from 2,666,896 to 692,864 parameters, a 74% reduction.
* Relation types cannot influence the model at all, so any future claim that they matter
  requires a new encoder and a controlled experiment, not a reinterpretation.
* The union mask means the three graphs are no longer separable inside the encoder. The
  ablations still remove them one at a time from the **input**, which is where Step 7
  found the measurable effect.

## Reversal conditions

Reverse this decision if any of these is shown:

* an encoder that reads relation types beats `A3Lite` at a matched budget on
  arrangement-sensitive metrics, by more than seed spread;
* a corpus in which relation type is the only route to the answer shows `A3Lite` at the
  relation-blind floor while a typed encoder is above it;
* head partitioning by graph kind is shown to matter on a corpus where the three graphs
  carry conflicting information, which this one does not.

A small metric difference is not sufficient. Step 7 retained this machinery for two steps
on exactly that basis.
