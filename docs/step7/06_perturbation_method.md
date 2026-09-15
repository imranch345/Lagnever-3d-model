# Experiments 1, 2 and 3 — Representation stress test. Method

The whole-organ benchmark asks how well a model reproduces an organ. These experiments
ask a narrower and more useful question: **which parts of the Anatomical World
Representation is the model actually reading?**

## Method

A model trained on intact input is evaluated with one channel of the representation
damaged at a time. Entities, presence, text features and sampled points are identical
across every perturbation, so a change in the output can only have arrived through the
relationship graph.

**Reading rule, fixed before running:** a channel the model depends on must degrade the
output when corrupted. A channel that can be deleted with no measurable effect is not
being used, whatever role the architecture assigns it, and its component is marked
SIMPLIFY or REMOVE.

This is a stronger test than comparing arms. An arm comparison can be confounded by
capacity or optimisation; deleting an input from a fixed model cannot be. If the output
does not move, the input was not being read.

## Experiment 1 and 2 — relationship and spatial perturbation

| Perturbation | What it does |
| --- | --- |
| `intact` | control |
| `drop_spatial` | every spatial edge removed |
| `invert_spatial` | every spatial relation replaced by its inverse, endpoints swapped with it |
| `shuffle_spatial_types` | spatial relation types permuted among the same edges |
| `randomise_spatial_endpoints` | spatial edges rewired to random entity pairs |
| `drop_functional` | every functional edge removed |
| `drop_structure` | every structural edge removed |

`invert_spatial` swaps the endpoints along with the relation, because the inverse of a
relation is the same edge read backwards. Leaving the endpoints in place would assert
something the ontology forbids and would test a different corruption than the one named.

The four spatial perturbations form a ladder. `drop_spatial` removes the information;
`shuffle_spatial_types` keeps the graph's shape but corrupts its labels;
`randomise_spatial_endpoints` keeps the labels but corrupts the shape; `invert_spatial`
keeps both and makes them systematically wrong. A model reading the graph should respond
differently to these. A model ignoring it responds to none.

## Experiment 3 — partial representation

| Case | Graphs kept |
| --- | --- |
| `A_full` | structure, spatial, functional |
| `B_no_spatial` | structure, functional |
| `C_no_functional` | structure, spatial |
| `D_no_structure` | spatial, functional |
| `E_entities_only` | none |

`E_entities_only` is the floor: the entity axis with no relationships at all. The gap
between `A_full` and `E_entities_only` is the total value of the relationship graph to
this model. If that gap is small, the AWR's relational half is not being used and the
finding is reported as such.

## Conditions

Every case is scored under both placement conditions. In the **given** condition
ground-truth entity frames are supplied, which hands the model the arrangement a graph
would otherwise have to supply; the perturbations are then expected to do little, and
that expectation is itself a check that the harness is measuring what it claims. The
**inferred** condition, where the model predicts placement, is the one the decision rests
on.

## Limitation to state with the results

These experiments evaluate a model trained on intact input. A model can be genuinely
robust to a corrupted input without ignoring it, and a model trained with corruption
might use the channel differently. What the experiment shows is whether the trained
model, as trained, depends on the channel. That is the right question for a KEEP or
REMOVE decision about the current design, and it is not the same as showing the channel
could never be useful.
