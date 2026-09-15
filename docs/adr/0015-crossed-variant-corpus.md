# ADR 0015: Anatomical variant and procedural family must be crossed

**Status:** ACCEPTED
**Date:** Step 7
**Relates to:** `datasets/whole_organ/corpus.py`, `tests/test_whole_organ_corpus.py`

## Context

Step 7 exists because Step 6's corpus carried one relationship graph for every scene,
which made the question "is the graph encoder necessary" unanswerable. The Step 7
corpus answers it by generating the same organ family in four arrangements, so that
presence, entity identity and text features are constant and the relationship graph is
the only channel carrying the arrangement.

The first Step 7 generator assigned

```python
family_id = index % families            # 40 families
variant   = variants[index % len(variants)]   # 4 variants
```

Because the variant count divides the family count, the second expression is a function
of the first. Every family appeared in exactly one arrangement. The validation split
held only `mirrored` and `rotated` scenes; the test split held only `normal` and
`transposed`. The contrast the corpus was built to present was never presented, and any
measured difference between arrangements was also a difference between families.

## Decision

Family, variant and level of detail are crossed independently. The variant advances once
per full pass over the families:

```python
family_id = index % families
variant   = variants[(index // families) % len(variants)]
```

The generator **refuses to write** a corpus in which any family covers fewer than every
variant, or any non-empty split is missing a variant. The crossing is recorded in the
manifest as `variants_per_family` and `split_variants`, so a reader can check it without
re-deriving it.

## Consequences

* Every family is seen in all four arrangements, ten times each at 1,600 scenes.
* Every split contains all four arrangements.
* A corpus with the old defect can no longer be produced silently.
* The first Step 7 suite is superseded and its arm comparison withdrawn. It is kept
  under `experiments/runs/step7-whole-organ-v1-confounded/`.

## Alternatives considered

**Shuffle the assignment randomly.** Rejected: a random assignment is balanced only in
expectation, and the failure mode being guarded against is a silent imbalance. The
deterministic crossing is exactly balanced and reproducible.

**Keep the confounded corpus and control for family statistically.** Rejected: with one
arrangement per family there is no within-family contrast to recover, so no amount of
analysis retrieves the comparison.
