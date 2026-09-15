# ADR-STEP8-003: Level-of-detail prefixes are made genuinely nested by objective

**Status:** ACCEPTED
**Date:** Step 8
**Relates to:** `generation/neural/nn/nested_lod.py`, `training/step8.py`,
`tests/test_step8_lod.py`

## Context

Step 5 proposed nested geometry token prefixes: a coarse level reads `tokens[0:K1]`, a
finer level reads `tokens[0:K2]` with `K1 < K2`, so the coarse representation is literally
contained in the fine one.

Step 6 supervised every prefix against the same full-detail target, which asked a coarse
prefix for detail it cannot represent, and the later tokens went largely unused.

Step 7 fixed the target, supervising each prefix against its own level. The result was
worse and more informative: detail gain was −0.14 while the token delta was 0.16. The
extra tokens were being **used** and the output got **worse**. The diagnosis recorded then:
nothing required the coarse prefix to be a prefix of the fine decode rather than an
independent summary, so the model was free to use the first eight tokens for one thing and
the next eight to overwrite it.

## Problem

The mechanism was already nested. `OccupancyFieldDecoder` reads a literal prefix, and the
corpus's own targets nest too: the 10 entities at level 1 are a subset of the 16 at level
2, a subset of the 20 at level 3. What was missing was an objective asking the
representation to behave the way the truth does.

## Alternatives considered

**A projection head per level.** Rejected, and this is the important rejection. Separate
heads would let each level learn an unrelated output while the prefix structure became
decorative, which is precisely the failure being fixed. `GeometryConfig.per_lod_blocks`
stays false and a test asserts it.

**A hard constraint: freeze the coarse decode when adding tokens.** Rejected. It forbids
the fine prefix from correcting the coarse prefix's mistakes, which is most of what
refinement is for.

**Weight the finer levels more heavily.** Rejected as a guess. It addresses none of the
mechanisms that could produce the observed behaviour.

## Decision

Two terms beside the per-level reconstruction, both computed between adjacent levels
decoded in the **same** forward pass:

| Term | What it asks | Weight |
| --- | --- | ---: |
| `reconstruction` | each prefix matches its own level's target | 1.0 |
| `containment` | a point the coarse prefix calls occupied stays occupied when tokens are added | 0.5 |
| `preservation` | where the coarse prefix is already correct **and the levels agree**, the fine prefix agrees | 0.5 |

Three details decide whether this works.

**Containment is one-sided.** The fine prefix may add occupancy freely, which is what
adding detail means; it may not remove what the coarse prefix established.

**Preservation carries two masks.** The first keeps points the coarse decode got right.
The second keeps points where the two levels' targets **agree**, which is what stops the
term punishing refinement itself. An earlier version took a single target and scored every
point the finer level legitimately added as a failure; a test now pins the corrected
behaviour.

**The coarse side of both terms is detached.** Otherwise the cheapest way to satisfy
nesting is for both levels to predict nothing.

All three levels are decoded every step. Step 7 rotated through one level per step, which
meant the containment relation between two levels was never present in a single graph and
could not be asked for.

## Evidence

Step 7's detail gain of −0.14 with a token delta of 0.16 is the motivation. The Step 8
evidence is `lod_containment_mean`, `lod_preservation_mean` and `lod_detail_gain` under
predicted placement, reported per arm in the completion report, together with the R11
non-nested comparison.

## Consequences

* Training costs four forward passes per step rather than two, because every level is
  decoded every step.
* Detail gain, containment and preservation are reported together. A high containment with
  a negative detail gain would mean the representation nests and does not refine, which is
  a different verdict from Step 7's and would be reported as such.

## Reversal conditions

Reverse this decision if:

* containment and preservation reach their ceiling while detail gain stays negative, which
  would mean nesting was enforced and refinement still did not happen, and the nested
  prefix design should be marked REVISE or REMOVE rather than tuned further;
* a per-level head is shown to beat the nested prefix on every level simultaneously, in
  which case the nesting property is not buying what it was proposed to buy.

This is the second objective tried for nested prefixes. A third should not be attempted
without a mechanism that explains why the first two failed.
