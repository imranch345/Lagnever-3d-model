# Experiment 7 — Level of detail. Method

## The Step 6 problem

Step 6 supervised every token prefix against the **same** full-detail target. A coarse
prefix was therefore asked to reproduce fine geometry it has no capacity to represent,
and the later tokens showed little use. The most likely reading was that the objective,
not the architecture, was at fault.

## The fix under test

Step 7 supervises each prefix against **that level's own target**. The corpus already
carries a per-level occupancy target: level 1 exposes 10 entities, level 2 adds valves
and septa for 16, level 3 all 20. A coarse prefix is now asked for a coarse organ, which
is a request it can satisfy.

```
level = (step % 3) + 1
prefix = LOD_TOKEN_SCHEDULE[level]
target = whole.lod_targets[level]
loss = binary_cross_entropy_with_logits(model(batch, token_prefix=prefix), target)
```

## Metrics

| Metric | Definition |
| --- | --- |
| `lod1_iou`, `lod2_iou`, `lod3_iou` | occupancy IoU at each prefix, against that level's own target |
| `lod_detail_gain` | `lod3_iou − lod1_iou`; positive means finer prefixes help |
| `lod_token_delta` | mean absolute difference between the coarsest and finest decoded fields; near zero means the extra tokens change nothing |

`lod_token_delta` is the honest check on `lod_detail_gain`. A model could score similarly
at both levels while genuinely using the extra tokens, or while ignoring them entirely.
Only the second produces a near-zero delta.

## What counts as a negative result

Stated before the runs: if higher levels of detail do not improve geometry against
level-specific targets, the nested level-of-detail design is marked REVISE or REMOVE. The
level-specific objective is the second attempt at making it work; a second failure is
reported as a failure of the design, not as a third objective to try.

One asymmetry to keep in mind when reading the numbers: level 3 is a harder target than
level 1, since it asks for twenty structures rather than ten. A lower `lod3_iou` than
`lod1_iou` is therefore not on its own evidence that the extra tokens are wasted. The
quantity that separates the two readings is `lod_token_delta`.
