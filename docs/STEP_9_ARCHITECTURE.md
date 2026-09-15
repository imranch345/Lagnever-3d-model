# Step 9 architecture

The architecture **as implemented**. Step 9 changes two things inside the frame predictor
and nothing else; everything not described here is the Step 8 architecture, documented in
[STEP_8_ARCHITECTURE.md](STEP_8_ARCHITECTURE.md), unchanged.

Synthetic research data throughout. Nothing here is anatomy, is medically validated, or
supports any clinical claim.

## What Step 9 changes

| Component | Step 8 | Step 9 |
| --- | --- | --- |
| Frame head input | the entity latent alone | optionally the latent plus a scene summary |
| Frame objective, translation | per-component L1 | optionally Euclidean distance |
| Frame objective, scale and rotation | L1 at 0.5 and 0.25 | unchanged |
| Frame target | centroid, log-extent, constant identity rotation | **unchanged** |
| Everything else | | **unchanged** |

Both switches default to off, so a Step 9 control is the Step 8 system bit for bit. That is
checked by a test rather than asserted: the default frame head is still 135,180 parameters
and every other parameter group is identical.

## Why the frame head is where the change goes

The audit that opened Step 9 found three things in the existing code.

**The frame target is six numbers, not twelve.** Translation is the entity's measured
centroid in scene coordinates and scale is the log of its measured extent, but the rotation
is the identity for every entity in every scene:

```text
rotation 6D over 40 scenes x 20 entities : exactly one unique row, std 0 everywhere
measured rotation_error, every Step 8 arm : 0.000000
```

The organ's own yaw, pitch and roll reach the target only indirectly, by moving centroids.

**The head reads a local latent and predicts a global quantity.** `FramePredictor` took
`entity_latent` and nothing else. The graph mask connects each entity to about 4.2
neighbours, there is no global pooling anywhere in the encoder, and the target is a
scene-global centroid. The information needed to place an entity had never reached the head
that places it.

**It is not interference, and it is not a missing slice of the latent.** Two read-only
probes on a frozen trained backbone: fitting the frame head alone on the frame objective for
600 steps moved position error from 0.1614 to 0.1603; probes on the identity subspace alone,
the context alone and the full latent reached 0.1675, 0.1611 and 0.1610. All three sit at
the same plateau.

## The scene summary

`generation/neural/nn/model.py`

```text
weights = present[..., None]                        [B, N, 1]
summary = (entity_latent * weights).sum(1) / weights.sum(1).clamp_min(1)
                                                    [B, W]
```

A masked mean of the present entities' latents. Three properties, each tested:

* **permutation invariant**, so it cannot become a positional shortcut;
* **masked by presence**, so a scene's summary does not depend on how many slots happen to
  be padded;
* **a function of latents the model already produced**, so it introduces no new information
  and cannot leak placement.

## The frame head with context

`generation/neural/nn/geometry.py`

```text
input    entity_latent  [B, N, 256]
         context        [B, 256]      required iff scene_context is set

         LayerNorm(entity_latent)  and  LayerNorm(context) broadcast over N
         concat                                   [B, N, 512]
         Linear(512 -> 256) -> GELU
         Linear(256 -> 256) -> GELU
         Linear(256 -> 12)
output   [B, N, 12]     translation[0:3], log-scale[3:6], 6D rotation[6:12]
```

135,180 parameters without context, 201,228 with. The output projection is zero-initialised
at the canonical frame in both, so the two start from the same scene and a difference
between them is something one of them learned.

Passing a context to a head built without one, or omitting it from a head built with one,
raises. Silently ignoring either would disable the mechanism under test while the experiment
appeared to run.

## The frame objective

`training/step9.py`

```text
l1         translation = |dx| + |dy| + |dz|        the Step 8 term, bit for bit
euclidean  translation = sqrt(dx^2 + dy^2 + dz^2)  the distance the metric measures

both       + 0.5 * sum|d log-scale|
           + 0.25 * sum|d rotation|                unchanged, and structurally zero
```

L1's optimum is the per-axis median; the reported `position_error` is Euclidean, whose
optimum is the geometric median. These are different points, and the gap between them is
not the model's fault. `euclidean` removes the discrepancy. Scale stays L1 because scale is
reported per axis.

## The placement-blind floor

`experiments/step9/placement_floor.py`

A first-class baseline, elevated to the same standing as the relation-blind floor. Three
reference points, computed from the corpus with no model:

| Predictor | What it uses |
| --- | --- |
| `global` | one frame for every entity in every scene |
| `per_entity` | **the floor**: each entity's mean frame from the training split, a lookup table keyed on identity and nothing else |
| `oracle` | the scene's own frame; zero by construction |

A model that does not beat `per_entity` has learned nothing about placement that identity
alone did not already supply.

**Masking is enforced structurally.** The floor is computed by driving the same loader and
the same presence mask the evaluation uses, not by iterating the corpus. The first version
of this measurement iterated the corpus and produced 0.1401, which is 0.02 optimistic and
made every Step 8 arm look worse than a lookup table. Measured correctly the floor is
0.1605 on `test_seen` and two arms are marginally above it. A test pins the difference.

## Reported metrics, and what each is for

| Metric | Meaning |
| --- | --- |
| `translation_error` | Euclidean centroid error, in scene units |
| `scale_error` | mean absolute log-scale error, a symmetric ratio error |
| `rotation_error` | geodesic angle. **Structurally zero** under the current target |
| `composite_frame_error` | the Step 8 quantity, `position + 0.5*scale + 0.25*rotation`, **unchanged** |
| `placement_error` | the Step 9 corrected metric, `position + 0.5*scale`, with the vacuous term removed |
| `rotation_information` | `0.25 * rotation_error`: what the rotation term contributes, which is zero |
| `placement_floor_gap` | `(floor - error) / floor`. Positive means better than the lookup table |

Both the Step 8 composite and the corrected metric are reported for every arm on every
split. Neither replaces the other: the composite preserves comparability with Step 8, and
the difference between them is exactly the information the rotation term carries.

## What did not change

The Anatomical World Representation, persistent entity identifiers, the entity hierarchy,
the three typed graphs, entity-local geometry, geometry correspondence through the entity
axis, the write-protected identity subspace, scene memory, the deterministic command path,
the explicit geometry decoder, mesh as an export, the anatomy/presentation separation, the
nested level-of-detail prefixes, predicted-placement evaluation, and every Step 8 leakage
control.

The teacher-forcing curriculum is unchanged in the Step 9 variants, so the two switches are
measured against the Step 8 schedule rather than confounded with a change to it.
