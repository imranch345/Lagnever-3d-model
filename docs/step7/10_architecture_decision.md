# Step 7 — Architecture decisions

Every component the Step 5 design proposed is assigned one of four verdicts. The rules
for assigning them are fixed here **before** the results section, so that a verdict is the
output of a rule rather than a preference.

## The four verdicts

| Verdict | Meaning | What triggers it |
| --- | --- | --- |
| **KEEP** | the component earns its place as designed | it measurably improves an outcome, or removing or corrupting it measurably degrades one |
| **SIMPLIFY** | the idea is right, the implementation is larger than it needs to be | a smaller or cheaper version matches the full one within noise |
| **REVISE** | the component does not work as built, but the reason is understood and addressable | it fails, and a specific mechanism explains the failure |
| **REMOVE** | the component is not doing anything | deleting or corrupting it produces no measurable effect, and no evidence suggests it would with more of anything |

## Rules of assignment

**1. Evidence, not intent.** A component is not kept because it ought to help. If the
experiments do not show it helping, it does not get KEEP.

**2. Untestable is not the same as unnecessary.** Where the setup could not have
distinguished the hypotheses, the correct verdict is that the question is **open**, and
the component is carried forward unchanged with the open question recorded. Three of
Step 7's earlier findings were of exactly this kind. Marking a component REMOVE on the
basis of an untestable comparison would be the same error as keeping one on the basis of
optimism.

**3. Condition-bound results are labelled.** A finding that holds only with placement
supplied, or only at a 900-step budget, is recorded with that condition attached. It does
not silently become a general claim.

**4. Structural facts are not achievements.** Where a result follows from the
construction rather than from anything a model learned, it is reported as structural and
does not support a KEEP.

**5. Withdrawal is cheap, silence is not.** Any verdict that later evidence contradicts
is withdrawn in writing, with the reason. Nothing is deleted to make a story tidier.

## Components under judgement

| Component | Source | Parameters |
| --- | --- | --- |
| Anatomical World Representation as the source of truth | Step 4 | none, it is data |
| Three typed graphs, structure / spatial / functional | Step 4 | none, it is data |
| Write-protected identity subspace | Step 5 | part of the entity composer |
| Partitioned relational graph transformer | Step 5 | 2,666,896 |
| Per-entity geometry token blocks | Step 5 | 101,888 plus decoder |
| Nested level-of-detail token prefixes | Step 5 | shared |
| Entity frame head | Step 5 | 3,084 |
| Prototype-hub multimodal alignment | Step 5 | 76,544 |
| Persistent local editing | Step 5, built in Step 7 | 83,777 in the edit head |
| Geometry correspondence as a tensor axis | Step 5 | none, it is a shape |

Verdicts are recorded in the completion report alongside the evidence for each.
