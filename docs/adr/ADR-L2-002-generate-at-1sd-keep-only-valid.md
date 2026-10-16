# ADR-L2-002: Generated thoraxes are drawn at 1 SD and kept only if they pass validity

**Status:** ACCEPTED
**Date:** Level 2a, 2026-10-16
**Relates to:** `shapes/generate.py` (`write_valid_sample_scene`), `viewer/server.py`,
`docs/LEVEL_2A_REPORT.md` §6, `docs/LEVEL_2_PLAN.md` §3.5

## Context

The plan (§3.5) said to sample at the largest spread whose samples pass the hard checks at a
stated rate. The validity sweep put that at **0.5 SD**: 50 of 50 samples passed there, against
78% at 1 SD. At 0.5 SD, though, samples are closer to the training set than real people are.

Plausibility and novelty were measured for four samplers. Novelty is the mean vertex distance
to the nearest training subject, median over 50 samples. Validity is the share passing the
calibrated checks.

| sampler | novelty | validity |
| --- | --- | --- |
| *a new real person (held out)* | *9.1 mm* | *90.5%* |
| Gaussian, 0.5 SD | 6.7 mm | 100% |
| Gaussian, 1 SD | 9.05 mm | 78% |
| around training subjects, h = 0.25 SD (kernel density) | 4.5 mm | 88% |
| around training subjects, h = 0.5 SD | 6.9 mm | 86% |
| **Gaussian, 1 SD, kept only if valid** | **8.8 mm** | **100% of shown** (80% of draws kept) |

Sampling around training subjects looked promising by pass rate, but it buys plausibility by
copying. At h = 0.25 a sample is half as far from a training subject as a new person is. At
h = 0.5 it is no more novel than 0.5 SD Gaussian draws, and passes less often. It was
rejected.

## Decision

**Generate at 1 SD, and keep a draw only if it passes every validity check.**
- Draws come from the seed's own sequence (`seed × 1000 + try`), so a seed always gives the
  same thorax.
- After 6 rejected draws, the thorax is drawn at 0.5 SD instead, where every swept sample
  passed.
- The manifest records the sampler, the number of tries and each rejected draw's failures.
- The viewer says which try succeeded.

## Consequences

* Shown thoraxes are as novel as a new real person (8.8 against 9.1 mm), and each passes
  checks that real anatomy passes 90.5% of the time.
* The distribution shown is truncated to the valid region. This is stated in every manifest.
  It is the same kind of gate that Level 1 applies to real scans (131 of 200 kept).
* Generation takes about 8 s per try for the validity check, so about 10 s on average. The
  pure-Python inside test is most of that.

## How this could be wrong

The checks are what define "valid". Anything they do not measure is not filtered: the shape
of a single structure beyond folding, for instance, or vessel positions, which are not
modelled. A thorax that passes is plausible by these checks, not proven anatomical.
