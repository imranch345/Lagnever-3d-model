# ADR 0010: Two dependency tiers, with PyTorch confined to the second

**Status:** ACCEPTED
**Date:** Step 6
**Relates to:** `pyproject.toml`, `tests/test_project_structure.py`

## Context

Step 5 deliberately added no deep-learning framework
([ADR 0009](0009-no-framework-in-step-5.md)). Step 6 needs one. The Step 4 rule
that the core imports only the standard library and PyYAML is enforced by test and
has been useful: it keeps the Anatomical World Representation installable,
inspectable and fast to test.

Deleting that rule to make room for PyTorch would trade a real architectural
property for convenience.

## Decision

Two tiers, both enforced by test.

**Tier 1: standard library plus PyYAML.** `awr`, `reasoning`, `editing`,
`animation`, `geometry`, `evaluation`, `datasets` (the schema), `generation` and
`generation.neural`. These may not reference `torch` or `numpy` anywhere, at module
level or inside a function.

**Tier 2: PyTorch and NumPy permitted.** `generation.neural.nn`,
`datasets.synthetic`, `training`, `visualization` and `experiments`. These are the
prototype, the corpus generator, the training loop and the inspection tools.

PyTorch and NumPy are an optional extra, `pip install -e ".[prototype]"`. Without them
the tier-one suite still runs in full: **375 tests pass on Python 3.11 with only PyYAML
and pytest installed**, and the prototype test modules are skipped rather than failing
at collection. With the extra installed the suite is 462 tests.

## Alternatives considered

**One tier, torch everywhere.** Simplest. Rejected: the AWR would stop being
usable without a 200 MB dependency, and the deterministic prototype's value is
partly that it is cheap to run anywhere.

**Separate repository for the neural work.** Clean separation. Rejected: the
contracts are shared, and two repositories would drift. Revisit if the prototype
grows its own release cycle.

**Lazy imports inside tier-1 functions.** Would let `evaluation` hold both metric
declarations and implementations. Rejected because it makes the rule unenforceable
by a simple check, and an unenforceable rule decays. Implemented metrics therefore
live in `generation.neural.nn.metrics`, and `evaluation.neural_metrics` keeps the
declarations and points at them.

## Consequences

* The AWR core stays installable with one runtime dependency.
* The boundary is checked by `test_dependency_tiers_are_respected`, which replaced
  the older flat check and is strictly stronger: it now asserts both that tier 1 is
  clean and that tier 2 is the only place the heavy dependencies appear.
* `tests/conftest.py` skips the prototype modules when PyTorch is absent, so the claim
  above is verified by running the suite in a PyTorch-free environment rather than
  asserted.
* Tier-1 tests run in under a second, which keeps the fast feedback loop.
* One awkwardness, accepted: metric declarations and metric implementations live in
  different packages.
