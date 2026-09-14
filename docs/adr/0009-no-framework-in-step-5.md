# ADR 0009: No deep-learning framework in the design phase

**Status:** ACCEPTED
**Date:** Step 5
**Relates to:** `generation/neural/contracts.py`

## Context

Step 5 is an architecture and representation design phase. The obvious way to
"design" a model is to start writing modules in a framework.

## Decision

Step 5 adds no framework dependency. The architecture is expressed as tensor
contracts, named dimensions, typed interfaces and symbolic bookkeeping that needs
no model. Arrays are a small framework-free type. Framework selection belongs to
Step 6.

## Alternatives considered

**Write the model in PyTorch now.** Concrete, and shapes become real. Rejected for
this phase: it converts open questions into code, and untrained modules invite the
impression that something works. Revisit at the start of Step 6, which is exactly
what it is for.

**Write it in JAX or another framework.** Same reasoning, plus it would pre-empt a
choice with nothing yet to justify it.

**Pseudo-code in the document only.** Cheaper still. Rejected: pseudo-code cannot
be tested, and contracts that are not executed drift from reality within weeks.

## Consequences

* The repository keeps its single runtime dependency, and Step 4's dependency rule
  still passes its test.
* The contracts are executable and tested: a shape trace runs end to end from a
  real heart scene.
* Some things cannot be checked without a framework, including numerical
  stability, memory behaviour and whether the attention pattern trains at all.
* Step 6 begins with a mechanical translation task rather than a design task.

## How this could be wrong

Contracts validated only symbolically can still be impractical in a real
framework, for instance if a dense per-head mask turns out to be a memory problem
at a larger entity count. The first Step 6 task is to build the prototype and find
out.
