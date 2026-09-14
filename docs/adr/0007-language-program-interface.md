# ADR 0007: Language emits verifiable AWR programs, never geometry

**Status:** PROPOSED (unvalidated)
**Date:** Step 5
**Relates to:** `generation/neural/language_encoder.py`

## Context

A text-to-3D system maps language to geometry. Lagnav's requirement is different:
language must operate on a persistent anatomical scene, and it must be impossible
for it to invent anatomy that does not exist.

## Decision

Two paths with different authority.

* **Control path.** Text becomes an AWR program: a short sequence of typed
  operations whose entity arguments are pointers into the closed ontology
  codebook. The program is validated and applied by the Step 4 engine.
* **Conditioning path.** A pooled text embedding influences appearance, style and
  register. It has no path to the AWR at all.

`ProgramTranscoder` turns the Step 4 parser and resolver into a program oracle, so
stage S1 has ground-truth supervision at no data cost.

## Alternatives considered

**Text straight to geometry conditioning.** What appearance-driven systems do, and
what the baseline arm does. Rejected for Lagnav: nothing is inspectable, nothing is
verifiable before it takes effect, and anatomy can be hallucinated.

**Free-form structured output, for example JSON from a language model.** Flexible,
and easy to prototype. Rejected: a free-form generator can emit entity names that
do not exist. Constrained decoding over a closed codebook is not an optimisation
here, it is the safety property.

**Latent instruction embedding applied directly to the scene latent.** Compact and
end-to-end differentiable. Deferred: no intermediate representation to show a user
or to validate, and no way to refuse an instruction with a reason.

## Consequences

* Anatomical hallucination is structurally impossible on the control path.
* Every instruction can be shown, logged, validated or refused before it applies.
* Step 4 becomes both the teacher and a permanent validator of the learned parser.
* The operation vocabulary is closed, so a genuinely new capability requires a
  deliberate vocabulary change rather than a prompt.

## How this could be wrong

A closed vocabulary may prove too rigid for open-ended instructions, and the
boundary between "style" (conditioning) and "content" (control) may turn out to be
blurrier than this split assumes.
