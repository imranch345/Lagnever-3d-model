# ADR-L1-005: Free text reaches Claude only as a request for a closed AWR program

**Status:** ACCEPTED
**Date:** Level 1, 2026-10-16
**Relates to:** `language/`, `viewer/server.py`, [ADR 0007](0007-language-program-interface.md),
`tests/test_language.py`

## Context

The rule parser understands a controlled vocabulary: "hide the trachea", "make the left
ventricle transparent", "show only the lungs". People do not talk like that. The user decided
on 2026-10-15 that the Claude API is the language model to use, when one is needed.

ADR 0007 sets the safety property. Language may only produce a verifiable program over the
closed ontology, never geometry, and anatomy must not be inventable.

## Decision

1. **Rules first, Claude second.** `LanguageLayer` sends text to the deterministic parser.
   If the parser understands it, the engine runs it as before, and no model is called.
   Only text the parser cannot read goes to the translator.
2. **Off by default.** The Claude translator exists only when `LAGNAV_LANGUAGE=claude` is set
   and the `language` extra (`anthropic`) is installed. Credentials are the SDK's to resolve.
   Nothing is spent unless someone turns it on.
3. **A closed program, three times checked.**
   - The model answers in structured output against `program_schema`. Ten operations and the
     ontology's entity ids are enums, so it can only express what the scene contains.
   - `validate_program` checks the answer again: operations, ids, value ranges, and no
     program that both refuses and acts.
   - The program runs as canonical commands through `CommandEngine`, which resolves and
     applies each one. It stops at the first rejection.
4. **The request.**
   - Model: `claude-opus-5-5`.
   - Effort: `low`. Turning a sentence into a few typed steps is simple and
     latency-sensitive. Thinking stays adaptive; it cannot be disabled on this model.
   - `fallbacks: "default"` under `server-side-fallback-2026-07-01`. A safety decline is
     retried on the server's recommended fallback, and a decline that survives is returned as
     a refusal.
   - The system prompt holds the instructions and the codebook. It is byte-stable and marked
     for caching.
5. **Visible.** The viewer's reply says which path ran. For Claude it includes the program and
   the commands applied, and the page logs them.

## Alternatives considered

**Free-form JSON or tool calls without enums.** Rejected by ADR 0007: a model can then name a
structure that does not exist, and the only defence left is after the fact.

**Claude for every request.** Rejected. It costs money and latency for sentences the parser
already handles exactly. The parser also stays the oracle a learned translator is judged
against.

**Let the model write commands as text.** Rejected. The parser would then have to guess what
the model meant. A typed program over ids leaves no second interpretation.

## Consequences

* Turning it on is one environment variable plus credentials. Turning it off returns exactly
  the previous behaviour.
* Asking for anatomy that is not in the scene ("show the spleen") yields a refusal with a
  reason, never a step.
* A new capability needs a vocabulary change: `OPERATIONS`, its command mapping and the prompt.
  It does not come from rewording a prompt.
* The tests run with a fake client and send nothing. They pin the request (model, effort,
  fallbacks, schema, cached system prompt) and the handling of every kind of answer: program,
  refusal, truncation, invalid JSON, invented anatomy, API failure.
