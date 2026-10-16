"""Free text to an AWR program with the Claude API, under the program schema (ADR 0007).

The model is asked for structured output against ``language.program.program_schema``, whose
operations and entity ids are enums, so it can only answer in the closed vocabulary; the
answer is then validated again before anything runs. It never sees or produces geometry.

Choices, all from the Claude API guidance for this SDK:

* ``claude-opus-5-5`` with effort ``low``: turning a sentence into a few typed steps is a
  simple, latency-sensitive task. Thinking stays adaptive (the model's default; it cannot be
  disabled on this model).
* ``fallbacks="default"`` under beta ``server-side-fallback-2026-07-01``: if a safety
  classifier declines, the server retries on its recommended fallback model. A decline that
  survives the fallback is reported as a refusal, never as an empty program.
* The system prompt -- instructions plus the ontology codebook -- is built deterministically
  and marked for caching; the user's text is the only part that varies.

Credentials are resolved by the SDK (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an
``ant auth login`` profile). Nothing here runs unless the language layer was enabled.

Needs the ``language`` extra.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from awr.ontology import AnatomyOntology
from language.program import LEVELS, Program, ProgramError, program_schema, validate_program

__all__ = ["DEFAULT_MODEL", "ClaudeProgramTranslator", "system_prompt"]

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_INSTRUCTIONS = """You translate a person's request about a 3D anatomy scene into a short \
program of typed steps. The scene holds exactly the structures listed below, by id. You never \
describe, invent or draw anatomy; you only choose steps.

Operations:
- show, hide, isolate: change which structures are visible. isolate shows only its targets.
- set_opacity: value is a number from 0 (invisible) to 1 (opaque); "transparent" is about 0.3.
- resize: exactly one target; value is the change in linear size in percent, from -30 to +30 \
("20% larger" is +20, "shrink by 10%" is -10). It reshapes that one structure as real anatomy \
varies, and is refused if the result would be implausible.
- set_lod: value is a level of detail from 0 (whole organs) to 4 (most detailed).
- set_level: value is one of {levels}; it sets the audience level.
- animate_flow and stop_animation: start or stop the blood-flow animation.
- explain: targets are the structures to explain; no targets explains the whole scene.
- reset: return the scene to its defaults.

Rules:
- Use only ids from the list. A group id (such as respiratory.lungs or heart.valves) stands for \
everything in it.
- Use as few steps as the request needs, in the order the person would expect them applied.
- Leave value empty when the operation does not take one.
- If the request asks for something these operations cannot do, or names a structure that is \
not in the list, return no steps and put a one-sentence reason in refusal. Otherwise refusal \
is empty.

Structures (id | name | other names):
{codebook}"""


def system_prompt(ontology: AnatomyOntology) -> str:
    """Instructions plus the codebook, byte-stable for a given ontology so it can be cached."""
    lines = []
    for entity_id in sorted(ontology.ids()):
        entity = ontology.get(entity_id)
        others = ", ".join(sorted({*entity.synonyms, *entity.abbreviations}))
        lines.append(f"{entity_id} | {entity.display_name} | {others}")
    return _INSTRUCTIONS.format(levels=", ".join(LEVELS), codebook="\n".join(lines))


class ClaudeProgramTranslator:
    """Ask Claude for a program; return it validated, or as a refusal with a reason."""

    def __init__(
        self,
        ontology: AnatomyOntology,
        *,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
    ) -> None:
        self.ontology = ontology
        self.client = client if client is not None else anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self._system = system_prompt(ontology)
        self._schema = program_schema(ontology)

    def request(self, text: str) -> dict[str, Any]:
        """The request body, separate from sending it so it can be inspected and tested."""
        return {
            "model": self.model,
            "max_tokens": 16000,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
            "system": [
                {"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}
            ],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self._schema},
            },
            "messages": [{"role": "user", "content": text}],
        }

    def translate(self, text: str) -> Program:
        """One request, one program.

        Raises:
            ProgramError: for an API failure, a truncated answer, or an answer that fails
                validation. A safety decline is not an error: it returns a refusal.

        """
        try:
            response = self.client.beta.messages.create(**self.request(text))
        except anthropic.RateLimitError as exc:
            raise ProgramError("The language service is busy; try again shortly.") from exc
        except anthropic.AuthenticationError as exc:
            raise ProgramError("The language service has no valid credentials.") from exc
        except anthropic.APIStatusError as exc:
            raise ProgramError(f"The language service answered {exc.status_code}.") from exc
        except anthropic.APIConnectionError as exc:
            raise ProgramError("The language service could not be reached.") from exc

        if response.stop_reason == "refusal":
            return Program(refusal="The language model declined this request.")
        if response.stop_reason == "max_tokens":
            raise ProgramError("The language model's answer was cut off.")
        texts = [block.text for block in response.content if block.type == "text"]
        if not texts:
            raise ProgramError("The language model returned no program.")
        try:
            raw = json.loads(texts[0])
        except json.JSONDecodeError as exc:
            raise ProgramError("The language model's program was not valid JSON.") from exc
        return validate_program(raw, self.ontology)
