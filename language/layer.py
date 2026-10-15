"""Which path a request takes: the free rule parser first, Claude only when needed and enabled.

* If the deterministic parser understands the text, the engine runs it as before. No model is
  called and nothing is spent.
* Otherwise, if a translator is configured, it turns the text into a program; the program is
  shown (it is part of the result), validated, and applied step by step as canonical commands
  through the same engine, stopping at the first step the engine rejects.
* Otherwise the parser's own "not understood" answer is returned, with a note that free-form
  requests need the language layer enabled.

The Claude translator is opt-in: ``translator_from_environment`` returns one only when
``LAGNAV_LANGUAGE=claude`` and the SDK is installed. Credentials are the SDK's to resolve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from awr.ontology import AnatomyOntology
from language.program import Program, ProgramError, to_commands
from reasoning.command_engine import CommandEngine

__all__ = ["LanguageLayer", "LanguageResult", "Translator", "translator_from_environment"]

ENABLE_VARIABLE = "LAGNAV_LANGUAGE"


class Translator(Protocol):
    """Anything that turns text into a validated program."""

    def translate(self, text: str) -> Program:
        """Return a program or a refusal; raise ``ProgramError`` on failure."""
        ...


@dataclass
class LanguageResult:
    """What happened to one request, including the program when one was made."""

    ok: bool
    message: str
    source: str
    program: dict[str, Any] | None = None
    commands: list[str] = field(default_factory=list)


def translator_from_environment(ontology: AnatomyOntology) -> Translator | None:
    """The Claude translator when explicitly enabled, else ``None``."""
    if os.environ.get(ENABLE_VARIABLE, "").strip().lower() != "claude":
        return None
    try:
        from language.claude import ClaudeProgramTranslator
    except ImportError:
        return None
    return ClaudeProgramTranslator(ontology)


class LanguageLayer:
    """Route one request through the parser or the translator, and the engine either way."""

    def __init__(self, engine: CommandEngine, translator: Translator | None = None) -> None:
        self.engine = engine
        self.translator = translator

    def execute(self, text: str) -> LanguageResult:
        """Apply one request; never raises for a request the system cannot carry out."""
        if self.engine.parser.parse(text).is_understood or self.translator is None:
            response = self.engine.execute(text)
            message = str(response.message)
            if not response.ok and self.translator is None:
                message += (
                    f" Free-form requests need the language layer: set {ENABLE_VARIABLE}=claude."
                )
            return LanguageResult(ok=bool(response.ok), message=message, source="rules")

        try:
            program = self.translator.translate(text)
        except ProgramError as exc:
            return LanguageResult(ok=False, message=str(exc), source="claude")
        if program.refusal:
            return LanguageResult(
                ok=False, message=program.refusal, source="claude", program=program.to_dict()
            )

        commands = to_commands(program)
        messages: list[str] = []
        for command in commands:
            response = self.engine.execute(command)
            messages.append(str(response.message))
            if not response.ok:
                return LanguageResult(
                    ok=False,
                    message=" ".join(messages),
                    source="claude",
                    program=program.to_dict(),
                    commands=commands,
                )
        return LanguageResult(
            ok=True,
            message=" ".join(messages),
            source="claude",
            program=program.to_dict(),
            commands=commands,
        )
