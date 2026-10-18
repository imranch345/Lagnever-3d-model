"""Which path a request takes: the free rule parser first, Claude only when needed and enabled.

* If the deterministic parser understands the text, the engine runs it as before. No model is
  called and nothing is spent.
* Otherwise, if a translator is configured, it turns the text into a program; the program is
  shown (it is part of the result), validated, and applied step by step as canonical commands
  through the same engine, stopping at the first step the engine rejects.
* Otherwise the parser's own "not understood" answer is returned, with a note that free-form
  requests need the language layer enabled.

Resizing ("make the left ventricle 20% larger") is read by a rule of its own before any of
that, so it works without the language model. It has no engine command: the host supplies a
``resize`` hook that reshapes the geometry through the shape model, and only when that
succeeds is the new size written to the AWR, as the entity's transform scale (ADR-L2-001).

The Claude translator is opt-in: ``translator_from_environment`` returns one only when
``LAGNAV_LANGUAGE=claude`` and the SDK is installed. Credentials are the SDK's to resolve.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from awr.ontology import AnatomyOntology
from awr.schema import Transform
from editing.operations import SetTransform
from language.program import Program, ProgramError, to_commands, validate_program
from reasoning.command_engine import CommandEngine

__all__ = [
    "LanguageLayer",
    "LanguageResult",
    "ResizeHook",
    "Translator",
    "load_user_env",
    "parse_resize",
    "translator_from_environment",
]

ENABLE_VARIABLE = "LAGNAV_LANGUAGE"

#: Reshape ``entity`` to ``factor`` times its linear size; return (applied, message).
ResizeHook = Callable[[str, float], tuple[bool, str]]

_PERCENT = r"(?P<sign>[+-])?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)"
_RESIZE_FORMS = (
    re.compile(
        r"^(?:please\s+)?(?:make|render)\s+(?:the\s+)?(?P<target>.+?)\s+"
        + _PERCENT
        + r"\s+(?P<direction>larger|bigger|smaller)[.!]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?(?P<verb>enlarge|grow|shrink|reduce|resize)\s+(?:the\s+)?"
        r"(?P<target>.+?)\s+(?:by\s+)?" + _PERCENT + r"[.!]?$",
        re.IGNORECASE,
    ),
)


def parse_resize(text: str) -> tuple[str, float] | None:
    """The target phrase and the signed percentage of a resize request, or None."""
    for form in _RESIZE_FORMS:
        match = form.match(text.strip())
        if match is None:
            continue
        percent = float(match["pct"])
        groups = match.groupdict()
        shrinking = (groups.get("direction") or "").lower() == "smaller" or (
            groups.get("verb") or ""
        ).lower() in {"shrink", "reduce"}
        if shrinking or match["sign"] == "-":
            percent = -percent
        return match["target"], percent
    return None


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
    #: Engine operations applied (``CommandResponse.operation``), e.g. "reset_view".
    operations: list[str] = field(default_factory=list)


#: Where the user keeps the API key: outside the repository, readable only by them.
USER_ENV_FILE = Path.home() / ".lagnav" / "anthropic.env"
#: The only variables that file may set.
_USER_ENV_KEYS = frozenset({"ANTHROPIC_API_KEY", ENABLE_VARIABLE})


def load_user_env(path: Path = USER_ENV_FILE) -> list[str]:
    """Set the allowed variables from the user's key file; return their names, never values.

    Lines are ``NAME=value``. Variables already in the environment win, so a shell can always
    override the file. Unknown names are ignored. If the key file can be read by the group or by
    others, it is refused, so a key is not used from a file anyone else could read.
    """
    if not path.is_file():
        return []
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"{path} is readable by others; run: chmod 600 {path}")
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        name, value = name.strip().removeprefix("export ").strip(), value.strip().strip("\"'")
        if not sep or name not in _USER_ENV_KEYS or not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


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

    def __init__(
        self,
        engine: CommandEngine,
        translator: Translator | None = None,
        resize: ResizeHook | None = None,
    ) -> None:
        self.engine = engine
        self.translator = translator
        self.resize = resize

    def execute(self, text: str) -> LanguageResult:
        """Apply one request; never raises for a request the system cannot carry out."""
        requested = parse_resize(text)
        if requested is not None:
            return self._resize_request(text, *requested)
        if self.engine.parser.parse(text).is_understood or self.translator is None:
            response = self.engine.execute(text)
            message = str(response.message)
            if not response.ok and self.translator is None:
                message += (
                    f" Free-form requests need the language layer: set {ENABLE_VARIABLE}=claude."
                )
            return LanguageResult(
                ok=bool(response.ok),
                message=message,
                source="rules",
                operations=[response.operation] if response.ok and response.operation else [],
            )

        try:
            program = self.translator.translate(text)
        except ProgramError as exc:
            return LanguageResult(ok=False, message=str(exc), source="claude")
        if program.refusal:
            return LanguageResult(
                ok=False, message=program.refusal, source="claude", program=program.to_dict()
            )

        commands: list[str] = []
        messages: list[str] = []
        operations: list[str] = []
        for step in program.steps:
            if step.op == "resize":
                commands.append(f"resize {step.targets[0]} {step.value}%")
                ok, message = self._apply_resize(step.targets[0], float(step.value), text)
                messages.append(message)
                if not ok:
                    break
                continue
            ok = True
            for command in to_commands(Program(steps=(step,))):
                commands.append(command)
                response = self.engine.execute(command)
                messages.append(str(response.message))
                if not response.ok:
                    ok = False
                    break
                if response.operation:
                    operations.append(str(response.operation))
            if not ok:
                break
        else:
            ok = True
        return LanguageResult(
            ok=ok,
            message=" ".join(messages),
            source="claude",
            program=program.to_dict(),
            commands=commands,
            operations=operations,
        )

    def _resize_request(self, text: str, phrase: str, percent: float) -> LanguageResult:
        """A resize read by the rule: resolve one structure, validate, apply."""
        resolution = self.engine.resolver.try_resolve(phrase)
        if resolution is None:
            return LanguageResult(False, f"No structure called {phrase!r}.", "rules")
        if resolution.is_group:
            return LanguageResult(
                False,
                f"{resolution.display_name} is a group; resize one structure at a time.",
                "rules",
            )
        try:
            program = validate_program(
                {"steps": [{"op": "resize", "targets": [resolution.entity_id], "value": percent}]},
                self.engine.ontology,
            )
        except ProgramError as exc:
            return LanguageResult(False, str(exc), "rules")
        step = program.steps[0]
        ok, message = self._apply_resize(step.targets[0], float(step.value), text)
        return LanguageResult(
            ok,
            message,
            "rules",
            program=program.to_dict(),
            commands=[f"resize {step.targets[0]} {step.value}%"],
        )

    def _apply_resize(self, entity: str, percent: float, text: str) -> tuple[bool, str]:
        """Geometry first, through the hook; the AWR records the size only if it succeeded."""
        if self.resize is None:
            return False, "Resizing needs a shape model: start the viewer with --model."
        factor = 1 + percent / 100
        ok, message = self.resize(entity, factor)
        scene, editor = self.engine.scene, self.engine.editor
        if ok and scene is not None and editor is not None:
            before = scene.get(entity).state.transform
            x, y, z = before.scale
            scale = (x * factor, y * factor, z * factor)
            editor.apply(
                SetTransform((entity,), Transform(before.translation, before.rotation, scale)),
                command_text=text,
            )
        return ok, message
