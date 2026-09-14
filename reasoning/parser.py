"""Command parsing: text to a typed, scene-independent command.

The parser answers one question: *what did the user ask for?* It never resolves
entity names, never reads the scene and never mutates anything. Its output is a
:class:`ParsedCommand` carrying an intent, an unresolved target selector and
typed parameters.

That boundary is the point of this module. A future language-model parser only
has to produce the same :class:`ParsedCommand`; the resolver, the AWR, the
editing operations and the scene never learn that the parser changed. The
deterministic rule parser stays useful afterwards as a fast path, a fallback and
a test oracle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from awr.errors import NotYetImplementedError

__all__ = [
    "Intent",
    "SelectorKind",
    "TargetSelector",
    "ParsedCommand",
    "CommandParser",
    "RuleBasedCommandParser",
    "LanguageModelCommandParser",
]


class Intent(StrEnum):
    """What the user wants done."""

    GENERATE = "generate"
    SHOW = "show"
    HIDE = "hide"
    ISOLATE = "isolate"
    SET_OPACITY = "set_opacity"
    SET_LOD = "set_lod"
    SET_EDUCATIONAL_LEVEL = "set_educational_level"
    RESET_VIEW = "reset_view"
    ANIMATE = "animate"
    STOP_ANIMATION = "stop_animation"
    EXPLAIN = "explain"
    UNKNOWN = "unknown"


class SelectorKind(StrEnum):
    """How the command's targets are specified."""

    NONE = "none"
    """No target, e.g. "switch to medical level"."""
    PHRASES = "phrases"
    """One or more unresolved anatomical phrases."""
    EVERYTHING = "everything"
    """All renderable entities."""
    EVERYTHING_EXCEPT = "everything_except"
    """All renderable entities except the listed phrases."""


@dataclass(frozen=True, slots=True)
class TargetSelector:
    """Unresolved description of which entities a command addresses."""

    kind: SelectorKind = SelectorKind.NONE
    phrases: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the selector addresses nothing."""
        return self.kind is SelectorKind.NONE and not self.phrases

    def describe(self) -> str:
        """One-line human-readable description."""
        if self.kind is SelectorKind.NONE:
            return "no target"
        if self.kind is SelectorKind.EVERYTHING:
            return "everything"
        if self.kind is SelectorKind.EVERYTHING_EXCEPT:
            return f"everything except {', '.join(self.phrases)}"
        return ", ".join(self.phrases)


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """A parsed, still scene-independent command."""

    intent: Intent
    raw_text: str
    selector: TargetSelector = field(default_factory=TargetSelector)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    parser_id: str = "unknown"
    notes: tuple[str, ...] = ()

    @property
    def is_understood(self) -> bool:
        """Whether the parser recognised an intent."""
        return self.intent is not Intent.UNKNOWN

    def describe(self) -> str:
        """One-line human-readable description."""
        params = ", ".join(f"{k}={v!r}" for k, v in sorted(self.parameters.items()))
        return f"{self.intent} [{self.selector.describe()}]" + (f" ({params})" if params else "")


class CommandParser(ABC):
    """Interface every Lagnav command parser must satisfy.

    A parser converts text into a :class:`ParsedCommand`. It must be free of
    scene state and free of side effects, so that parsers can be swapped,
    chained or compared without affecting the AWR.
    """

    parser_id: str = "abstract"

    @abstractmethod
    def parse(self, text: str) -> ParsedCommand:
        """Parse one utterance into a typed command."""

    def parse_all(self, texts: Sequence[str]) -> tuple[ParsedCommand, ...]:
        """Parse a sequence of utterances, preserving order."""
        return tuple(self.parse(text) for text in texts)


# --- rule table ---------------------------------------------------------------
# Ordered, fully anchored patterns. Order matters: more specific patterns come
# first so that "hide everything except the chambers" is not read as "hide X".

_ORGAN = r"(?:human\s+)?(?P<organ>heart|human heart)"
_LEVELS = r"(?P<level>primary|school|high[\s-]?school|university|medical)"
_FLOW = (
    r"(?:blood\s*flow|blood\s*circulation|circulation|flow|cardiac\s*cycle"
    r"|heart\s*beat|heartbeat)"
)


@dataclass(frozen=True, slots=True)
class _Rule:
    """One parsing rule: a regex plus how to build the command from its groups."""

    intent: Intent
    pattern: re.Pattern[str]
    selector_kind: SelectorKind = SelectorKind.NONE
    target_group: str | None = None
    parameter_groups: tuple[str, ...] = ()
    fixed_parameters: Mapping[str, Any] = field(default_factory=dict)
    note: str | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*{pattern}\s*$", re.IGNORECASE)


_RULES: tuple[_Rule, ...] = (
    # --- generation ----------------------------------------------------
    _Rule(
        Intent.GENERATE,
        _rx(rf"(?:generate|create|build|make|give me)\s+(?:a|an|the)?\s*{_ORGAN}"),
        SelectorKind.PHRASES,
        target_group="organ",
    ),
    # --- isolate (before plain show/hide) ------------------------------
    _Rule(
        Intent.ISOLATE,
        _rx(
            r"(?:hide|remove)\s+(?:everything|all|all structures)\s+(?:except|but|besides)"
            r"(?:\s+for)?\s+(?:the\s+)?(?P<target>.+)"
        ),
        SelectorKind.EVERYTHING_EXCEPT,
        target_group="target",
    ),
    _Rule(
        Intent.ISOLATE,
        _rx(r"(?:show|display)\s+(?:only|just)\s+(?:the\s+)?(?P<target>.+)"),
        SelectorKind.EVERYTHING_EXCEPT,
        target_group="target",
    ),
    _Rule(
        Intent.ISOLATE,
        _rx(r"(?:isolate|focus on)\s+(?:the\s+)?(?P<target>.+)"),
        SelectorKind.EVERYTHING_EXCEPT,
        target_group="target",
    ),
    # --- level of detail -----------------------------------------------
    _Rule(
        Intent.SET_LOD,
        _rx(r"(?:show|give me|add)\s+(?:me\s+)?more\s+detail(?:s)?"),
        fixed_parameters={"lod_delta": 1},
    ),
    _Rule(
        Intent.SET_LOD,
        _rx(r"(?:show|give me)\s+(?:me\s+)?less\s+detail(?:s)?"),
        fixed_parameters={"lod_delta": -1},
    ),
    _Rule(Intent.SET_LOD, _rx(r"increase\s+(?:the\s+)?detail"), fixed_parameters={"lod_delta": 1}),
    _Rule(
        Intent.SET_LOD,
        _rx(r"(?:decrease|reduce)\s+(?:the\s+)?detail"),
        fixed_parameters={"lod_delta": -1},
    ),
    _Rule(
        Intent.SET_LOD,
        _rx(r"(?:set|switch to|go to|use)\s+(?:lod|level of detail)\s*(?P<lod>\d+)"),
        parameter_groups=("lod",),
    ),
    # --- educational level ---------------------------------------------
    _Rule(
        Intent.SET_EDUCATIONAL_LEVEL,
        _rx(
            rf"(?:switch|change|set|go)\s*(?:to|into)?\s+(?:the\s+)?{_LEVELS}"
            r"(?:\s+(?:level|mode|view|detail))?"
        ),
        parameter_groups=("level",),
    ),
    _Rule(
        Intent.SET_EDUCATIONAL_LEVEL,
        _rx(rf"{_LEVELS}\s+(?:level|mode|view)"),
        parameter_groups=("level",),
    ),
    # --- opacity --------------------------------------------------------
    _Rule(
        Intent.SET_OPACITY,
        _rx(
            r"(?:make|render|set)\s+(?:the\s+)?(?P<target>.+?)\s+"
            r"(?P<preset>fully\s+transparent|semi[\s-]?transparent|transparent|opaque|solid|ghost(?:ed)?)"
        ),
        SelectorKind.PHRASES,
        target_group="target",
        parameter_groups=("preset",),
    ),
    _Rule(
        Intent.SET_OPACITY,
        _rx(
            r"set\s+(?:the\s+)?(?P<target>.+?)\s+opacity\s+to\s+(?P<opacity>\d+(?:\.\d+)?)\s*(?P<percent>%)?"
        ),
        SelectorKind.PHRASES,
        target_group="target",
        parameter_groups=("opacity", "percent"),
    ),
    # --- animation ------------------------------------------------------
    _Rule(
        Intent.ANIMATE,
        _rx(rf"(?:animate|start|play|show)\s+(?:the\s+)?{_FLOW}"),
        fixed_parameters={"clip": "blood_flow"},
    ),
    _Rule(
        Intent.STOP_ANIMATION,
        _rx(r"(?:stop|pause|end)\s+(?:the\s+)?(?:animation|flow|blood\s*flow)"),
    ),
    # --- explanation ----------------------------------------------------
    _Rule(
        Intent.EXPLAIN,
        _rx(r"explain\s+(?:what\s+is\s+happening|what's\s+happening|this|the\s+scene)"),
    ),
    _Rule(
        Intent.EXPLAIN,
        _rx(r"what\s+(?:is|'s)\s+happening(?:\s+here)?"),
    ),
    _Rule(
        Intent.EXPLAIN,
        _rx(r"(?:explain|describe|tell me about)\s+(?:the\s+)?(?P<target>.+)"),
        SelectorKind.PHRASES,
        target_group="target",
    ),
    # --- reset ----------------------------------------------------------
    _Rule(
        Intent.RESET_VIEW,
        _rx(r"(?:reset|restore)(?:\s+(?:the\s+)?(?:scene|view|visibility|defaults))?"),
    ),
    # --- show / hide (last, most general) -------------------------------
    _Rule(
        Intent.SHOW,
        _rx(r"(?:show|display|reveal)\s+(?:me\s+)?(?:everything|all)"),
        SelectorKind.EVERYTHING,
    ),
    _Rule(Intent.HIDE, _rx(r"(?:hide|conceal)\s+(?:everything|all)"), SelectorKind.EVERYTHING),
    _Rule(
        Intent.SHOW,
        _rx(r"(?:show|display|reveal|expose)\s+(?:me\s+)?(?:the\s+)?(?P<target>.+)"),
        SelectorKind.PHRASES,
        target_group="target",
    ),
    _Rule(
        Intent.HIDE,
        _rx(r"(?:hide|conceal|remove)\s+(?:the\s+)?(?P<target>.+)"),
        SelectorKind.PHRASES,
        target_group="target",
    ),
)

_OPACITY_PRESET_ALIASES: Mapping[str, str] = {
    "transparent": "transparent",
    "fully transparent": "ghost",
    "semi transparent": "semi_transparent",
    "semi-transparent": "semi_transparent",
    "semitransparent": "semi_transparent",
    "opaque": "opaque",
    "solid": "opaque",
    "ghost": "ghost",
    "ghosted": "ghost",
}

_CONJUNCTION = re.compile(r"\s*(?:,|\band\b|\bplus\b)\s*", re.IGNORECASE)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _split_targets(raw: str) -> tuple[str, ...]:
    """Split a target phrase on conjunctions, keeping anatomical words intact.

    Only leading articles are stripped. Anatomical words are never dropped: the
    resolver, not the parser, decides what a phrase means.
    """
    parts = [
        _LEADING_ARTICLE.sub("", part.strip()).strip()
        for part in _CONJUNCTION.split(raw)
        if part.strip()
    ]
    parts = [part for part in parts if part]
    return tuple(parts) or (raw.strip(),)


class RuleBasedCommandParser(CommandParser):
    """Deterministic pattern parser.

    Covers the controlled command vocabulary of the prototype. It is intentionally
    strict: an utterance it does not recognise becomes
    :attr:`Intent.UNKNOWN` with zero confidence rather than a guess, so the
    command engine can report the failure instead of mutating the scene.
    """

    parser_id = "rule-based-v0.1"

    def parse(self, text: str) -> ParsedCommand:
        """Parse one utterance."""
        cleaned = " ".join(text.strip().split()).rstrip(".!")
        if not cleaned:
            return ParsedCommand(
                intent=Intent.UNKNOWN,
                raw_text=text,
                confidence=0.0,
                parser_id=self.parser_id,
                notes=("Empty command.",),
            )
        for rule in _RULES:
            match = rule.pattern.match(cleaned)
            if match is None:
                continue
            groups = match.groupdict()
            parameters: dict[str, Any] = dict(rule.fixed_parameters)
            for name in rule.parameter_groups:
                value = groups.get(name)
                if value is None:
                    continue
                parameters.update(self._coerce_parameter(name, value))
            selector = self._build_selector(rule, groups)
            return ParsedCommand(
                intent=rule.intent,
                raw_text=text,
                selector=selector,
                parameters=parameters,
                confidence=1.0,
                parser_id=self.parser_id,
                notes=(rule.note,) if rule.note else (),
            )
        return ParsedCommand(
            intent=Intent.UNKNOWN,
            raw_text=text,
            confidence=0.0,
            parser_id=self.parser_id,
            notes=(
                "No rule matched. The deterministic parser supports a controlled "
                "vocabulary; see reasoning/parser.py for the supported forms.",
            ),
        )

    @staticmethod
    def _coerce_parameter(name: str, value: str) -> Mapping[str, Any]:
        """Convert a captured group into a typed parameter."""
        if name == "lod":
            return {"lod": int(value)}
        if name == "opacity":
            return {"opacity_raw": float(value)}
        if name == "percent":
            return {"opacity_is_percent": True}
        if name == "preset":
            key = " ".join(value.lower().split())
            return {"opacity_preset": _OPACITY_PRESET_ALIASES.get(key, key)}
        if name == "level":
            return {"educational_level": value.lower().replace(" ", "_").replace("-", "_")}
        return {name: value}

    @staticmethod
    def _build_selector(rule: _Rule, groups: Mapping[str, str | None]) -> TargetSelector:
        """Build the (still unresolved) target selector for a matched rule."""
        if rule.selector_kind is SelectorKind.NONE:
            return TargetSelector()
        if rule.selector_kind is SelectorKind.EVERYTHING:
            return TargetSelector(kind=SelectorKind.EVERYTHING)
        raw = groups.get(rule.target_group or "target")
        if not raw:
            return TargetSelector()
        return TargetSelector(kind=rule.selector_kind, phrases=_split_targets(raw))

    @staticmethod
    def supported_forms() -> tuple[str, ...]:
        """Example utterances the parser accepts, for documentation and tests."""
        return (
            "generate a human heart",
            "show the four chambers",
            "hide everything except the chambers",
            "show the valves",
            "hide the valves",
            "make the left ventricle transparent",
            "make the left ventricle opaque",
            "set the left ventricle opacity to 0.45",
            "show more detail",
            "show less detail",
            "set lod 3",
            "switch to medical level",
            "switch to school level",
            "show the left ventricle",
            "hide the left ventricle",
            "isolate the mitral valve",
            "show only the valves",
            "show everything",
            "reset the scene",
            "animate blood flow",
            "stop the animation",
            "explain what is happening",
            "explain the left ventricle",
        )


class LanguageModelCommandParser(CommandParser):
    """Language-model parser. Declared, not implemented.

    TODO(step-5 and later): a learned parser replaces only this class. It must
    return the same :class:`ParsedCommand` type, so the AWR, scene, editing and
    validation layers are unaffected. Open questions: which model, whether
    parsing and anatomical reasoning stay separate stages, how confidence is
    calibrated, and how ambiguity is escalated back to the user.
    """

    parser_id = "language-model-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "LanguageModelCommandParser",
            planned_in="a later R&D step, after Step 5 defines the language encoder",
        )

    def parse(self, text: str) -> ParsedCommand:  # pragma: no cover - construction raises
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("LanguageModelCommandParser.parse")
