"""Deterministic parser: coverage, typed parameters and refusal to guess."""

from __future__ import annotations

import pytest

from awr.errors import NotYetImplementedError
from reasoning.parser import (
    Intent,
    LanguageModelCommandParser,
    RuleBasedCommandParser,
    SelectorKind,
)


@pytest.fixture
def parser() -> RuleBasedCommandParser:
    """The rule-based parser."""
    return RuleBasedCommandParser()


@pytest.mark.parametrize("text", RuleBasedCommandParser.supported_forms())
def test_every_documented_form_parses(parser: RuleBasedCommandParser, text: str) -> None:
    """Each documented example maps to a known intent."""
    command = parser.parse(text)
    assert command.is_understood, text
    assert command.confidence == 1.0


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("generate a human heart", Intent.GENERATE),
        ("Show the four chambers", Intent.SHOW),
        ("hide the valves", Intent.HIDE),
        ("hide everything except the chambers", Intent.ISOLATE),
        ("show only the valves", Intent.ISOLATE),
        ("make the left ventricle transparent", Intent.SET_OPACITY),
        ("show more detail", Intent.SET_LOD),
        ("switch to medical level", Intent.SET_EDUCATIONAL_LEVEL),
        ("animate blood flow", Intent.ANIMATE),
        ("explain what is happening", Intent.EXPLAIN),
        ("reset the scene", Intent.RESET_VIEW),
    ],
)
def test_intents(parser: RuleBasedCommandParser, text: str, intent: Intent) -> None:
    """Utterances map to the expected intent."""
    assert parser.parse(text).intent is intent


def test_case_and_punctuation_are_ignored(parser: RuleBasedCommandParser) -> None:
    """Capitalisation and trailing punctuation do not change the parse."""
    first = parser.parse("Make the Left Ventricle Transparent.")
    second = parser.parse("make the left ventricle transparent")
    assert first.intent is second.intent
    assert first.parameters == second.parameters


def test_parameters_are_typed(parser: RuleBasedCommandParser) -> None:
    """Captured values arrive as typed parameters, not raw strings."""
    assert parser.parse("set lod 3").parameters["lod"] == 3
    assert parser.parse("show more detail").parameters["lod_delta"] == 1
    assert parser.parse("show less detail").parameters["lod_delta"] == -1
    assert parser.parse("make the aorta transparent").parameters["opacity_preset"] == "transparent"
    assert parser.parse("set the aorta opacity to 0.4").parameters["opacity_raw"] == 0.4
    assert parser.parse("switch to high school level").parameters["educational_level"] == (
        "high_school"
    )


def test_selectors_are_unresolved_text(parser: RuleBasedCommandParser) -> None:
    """The parser does not resolve entities; that is the resolver's job."""
    command = parser.parse("show the four chambers")
    assert command.selector.kind is SelectorKind.PHRASES
    assert command.selector.phrases == ("four chambers",)

    everything = parser.parse("show everything")
    assert everything.selector.kind is SelectorKind.EVERYTHING

    isolate = parser.parse("hide everything except the chambers")
    assert isolate.selector.kind is SelectorKind.EVERYTHING_EXCEPT
    assert isolate.selector.phrases == ("chambers",)


def test_conjunctions_split_into_several_targets(parser: RuleBasedCommandParser) -> None:
    """Multiple targets in one command are split, not concatenated."""
    command = parser.parse("show the aorta and the pulmonary trunk")
    assert command.selector.phrases == ("aorta", "pulmonary trunk")


def test_unknown_text_is_not_guessed(parser: RuleBasedCommandParser) -> None:
    """An unrecognised sentence has zero confidence and no target."""
    command = parser.parse("please rotate the camera around the model")
    assert command.intent is Intent.UNKNOWN
    assert command.confidence == 0.0
    assert command.selector.is_empty


def test_empty_input_is_handled(parser: RuleBasedCommandParser) -> None:
    """Empty input is refused cleanly."""
    assert parser.parse("   ").intent is Intent.UNKNOWN


def test_parser_is_stateless(parser: RuleBasedCommandParser) -> None:
    """Parsing the same text twice gives the same result."""
    first = parser.parse("hide everything except the chambers")
    second = parser.parse("hide everything except the chambers")
    assert first == second


def test_language_model_parser_is_declared_only() -> None:
    """The future parser holds the interface without faking behaviour."""
    with pytest.raises(NotYetImplementedError):
        LanguageModelCommandParser()
