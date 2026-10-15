"""Level 1: the language layer -- programs, their schema and validation, and routing.

No request leaves the machine: the Claude translator is exercised with a fake client that
records the request and returns a canned response, so these tests pin what would be sent and
how every kind of answer is handled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from awr.config import load_domain_config
from awr.ontology import load_ontology
from language.layer import ENABLE_VARIABLE, LanguageLayer, translator_from_environment
from language.program import (
    OPERATIONS,
    Program,
    ProgramError,
    Step,
    program_schema,
    to_commands,
    validate_program,
)
from reasoning.command_engine import CommandEngine

THORAX = Path("ontology/thorax")


@pytest.fixture(scope="module")
def thorax():
    """The ontology programs are written against."""
    return load_ontology(THORAX, expected_version="0.1.0")


@pytest.fixture()
def engine() -> CommandEngine:
    """A thorax scene at LOD 1."""
    engine = CommandEngine(config=load_domain_config(domain="thorax"))
    engine.execute("generate the heart and lungs")
    return engine


def _visible(engine: CommandEngine) -> set[str]:
    return {i for i in engine.scene.ids() if engine.scene.get(i).state.visibility}


class TestTheSchema:
    """What a model is allowed to answer."""

    def test_operations_and_ids_are_closed(self, thorax) -> None:
        """Both enums; nothing else can be expressed."""
        schema = program_schema(thorax)
        step = schema["properties"]["steps"]["items"]
        assert step["properties"]["op"]["enum"] == list(OPERATIONS)
        assert step["properties"]["targets"]["items"]["enum"] == sorted(thorax.ids())
        assert schema["additionalProperties"] is False and step["additionalProperties"] is False

    def test_it_is_byte_stable(self, thorax) -> None:
        """Same ontology, same schema text: a precondition for prompt caching."""
        assert json.dumps(program_schema(thorax)) == json.dumps(program_schema(thorax))


class TestValidation:
    """The guarantee, independent of what the schema asked for."""

    def test_a_good_program_validates(self, thorax) -> None:
        """Values are normalised: opacity as a number, LOD as an integer."""
        program = validate_program(
            {
                "steps": [
                    {"op": "hide", "targets": ["respiratory.lungs"], "value": ""},
                    {"op": "set_opacity", "targets": ["heart.left_ventricle"], "value": "0.30"},
                    {"op": "set_lod", "targets": [], "value": "2.0"},
                ],
                "refusal": "",
            },
            thorax,
        )
        assert [s.value for s in program.steps] == ["", "0.3", "2"]

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            (
                {"steps": [{"op": "draw", "targets": [], "value": ""}], "refusal": ""},
                "unknown operation",
            ),
            (
                {
                    "steps": [{"op": "hide", "targets": ["heart.spleen"], "value": ""}],
                    "refusal": "",
                },
                "no such structure",
            ),
            (
                {"steps": [{"op": "hide", "targets": [], "value": ""}], "refusal": ""},
                "needs at least one",
            ),
            (
                {
                    "steps": [{"op": "set_opacity", "targets": ["heart"], "value": "1.5"}],
                    "refusal": "",
                },
                "outside 0-1",
            ),
            (
                {
                    "steps": [{"op": "set_opacity", "targets": ["heart"], "value": "clear"}],
                    "refusal": "",
                },
                "needs a number",
            ),
            (
                {"steps": [{"op": "set_lod", "targets": [], "value": "9"}], "refusal": ""},
                "level of detail",
            ),
            (
                {"steps": [{"op": "set_level", "targets": [], "value": "expert"}], "refusal": ""},
                "level must be",
            ),
            ({"steps": [], "refusal": ""}, "no steps and no refusal"),
            (
                {"steps": [{"op": "reset", "targets": [], "value": ""}], "refusal": "no"},
                "both refuse and act",
            ),
            (
                {"steps": [{"op": "reset", "targets": [], "value": ""}] * 13, "refusal": ""},
                "at most",
            ),
            ([], "JSON object"),
        ],
        ids=[
            "unknown-op",
            "invented-anatomy",
            "no-targets",
            "opacity-range",
            "opacity-text",
            "lod-range",
            "unknown-level",
            "empty",
            "refuse-and-act",
            "too-long",
            "not-object",
        ],
    )
    def test_bad_programs_are_refused_with_a_reason(self, thorax, raw: Any, reason: str) -> None:
        """Invented anatomy above all: ADR 0007's safety property."""
        with pytest.raises(ProgramError, match=reason):
            validate_program(raw, thorax)

    def test_a_refusal_is_a_program_with_a_reason(self, thorax) -> None:
        """No steps, a reason, and nothing to apply."""
        program = validate_program({"steps": [], "refusal": "There is no spleen here."}, thorax)
        assert program.steps == () and program.refusal == "There is no spleen here."


class TestCommands:
    """Programs become canonical commands the deterministic engine runs."""

    def test_every_operation_runs_in_the_engine(self, engine) -> None:
        """One step of each operation, applied through the engine, all accepted."""
        program = Program(
            steps=(
                Step("hide", ("respiratory.lungs",)),
                Step("show", ("respiratory.lung_middle_lobe_right",)),
                Step("isolate", ("heart.left_ventricle", "heart.right_ventricle")),
                Step("set_opacity", ("heart.left_ventricle", "heart.right_ventricle"), "0.3"),
                Step("set_lod", (), "2"),
                Step("set_level", (), "high_school"),
                Step("animate_flow"),
                Step("stop_animation"),
                Step("explain", ("heart.aorta",)),
                Step("explain"),
                Step("reset"),
            )
        )
        assert {s.op for s in program.steps} == set(OPERATIONS)
        for command in to_commands(program):
            assert engine.execute(command).ok, command

    def test_isolate_means_only_those(self, engine) -> None:
        """'isolate' becomes 'show only', and leaves exactly its targets visible."""
        for command in to_commands(Program(steps=(Step("isolate", ("respiratory.trachea",)),))):
            engine.execute(command)
        assert _visible(engine) == {"respiratory.trachea"}


class _ScriptedTranslator:
    def __init__(self, result: Program | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    def translate(self, text: str) -> Program:
        self.calls.append(text)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class TestRouting:
    """The rule parser first; the translator only when it is needed and present."""

    def test_understood_text_never_reaches_the_translator(self, engine) -> None:
        """Free and instant for everything the parser already knows."""
        translator = _ScriptedTranslator(Program(steps=(Step("reset"),)))
        result = LanguageLayer(engine, translator).execute("hide the trachea")
        assert result.ok and result.source == "rules" and translator.calls == []

    def test_free_text_goes_through_a_shown_program(self, engine) -> None:
        """The program and its commands are part of the result, and are applied."""
        translator = _ScriptedTranslator(
            Program(
                steps=(
                    Step("hide", ("respiratory.lungs",)),
                    Step("set_opacity", ("heart.right_ventricle",), "0.3"),
                )
            )
        )
        result = LanguageLayer(engine, translator).execute(
            "get the lungs out of the way and let me see through the right ventricle"
        )
        assert result.ok and result.source == "claude"
        assert result.commands == [
            "hide respiratory.lungs",
            "set heart.right_ventricle opacity to 0.3",
        ]
        assert result.program["steps"][0]["op"] == "hide"
        assert not any(i.startswith("respiratory.lung_") for i in _visible(engine))
        assert engine.scene.get("heart.right_ventricle").state.opacity == pytest.approx(0.3)

    def test_a_refusal_changes_nothing(self, engine) -> None:
        """The reason is returned and the scene is untouched."""
        before = engine.scene.version
        result = LanguageLayer(
            engine, _ScriptedTranslator(Program(refusal="There is no spleen in this scene."))
        ).execute("show me the spleen please")
        assert not result.ok and "spleen" in result.message
        assert engine.scene.version == before

    def test_a_translator_failure_is_reported(self, engine) -> None:
        """An unreachable service is an answer, not a crash."""
        result = LanguageLayer(
            engine, _ScriptedTranslator(ProgramError("The language service could not be reached."))
        ).execute("do the thing")
        assert not result.ok and "could not be reached" in result.message

    def test_without_a_translator_free_text_explains_how_to_enable_it(self, engine) -> None:
        """The parser's answer, plus the switch."""
        result = LanguageLayer(engine, None).execute("could you tidy the scene up a bit")
        assert not result.ok and ENABLE_VARIABLE in result.message

    def test_the_translator_is_off_unless_enabled(self, thorax, monkeypatch) -> None:
        """No variable, no translator: nothing is ever spent by default."""
        monkeypatch.delenv(ENABLE_VARIABLE, raising=False)
        assert translator_from_environment(thorax) is None
        monkeypatch.setenv(ENABLE_VARIABLE, "rules")
        assert translator_from_environment(thorax) is None
