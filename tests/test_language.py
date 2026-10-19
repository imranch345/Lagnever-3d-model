"""Level 1: the language layer -- programs, their schema and validation, and routing.

No request leaves the machine: the Claude translator is exercised with a fake client that
records the request and returns a canned response, so these tests pin what would be sent and
how every kind of answer is handled.
"""

from __future__ import annotations

import json
import os
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
        # resize is the one operation without an engine command (ADR-L2-001).
        assert {s.op for s in program.steps} == set(OPERATIONS) - {"resize"}
        for command in to_commands(program):
            assert engine.execute(command).ok, command
        assert to_commands(Program(steps=(Step("resize", ("heart.left_ventricle",), "+10"),))) == []

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

    def test_a_name_the_rules_cannot_resolve_goes_to_the_translator(self, engine) -> None:
        """'Show me the chamber that pumps blood to the body': form understood, name not."""
        translator = _ScriptedTranslator(Program(steps=(Step("show", ("heart.left_ventricle",)),)))
        result = LanguageLayer(engine, translator).execute(
            "show me the chamber that pumps blood to the whole body"
        )
        assert result.ok and result.source == "claude"
        assert translator.calls == ["show me the chamber that pumps blood to the whole body"]

    def test_anatomy_not_in_the_scene_gets_the_models_refusal(self, engine) -> None:
        """'Show me the spleen': the language model explains instead of a bare 'no match'."""
        translator = _ScriptedTranslator(Program(refusal="There is no spleen in this scene."))
        result = LanguageLayer(engine, translator).execute("show me the spleen")
        assert not result.ok and result.source == "claude"
        assert result.message == "There is no spleen in this scene."

    def test_without_a_translator_an_unresolved_name_stays_with_the_rules(self, engine) -> None:
        """No model configured: the rules' own answer, and how to enable the model."""
        result = LanguageLayer(engine).execute("show me the spleen")
        assert not result.ok and result.source == "rules" and ENABLE_VARIABLE in result.message

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


class TestResize:
    """'Make the left ventricle 20% larger': read by a rule, applied by a hook, kept in the AWR."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("make the left ventricle 20% larger", ("left ventricle", 20.0)),
            ("Make the LV 10 percent smaller.", ("LV", -10.0)),
            ("shrink the right atrium by 15%", ("right atrium", -15.0)),
            ("enlarge the left lung 5%", ("left lung", 5.0)),
            ("resize the myocardium by -10%", ("myocardium", -10.0)),
            ("make the left ventricle transparent", None),
            ("set the left ventricle opacity to 30%", None),
        ],
    )
    def test_the_rule_reads_size_requests(self, text: str, expected: object) -> None:
        """Direction from the words or the sign; anything else is not a resize."""
        from language.layer import parse_resize

        assert parse_resize(text) == expected

    def test_a_resize_step_is_validated(self, thorax) -> None:
        """One structure, a non-zero change within 30%, normalised with its sign."""

        def step(targets: list[str], value: str) -> dict[str, Any]:
            return {"steps": [{"op": "resize", "targets": targets, "value": value}]}

        lv, rv = "heart.left_ventricle", "heart.right_ventricle"
        assert validate_program(step([lv], "20"), thorax).steps[0].value == "+20"
        for bad in (step([lv, rv], "10"), step([lv], "40"), step([lv], "0"), step([], "10")):
            with pytest.raises(ProgramError):
                validate_program(bad, thorax)

    def test_the_awr_records_size_only_when_the_geometry_changed(self, engine) -> None:
        """Cumulative scale on success; nothing recorded on a refusal or without a model."""
        calls: list[tuple[str, float]] = []
        answers = iter([(True, "grown"), (True, "grown again"), (False, "would overlap")])

        def hook(entity: str, factor: float) -> tuple[bool, str]:
            calls.append((entity, factor))
            return next(answers)

        layer = LanguageLayer(engine, resize=hook)

        def scale() -> tuple[float, ...]:
            return engine.scene.get("heart.left_ventricle").state.transform.scale

        first = layer.execute("make the left ventricle 20% larger")
        assert first.ok and first.source == "rules" and first.message == "grown"
        assert first.commands == ["resize heart.left_ventricle +20%"]
        assert calls == [("heart.left_ventricle", 1.2)]
        layer.execute("make the LV 10% larger")
        assert scale() == pytest.approx((1.32, 1.32, 1.32))
        refused = layer.execute("enlarge the left ventricle by 10%")
        assert not refused.ok and refused.message == "would overlap"
        assert scale() == pytest.approx((1.32, 1.32, 1.32))
        no_model = LanguageLayer(engine).execute("make the left ventricle 5% smaller")
        assert not no_model.ok and "--model" in no_model.message

    def test_groups_unknown_names_and_big_changes_are_refused(self, engine) -> None:
        """One structure at a time, a real one, within the limit; the hook is never called."""
        calls: list[str] = []
        layer = LanguageLayer(engine, resize=lambda e, f: (calls.append(e) or True, "x"))
        assert "group" in layer.execute("make the lungs 10% larger").message
        assert not layer.execute("make the spleen 10% larger").ok
        assert "30%" in layer.execute("make the left ventricle 50% larger").message
        assert calls == []

    def test_a_program_mixes_resize_with_engine_steps(self, engine) -> None:
        """Claude's program: the resize goes to the hook, the rest to the engine, in order."""
        translator = _ScriptedTranslator(
            Program(
                steps=(
                    Step("resize", ("heart.left_ventricle",), "+20"),
                    Step("hide", ("respiratory.lungs",)),
                )
            )
        )
        layer = LanguageLayer(engine, translator, resize=lambda e, f: (True, f"{e} x{f:g}"))
        result = layer.execute("give me a bigger left ventricle and get the lungs out of the way")
        assert result.ok and result.source == "claude"
        assert result.commands == ["resize heart.left_ventricle +20%", "hide respiratory.lungs"]
        assert "heart.left_ventricle x1.2" in result.message
        assert not _visible(engine) & {"respiratory.left_lung", "respiratory.right_lung"}


class TestTheUserKeyFile:
    """The API key comes from a private file outside the repository, never from code or chat."""

    def _file(self, tmp_path: Path, text: str, mode: int = 0o600) -> Path:
        path = tmp_path / "anthropic.env"
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)
        return path

    def test_allowed_names_are_loaded_and_values_never_returned(
        self, tmp_path, monkeypatch
    ) -> None:
        """Only the key and the switch are set; the return lists names only."""
        from language.layer import load_user_env

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv(ENABLE_VARIABLE, raising=False)
        monkeypatch.delenv("PATH_HIJACK", raising=False)
        path = self._file(
            tmp_path,
            'export ANTHROPIC_API_KEY="test-not-a-real-key"\n'
            "LAGNAV_LANGUAGE=claude\nPATH_HIJACK=x\n",
        )
        loaded = load_user_env(path)
        assert sorted(loaded) == ["ANTHROPIC_API_KEY", ENABLE_VARIABLE]
        assert os.environ["ANTHROPIC_API_KEY"] == "test-not-a-real-key"
        assert "PATH_HIJACK" not in os.environ
        assert all("test-not" not in name for name in loaded)

    def test_the_environment_wins_over_the_file(self, tmp_path, monkeypatch) -> None:
        """A shell can always override the file."""
        from language.layer import load_user_env

        monkeypatch.setenv(ENABLE_VARIABLE, "rules")
        path = self._file(tmp_path, "LAGNAV_LANGUAGE=claude\n")
        assert load_user_env(path) == [] and os.environ[ENABLE_VARIABLE] == "rules"

    def test_a_file_others_can_read_is_refused(self, tmp_path) -> None:
        """A key in a group- or world-readable file is not used."""
        from language.layer import load_user_env

        path = self._file(tmp_path, "ANTHROPIC_API_KEY=x\n", mode=0o644)
        with pytest.raises(PermissionError, match="chmod 600"):
            load_user_env(path)

    def test_no_file_is_fine(self, tmp_path) -> None:
        """Without the file nothing changes, and the rules path stays the default."""
        from language.layer import load_user_env

        assert load_user_env(tmp_path / "missing.env") == []
