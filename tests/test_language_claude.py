"""Level 1: the Claude translator -- what would be sent, and how each answer is handled.

No request leaves the machine: a fake client records the request and returns a canned
response. Skipped without the ``language`` extra; the program and routing tests in
``test_language.py`` do not need it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from awr.ontology import load_ontology
from language.program import OPERATIONS, ProgramError, Step, program_schema

claude = pytest.importorskip("language.claude")


@pytest.fixture(scope="module")
def thorax():
    """The ontology programs are written against."""
    return load_ontology(Path("ontology/thorax"), expected_version="0.1.0")


class _FakeMessages:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _client(reply: Any) -> Any:
    messages = _FakeMessages(reply)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _response(program: dict[str, Any] | None, stop_reason: str = "end_turn") -> Any:
    content = [] if program is None else [SimpleNamespace(type="text", text=json.dumps(program))]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class TestTheClaudeTranslator:
    """What would be sent, and how each kind of answer is handled -- with no network."""

    def test_the_request(self, thorax) -> None:
        """Opus 5.5, low effort, structured output, default fallbacks, cached codebook."""
        client, messages = _client(
            _response({"steps": [{"op": "reset", "targets": [], "value": ""}], "refusal": ""})
        )
        claude.ClaudeProgramTranslator(thorax, client=client).translate("start over")
        request = messages.requests[0]
        assert request["model"] == "claude-opus-5-5"
        assert request["betas"] == ["server-side-fallback-2026-07-01"]
        assert request["fallbacks"] == "default"
        assert request["output_config"]["effort"] == "low"
        assert request["output_config"]["format"] == {
            "type": "json_schema",
            "schema": program_schema(thorax),
        }
        system = request["system"][0]
        assert system["cache_control"] == {"type": "ephemeral"}
        assert "respiratory.lung_middle_lobe_right | Right middle lobe" in system["text"]
        assert request["messages"] == [{"role": "user", "content": "start over"}]
        assert "thinking" not in request

    def test_every_operation_is_described_to_the_model(self, thorax) -> None:
        """An operation the prompt never explains is one the model will misuse."""
        prompt = claude.system_prompt(thorax)
        for op in OPERATIONS:
            assert op in prompt

    def test_the_prompt_is_byte_stable(self, thorax) -> None:
        """Deterministic codebook order, so the cached prefix is reused."""
        assert claude.system_prompt(thorax) == claude.system_prompt(thorax)

    def test_a_good_answer_becomes_a_validated_program(self, thorax) -> None:
        """The JSON in the text block, validated."""
        client, _ = _client(
            _response(
                {
                    "steps": [{"op": "hide", "targets": ["respiratory.lungs"], "value": ""}],
                    "refusal": "",
                }
            )
        )
        program = claude.ClaudeProgramTranslator(thorax, client=client).translate("lungs away")
        assert program.steps == (Step("hide", ("respiratory.lungs",), ""),)

    def test_a_safety_decline_is_a_refusal_not_an_error(self, thorax) -> None:
        """stop_reason 'refusal' survives the server-side fallback: reported, nothing applied."""
        client, _ = _client(_response(None, stop_reason="refusal"))
        program = claude.ClaudeProgramTranslator(thorax, client=client).translate("x")
        assert program.steps == () and "declined" in program.refusal

    @pytest.mark.parametrize(
        ("reply", "reason"),
        [
            (_response(None, stop_reason="max_tokens"), "cut off"),
            (_response(None), "no program"),
            (
                SimpleNamespace(
                    stop_reason="end_turn", content=[SimpleNamespace(type="text", text="not json")]
                ),
                "not valid JSON",
            ),
            (
                _response(
                    {
                        "steps": [{"op": "hide", "targets": ["heart.spleen"], "value": ""}],
                        "refusal": "",
                    }
                ),
                "no such structure",
            ),
        ],
        ids=["truncated", "empty", "not-json", "invented-anatomy"],
    )
    def test_bad_answers_raise_program_errors(self, thorax, reply: Any, reason: str) -> None:
        """Each is a reason the person can read; none reaches the engine."""
        client, _ = _client(reply)
        with pytest.raises(ProgramError, match=reason):
            claude.ClaudeProgramTranslator(thorax, client=client).translate("x")

    def test_api_failures_become_program_errors(self, thorax) -> None:
        """A connection failure is reported, not raised through the server."""
        import anthropic
        import httpx2

        failure = anthropic.APIConnectionError(
            request=httpx2.Request("POST", "https://example.invalid")
        )
        client, _ = _client(failure)
        with pytest.raises(ProgramError, match="could not be reached"):
            claude.ClaudeProgramTranslator(thorax, client=client).translate("x")
