"""AWR programs: a closed operation vocabulary over a closed entity codebook (ADR 0007).

A program is a short list of steps. Every step names an operation from ``OPERATIONS`` and, where
it needs them, entity ids from the ontology -- nothing else can be expressed, so a language
model asked for a program cannot invent anatomy. The JSON schema handed to the model enumerates
the ids, and ``validate_program`` checks the same rules again on what comes back, because a
schema is a request, and validation is the guarantee.

A valid program is executed as canonical commands through the existing ``CommandEngine``
(``to_commands``), so the deterministic engine stays the single writer of scene state and
validates every step a third time as it applies it.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.errors import CommandError
from awr.ontology import AnatomyOntology

__all__ = [
    "LEVELS",
    "MAX_RESIZE_PERCENT",
    "OPERATIONS",
    "Program",
    "ProgramError",
    "Step",
    "program_schema",
    "to_commands",
    "validate_program",
]

#: The whole vocabulary. A new capability is a deliberate change here, not a prompt.
OPERATIONS: tuple[str, ...] = (
    "show",
    "hide",
    "isolate",
    "set_opacity",
    "resize",
    "set_lod",
    "set_level",
    "animate_flow",
    "stop_animation",
    "explain",
    "reset",
)
NEEDS_TARGETS = frozenset({"show", "hide", "isolate", "set_opacity", "resize"})
#: Largest change of linear size one resize may ask for, in percent (ADR-L2-001).
MAX_RESIZE_PERCENT = 30.0
LEVELS: tuple[str, ...] = ("primary", "school", "high_school", "university", "medical")
MAX_STEPS = 12


class ProgramError(CommandError):
    """A program that is malformed, out of vocabulary, or names anatomy that does not exist."""


@dataclass(frozen=True, slots=True)
class Step:
    """One operation. ``value`` carries the opacity, LOD or level where the operation needs one."""

    op: str
    targets: tuple[str, ...] = ()
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for logs and the viewer."""
        return {"op": self.op, "targets": list(self.targets), "value": self.value}


@dataclass(frozen=True, slots=True)
class Program:
    """Steps to apply in order, or a refusal with the reason, never both."""

    steps: tuple[Step, ...] = ()
    refusal: str = ""
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for logs and the viewer."""
        return {"steps": [s.to_dict() for s in self.steps], "refusal": self.refusal}


def program_schema(ontology: AnatomyOntology) -> dict[str, Any]:
    """The JSON schema a model must answer in: operations and entity ids are enums.

    Every field is required and a string or array, the plainest shapes structured output
    supports; unused fields are empty. Ids are sorted so the schema, and any prompt built from
    it, is byte-stable across runs.
    """
    return {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": list(OPERATIONS)},
                        "targets": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(ontology.ids())},
                        },
                        "value": {"type": "string"},
                    },
                    "required": ["op", "targets", "value"],
                    "additionalProperties": False,
                },
            },
            "refusal": {"type": "string"},
        },
        "required": ["steps", "refusal"],
        "additionalProperties": False,
    }


def _number(value: str, op: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise ProgramError(f"{op} needs a number, got {value!r}.") from None


def validate_program(raw: Mapping[str, Any], ontology: AnatomyOntology) -> Program:
    """Check a program against the vocabulary and the ontology; return it typed.

    Raises:
        ProgramError: naming the first rule broken. Nothing of a rejected program is applied.

    """
    if not isinstance(raw, Mapping):
        raise ProgramError("A program must be a JSON object.")
    refusal = str(raw.get("refusal", "") or "").strip()
    raw_steps = raw.get("steps", [])
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes):
        raise ProgramError("'steps' must be a list.")
    if refusal and raw_steps:
        raise ProgramError("A program cannot both refuse and act.")
    if refusal:
        return Program(refusal=refusal)
    if not raw_steps:
        raise ProgramError("The program has no steps and no refusal.")
    if len(raw_steps) > MAX_STEPS:
        raise ProgramError(f"The program has {len(raw_steps)} steps; at most {MAX_STEPS}.")

    max_lod = int(ontology.max_lod())
    steps: list[Step] = []
    for index, item in enumerate(raw_steps):
        if not isinstance(item, Mapping):
            raise ProgramError(f"Step {index + 1} is not an object.")
        op = str(item.get("op", ""))
        if op not in OPERATIONS:
            raise ProgramError(f"Step {index + 1}: unknown operation {op!r}.")
        targets = tuple(str(t) for t in item.get("targets", []) or [])
        unknown = [t for t in targets if t not in ontology]
        if unknown:
            raise ProgramError(f"Step {index + 1}: no such structure {unknown}.")
        if op in NEEDS_TARGETS and not targets:
            raise ProgramError(f"Step {index + 1}: {op} needs at least one structure.")
        value = str(item.get("value", "") or "").strip()
        if op == "set_opacity":
            opacity = _number(value, op)
            if not 0.0 <= opacity <= 1.0:
                raise ProgramError(f"Step {index + 1}: opacity {opacity} is outside 0-1.")
            value = f"{opacity:g}"
        elif op == "set_lod":
            lod = _number(value, op)
            if lod != int(lod) or not 0 <= lod <= max_lod:
                raise ProgramError(f"Step {index + 1}: level of detail must be 0-{max_lod}.")
            value = str(int(lod))
        elif op == "resize":
            if len(targets) != 1:
                raise ProgramError(f"Step {index + 1}: resize changes one structure at a time.")
            percent = _number(value, op)
            if percent == 0 or abs(percent) > MAX_RESIZE_PERCENT:
                raise ProgramError(
                    f"Step {index + 1}: resize takes a change from -{MAX_RESIZE_PERCENT:g}% "
                    f"to +{MAX_RESIZE_PERCENT:g}%, not {percent:g}%."
                )
            value = f"{percent:+g}"
        elif op == "set_level" and value not in LEVELS:
            raise ProgramError(f"Step {index + 1}: level must be one of {list(LEVELS)}.")
        steps.append(Step(op=op, targets=targets, value=value))
    return Program(steps=tuple(steps))


def to_commands(program: Program) -> list[str]:
    """Canonical command text for each step, in the forms the rule parser accepts.

    Entity ids are used verbatim: the resolver accepts them as exact surface forms, so there
    is no second, lossy name lookup between the program and the engine.

    ``resize`` has no engine command: the language layer applies it through the shape model
    and records it in the AWR itself (ADR-L2-001).
    """
    commands: list[str] = []
    for step in program.steps:
        listed = ", ".join(step.targets)
        if step.op == "show":
            commands.append(f"show {listed}")
        elif step.op == "hide":
            commands.append(f"hide {listed}")
        elif step.op == "isolate":
            commands.append(f"show only {listed}")
        elif step.op == "set_opacity":
            commands.extend(f"set {t} opacity to {step.value}" for t in step.targets)
        elif step.op == "set_lod":
            commands.append(f"set lod {step.value}")
        elif step.op == "set_level":
            commands.append(f"switch to {step.value.replace('_', ' ')} level")
        elif step.op == "animate_flow":
            commands.append("animate blood flow")
        elif step.op == "stop_animation":
            commands.append("stop the animation")
        elif step.op == "explain":
            commands.extend([f"explain {t}" for t in step.targets] or ["explain what is happening"])
        elif step.op == "reset":
            commands.append("reset the scene")
    return commands
