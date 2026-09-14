"""Language to anatomy: a discrete control path and a continuous conditioning path.

STATUS: the transcoder and the program contract are implemented and tested; the
learned models are declared interfaces only.

The central commitment
----------------------

Language never produces geometry. It produces an **AWR program**: a short sequence
of typed operations whose entity arguments are drawn from the ontology's closed
codebook. The program is then validated and applied by the Step 4 engine, and only
the resulting AWR is decoded into geometry.

Two consequences follow, and they are the reason for the design:

* **Anatomical hallucination is structurally impossible.** A model that selects
  entity arguments from a 42-way codebook cannot invent a structure that does not
  exist. An appearance-driven system, which maps text straight to geometry, has no
  equivalent guarantee.
* **Every instruction is inspectable and reversible before it takes effect.** The
  program can be shown to a user, validated against the scene, rejected with a
  reason, or logged. Step 4's operations already are that validation layer.

Two paths, not one
------------------

============  ===================================  ================================
path          output                               used for
============  ===================================  ================================
control       AWR program (discrete, verifiable)   what changes: entities, state, level of detail
conditioning  pooled text embedding (continuous)   how it looks: style, emphasis, audience register
============  ===================================  ================================

The conditioning path may influence decoding. It may never introduce, remove or
rename an entity: that authority belongs to the control path alone.

Supervision comes from Step 4
-----------------------------

:class:`ProgramTranscoder` turns the deterministic Step 4 parser and resolver into
a program oracle. Templated utterances produce ground-truth programs at no data
cost, which is what makes the first training stage possible with no external
dataset and no licensing exposure. The rule parser is also a permanent validator:
where it is confident, a learned parser that disagrees with it is a candidate
regression.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from awr.config import load_domain_config
from awr.errors import ContractError, NotYetImplementedError
from awr.ontology import AnatomyOntology
from awr.schema import EducationalLevel
from generation.neural.contracts import (
    CONTRACTS,
    Contract,
    DType,
    PlainArray,
    TensorSpec,
    dim,
)
from generation.neural.features import EntityCodebook, Vocabulary
from reasoning.entity_resolver import EntityResolver
from reasoning.parser import (
    CommandParser,
    Intent,
    ParsedCommand,
    RuleBasedCommandParser,
    SelectorKind,
)

__all__ = [
    "ProgramOpType",
    "ProgramOp",
    "AWRProgram",
    "PROGRAM_CONTRACT",
    "TEXT_CONTRACT",
    "ProgramTranscoder",
    "LanguageEncoderSpec",
    "TextToProgramModel",
    "ConditioningLanguageEncoder",
]


class ProgramOpType(StrEnum):
    """Operation vocabulary of an AWR program.

    Closed by design and aligned one-to-one with Step 4 intents, so a learned
    decoder and the deterministic parser produce the same object.
    """

    NOOP = "noop"
    GENERATE = "generate"
    SHOW = "show"
    HIDE = "hide"
    ISOLATE = "isolate"
    SHOW_ALL = "show_all"
    SET_OPACITY = "set_opacity"
    SET_LOD = "set_lod"
    SET_EDUCATIONAL_LEVEL = "set_educational_level"
    RESET_VIEW = "reset_view"
    ANIMATE = "animate"
    STOP_ANIMATION = "stop_animation"
    EXPLAIN = "explain"

    @classmethod
    def from_intent(cls, intent: Intent, selector_kind: SelectorKind) -> ProgramOpType:
        """Map a Step 4 intent onto a program operation."""
        if intent is Intent.SHOW and selector_kind is SelectorKind.EVERYTHING:
            return cls.SHOW_ALL
        mapping = {
            Intent.GENERATE: cls.GENERATE,
            Intent.SHOW: cls.SHOW,
            Intent.HIDE: cls.HIDE,
            Intent.ISOLATE: cls.ISOLATE,
            Intent.SET_OPACITY: cls.SET_OPACITY,
            Intent.SET_LOD: cls.SET_LOD,
            Intent.SET_EDUCATIONAL_LEVEL: cls.SET_EDUCATIONAL_LEVEL,
            Intent.RESET_VIEW: cls.RESET_VIEW,
            Intent.ANIMATE: cls.ANIMATE,
            Intent.STOP_ANIMATION: cls.STOP_ANIMATION,
            Intent.EXPLAIN: cls.EXPLAIN,
        }
        if intent not in mapping:
            raise ContractError(
                f"Intent {intent!r} has no program operation. The operation vocabulary is "
                "closed; add a member deliberately rather than emitting a free-form action."
            )
        return mapping[intent]


SCALAR_SLOTS: tuple[str, ...] = ("opacity", "lod", "lod_delta", "educational_level")
"""Named scalar argument slots, in tensor order. Unused slots hold the padding value."""


@dataclass(frozen=True, slots=True)
class ProgramOp:
    """One typed operation of an AWR program."""

    op_type: ProgramOpType
    entity_args: tuple[str, ...] = ()
    scalars: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        limit = dim("A_ENT").size or 4
        if len(self.entity_args) > limit:
            raise ContractError(
                f"{self.op_type}: {len(self.entity_args)} entity arguments exceed the {limit} "
                "declared slots. Reference a group entity instead of listing its members."
            )
        unknown = sorted(set(self.scalars) - set(SCALAR_SLOTS))
        if unknown:
            raise ContractError(f"{self.op_type}: unknown scalar arguments {unknown}.")

    def describe(self) -> str:
        """One-line human-readable description."""
        args = ", ".join(self.entity_args)
        scalars = ", ".join(f"{k}={v:g}" for k, v in sorted(self.scalars.items()))
        parts = [part for part in (args, scalars) if part]
        return f"{self.op_type}({'; '.join(parts)})"


@dataclass(frozen=True, slots=True)
class AWRProgram:
    """A short, typed, verifiable program over the AWR."""

    ops: tuple[ProgramOp, ...]
    source_text: str = ""
    producer: str = "unknown"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        limit = dim("L_OP").size or 16
        if len(self.ops) > limit:
            raise ContractError(
                f"Program has {len(self.ops)} operations, more than the {limit} declared slots."
            )

    def __len__(self) -> int:
        return len(self.ops)

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Every entity referenced by the program, in order of first appearance."""
        seen: set[str] = set()
        out: list[str] = []
        for op in self.ops:
            for entity_id in op.entity_args:
                if entity_id not in seen:
                    seen.add(entity_id)
                    out.append(entity_id)
        return tuple(out)

    def describe(self) -> str:
        """Multi-line human-readable description."""
        return "\n".join(f"  {i}: {op.describe()}" for i, op in enumerate(self.ops))


PROGRAM_CONTRACT = CONTRACTS.register(
    Contract(
        name="awr_program",
        purpose=(
            "The discrete control path's output: a batch of typed AWR programs. Entity "
            "arguments index the ontology codebook, so the action space is closed."
        ),
        specs=(
            TensorSpec(
                "program_mask",
                ("B", "L_OP"),
                DType.BOOL,
                "True for real operation slots.",
            ),
            TensorSpec(
                "program_ops",
                ("B", "L_OP"),
                DType.INT32,
                "Operation type id from the closed ProgramOpType vocabulary.",
                mask="program_mask",
                padding_value=0,
            ),
            TensorSpec(
                "program_entity_args",
                ("B", "L_OP", "A_ENT"),
                DType.INT32,
                "Entity codebook indices; -1 for an unused slot. A closed argument space is "
                "what makes anatomical hallucination impossible.",
                mask="program_mask",
                padding_value=-1,
            ),
            TensorSpec(
                "program_scalar_args",
                ("B", "L_OP", "A_SCALAR"),
                DType.FLOAT32,
                "Scalar arguments in the fixed slot order opacity, lod, lod_delta, "
                "educational_level; -1.0 marks an unused slot.",
                normalization="opacity in [0,1]; levels as raw integers cast to float",
                mask="program_mask",
                padding_value=-1.0,
            ),
        ),
    )
)
"""Contract for a batch of AWR programs."""


TEXT_CONTRACT = CONTRACTS.register(
    Contract(
        name="language_io",
        purpose="Inputs and outputs of the language stack, for both paths.",
        specs=(
            TensorSpec("text_mask", ("B", "T_TOK"), DType.BOOL, "True for real text tokens."),
            TensorSpec(
                "text_tokens",
                ("B", "T_TOK"),
                DType.INT32,
                "Tokenised utterance. The tokeniser is a Step 6 decision.",
                mask="text_mask",
                padding_value=0,
            ),
            TensorSpec(
                "text_hidden",
                ("B", "T_TOK", "D_TEXT"),
                DType.FLOAT32,
                "Per-token hidden states, used by the control path's pointer over entities.",
                normalization="layer-normalised",
                mask="text_mask",
            ),
            TensorSpec(
                "text_pooled",
                ("B", "D_TEXT"),
                DType.FLOAT32,
                "Pooled utterance embedding. The conditioning path's only output.",
                normalization="unit L2 norm, so it can be compared with entity prototypes",
            ),
        ),
    )
)
"""Contract for language inputs and outputs."""


class ProgramTranscoder:
    """Turns Step 4's deterministic parse into a typed AWR program.

    Implemented, because it needs no model: the parser gives the intent, the
    resolver gives persistent entity ids, and this class packages them. It serves
    three purposes at once - a supervision generator for training the learned
    parser, a validator for its predictions, and a worked example of the target
    output format.
    """

    producer = "step4-rule-transcoder-v0.1"

    def __init__(
        self,
        ontology: AnatomyOntology,
        *,
        resolver: EntityResolver | None = None,
        parser: CommandParser | None = None,
        opacity_presets: Mapping[str, float] | None = None,
    ) -> None:
        self.ontology = ontology
        self.codebook = EntityCodebook.from_ontology(ontology)
        self.resolver = resolver or EntityResolver(ontology)
        self.parser = parser or RuleBasedCommandParser()
        self.op_vocabulary = Vocabulary.from_enum("program_op", ProgramOpType)
        self.level_vocabulary = Vocabulary.from_enum("educational_level", EducationalLevel)
        if opacity_presets is None:
            opacity_presets = dict(load_domain_config().opacity_presets.values)
        self.opacity_presets: Mapping[str, float] = dict(opacity_presets)

    def transcode(self, text: str) -> AWRProgram:
        """Parse an utterance and express it as an AWR program."""
        command = self.parser.parse(text)
        return self.from_command(command)

    def from_command(self, command: ParsedCommand) -> AWRProgram:
        """Express an already-parsed command as an AWR program."""
        if not command.is_understood:
            raise ContractError(
                f"Cannot transcode {command.raw_text!r}: the parser recognised no intent. "
                "An unparsed utterance must not become a program."
            )
        op_type = ProgramOpType.from_intent(command.intent, command.selector.kind)
        entity_args: list[str] = []
        for phrase in command.selector.phrases:
            resolution = self.resolver.resolve(phrase)
            entity_args.append(resolution.entity_id)
        scalars: dict[str, float] = {}
        preset = command.parameters.get("opacity_preset")
        if preset is not None:
            name = str(preset)
            if name not in self.opacity_presets:
                raise ContractError(
                    f"Opacity preset {name!r} is not configured. A program carries resolved "
                    "numbers, never preset names, so the model never learns a magic string."
                )
            scalars["opacity"] = float(self.opacity_presets[name])
        if "opacity_raw" in command.parameters:
            raw = float(command.parameters["opacity_raw"])
            percent = bool(command.parameters.get("opacity_is_percent"))
            scalars["opacity"] = raw / 100.0 if percent else raw
        if "lod" in command.parameters:
            scalars["lod"] = float(command.parameters["lod"])
        if "lod_delta" in command.parameters:
            scalars["lod_delta"] = float(command.parameters["lod_delta"])
        if "educational_level" in command.parameters:
            scalars["educational_level"] = float(
                self.level_vocabulary.index(str(command.parameters["educational_level"]))
            )
        return AWRProgram(
            ops=(ProgramOp(op_type, tuple(entity_args), scalars),),
            source_text=command.raw_text,
            producer=self.producer,
            confidence=command.confidence,
        )

    def encode(self, programs: Sequence[AWRProgram]) -> dict[str, PlainArray]:
        """Encode programs into a contract-valid batch."""
        if not programs:
            raise ContractError("A program batch must contain at least one program.")
        slots = dim("L_OP").size or 16
        entity_slots = dim("A_ENT").size or 4
        scalar_slots = dim("A_SCALAR").size or 4
        batch = len(programs)

        mask: list[object] = []
        ops: list[object] = []
        entity_args: list[object] = []
        scalar_args: list[object] = []
        for program in programs:
            for position in range(slots):
                live = position < len(program.ops)
                mask.append(live)
                if not live:
                    ops.append(0)
                    entity_args.extend([-1] * entity_slots)
                    scalar_args.extend([-1.0] * scalar_slots)
                    continue
                op = program.ops[position]
                ops.append(self.op_vocabulary.index(str(op.op_type)))
                indices = [self.codebook.index(entity) for entity in op.entity_args]
                indices.extend([-1] * (entity_slots - len(indices)))
                entity_args.extend(indices)
                scalar_args.extend(
                    float(op.scalars.get(name, -1.0)) for name in SCALAR_SLOTS[:scalar_slots]
                )

        payload = {
            "program_mask": PlainArray((batch, slots), DType.BOOL, tuple(mask)),
            "program_ops": PlainArray((batch, slots), DType.INT32, tuple(ops)),
            "program_entity_args": PlainArray(
                (batch, slots, entity_slots), DType.INT32, tuple(entity_args)
            ),
            "program_scalar_args": PlainArray(
                (batch, slots, scalar_slots), DType.FLOAT32, tuple(scalar_args)
            ),
        }
        PROGRAM_CONTRACT.validate_payload(payload, bindings={"B": batch})
        return payload

    def decode(self, payload: Mapping[str, PlainArray], batch_index: int = 0) -> AWRProgram:
        """Rebuild a program from its tensor form, for round-trip checking."""
        slots = payload["program_ops"].shape[1]
        entity_slots = payload["program_entity_args"].shape[2]
        scalar_slots = payload["program_scalar_args"].shape[2]
        entity_args = payload["program_entity_args"]
        scalar_args = payload["program_scalar_args"]
        ops: list[ProgramOp] = []
        for position in range(slots):
            if not payload["program_mask"].at(batch_index, position):
                continue
            op_type = ProgramOpType(
                self.op_vocabulary.symbol(int(payload["program_ops"].at(batch_index, position)))
            )
            entities = tuple(
                self.codebook.entity(int(entity_args.at(batch_index, position, slot)))
                for slot in range(entity_slots)
                if int(entity_args.at(batch_index, position, slot)) >= 0
            )
            scalars = {
                name: float(scalar_args.at(batch_index, position, slot))
                for slot, name in enumerate(SCALAR_SLOTS[:scalar_slots])
                if float(scalar_args.at(batch_index, position, slot)) >= 0.0
            }
            ops.append(ProgramOp(op_type, entities, scalars))
        return AWRProgram(tuple(ops), producer=f"{self.producer}:decoded")


@dataclass(frozen=True, slots=True)
class LanguageEncoderSpec:
    """Configuration of the proposed language stack."""

    hidden_width: int = 384
    layers: int = 6
    heads: int = 6
    max_tokens: int = 128
    vocabulary_size: int = 32_000
    pretrained_initialisation: str | None = None
    frozen: bool = False
    pointer_over_codebook: bool = True
    status: str = "PROPOSED"

    def __post_init__(self) -> None:
        if self.hidden_width % self.heads != 0:
            raise ContractError(
                f"Hidden width {self.hidden_width} must divide among {self.heads} heads."
            )


class TextToProgramModel:
    """The learned control path. Declared, not implemented.

    Contract for any implementation: emit a :class:`AWRProgram` whose entity
    arguments are codebook indices, produced by a pointer over entity embeddings
    rather than by free generation. Constrained decoding is not an optimisation
    here, it is the safety property.
    """

    model_id = "lagnav-text-to-program-undefined"

    def __init__(self, spec: LanguageEncoderSpec | None = None) -> None:
        self.spec = spec or LanguageEncoderSpec()
        raise NotYetImplementedError(
            "TextToProgramModel", planned_in="Step 6, curriculum stage S1"
        )

    def predict(self, text: str) -> AWRProgram:  # pragma: no cover - construction raises
        """Predict a program for an utterance."""
        raise NotYetImplementedError("TextToProgramModel.predict", planned_in="Step 6")


class ConditioningLanguageEncoder:
    """The learned conditioning path. Declared, not implemented.

    Produces ``text_pooled`` only. It has no authority over which entities exist,
    which is enforced by the fact that it has no path to the AWR at all.
    """

    model_id = "lagnav-conditioning-encoder-undefined"

    def __init__(self, spec: LanguageEncoderSpec | None = None) -> None:
        self.spec = spec or LanguageEncoderSpec()
        raise NotYetImplementedError(
            "ConditioningLanguageEncoder", planned_in="Step 6, curriculum stage S1"
        )

    def encode(self, text: str) -> PlainArray:  # pragma: no cover - construction raises
        """Encode an utterance into a pooled embedding."""
        raise NotYetImplementedError("ConditioningLanguageEncoder.encode", planned_in="Step 6")
