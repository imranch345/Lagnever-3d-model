"""Command engine: natural language in, scene changes out.

The engine is the only component that knows about all the others, and it stays
thin on purpose. Its job is orchestration:

    text -> parser -> entity resolver -> editing operation -> scene -> response

Everything it touches is replaceable. Swapping the rule parser for a language
model changes one constructor argument. Nothing below the parser knows or cares
where the intent came from, which is the property that lets the AWR survive the
arrival of the neural stack.

User-input errors (an unknown structure, an ambiguous phrase, an out-of-range
opacity) produce a failed :class:`CommandResponse` carrying an error code and a
message that says what to do instead. They never half-apply a change: an
operation either validates and applies, or the scene is untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from animation.heart_flow import CardiacFlowModel
from awr.config import DomainConfig, load_domain_config
from awr.errors import (
    AmbiguousEntityReferenceError,
    CommandError,
    LagnavError,
    UnknownEntityReferenceError,
)
from awr.ontology import AnatomyOntology, load_ontology
from awr.scene import AWRScene, EntityDelta
from awr.schema import EducationalLevel, coerce_enum
from editing.operations import (
    ClearAnimation,
    HideEntities,
    IsolateEntities,
    ResetToLodDefaults,
    SetEducationalLevel,
    SetLod,
    SetOpacity,
    ShowEntities,
)
from editing.scene_editor import SceneEditor
from generation.generator import GenerationRequest, HeartSceneGenerator, SceneGenerator
from reasoning.entity_resolver import EntityResolver
from reasoning.explanation import SceneExplainer, TemplateSceneExplainer
from reasoning.parser import (
    CommandParser,
    Intent,
    ParsedCommand,
    RuleBasedCommandParser,
    SelectorKind,
)

__all__ = ["CommandResponse", "CommandEngine"]


@dataclass(frozen=True, slots=True)
class CommandResponse:
    """Structured result of executing one command."""

    ok: bool
    message: str
    command: ParsedCommand | None = None
    operation: str | None = None
    affected: tuple[str, ...] = ()
    deltas: tuple[EntityDelta, ...] = ()
    scene_version: int | None = None
    error_code: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def intent(self) -> Intent | None:
        """Intent the parser assigned, if the command was parsed."""
        return self.command.intent if self.command else None

    def describe(self) -> str:
        """One-line human-readable description."""
        marker = "ok" if self.ok else f"error[{self.error_code}]"
        return f"{marker}: {self.message}"


class CommandEngine:
    """Executes natural-language commands against a persistent AWR scene."""

    def __init__(
        self,
        *,
        config: DomainConfig | None = None,
        ontology: AnatomyOntology | None = None,
        scene: AWRScene | None = None,
        parser: CommandParser | None = None,
        generator: SceneGenerator | None = None,
        explainer: SceneExplainer | None = None,
    ) -> None:
        self.config = config or load_domain_config()
        self.ontology = ontology or load_ontology(
            self.config.domain.ontology_dir,
            expected_version=self.config.domain.ontology_version,
        )
        self.parser = parser or RuleBasedCommandParser()
        self.resolver = EntityResolver(self.ontology, self.config.resolver)
        self.generator = generator or HeartSceneGenerator(
            config=self.config, ontology=self.ontology
        )
        self.explainer = explainer or TemplateSceneExplainer()
        self._scene: AWRScene | None = scene
        self._editor: SceneEditor | None = SceneEditor(scene) if scene else None

    # ------------------------------------------------------------------
    @property
    def scene(self) -> AWRScene | None:
        """The scene under edit, or ``None`` before anything was generated."""
        return self._scene

    @property
    def editor(self) -> SceneEditor | None:
        """The editor bound to the current scene."""
        return self._editor

    def require_scene(self) -> AWRScene:
        """Return the current scene or raise a :class:`CommandError`."""
        if self._scene is None:
            raise CommandError(
                "There is no scene yet. Say 'generate a human heart' first."
            )
        return self._scene

    def attach_scene(self, scene: AWRScene) -> None:
        """Bind an existing scene, for example one loaded from disk."""
        self._scene = scene
        self._editor = SceneEditor(scene)

    # ------------------------------------------------------------------
    def execute(self, text: str, *, strict: bool = False) -> CommandResponse:
        """Parse and execute one command.

        Args:
            text: The utterance.
            strict: Re-raise errors instead of returning a failed response.

        """
        command = self.parser.parse(text)
        try:
            return self._dispatch(command)
        except LagnavError as exc:
            if strict:
                raise
            return CommandResponse(
                ok=False,
                message=str(exc),
                command=command,
                error_code=type(exc).__name__,
                scene_version=self._scene.version if self._scene else None,
            )

    def run(self, texts: Sequence[str], *, strict: bool = False) -> tuple[CommandResponse, ...]:
        """Execute a sequence of commands against the same persistent scene."""
        return tuple(self.execute(text, strict=strict) for text in texts)

    # ------------------------------------------------------------------
    def _dispatch(self, command: ParsedCommand) -> CommandResponse:
        handlers = {
            Intent.GENERATE: self._handle_generate,
            Intent.SHOW: self._handle_show,
            Intent.HIDE: self._handle_hide,
            Intent.ISOLATE: self._handle_isolate,
            Intent.SET_OPACITY: self._handle_set_opacity,
            Intent.SET_LOD: self._handle_set_lod,
            Intent.SET_EDUCATIONAL_LEVEL: self._handle_set_level,
            Intent.RESET_VIEW: self._handle_reset,
            Intent.ANIMATE: self._handle_animate,
            Intent.STOP_ANIMATION: self._handle_stop_animation,
            Intent.EXPLAIN: self._handle_explain,
        }
        handler = handlers.get(command.intent)
        if handler is None:
            examples = ", ".join(
                f'"{form}"' for form in RuleBasedCommandParser.supported_forms()[:5]
            )
            raise CommandError(
                f"I could not interpret {command.raw_text!r}. The deterministic parser "
                f"understands a controlled vocabulary, for example: {examples}."
            )
        return handler(command)

    # --- targets -------------------------------------------------------
    def _targets(self, command: ParsedCommand) -> tuple[str, ...]:
        """Resolve a command's selector into concrete entity ids."""
        scene = self.require_scene()
        selector = command.selector
        if selector.kind is SelectorKind.EVERYTHING:
            return scene.renderable_ids()
        if not selector.phrases:
            raise CommandError(
                f"{command.intent} needs a target, for example 'show the valves'."
            )
        try:
            return self.resolver.targets_for(selector.phrases)
        except (UnknownEntityReferenceError, AmbiguousEntityReferenceError):
            raise  # already carries suggestions or candidates

    def _respond(
        self,
        command: ParsedCommand,
        applied: Any,
        *,
        message: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> CommandResponse:
        scene = self.require_scene()
        result = applied.result
        return CommandResponse(
            ok=True,
            message=message or result.summary,
            command=command,
            operation=result.operation,
            affected=result.affected,
            deltas=result.deltas,
            scene_version=scene.version,
            data=dict(data or {}),
        )

    # --- handlers ------------------------------------------------------
    def _handle_generate(self, command: ParsedCommand) -> CommandResponse:
        result = self.generator.generate(GenerationRequest(text=command.raw_text))
        self.attach_scene(result.scene)
        scene = result.scene
        return CommandResponse(
            ok=True,
            message=(
                f"Built {scene.name.lower()} as {len(scene)} anatomical entities "
                f"({len(scene.visible_ids())} visible at level of detail {scene.active_lod}), "
                f"{len(scene.relationships)} typed relationships and "
                f"{len(result.correspondence)} reserved geometry components. "
                "No geometry has been generated."
            ),
            command=command,
            operation="generate",
            affected=scene.ids(),
            scene_version=scene.version,
            data={"generation": result.summary()},
        )

    def _handle_show(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        targets = self._targets(command)
        applied = editor.apply(ShowEntities(targets), command_text=command.raw_text)
        return self._respond(command, applied)

    def _handle_hide(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        targets = self._targets(command)
        applied = editor.apply(HideEntities(targets), command_text=command.raw_text)
        return self._respond(command, applied)

    def _handle_isolate(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        targets = self._targets(command)
        applied = editor.apply(IsolateEntities(targets), command_text=command.raw_text)
        return self._respond(command, applied)

    def _handle_set_opacity(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        targets = self._targets(command)
        opacity, preset = self._opacity_value(command)
        applied = editor.apply(
            SetOpacity(tuple(targets), opacity, preset_name=preset),
            command_text=command.raw_text,
        )
        return self._respond(command, applied)

    def _opacity_value(self, command: ParsedCommand) -> tuple[float, str | None]:
        """Turn opacity parameters into a value in ``[0, 1]``."""
        preset = command.parameters.get("opacity_preset")
        if preset:
            return self.config.opacity_presets.get(str(preset)), str(preset)
        raw = command.parameters.get("opacity_raw")
        if raw is None:
            raise CommandError(
                "No opacity value was given. Try 'make the left ventricle transparent' or "
                "'set the left ventricle opacity to 0.3'."
            )
        value = float(raw)
        if command.parameters.get("opacity_is_percent"):
            value /= 100.0
        return value, None

    def _handle_set_lod(self, command: ParsedCommand) -> CommandResponse:
        scene = self.require_scene()
        editor = self._require_editor()
        if "lod" in command.parameters:
            target = int(command.parameters["lod"])
        else:
            delta = int(command.parameters.get("lod_delta", 0))
            target = scene.lod_ladder.step(scene.active_lod, delta)
            if target == scene.active_lod:
                bound = "highest" if delta > 0 else "lowest"
                return CommandResponse(
                    ok=True,
                    message=(
                        f"Already at the {bound} level of detail "
                        f"({scene.active_lod}, {scene.lod_spec().name})."
                    ),
                    command=command,
                    operation="set_lod",
                    scene_version=scene.version,
                    data={"lod": scene.active_lod},
                )
        applied = editor.apply(SetLod(target), command_text=command.raw_text)
        return self._respond(command, applied, data=dict(applied.result.data))

    def _handle_set_level(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        raw = str(command.parameters.get("educational_level", ""))
        level = coerce_enum(EducationalLevel, raw, field_name="educational level")
        applied = editor.apply(SetEducationalLevel(level), command_text=command.raw_text)
        return self._respond(command, applied, data=dict(applied.result.data))

    def _handle_reset(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        applied = editor.apply(
            ResetToLodDefaults(default_opacity=self.config.scene_defaults.default_opacity),
            command_text=command.raw_text,
        )
        return self._respond(command, applied)

    def _handle_animate(self, command: ParsedCommand) -> CommandResponse:
        scene = self.require_scene()
        editor = self._require_editor()
        model = CardiacFlowModel(scene, self.config.cardiac_cycle)
        clip = model.build_clip()
        applied = editor.bind_animation(
            clip.bindings(),
            clip.clip_id,
            summary_note=f"{len(clip.paths)} flow paths at {clip.granularity} granularity.",
            command_text=command.raw_text,
        )
        message = (
            f"Bound {len(clip.bindings())} structures to the cardiac cycle across "
            f"{len(clip.phases)} phases, following {len(clip.paths)} flow paths derived from the "
            "functional graph. This is a semantic schedule; no geometry is animated."
        )
        return self._respond(command, applied, message=message, data={"clip": clip.summary()})

    def _handle_stop_animation(self, command: ParsedCommand) -> CommandResponse:
        editor = self._require_editor()
        applied = editor.apply(ClearAnimation(), command_text=command.raw_text)
        return self._respond(command, applied)

    def _handle_explain(self, command: ParsedCommand) -> CommandResponse:
        scene = self.require_scene()
        if command.selector.phrases:
            targets = self._targets(command)
            explanation = (
                self.explainer.explain_entity(scene, targets[0])
                if len(targets) == 1
                else TemplateSceneExplainer().explain_entities(scene, targets)
            )
        else:
            explanation = self.explainer.explain_scene(scene)
        return CommandResponse(
            ok=True,
            message=explanation.rendered(),
            command=command,
            operation="explain",
            affected=explanation.entity_ids,
            scene_version=scene.version,
            data={"explanation_source": explanation.source},
        )

    def _require_editor(self) -> SceneEditor:
        self.require_scene()
        assert self._editor is not None  # set together with the scene
        return self._editor

    # ------------------------------------------------------------------
    @staticmethod
    def supported_commands() -> tuple[str, ...]:
        """Example commands the deterministic engine understands."""
        return RuleBasedCommandParser.supported_forms()
