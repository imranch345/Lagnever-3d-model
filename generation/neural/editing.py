"""Persistent editing: what an instruction actually has to recompute.

STATUS: the edit planner is implemented and tested; the learned local editor is a
declared interface.

The claim being designed for
----------------------------

"Generate a heart" followed by "make the left ventricle transparent" must not
regenerate the heart. In this architecture that is not an optimisation, it is a
consequence of where information lives:

* Visibility, opacity, level of detail and animation live in the **scene**, not in
  the latent. Changing them is symbolic work that Step 4 already does, and it
  requires **zero neural computation**.
* Geometry lives in per-entity token blocks. Changing one structure's shape
  re-decodes one block. Every other block is bit-identical, so the rest of the
  scene is untouched by construction rather than by a locality loss.

:func:`plan_edit` makes this concrete and checkable: given a program and a scene,
it returns exactly which entities need geometry recomputed and which are
guaranteed frozen. An implementation that recomputes more than the plan says is
violating the architecture, and the plan is the thing to test it against.

Four kinds of edit
------------------

==================  ==========================================  =========================
kind                example                                     geometry recomputed
==================  ==========================================  =========================
state only          "make the left ventricle transparent"       none
level of detail     "show more detail"                          entities whose token prefix grows
structural          "remove the pericardium" (future)           the entity and its immediate context
geometric           "thicken the ventricle wall" (future)       the entity alone
==================  ==========================================  =========================
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from awr.errors import NotYetImplementedError
from awr.scene import AWRScene
from generation.neural.language_encoder import AWRProgram, ProgramOpType
from generation.neural.three_d_latent import tokens_for_lod

__all__ = [
    "EditKind",
    "EditPlan",
    "plan_edit",
    "EditStrategy",
    "EDIT_STRATEGIES",
    "RECOMMENDED_EDIT_STRATEGY",
    "LocalGeometryEditor",
]


class EditKind(StrEnum):
    """How much of the pipeline an edit reaches."""

    STATE_ONLY = "state_only"
    LOD_REFINEMENT = "lod_refinement"
    STRUCTURAL = "structural"
    GEOMETRIC = "geometric"

    @property
    def touches_geometry(self) -> bool:
        """Whether the edit can require any geometry decoding at all."""
        return self is not EditKind.STATE_ONLY


_STATE_ONLY_OPS = frozenset(
    {
        ProgramOpType.SHOW,
        ProgramOpType.HIDE,
        ProgramOpType.ISOLATE,
        ProgramOpType.SHOW_ALL,
        ProgramOpType.SET_OPACITY,
        ProgramOpType.RESET_VIEW,
        ProgramOpType.ANIMATE,
        ProgramOpType.STOP_ANIMATION,
        ProgramOpType.EXPLAIN,
        ProgramOpType.NOOP,
    }
)
_LOD_OPS = frozenset({ProgramOpType.SET_LOD, ProgramOpType.SET_EDUCATIONAL_LEVEL})


@dataclass(frozen=True, slots=True)
class EditPlan:
    """What an edit changes, what it must recompute, and what is guaranteed frozen."""

    kind: EditKind
    operations: tuple[str, ...]
    touched_entities: tuple[str, ...]
    recompute_entities: tuple[str, ...]
    frozen_entities: tuple[str, ...]
    guarantees: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    requires_full_regeneration: bool = False

    @property
    def recompute_fraction(self) -> float:
        """Share of renderable entities whose geometry must be decoded again."""
        total = len(self.recompute_entities) + len(self.frozen_entities)
        return 0.0 if total == 0 else len(self.recompute_entities) / total

    def summary(self) -> dict[str, object]:
        """Compact description for reports and tests."""
        return {
            "kind": str(self.kind),
            "operations": list(self.operations),
            "touched": len(self.touched_entities),
            "recompute": len(self.recompute_entities),
            "frozen": len(self.frozen_entities),
            "recompute_fraction": round(self.recompute_fraction, 4),
            "full_regeneration": self.requires_full_regeneration,
        }


def plan_edit(
    program: AWRProgram,
    scene: AWRScene,
    *,
    decoded_entities: Iterable[str] | None = None,
    target_lod: int | None = None,
) -> EditPlan:
    """Work out what an edit must recompute.

    Args:
        program: The AWR program about to be applied.
        scene: The scene it will be applied to.
        decoded_entities: Entities whose geometry has already been decoded. Pass
            ``None`` to assume eager decoding of every renderable entity, which is
            the prototype's behaviour.
        target_lod: Level of detail after the edit, when the program changes it.

    Returns:
        An :class:`EditPlan`. For a state-only edit ``recompute_entities`` is
        empty, which is the architecture's central editing claim in testable form.

    """
    renderable = set(scene.renderable_ids())
    cached = set(decoded_entities) if decoded_entities is not None else set(renderable)
    op_types = [op.op_type for op in program.ops]
    operations = tuple(str(op) for op in op_types)

    if any(op is ProgramOpType.GENERATE for op in op_types):
        return EditPlan(
            kind=EditKind.STRUCTURAL,
            operations=operations,
            touched_entities=tuple(scene.ids()),
            recompute_entities=tuple(sorted(renderable)),
            frozen_entities=(),
            guarantees=("A generate request builds a scene; nothing is preserved from before.",),
            requires_full_regeneration=True,
        )

    touched: list[str] = []
    for op in program.ops:
        for entity_id in op.entity_args:
            for target in scene.expand([entity_id]):
                if target not in touched:
                    touched.append(target)

    if all(op in _STATE_ONLY_OPS for op in op_types):
        missing = tuple(sorted(entity for entity in touched if entity not in cached))
        return EditPlan(
            kind=EditKind.STATE_ONLY,
            operations=operations,
            touched_entities=tuple(touched),
            recompute_entities=missing,
            frozen_entities=tuple(sorted(renderable - set(missing))),
            guarantees=(
                "No geometry token block is modified.",
                "Entity identity, hierarchy and relationships are untouched.",
                "With eager decoding this edit costs zero neural computation.",
            ),
            notes=(
                ()
                if not missing
                else (
                    f"{len(missing)} entities have no cached geometry yet and would be decoded "
                    "on first display under a lazy decoding policy.",
                )
            ),
        )

    if all(op in _LOD_OPS | _STATE_ONLY_OPS for op in op_types):
        current = scene.active_lod
        new_lod = current if target_lod is None else target_lod
        grows = tokens_for_lod(new_lod) > tokens_for_lod(current)
        affected = (
            tuple(
                sorted(
                    entity
                    for entity in renderable
                    if scene.default_visibility(entity, new_lod)
                    and (grows or entity not in cached)
                )
            )
            if new_lod != current
            else ()
        )
        return EditPlan(
            kind=EditKind.LOD_REFINEMENT,
            operations=operations,
            touched_entities=tuple(sorted(renderable)),
            recompute_entities=affected,
            frozen_entities=tuple(sorted(renderable - set(affected))),
            guarantees=(
                "Entity identity is unchanged: a level of detail reads a longer prefix of the "
                "same token block, it does not select a different latent.",
                "Structures hidden at the new level still exist in the representation.",
            ),
            notes=(
                f"Token prefix moves from {tokens_for_lod(current)} to {tokens_for_lod(new_lod)} "
                "tokens per entity.",
            ),
        )

    return EditPlan(
        kind=EditKind.GEOMETRIC,
        operations=operations,
        touched_entities=tuple(touched),
        recompute_entities=tuple(sorted(set(touched) & renderable)),
        frozen_entities=tuple(sorted(renderable - set(touched))),
        guarantees=(
            "Only the named entities' token blocks are rewritten; every other block is "
            "bit-identical after the edit.",
        ),
        notes=("Geometric editing is declared, not implemented; see LocalGeometryEditor.",),
    )


@dataclass(frozen=True, slots=True)
class EditStrategy:
    """A candidate mechanism for geometric editing."""

    name: str
    summary: str
    advantages: tuple[str, ...]
    disadvantages: tuple[str, ...]
    verdict: str
    revisit_when: str | None = None


EDIT_STRATEGIES: tuple[EditStrategy, ...] = (
    EditStrategy(
        name="full_regeneration",
        summary="Re-run generation with an amended prompt.",
        advantages=("Trivial to implement; no extra machinery.",),
        disadvantages=(
            "Produces a different object, so identity and every prior edit are lost.",
            "This is the behaviour the project exists to replace.",
        ),
        verdict="REJECTED",
    ),
    EditStrategy(
        name="global_latent_modification",
        summary="Edit a single scene-level latent and decode everything again.",
        advantages=("Simple; one latent to manipulate.",),
        disadvantages=(
            "No locality: every structure changes because every structure shares the latent.",
            "Requires a locality loss to approximate what factorisation gives for free.",
        ),
        verdict="REJECTED",
    ),
    EditStrategy(
        name="masked_local_redecoding",
        summary=(
            "Rewrite only the affected entity's geometry tokens and re-decode that entity, "
            "with neighbouring entities and the scene latent as frozen context."
        ),
        advantages=(
            "Locality is structural: untouched blocks are not written, so they cannot change.",
            "Cost scales with the edit, not with the scene.",
            "Directly testable: compare frozen blocks before and after, expect equality.",
        ),
        disadvantages=(
            "Seams at boundaries with neighbouring entities must be managed.",
            "Needs context conditioning so the edited part still fits its neighbours.",
        ),
        verdict="RECOMMENDED for the first prototype",
    ),
    EditStrategy(
        name="local_diffusion_or_flow",
        summary="Run a diffusion or flow model over the affected entity's tokens.",
        advantages=("Strong generative quality; natural handling of ambiguity in an edit.",),
        disadvantages=(
            "Slower at inference; another training loop before the basics are working.",
            "Stochastic, so repeated identical edits need seed discipline to stay reproducible.",
        ),
        verdict="DEFERRED",
        revisit_when="Deterministic re-decoding proves too blurry or too low in diversity.",
    ),
    EditStrategy(
        name="graph_modification_only",
        summary="Change the AWR graph and let the decoder follow.",
        advantages=(
            "Exactly right for structural and state edits, which is most of the command set.",
        ),
        disadvantages=("Cannot express a purely geometric change such as wall thickness.",),
        verdict="ADOPTED for state and structural edits",
    ),
)
"""Candidate editing mechanisms. See ADR 0005."""


RECOMMENDED_EDIT_STRATEGY = "masked_local_redecoding"
"""Proposed geometric editing mechanism for the first prototype."""


class LocalGeometryEditor:
    """Interface for entity-local geometry editing. Declared, not implemented.

    Contract for any implementation: given a scene latent, an entity and an edit
    instruction, return a new token block for **that entity only**. Writing any
    other entity's block is a contract violation, and the test for it is exact
    equality, not a tolerance.
    """

    editor_id = "lagnav-local-editor-undefined"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotYetImplementedError(
            "LocalGeometryEditor", planned_in="Step 6, curriculum stage S4"
        )

    def edit(
        self, latent: Mapping[str, object], entity_id: str, instruction: str
    ) -> Mapping[str, object]:  # pragma: no cover - construction raises
        """Produce a replacement token block for one entity."""
        raise NotYetImplementedError("LocalGeometryEditor.edit", planned_in="Step 6")

    @staticmethod
    def locality_guarantee() -> Sequence[str]:
        """The properties an implementation is required to satisfy."""
        return (
            "Token blocks of entities outside the edit are bit-identical before and after.",
            "Entity identity slices are never written by an edit.",
            "Scene history records the edit as one versioned event, as in Step 4.",
        )
