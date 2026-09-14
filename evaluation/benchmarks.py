"""Command-sequence benchmarks for the deterministic prototype.

A benchmark case is a sequence of natural-language commands plus assertions about
the AWR state afterwards. This is the closest thing the milestone has to an
end-to-end score, and it measures the right thing: whether language reliably
produces the intended change in a persistent, structured representation.

Every case is deterministic. Running the suite twice on the same code and the
same ontology must produce identical results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.scene import AWRScene
from reasoning.command_engine import CommandEngine, CommandResponse

__all__ = [
    "CaseOutcome",
    "BenchmarkCase",
    "BenchmarkReport",
    "HEART_V01_BENCHMARK",
    "run_benchmark",
    "run_benchmark_suite",
]

Check = Callable[[AWRScene, "tuple[CommandResponse, ...]"], "list[str]"]
"""A check returns a list of failure messages; an empty list means it passed."""


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One command sequence with assertions about the resulting scene."""

    name: str
    commands: tuple[str, ...]
    checks: tuple[Check, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Result of running one benchmark case."""

    name: str
    passed: bool
    failures: tuple[str, ...] = ()
    responses: tuple[CommandResponse, ...] = ()
    scene_version: int | None = None

    def describe(self) -> str:
        """One-line human-readable description."""
        status = "PASS" if self.passed else "FAIL"
        detail = "" if self.passed else f" ({'; '.join(self.failures)})"
        return f"{status} {self.name}{detail}"


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate result of a benchmark run."""

    outcomes: tuple[CaseOutcome, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        """Number of passing cases."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def failed(self) -> int:
        """Number of failing cases."""
        return len(self.outcomes) - self.passed

    @property
    def is_green(self) -> bool:
        """Whether every case passed."""
        return self.failed == 0

    def summary(self) -> dict[str, Any]:
        """Compact result summary."""
        return {
            "cases": len(self.outcomes),
            "passed": self.passed,
            "failed": self.failed,
            "failures": [outcome.describe() for outcome in self.outcomes if not outcome.passed],
        }


# --- checks -------------------------------------------------------------------


def _visible(scene: AWRScene) -> set[str]:
    return set(scene.visible_ids())


def check_only_chambers_visible(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """Exactly the four chambers are visible."""
    chambers = set(scene.expand(["heart.chambers"]))
    visible = _visible(scene)
    if visible == chambers:
        return []
    return [f"expected only the four chambers visible, got {sorted(visible)}"]


def check_lv_transparent(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """The left ventricle is visible but not opaque, and kept its id."""
    lv = scene.get("heart.left_ventricle")
    failures: list[str] = []
    if lv.entity_id != "heart.left_ventricle":
        failures.append("left ventricle entity id changed")
    if lv.opacity >= 1.0:
        failures.append(f"expected reduced opacity, got {lv.opacity}")
    if not lv.visibility:
        failures.append("left ventricle should stay visible when made transparent")
    return failures


def check_valves_visible(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """All four valves are visible."""
    valves = set(scene.expand(["heart.valves"]))
    missing = sorted(valves - _visible(scene))
    return [f"valves not visible: {missing}"] if missing else []


def check_medical_level(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """Medical level maps to the deepest LOD and reveals the conduction system."""
    failures: list[str] = []
    if str(scene.educational_level) != "medical":
        failures.append(f"educational level is {scene.educational_level}")
    if scene.active_lod != scene.lod_ladder.max_level:
        failures.append(f"expected LOD {scene.lod_ladder.max_level}, got {scene.active_lod}")
    if not scene.get("heart.sinoatrial_node").visibility:
        failures.append("sinoatrial node should be visible at medical level")
    return failures


def check_identity_preserved(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """The full ontology is still present after every edit."""
    if len(scene) != 42:
        return [f"expected 42 entities, got {len(scene)}"]
    return []


def check_all_ok(scene: AWRScene, responses: Sequence[CommandResponse]) -> list[str]:
    """Every command in the sequence succeeded."""
    return [
        f"command failed: {r.command.raw_text if r.command else '?'} -> {r.message}"
        for r in responses
        if not r.ok
    ]


def check_animation_bound(scene: AWRScene, _: Sequence[CommandResponse]) -> list[str]:
    """The cardiac cycle is bound to the chambers."""
    animated = [e.entity_id for e in scene.iter_entities() if e.animation_state.is_animated]
    if not animated:
        return ["no entity is bound to an animation clip"]
    missing = [
        entity_id
        for entity_id in scene.expand(["heart.chambers"])
        if entity_id not in animated
    ]
    return [f"chambers missing from the clip: {missing}"] if missing else []


def check_history_matches_commands(
    scene: AWRScene, responses: Sequence[CommandResponse]
) -> list[str]:
    """Scene history records at least one event per state-changing command."""
    if not scene.history:
        return ["scene history is empty after a sequence of commands"]
    if scene.version != len(scene.history):
        return [f"version {scene.version} does not match {len(scene.history)} events"]
    return []


HEART_V01_BENCHMARK: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="generate_and_inspect",
        commands=("generate a human heart", "explain what is happening"),
        checks=(check_all_ok, check_identity_preserved),
        description="A request builds the full AWR and the scene can describe itself.",
    ),
    BenchmarkCase(
        name="chambers_workflow",
        commands=(
            "generate a human heart",
            "show the four chambers",
            "hide everything except the chambers",
        ),
        checks=(check_all_ok, check_only_chambers_visible, check_identity_preserved),
        description="The canonical isolation workflow from the specification.",
    ),
    BenchmarkCase(
        name="transparency_persists",
        commands=(
            "generate a human heart",
            "hide everything except the chambers",
            "make the left ventricle transparent",
            "show the valves",
        ),
        checks=(check_all_ok, check_lv_transparent, check_valves_visible),
        description="An opacity edit survives later visibility commands.",
    ),
    BenchmarkCase(
        name="level_switching",
        commands=(
            "generate a human heart",
            "make the left ventricle transparent",
            "switch to medical level",
        ),
        checks=(check_all_ok, check_medical_level, check_lv_transparent),
        description="Changing audience level changes visibility but not material state.",
    ),
    BenchmarkCase(
        name="animation_binding",
        commands=("generate a human heart", "show more detail", "animate blood flow"),
        checks=(check_all_ok, check_animation_bound, check_history_matches_commands),
        description="Blood flow is bound from the functional graph.",
    ),
)
"""The v0.1 benchmark suite. Deterministic and ontology-driven."""


def run_benchmark(
    case: BenchmarkCase, engine_factory: Callable[[], CommandEngine] | None = None
) -> CaseOutcome:
    """Run one benchmark case on a fresh engine."""
    engine = (engine_factory or CommandEngine)()
    responses = engine.run(case.commands)
    scene = engine.scene
    if scene is None:
        return CaseOutcome(
            name=case.name,
            passed=False,
            failures=("no scene was produced",),
            responses=responses,
        )
    failures: list[str] = []
    for check in case.checks:
        failures.extend(check(scene, responses))
    return CaseOutcome(
        name=case.name,
        passed=not failures,
        failures=tuple(failures),
        responses=responses,
        scene_version=scene.version,
    )


def run_benchmark_suite(
    cases: Sequence[BenchmarkCase] = HEART_V01_BENCHMARK,
    engine_factory: Callable[[], CommandEngine] | None = None,
) -> BenchmarkReport:
    """Run a whole suite and return an aggregate report."""
    outcomes = tuple(run_benchmark(case, engine_factory) for case in cases)
    return BenchmarkReport(
        outcomes=outcomes,
        metadata={"suite_size": len(outcomes), "deterministic": True},
    )
