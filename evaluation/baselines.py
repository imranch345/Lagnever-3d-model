"""Baseline interfaces for comparing Lagnav against other systems.

STATUS: interfaces only. No external system is integrated, downloaded or wrapped.

Two kinds of baseline, and the distinction matters
--------------------------------------------------

* **Internal ablation.** Our own appearance-driven model, trained on the same
  data, for the same steps, at a matched parameter count. This is the comparison
  the research hypothesis actually rests on, because everything except the
  representation is held constant.
* **External system.** An existing open text-to-3D system such as TRELLIS or
  Hunyuan3D. These are reference points, not the architecture, and they are not
  controlled comparisons: different training data, different scale, different
  objectives. A number from them says where the field is, not whether the
  hypothesis holds.

Reporting rule
--------------

Several axes cannot be measured on an external system at all, because the system
has no interface for them. Persistent editing is the clearest case: a system that
regenerates from a new prompt is not scoring badly at persistent editing, it is
not doing the task. Those cells are reported as ``not supported`` and never as
zero, which :class:`CapabilityMatrix` enforces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from awr.errors import NotYetImplementedError

__all__ = [
    "BaselineKind",
    "ComparisonAxis",
    "Capability",
    "BaselineSpec",
    "BASELINES",
    "CapabilityMatrix",
    "BaselineAdapter",
]


class BaselineKind(StrEnum):
    """What sort of comparison a baseline provides."""

    INTERNAL_ABLATION = "internal_ablation"
    EXTERNAL_SYSTEM = "external_system"
    DETERMINISTIC_REFERENCE = "deterministic_reference"


class ComparisonAxis(StrEnum):
    """Axes on which systems are compared."""

    ANATOMICAL_ACCURACY = "anatomical_accuracy"
    SEMANTIC_PART_CONTROL = "semantic_part_control"
    PERSISTENT_EDITING = "persistent_editing"
    MULTI_VIEW_CONSISTENCY = "multi_view_consistency"
    TOPOLOGY = "topology"
    INTERNAL_STRUCTURE = "internal_structure"
    FUNCTIONAL_BEHAVIOUR = "functional_behaviour"


class Capability(StrEnum):
    """Whether a system can be asked to do a thing at all."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """One baseline, with its licensing and integration status."""

    name: str
    kind: BaselineKind
    description: str
    capabilities: Mapping[ComparisonAxis, Capability]
    licence_status: str = "not reviewed"
    integration_status: str = "not integrated"
    cautions: tuple[str, ...] = ()

    def capability(self, axis: ComparisonAxis) -> Capability:
        """Capability on one axis."""
        return self.capabilities.get(axis, Capability.UNKNOWN)

    def comparable_axes(self) -> tuple[ComparisonAxis, ...]:
        """Axes on which a number from this baseline would mean something."""
        return tuple(
            axis
            for axis in ComparisonAxis
            if self.capability(axis) in (Capability.SUPPORTED, Capability.PARTIAL)
        )


_FULL = dict.fromkeys(ComparisonAxis, Capability.SUPPORTED)


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="step4_deterministic",
        kind=BaselineKind.DETERMINISTIC_REFERENCE,
        description=(
            "The Step 4 AWR prototype. Perfect on identity, control and edit locality by "
            "construction, and incapable of producing geometry. It is the upper bound for the "
            "semantic axes and the floor for the geometric ones."
        ),
        capabilities={
            ComparisonAxis.ANATOMICAL_ACCURACY: Capability.PARTIAL,
            ComparisonAxis.SEMANTIC_PART_CONTROL: Capability.SUPPORTED,
            ComparisonAxis.PERSISTENT_EDITING: Capability.SUPPORTED,
            ComparisonAxis.MULTI_VIEW_CONSISTENCY: Capability.NOT_SUPPORTED,
            ComparisonAxis.TOPOLOGY: Capability.NOT_SUPPORTED,
            ComparisonAxis.INTERNAL_STRUCTURE: Capability.SUPPORTED,
            ComparisonAxis.FUNCTIONAL_BEHAVIOUR: Capability.PARTIAL,
        },
        licence_status="internal",
        integration_status="available in this repository",
        cautions=(
            "Scoring 1.0 on the semantic axes is a property of a symbolic system, not evidence "
            "that a learned model can do the same.",
        ),
    ),
    BaselineSpec(
        name="appearance_driven_ablation",
        kind=BaselineKind.INTERNAL_ABLATION,
        description=(
            "Our own model with the entity axis, the graphs and per-entity geometry removed. "
            "Same data, same steps, matched parameters. The controlled comparison."
        ),
        capabilities={
            ComparisonAxis.ANATOMICAL_ACCURACY: Capability.SUPPORTED,
            ComparisonAxis.SEMANTIC_PART_CONTROL: Capability.PARTIAL,
            ComparisonAxis.PERSISTENT_EDITING: Capability.PARTIAL,
            ComparisonAxis.MULTI_VIEW_CONSISTENCY: Capability.SUPPORTED,
            ComparisonAxis.TOPOLOGY: Capability.SUPPORTED,
            ComparisonAxis.INTERNAL_STRUCTURE: Capability.PARTIAL,
            ComparisonAxis.FUNCTIONAL_BEHAVIOUR: Capability.NOT_SUPPORTED,
        },
        licence_status="internal",
        integration_status="specified in configs/neural/baseline.yaml, not built",
        cautions=(
            "Must be parameter-matched and trained identically, or the comparison measures "
            "budget rather than architecture.",
        ),
    ),
    BaselineSpec(
        name="external_text_to_3d",
        kind=BaselineKind.EXTERNAL_SYSTEM,
        description=(
            "An existing open text-to-3D system, for example TRELLIS or Hunyuan3D, evaluated "
            "as a reference point for geometry quality only."
        ),
        capabilities={
            ComparisonAxis.ANATOMICAL_ACCURACY: Capability.UNKNOWN,
            ComparisonAxis.SEMANTIC_PART_CONTROL: Capability.NOT_SUPPORTED,
            ComparisonAxis.PERSISTENT_EDITING: Capability.NOT_SUPPORTED,
            ComparisonAxis.MULTI_VIEW_CONSISTENCY: Capability.SUPPORTED,
            ComparisonAxis.TOPOLOGY: Capability.SUPPORTED,
            ComparisonAxis.INTERNAL_STRUCTURE: Capability.NOT_SUPPORTED,
            ComparisonAxis.FUNCTIONAL_BEHAVIOUR: Capability.NOT_SUPPORTED,
        },
        licence_status="must be reviewed before any use, including evaluation",
        integration_status="not integrated; interface only",
        cautions=(
            "Not a controlled comparison: different data, scale and objectives.",
            "Never part of the Lagnav architecture, at any stage.",
            "Geometry quality gaps are expected and are not evidence about the hypothesis.",
        ),
    ),
)
"""Declared baselines. None is integrated."""


@dataclass(frozen=True, slots=True)
class CapabilityMatrix:
    """Which baselines can be asked which questions."""

    baselines: tuple[BaselineSpec, ...] = field(default_factory=lambda: BASELINES)

    def comparable(self, axis: ComparisonAxis) -> tuple[str, ...]:
        """Baselines that can meaningfully be scored on an axis."""
        return tuple(
            baseline.name
            for baseline in self.baselines
            if baseline.capability(axis) in (Capability.SUPPORTED, Capability.PARTIAL)
        )

    def unsupported(self, axis: ComparisonAxis) -> tuple[str, ...]:
        """Baselines that must be reported as 'not supported' on an axis."""
        return tuple(
            baseline.name
            for baseline in self.baselines
            if baseline.capability(axis) is Capability.NOT_SUPPORTED
        )

    def table(self) -> str:
        """Markdown capability table for the design document."""
        header = "| axis | " + " | ".join(b.name for b in self.baselines) + " |"
        divider = "| --- " * (len(self.baselines) + 1) + "|"
        lines = [header, divider]
        for axis in ComparisonAxis:
            cells = " | ".join(str(b.capability(axis)) for b in self.baselines)
            lines.append(f"| `{axis}` | {cells} |")
        return "\n".join(lines)


class BaselineAdapter:
    """Interface a baseline must expose to be evaluated. Declared, not implemented.

    Deliberately narrow: a prompt goes in, a scene-like result comes out, and the
    adapter says which axes it can answer. Nothing about a baseline's internals
    enters Lagnav, and no baseline becomes a dependency of the architecture.
    """

    baseline_name: str = "undefined"

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotYetImplementedError(
            "BaselineAdapter",
            planned_in="the evaluation step that follows the first experiment",
        )

    def generate(self, prompt: str) -> object:  # pragma: no cover - construction raises
        """Produce a result for a prompt."""
        raise NotYetImplementedError("BaselineAdapter.generate")

    def supports(self, axis: ComparisonAxis) -> Capability:  # pragma: no cover
        """Whether this baseline can be asked about an axis."""
        raise NotYetImplementedError("BaselineAdapter.supports")

    @staticmethod
    def reporting_rules() -> Sequence[str]:
        """Rules any comparison report must follow."""
        return (
            "An unsupported axis is reported as 'not supported', never as a score of zero.",
            "External systems are reported separately from the controlled internal ablation.",
            "Parameter count, training data and training steps are reported alongside every "
            "number, because a comparison without them is not a comparison.",
        )
