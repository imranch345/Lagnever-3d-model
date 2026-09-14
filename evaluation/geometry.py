"""Geometry evaluation.

Almost everything in this module is declared rather than implemented, because
there is no geometry to measure. That is the honest state of the project and the
code says so: every declared metric raises
:class:`~awr.errors.NotYetImplementedError` instead of returning a placeholder
number that could leak into a report.

What *is* implemented is the check that matters today:
:func:`evaluate_correspondence` verifies that anatomy and geometry stay linked -
every renderable entity owns a component, no component is shared, and no
component is orphaned.
"""

from __future__ import annotations

from dataclasses import dataclass

from awr.errors import NotYetImplementedError
from awr.scene import AWRScene
from awr.validation import Severity, ValidationReport
from geometry.correspondence import GeometryCorrespondence

__all__ = [
    "DeclaredMetric",
    "DECLARED_GEOMETRY_METRICS",
    "evaluate_correspondence",
    "measure",
]


@dataclass(frozen=True, slots=True)
class DeclaredMetric:
    """A metric the finished system must report, with the reason it cannot yet."""

    name: str
    description: str
    blocked_by: str


DECLARED_GEOMETRY_METRICS: tuple[DeclaredMetric, ...] = (
    DeclaredMetric(
        "watertightness",
        "Fraction of components whose surface is closed.",
        "no geometry decoder exists",
    ),
    DeclaredMetric(
        "manifoldness",
        "Fraction of components that are two-manifold.",
        "no geometry decoder exists",
    ),
    DeclaredMetric(
        "self_intersection",
        "Self-intersecting face count per component.",
        "no geometry decoder exists",
    ),
    DeclaredMetric(
        "part_boundary_agreement",
        "Whether geometric part boundaries agree with anatomical boundaries.",
        "needs both geometry and annotated reference anatomy",
    ),
    DeclaredMetric(
        "surface_distance",
        "Chamfer or Hausdorff distance to a reference surface.",
        "needs a licensed reference dataset",
    ),
    DeclaredMetric(
        "multi_view_consistency",
        "Agreement of renderings across viewpoints.",
        "needs a renderer and generated geometry",
    ),
    DeclaredMetric(
        "text_compliance",
        "Whether produced geometry satisfies the text request.",
        "needs generated geometry and a grounded scorer",
    ),
    DeclaredMetric(
        "educational_appropriateness",
        "Whether the visible detail suits the audience level.",
        "needs expert rubric and human evaluation",
    ),
)
"""Metrics the finished system must report. None of them can be computed yet."""


def measure(metric_name: str, *args: object, **kwargs: object) -> float:
    """Compute a declared geometry metric.

    Always raises: none of these metrics can be computed before the geometry
    decoder exists. The function is the single entry point so that the failure is
    uniform and traceable.
    """
    known = {metric.name for metric in DECLARED_GEOMETRY_METRICS}
    if metric_name not in known:
        raise NotYetImplementedError(
            f"Geometry metric {metric_name!r} is not declared. Known metrics: "
            f"{', '.join(sorted(known))}",
            planned_in="the evaluation step that follows geometry decoding",
        )
    blocked = next(m for m in DECLARED_GEOMETRY_METRICS if m.name == metric_name)
    raise NotYetImplementedError(
        f"Geometry metric {metric_name!r} ({blocked.blocked_by})",
        planned_in="the evaluation step that follows geometry decoding",
    )


def evaluate_correspondence(
    scene: AWRScene, correspondence: GeometryCorrespondence | None = None
) -> ValidationReport:
    """Check that every renderable entity keeps an unambiguous geometry link."""
    report = ValidationReport(subject=f"geometry correspondence {scene.scene_id}")

    owners: dict[str, str] = {}
    for entity in scene.iter_entities():
        reference = entity.geometry_reference
        if entity.renderable and reference is None:
            report.add(
                "MISSING_GEOMETRY_REFERENCE",
                Severity.ERROR,
                "Renderable entity has no reserved geometry component.",
                [entity.entity_id],
            )
            continue
        if reference is None:
            continue
        if not entity.renderable:
            report.add(
                "GEOMETRY_ON_GROUP",
                Severity.WARNING,
                "An organisational group owns a geometry component.",
                [entity.entity_id],
            )
        previous = owners.get(reference.component_id)
        if previous:
            report.add(
                "SHARED_COMPONENT",
                Severity.ERROR,
                f"Component {reference.component_id!r} is claimed by two entities.",
                [previous, entity.entity_id],
            )
        owners[reference.component_id] = entity.entity_id

    if correspondence is not None:
        for component_id, entity_id in owners.items():
            if component_id not in correspondence:
                report.add(
                    "COMPONENT_NOT_IN_MAP",
                    Severity.ERROR,
                    f"Component {component_id!r} is on an entity but missing from the "
                    "correspondence map.",
                    [entity_id],
                )
        for entry in correspondence.entries():
            if entry.entity_id not in scene:
                report.add(
                    "ORPHAN_COMPONENT",
                    Severity.ERROR,
                    f"Component {entry.component_id!r} belongs to {entry.entity_id!r}, which is "
                    "not in the scene. Part identity has been lost.",
                    [entry.entity_id],
                )

    resolved = sum(
        1
        for entity in scene.iter_entities()
        if entity.geometry_reference and entity.geometry_reference.is_resolved
    )
    report.add(
        "NO_GEOMETRY_PAYLOAD",
        Severity.INFO,
        f"{resolved} of {len(owners)} components carry geometry data. The prototype "
        "reserves component slots only.",
    )
    return report
