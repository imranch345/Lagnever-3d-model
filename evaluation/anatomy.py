"""Anatomical completeness and structural checks for an ontology.

What is checked here is *coverage and structure*, not medical truth. The
prototype can verify that the heart ontology contains the entities and
relationships the specification requires, that valves sit between the right
chambers, that laterality is consistent. It cannot verify that the anatomy is
clinically correct, and it does not pretend to: that requires expert review and
is declared as future work.
"""

from __future__ import annotations

from awr.errors import NotYetImplementedError
from awr.ontology import AnatomyOntology
from awr.relationships import GraphKind
from awr.schema import AnatomyType, Laterality
from awr.validation import Severity, ValidationReport

__all__ = [
    "MANDATORY_HEART_ENTITIES",
    "REQUIRED_FUNCTIONAL_EDGES",
    "evaluate_anatomy",
    "evaluate_against_medical_reference",
]

MANDATORY_HEART_ENTITIES: tuple[str, ...] = (
    # organ
    "heart",
    # chambers
    "heart.right_atrium",
    "heart.right_ventricle",
    "heart.left_atrium",
    "heart.left_ventricle",
    # valves
    "heart.tricuspid_valve",
    "heart.pulmonary_valve",
    "heart.mitral_valve",
    "heart.aortic_valve",
    # great vessels
    "heart.superior_vena_cava",
    "heart.inferior_vena_cava",
    "heart.pulmonary_trunk",
    "heart.pulmonary_arteries",
    "heart.pulmonary_veins",
    "heart.aorta",
    # structural extensions
    "heart.interatrial_septum",
    "heart.interventricular_septum",
    "heart.endocardium",
    "heart.myocardium",
    "heart.epicardium",
    "heart.pericardium",
    # richer / medical extensions
    "heart.papillary_muscles",
    "heart.chordae_tendineae",
    "heart.trabeculae_carneae",
    "heart.moderator_band",
    "heart.fossa_ovalis",
    "heart.coronary_circulation",
    "heart.cardiac_conduction_system",
)
"""Entities Heart Ontology v0.1 must contain, per the project specification."""

REQUIRED_FUNCTIONAL_EDGES: tuple[tuple[str, str, str], ...] = (
    ("heart.right_atrium", "receives_from", "heart.superior_vena_cava"),
    ("heart.right_atrium", "receives_from", "heart.inferior_vena_cava"),
    ("heart.right_ventricle", "pumps_to", "heart.pulmonary_trunk"),
    ("heart.left_atrium", "receives_from", "heart.pulmonary_veins"),
    ("heart.left_ventricle", "pumps_to", "heart.aorta"),
)
"""Blood-flow edges the specification names explicitly."""

REQUIRED_STRUCTURAL_EDGES: tuple[tuple[str, str, str], ...] = (
    ("heart.right_atrium", "connects_to", "heart.right_ventricle"),
    ("heart.left_atrium", "connects_to", "heart.left_ventricle"),
)

_EXPECTED_LATERALITY: dict[str, Laterality] = {
    "heart.right_atrium": Laterality.RIGHT,
    "heart.right_ventricle": Laterality.RIGHT,
    "heart.left_atrium": Laterality.LEFT,
    "heart.left_ventricle": Laterality.LEFT,
    "heart.tricuspid_valve": Laterality.RIGHT,
    "heart.pulmonary_valve": Laterality.RIGHT,
    "heart.mitral_valve": Laterality.LEFT,
    "heart.aortic_valve": Laterality.LEFT,
}


def evaluate_anatomy(ontology: AnatomyOntology) -> ValidationReport:
    """Check ontology coverage, mandated relationships and valve placement."""
    report = ValidationReport(subject=f"anatomy coverage {ontology.ontology_id}@{ontology.version}")

    for entity_id in MANDATORY_HEART_ENTITIES:
        if entity_id not in ontology:
            report.add(
                "MISSING_MANDATORY_ENTITY",
                Severity.ERROR,
                "Entity required by the Heart Ontology v0.1 specification is absent.",
                [entity_id],
            )

    for subject, relation, obj in REQUIRED_FUNCTIONAL_EDGES + REQUIRED_STRUCTURAL_EDGES:
        if subject not in ontology or obj not in ontology:
            continue
        if obj not in ontology.relationships.related(subject, relation):
            report.add(
                "MISSING_REQUIRED_RELATION",
                Severity.ERROR,
                f"Required relationship {subject} {relation} {obj} is missing.",
                [subject, obj],
            )

    for entity_id, expected in _EXPECTED_LATERALITY.items():
        if entity_id not in ontology:
            continue
        actual = ontology.get(entity_id).laterality
        if actual is not expected:
            report.add(
                "LATERALITY_MISMATCH",
                Severity.WARNING,
                f"Laterality is {actual}, expected {expected}.",
                [entity_id],
            )

    # Every valve must sit on a flow path: something opens into it and it opens
    # into something. A valve that leads nowhere is a modelling error.
    functional = ontology.relationships.graph(GraphKind.FUNCTIONAL)
    for valve_id in ontology.by_type(AnatomyType.VALVE):
        downstream = functional.related(valve_id, "opens_into")
        upstream = [edge.subject for edge in functional.incoming(valve_id, "opens_into")]
        if not downstream:
            report.add(
                "VALVE_WITHOUT_OUTFLOW", Severity.ERROR,
                "Valve does not open into anything.", [valve_id],
            )
        if not upstream:
            report.add(
                "VALVE_WITHOUT_INFLOW", Severity.ERROR,
                "Nothing opens into this valve.", [valve_id],
            )

    # Every chamber should be reachable in the functional graph.
    for chamber_id in ontology.by_type(AnatomyType.CHAMBER):
        if not functional.incident(chamber_id):
            report.add(
                "CHAMBER_NOT_IN_FLOW", Severity.ERROR,
                "Chamber has no functional relationships.", [chamber_id],
            )

    report.add(
        "COVERAGE_ONLY",
        Severity.INFO,
        "These checks verify coverage and structure, not clinical correctness. "
        "Medical validation is a separate, later R&D step.",
    )
    return report


def evaluate_against_medical_reference(ontology: AnatomyOntology, reference: object) -> None:
    """Compare an ontology against a validated medical reference.

    TODO: requires a licensed anatomical reference and expert review, which the
    project has deliberately not acquired yet. Declared so that the gap between
    "structurally complete" and "medically validated" stays visible in code.
    """
    raise NotYetImplementedError(
        "evaluate_against_medical_reference",
        planned_in="the medical validation step, after licensing is settled",
    )
