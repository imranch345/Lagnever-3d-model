"""Entity resolution: synonyms, abbreviations, groups, ambiguity and errors."""

from __future__ import annotations

import pytest

from awr.errors import AmbiguousEntityReferenceError, UnknownEntityReferenceError
from reasoning.entity_resolver import EntityResolver, MatchKind, normalize_phrase


@pytest.mark.parametrize(
    "phrase",
    [
        "left ventricle",
        "Left Ventricle",
        "  the LEFT   ventricle ",
        "LV",
        "lv",
        "left ventricular chamber",
        "left_ventricle",
        "heart.left_ventricle",
        "the left ventricle.",
    ],
)
def test_variants_resolve_to_the_same_entity(resolver: EntityResolver, phrase: str) -> None:
    """Every accepted surface form returns the same persistent id."""
    assert resolver.resolve(phrase).entity_id == "heart.left_ventricle"


def test_resolution_reports_how_it_matched(resolver: EntityResolver) -> None:
    """The resolution says which kind of match was used."""
    assert resolver.resolve("left ventricle").kind is MatchKind.CANONICAL
    assert resolver.resolve("LV").kind is MatchKind.ABBREVIATION
    assert resolver.resolve("left ventricular chamber").kind is MatchKind.SYNONYM
    assert resolver.resolve("heart.left_ventricle").kind is MatchKind.ENTITY_ID


def test_group_phrases_expand_to_members(resolver: EntityResolver) -> None:
    """A group reference resolves to the group and lists its renderable members."""
    resolution = resolver.resolve("the four chambers")
    assert resolution.entity_id == "heart.chambers"
    assert resolution.is_group is True
    assert set(resolution.targets) == {
        "heart.right_atrium",
        "heart.right_ventricle",
        "heart.left_atrium",
        "heart.left_ventricle",
    }


def test_valves_group_resolves(resolver: EntityResolver) -> None:
    """"the valves" expands to the four valves."""
    assert len(resolver.resolve("the valves").targets) == 4


def test_abbreviations_are_case_insensitive(resolver: EntityResolver) -> None:
    """Common clinical abbreviations resolve."""
    assert resolver.resolve("SVC").entity_id == "heart.superior_vena_cava"
    assert resolver.resolve("ivc").entity_id == "heart.inferior_vena_cava"
    assert resolver.resolve("SA node").entity_id == "heart.sinoatrial_node"


def test_ambiguous_abbreviation_is_not_registered(resolver: EntityResolver) -> None:
    """'AV' is not bound to the aortic valve, because it means atrioventricular."""
    with pytest.raises((UnknownEntityReferenceError, AmbiguousEntityReferenceError)):
        resolver.resolve("AV")


def test_partial_words_resolve_when_unambiguous(resolver: EntityResolver) -> None:
    """A distinctive fragment resolves; the match kind says it was partial."""
    resolution = resolver.resolve("mitral")
    assert resolution.entity_id == "heart.mitral_valve"
    assert resolver.resolve("purkinje").kind is MatchKind.PARTIAL


def test_ambiguous_reference_raises_with_candidates(resolver: EntityResolver) -> None:
    """An ambiguous word is never silently resolved to one side."""
    with pytest.raises(AmbiguousEntityReferenceError) as exc:
        resolver.resolve("ventricle")
    assert set(exc.value.candidates) == {"heart.left_ventricle", "heart.right_ventricle"}


def test_unknown_reference_raises_with_suggestions(resolver: EntityResolver) -> None:
    """An unknown structure produces a useful error, not a silent no-op."""
    with pytest.raises(UnknownEntityReferenceError) as exc:
        resolver.resolve("spleen")
    assert exc.value.phrase == "spleen"


def test_near_miss_typo_resolves_by_fuzzy_match(resolver: EntityResolver) -> None:
    """A close typo resolves, and is reported as a fuzzy match."""
    resolution = resolver.resolve("left ventrical")
    assert resolution.entity_id == "heart.left_ventricle"
    assert resolution.kind is MatchKind.FUZZY
    assert resolution.score < 1.0


def test_no_surface_form_is_ambiguous_in_the_shipped_ontology(resolver: EntityResolver) -> None:
    """No two entities share an indexed phrase."""
    assert resolver.ambiguous_forms() == {}


def test_try_resolve_returns_none_instead_of_raising(resolver: EntityResolver) -> None:
    """The non-raising variant is available for optional lookups."""
    assert resolver.try_resolve("spleen") is None
    assert resolver.try_resolve("aorta") is not None


def test_targets_for_deduplicates(resolver: EntityResolver) -> None:
    """Overlapping phrases produce each target once."""
    targets = resolver.targets_for(["the four chambers", "left ventricle"])
    assert len(targets) == len(set(targets)) == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Left Ventricle", "left ventricle"),
        ("  mitral-valve  ", "mitral valve"),
        ("the heart's chambers", "heart chambers"),
        ("aorta.", "aorta"),
    ],
)
def test_normalisation(raw: str, expected: str) -> None:
    """Normalisation is conservative and predictable."""
    assert normalize_phrase(raw) == expected
