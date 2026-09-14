"""Entity resolution: natural-language phrase to persistent entity id.

The resolver is the only component allowed to turn text into identity. Every
other layer works with entity ids, which is why a future language model can
replace the parser without touching the AWR.

Resolution is staged and deterministic. Stages are tried in order and the first
one that produces a unique answer wins:

1. exact entity id (``heart.left_ventricle``)
2. exact normalised surface form (canonical name, display name, synonym, abbreviation)
3. naive morphological variant (singular / plural)
4. whole-word containment ("mitral" inside "mitral valve")
5. deterministic fuzzy match above a configured threshold (``difflib``)

Ambiguity is never resolved by guessing. A phrase that matches two entities
raises :class:`~awr.errors.AmbiguousEntityReferenceError` listing the candidates;
an unknown phrase raises :class:`~awr.errors.UnknownEntityReferenceError` with
suggestions.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from awr.config import ResolverConfig
from awr.errors import AmbiguousEntityReferenceError, UnknownEntityReferenceError
from awr.ontology import AnatomyOntology
from awr.schema import unique

__all__ = ["MatchKind", "Resolution", "EntityResolver", "normalize_phrase"]

_PUNCTUATION = re.compile(r"[.,;:!?\"'`()\[\]]+")
_WHITESPACE = re.compile(r"\s+")
_LEADING_FILLER = ("the ", "a ", "an ", "this ", "that ", "my ", "our ")
_MIN_CONTAINMENT_LENGTH = 4
"""Shortest single token allowed to match by containment; below this an exact match is required."""
_TRAILING_FILLER = (" please", " now")


def normalize_phrase(phrase: str) -> str:
    """Normalise a phrase for lookup.

    Lowercases, removes punctuation, converts separators to spaces, collapses
    whitespace and strips leading articles. Deliberately conservative: it never
    drops anatomical words, because dropping a word can silently change which
    structure is meant.
    """
    text = phrase.strip().lower().replace("_", " ").replace("-", " ")
    text = text.replace("'s ", " ").removesuffix("'s")
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    changed = True
    while changed:
        changed = False
        for filler in _LEADING_FILLER:
            if text.startswith(filler):
                text = text[len(filler) :]
                changed = True
        for filler in _TRAILING_FILLER:
            if text.endswith(filler):
                text = text[: -len(filler)]
                changed = True
        text = text.strip()
    return text


def _morphological_variants(text: str) -> tuple[str, ...]:
    """Naive singular/plural variants of a normalised phrase."""
    variants: list[str] = []
    if text.endswith("ies"):
        variants.append(f"{text[:-3]}y")
    if text.endswith("es"):
        variants.append(text[:-2])
    if text.endswith("s") and not text.endswith("ss"):
        variants.append(text[:-1])
    variants.extend((f"{text}s", f"{text}es"))
    if text.endswith("y"):
        variants.append(f"{text[:-1]}ies")
    return unique(variant for variant in variants if variant and variant != text)


class MatchKind(StrEnum):
    """How a phrase was matched to an entity."""

    ENTITY_ID = "entity_id"
    CANONICAL = "canonical"
    DISPLAY_NAME = "display_name"
    SYNONYM = "synonym"
    ABBREVIATION = "abbreviation"
    MORPHOLOGICAL = "morphological"
    PARTIAL = "partial"
    FUZZY = "fuzzy"


@dataclass(frozen=True, slots=True)
class Resolution:
    """The result of resolving one phrase.

    ``entity_id`` is the persistent id of the matched entity. When the match is
    an organisational group, ``members`` holds the renderable entities the group
    expands to; callers that change state should act on ``targets``.
    """

    phrase: str
    normalized: str
    entity_id: str
    display_name: str
    kind: MatchKind
    score: float
    is_group: bool = False
    members: tuple[str, ...] = ()
    matched_form: str = ""

    @property
    def targets(self) -> tuple[str, ...]:
        """Entities an operation should act on (group members, or the entity itself)."""
        return self.members if self.is_group else (self.entity_id,)

    def describe(self) -> str:
        """One-line human-readable description."""
        suffix = f" -> {len(self.members)} members" if self.is_group else ""
        return f"{self.phrase!r} -> {self.entity_id} ({self.kind}, {self.score:.2f}){suffix}"


@dataclass(slots=True)
class EntityResolver:
    """Deterministic phrase-to-entity resolver backed by ontology synonyms."""

    ontology: AnatomyOntology
    config: ResolverConfig = field(default_factory=ResolverConfig)
    _index: dict[str, dict[str, MatchKind]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _register(self, form: str, entity_id: str, kind: MatchKind) -> None:
        key = normalize_phrase(form)
        if not key:
            return
        owners = self._index.setdefault(key, {})
        # A stronger match kind wins for the same entity; different entities stay ambiguous.
        owners.setdefault(entity_id, kind)

    def _build_index(self) -> None:
        self._index.clear()
        for entity in self.ontology.iter_entities():
            self._register(entity.entity_id, entity.entity_id, MatchKind.ENTITY_ID)
            self._register(entity.canonical_name, entity.entity_id, MatchKind.CANONICAL)
            self._register(entity.display_name, entity.entity_id, MatchKind.DISPLAY_NAME)
            for synonym in entity.synonyms:
                self._register(synonym, entity.entity_id, MatchKind.SYNONYM)
            for abbreviation in entity.abbreviations:
                self._register(abbreviation, entity.entity_id, MatchKind.ABBREVIATION)

    # ------------------------------------------------------------------
    def known_forms(self) -> tuple[str, ...]:
        """Every indexed surface form, sorted. Useful for diagnostics and tests."""
        return tuple(sorted(self._index))

    def ambiguous_forms(self) -> dict[str, tuple[str, ...]]:
        """Forms that map to more than one entity."""
        return {
            form: tuple(sorted(owners))
            for form, owners in self._index.items()
            if len(owners) > 1
        }

    def suggest(self, phrase: str, limit: int | None = None) -> tuple[str, ...]:
        """Closest known forms to ``phrase``, for error messages."""
        normalized = normalize_phrase(phrase)
        count = limit if limit is not None else self.config.max_suggestions
        return tuple(
            difflib.get_close_matches(normalized, self._index.keys(), n=count, cutoff=0.45)
        )

    def resolve(self, phrase: str) -> Resolution:
        """Resolve a phrase to a persistent entity id.

        Raises:
            UnknownEntityReferenceError: No entity matches.
            AmbiguousEntityReferenceError: Several entities match equally well.

        """
        normalized = normalize_phrase(phrase)
        if not normalized:
            raise UnknownEntityReferenceError(phrase, self.suggest("heart"))

        exact = self._index.get(normalized)
        if exact:
            return self._build(phrase, normalized, exact, normalized, score=1.0)

        for variant in _morphological_variants(normalized):
            owners = self._index.get(variant)
            if owners:
                return self._build(
                    phrase,
                    normalized,
                    owners,
                    variant,
                    score=0.95,
                    override_kind=MatchKind.MORPHOLOGICAL,
                )

        contained = self._containment_candidates(normalized)
        if contained:
            owners, matched_form = contained
            return self._build(
                phrase,
                normalized,
                owners,
                matched_form,
                score=0.9,
                override_kind=MatchKind.PARTIAL,
            )

        close = difflib.get_close_matches(
            normalized, self._index.keys(), n=3, cutoff=self.config.fuzzy_threshold
        )
        if close:
            candidates: dict[str, MatchKind] = {}
            best_form = close[0]
            for form in close:
                candidates.update(self._index[form])
            score = difflib.SequenceMatcher(None, normalized, best_form).ratio()
            return self._build(
                phrase,
                normalized,
                candidates,
                best_form,
                score=score,
                override_kind=MatchKind.FUZZY,
            )

        raise UnknownEntityReferenceError(phrase, self.suggest(phrase))

    def _containment_candidates(
        self, normalized: str
    ) -> tuple[dict[str, MatchKind], str] | None:
        """Forms that contain the phrase as a whole word.

        Lets "mitral" reach the mitral valve, while a genuinely ambiguous word
        such as "ventricle" collects both ventricles and is reported as
        ambiguous rather than guessed.

        Short single tokens are excluded. They are abbreviation-shaped, and an
        abbreviation must match the ontology exactly: guessing that "AV" means
        the atrioventricular node when the speaker may have meant the aortic
        valve is exactly the silent error this project must not make.
        """
        if len(normalized) < _MIN_CONTAINMENT_LENGTH and " " not in normalized:
            return None
        pattern = re.compile(rf"\b{re.escape(normalized)}\b")
        owners: dict[str, MatchKind] = {}
        matched_form = ""
        for form, form_owners in sorted(self._index.items()):
            if pattern.search(form):
                if not matched_form:
                    matched_form = form
                for entity_id, kind in form_owners.items():
                    owners.setdefault(entity_id, kind)
        return (owners, matched_form) if owners else None

    def _build(
        self,
        phrase: str,
        normalized: str,
        owners: dict[str, MatchKind],
        matched_form: str,
        *,
        score: float,
        override_kind: MatchKind | None = None,
    ) -> Resolution:
        if len(owners) > 1 and not self.config.allow_ambiguous_autopick:
            raise AmbiguousEntityReferenceError(phrase, sorted(owners))
        entity_id, kind = sorted(owners.items())[0]
        entity = self.ontology.get(entity_id)
        return Resolution(
            phrase=phrase,
            normalized=normalized,
            entity_id=entity_id,
            display_name=entity.display_name,
            kind=override_kind or kind,
            score=score,
            is_group=entity.is_group,
            members=(
                self.ontology.renderable_descendants(entity_id, include_self=False)
                if entity.is_group
                else ()
            ),
            matched_form=matched_form,
        )

    def try_resolve(self, phrase: str) -> Resolution | None:
        """Resolve, returning ``None`` instead of raising when unresolvable."""
        try:
            return self.resolve(phrase)
        except (UnknownEntityReferenceError, AmbiguousEntityReferenceError):
            return None

    def resolve_all(self, phrases: Iterable[str]) -> tuple[Resolution, ...]:
        """Resolve several phrases, preserving order."""
        return tuple(self.resolve(phrase) for phrase in phrases)

    def targets_for(self, phrases: Sequence[str]) -> tuple[str, ...]:
        """Resolve phrases and flatten to the entity ids an operation should touch."""
        out: list[str] = []
        seen: set[str] = set()
        for resolution in self.resolve_all(phrases):
            for target in resolution.targets:
                if target not in seen:
                    seen.add(target)
                    out.append(target)
        return tuple(out)
