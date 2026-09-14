"""Language encoding interfaces.

The long-term pipeline starts with a language encoder. That encoder does not
exist yet, and this module does not pretend otherwise: the only implementation
here is lexical and symbolic. It produces **no embedding vector**, because a
fabricated vector would be worse than none - it would let downstream code look
finished while carrying no information.

What the module does fix is the contract: text in, a typed
:class:`LanguageEncoding` out, with ontology term matches attached. A neural
encoder later fills in :attr:`LanguageEncoding.embedding` and nothing downstream
changes shape.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.errors import NotYetImplementedError
from awr.ontology import AnatomyOntology

__all__ = [
    "TermMatch",
    "LanguageEncoding",
    "LanguageEncoder",
    "LexicalLanguageEncoder",
    "NeuralLanguageEncoder",
]

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class TermMatch:
    """An ontology term found in the input text."""

    entity_id: str
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class LanguageEncoding:
    """Typed result of encoding one utterance.

    ``embedding`` is ``None`` for every encoder in this milestone. Its type is
    deliberately ``Any``: Step 5 decides the tensor schema and dimensionality,
    and nothing here may pre-empt that decision.
    """

    raw_text: str
    normalized_text: str
    tokens: tuple[str, ...]
    term_matches: tuple[TermMatch, ...] = ()
    encoder_id: str = "unknown"
    embedding: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Distinct ontology entities mentioned, in order of first appearance."""
        seen: set[str] = set()
        out: list[str] = []
        for match in self.term_matches:
            if match.entity_id not in seen:
                seen.add(match.entity_id)
                out.append(match.entity_id)
        return tuple(out)

    @property
    def is_neural(self) -> bool:
        """Whether a learned representation backs this encoding."""
        return self.embedding is not None


class LanguageEncoder(ABC):
    """Interface for every Lagnav language encoder.

    TODO(step-5): Step 5 defines the neural encoder - architecture, tokenizer,
    hidden size, whether it is pretrained or trained jointly, and how its output
    is fused with the anatomical reasoner. Implementations must stay stateless
    with respect to the scene.
    """

    encoder_id: str = "abstract"

    @abstractmethod
    def encode(self, text: str) -> LanguageEncoding:
        """Encode one utterance."""

    def encode_all(self, texts: Sequence[str]) -> tuple[LanguageEncoding, ...]:
        """Encode several utterances, preserving order."""
        return tuple(self.encode(text) for text in texts)


class LexicalLanguageEncoder(LanguageEncoder):
    """Deterministic symbolic encoder: tokens plus ontology term matches.

    This is a scaffold, not a model. It exists so the generation pipeline has a
    working first stage that is fully explainable and reproducible while the real
    encoder is still an open research question.
    """

    encoder_id = "lexical-v0.1"

    def __init__(self, ontology: AnatomyOntology) -> None:
        self.ontology = ontology
        self._forms: list[tuple[str, str]] = []
        for entity in ontology.iter_entities():
            for form in entity.surface_forms():
                normalized = " ".join(form.lower().replace("_", " ").split())
                if normalized:
                    self._forms.append((normalized, entity.entity_id))
        # Longest forms first so "left ventricular chamber" wins over "left ventricle".
        self._forms.sort(key=lambda item: (-len(item[0]), item[0]))

    def encode(self, text: str) -> LanguageEncoding:
        """Tokenise and find ontology terms, without producing any embedding."""
        normalized = " ".join(text.lower().replace("_", " ").split())
        tokens = tuple(_TOKEN.findall(normalized))
        matches: list[TermMatch] = []
        claimed: list[tuple[int, int]] = []
        for form, entity_id in self._forms:
            for match in re.finditer(rf"\b{re.escape(form)}\b", normalized):
                span = (match.start(), match.end())
                if any(span[0] < end and start < span[1] for start, end in claimed):
                    continue
                claimed.append(span)
                matches.append(TermMatch(entity_id, form, span[0], span[1]))
        matches.sort(key=lambda m: m.start)
        return LanguageEncoding(
            raw_text=text,
            normalized_text=normalized,
            tokens=tokens,
            term_matches=tuple(matches),
            encoder_id=self.encoder_id,
            embedding=None,
            metadata={
                "representation": "symbolic",
                "note": "No learned representation. Step 5 defines the neural encoder.",
            },
        )


class NeuralLanguageEncoder(LanguageEncoder):
    """The Lagnav language encoder. Declared, not implemented.

    TODO(step-5): decide architecture, tensor schema, latent dimensionality,
    training objectives, attention mechanisms and multimodal fusion. Until then
    this class exists only to hold the interface in place.
    """

    encoder_id = "lagnav-language-encoder-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "NeuralLanguageEncoder", planned_in="Step 5 (Lagnav neural architecture)"
        )

    def encode(self, text: str) -> LanguageEncoding:  # pragma: no cover - construction raises
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("NeuralLanguageEncoder.encode", planned_in="Step 5")
