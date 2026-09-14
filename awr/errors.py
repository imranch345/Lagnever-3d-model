"""Exception hierarchy for Lagnav 3D.

Every failure mode in the prototype raises a typed error carrying enough context
for a caller to explain what went wrong. Silent corruption is never acceptable:
a command that cannot be resolved must fail loudly rather than mutate the scene
partially.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "LagnavError",
    "ConfigError",
    "OntologyError",
    "SchemaError",
    "InvalidStateError",
    "EntityNotFoundError",
    "UnknownEntityReferenceError",
    "AmbiguousEntityReferenceError",
    "RelationshipError",
    "UnknownRelationTypeError",
    "SceneError",
    "OperationError",
    "CommandError",
    "UnsupportedCommandError",
    "GeometryError",
    "ContractError",
    "NotYetImplementedError",
]


class LagnavError(Exception):
    """Base class for every Lagnav 3D error."""


class ConfigError(LagnavError):
    """Configuration is missing, unreadable or internally inconsistent."""


class OntologyError(LagnavError):
    """Ontology data is missing, malformed or self-inconsistent."""


class SchemaError(LagnavError):
    """A serialised payload does not match the expected AWR schema."""


class InvalidStateError(LagnavError):
    """An entity or scene field was given a value outside its allowed domain."""


class EntityNotFoundError(LagnavError):
    """An entity id is not present in the scene or ontology."""

    def __init__(self, entity_id: str, *, available: int | None = None) -> None:
        self.entity_id = entity_id
        detail = f" (scene holds {available} entities)" if available is not None else ""
        super().__init__(f"Unknown entity id {entity_id!r}{detail}.")


class UnknownEntityReferenceError(LagnavError):
    """A natural-language phrase could not be resolved to any known entity."""

    def __init__(self, phrase: str, suggestions: Sequence[str] = ()) -> None:
        self.phrase = phrase
        self.suggestions = tuple(suggestions)
        hint = f" Did you mean: {', '.join(self.suggestions)}?" if self.suggestions else ""
        super().__init__(f"No anatomical entity matches {phrase!r}.{hint}")


class AmbiguousEntityReferenceError(LagnavError):
    """A phrase matches several entities and the prototype refuses to guess."""

    def __init__(self, phrase: str, candidates: Sequence[str]) -> None:
        self.phrase = phrase
        self.candidates = tuple(candidates)
        super().__init__(
            f"The reference {phrase!r} is ambiguous between: {', '.join(self.candidates)}. "
            "Please be more specific."
        )


class RelationshipError(LagnavError):
    """A relationship edge is invalid or refers to a missing entity."""


class UnknownRelationTypeError(RelationshipError):
    """A relation name is not registered in the relation-type registry."""

    def __init__(self, relation: str, known: Sequence[str] = ()) -> None:
        self.relation = relation
        self.known = tuple(known)
        hint = f" Registered types: {', '.join(sorted(self.known))}." if self.known else ""
        super().__init__(
            f"Relation type {relation!r} is not registered. Register it in "
            f"awr/relationships.py before using it in ontology data.{hint}"
        )


class SceneError(LagnavError):
    """The scene is in, or would be put into, an invalid state."""


class OperationError(LagnavError):
    """An editing operation could not be applied."""


class CommandError(LagnavError):
    """A natural-language command could not be parsed or executed."""


class UnsupportedCommandError(CommandError):
    """The command was understood but the prototype does not implement it yet."""

    def __init__(self, message: str, *, planned_in: str | None = None) -> None:
        self.planned_in = planned_in
        suffix = f" Planned in: {planned_in}." if planned_in else ""
        super().__init__(f"{message}{suffix}")


class GeometryError(LagnavError):
    """Geometry correspondence or geometry reference is invalid."""


class ContractError(LagnavError):
    """A tensor or data contract was violated.

    Raised when a declared representation does not match its specification:
    wrong rank, wrong dimension size, wrong dtype, or a dimension bound to two
    different sizes in one batch.
    """


class NotYetImplementedError(LagnavError):
    """A deliberately declared but unimplemented research component was invoked.

    Raised instead of the builtin ``NotImplementedError`` so that callers can
    distinguish "this is a planned Lagnav component" from a genuine bug.
    """

    def __init__(self, component: str, *, planned_in: str = "a later R&D step") -> None:
        self.component = component
        self.planned_in = planned_in
        super().__init__(
            f"{component} is declared but not implemented yet; it will be defined in "
            f"{planned_in}. This is intentional in the current prototype."
        )
