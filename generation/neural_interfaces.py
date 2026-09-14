"""Placeholders for the future Lagnav neural components.

Nothing in this module is implemented, and that is the point. The interfaces pin
down *where* each learned component plugs into the pipeline so that Step 5 can
design the models without re-litigating the architecture boundaries, and so that
no part of the deterministic prototype quietly depends on a model decision.

Pipeline position of each interface:

    text
      -> LanguageEncoder                (generation/text_encoder.py)
      -> AnatomicalReasoner             (generation/anatomical_reasoner.py)
      -> AWR  (structure / spatial / functional graphs)
      -> ThreeDLatentModel              (here)
      -> GeometryDecoder                (geometry/decoder.py)
         MaterialDecoder                (here)
         AnimationDecoder               (here)
      -> structured 3D anatomy + scene memory

Every class raises :class:`~awr.errors.NotYetImplementedError` on construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.errors import NotYetImplementedError
from awr.schema import AnimationState, LodLevel, MaterialState

__all__ = [
    "STEP5_OPEN_DECISIONS",
    "LatentRequest",
    "ThreeDLatent",
    "ThreeDLatentModel",
    "MaterialDecoder",
    "AnimationDecoder",
]

STEP5_OPEN_DECISIONS: tuple[str, ...] = (
    "model architecture",
    "tensor schema",
    "latent dimensionality",
    "training objectives",
    "attention mechanisms",
    "multimodal fusion",
    "3D representation",
    "geometry decoding strategy",
)
"""Decisions Step 5 owns. Asserted by a test so this list cannot rot silently."""


@dataclass(frozen=True, slots=True)
class LatentRequest:
    """Input to the 3D latent model.

    Carries *semantics*, not tensors: the entities to represent, the graphs that
    relate them, and the audience level. How this becomes tensors is Step 5's
    decision.
    """

    scene_id: str
    entity_ids: tuple[str, ...]
    lod: LodLevel
    language_embedding: Any | None = None
    graph_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ThreeDLatent:
    """A 3D latent representation.

    ``payload`` is intentionally untyped. The latent's shape, dimensionality and
    per-entity factorisation are exactly what Step 5 must decide; asserting a
    type here would be an arbitrary architectural choice.
    """

    scene_id: str
    entity_ids: tuple[str, ...]
    payload: Any = None
    producer: str = "undefined"

    @property
    def is_populated(self) -> bool:
        """Whether the latent actually carries data."""
        return self.payload is not None


class ThreeDLatentModel(ABC):
    """Maps an AWR plus language conditioning to a 3D latent.

    TODO(step-5): This is the central open research question of the project. It
    must produce a latent from which per-entity geometry can be decoded
    *separately*, so that regenerating one component does not disturb the others.
    """

    model_id: str = "lagnav-3d-latent-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "ThreeDLatentModel", planned_in="Step 5 (Lagnav neural architecture)"
        )

    @abstractmethod
    def encode(self, request: LatentRequest) -> ThreeDLatent:  # pragma: no cover
        """Produce a latent for the requested entities."""

    def supports_per_entity_decode(self) -> bool:  # pragma: no cover
        """Whether the latent is factorised per anatomical entity.

        Lagnav requires ``True``. It is declared here so the requirement is part
        of the interface rather than a note in a document.
        """
        return False


class MaterialDecoder(ABC):
    """Decodes material appearance for anatomical entities.

    TODO(step-5): parameterisation (PBR, tissue-specific, spectral), conditioning
    (per entity, per scene, per audience level) and supervision are undecided.
    The deterministic prototype writes semantic material state only.
    """

    decoder_id: str = "lagnav-material-decoder-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "MaterialDecoder", planned_in="Step 5 (Lagnav neural architecture)"
        )

    @abstractmethod
    def decode(
        self, latent: ThreeDLatent, entity_ids: Sequence[str]
    ) -> Mapping[str, MaterialState]:  # pragma: no cover
        """Produce material state per entity."""


class AnimationDecoder(ABC):
    """Decodes time-varying behaviour for anatomical entities.

    TODO(step-5): temporal representation (keyframes, latent trajectories, physical
    simulation parameters), how the cardiac cycle is conditioned, and how flow is
    represented geometrically are undecided. The prototype produces a semantic,
    graph-derived cycle schedule instead (see ``animation/heart_flow.py``).
    """

    decoder_id: str = "lagnav-animation-decoder-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "AnimationDecoder", planned_in="Step 5 (Lagnav neural architecture)"
        )

    @abstractmethod
    def decode(
        self, latent: ThreeDLatent, entity_ids: Sequence[str]
    ) -> Mapping[str, AnimationState]:  # pragma: no cover
        """Produce animation state per entity."""
