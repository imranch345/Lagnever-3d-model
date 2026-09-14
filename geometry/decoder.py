"""Geometry decoder interfaces.

The geometry decoder turns a 3D latent representation into structured anatomy.
That model does not exist yet and is not designed here. What this module fixes is
the contract every future decoder must honour:

1. A decode request names *entities*, not vertices.
2. A decode result returns a part-addressable representation plus a
   correspondence map, so anatomy and geometry stay linked.
3. A decoder may be asked to regenerate a **single** component, because the
   editing model requires that "regenerate the mitral valve" never rebuilds the
   whole scene.

:class:`SymbolicGeometryDecoder` is the only implementation in this milestone. It
allocates stable component slots and produces **no geometry at all**. That is a
deliberate choice over a random mesh generator, which would produce output that
looks like progress while being anatomically meaningless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from awr.errors import GeometryError, NotYetImplementedError
from awr.schema import GeometryRepresentationKind, LodLevel
from geometry.correspondence import (
    ComponentIdAllocator,
    CorrespondenceEntry,
    GeometryCorrespondence,
)
from geometry.representation import (
    Geometry3DRepresentation,
    GeometryComponent,
    SymbolicGeometry,
)

__all__ = [
    "GeometryDecodeRequest",
    "GeometryDecodeResult",
    "GeometryDecoder",
    "SymbolicGeometryDecoder",
    "NeuralGeometryDecoder",
]


@dataclass(frozen=True, slots=True)
class GeometryDecodeRequest:
    """What a caller asks a geometry decoder to produce."""

    scene_id: str
    entity_ids: tuple[str, ...]
    lod: LodLevel
    target_kind: GeometryRepresentationKind = GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER
    latent: Any | None = None
    """Opaque 3D latent. Its type and shape are defined in Step 5, not here."""
    partial: bool = False
    """True when only the listed entities are being regenerated."""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_ids:
            raise GeometryError("A decode request must name at least one entity.")


@dataclass(frozen=True, slots=True)
class GeometryDecodeResult:
    """What a decoder returns: geometry plus its anatomical correspondence."""

    representation: Geometry3DRepresentation
    correspondence: GeometryCorrespondence
    decoder_id: str
    notes: tuple[str, ...] = ()

    def references(self) -> dict[str, Any]:
        """Geometry references to store on each entity, keyed by entity id."""
        out: dict[str, Any] = {}
        for entry in self.correspondence.entries():
            out[entry.entity_id] = entry.to_reference(producer=self.decoder_id)
        return out

    def summary(self) -> dict[str, Any]:
        """Compact description for reports and tests."""
        return {
            "decoder": self.decoder_id,
            "representation": self.representation.stats(),
            "links": len(self.correspondence),
            "notes": list(self.notes),
        }


class GeometryDecoder(ABC):
    """Abstract geometry decoder.

    TODO(step-5): Step 5 defines the decoder architecture, the latent it consumes,
    its output representation, the training objectives, and how per-part decoding
    is conditioned. Implementations must keep :meth:`decode` side-effect free with
    respect to the AWR: they return geometry, they never mutate anatomy.
    """

    decoder_id: str = "abstract"

    @abstractmethod
    def decode(self, request: GeometryDecodeRequest) -> GeometryDecodeResult:
        """Produce geometry for the requested entities."""

    def supports_partial_decode(self) -> bool:
        """Whether single components can be regenerated without a full rebuild.

        Lagnav requires this to be ``True`` for any production decoder; it is
        declared here so the requirement is visible in the interface.
        """
        return False


class SymbolicGeometryDecoder(GeometryDecoder):
    """Allocates stable component slots. Produces no geometry.

    Use this to exercise and test the correspondence layer before a real decoder
    exists. Every component it returns has
    ``kind == SYMBOLIC_PLACEHOLDER`` and no payload, which keeps the prototype
    honest about what it has and has not built.
    """

    decoder_id = "symbolic-allocator-v0.1"

    def __init__(self, allocator: ComponentIdAllocator | None = None) -> None:
        self.allocator = allocator or ComponentIdAllocator()

    def decode(self, request: GeometryDecodeRequest) -> GeometryDecodeResult:
        """Allocate one component slot per requested entity, in request order."""
        if request.target_kind is not GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER:
            raise NotYetImplementedError(
                f"Decoding to {request.target_kind}",
                planned_in="Step 5 (Lagnav neural architecture)",
            )
        representation = SymbolicGeometry()
        correspondence = GeometryCorrespondence()
        for index, entity_id in enumerate(request.entity_ids):
            component_id = self.allocator.id_for_index(index)
            representation.add(
                GeometryComponent(
                    component_id=component_id,
                    entity_id=entity_id,
                    kind=GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER,
                    lod=request.lod,
                    stats={"payload": "none"},
                )
            )
            correspondence.assign(
                CorrespondenceEntry(
                    entity_id=entity_id,
                    component_id=component_id,
                    kind=GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER,
                    lod=request.lod,
                )
            )
        return GeometryDecodeResult(
            representation=representation,
            correspondence=correspondence,
            decoder_id=self.decoder_id,
            notes=(
                "No geometry was produced. Component slots reserve stable ids so that a future "
                "decoder can fill them without changing anatomical identity.",
            ),
        )

    def supports_partial_decode(self) -> bool:
        """Slot allocation is per entity, so partial decoding is trivially supported."""
        return True


class NeuralGeometryDecoder(GeometryDecoder):
    """The Lagnav geometry decoder. Declared, not implemented.

    TODO(step-5): architecture, tensor schema, latent dimensionality, training
    objectives, attention mechanisms, multimodal fusion, 3D representation and
    geometry decoding strategy are all open. Nothing in this repository may
    assume any of them.
    """

    decoder_id = "lagnav-neural-geometry-decoder-undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotYetImplementedError(
            "NeuralGeometryDecoder", planned_in="Step 5 (Lagnav neural architecture)"
        )

    def decode(self, request: GeometryDecodeRequest) -> GeometryDecodeResult:  # pragma: no cover
        """Unreachable: construction always raises."""
        raise NotYetImplementedError("NeuralGeometryDecoder.decode", planned_in="Step 5")


def allocate_scene_components(
    entity_ids: Sequence[str], *, allocator: ComponentIdAllocator | None = None
) -> GeometryCorrespondence:
    """Convenience helper: allocate one component per entity, deterministically."""
    active = allocator or ComponentIdAllocator()
    mapping = active.allocate(entity_ids)
    return GeometryCorrespondence(
        CorrespondenceEntry(entity_id=entity_id, component_id=component_id)
        for entity_id, component_id in mapping.items()
    )
