"""Geometry layer: representation, correspondence and decoder interfaces.

No geometry is generated in this milestone. The layer exists so that anatomical
entities can own stable geometry component references from day one, and so that
the future decoder has a contract it must satisfy.
"""

from geometry.correspondence import (
    ComponentIdAllocator,
    CorrespondenceEntry,
    GeometryCorrespondence,
)
from geometry.decoder import (
    GeometryDecoder,
    GeometryDecodeRequest,
    GeometryDecodeResult,
    NeuralGeometryDecoder,
    SymbolicGeometryDecoder,
    allocate_scene_components,
)
from geometry.representation import (
    Geometry3DRepresentation,
    GeometryComponent,
    ImplicitFieldRepresentation,
    MeshRepresentation,
    SymbolicGeometry,
    ThreeDRepresentation,
)

__all__ = [
    "ComponentIdAllocator",
    "CorrespondenceEntry",
    "GeometryCorrespondence",
    "GeometryDecodeRequest",
    "GeometryDecodeResult",
    "GeometryDecoder",
    "NeuralGeometryDecoder",
    "SymbolicGeometryDecoder",
    "allocate_scene_components",
    "Geometry3DRepresentation",
    "GeometryComponent",
    "ImplicitFieldRepresentation",
    "MeshRepresentation",
    "SymbolicGeometry",
    "ThreeDRepresentation",
]
