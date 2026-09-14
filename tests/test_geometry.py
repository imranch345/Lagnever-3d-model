"""Geometry correspondence, symbolic allocation and declared-only geometry."""

from __future__ import annotations

import pytest

from awr.errors import GeometryError, NotYetImplementedError
from awr.scene import AWRScene
from awr.schema import GeometryReference, GeometryRepresentationKind
from editing.operations import AttachGeometryReferences
from editing.scene_editor import SceneEditor
from evaluation.geometry import DECLARED_GEOMETRY_METRICS, evaluate_correspondence, measure
from geometry.correspondence import (
    ComponentIdAllocator,
    CorrespondenceEntry,
    GeometryCorrespondence,
)
from geometry.decoder import (
    GeometryDecodeRequest,
    NeuralGeometryDecoder,
    SymbolicGeometryDecoder,
)
from geometry.representation import (
    GeometryComponent,
    ImplicitFieldRepresentation,
    MeshRepresentation,
    SymbolicGeometry,
)


def test_every_renderable_entity_owns_a_component(scene: AWRScene) -> None:
    """Generation reserves one geometry slot per renderable entity."""
    for entity in scene.iter_entities():
        reference = entity.geometry_reference
        if entity.renderable:
            assert reference is not None
            assert reference.kind is GeometryRepresentationKind.SYMBOLIC_PLACEHOLDER
            assert reference.is_resolved is False
        else:
            assert reference is None


def test_component_ids_are_deterministic() -> None:
    """Allocation is reproducible and zero-padded as specified."""
    allocator = ComponentIdAllocator()
    assert allocator.id_for_index(42) == "geometry_part_0042"
    assert allocator.allocate(("a", "b")) == {"a": "geometry_part_0000", "b": "geometry_part_0001"}


def test_correspondence_is_bidirectional() -> None:
    """A component resolves back to its anatomical owner."""
    correspondence = GeometryCorrespondence(
        [CorrespondenceEntry("heart.left_ventricle", "geometry_part_0042")]
    )
    assert correspondence.entity_for("geometry_part_0042") == "heart.left_ventricle"
    assert correspondence.components_for("heart.left_ventricle") == ("geometry_part_0042",)


def test_a_component_cannot_belong_to_two_entities() -> None:
    """Part identity must stay unambiguous."""
    correspondence = GeometryCorrespondence(
        [CorrespondenceEntry("heart.left_ventricle", "geometry_part_0042")]
    )
    with pytest.raises(GeometryError, match="already assigned"):
        correspondence.assign(CorrespondenceEntry("heart.aorta", "geometry_part_0042"))


def test_orphan_component_lookup_is_an_error() -> None:
    """A component with no anatomical owner is reported, not tolerated."""
    with pytest.raises(GeometryError, match="no anatomical owner"):
        GeometryCorrespondence().entity_for("geometry_part_0001")


def test_component_requires_an_entity() -> None:
    """Geometry components carry their anatomy binding as part of identity."""
    with pytest.raises(GeometryError, match="must name the anatomical entity"):
        GeometryComponent(component_id="geometry_part_0001", entity_id="")


def test_symbolic_decoder_produces_no_geometry() -> None:
    """The only implemented decoder reserves slots and says it made nothing."""
    decoder = SymbolicGeometryDecoder()
    result = decoder.decode(
        GeometryDecodeRequest(
            scene_id="s", entity_ids=("heart.left_ventricle", "heart.aorta"), lod=1
        )
    )
    assert decoder.supports_partial_decode() is True
    assert result.representation.stats()["components_with_payload"] == 0
    assert all(not component.has_payload for component in result.representation)
    assert "No geometry was produced" in result.notes[0]


def test_decoding_to_a_real_representation_is_refused() -> None:
    """Asking for a mesh fails with a pointer to Step 5."""
    with pytest.raises(NotYetImplementedError, match="Step 5"):
        SymbolicGeometryDecoder().decode(
            GeometryDecodeRequest(
                scene_id="s",
                entity_ids=("heart.aorta",),
                lod=1,
                target_kind=GeometryRepresentationKind.MESH,
            )
        )


def test_empty_decode_request_is_rejected() -> None:
    """A request must name at least one entity."""
    with pytest.raises(GeometryError, match="at least one entity"):
        GeometryDecodeRequest(scene_id="s", entity_ids=(), lod=1)


@pytest.mark.parametrize(
    "cls", [MeshRepresentation, ImplicitFieldRepresentation, NeuralGeometryDecoder]
)
def test_future_geometry_classes_are_declared_only(cls: type) -> None:
    """Future representations and decoders refuse to be constructed."""
    with pytest.raises(NotYetImplementedError):
        cls()


def test_representation_is_part_addressable() -> None:
    """A representation can always answer which entity a component belongs to."""
    representation = SymbolicGeometry(
        [
            GeometryComponent("geometry_part_0000", "heart.left_ventricle"),
            GeometryComponent("geometry_part_0001", "heart.aorta"),
        ]
    )
    assert representation.entity_ids() == ("heart.left_ventricle", "heart.aorta")
    assert representation.component("geometry_part_0001").entity_id == "heart.aorta"
    assert len(representation.components_for("heart.aorta")) == 1


def test_duplicate_component_id_is_rejected() -> None:
    """Two components cannot share an id."""
    representation = SymbolicGeometry([GeometryComponent("c0", "heart.aorta")])
    with pytest.raises(GeometryError, match="already registered"):
        representation.add(GeometryComponent("c0", "heart.left_ventricle"))


def test_correspondence_survives_edits(scene: AWRScene) -> None:
    """Geometry references are untouched by visibility and opacity edits."""
    editor = SceneEditor(scene)
    before = scene.get("heart.left_ventricle").geometry_reference
    editor.hide("heart.left_ventricle")
    editor.set_opacity(["heart.left_ventricle"], 0.3)
    assert scene.get("heart.left_ventricle").geometry_reference == before
    assert evaluate_correspondence(scene).is_valid


def test_attaching_a_shared_component_is_refused(scene: AWRScene) -> None:
    """The editing layer refuses to give one component to two entities."""
    from awr.errors import OperationError

    reference = GeometryReference(component_id="geometry_part_9999")
    with pytest.raises(OperationError, match="would be attached to both"):
        SceneEditor(scene).apply(
            AttachGeometryReferences(
                {"heart.aorta": reference, "heart.left_ventricle": reference}
            )
        )


def test_declared_geometry_metrics_refuse_to_return_numbers() -> None:
    """No geometry metric returns a placeholder score."""
    assert len(DECLARED_GEOMETRY_METRICS) >= 6
    for metric in DECLARED_GEOMETRY_METRICS:
        with pytest.raises(NotYetImplementedError):
            measure(metric.name)
    with pytest.raises(NotYetImplementedError, match="not declared"):
        measure("vibes")
