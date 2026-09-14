"""Dataset specification for Lagnav training samples.

No data is downloaded, scraped or bundled by this module, and none should be
added without clearing the licence first. What exists here is the *shape* a
training sample must have, plus validation that refuses a sample whose provenance
or licence is unknown.

A sample is anchored to the ontology, not to a mesh: its parts are entity ids.
That is what lets several assets (images, a mesh, an annotation set) describe the
same anatomy, and what makes a correspondence between a mesh part and
``heart.left_ventricle`` checkable rather than assumed.

Fields marked "declared" describe assets the project does not have yet. They are
part of the schema so that acquisition, licensing and annotation work has a
target to fill.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from awr.errors import SchemaError
from awr.ontology import AnatomyOntology
from awr.schema import EducationalLevel, coerce_enum
from awr.validation import Severity, ValidationReport

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "LICENSING_POLICY",
    "Provenance",
    "TextAnnotation",
    "PartAnnotation",
    "ImageAsset",
    "MultiViewSet",
    "Model3DAsset",
    "FunctionalAnnotation",
    "ValidationRecord",
    "HeartSample",
    "load_sample",
    "validate_sample",
    "example_sample",
]

DATASET_SCHEMA_VERSION = "lagnav-dataset-1"

LICENSING_POLICY = (
    "Every asset must carry an explicit licence and source before it enters the dataset. "
    "Samples without provenance are rejected at load time. Nothing is scraped, and no "
    "third-party anatomical atlas, scan archive or model library is used until its terms "
    "have been reviewed and recorded here."
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an asset came from and what may be done with it."""

    source_name: str
    licence: str
    source_url: str | None = None
    licence_url: str | None = None
    attribution: str | None = None
    redistributable: bool = False
    acquired_on: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip() or not self.licence.strip():
            raise SchemaError(
                "Provenance requires both a source name and a licence. "
                f"Policy: {LICENSING_POLICY}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "source_name": self.source_name,
            "licence": self.licence,
            "source_url": self.source_url,
            "licence_url": self.licence_url,
            "attribution": self.attribution,
            "redistributable": self.redistributable,
            "acquired_on": self.acquired_on,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            source_name=str(payload["source_name"]),
            licence=str(payload["licence"]),
            source_url=payload.get("source_url"),
            licence_url=payload.get("licence_url"),
            attribution=payload.get("attribution"),
            redistributable=bool(payload.get("redistributable", False)),
            acquired_on=payload.get("acquired_on"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class TextAnnotation:
    """A natural-language request paired with the sample."""

    prompt: str
    description: str = ""
    educational_level: EducationalLevel = EducationalLevel.SCHOOL
    language: str = "en"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "prompt": self.prompt,
            "description": self.description,
            "educational_level": str(self.educational_level),
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            prompt=str(payload["prompt"]),
            description=str(payload.get("description", "")),
            educational_level=coerce_enum(
                EducationalLevel,
                str(payload.get("educational_level", "school")),
                field_name="educational_level",
            ),
            language=str(payload.get("language", "en")),
        )


@dataclass(frozen=True, slots=True)
class PartAnnotation:
    """One anatomical part of the sample, anchored to an ontology entity."""

    entity_id: str
    present: bool = True
    geometry_component_id: str | None = None
    label: str | None = None
    annotator: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "entity_id": self.entity_id,
            "present": self.present,
            "geometry_component_id": self.geometry_component_id,
            "label": self.label,
            "annotator": self.annotator,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            entity_id=str(payload["entity_id"]),
            present=bool(payload.get("present", True)),
            geometry_component_id=payload.get("geometry_component_id"),
            label=payload.get("label"),
            annotator=payload.get("annotator"),
            confidence=payload.get("confidence"),
        )


@dataclass(frozen=True, slots=True)
class ImageAsset:
    """A 2D image. Declared: the project holds no image assets yet."""

    asset_id: str
    relative_path: str
    view: str
    provenance: Provenance
    width: int | None = None
    height: int | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "view": self.view,
            "provenance": self.provenance.to_dict(),
            "width": self.width,
            "height": self.height,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            asset_id=str(payload["asset_id"]),
            relative_path=str(payload["relative_path"]),
            view=str(payload.get("view", "unspecified")),
            provenance=Provenance.from_dict(payload["provenance"]),
            width=payload.get("width"),
            height=payload.get("height"),
            checksum=payload.get("checksum"),
        )


@dataclass(frozen=True, slots=True)
class MultiViewSet:
    """A set of views of the same anatomy, with its camera convention."""

    set_id: str
    asset_ids: tuple[str, ...] = ()
    camera_convention: str = "undefined"
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "set_id": self.set_id,
            "asset_ids": list(self.asset_ids),
            "camera_convention": self.camera_convention,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            set_id=str(payload["set_id"]),
            asset_ids=tuple(payload.get("asset_ids", ())),
            camera_convention=str(payload.get("camera_convention", "undefined")),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class Model3DAsset:
    """A 3D model. Declared: the project holds no 3D assets yet."""

    asset_id: str
    relative_path: str
    file_format: str
    provenance: Provenance
    units: str = "millimetre"
    coordinate_system: str = "RAS"
    part_count: int | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "file_format": self.file_format,
            "provenance": self.provenance.to_dict(),
            "units": self.units,
            "coordinate_system": self.coordinate_system,
            "part_count": self.part_count,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            asset_id=str(payload["asset_id"]),
            relative_path=str(payload["relative_path"]),
            file_format=str(payload["file_format"]),
            provenance=Provenance.from_dict(payload["provenance"]),
            units=str(payload.get("units", "millimetre")),
            coordinate_system=str(payload.get("coordinate_system", "RAS")),
            part_count=payload.get("part_count"),
            checksum=payload.get("checksum"),
        )


@dataclass(frozen=True, slots=True)
class FunctionalAnnotation:
    """Functional and animation data attached to a sample."""

    flow_paths: tuple[tuple[str, ...], ...] = ()
    cycle_phases: tuple[str, ...] = ()
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "flow_paths": [list(path) for path in self.flow_paths],
            "cycle_phases": list(self.cycle_phases),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            flow_paths=tuple(tuple(path) for path in payload.get("flow_paths", ())),
            cycle_phases=tuple(payload.get("cycle_phases", ())),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    """Who checked the sample, and what the check concluded."""

    status: str = "unvalidated"
    reviewer: str | None = None
    reviewed_on: str | None = None
    notes: str | None = None

    @property
    def is_validated(self) -> bool:
        """Whether a reviewer signed the sample off."""
        return self.status == "validated"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "status": self.status,
            "reviewer": self.reviewer,
            "reviewed_on": self.reviewed_on,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            status=str(payload.get("status", "unvalidated")),
            reviewer=payload.get("reviewer"),
            reviewed_on=payload.get("reviewed_on"),
            notes=payload.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class HeartSample:
    """One training sample for the heart domain."""

    sample_id: str
    text: TextAnnotation
    ontology_id: str
    ontology_version: str
    root_entity_id: str
    parts: tuple[PartAnnotation, ...] = ()
    images: tuple[ImageAsset, ...] = ()
    multi_view: tuple[MultiViewSet, ...] = ()
    models_3d: tuple[Model3DAsset, ...] = ()
    functional: FunctionalAnnotation = field(default_factory=FunctionalAnnotation)
    validation: ValidationRecord = field(default_factory=ValidationRecord)
    provenance: Provenance | None = None
    schema_version: str = DATASET_SCHEMA_VERSION
    synthetic: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Ontology entities the sample annotates."""
        return tuple(part.entity_id for part in self.parts)

    @property
    def has_assets(self) -> bool:
        """Whether any binary asset is attached."""
        return bool(self.images or self.models_3d)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "synthetic": self.synthetic,
            "text": self.text.to_dict(),
            "anatomy": {
                "ontology_id": self.ontology_id,
                "ontology_version": self.ontology_version,
                "root_entity_id": self.root_entity_id,
            },
            "parts": [part.to_dict() for part in self.parts],
            "images": [image.to_dict() for image in self.images],
            "multi_view": [view.to_dict() for view in self.multi_view],
            "models_3d": [model.to_dict() for model in self.models_3d],
            "functional": self.functional.to_dict(),
            "validation": self.validation.to_dict(),
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HeartSample:
        """Rebuild from :meth:`to_dict` output."""
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != DATASET_SCHEMA_VERSION:
            raise SchemaError(
                f"Sample declares schema_version {schema_version!r}, expected "
                f"{DATASET_SCHEMA_VERSION!r}."
            )
        anatomy = payload.get("anatomy", {})
        provenance = payload.get("provenance")
        return cls(
            sample_id=str(payload["sample_id"]),
            text=TextAnnotation.from_dict(payload["text"]),
            ontology_id=str(anatomy["ontology_id"]),
            ontology_version=str(anatomy["ontology_version"]),
            root_entity_id=str(anatomy.get("root_entity_id", "heart")),
            parts=tuple(PartAnnotation.from_dict(p) for p in payload.get("parts", ())),
            images=tuple(ImageAsset.from_dict(i) for i in payload.get("images", ())),
            multi_view=tuple(MultiViewSet.from_dict(v) for v in payload.get("multi_view", ())),
            models_3d=tuple(Model3DAsset.from_dict(m) for m in payload.get("models_3d", ())),
            functional=FunctionalAnnotation.from_dict(payload.get("functional", {})),
            validation=ValidationRecord.from_dict(payload.get("validation", {})),
            provenance=Provenance.from_dict(provenance) if provenance else None,
            schema_version=schema_version,
            synthetic=bool(payload.get("synthetic", True)),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, path: str | Path) -> Path:
        """Write the sample to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target


def load_sample(path: str | Path) -> HeartSample:
    """Load one sample from JSON, rejecting unknown schema versions."""
    source = Path(path)
    if not source.is_file():
        raise SchemaError(f"Dataset sample not found: {source}")
    return HeartSample.from_dict(json.loads(source.read_text(encoding="utf-8")))


def validate_sample(
    sample: HeartSample, ontology: AnatomyOntology | None = None
) -> ValidationReport:
    """Check a sample's anatomy anchoring, licensing and asset consistency."""
    report = ValidationReport(subject=f"dataset sample {sample.sample_id}")

    if ontology is not None:
        if sample.ontology_id != ontology.ontology_id:
            report.add(
                "ONTOLOGY_MISMATCH",
                Severity.ERROR,
                f"Sample targets {sample.ontology_id!r} but was validated against "
                f"{ontology.ontology_id!r}.",
            )
        if sample.ontology_version != ontology.version:
            report.add(
                "ONTOLOGY_VERSION_MISMATCH",
                Severity.WARNING,
                f"Sample was annotated against ontology {sample.ontology_version}, "
                f"current version is {ontology.version}.",
            )
        for part in sample.parts:
            if part.entity_id not in ontology:
                report.add(
                    "UNKNOWN_ENTITY",
                    Severity.ERROR,
                    "Annotated part does not exist in the ontology.",
                    [part.entity_id],
                )

    seen: set[str] = set()
    for part in sample.parts:
        if part.entity_id in seen:
            report.add(
                "DUPLICATE_PART", Severity.ERROR, "Entity is annotated twice.", [part.entity_id]
            )
        seen.add(part.entity_id)

    components: dict[str, str] = {}
    for part in sample.parts:
        if not part.geometry_component_id:
            continue
        owner = components.get(part.geometry_component_id)
        if owner:
            report.add(
                "SHARED_COMPONENT",
                Severity.ERROR,
                f"Geometry component {part.geometry_component_id!r} is annotated for two entities.",
                [owner, part.entity_id],
            )
        components[part.geometry_component_id] = part.entity_id

    for asset_id, provenance in [
        *[(image.asset_id, image.provenance) for image in sample.images],
        *[(model.asset_id, model.provenance) for model in sample.models_3d],
    ]:
        if not provenance.licence.strip():
            report.add(
                "MISSING_LICENCE", Severity.ERROR, f"Asset {asset_id!r} has no licence recorded."
            )

    known_assets = {image.asset_id for image in sample.images}
    for view_set in sample.multi_view:
        missing = [asset_id for asset_id in view_set.asset_ids if asset_id not in known_assets]
        if missing:
            report.add(
                "DANGLING_VIEW_ASSET",
                Severity.ERROR,
                f"Multi-view set {view_set.set_id!r} references unknown assets: {missing}.",
            )

    if not sample.validation.is_validated:
        report.add(
            "SAMPLE_UNVALIDATED",
            Severity.INFO,
            "The sample has not been reviewed; it must not be treated as ground truth.",
        )
    if sample.synthetic:
        report.add(
            "SYNTHETIC_SAMPLE",
            Severity.INFO,
            "Synthetic metadata only. No real asset is attached to this sample.",
        )
    return report


def example_sample(ontology: AnatomyOntology | None = None) -> HeartSample:
    """Build the synthetic example sample used in tests and documentation.

    Entirely metadata: no image, no mesh, no downloaded file. It exists to show
    the shape a real sample must take once assets are licensed.
    """
    entity_ids: Sequence[str] = (
        ontology.ids()
        if ontology is not None
        else (
            "heart",
            "heart.right_atrium",
            "heart.right_ventricle",
            "heart.left_atrium",
            "heart.left_ventricle",
        )
    )
    parts = tuple(
        PartAnnotation(
            entity_id=entity_id,
            geometry_component_id=f"geometry_part_{index:04d}",
            annotator="synthetic",
        )
        for index, entity_id in enumerate(entity_ids)
    )
    return HeartSample(
        sample_id="heart_sample_0001",
        text=TextAnnotation(
            prompt="Generate a human heart and show the four chambers.",
            description="Whole heart with the four chambers highlighted.",
            educational_level=EducationalLevel.SCHOOL,
        ),
        ontology_id=ontology.ontology_id if ontology else "lagnav.heart",
        ontology_version=ontology.version if ontology else "0.1.0",
        root_entity_id=ontology.root_id if ontology else "heart",
        parts=parts,
        functional=FunctionalAnnotation(
            flow_paths=(
                (
                    "heart.superior_vena_cava",
                    "heart.right_atrium",
                    "heart.tricuspid_valve",
                    "heart.right_ventricle",
                ),
            ),
            cycle_phases=("atrial_systole", "ventricular_ejection"),
            notes="Derived from the functional graph, not from measurement.",
        ),
        validation=ValidationRecord(
            status="unvalidated", notes="Synthetic example; no clinical review."
        ),
        provenance=Provenance(
            source_name="Lagnav 3D R&D (synthetic)",
            licence="internal-rnd",
            redistributable=False,
            notes=LICENSING_POLICY,
        ),
        synthetic=True,
        metadata={"purpose": "schema example", "assets": "none"},
    )
