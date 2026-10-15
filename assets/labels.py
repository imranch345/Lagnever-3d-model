"""The map from dataset labels to AWR entities, and its validation against an ontology.

A label map is data, not code: ``assets/labels/*.json``. It names a source dataset (with its
license, which travels with every derived mesh -- ADR-L1-004), the target ontology by id and
version, the dataset's class ids, and for each entity the labels whose union is its mesh.

Validation is strict for the same reason the ontology loader is: a typo here would attach a
mesh to the wrong entity, or silently give an entity none, and nothing downstream could tell.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from awr.errors import OntologyError
from awr.ontology import AnatomyOntology

__all__ = [
    "DEFAULT_LABEL_MAP",
    "EntityLabels",
    "LabelMap",
    "LabelSource",
    "load_label_map",
    "validate_label_map",
]

DEFAULT_LABEL_MAP = Path(__file__).parent / "labels" / "totalsegmentator_v1.json"


@dataclass(frozen=True, slots=True)
class LabelSource:
    """Where the labels come from, and the terms they come under."""

    dataset: str
    version: str
    doi: str
    url: str
    license: str
    attribution: str


@dataclass(frozen=True, slots=True)
class EntityLabels:
    """The labels whose union becomes one entity's mesh."""

    entity_id: str
    labels: tuple[str, ...]
    crop: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class LabelMap:
    """A validated-shape label map; see :func:`validate_label_map` for the ontology check."""

    label_map_id: str
    version: str
    source: LabelSource
    ontology_id: str
    ontology_version: str
    classes: Mapping[str, int]
    crops: Mapping[str, str]
    entities: tuple[EntityLabels, ...]

    def for_entity(self, entity_id: str) -> EntityLabels:
        """The entry for one entity, or ``KeyError``."""
        for entry in self.entities:
            if entry.entity_id == entity_id:
                return entry
        raise KeyError(entity_id)

    def entity_ids(self) -> tuple[str, ...]:
        """Every entity that gets a mesh, in declaration order."""
        return tuple(entry.entity_id for entry in self.entities)

    def labels_needed(self) -> tuple[str, ...]:
        """Every dataset label any entity uses, sorted -- what has to be fetched."""
        return tuple(sorted({label for entry in self.entities for label in entry.labels}))


def _require(raw: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in raw:
        raise OntologyError(f"Label map {where} is missing {key!r}.")
    return raw[key]


def load_label_map(path: str | Path = DEFAULT_LABEL_MAP) -> LabelMap:
    """Read a label map and check its internal consistency.

    Raises:
        OntologyError: on a missing field, a label used but not declared among ``classes``,
            a duplicate class id or entity, an unknown crop, or an entity with no labels.

    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    source = _require(raw, "source", "")
    target = _require(raw, "target_ontology", "")
    classes = {str(k): int(v) for k, v in _require(raw, "classes", "").items()}
    crops = {str(k): str(v) for k, v in raw.get("crops", {}).items()}

    ids = list(classes.values())
    if len(set(ids)) != len(ids):
        raise OntologyError("Label map declares the same class id twice.")

    entities: list[EntityLabels] = []
    seen: set[str] = set()
    for index, entry in enumerate(_require(raw, "entities", "")):
        entity_id = str(_require(entry, "entity_id", f"entity #{index}"))
        labels = tuple(str(label) for label in _require(entry, "labels", entity_id))
        if not labels:
            raise OntologyError(f"Label map entity {entity_id!r} has no labels.")
        unknown = [label for label in labels if label not in classes]
        if unknown:
            raise OntologyError(f"Label map entity {entity_id!r} uses undeclared {unknown}.")
        crop = entry.get("crop")
        if crop is not None and crop not in crops:
            raise OntologyError(f"Label map entity {entity_id!r} uses unknown crop {crop!r}.")
        if entity_id in seen:
            raise OntologyError(f"Label map lists entity {entity_id!r} twice.")
        seen.add(entity_id)
        entities.append(EntityLabels(entity_id, labels, crop, entry.get("note")))

    return LabelMap(
        label_map_id=str(_require(raw, "label_map_id", "")),
        version=str(_require(raw, "version", "")),
        source=LabelSource(
            dataset=str(_require(source, "dataset", "source")),
            version=str(_require(source, "version", "source")),
            doi=str(_require(source, "doi", "source")),
            url=str(_require(source, "url", "source")),
            license=str(_require(source, "license", "source")),
            attribution=str(_require(source, "attribution", "source")),
        ),
        ontology_id=str(_require(target, "ontology_id", "target_ontology")),
        ontology_version=str(_require(target, "version", "target_ontology")),
        classes=classes,
        crops=crops,
        entities=tuple(entities),
    )


def validate_label_map(label_map: LabelMap, ontology: AnatomyOntology) -> None:
    """Check the map against the ontology it names.

    Every mapped entity must exist, be renderable and not be an organisational group, and the
    ontology must be the one the map was written for.

    Raises:
        OntologyError: naming every problem found, not only the first.

    """
    problems: list[str] = []
    if (ontology.ontology_id, ontology.version) != (
        label_map.ontology_id,
        label_map.ontology_version,
    ):
        problems.append(
            f"map targets {label_map.ontology_id}@{label_map.ontology_version}, "
            f"ontology is {ontology.ontology_id}@{ontology.version}"
        )
    for entry in label_map.entities:
        if entry.entity_id not in ontology:
            problems.append(f"{entry.entity_id!r} is not in the ontology")
            continue
        entity = ontology.get(entry.entity_id)
        if entity.is_group or not entity.renderable:
            problems.append(f"{entry.entity_id!r} is a group or not renderable")
    if problems:
        raise OntologyError("Label map does not fit the ontology: " + "; ".join(problems) + ".")
