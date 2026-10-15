"""Level 1: the TotalSegmentator v1 label map and its fit to the thorax ontology.

The map decides which mesh belongs to which entity. A wrong id here would put the left
ventricle's shape on the right ventricle and nothing downstream could notice, so the class
ids are pinned to the dataset's published v1 class map and every entry is checked against
the ontology it names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from assets.labels import DEFAULT_LABEL_MAP, load_label_map, validate_label_map
from awr.errors import OntologyError
from awr.ontology import load_ontology
from awr.schema import Laterality

THORAX = Path("ontology/thorax")

#: From TotalSegmentator v1.5.7 ``map_to_binary.class_map["total"]``, the class map of the
#: 104-structure v1 model the v1.0 dataset was labelled with.
PUBLISHED_V1_IDS = {
    "aorta": 7,
    "inferior_vena_cava": 8,
    "lung_upper_lobe_left": 13,
    "lung_lower_lobe_left": 14,
    "lung_upper_lobe_right": 15,
    "lung_middle_lobe_right": 16,
    "lung_lower_lobe_right": 17,
    "trachea": 43,
    "heart_myocardium": 44,
    "heart_atrium_left": 45,
    "heart_ventricle_left": 46,
    "heart_atrium_right": 47,
    "heart_ventricle_right": 48,
    "pulmonary_artery": 49,
}


@pytest.fixture(scope="module")
def label_map():
    """The shipped map."""
    return load_label_map()


@pytest.fixture(scope="module")
def thorax():
    """The ontology the map targets."""
    return load_ontology(THORAX, expected_version="0.1.0")


class TestTheShippedMap:
    """What Level 1 will build meshes from."""

    def test_it_fits_the_thorax_ontology(self, label_map, thorax) -> None:
        """Every mapped entity exists, is renderable and is not a group."""
        validate_label_map(label_map, thorax)

    def test_class_ids_are_the_published_v1_ids(self, label_map) -> None:
        """Pinned, so an edit to the JSON cannot quietly renumber a structure."""
        assert dict(label_map.classes) == PUBLISHED_V1_IDS

    def test_the_license_and_attribution_travel_with_it(self, label_map) -> None:
        """ADR-L1-004: CC BY 4.0 requires attribution."""
        assert label_map.source.license == "CC BY 4.0"
        assert label_map.source.doi == "10.5281/zenodo.6802614"
        assert "TotalSegmentator" in label_map.source.attribution

    def test_fourteen_labels_are_needed(self, label_map) -> None:
        """What the subset download has to fetch per subject, and nothing more."""
        assert set(label_map.labels_needed()) == set(PUBLISHED_V1_IDS)

    def test_sides_never_cross(self, label_map, thorax) -> None:
        """A left entity never uses a right label, and a right entity never a left one.

        Unsided labels are allowed on sided entities: the IVC is a right-sided structure whose
        label carries no side.
        """
        for entry in label_map.entities:
            side = thorax.get(entry.entity_id).laterality
            if side is Laterality.LEFT:
                assert not any("right" in label for label in entry.labels), entry.entity_id
            if side is Laterality.RIGHT:
                assert not any("left" in label for label in entry.labels), entry.entity_id

    def test_shells_are_unions_of_their_parts(self, label_map, thorax) -> None:
        """An LOD-0 shell is exactly the union of its mapped children's labels."""
        for shell in ("heart", "respiratory.left_lung", "respiratory.right_lung"):
            children = [
                c for c in thorax.children(shell) if c in label_map.entity_ids()
            ]
            parts = {label for c in children for label in label_map.for_entity(c).labels}
            if shell == "heart":
                parts = {
                    label
                    for c in ("heart.myocardium", "heart.left_atrium", "heart.left_ventricle",
                              "heart.right_atrium", "heart.right_ventricle")
                    for label in label_map.for_entity(c).labels
                }
            assert set(label_map.for_entity(shell).labels) == parts, shell

    def test_vessels_that_run_into_the_abdomen_are_cropped(self, label_map) -> None:
        """The aorta and IVC labels extend far below the thorax."""
        for entity_id in ("heart.aorta", "heart.inferior_vena_cava"):
            assert label_map.for_entity(entity_id).crop == "thorax"

    def test_what_has_no_geometry_is_known(self, label_map, thorax) -> None:
        """Valves, the conduction system and the like stay symbolic (ADR-L1-002)."""
        mapped = set(label_map.entity_ids())
        renderable = {e for e in thorax.ids() if thorax.get(e).renderable}
        assert mapped <= renderable
        symbolic = renderable - mapped
        for entity_id in ("heart.mitral_valve", "heart.superior_vena_cava",
                          "heart.sinoatrial_node", "heart.left_coronary_artery"):
            assert entity_id in symbolic


def _write(tmp_path: Path, mutate) -> Path:
    raw: dict[str, Any] = json.loads(DEFAULT_LABEL_MAP.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "map.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class TestBadMapsAreRefused:
    """Each of these would attach a mesh to the wrong thing, or to nothing."""

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda raw: raw["entities"][1]["labels"].append("heart_septum"),
            lambda raw: raw["entities"][1].update(crop="abdomen"),
            lambda raw: raw["entities"].append(dict(raw["entities"][1])),
            lambda raw: raw["classes"].update(trachea=44),
            lambda raw: raw["entities"][1].update(labels=[]),
            lambda raw: raw.pop("source"),
        ],
        ids=["undeclared-label", "unknown-crop", "duplicate-entity", "duplicate-class-id",
             "no-labels", "no-source"],
    )
    def test_internally_inconsistent_maps(self, tmp_path: Path, mutate) -> None:
        """Caught when the map is read, before any ontology is involved."""
        with pytest.raises(OntologyError):
            load_label_map(_write(tmp_path, mutate))

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda raw: raw["entities"].append({"entity_id": "thorax", "labels": ["trachea"]}),
            lambda raw: raw["entities"].append({"entity_id": "heart.nowhere", "labels": ["aorta"]}),
            lambda raw: raw["target_ontology"].update(version="0.2.0"),
        ],
        ids=["group-entity", "unknown-entity", "wrong-ontology-version"],
    )
    def test_maps_that_do_not_fit_the_ontology(self, tmp_path: Path, thorax, mutate) -> None:
        """Caught by validation against the named ontology."""
        with pytest.raises(OntologyError):
            validate_label_map(load_label_map(_write(tmp_path, mutate)), thorax)
