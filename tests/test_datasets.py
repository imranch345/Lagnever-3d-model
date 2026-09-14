"""Dataset schema: anchoring to the ontology and licence enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from awr.errors import SchemaError
from awr.ontology import AnatomyOntology
from awr.paths import datasets_dir, repo_root
from datasets.schema import (
    DATASET_SCHEMA_VERSION,
    HeartSample,
    ImageAsset,
    PartAnnotation,
    Provenance,
    example_sample,
    load_sample,
    validate_sample,
)


def test_example_sample_validates(ontology: AnatomyOntology) -> None:
    """The synthetic example passes validation and is flagged as synthetic."""
    sample = example_sample(ontology)
    report = validate_sample(sample, ontology)
    assert report.is_valid, str(report)
    assert any(issue.code == "SYNTHETIC_SAMPLE" for issue in report.issues)
    assert any(issue.code == "SAMPLE_UNVALIDATED" for issue in report.issues)


def test_committed_example_file_loads(ontology: AnatomyOntology) -> None:
    """The checked-in annotation file matches the current schema."""
    path = datasets_dir() / "annotations" / "heart_sample_0001.json"
    sample = load_sample(path)
    assert sample.schema_version == DATASET_SCHEMA_VERSION
    assert sample.synthetic is True
    assert sample.has_assets is False
    assert validate_sample(sample, ontology).is_valid


def test_sample_parts_are_ontology_entities(ontology: AnatomyOntology) -> None:
    """Parts are entity ids, so anatomy and assets can be cross-checked."""
    sample = example_sample(ontology)
    assert set(sample.entity_ids) <= set(ontology.ids())


def test_unknown_entity_in_a_sample_is_reported(ontology: AnatomyOntology) -> None:
    """A part naming an entity outside the ontology is an error."""
    sample = example_sample(ontology)
    broken = HeartSample(
        sample_id=sample.sample_id,
        text=sample.text,
        ontology_id=sample.ontology_id,
        ontology_version=sample.ontology_version,
        root_entity_id=sample.root_entity_id,
        parts=(*sample.parts, PartAnnotation(entity_id="heart.spleen")),
    )
    report = validate_sample(broken, ontology)
    assert any(issue.code == "UNKNOWN_ENTITY" for issue in report.errors)


def test_assets_without_a_licence_cannot_be_constructed() -> None:
    """Provenance requires a source and a licence."""
    with pytest.raises(SchemaError, match="licence"):
        Provenance(source_name="somewhere", licence="")


def test_image_asset_requires_provenance() -> None:
    """Every asset carries its provenance by construction."""
    asset = ImageAsset(
        asset_id="img-1",
        relative_path="raw/img-1.png",
        view="anterior",
        provenance=Provenance(source_name="synthetic", licence="internal-rnd"),
    )
    assert asset.provenance.redistributable is False


def test_shared_geometry_component_in_a_sample_is_reported(ontology: AnatomyOntology) -> None:
    """Two parts must not annotate the same geometry component."""
    broken = HeartSample(
        sample_id="s",
        text=example_sample(ontology).text,
        ontology_id=ontology.ontology_id,
        ontology_version=ontology.version,
        root_entity_id=ontology.root_id,
        parts=(
            PartAnnotation("heart.aorta", geometry_component_id="geometry_part_0000"),
            PartAnnotation("heart.left_ventricle", geometry_component_id="geometry_part_0000"),
        ),
    )
    report = validate_sample(broken, ontology)
    assert any(issue.code == "SHARED_COMPONENT" for issue in report.errors)


def test_sample_round_trips(ontology: AnatomyOntology, tmp_path: Path) -> None:
    """A sample survives a save and load cycle."""
    sample = example_sample(ontology)
    restored = load_sample(sample.save(tmp_path / "sample.json"))
    assert restored.to_dict() == sample.to_dict()


def test_unknown_schema_version_is_refused(ontology: AnatomyOntology, tmp_path: Path) -> None:
    """A sample from a different schema version is not silently accepted."""
    payload = example_sample(ontology).to_dict()
    payload["schema_version"] = "lagnav-dataset-99"
    path = tmp_path / "bad.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    with pytest.raises(SchemaError, match="schema_version"):
        load_sample(path)


def test_no_licensed_assets_are_acquired() -> None:
    """Nothing has been downloaded: the raw directory stays empty."""
    contents = [p.name for p in (datasets_dir() / "raw").iterdir() if p.name != ".gitkeep"]
    assert contents == []


def test_processed_holds_only_generated_text_data() -> None:
    """Step 6 generates a synthetic corpus here. It must be text, and it must be ignored by git.

    Stricter than the pre-Step-6 emptiness check it replaces: the corpus is allowed,
    binary blobs and licensed assets are still not.
    """
    allowed = {".json", ".jsonl", ".gitkeep"}
    for path in (datasets_dir() / "processed").rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix or path.name
        assert suffix in allowed, f"unexpected file in datasets/processed: {path.name}"

    ignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "datasets/processed/*" in ignore
    assert "datasets/raw/*" in ignore
