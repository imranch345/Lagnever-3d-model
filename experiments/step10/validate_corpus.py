"""Step 10 Change 2 §14–§16: everything the rotated corpus must pass before training.

One entry point, one JSON report, and a non-zero exit on the first failure. Nothing here
trains anything or looks at a model; it is about the data and the metadata.

What is checked
---------------

``manifest``
    The files on disk hold what the manifest says, and the manifest records what §13
    requires: parentage, generator version and commit, seed, rotation flag.

``pairing``
    Scene for scene against the parent corpus: same scene ids, same centroids, same
    relationship graphs, same presence, same levels of detail. This is the evidence that
    rotation is the only intended change; the extents are expected to differ and are
    reported as the one consequence of it.

``rotations``
    Every stored rotation is the generator's, compared against each scene's own parameters.
    This is the check that catches a rotation which is well formed but wrong.

``placement``
    The §27 gate on a batch from every split: parent table against construction, slot table,
    composition order, stored bases, and the round trip — now informative on stored frames,
    because the rotations are real.

``levels``
    Levels of detail nest, and every scene carries all twenty entities, so presence cannot
    carry the arrangement.

``splits``
    Every scene classifies into the region its split is meant to draw from, by the same
    function the hold-out is defined with.

``holdout``
    Which rotation bands each split occupies, and that the training band excludes the
    interpolation hole and the extrapolation edge the test splits live in.

    python -m experiments.step10.validate_corpus --corpus datasets/processed/step10_rotated
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import (
    YAW_EXTRAPOLATION_EDGE,
    YAW_INTERPOLATION_HOLE,
    classify_arrangement,
)
from datasets.whole_organ.continuous_corpus import (
    STEP8_SPLITS,
    load_step8_manifest,
    load_step8_split,
    split_regions,
)
from datasets.whole_organ.corpus import LOD_ENTITIES
from datasets.whole_organ.field import WHOLE_ORGAN_ENTITIES
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.placement_integrity import (
    PlacementIntegrityError,
    check_corpus_rotations,
    validate_placement,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = ["validate_corpus", "main"]

#: Which arrangement region each split must classify into.
EXPECTED_REGION = {
    "train": "in_distribution",
    "validation": "in_distribution",
    "test_seen": "in_distribution",
    "test_arrangement": "test_arrangement",
    "test_transform": "test_transform",
    "test_combination": "test_combination",
}


class CorpusValidationError(ValueError):
    """The corpus is not what its manifest says, or not what training requires."""


def _check_manifest(corpus_dir: Path) -> dict[str, Any]:
    manifest = load_step8_manifest(corpus_dir)
    counted = {
        split: sum(1 for _ in (corpus_dir / f"{split}.jsonl").open(encoding="utf-8"))
        for split in STEP8_SPLITS
    }
    if counted != dict(manifest.split_counts):
        raise CorpusValidationError(
            f"the files hold {counted} scenes but the manifest claims {dict(manifest.split_counts)}"
        )
    if not manifest.rotation_enabled:
        raise CorpusValidationError("rotation_enabled is false; this is not the rotated corpus")
    missing = [
        name
        for name in ("parent_corpus_id", "generator_commit", "random_seed")
        if getattr(manifest, name) is None
    ]
    if missing:
        raise CorpusValidationError(f"the manifest is missing provenance: {missing}")
    return {
        "corpus_id": manifest.corpus_id,
        "parent_corpus_id": manifest.parent_corpus_id,
        "format_version": manifest.format_version,
        "generator_version": manifest.generator_version,
        "generator_commit": manifest.generator_commit,
        "random_seed": manifest.random_seed,
        "rotation_enabled": manifest.rotation_enabled,
        "scenes": manifest.scenes,
        "families": manifest.family_count,
        "arrangements": manifest.arrangement_count,
        "entity_count": manifest.entity_count,
        "split_counts": counted,
        "split_definition": manifest.split_definition,
        "distinct_relation_graphs": manifest.distinct_relation_graphs,
        "changes_from_parent": list(manifest.changes_from_parent),
    }


def _check_pairing(corpus_dir: Path, parent_dir: Path) -> dict[str, Any]:
    """Scene for scene against the parent corpus: only the frame measurement may differ."""
    out: dict[str, Any] = {"parent_corpus_dir": str(parent_dir), "splits": {}}
    for split in STEP8_SPLITS:
        rotated = load_step8_split(corpus_dir, split)
        plain = load_step8_split(parent_dir, split)
        if len(rotated) != len(plain):
            raise CorpusValidationError(
                f"{split}: {len(rotated)} rotated scenes against {len(plain)} parent scenes"
            )
        worst_centroid = 0.0
        extents_moved = 0
        for new, old in zip(rotated, plain, strict=True):
            if new.scene_id != old.scene_id:
                raise CorpusValidationError(
                    f"{split}: {new.scene_id} is paired with {old.scene_id}"
                )
            if new.edges != old.edges:
                raise CorpusValidationError(f"{new.scene_id}: the relationship graph changed")
            if new.counts != old.counts:
                raise CorpusValidationError(f"{new.scene_id}: entity point counts changed")
            if new.entity_ids != old.entity_ids:
                raise CorpusValidationError(f"{new.scene_id}: the entity set changed")
            if new.active_lod != old.active_lod:
                raise CorpusValidationError(f"{new.scene_id}: the level of detail changed")
            for entity_id, centroid in old.centroids.items():
                deviation = float(
                    np.abs(np.asarray(new.centroids[entity_id]) - np.asarray(centroid)).max()
                )
                worst_centroid = max(worst_centroid, deviation)
                if not np.allclose(new.extents[entity_id], old.extents[entity_id], atol=1e-6):
                    extents_moved += 1
        if worst_centroid > 0.0:
            raise CorpusValidationError(
                f"{split}: centroids moved by up to {worst_centroid:.3e}; rotation was meant to "
                "be the only change"
            )
        out["splits"][split] = {
            "scenes": len(rotated),
            "worst_centroid_deviation": worst_centroid,
            "entities_whose_extent_moved": extents_moved,
        }
    return out


def _check_rotations(corpus_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}}
    total = 0
    worst = 0.0
    for split in STEP8_SPLITS:
        report = check_corpus_rotations(load_step8_split(corpus_dir, split))
        out["splits"][split] = report
        total += int(report["rotations_checked"])
        worst = max(worst, float(report["worst_deviation"]))
    out["rotations_checked"] = total
    out["worst_deviation"] = worst
    out["compared_against"] = "each scene's own generator parameters, rebuilt"
    return out


def _check_placement(corpus_dir: Path) -> dict[str, Any]:
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    out: dict[str, Any] = {}
    for split in STEP8_SPLITS:
        scenes = load_step8_split(corpus_dir, split, limit=32)
        loader = WholeOrganLoader(
            scenes, builder, batch_size=16, seed=0, shuffle=False, drop_last=False
        )
        batch = next(iter(loader.epoch(0))).batch
        placement = HierarchicalPlacement(
            dict(builder.slot_of), slots=int(batch.entity_present.shape[1])
        )
        out[split] = validate_placement(
            placement,
            slot_of=builder.slot_of,
            hierarchy="spatial",
            frames=batch.entity_frames,
            present=batch.entity_present,
        )
    return out


def _check_levels(corpus_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}}
    for split in STEP8_SPLITS:
        levels: dict[int, int] = {}
        for scene in load_step8_split(corpus_dir, split):
            level = int(scene.active_lod)
            levels[level] = levels.get(level, 0) + 1
            if level not in (1, 2, 3):
                raise CorpusValidationError(f"{scene.scene_id}: level of detail {level}")
            if len(scene.centroids) != len(WHOLE_ORGAN_ENTITIES):
                raise CorpusValidationError(
                    f"{scene.scene_id}: {len(scene.centroids)} entities, not "
                    f"{len(WHOLE_ORGAN_ENTITIES)}; presence would carry the arrangement"
                )
            first = set(scene.visible_entity_ids(1))
            second = set(scene.visible_entity_ids(2))
            third = set(scene.visible_entity_ids(3))
            if not (first <= second <= third):
                raise CorpusValidationError(f"{scene.scene_id}: levels of detail do not nest")
        out["splits"][split] = {"lod_counts": {str(k): v for k, v in sorted(levels.items())}}
    out["declared_lod_entities"] = {str(k): len(v) for k, v in sorted(LOD_ENTITIES.items())}
    return out


def _check_splits(corpus_dir: Path) -> dict[str, Any]:
    manifest = load_step8_manifest(corpus_dir)
    regions = split_regions(corpus_dir)
    for split, expected in EXPECTED_REGION.items():
        observed = regions.get(split, {})
        if set(observed) != {expected}:
            raise CorpusValidationError(
                f"{split}: scenes classify into {set(observed)}, expected only {expected!r}"
            )
    families = {split: set(values) for split, values in manifest.split_families.items()}
    overlap = families["train"] & families["test_seen"]
    if overlap:
        raise CorpusValidationError(f"test_seen shares families with train: {sorted(overlap)}")
    return {
        "regions": {split: dict(values) for split, values in regions.items()},
        "classified_by": "classify_arrangement, the same function the hold-out is defined with",
        "train_families": len(families["train"]),
        "test_seen_families": len(families["test_seen"]),
        "test_seen_families_disjoint_from_train": True,
        "other_test_splits_reuse_train_families": sorted(
            split
            for split in ("test_arrangement", "test_transform", "test_combination")
            if families.get(split, set()) <= families["train"]
        ),
    }


def _check_holdout(corpus_dir: Path) -> dict[str, Any]:
    """Which rotation band each split occupies, and that training excludes the held-out ones."""
    out: dict[str, Any] = {
        "yaw_interpolation_hole": list(YAW_INTERPOLATION_HOLE),
        "yaw_extrapolation_edge": YAW_EXTRAPOLATION_EDGE,
        "splits": {},
    }
    for split in STEP8_SPLITS:
        scenes = load_step8_split(corpus_dir, split)
        yaw = np.abs(np.asarray([float(scene.arrangement.yaw) for scene in scenes]))
        mirrored = float(np.mean([scene.arrangement.mirror for scene in scenes]))
        transposed = float(np.mean([scene.arrangement.transpose for scene in scenes]))
        out["splits"][split] = {
            "abs_yaw_min": float(yaw.min()),
            "abs_yaw_max": float(yaw.max()),
            "fraction_in_interpolation_hole": float(
                ((yaw >= YAW_INTERPOLATION_HOLE[0]) & (yaw <= YAW_INTERPOLATION_HOLE[1])).mean()
            ),
            "fraction_beyond_extrapolation_edge": float((yaw > YAW_EXTRAPOLATION_EDGE).mean()),
            "fraction_mirrored": mirrored,
            "fraction_transposed": transposed,
            "regions": dict(classify_counts(scenes)),
        }
    train = out["splits"]["train"]
    if train["fraction_in_interpolation_hole"] > 0.0:
        raise CorpusValidationError("train contains scenes from the interpolation hole")
    if train["fraction_beyond_extrapolation_edge"] > 0.0:
        raise CorpusValidationError("train contains scenes beyond the extrapolation edge")
    if out["splits"]["test_arrangement"]["fraction_in_interpolation_hole"] == 0.0:
        raise CorpusValidationError("test_arrangement holds nothing out of the hole")
    if out["splits"]["test_transform"]["fraction_beyond_extrapolation_edge"] == 0.0:
        raise CorpusValidationError("test_transform holds nothing beyond the edge")
    return out


def classify_counts(scenes: Sequence[Any]) -> dict[str, int]:
    """How many scenes fall in each arrangement region."""
    counts: dict[str, int] = {}
    for scene in scenes:
        region = classify_arrangement(scene.arrangement)
        counts[region] = counts.get(region, 0) + 1
    return counts


def validate_corpus(corpus_dir: str | Path, parent_dir: str | Path) -> dict[str, Any]:
    """Run every corpus check; raise on the first failure."""
    target, parent = Path(corpus_dir), Path(parent_dir)
    return {
        "corpus_dir": str(target),
        "manifest": _check_manifest(target),
        "pairing": _check_pairing(target, parent),
        "rotations": _check_rotations(target),
        "placement": _check_placement(target),
        "levels": _check_levels(target),
        "splits": _check_splits(target),
        "holdout": _check_holdout(target),
        "notes": [
            "Every check raises rather than reports; reaching the end is the pass.",
            "Extents differ from the parent corpus by design: an oriented frame measures "
            "them on its own axes.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Validate the Step 10 rotated corpus.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--parent", type=Path, default=Path("datasets/processed/step8_continuous")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/runs/step10-change2/corpus_validation.json"),
    )
    args = parser.parse_args(argv)

    try:
        report = validate_corpus(args.corpus, args.parent)
    except (CorpusValidationError, PlacementIntegrityError) as failure:
        print(f"FAILED: {failure}")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"corpus {report['manifest']['corpus_id']} passed every check")
    print(f"  rotations checked against the generator: {report['rotations']['rotations_checked']}")
    print(f"  worst rotation deviation: {report['rotations']['worst_deviation']:.2e}")
    print(f"  scene-for-scene pairing with {report['manifest']['parent_corpus_id']}: exact")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
