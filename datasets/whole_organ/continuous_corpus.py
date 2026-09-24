"""Step 8 corpus: continuous arrangements with held-out regions.

Step 7's corpus offered four arrangements crossed with forty families. That was enough
to show the relationship graph carried scene-specific information. It was not enough to
show a model was *inferring* from it, because four labels can be classified.

This generator draws each scene's arrangement from the continuous space in
:mod:`datasets.whole_organ.arrangement`. Two scenes essentially never share an
arrangement, so there is nothing to memorise and relational inference is the only route
to placement.

Splits
------

Six, and the four test splits ask four different questions:

``train`` / ``validation``
    In-distribution arrangements, families reserved for training.
``test_seen``
    In-distribution arrangements, **held-out families**. Does it generalise to a new
    organ shape?
``test_arrangement``
    Seen families, arrangements from a band removed from training. Does it interpolate
    into a hole in the arrangement space?
``test_transform``
    Rotations larger than any seen in training. Does it extrapolate past the edge?
``test_combination``
    Mirrored and transposed together, a pair that never co-occurs in training. Does it
    compose two transformations it has only seen separately?

The boundaries live in :func:`~datasets.whole_organ.arrangement.classify_arrangement`,
so the generator and the leakage tests read the same definition and cannot disagree.

What is deliberately **not** varied: which entities are present, the text features, and
the entity ordering. The relationship graph remains the only channel carrying the
arrangement.

Synthetic research data. Not anatomy, not validated, not clinical.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from datasets.whole_organ.arrangement import (
    ARRANGEMENT_FIELDS,
    Arrangement,
    axis_ranges,
    classify_arrangement,
    sample_arrangement,
)
from datasets.whole_organ.corpus import (
    CORPUS_FORMAT_VERSION,
    WholeOrganScene,
    build_scene,
)
from datasets.whole_organ.field import WHOLE_ORGAN_ENTITIES
from datasets.whole_organ.parameters import DATA_LABEL
from datasets.whole_organ.rotation_stats import scene_angles, summarise_angles

__all__ = [
    "STEP8_SPLITS",
    "TEST_SPLITS",
    "derive_rotated_corpus",
    "Step8Manifest",
    "generate_step8_corpus",
    "load_step8_manifest",
    "load_step8_split",
    "iter_step8_split",
]

#: Every split written by :func:`generate_step8_corpus`, in a fixed order.
STEP8_SPLITS: tuple[str, ...] = (
    "train",
    "validation",
    "test_seen",
    "test_arrangement",
    "test_transform",
    "test_combination",
)

#: The four that answer a generalisation question.
#: Bumped when the generator changes what it writes. Change 2 added measured rotations.
GENERATOR_VERSION = "whole-organ-generator-2"

#: What wrote every manifest that carries no ``generator_version`` field, the Change 1 corpus
#: among them. Defaulting those to the current version would relabel the frozen corpus.
PARENT_GENERATOR_VERSION = "whole-organ-generator-1"

TEST_SPLITS: tuple[str, ...] = (
    "test_seen",
    "test_arrangement",
    "test_transform",
    "test_combination",
)

#: Which arrangement region each split draws from.
_REGION_OF: Mapping[str, str] = {
    "train": "in_distribution",
    "validation": "in_distribution",
    "test_seen": "in_distribution",
    "test_arrangement": "test_arrangement",
    "test_transform": "test_transform",
    "test_combination": "test_combination",
}


@dataclass(frozen=True, slots=True)
class Step8Manifest:
    """What a Step 8 corpus contains, and how its hold-outs are defined."""

    corpus_id: str
    format_version: str
    scenes: int
    families: int
    split_counts: Mapping[str, int]
    split_families: Mapping[str, list[int]]
    lod_distribution: Mapping[int, int]
    distinct_relation_graphs: int
    entity_count: int
    arrangement_fields: tuple[str, ...] = ARRANGEMENT_FIELDS
    arrangement_span: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    data_label: str = DATA_LABEL
    notes: tuple[str, ...] = ()
    rotation_enabled: bool = False
    """Whether entity frames carry measured rotations or the identity."""
    parent_corpus_id: str | None = None
    """The corpus this one is derived from, when it is a variant of an earlier one."""
    generator_version: str = GENERATOR_VERSION
    generator_commit: str | None = None
    random_seed: int | None = None
    rotation_distribution: Mapping[str, Any] = field(default_factory=dict)
    """How rotations were produced, and the measured angle statistics per split."""
    changes_from_parent: tuple[str, ...] = ()
    """Everything that differs from the parent corpus, rotation included."""

    @property
    def arrangement_count(self) -> int:
        """Arrangements drawn: one per scene, from a continuous space."""
        return self.scenes

    @property
    def family_count(self) -> int:
        """Archetype families the scenes are drawn from."""
        return self.families

    @property
    def split_definition(self) -> dict[str, str]:
        """Which arrangement region each split draws from."""
        return dict(_REGION_OF)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "corpus_id": self.corpus_id,
            "format_version": self.format_version,
            "scenes": self.scenes,
            "families": self.families,
            "split_counts": dict(self.split_counts),
            "split_families": {k: list(v) for k, v in self.split_families.items()},
            "lod_distribution": {str(k): v for k, v in self.lod_distribution.items()},
            "distinct_relation_graphs": self.distinct_relation_graphs,
            "entity_count": self.entity_count,
            "arrangement_fields": list(self.arrangement_fields),
            "arrangement_span": {
                split: dict(values) for split, values in self.arrangement_span.items()
            },
            "data_label": self.data_label,
            "notes": list(self.notes),
            "rotation_enabled": self.rotation_enabled,
            "parent_corpus_id": self.parent_corpus_id,
            "generator_version": self.generator_version,
            "generator_commit": self.generator_commit,
            "random_seed": self.random_seed,
            "rotation_distribution": dict(self.rotation_distribution),
            "changes_from_parent": list(self.changes_from_parent),
            "arrangement_count": self.arrangement_count,
            "family_count": self.family_count,
            "split_definition": self.split_definition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Step8Manifest:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            corpus_id=str(payload["corpus_id"]),
            format_version=str(payload["format_version"]),
            scenes=int(payload["scenes"]),
            families=int(payload["families"]),
            split_counts={str(k): int(v) for k, v in payload["split_counts"].items()},
            split_families={
                str(k): [int(f) for f in v] for k, v in payload["split_families"].items()
            },
            lod_distribution={int(k): int(v) for k, v in payload["lod_distribution"].items()},
            distinct_relation_graphs=int(payload["distinct_relation_graphs"]),
            entity_count=int(payload["entity_count"]),
            arrangement_fields=tuple(payload.get("arrangement_fields", ARRANGEMENT_FIELDS)),
            arrangement_span={
                str(split): {str(k): float(v) for k, v in values.items()}
                for split, values in payload.get("arrangement_span", {}).items()
            },
            data_label=str(payload.get("data_label", DATA_LABEL)),
            notes=tuple(payload.get("notes", ())),
            rotation_enabled=bool(payload.get("rotation_enabled", False)),
            parent_corpus_id=(
                str(payload["parent_corpus_id"])
                if payload.get("parent_corpus_id") is not None
                else None
            ),
            generator_version=str(payload.get("generator_version", PARENT_GENERATOR_VERSION)),
            generator_commit=(
                str(payload["generator_commit"])
                if payload.get("generator_commit") is not None
                else None
            ),
            random_seed=(
                int(payload["random_seed"]) if payload.get("random_seed") is not None else None
            ),
            rotation_distribution=dict(payload.get("rotation_distribution", {})),
            changes_from_parent=tuple(payload.get("changes_from_parent", ())),
        )


def _family_assignment(families: int) -> dict[str, list[int]]:
    """Which families each split may use.

    Training families are reserved. ``test_seen`` uses families no training scene ever
    saw, which is what makes it a shape-generalisation test. The three arrangement
    hold-outs deliberately reuse **training** families, so a failure there is about the
    arrangement and not about an unfamiliar organ.
    """
    held_out = max(2, families // 8)
    unseen = list(range(families - held_out, families))
    seen = list(range(families - held_out))
    return {
        "train": seen,
        "validation": seen,
        "test_seen": unseen,
        "test_arrangement": seen,
        "test_transform": seen,
        "test_combination": seen,
    }


def _rotation_provenance() -> dict[str, Any]:
    """How a rotated corpus's rotations were produced, recorded in its manifest."""
    return {
        "source": (
            "declared from the generator's construction: an annulus or a tube takes its own "
            "axis as the frame's third row, with the other two fixed by convention because a "
            "circular vessel is symmetric about its axis; every other entity takes the "
            "organ-to-scene rotation built from the arrangement's yaw, pitch and roll. "
            "Ground truth, not an augmentation applied afterwards."
        ),
        "why_declared_not_principal_axes": (
            "a chamber is nearly an ellipsoid of revolution, so its second and third "
            "principal axes are decided by sampling noise and the target would be a coin flip"
        ),
        "applied": "before measurement: the geometry is rotated, then measured",
        "hierarchy_conditioned": (
            "yes: every entity in a scene shares that scene's organ rotation, and an annulus "
            "or tube composes its construction axis on top of it"
        ),
        "independent_per_entity": False,
        "may_rotate": "every entity; ten carry the organ rotation alone, ten also an axis",
        "scene_rotation_fields": ["yaw", "pitch", "roll"],
        "scene_rotation_bounds_rad": {
            name: list(axis_ranges()[name]) for name in ("yaw", "pitch", "roll")
        },
        "ground_truth": True,
    }


def generate_step8_corpus(
    output_dir: str | Path,
    *,
    train_scenes: int = 1_200,
    validation_scenes: int = 150,
    test_scenes: int = 150,
    families: int = 40,
    lod_choices: Sequence[int] = (1, 2, 3),
    corpus_id: str | None = None,
    seed: int = 20_250_915,
    rotations: bool = False,
    parent_corpus_id: str | None = None,
    generator_commit: str | None = None,
) -> Step8Manifest:
    """Generate a corpus and write one JSONL file per split.

    ``rotations`` records each entity's declared rotation and measures its extents in that
    frame; everything else about a scene is drawn identically either way. Left false, this
    reproduces the Change 1 corpus exactly.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    assignment = _family_assignment(families)
    wanted = {
        "train": train_scenes,
        "validation": validation_scenes,
        **{name: test_scenes for name in TEST_SPLITS},
    }

    counts: dict[str, int] = dict.fromkeys(STEP8_SPLITS, 0)
    angles: dict[str, list[float]] = {name: [] for name in STEP8_SPLITS}
    entity_angles: dict[str, list[float]] = {}
    lod_counts: dict[int, int] = {}
    signatures: set[str] = set()
    span: dict[str, dict[str, float]] = {}
    index = 0

    handles = {
        name: (target / f"{name}.jsonl").open("w", encoding="utf-8") for name in STEP8_SPLITS
    }
    try:
        for split in STEP8_SPLITS:
            rng = np.random.default_rng(seed + 1000 * STEP8_SPLITS.index(split))
            pool = assignment[split]
            region = _REGION_OF[split]
            observed: list[Arrangement] = []
            attempts = 0
            while counts[split] < wanted[split] and attempts < wanted[split] * 12:
                attempts += 1
                family_id = int(pool[counts[split] % len(pool)])
                lod = int(lod_choices[index % len(lod_choices)])
                arrangement = sample_arrangement(rng, region=region)
                try:
                    scene = build_scene(
                        scene_index=index,
                        family_id=family_id,
                        lod=lod,
                        arrangement=arrangement,
                        rotations=rotations,
                    )
                except RuntimeError:
                    index += 1
                    continue
                if rotations:
                    for entity_id, angle in scene_angles(scene.frames()).items():
                        angles[split].append(angle)
                        entity_angles.setdefault(entity_id, []).append(angle)
                handles[split].write(json.dumps(scene.to_dict(), separators=(",", ":")) + "\n")
                counts[split] += 1
                lod_counts[lod] = lod_counts.get(lod, 0) + 1
                signatures.add(
                    "|".join(sorted(f"{e.subject}|{e.relation}|{e.object}" for e in scene.edges))
                )
                observed.append(arrangement)
                index += 1
            if counts[split] < wanted[split]:
                raise RuntimeError(
                    f"Split {split!r} produced {counts[split]} of {wanted[split]} scenes. "
                    "The generator is rejecting too many arrangements; fix it rather than "
                    "accepting a short split."
                )
            coordinates = np.asarray([a.coordinates() for a in observed])
            span[split] = {
                f"{name}_min": float(coordinates[:, position].min())
                for position, name in enumerate(ARRANGEMENT_FIELDS)
            }
            span[split].update(
                {
                    f"{name}_max": float(coordinates[:, position].max())
                    for position, name in enumerate(ARRANGEMENT_FIELDS)
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    distribution: dict[str, Any] = {}
    if rotations:
        distribution = {
            **_rotation_provenance(),
            "splits": {name: summarise_angles(values) for name, values in angles.items()},
            "entities": {
                entity_id: summarise_angles(values)
                for entity_id, values in sorted(entity_angles.items())
            },
        }

    manifest = Step8Manifest(
        corpus_id=corpus_id or f"step8-continuous-{sum(counts.values())}-{families}",
        format_version=CORPUS_FORMAT_VERSION,
        scenes=sum(counts.values()),
        families=families,
        split_counts=counts,
        split_families=assignment,
        lod_distribution=lod_counts,
        distinct_relation_graphs=len(signatures),
        entity_count=len(WHOLE_ORGAN_ENTITIES),
        arrangement_span=span,
        notes=(
            "Arrangements are drawn from a continuous space, so two scenes essentially "
            "never share one and arrangement classification is not a shortcut.",
            "Hold-out boundaries come from classify_arrangement, the same function the "
            "leakage tests use.",
            "test_seen holds out families; the other three hold out arrangements while "
            "reusing training families, so a failure is about the arrangement.",
            "Presence, text features and entity ordering are identical across every "
            "arrangement. The relationship graph is the only channel that carries it.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ),
        rotation_enabled=rotations,
        parent_corpus_id=parent_corpus_id,
        generator_commit=generator_commit,
        random_seed=seed,
        rotation_distribution=distribution,
        changes_from_parent=(
            (
                "entity frames carry measured rotations instead of the identity",
                "extents are measured on each entity's own frame axes, not the world axes, "
                "because an oriented frame with world-axis extents describes a different box",
            )
            if rotations
            else ()
        ),
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest


def derive_rotated_corpus(
    parent_dir: str | Path,
    output_dir: str | Path,
    *,
    corpus_id: str | None = None,
    generator_commit: str | None = None,
    centroid_tolerance: float = 1e-4,
    declared_parent_seed: int | None = None,
) -> Step8Manifest:
    """Re-measure an existing corpus's frames with rotations, changing nothing else.

    Every stored input is carried over verbatim — the generator parameters, the centroids, the
    relationship graph, the point counts, presence, the level of detail, the split and family
    assignment — and only two things are replaced: each entity's rotation, and its extents,
    which are re-measured on that rotation's own axes.

    Deriving rather than generating afresh is deliberate. The relationship graph is the channel
    that carries the arrangement to the model, and until this pass the adjacency draws iterated
    a set of entity ids, so they depended on hash randomisation and a freshly generated corpus
    did not reproduce its predecessor's graphs. That is fixed, but the parent corpus was written
    before the fix and is frozen, so the only way to hold the graphs exactly constant is to copy
    them. What remains is a corpus that differs from its parent in the frame measurement and in
    nothing else.

    Raises:
        ValueError: if a re-measured centroid does not reproduce the stored one, which would
            mean the derivation is not measuring the same scene.

    """
    parent_path, target = Path(parent_dir), Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    parent = load_step8_manifest(parent_path)

    angles: dict[str, list[float]] = {name: [] for name in STEP8_SPLITS}
    entity_angles: dict[str, list[float]] = {}
    counts: dict[str, int] = dict.fromkeys(STEP8_SPLITS, 0)
    lod_counts: dict[int, int] = {}
    signatures: set[str] = set()
    worst_centroid = 0.0

    handles = {
        name: (target / f"{name}.jsonl").open("w", encoding="utf-8") for name in STEP8_SPLITS
    }
    try:
        for split in STEP8_SPLITS:
            for scene in iter_step8_split(parent_path, split):
                organ = scene.field()
                measured = organ.oriented_statistics(np.random.default_rng(scene.seed + 11))
                missing = set(scene.centroids) - set(measured)
                if missing:
                    raise ValueError(
                        f"{scene.scene_id}: re-measuring lost {sorted(missing)}; the "
                        "derivation is not reproducing the parent's scene"
                    )
                for entity_id, stored in scene.centroids.items():
                    deviation = float(
                        np.abs(measured[entity_id]["centroid"] - np.asarray(stored)).max()
                    )
                    worst_centroid = max(worst_centroid, deviation)
                    if deviation > centroid_tolerance:
                        raise ValueError(
                            f"{scene.scene_id}/{entity_id}: re-measured centroid differs from "
                            f"the stored one by {deviation:.3e}"
                        )
                rotated = replace(
                    scene,
                    extents={
                        entity_id: measured[entity_id]["frame_extent"].tolist()
                        for entity_id in scene.centroids
                    },
                    rotations={
                        entity_id: measured[entity_id]["rotation"].tolist()
                        for entity_id in scene.centroids
                    },
                )
                handles[split].write(
                    json.dumps(rotated.to_dict(), separators=(",", ":")) + "\n"
                )
                counts[split] += 1
                lod_counts[rotated.active_lod] = lod_counts.get(rotated.active_lod, 0) + 1
                signatures.add(
                    "|".join(
                        sorted(f"{e.subject}|{e.relation}|{e.object}" for e in rotated.edges)
                    )
                )
                for entity_id, angle in scene_angles(rotated.frames()).items():
                    angles[split].append(angle)
                    entity_angles.setdefault(entity_id, []).append(angle)
    finally:
        for handle in handles.values():
            handle.close()

    if counts != dict(parent.split_counts):
        raise ValueError(
            f"derived {counts} scenes but the parent holds {dict(parent.split_counts)}"
        )

    manifest = Step8Manifest(
        corpus_id=corpus_id or f"step10-rotated-{sum(counts.values())}-{parent.families}",
        format_version=parent.format_version,
        scenes=sum(counts.values()),
        families=parent.families,
        split_counts=counts,
        split_families=parent.split_families,
        lod_distribution=lod_counts,
        distinct_relation_graphs=len(signatures),
        entity_count=parent.entity_count,
        arrangement_span=parent.arrangement_span,
        notes=(
            *parent.notes,
            "Derived from the parent corpus by re-measuring frames; every other stored "
            "field is the parent's, so the relationship graphs are identical rather than "
            "merely similar.",
        ),
        rotation_enabled=True,
        parent_corpus_id=parent.corpus_id,
        generator_commit=generator_commit,
        random_seed=(
            parent.random_seed if parent.random_seed is not None else declared_parent_seed
        ),
        rotation_distribution={
            **_rotation_provenance(),
            "derived_from": parent.corpus_id,
            "seed_provenance": (
                "the parent manifest records its own seed"
                if parent.random_seed is not None
                else "the parent manifest predates the random_seed field; the value recorded "
                "here is the generating CLI's default, declared rather than read back"
            ),
            "per_scene_seed": (
                "each scene carries its own seed; re-measuring uses scene.seed + 11, the same "
                "generator state the parent measured with"
            ),
            "worst_centroid_deviation_on_rederivation": worst_centroid,
            "splits": {name: summarise_angles(values) for name, values in angles.items()},
            "entities": {
                entity_id: summarise_angles(values)
                for entity_id, values in sorted(entity_angles.items())
            },
        },
        changes_from_parent=(
            "entity frames carry measured rotations instead of the identity",
            "extents are measured on each entity's own frame axes, not the world axes, "
            "because an oriented frame with world-axis extents describes a different box",
        ),
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest


def load_step8_manifest(corpus_dir: str | Path) -> Step8Manifest:
    """Read a Step 8 corpus manifest."""
    path = Path(corpus_dir) / "manifest.json"
    return Step8Manifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def iter_step8_split(corpus_dir: str | Path, split: str) -> Iterator[WholeOrganScene]:
    """Stream one split's scenes."""
    path = Path(corpus_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No split file at {path}.")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield WholeOrganScene.from_dict(json.loads(line))


def load_step8_split(
    corpus_dir: str | Path, split: str, *, limit: int | None = None
) -> list[WholeOrganScene]:
    """Read one split into memory."""
    out: list[WholeOrganScene] = []
    for scene in iter_step8_split(corpus_dir, split):
        out.append(scene)
        if limit is not None and len(out) >= limit:
            break
    return out


def split_regions(corpus_dir: str | Path) -> dict[str, dict[str, int]]:
    """Which arrangement regions each split actually contains.

    A split whose scenes fall in the wrong region is a leak, and this is what the test
    checks. Recomputed from the scenes rather than trusted from the manifest.
    """
    out: dict[str, dict[str, int]] = {}
    for split in STEP8_SPLITS:
        counts: dict[str, int] = {}
        for scene in iter_step8_split(corpus_dir, split):
            region = classify_arrangement(scene.parameters.arrangement)
            counts[region] = counts.get(region, 0) + 1
        out[split] = counts
    return out
