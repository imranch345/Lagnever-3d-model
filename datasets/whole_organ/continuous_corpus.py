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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datasets.whole_organ.arrangement import (
    ARRANGEMENT_FIELDS,
    Arrangement,
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

__all__ = [
    "STEP8_SPLITS",
    "TEST_SPLITS",
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
) -> Step8Manifest:
    """Generate a Step 8 corpus and write one JSONL file per split."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    assignment = _family_assignment(families)
    wanted = {
        "train": train_scenes,
        "validation": validation_scenes,
        **{name: test_scenes for name in TEST_SPLITS},
    }

    counts: dict[str, int] = dict.fromkeys(STEP8_SPLITS, 0)
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
                    )
                except RuntimeError:
                    index += 1
                    continue
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
