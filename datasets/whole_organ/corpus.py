"""Whole-organ scenes, corpus generation and deterministic splits.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED.

A scene stores the global organ parameters, the measured per-entity statistics and the
measured relationship graph. Geometry is recomputed analytically from the parameters, so
a corpus of thousands of scenes is a few megabytes and is reproducible exactly.

Scenes are rejected and resampled if any of the twenty entities fails to own measurable
volume. Uniform presence matters: if presence varied by variant, the variant would leak
through a channel other than the relationship graph and the experiment would be void.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from datasets.whole_organ.arrangement import Arrangement, classify_arrangement
from datasets.whole_organ.field import (
    CHAMBERS,
    SEPTA,
    VALVES,
    VESSELS,
    WHOLE_ORGAN_ENTITIES,
    WholeOrganField,
)
from datasets.whole_organ.parameters import DATA_LABEL, OrganParameters, Variant, sample_parameters
from datasets.whole_organ.relations import MeasuredEdge, measure_relations

__all__ = [
    "CORPUS_FORMAT_VERSION",
    "LOD_ENTITIES",
    "WholeOrganScene",
    "build_scene",
    "generate_corpus",
    "load_manifest",
    "load_split",
    "WholeOrganManifest",
]

FloatArray = npt.NDArray[np.float64]

CORPUS_FORMAT_VERSION = "lagnav-whole-organ-1"
_SPLITS: tuple[str, ...] = ("train", "validation", "test")

LOD_ENTITIES: Mapping[int, tuple[str, ...]] = {
    0: (),
    1: (*CHAMBERS, *VESSELS),
    2: (*CHAMBERS, *VESSELS, *VALVES, *SEPTA),
    3: WHOLE_ORGAN_ENTITIES,
    4: WHOLE_ORGAN_ENTITIES,
}
"""Which entities each level of detail shows.

Matches the ontology's declared ``min_lod`` for these entities. Level-specific targets
are what Step 6's level-of-detail objective lacked: every prefix was trained against the
same full-detail target, so later tokens had nothing to add.
"""


@dataclass(slots=True)
class WholeOrganScene:
    """One generated organ."""

    scene_id: str
    family_id: int
    seed: int
    ontology_id: str
    ontology_version: str
    active_lod: int
    parameters: OrganParameters
    centroids: dict[str, list[float]]
    extents: dict[str, list[float]]
    counts: dict[str, int]
    edges: tuple[MeasuredEdge, ...]
    data_label: str = DATA_LABEL

    @property
    def arrangement(self) -> Arrangement:
        """Where this scene sits in the continuous arrangement space."""
        return self.parameters.arrangement

    @property
    def variant(self) -> Variant:
        """Coarse label derived from the arrangement. Reporting only, never an input."""
        return self.parameters.variant

    @property
    def region(self) -> str:
        """Which held-out region this scene's arrangement belongs to."""
        return classify_arrangement(self.parameters.arrangement)

    def field(self) -> WholeOrganField:
        """Rebuild the analytic field from the stored parameters."""
        return WholeOrganField(self.parameters)

    @property
    def entity_ids(self) -> tuple[str, ...]:
        """Entities present in this scene."""
        return tuple(entity_id for entity_id in WHOLE_ORGAN_ENTITIES if entity_id in self.centroids)

    def visible_entity_ids(self, lod: int | None = None) -> tuple[str, ...]:
        """Entities visible at a level of detail."""
        level = self.active_lod if lod is None else lod
        allowed = set(LOD_ENTITIES[min(max(level, 0), 4)])
        return tuple(entity_id for entity_id in self.entity_ids if entity_id in allowed)

    def frames(self) -> dict[str, list[float]]:
        """Canonical frames, measured from the labelled field."""
        out: dict[str, list[float]] = {}
        for entity_id, centroid in self.centroids.items():
            extent = np.maximum(np.asarray(self.extents[entity_id]), 1e-3)
            out[entity_id] = [
                *centroid,
                *np.log(extent).tolist(),
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "scene_id": self.scene_id,
            "family_id": self.family_id,
            "variant": str(self.variant),
            "region": self.region,
            "seed": self.seed,
            "data_label": self.data_label,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "active_lod": self.active_lod,
            "parameters": self.parameters.to_dict(),
            "centroids": {k: [round(x, 5) for x in v] for k, v in self.centroids.items()},
            "extents": {k: [round(x, 5) for x in v] for k, v in self.extents.items()},
            "counts": dict(self.counts),
            "edges": [[e.subject, e.relation, e.object] for e in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WholeOrganScene:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            scene_id=str(payload["scene_id"]),
            family_id=int(payload["family_id"]),
            seed=int(payload["seed"]),
            ontology_id=str(payload["ontology_id"]),
            ontology_version=str(payload["ontology_version"]),
            active_lod=int(payload["active_lod"]),
            parameters=OrganParameters.from_dict(payload["parameters"]),
            centroids={k: [float(x) for x in v] for k, v in payload["centroids"].items()},
            extents={k: [float(x) for x in v] for k, v in payload["extents"].items()},
            counts={k: int(v) for k, v in payload["counts"].items()},
            edges=tuple(MeasuredEdge(*triple) for triple in payload["edges"]),
            data_label=str(payload.get("data_label", DATA_LABEL)),
        )


def build_scene(
    *,
    scene_index: int,
    family_id: int,
    lod: int,
    variant: Variant | None = None,
    arrangement: Arrangement | None = None,
    ontology_id: str = "lagnav.heart",
    ontology_version: str = "0.1.0",
    max_attempts: int = 24,
) -> WholeOrganScene:
    """Generate one organ, resampling until all twenty entities own volume.

    Raises:
        RuntimeError: if no sample succeeds. Presence must be uniform across variants or
            the variant leaks through a channel other than the relationship graph.

    """
    for attempt in range(max_attempts):
        seed = scene_index * 97 + attempt
        rng = np.random.default_rng(seed)
        parameters = sample_parameters(rng, family_id, variant, arrangement=arrangement)
        organ = WholeOrganField(parameters)
        statistics = organ.entity_statistics(np.random.default_rng(seed + 11))
        if len(statistics) < len(WHOLE_ORGAN_ENTITIES):
            continue
        edges = measure_relations(organ, statistics, np.random.default_rng(seed + 23))
        return WholeOrganScene(
            scene_id=f"whole-{scene_index:06d}",
            family_id=family_id,
            seed=seed,
            ontology_id=ontology_id,
            ontology_version=ontology_version,
            active_lod=lod,
            parameters=parameters,
            centroids={k: v["centroid"].tolist() for k, v in statistics.items()},
            extents={k: v["extent"].tolist() for k, v in statistics.items()},
            counts={k: int(v["count"][0]) for k, v in statistics.items()},
            edges=edges,
        )
    raise RuntimeError(
        f"Could not generate a complete organ for family {family_id} in "
        f"{max_attempts} attempts. Fix the generator rather than accepting a scene with a "
        "missing entity."
    )


@dataclass(frozen=True, slots=True)
class WholeOrganManifest:
    """What a whole-organ corpus contains."""

    corpus_id: str
    format_version: str
    scenes: int
    families: int
    variants: Mapping[str, int]
    split_counts: Mapping[str, int]
    lod_distribution: Mapping[int, int]
    distinct_relation_graphs: int
    entity_count: int
    variants_per_family: Mapping[str, int] = field(default_factory=dict)
    split_variants: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    data_label: str = DATA_LABEL
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "corpus_id": self.corpus_id,
            "format_version": self.format_version,
            "scenes": self.scenes,
            "families": self.families,
            "variants": dict(self.variants),
            "split_counts": dict(self.split_counts),
            "lod_distribution": {str(k): v for k, v in self.lod_distribution.items()},
            "distinct_relation_graphs": self.distinct_relation_graphs,
            "entity_count": self.entity_count,
            "variants_per_family": dict(self.variants_per_family),
            "split_variants": {
                split: dict(counts) for split, counts in self.split_variants.items()
            },
            "data_label": self.data_label,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WholeOrganManifest:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            corpus_id=str(payload["corpus_id"]),
            format_version=str(payload["format_version"]),
            scenes=int(payload["scenes"]),
            families=int(payload["families"]),
            variants={str(k): int(v) for k, v in payload["variants"].items()},
            split_counts={str(k): int(v) for k, v in payload["split_counts"].items()},
            lod_distribution={int(k): int(v) for k, v in payload["lod_distribution"].items()},
            distinct_relation_graphs=int(payload["distinct_relation_graphs"]),
            entity_count=int(payload["entity_count"]),
            variants_per_family={
                str(k): int(v) for k, v in payload.get("variants_per_family", {}).items()
            },
            split_variants={
                str(split): {str(k): int(v) for k, v in counts.items()}
                for split, counts in payload.get("split_variants", {}).items()
            },
            data_label=str(payload.get("data_label", DATA_LABEL)),
            notes=tuple(payload.get("notes", ())),
        )


def assign_splits(families: int, fractions: Mapping[str, float]) -> dict[int, str]:
    """Assign families to splits deterministically and in proportion."""
    ordered = sorted(
        range(families),
        key=lambda family: hashlib.blake2s(
            f"whole-family-{family}".encode(), digest_size=8
        ).hexdigest(),
    )
    assignment: dict[int, str] = {}
    start = 0
    for position, name in enumerate(_SPLITS):
        share = float(fractions.get(name, 0.0))
        end = families if position == len(_SPLITS) - 1 else start + int(round(share * families))
        for family in ordered[start:end]:
            assignment[family] = name
        start = end
    for family in range(families):
        assignment.setdefault(family, _SPLITS[0])
    return assignment


def generate_corpus(
    output_dir: str | Path,
    *,
    scenes: int = 1_600,
    families: int = 40,
    fractions: Mapping[str, float] | None = None,
    lod_choices: Sequence[int] = (1, 2, 3),
    corpus_id: str | None = None,
) -> WholeOrganManifest:
    """Generate a whole-organ corpus with every variant represented in every split."""
    split_fractions = dict(fractions or {"train": 0.8, "validation": 0.1, "test": 0.1})
    assignment = assign_splits(families, split_fractions)
    empty = [name for name in _SPLITS if name not in set(assignment.values())]
    if empty:
        raise ValueError(f"Splits {empty} would be empty with {families} families.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    handles = {name: (target / f"{name}.jsonl").open("w", encoding="utf-8") for name in _SPLITS}
    counts = dict.fromkeys(_SPLITS, 0)
    variant_counts: dict[str, int] = {}
    lod_counts: dict[int, int] = {}
    signatures: set[str] = set()
    variants = list(Variant)
    family_variants: dict[int, set[str]] = {}
    split_variants: dict[str, dict[str, int]] = {name: {} for name in _SPLITS}
    try:
        for index in range(scenes):
            # Family, variant and level of detail must vary *independently*. Deriving
            # the variant from ``index % len(variants)`` aliases it onto the family
            # whenever the variant count divides the family count, which confounds the
            # two and leaves each family stuck in a single arrangement. Advancing the
            # variant once per full pass over the families removes the aliasing, so
            # every family is seen in every variant.
            family_id = index % families
            variant = variants[(index // families) % len(variants)]
            lod = int(lod_choices[index % len(lod_choices)])
            scene = build_scene(scene_index=index, family_id=family_id, variant=variant, lod=lod)
            split = assignment[family_id]
            handles[split].write(json.dumps(scene.to_dict(), separators=(",", ":")) + "\n")
            counts[split] += 1
            variant_counts[str(variant)] = variant_counts.get(str(variant), 0) + 1
            family_variants.setdefault(family_id, set()).add(str(variant))
            bucket = split_variants[split]
            bucket[str(variant)] = bucket.get(str(variant), 0) + 1
            lod_counts[lod] = lod_counts.get(lod, 0) + 1
            signatures.add(
                "|".join(sorted(f"{e.subject}|{e.relation}|{e.object}" for e in scene.edges))
            )
    finally:
        for handle in handles.values():
            handle.close()

    # Guard against the confound that variant aliasing produces: if a family only ever
    # appears in one arrangement, or a split is missing an arrangement entirely, then
    # "same organ, different relations" is never actually presented and the graph
    # experiments measure family differences rather than relation differences.
    starved = sorted(f for f, seen in family_variants.items() if len(seen) < len(variants))
    if starved:
        raise ValueError(
            f"{len(starved)} of {families} families cover fewer than {len(variants)} "
            f"variants (first: {starved[:5]}). Variant and family are confounded."
        )
    thin = {
        split: sorted(set(str(v) for v in variants) - set(seen))
        for split, seen in split_variants.items()
        if counts[split] and set(str(v) for v in variants) - set(seen)
    }
    if thin:
        raise ValueError(f"Splits missing variants: {thin}.")

    manifest = WholeOrganManifest(
        corpus_id=corpus_id or f"whole-organ-{scenes}-{families}",
        format_version=CORPUS_FORMAT_VERSION,
        scenes=scenes,
        families=families,
        variants=variant_counts,
        split_counts=counts,
        lod_distribution=lod_counts,
        distinct_relation_graphs=len(signatures),
        entity_count=len(WHOLE_ORGAN_ENTITIES),
        variants_per_family={
            str(family): len(seen) for family, seen in sorted(family_variants.items())
        },
        split_variants={split: dict(seen) for split, seen in split_variants.items()},
        notes=(
            "The organ is generated as a whole and then segmented; no entity is placed "
            "independently.",
            "Relationships are measured from the finished organ, not taken from the ontology.",
            "Variants change the arrangement and therefore the relations; presence is identical.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ),
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(corpus_dir: str | Path) -> WholeOrganManifest:
    """Read a corpus manifest, checking the format version."""
    path = Path(corpus_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No whole-organ manifest at {path}.")
    manifest = WholeOrganManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if manifest.format_version != CORPUS_FORMAT_VERSION:
        raise ValueError(
            f"Corpus format {manifest.format_version!r} does not match {CORPUS_FORMAT_VERSION!r}."
        )
    return manifest


def iter_split(corpus_dir: str | Path, split: str) -> Iterator[WholeOrganScene]:
    """Stream one split."""
    path = Path(corpus_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No split file at {path}.")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield WholeOrganScene.from_dict(json.loads(line))


def load_split(
    corpus_dir: str | Path, split: str, limit: int | None = None
) -> list[WholeOrganScene]:
    """Load one split into memory."""
    out: list[WholeOrganScene] = []
    for scene in iter_split(corpus_dir, split):
        out.append(scene)
        if limit is not None and len(out) >= limit:
            break
    return out
