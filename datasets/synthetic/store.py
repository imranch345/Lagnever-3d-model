"""Corpus storage and deterministic splits.

Scenes are stored as **parameters**, not as geometry. Every primitive, frame,
occupancy value and label is recomputed analytically from those parameters, so a
corpus of 10,000 scenes is a few megabytes of JSON Lines and is reproducible exactly.

Splits are by **procedural family**, not by scene. Scenes in one family share an
archetype's proportions, so splitting by scene would put near-duplicates on both
sides of the boundary and inflate every number.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import DATA_LABEL, SceneSpec, build_scene

__all__ = [
    "CORPUS_FORMAT_VERSION",
    "SplitName",
    "CorpusManifest",
    "assign_splits",
    "iter_split",
    "generate_corpus",
    "load_manifest",
    "load_split",
    "split_for_family",
]

CORPUS_FORMAT_VERSION = "lagnav-tier0-1"
SplitName = str
_SPLITS: tuple[str, ...] = ("train", "validation", "test")


def assign_splits(families: int, fractions: Mapping[str, float]) -> dict[int, SplitName]:
    """Assign every procedural family to a split, deterministically and in proportion.

    Families are ordered by a stable hash and then sliced by the requested
    fractions. Hashing each family independently would be simpler but gives
    proportions that only hold on average; with 40 families that can easily leave a
    split empty, which is how a quiet evaluation bug starts.
    """
    if families <= 0:
        raise ValueError("A corpus needs at least one family.")
    ordered = sorted(
        range(families),
        key=lambda family: hashlib.blake2s(
            f"family-{family}".encode(), digest_size=8
        ).hexdigest(),
    )
    assignment: dict[int, SplitName] = {}
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


def split_for_family(
    family_id: int, families: int, fractions: Mapping[str, float]
) -> SplitName:
    """Split a single family belongs to."""
    return assign_splits(families, fractions)[family_id]


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """What a generated corpus contains and how it was produced."""

    corpus_id: str
    format_version: str
    ontology_id: str
    ontology_version: str
    scenes: int
    families: int
    lod_distribution: Mapping[int, int]
    split_counts: Mapping[str, int]
    fractions: Mapping[str, float]
    generator_seed: int
    data_label: str = DATA_LABEL
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain data."""
        return {
            "corpus_id": self.corpus_id,
            "format_version": self.format_version,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "scenes": self.scenes,
            "families": self.families,
            "lod_distribution": {str(k): v for k, v in self.lod_distribution.items()},
            "split_counts": dict(self.split_counts),
            "fractions": dict(self.fractions),
            "generator_seed": self.generator_seed,
            "data_label": self.data_label,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CorpusManifest:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            corpus_id=str(payload["corpus_id"]),
            format_version=str(payload["format_version"]),
            ontology_id=str(payload["ontology_id"]),
            ontology_version=str(payload["ontology_version"]),
            scenes=int(payload["scenes"]),
            families=int(payload["families"]),
            lod_distribution={int(k): int(v) for k, v in payload["lod_distribution"].items()},
            split_counts={str(k): int(v) for k, v in payload["split_counts"].items()},
            fractions={str(k): float(v) for k, v in payload["fractions"].items()},
            generator_seed=int(payload["generator_seed"]),
            data_label=str(payload.get("data_label", DATA_LABEL)),
            notes=tuple(payload.get("notes", ())),
        )


def generate_corpus(
    ontology: AnatomyOntology,
    output_dir: str | Path,
    *,
    scenes: int = 2_000,
    families: int = 40,
    fractions: Mapping[str, float] | None = None,
    lod_choices: Sequence[int] = (1, 2, 3),
    seed: int = 0,
    corpus_id: str | None = None,
) -> CorpusManifest:
    """Generate a corpus and write it as JSON Lines plus a manifest."""
    split_fractions = dict(fractions or {"train": 0.8, "validation": 0.1, "test": 0.1})
    total = sum(split_fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}.")
    if families < len(_SPLITS):
        raise ValueError("A corpus needs at least one family per split.")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    assignment = assign_splits(families, split_fractions)
    empty = [name for name in _SPLITS if name not in set(assignment.values())]
    if empty:
        raise ValueError(
            f"Splits {empty} would be empty with {families} families. Increase the family "
            "count or change the fractions; an empty evaluation split is never acceptable."
        )
    handles = {name: (target / f"{name}.jsonl").open("w", encoding="utf-8") for name in _SPLITS}
    counts = dict.fromkeys(_SPLITS, 0)
    lod_counts: dict[int, int] = {}
    try:
        for index in range(scenes):
            family_id = index % families
            lod = int(lod_choices[index % len(lod_choices)])
            spec = build_scene(
                ontology, scene_index=seed * 1_000_000 + index, family_id=family_id, lod=lod
            )
            split = assignment[family_id]
            handles[split].write(json.dumps(spec.to_dict(), separators=(",", ":")) + "\n")
            counts[split] += 1
            lod_counts[lod] = lod_counts.get(lod, 0) + 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest = CorpusManifest(
        corpus_id=corpus_id or f"tier0-{scenes}-{families}-{seed}",
        format_version=CORPUS_FORMAT_VERSION,
        ontology_id=ontology.ontology_id,
        ontology_version=ontology.version,
        scenes=scenes,
        families=families,
        lod_distribution=lod_counts,
        split_counts=counts,
        fractions=split_fractions,
        generator_seed=seed,
        notes=(
            "Scenes store parameters only; geometry is recomputed analytically.",
            "Splits are by procedural family, so no family appears in two splits.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ),
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    return manifest


def load_manifest(corpus_dir: str | Path) -> CorpusManifest:
    """Read a corpus manifest, checking the format version."""
    path = Path(corpus_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No corpus manifest at {path}. Generate one with "
            "'python -m datasets.synthetic.cli'."
        )
    manifest = CorpusManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if manifest.format_version != CORPUS_FORMAT_VERSION:
        raise ValueError(
            f"Corpus format {manifest.format_version!r} does not match the expected "
            f"{CORPUS_FORMAT_VERSION!r}."
        )
    return manifest


def iter_split(corpus_dir: str | Path, split: SplitName) -> Iterator[SceneSpec]:
    """Stream the scenes of one split."""
    path = Path(corpus_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No split file at {path}.")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield SceneSpec.from_dict(json.loads(line))


def load_split(
    corpus_dir: str | Path, split: SplitName, limit: int | None = None
) -> list[SceneSpec]:
    """Load a split into memory, optionally truncated."""
    out: list[SceneSpec] = []
    for spec in iter_split(corpus_dir, split):
        out.append(spec)
        if limit is not None and len(out) >= limit:
            break
    return out
