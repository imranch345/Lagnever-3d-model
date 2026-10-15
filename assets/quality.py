"""Rank built subjects and choose the one Level 1 shows.

Checks are structural and relational, read from each subject's ``manifest.json``. None is a
clinical reference range: this project has no validated normal values, and inventing some
would be worse than not checking.

**Hard** -- a subject that fails any of these is not usable:

* every mapped entity has a mesh;
* every left and right entity lies on its own side of the heart;
* every mesh is watertight, and its mask is one connected piece once specks below the
  pipeline's component fraction are ignored (a hollow myocardium counts as one);
* no heart or lung structure touches a face of the scanned volume -- touching means the field
  of view cut it off. Cropped vessels and the trachea, which run out of the thorax by nature,
  are exempt.

**Soft** -- reported, and used only to break ties among usable subjects:

* each lung is larger than the heart shell;
* every chamber's centroid lies inside the heart shell's bounds;
* the trachea's centroid is superior to the heart's.

    python -m assets.quality --scenes datasets/processed/level1
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["SubjectQuality", "assess", "choose"]

_HEART_AND_LUNGS = ("heart", "respiratory.")
#: Structures that run out of the thorax by nature; reaching a face is expected, not a cut.
_RUNS_OUT = frozenset({"respiratory.trachea"})
_CHAMBERS = ("heart.left_atrium", "heart.left_ventricle", "heart.right_atrium",
             "heart.right_ventricle")


@dataclass
class SubjectQuality:
    """One subject's verdict."""

    subject: str
    usable: bool
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    dropped_components: int = 0


def _inside(point: Sequence[float], bounds: Sequence[Sequence[float]]) -> bool:
    return all(bounds[0][i] <= point[i] <= bounds[1][i] for i in range(3))


def assess(manifest: Mapping[str, Any]) -> SubjectQuality:
    """Apply the hard and soft checks to one built subject."""
    entities: Mapping[str, Any] = manifest.get("entities", {})
    hard: list[str] = []
    soft: list[str] = []

    if manifest.get("missing"):
        hard.append(f"missing {manifest['missing']}")
    wrong_side = [e for e, ok in manifest.get("sides_correct", {}).items() if not ok]
    if wrong_side:
        hard.append(f"wrong side {wrong_side}")
    dropped = 0
    for entity_id, entry in entities.items():
        quality = entry["quality"]
        if not quality["watertight"]:
            hard.append(f"{entity_id} not watertight")
        if quality.get("voxel_components", 1) != 1:
            hard.append(f"{entity_id} is in {quality['voxel_components']} separate pieces")
        checked = entity_id.startswith(_HEART_AND_LUNGS) and entity_id not in _RUNS_OUT
        if checked and entry.get("crop") is None:
            touched = quality.get("touches_boundary", [])
            if touched:
                hard.append(f"{entity_id} cut by the field of view ({', '.join(touched)})")
        dropped += sum(1 for note in quality.get("notes", []) if note.startswith("dropped"))

    def volume(entity_id: str) -> float | None:
        entry = entities.get(entity_id)
        return None if entry is None else entry["quality"].get("volume_ml")

    heart = volume("heart")
    for lung in ("respiratory.left_lung", "respiratory.right_lung"):
        lung_volume = volume(lung)
        if heart is not None and lung_volume is not None and lung_volume <= heart:
            soft.append(f"{lung} not larger than the heart")
    if "heart" in entities:
        bounds = entities["heart"]["quality"]["bounds_mm"]
        for chamber in _CHAMBERS:
            if chamber in entities and not _inside(
                entities[chamber]["quality"]["centroid_mm"], bounds
            ):
                soft.append(f"{chamber} centroid outside the heart")
        trachea = entities.get("respiratory.trachea")
        if trachea is not None:
            # Viewer frame: +Y is superior.
            if trachea["centroid_viewer_mm"][1] <= entities["heart"]["centroid_viewer_mm"][1]:
                soft.append("trachea not superior to the heart")

    return SubjectQuality(
        subject=str(manifest.get("subject", "?")),
        usable=not hard,
        hard=hard,
        soft=soft,
        dropped_components=dropped,
    )


def choose(qualities: Sequence[SubjectQuality]) -> SubjectQuality | None:
    """The usable subject with the fewest soft findings, then fewest dropped components."""
    usable = [q for q in qualities if q.usable]
    if not usable:
        return None
    return min(usable, key=lambda q: (len(q.soft), q.dropped_components, q.subject))


def build_all(raw_root: Path, scenes: Path) -> dict[str, str]:
    """Build every subject under ``raw_root`` into ``scenes``; return build failures by subject.

    A subject that cannot be built is recorded with its error and is simply not usable; one
    bad scan must not stop the others.
    """
    from assets.meshes import build_subject

    failures: dict[str, str] = {}
    for subject in sorted(p for p in raw_root.iterdir() if (p / "segmentations").is_dir()):
        try:
            build_subject(subject, scenes / subject.name)
        except Exception as exc:  # noqa: BLE001 - recorded per subject, not swallowed
            failures[subject.name] = f"{type(exc).__name__}: {exc}"
        print(f"built {subject.name}: {failures.get(subject.name, 'ok')}", flush=True)
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """Assess every built subject under a folder and write ``quality_report.json`` there."""
    parser = argparse.ArgumentParser(description="Rank built Level 1 subjects.")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--build-from", type=Path, default=None,
                        help="raw subjects to build into --scenes first")
    args = parser.parse_args(argv)
    failures = build_all(args.build_from, args.scenes) if args.build_from else {}
    qualities = [
        assess(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.scenes.glob("*/manifest.json"))
    ]
    qualities += [
        SubjectQuality(subject=name, usable=False, hard=[f"build failed: {error}"])
        for name, error in failures.items()
    ]
    chosen = choose(qualities)
    report = {
        "chosen": chosen.subject if chosen else None,
        "subjects": [asdict(q) for q in qualities],
        "checks": "structural and relational only; no clinical reference ranges",
    }
    (args.scenes / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    usable = sum(q.usable for q in qualities)
    print(f"{usable} of {len(qualities)} usable; chosen: {report['chosen']}")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
