"""Step 10 Change 2 §9: whether the corpus's rotation target actually carries rotation.

The audit reads the corpus through the **evaluation loader**, the same path the floor and any
future training run use, and recomputes everything from the frames it receives. It does not
read the manifest's recorded statistics except to check them against its own, because a
manifest is a claim and this is the check on it.

Three things are established here.

*Rotation is present.* Angle statistics per split and per entity, with the thresholds the
brief names. A corpus whose rotations were the identity, or nearly so, would show it here and
would be rejected: predicting the identity would then be almost free.

*Rotation is well formed.* Every stored basis is orthonormal and right-handed, read before
the model's Gram-Schmidt can repair it, and no frame is non-finite.

*Rotation is not fully exposed to training.* The rotation of a scene comes from its
arrangement, and the arrangement is what the held-out splits hold out. The audit measures the
yaw band each split occupies and the organ-rotation angle that follows from it, so the
compositional hold-out that Step 9 defined can be seen to carry over to rotation rather than
being assumed to.

    python -m experiments.step10.rotation_audit --corpus datasets/processed/step10_rotated
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import (
    YAW_EXTRAPOLATION_EDGE,
    YAW_INTERPOLATION_HOLE,
)
from datasets.whole_organ.continuous_corpus import (
    STEP8_SPLITS,
    load_step8_manifest,
    load_step8_split,
)
from datasets.whole_organ.rotation_stats import (
    basis_to_matrix,
    geodesic_angle_degrees,
    summarise_angles,
)
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.whole_organ import WholeOrganLoader

__all__ = ["audit_rotations", "main"]

ORTHONORMAL_TOLERANCE = 1e-4


def _angles_through_the_loader(
    corpus_dir: Path, split: str, builder: WholeOrganBatchBuilder, *, batch_size: int = 16
) -> tuple[list[float], dict[str, list[float]], dict[str, float]]:
    """Angles per entity as the loader delivers them, plus the well-formedness worst cases."""
    name_of = {slot: entity_id for entity_id, slot in builder.slot_of.items()}
    scenes = load_step8_split(corpus_dir, split)
    loader = WholeOrganLoader(
        scenes, builder, batch_size=batch_size, seed=0, shuffle=False, drop_last=False
    )
    angles: list[float] = []
    per_entity: dict[str, list[float]] = defaultdict(list)
    worst = {"unit_deviation": 0.0, "orthogonality": 0.0, "handedness": 0.0}
    for whole in loader.epoch(0):
        frames = whole.batch.entity_frames.double()
        present = whole.batch.entity_present.bool()
        if not bool(torch.isfinite(frames[present]).all()):
            raise ValueError(f"{split}: a present entity has a non-finite frame")
        first, second = frames[..., 6:9], frames[..., 9:12]
        worst["unit_deviation"] = max(
            worst["unit_deviation"],
            float((first[present].norm(dim=-1) - 1.0).abs().max()),
            float((second[present].norm(dim=-1) - 1.0).abs().max()),
        )
        worst["orthogonality"] = max(
            worst["orthogonality"], float((first * second).sum(dim=-1)[present].abs().max())
        )
        for row in range(frames.shape[0]):
            for slot in torch.nonzero(present[row]).flatten().tolist():
                rotation = basis_to_matrix(frames[row, slot, 6:12].tolist())
                worst["handedness"] = max(
                    worst["handedness"], abs(float(np.linalg.det(rotation)) - 1.0)
                )
                angle = geodesic_angle_degrees(rotation)
                angles.append(angle)
                per_entity[name_of[int(slot)]].append(angle)
    return angles, per_entity, worst


def _yaw_bands(corpus_dir: Path, split: str) -> dict[str, float]:
    """The absolute yaw band a split occupies, which is what its rotations follow from."""
    values = [abs(float(scene.arrangement.yaw)) for scene in load_step8_split(corpus_dir, split)]
    array = np.asarray(values)
    return {
        "abs_yaw_min": float(array.min()),
        "abs_yaw_max": float(array.max()),
        "fraction_in_interpolation_hole": float(
            ((array >= YAW_INTERPOLATION_HOLE[0]) & (array <= YAW_INTERPOLATION_HOLE[1])).mean()
        ),
        "fraction_beyond_extrapolation_edge": float((array > YAW_EXTRAPOLATION_EDGE).mean()),
    }


def audit_rotations(
    corpus_dir: str | Path, *, splits: Sequence[str] = STEP8_SPLITS
) -> dict[str, Any]:
    """Rotation statistics, well-formedness and hold-out structure, per split and per entity."""
    target = Path(corpus_dir)
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_step8_manifest(target)

    per_split: dict[str, Any] = {}
    per_entity_all: dict[str, list[float]] = defaultdict(list)
    worst_overall = {"unit_deviation": 0.0, "orthogonality": 0.0, "handedness": 0.0}
    for split in splits:
        angles, per_entity, worst = _angles_through_the_loader(target, split, builder)
        for key, value in worst.items():
            worst_overall[key] = max(worst_overall[key], value)
        for entity_id, values in per_entity.items():
            per_entity_all[entity_id].extend(values)
        per_split[split] = {
            **summarise_angles(angles),
            "fraction_identity": float(np.mean(np.asarray(angles) < 1e-6)),
            "yaw": _yaw_bands(target, split),
            "entities": {
                entity_id: summarise_angles(values)
                for entity_id, values in sorted(per_entity.items())
            },
        }

    if worst_overall["unit_deviation"] > ORTHONORMAL_TOLERANCE:
        raise ValueError(f"stored rotation bases are not unit: {worst_overall}")
    if worst_overall["orthogonality"] > ORTHONORMAL_TOLERANCE:
        raise ValueError(f"stored rotation bases are not orthogonal: {worst_overall}")
    if worst_overall["handedness"] > ORTHONORMAL_TOLERANCE:
        raise ValueError(f"stored rotations are not right-handed: {worst_overall}")

    overall = [angle for values in per_entity_all.values() for angle in values]
    declared = manifest.rotation_distribution.get("splits", {})
    manifest_difference = {
        split: {
            "manifest_mean_deg": float(declared[split]["mean_deg"]),
            "audited_mean_deg": float(per_split[split]["mean_deg"]),
            "difference_deg": float(per_split[split]["mean_deg"])
            - float(declared[split]["mean_deg"]),
        }
        for split in splits
        if split in declared and "mean_deg" in declared[split]
    }

    return {
        "corpus_id": manifest.corpus_id,
        "parent_corpus_id": manifest.parent_corpus_id,
        "rotation_enabled": manifest.rotation_enabled,
        "read_through": "the evaluation loader, the same path the floor and training use",
        "overall": summarise_angles(overall),
        "max_abs_rotation_minus_identity": _max_deviation(per_entity_all),
        "well_formed": worst_overall,
        "splits": per_split,
        "entities": {
            entity_id: summarise_angles(values)
            for entity_id, values in sorted(per_entity_all.items())
        },
        "manifest_difference": {
            "why": (
                "the manifest averages every entity of every scene; this audit averages the "
                "entities the loader actually delivers, which at a coarse level of detail is "
                "fewer. The audited figure is the one a model is scored against, and the two "
                "are expected to differ, not to match"
            ),
            "splits": manifest_difference,
        },
        "holdout": {
            "yaw_interpolation_hole": list(YAW_INTERPOLATION_HOLE),
            "yaw_extrapolation_edge": YAW_EXTRAPOLATION_EDGE,
            "note": (
                "Rotation follows the arrangement, so the yaw band a split occupies is the "
                "rotation band it occupies. train avoids the hole and the edge; "
                "test_arrangement sits in the hole and test_transform beyond the edge, so "
                "their rotations are held out rather than merely their labels."
            ),
        },
        "notes": [
            "Angles are geodesic, from the identity, in degrees.",
            "A corpus where predicting the identity is nearly free would show a mean near "
            "zero here; this one does not.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }


def _max_deviation(per_entity: dict[str, list[float]]) -> float:
    """The largest angle seen anywhere, which is ``max |R - I|`` expressed as an angle."""
    return max((max(values) for values in per_entity.values() if values), default=0.0)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 10 Change 2 rotation audit.")
    parser.add_argument("--corpus", type=Path, default=Path("datasets/processed/step10_rotated"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/runs/step10-change2/rotation_audit.json")
    )
    args = parser.parse_args(argv)

    report = audit_rotations(args.corpus)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"corpus {report['corpus_id']}, rotations enabled: {report['rotation_enabled']}")
    print(f"max angle anywhere: {report['max_abs_rotation_minus_identity']:.2f} deg")
    for split, entry in report["splits"].items():
        print(
            f"  {split:<18} mean {entry['mean_deg']:6.2f}  median {entry['median_deg']:6.2f}  "
            f"identity {entry['fraction_identity']:.3f}  >30 deg "
            f"{entry['fraction_above_30_deg']:.2f}  |yaw| "
            f"{entry['yaw']['abs_yaw_min']:.3f}-{entry['yaw']['abs_yaw_max']:.3f}"
        )
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
