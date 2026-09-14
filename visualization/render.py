"""Research visualisations: the eight views the Step 6 brief asks for.

Each view answers a question about the representation rather than decorating it:

1. whole heart, 2. per-entity geometry, 3. the four chambers, 4. the valves,
5. entity ownership, 6. level-of-detail comparison, 7. before and after an edit,
8. untouched-entity drift.

Views are rendered from the ground-truth corpus and, when a checkpoint is given,
from the model, so the two can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import AnatomyOntology, load_ontology
from datasets.synthetic.heart_corpus import SceneSpec
from datasets.synthetic.store import load_split
from generation.neural.nn.tensors import BatchBuilder
from visualization.export import (
    render_difference_slice,
    render_ownership_slice,
    render_slice,
    slice_points,
    write_png,
    write_point_cloud_ply,
    write_voxel_obj,
)

__all__ = ["ViewSet", "render_ground_truth_views", "render_model_views", "main"]

CHAMBERS = (
    "heart.right_atrium",
    "heart.right_ventricle",
    "heart.left_atrium",
    "heart.left_ventricle",
)
VALVES = (
    "heart.tricuspid_valve",
    "heart.pulmonary_valve",
    "heart.mitral_valve",
    "heart.aortic_valve",
)


@dataclass(slots=True)
class ViewSet:
    """Artefacts produced by one rendering pass."""

    output_dir: Path
    files: list[Path]

    def summary(self) -> dict[str, object]:
        """Names of the written files."""
        return {
            "output_dir": str(self.output_dir),
            "count": len(self.files),
            "files": sorted(path.name for path in self.files),
        }


def _entity_occupancy(spec: SceneSpec, entity_ids: Sequence[str], points: np.ndarray) -> np.ndarray:
    occupancy = np.zeros(points.shape[0], dtype=bool)
    for entity_id in entity_ids:
        if entity_id in spec.primitives:
            occupancy |= spec.primitives[entity_id].occupancy(points)
    return occupancy


def _ownership(
    spec: SceneSpec,
    entity_ids: Sequence[str],
    points: np.ndarray,
    slots: Mapping[str, int],
) -> np.ndarray:
    owner = np.full(points.shape[0], -1, dtype=np.int64)
    best = np.full(points.shape[0], 1 << 20, dtype=np.int64)
    for entity_id in entity_ids:
        primitive = spec.primitives.get(entity_id)
        if primitive is None:
            continue
        inside = primitive.occupancy(points)
        rank = spec.ownership_priority[entity_id]
        better = inside & (rank < best)
        owner[better] = slots[entity_id]
        best[better] = rank
    return owner


def render_ground_truth_views(
    spec: SceneSpec,
    ontology: AnatomyOntology,
    output_dir: Path,
    *,
    resolution: int = 160,
    axis: int = 2,
    grid_resolution: int = 48,
) -> ViewSet:
    """Render the corpus views for one scene."""
    slots = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    points = slice_points(resolution, axis, 0.0)
    visible = list(spec.visible_entity_ids(ontology))
    files: list[Path] = []

    files.append(
        write_png(
            output_dir / "01_whole_heart.png",
            render_slice(_entity_occupancy(spec, visible, points), resolution),
        )
    )
    for entity_id in ("heart.left_ventricle", "heart.aorta", "heart.mitral_valve"):
        if entity_id in spec.primitives:
            files.append(
                write_png(
                    output_dir / f"02_entity_{entity_id.split('.')[-1]}.png",
                    render_slice(_entity_occupancy(spec, [entity_id], points), resolution),
                )
            )
    files.append(
        write_png(
            output_dir / "03_four_chambers.png",
            render_ownership_slice(_ownership(spec, CHAMBERS, points, slots), resolution),
        )
    )
    files.append(
        write_png(
            output_dir / "04_valves.png",
            render_ownership_slice(_ownership(spec, VALVES, points, slots), resolution),
        )
    )
    files.append(
        write_png(
            output_dir / "05_ownership.png",
            render_ownership_slice(_ownership(spec, visible, points, slots), resolution),
        )
    )

    lod_images = []
    for lod in (1, 2, 3):
        entities = list(spec.visible_entity_ids(ontology, lod))
        lod_images.append(render_slice(_entity_occupancy(spec, entities, points), resolution))
        files.append(
            write_png(
                output_dir / f"06_lod{lod}.png",
                render_ownership_slice(_ownership(spec, entities, points, slots), resolution),
            )
        )

    kept = [entity for entity in visible if entity != "heart.aorta"]
    before = _entity_occupancy(spec, visible, points)
    after = _entity_occupancy(spec, kept, points)
    files.append(write_png(output_dir / "07_edit_before.png", render_slice(before, resolution)))
    files.append(write_png(output_dir / "07_edit_after.png", render_slice(after, resolution)))
    files.append(
        write_png(
            output_dir / "08_edit_difference.png",
            render_difference_slice(before.astype(float), after.astype(float), resolution),
        )
    )

    line = np.linspace(-1.0, 1.0, grid_resolution)
    grid_points = np.stack(np.meshgrid(line, line, line, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = _entity_occupancy(spec, visible, grid_points).reshape(
        grid_resolution, grid_resolution, grid_resolution
    )
    files.append(write_voxel_obj(output_dir / "whole_heart.obj", grid))
    cloud = np.concatenate(
        [
            spec.primitives[entity].sample_surface(200, np.random.default_rng(0))
            for entity in visible
        ]
    )
    files.append(write_point_cloud_ply(output_dir / "surface_points.ply", cloud))
    return ViewSet(output_dir, files)


def render_model_views(
    checkpoint: Path,
    spec: SceneSpec,
    ontology: AnatomyOntology,
    output_dir: Path,
    *,
    resolution: int = 160,
    axis: int = 2,
) -> ViewSet:
    """Render what a trained model produces for one scene, for side-by-side comparison."""
    from training.loop import ArmName, build_model

    domain = load_domain_config()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arm = str(payload["config"]["arm"])
    builder = BatchBuilder(ontology, domain)
    batch = builder.build([spec])
    model, _ = build_model(
        cast(ArmName, arm), builder, text_features=int(batch.text_features.shape[1])
    )
    model.load_state_dict(payload["model"])
    model.eval()

    points = torch.tensor(slice_points(resolution, axis, 0.0), dtype=torch.float32)
    probe = batch
    probe.scene_points = points.unsqueeze(0)
    probe.scene_occupancy = torch.zeros(1, points.shape[0])
    probe.part_owner = torch.full((1, points.shape[0]), -1, dtype=torch.int64)
    with torch.no_grad():
        output = model(probe)
        occupancy = torch.sigmoid(output.scene_logits)[0].numpy()
        owners = output.part_logits.argmax(dim=-1)[0].numpy()
    entity_count = batch.structure.entity_count
    owners = np.where(owners >= entity_count, -1, owners)

    files = [
        write_png(output_dir / f"model_{arm}_occupancy.png", render_slice(occupancy, resolution)),
        write_png(
            output_dir / f"model_{arm}_ownership.png",
            render_ownership_slice(owners.astype(np.int64), resolution),
        ),
    ]

    edited = batch
    hidden = int(batch.structure.visible_slots[-1].item())
    edited_features = batch.text_features.clone()
    edited_features[:, hidden] = 0.0
    probe.text_features = edited_features
    with torch.no_grad():
        edited_output = model(probe)
        edited_occupancy = torch.sigmoid(edited_output.scene_logits)[0].numpy()
    files.append(
        write_png(
            output_dir / f"model_{arm}_edit_drift.png",
            render_difference_slice(occupancy, edited_occupancy, resolution),
        )
    )
    del edited
    return ViewSet(output_dir, files)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the inspection views."""
    parser = argparse.ArgumentParser(description="Render Lagnav research views.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/views"))
    args = parser.parse_args(argv)

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    specs = load_split(args.corpus, args.split, limit=args.scene + 1)
    spec = specs[args.scene]
    views = render_ground_truth_views(spec, ontology, args.out / "ground_truth")
    summary = {"ground_truth": views.summary()}
    if args.checkpoint is not None:
        model_views = render_model_views(args.checkpoint, spec, ontology, args.out / "model")
        summary["model"] = model_views.summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
