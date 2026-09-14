"""Query-point sampling and exact supervision labels for the tier-0 corpus.

SYNTHETIC_RESEARCH_DATA. NOT_MEDICALLY_VALIDATED.

Every label here is computed analytically from the primitives, so occupancy, part
ownership and frames are exact rather than annotated. Sampling is seeded per scene,
which makes a training run reproducible down to the query points.

Two families of points are produced, because the architecture needs both:

* **scene points** in scene space, with scene occupancy and a part owner. These
  supervise the shared field and the part-label head, and they are what the drift
  and part-control metrics are measured on.
* **entity points** per entity, concentrated around that entity, with its own
  occupancy. These supervise the per-entity fields, which is the representation
  under test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec
from datasets.synthetic.primitives import Primitive

__all__ = ["SampledScene", "SamplingConfig", "sample_scene", "surface_points"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How many points to draw and where from."""

    scene_points: int = 1024
    entity_points: int = 32
    near_surface_fraction: float = 0.5
    inside_fraction: float = 0.2
    surface_jitter: float = 0.02
    entity_surface_fraction: float = 0.6
    resample_per_epoch: bool = False
    entity_slots: int = 64
    """Padded entity-slot count, matching the ``N_ENT`` dimension of the Step 5 contracts."""

    def __post_init__(self) -> None:
        if self.scene_points <= 0 or self.entity_points <= 0:
            raise ValueError("Point counts must be positive.")
        if self.entity_slots <= 0:
            raise ValueError("entity_slots must be positive.")
        if self.near_surface_fraction + self.inside_fraction > 1.0:
            raise ValueError(
                "near_surface_fraction plus inside_fraction cannot exceed 1.0; the remainder "
                "is the uniform share."
            )


@dataclass(frozen=True, slots=True)
class SampledScene:
    """One scene's points and exact labels."""

    scene_id: str
    entity_slots: tuple[int, ...]
    entity_ids: tuple[str, ...]
    scene_points: FloatArray
    scene_occupancy: BoolArray
    part_owner: IntArray
    entity_points: FloatArray
    entity_occupancy: BoolArray
    entity_frames: FloatArray
    visible_mask: BoolArray

    def counts(self) -> dict[str, int]:
        """Shapes as a dictionary, for tests and logging."""
        return {
            "scene_points": int(self.scene_points.shape[0]),
            "entities": int(self.entity_points.shape[0]),
            "entity_points": int(self.entity_points.shape[1]),
            "occupied_scene_points": int(self.scene_occupancy.sum()),
            "background_points": int((self.part_owner < 0).sum()),
        }


def surface_points(
    primitive: Primitive, count: int, rng: np.random.Generator, jitter: float
) -> FloatArray:
    """Points near a primitive's surface, jittered inward and outward."""
    samples = primitive.sample_surface(count, rng)
    offsets = rng.normal(scale=jitter, size=samples.shape)
    return np.asarray(samples + offsets, dtype=np.float64)


def _owner_of(
    points: FloatArray,
    primitives: Mapping[str, Primitive],
    slots: Mapping[str, int],
    priority: Mapping[str, int],
) -> IntArray:
    """Part owner per point: the containing entity with the smallest priority."""
    best_priority = np.full(points.shape[0], 1 << 20, dtype=np.int64)
    owner = np.full(points.shape[0], -1, dtype=np.int64)
    for entity_id, primitive in primitives.items():
        inside = primitive.occupancy(points)
        if not inside.any():
            continue
        rank = int(priority[entity_id])
        better = inside & (rank < best_priority)
        owner[better] = slots[entity_id]
        best_priority[better] = rank
    return owner


def sample_scene(
    spec: SceneSpec,
    ontology: AnatomyOntology,
    config: SamplingConfig | None = None,
    *,
    epoch: int = 0,
) -> SampledScene:
    """Sample points and compute exact labels for one scene."""
    settings = config or SamplingConfig()
    salt = epoch if settings.resample_per_epoch else 0
    rng = np.random.default_rng(spec.seed * 7919 + salt)

    visible = spec.visible_entity_ids(ontology)
    if not visible:
        raise ValueError(f"Scene {spec.scene_id} has no visible entity at LOD {spec.active_lod}.")
    slots = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
    visible_primitives = {entity_id: spec.primitives[entity_id] for entity_id in visible}

    total = settings.scene_points
    near = int(total * settings.near_surface_fraction)
    inside = int(total * settings.inside_fraction)
    uniform = total - near - inside

    chunks: list[FloatArray] = [rng.uniform(-1.0, 1.0, size=(uniform, 3))]
    order = list(visible)
    if near:
        per_entity = max(1, near // len(order))
        near_chunks = [
            surface_points(visible_primitives[entity_id], per_entity, rng, settings.surface_jitter)
            for entity_id in order
        ]
        chunks.append(np.concatenate(near_chunks)[:near])
    if inside:
        per_entity = max(1, inside // len(order))
        inside_chunks: list[FloatArray] = []
        for entity_id in order:
            primitive = visible_primitives[entity_id]
            candidates = primitive.sample_surface(per_entity, rng)
            pulled = primitive.centre + (candidates - primitive.centre) * rng.uniform(
                0.0, 0.85, size=(candidates.shape[0], 1)
            )
            inside_chunks.append(pulled)
        chunks.append(np.concatenate(inside_chunks)[:inside])

    scene_pts = np.clip(np.concatenate(chunks)[:total], -1.0, 1.0)
    if scene_pts.shape[0] < total:  # pragma: no cover - only when a scene has few entities
        extra = rng.uniform(-1.0, 1.0, size=(total - scene_pts.shape[0], 3))
        scene_pts = np.concatenate([scene_pts, extra])

    scene_occ = np.zeros(scene_pts.shape[0], dtype=np.bool_)
    for primitive in visible_primitives.values():
        scene_occ |= primitive.occupancy(scene_pts)
    owner = _owner_of(scene_pts, visible_primitives, slots, spec.ownership_priority)

    entity_count = settings.entity_slots
    if entity_count < len(ontology.ids()):
        raise ValueError(
            f"entity_slots={entity_count} is smaller than the ontology's "
            f"{len(ontology.ids())} entities."
        )
    entity_pts = np.zeros((entity_count, settings.entity_points, 3), dtype=np.float64)
    entity_occ = np.zeros((entity_count, settings.entity_points), dtype=np.bool_)
    frames = np.zeros((entity_count, 12), dtype=np.float64)
    visible_flags = np.zeros(entity_count, dtype=np.bool_)

    surface_count = int(settings.entity_points * settings.entity_surface_fraction)
    for entity_id, primitive in visible_primitives.items():
        slot = slots[entity_id]
        visible_flags[slot] = True
        frames[slot] = primitive.frame()
        near_pts = surface_points(primitive, surface_count, rng, settings.surface_jitter)
        box = np.maximum(primitive.extent, 1e-3) * 1.6
        far_pts = primitive.centre + rng.uniform(
            -1.0, 1.0, size=(settings.entity_points - surface_count, 3)
        ) * box
        points = np.clip(np.concatenate([near_pts, far_pts]), -1.0, 1.0)
        entity_pts[slot] = points
        entity_occ[slot] = primitive.occupancy(points)

    return SampledScene(
        scene_id=spec.scene_id,
        entity_slots=tuple(slots[entity_id] for entity_id in visible),
        entity_ids=tuple(visible),
        scene_points=scene_pts,
        scene_occupancy=scene_occ,
        part_owner=owner,
        entity_points=entity_pts,
        entity_occupancy=entity_occ,
        entity_frames=frames,
        visible_mask=visible_flags,
    )


def surface_point_cloud(
    spec: SceneSpec,
    ontology: AnatomyOntology,
    count: int = 2048,
    *,
    lod: int | None = None,
    seed: int = 0,
    entity_ids: Sequence[str] | None = None,
) -> FloatArray:
    """Reference surface points for a scene, used by the Chamfer metric."""
    rng = np.random.default_rng(seed)
    chosen = tuple(entity_ids) if entity_ids is not None else spec.visible_entity_ids(ontology, lod)
    if not chosen:
        return np.zeros((0, 3), dtype=np.float64)
    per_entity = max(1, count // len(chosen))
    clouds = [
        spec.primitives[entity_id].sample_surface(per_entity, rng) for entity_id in chosen
    ]
    return np.asarray(np.concatenate(clouds)[:count], dtype=np.float64)
