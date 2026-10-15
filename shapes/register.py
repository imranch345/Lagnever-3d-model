"""Rigid alignment of subjects, and template correspondence per entity.

**Alignment.** Every Level 1 scene is already in the heart-centred viewer frame, so what is
left is each subject's rotation. The subject's heart shell is aligned to the reference's by
rigid ICP: no scaling, because size is one of the things the shape model has to capture. The
same transform then applies to every entity of that subject.

**Correspondence.** One template per entity, taken from the reference subject and decimated,
is deformed onto each subject's surface with Amberg's optimal-step non-rigid ICP. Afterwards
vertex ``i`` is the same place on every subject. The fit is measured both ways (template to
surface, surface to template) so a template that only covers part of the target cannot look
good. Surface fit does not prove anatomical correspondence; it is the necessary part that can
be measured.

Needs the ``assets`` extra plus ``rtree``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh
from trimesh.registration import icp, nricp_amberg

__all__ = [
    "DEFAULT_STEPS",
    "fit_error",
    "load_scene",
    "make_template",
    "register",
    "rigid_to",
]

FloatArray = npt.NDArray[np.float64]

#: (smoothness, landmark weight, normal weight, iterations) per stage, stiff to loose and a
#: final stage without normals. Checked on real left ventricles: about 0.2 mm mean surface fit.
DEFAULT_STEPS: tuple[tuple[float, float, float, int], ...] = (
    (0.01, 0.0, 0.5, 10),
    (0.02, 0.0, 0.5, 10),
    (0.03, 0.0, 0.5, 10),
    (0.01, 0.0, 0.0, 10),
)


def load_scene(scene_dir: Path) -> dict[str, trimesh.Trimesh]:
    """A built Level 1 scene's meshes by entity id, in its viewer frame."""
    scene = trimesh.load(scene_dir / "scene.glb", force="scene")
    assert isinstance(scene, trimesh.Scene)
    return {str(name): geometry for name, geometry in scene.geometry.items()}


def rigid_to(
    reference: trimesh.Trimesh,
    moving: trimesh.Trimesh,
    *,
    samples: int = 2000,
    iterations: int = 30,
    seed: int = 0,
) -> tuple[FloatArray, float]:
    """The rotation and translation that best moves ``moving`` onto ``reference``; and its cost.

    Two stages. Point-to-point ICP between surface samples (k-d tree) gets close quickly, but
    two independent random samplings bias it: on a 40 x 25 x 15 mm ellipsoid it recovers only
    about 8 of 10 degrees. Point-to-surface ICP against the reference, decimated for speed,
    then refines from there. Seeded sampling keeps the result reproducible.
    """
    a, _ = trimesh.sample.sample_surface(moving, samples, seed=seed)
    b, _ = trimesh.sample.sample_surface(reference, samples * 2, seed=seed + 1)
    points = np.asarray(a)
    coarse, _, _ = icp(
        points, np.asarray(b), max_iterations=iterations, reflection=False, scale=False
    )
    surface = (
        reference.simplify_quadric_decimation(face_count=4000)
        if len(reference.faces) > 4000
        else reference
    )
    matrix, _, cost = icp(
        points,
        surface,
        initial=coarse,
        max_iterations=iterations * 2,
        reflection=False,
        scale=False,
    )
    return np.asarray(matrix, dtype=np.float64), float(cost)


def make_template(mesh: trimesh.Trimesh, faces: int = 2000) -> trimesh.Trimesh:
    """The reference entity, decimated to a face budget, as the template to deform."""
    template = (
        mesh.simplify_quadric_decimation(face_count=faces)
        if len(mesh.faces) > faces
        else mesh.copy()
    )
    template.remove_unreferenced_vertices()
    return template


def register(
    template: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    steps: Sequence[Sequence[float]] = DEFAULT_STEPS,
    distance_threshold: float = 15.0,
) -> FloatArray:
    """The template's vertices, deformed onto ``target``. Same order, same faces."""
    vertices = nricp_amberg(
        template, target, steps=[list(s) for s in steps], distance_threshold=distance_threshold
    )
    return np.asarray(vertices, dtype=np.float64)


def fit_error(
    vertices: FloatArray, faces: npt.ArrayLike, target: trimesh.Trimesh, *, samples: int = 3000
) -> dict[str, float]:
    """Surface distance in millimetres, both ways, mean and max."""
    fitted = trimesh.Trimesh(vertices, np.asarray(faces), process=False)
    _, to_target, _ = trimesh.proximity.closest_point(target, vertices)
    points, _ = trimesh.sample.sample_surface(target, samples, seed=0)
    _, from_target, _ = trimesh.proximity.closest_point(fitted, points)
    return {
        "mean_to_target": float(np.mean(to_target)),
        "mean_from_target": float(np.mean(from_target)),
        "max_from_target": float(np.max(from_target)),
    }
