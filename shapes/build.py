"""Build Level 2a shape models from built Level 1 subjects, with a held-out evaluation.

For every usable subject (``quality_report.json``) and every requested entity:

1. align the subject to the reference subject (the one Level 1 chose) by its heart shell;
2. deform the reference's template onto the subject's entity;
3. keep the registration only if its surface fit is within ``max_fit_mm``, else record it as
   excluded with its error -- counted and reported, never silently dropped. A subject whose
   loading, alignment or registration raises is excluded the same way, with the error.

Registrations are cached per subject and entity, so an interrupted build resumes and a rebuild
with new model settings costs nothing.

Subjects are split once, deterministically, into training and held-out sets. Models are fitted
on training subjects only. The report, per entity, gives:

* **generality** -- mean vertex error reconstructing each held-out shape with the model,
  against the mean-shape baseline (the error of predicting the mean for everyone);
* **specificity** -- mean vertex distance from random samples (weights drawn from N(0, 1),
  clipped at 3) to the nearest training shape: low means samples look like real subjects;
* **compactness** -- modes kept and the variance they explain.

    python -m shapes.build --scenes datasets/processed/level1/real --out datasets/processed/level2a
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shapes.joint import fit_joint
from shapes.register import fit_error, load_scene, make_template, register, rigid_to
from shapes.ssm import ShapeModel, fit

__all__ = ["DEFAULT_ENTITIES", "build_models", "split_subjects"]

#: The first models: the four chambers, the myocardium and both lungs.
DEFAULT_ENTITIES: tuple[str, ...] = (
    "heart.left_atrium",
    "heart.left_ventricle",
    "heart.right_atrium",
    "heart.right_ventricle",
    "heart.myocardium",
    "respiratory.left_lung",
    "respiratory.right_lung",
)


def split_subjects(
    subjects: Sequence[str], holdout_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministic train / held-out split; at least one subject on each side."""
    ordered = sorted(subjects)
    order = np.random.default_rng(seed).permutation(len(ordered))
    n_hold = min(max(1, round(len(ordered) * holdout_fraction)), len(ordered) - 2)
    held = sorted(ordered[i] for i in order[:n_hold])
    train = sorted(ordered[i] for i in order[n_hold:])
    return train, held


def _mean_vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean())


def evaluate(
    model: ShapeModel, held_out: dict[str, np.ndarray], train: dict[str, np.ndarray], seed: int
) -> dict[str, Any]:
    """Generality against the mean baseline, specificity, and compactness."""
    generality = {s: _mean_vertex_distance(model.reconstruct(v), v) for s, v in held_out.items()}
    baseline = {s: _mean_vertex_distance(model.mean, v) for s, v in held_out.items()}
    rng = np.random.default_rng(seed)
    training = np.stack(list(train.values()))
    specificity = []
    for _ in range(50):
        sample = model.sample(np.clip(rng.standard_normal(model.n_modes), -3, 3))
        specificity.append(min(_mean_vertex_distance(sample, t) for t in training))
    return {
        "modes": model.n_modes,
        "explained": [round(float(x), 4) for x in model.explained],
        "explained_total": round(float(model.explained.sum()), 4),
        "generality_mm": round(float(np.mean(list(generality.values()))), 3) if held_out else None,
        "mean_baseline_mm": round(float(np.mean(list(baseline.values()))), 3) if held_out else None,
        "specificity_mm": round(float(np.mean(specificity)), 3),
        "held_out": {s: round(e, 3) for s, e in generality.items()},
    }


def build_models(
    scenes_root: Path,
    out_dir: Path,
    *,
    entities: Sequence[str] = DEFAULT_ENTITIES,
    template_faces: int = 2000,
    holdout_fraction: float = 0.2,
    max_fit_mm: float = 2.0,
    variance: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Register, fit, evaluate and save; return the report (also written to ``report.json``)."""
    quality = json.loads((scenes_root / "quality_report.json").read_text(encoding="utf-8"))
    reference_id = str(quality["chosen"])
    usable = sorted(q["subject"] for q in quality["subjects"] if q["usable"])
    reference = load_scene(scenes_root / reference_id)
    templates = {e: make_template(reference[e], template_faces) for e in entities}
    cache_dir = out_dir / "registrations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    shapes: dict[str, dict[str, np.ndarray]] = {e: {} for e in entities}
    excluded: dict[str, dict[str, str]] = {e: {} for e in entities}
    fits: dict[str, dict[str, dict[str, float]]] = {e: {} for e in entities}
    for subject in usable:
        cached = {e: cache_dir / f"{subject}__{e}.npz" for e in entities}
        scene = None
        matrix = np.eye(4)
        rigid_cost = 0.0
        for entity in entities:
            try:
                if cached[entity].is_file():
                    stored = np.load(cached[entity])
                    vertices, error = stored["vertices"], json.loads(str(stored["error"]))
                else:
                    if scene is None:
                        # The scene counts as loaded only once it is aligned, so a failed
                        # alignment can never leave later entities registered unaligned.
                        loaded = load_scene(scenes_root / subject)
                        matrix, rigid_cost = rigid_to(
                            reference["heart"], loaded["heart"], seed=seed
                        )
                        scene = loaded
                    if entity not in scene:
                        excluded[entity][subject] = "no mesh"
                        continue
                    target = scene[entity].copy()
                    target.apply_transform(matrix)
                    vertices = register(templates[entity], target)
                    error = fit_error(vertices, templates[entity].faces, target)
                    error["rigid_cost"] = rigid_cost
                    np.savez_compressed(cached[entity], vertices=vertices, error=json.dumps(error))
            except Exception as failure:
                # One bad subject must not end a build of hundreds. It is recorded, not
                # cached, so a rebuild tries it again.
                excluded[entity][subject] = f"failed: {type(failure).__name__}: {failure}"
                print(f"failed {subject} {entity}: {failure!r}", flush=True)
                continue
            fits[entity][subject] = error
            if error["mean_from_target"] > max_fit_mm:
                excluded[entity][subject] = f"fit {error['mean_from_target']:.2f} mm"
            else:
                shapes[entity][subject] = vertices
        print(f"registered {subject}", flush=True)

    train_ids, held_ids = split_subjects(usable, holdout_fraction, seed)
    report: dict[str, Any] = {
        "reference": reference_id,
        "subjects_usable": len(usable),
        "train": train_ids,
        "held_out": held_ids,
        "settings": {
            "template_faces": template_faces,
            "max_fit_mm": max_fit_mm,
            "variance": variance,
            "holdout_fraction": holdout_fraction,
            "seed": seed,
        },
        "entities": {},
    }
    for entity in entities:
        train = {s: shapes[entity][s] for s in train_ids if s in shapes[entity]}
        held = {s: shapes[entity][s] for s in held_ids if s in shapes[entity]}
        entry: dict[str, Any] = {
            "registered": len(fits[entity]),
            "excluded": excluded[entity],
            "fit_mean_mm": round(
                float(np.mean([f["mean_from_target"] for f in fits[entity].values()])), 3
            )
            if fits[entity]
            else None,
            "train": len(train),
            "held_out": len(held),
        }
        if len(train) >= 2:
            model = fit(
                entity,
                np.stack(list(train.values())),
                templates[entity].faces,
                tuple(train),
                variance=variance,
            )
            model.save(out_dir / "models" / f"{entity}.npz")
            entry.update(evaluate(model, held, train, seed))
        report["entities"][entity] = entry

    # One model over every entity, so samples stay consistent (shapes.joint). Only subjects
    # with every entity registered within the threshold contribute or are evaluated.
    complete_train = [s for s in train_ids if all(s in shapes[e] for e in entities)]
    complete_held = [s for s in held_ids if all(s in shapes[e] for e in entities)]
    if len(complete_train) >= 2:
        joint = fit_joint(
            shapes, {e: templates[e].faces for e in entities}, complete_train, variance=variance
        )
        joint.save(out_dir / "models" / "joint.npz")
        stacked = {
            s: np.concatenate([shapes[e][s] for e in entities])
            for s in complete_train + complete_held
        }
        joint_eval = evaluate(
            joint.model,
            {s: stacked[s] for s in complete_held},
            {s: stacked[s] for s in complete_train},
            seed,
        )
        report["joint"] = {
            "train": len(complete_train),
            "held_out": len(complete_held),
            **joint_eval,
        }
    reference_manifest = json.loads(
        (scenes_root / reference_id / "manifest.json").read_text(encoding="utf-8")
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(reference_manifest.get("source", {}), indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Build the models and print a one-line summary per entity."""
    parser = argparse.ArgumentParser(description="Build Level 2a shape models.")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--entities", nargs="+", default=list(DEFAULT_ENTITIES))
    parser.add_argument("--max-fit-mm", type=float, default=2.0)
    args = parser.parse_args(argv)
    report = build_models(args.scenes, args.out, entities=args.entities, max_fit_mm=args.max_fit_mm)
    joint = report.get("joint")
    if joint:
        print(
            f"joint: {joint['train']} train, {joint['modes']} modes "
            f"({joint['explained_total']}), generality {joint['generality_mm']} mm "
            f"vs mean {joint['mean_baseline_mm']} mm, specificity {joint['specificity_mm']} mm"
        )
    for entity, entry in report["entities"].items():
        print(
            f"{entity}: {entry.get('train')} train, {len(entry['excluded'])} excluded, "
            f"{entry.get('modes')} modes ({entry.get('explained_total')}), generality "
            f"{entry.get('generality_mm')} mm vs mean {entry.get('mean_baseline_mm')} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
