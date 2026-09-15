"""A machine-readable persistent-editing regression, so the report cites a run.

Step 8 changed the arrangement space, made placement predicted and rewrote the
level-of-detail objective. None of that is about editing, and all of it could break
editing by accident. This produces the artefact the completion report points at, rather
than the report asserting that editing still works.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.scene import AWRScene
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from datasets.whole_organ.sampling import EditOperation
from editing.scene_editor import SceneEditor
from experiments.step7.editing import OPERATIONS, build_edit_case
from generation.neural.nn.editing_head import EditHeadConfig, LocalEditHead, encode_edit

__all__ = ["editing_regression", "main"]


def _scene() -> AWRScene:
    from generation.generator import GenerationRequest, HeartSceneGenerator

    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    generator = HeartSceneGenerator(config=domain, ontology=ontology)
    return generator.generate(GenerationRequest(text="generate a human heart")).scene


def editing_regression() -> dict[str, Any]:
    """Run every check and return what happened, pass or fail."""
    checks: dict[str, dict[str, Any]] = {}

    scene = _scene()
    editor = SceneEditor(scene)
    before = {entity.entity_id: entity.state.opacity for entity in scene.iter_entities()}
    editor.set_opacity(["heart.left_ventricle"], 0.25)
    after = {entity.entity_id: entity.state.opacity for entity in scene.iter_entities()}
    drifted = [
        entity_id
        for entity_id, opacity in before.items()
        if entity_id != "heart.left_ventricle" and abs(after[entity_id] - opacity) > 1e-9
    ]
    checks["edit_is_local"] = {
        "passed": not drifted and abs(after["heart.left_ventricle"] - 0.25) < 1e-9,
        "untouched_entities_that_moved": drifted,
        "target_opacity": after["heart.left_ventricle"],
    }

    editor.hide("heart.pericardium")
    restored = AWRScene.from_dict(scene.to_dict())
    checks["edit_survives_reload"] = {
        "passed": (
            abs(restored.get("heart.left_ventricle").state.opacity - 0.25) < 1e-9
            and restored.get("heart.pericardium").state.visibility is False
        ),
        "opacity_after_reload": restored.get("heart.left_ventricle").state.opacity,
    }

    second = SceneEditor(restored)
    second.set_opacity(["heart.tricuspid_valve"], 0.8)
    checks["first_edit_survives_a_second"] = {
        "passed": abs(restored.get("heart.left_ventricle").state.opacity - 0.25) < 1e-9,
        "first_edit_opacity": restored.get("heart.left_ventricle").state.opacity,
        "second_edit_opacity": restored.get("heart.tricuspid_valve").state.opacity,
    }

    identity_before = [entity.entity_id for entity in restored.iter_entities()]
    second.set_lod(2)
    identity_after = [entity.entity_id for entity in restored.iter_entities()]
    checks["identity_is_stable"] = {
        "passed": identity_before == identity_after,
        "entities": len(identity_after),
    }

    head = LocalEditHead(
        EditHeadConfig(
            entity_width=256,
            token_width=64,
            tokens=32,
            operations=len(OPERATIONS),
            identity_width=96,
            scope="target",
        )
    )
    with torch.no_grad():
        for parameter in head.delta[-1].parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
        head.gate.bias.add_(4.0)
    tokens = torch.randn(1, 6, 32, 64)
    latent = torch.randn(1, 6, 256)
    mask = torch.zeros(1, 6)
    mask[0, 2] = 1.0
    edited = head(tokens, latent, encode_edit([0], [1.4], operations=len(OPERATIONS)), mask)
    leaked = [
        slot for slot in range(6) if slot != 2 and not torch.equal(edited[0, slot], tokens[0, slot])
    ]
    checks["edit_head_scopes_to_target"] = {
        "passed": not leaked and not torch.equal(edited[0, 2], tokens[0, 2]),
        "entities_that_leaked": leaked,
    }

    rng = np.random.default_rng(0)
    organ = build_scene(scene_index=0, family_id=0, lod=3, arrangement=sample_arrangement(rng))
    case = build_edit_case(organ, EditOperation.THICKEN_VALVE)
    checks["edit_pairs_build_on_continuous_arrangements"] = {
        "passed": bool(case.moved)
        and bool(case.still)
        and case.change_fraction[case.target] > 0.05,
        "target": case.target,
        "target_change_fraction": case.change_fraction[case.target],
        "entities_moved": len(case.moved),
        "entities_still": len(case.still),
    }

    return {
        "experiment_id": "step8-editing-regression",
        "scope": (
            "Regression only. Step 7 recorded persistent editing as REVISE, with locality "
            "working and accuracy not; Step 8 deliberately did not spend compute on it."
        ),
        "checks": checks,
        "all_passed": all(entry["passed"] for entry in checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 8 editing regression.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = editing_regression()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, entry in report["checks"].items():
        print(f"{'PASS' if entry['passed'] else 'FAIL'}  {name}")
    print(f"all passed: {report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
