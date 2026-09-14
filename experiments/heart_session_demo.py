"""Runnable demonstration of one Lagnav session.

Runs the canonical command sequence from the project specification against a
single persistent scene and prints what the AWR does in response. Nothing is
rendered: what you see is the representation changing.

Run it with::

    python -m experiments.heart_session_demo
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from awr.validation import validate_scene
from evaluation.consistency import evaluate_persistence
from evaluation.geometry import evaluate_correspondence
from reasoning.command_engine import CommandEngine

DEMO_SESSION: tuple[str, ...] = (
    "Generate a human heart",
    "Show the four chambers",
    "Hide everything except the chambers",
    "Make the left ventricle transparent",
    "Show the valves",
    "Show more detail",
    "Switch to medical level",
    "Animate blood flow",
    "Explain what is happening",
)


def run_session(commands: Sequence[str] = DEMO_SESSION, *, verbose: bool = True) -> CommandEngine:
    """Run a command sequence against one persistent scene."""
    engine = CommandEngine()
    for text in commands:
        response = engine.execute(text)
        if not verbose:
            continue
        print(f"\n> {text}")
        status = "ok" if response.ok else f"error [{response.error_code}]"
        print(f"  {status}: {response.message}")
        scene = engine.scene
        if scene is not None:
            print(
                f"  scene v{scene.version} | lod {scene.active_lod} ({scene.lod_spec().name}) "
                f"| visible {len(scene.visible_ids())}/{len(scene.renderable_ids())} "
                f"| entities {len(scene)}"
            )
    return engine


def print_report(engine: CommandEngine) -> None:
    """Print identity, validation and persistence checks after the session."""
    scene = engine.scene
    if scene is None:
        print("No scene was produced.")
        return

    lv = scene.get("heart.left_ventricle")
    print("\n--- persistent identity ---")
    print(f"  entity_id           : {lv.entity_id}")
    print(f"  visibility / opacity: {lv.visibility} / {lv.opacity:g}")
    component = lv.geometry_reference.component_id if lv.geometry_reference else None
    print(f"  geometry component  : {component}")
    print(f"  animation           : {lv.animation_state.clip_id} ({lv.animation_state.role})")

    print("\n--- scene history ---")
    for event in scene.history:
        print(f"  v{event.version:<2} {event.operation:<22} {event.summary}")

    print("\n--- checks ---")
    checks = (validate_scene(scene), evaluate_correspondence(scene), evaluate_persistence(scene))
    for report in checks:
        print(f"  {report.summary()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m experiments.heart_session_demo``."""
    parser = argparse.ArgumentParser(description="Run a Lagnav 3D heart session demo.")
    parser.add_argument("--quiet", action="store_true", help="only print the final report")
    parser.add_argument(
        "--save", metavar="PATH", default=None, help="write the resulting scene to a JSON file"
    )
    args = parser.parse_args(argv)

    engine = run_session(verbose=not args.quiet)
    print_report(engine)
    if args.save and engine.scene is not None:
        path = engine.scene.save(args.save)
        print(f"\nScene written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
