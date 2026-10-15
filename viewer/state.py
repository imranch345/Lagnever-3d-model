"""What the page needs to draw a scene: per-entity state, joined with what has geometry.

Kept separate from the server so it can be tested without sockets. The page never infers
anything from the AWR itself; everything it shows -- including "no geometry yet" for
structures the data lacks (ADR-L1-002) -- is decided here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from animation import CardiacFlowModel
from reasoning.command_engine import CommandEngine

__all__ = ["DEFAULT_COLOURS", "animation_schedule", "colour_for", "scene_state"]

#: Presentation only. Left-heart and systemic-arterial structures red (oxygenated),
#: right-heart and venous structures blue, lungs pink, airway pale.
DEFAULT_COLOURS: dict[str, str] = {
    "heart": "#b8323a",
    "heart.myocardium": "#8e2430",
    "heart.left_atrium": "#d9414b",
    "heart.left_ventricle": "#c7303b",
    "heart.right_atrium": "#4a6fd1",
    "heart.right_ventricle": "#3b5cc0",
    "heart.aorta": "#e0454f",
    "heart.pulmonary_trunk": "#5b7fe0",
    "heart.inferior_vena_cava": "#3d5bb5",
    "respiratory.left_lung": "#e8a3ad",
    "respiratory.right_lung": "#e8a3ad",
    "respiratory.lung_upper_lobe_left": "#eab0b8",
    "respiratory.lung_lower_lobe_left": "#de96a1",
    "respiratory.lung_upper_lobe_right": "#eab0b8",
    "respiratory.lung_middle_lobe_right": "#f0c1c7",
    "respiratory.lung_lower_lobe_right": "#de96a1",
    "respiratory.trachea": "#ece3cf",
}
_FALLBACK = "#b9b3a9"


def colour_for(entity_id: str, base_color: Any = None) -> str:
    """The engine's material colour if one was set, else the default for the entity."""
    if base_color is not None and len(base_color) == 3:
        r, g, b = (max(0, min(255, round(float(c) * 255))) for c in base_color)
        return f"#{r:02x}{g:02x}{b:02x}"
    return DEFAULT_COLOURS.get(entity_id, _FALLBACK)


def animation_schedule(engine: CommandEngine) -> dict[str, Any] | None:
    """The playing clip's phases and per-phase roles, or ``None`` when nothing is animated.

    The engine stores on each entity only the role it plays at its own phase; playback needs
    every phase's roles, so the clip is rebuilt from the same scene and cycle configuration the
    engine bound it from. Rebuilding is deterministic: the clip is a function of the graph.
    """
    scene = engine.scene
    if scene is None:
        return None
    clips = {
        scene.get(i).state.animation.clip_id
        for i in scene.ids()
        if scene.get(i).state.animation.playing
    }
    if not clips:
        return None
    clip = CardiacFlowModel(scene, engine.config.cardiac_cycle).build_clip()
    phased = {entity for phase in clip.phases for entity in phase.roles}
    on_paths = {entity for path in clip.paths for entity in path.entity_ids}
    return {
        "clip_id": clip.clip_id,
        "duration_seconds": clip.duration_seconds,
        "phases": [phase.to_dict() for phase in clip.phases],
        # On a flow path but given no phase role (the lungs, the coronary vessels): they carry
        # flow throughout the cycle, which is how the engine binds them too.
        "carrying_flow": sorted(on_paths - phased),
        "semantic_only": True,
    }


def scene_state(engine: CommandEngine, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The scene as the page draws it.

    ``has_geometry`` is true only for entities with a mesh node in the manifest. An entity
    that is visible but has no geometry is still reported visible, so the page can say so
    instead of silently drawing nothing.
    """
    scene = engine.scene
    if scene is None:
        raise RuntimeError("no scene: generate one before asking for its state")
    with_mesh = set(manifest.get("entities", {}))
    entities = []
    for entity_id in scene.ids():
        entity = scene.get(entity_id)
        state = entity.state
        entities.append(
            {
                "id": entity_id,
                "name": entity.display_name,
                "parent": entity.parent_id,
                "group": entity.is_group,
                "renderable": entity.renderable,
                "visible": bool(state.visibility),
                "opacity": float(state.opacity),
                "colour": colour_for(entity_id, state.material.base_color),
                "has_geometry": entity_id in with_mesh,
                "lod": int(entity.lod),
            }
        )
    source = manifest.get("source", {})
    return {
        "scene_version": scene.version,
        "active_lod": scene.active_lod,
        "educational_level": str(scene.educational_level),
        "entities": entities,
        "credits": {
            "dataset": source.get("dataset"),
            "version": source.get("version"),
            "license": source.get("license"),
            "attribution": source.get("attribution"),
            "url": source.get("url"),
            "subject": manifest.get("subject"),
        },
        "medical_status": manifest.get(
            "medical_status", "research data, not validated; not for clinical use"
        ),
        "animation": animation_schedule(engine),
    }
