"""Animation layer: semantic, graph-derived cardiac behaviour.

No geometry is deformed here. The layer turns the functional graph into a typed
schedule of entities, phases and roles that a future animation decoder can
consume.
"""

from animation.heart_flow import (
    AnimationClip,
    CardiacFlowModel,
    CyclePhase,
    FlowPath,
    FlowRole,
    FlowSegment,
)

__all__ = [
    "AnimationClip",
    "CardiacFlowModel",
    "CyclePhase",
    "FlowPath",
    "FlowRole",
    "FlowSegment",
]
