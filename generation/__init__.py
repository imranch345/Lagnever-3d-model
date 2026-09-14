"""Generation layer: text encoding, anatomical reasoning, scene construction.

The deterministic pipeline implemented here proves the representation, not the
graphics: it produces a fully structured AWR scene and reserved geometry slots,
and no geometry at all. Future neural components are declared in
``generation/neural_interfaces.py``.
"""

from generation.anatomical_reasoner import (
    AnatomicalReasoner,
    AWRConstructionPlan,
    NeuralAnatomicalReasoner,
    OntologyDrivenReasoner,
)
from generation.generator import (
    GenerationRequest,
    GenerationResult,
    HeartSceneGenerator,
    SceneGenerator,
)
from generation.neural_interfaces import (
    STEP5_OPEN_DECISIONS,
    AnimationDecoder,
    LatentRequest,
    MaterialDecoder,
    ThreeDLatent,
    ThreeDLatentModel,
)
from generation.text_encoder import (
    LanguageEncoder,
    LanguageEncoding,
    LexicalLanguageEncoder,
    NeuralLanguageEncoder,
    TermMatch,
)

__all__ = [
    "STEP5_OPEN_DECISIONS",
    "AWRConstructionPlan",
    "AnatomicalReasoner",
    "AnimationDecoder",
    "GenerationRequest",
    "GenerationResult",
    "HeartSceneGenerator",
    "LanguageEncoder",
    "LanguageEncoding",
    "LatentRequest",
    "LexicalLanguageEncoder",
    "MaterialDecoder",
    "NeuralAnatomicalReasoner",
    "NeuralLanguageEncoder",
    "OntologyDrivenReasoner",
    "SceneGenerator",
    "TermMatch",
    "ThreeDLatent",
    "ThreeDLatentModel",
]
