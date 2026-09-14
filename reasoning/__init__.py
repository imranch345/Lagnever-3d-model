"""Reasoning layer: parsing, entity resolution, explanation and orchestration.

The layer is deliberately free of anatomy knowledge. Anatomy lives in the
ontology and the AWR; this layer only turns language into typed operations
against them.
"""

from reasoning.command_engine import CommandEngine, CommandResponse
from reasoning.entity_resolver import EntityResolver, MatchKind, Resolution, normalize_phrase
from reasoning.explanation import (
    Explanation,
    LanguageModelExplainer,
    SceneExplainer,
    TemplateSceneExplainer,
)
from reasoning.parser import (
    CommandParser,
    Intent,
    LanguageModelCommandParser,
    ParsedCommand,
    RuleBasedCommandParser,
    SelectorKind,
    TargetSelector,
)

__all__ = [
    "CommandEngine",
    "CommandParser",
    "CommandResponse",
    "EntityResolver",
    "Explanation",
    "Intent",
    "LanguageModelCommandParser",
    "LanguageModelExplainer",
    "MatchKind",
    "ParsedCommand",
    "Resolution",
    "RuleBasedCommandParser",
    "SceneExplainer",
    "SelectorKind",
    "TargetSelector",
    "TemplateSceneExplainer",
    "normalize_phrase",
]
