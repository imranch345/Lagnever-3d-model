"""Shared fixtures.

The ontology and configuration are immutable, so they are session scoped. Scenes
and engines are function scoped because tests mutate them, and a leaked mutation
between tests would hide exactly the persistence bugs these tests exist to catch.
"""

from __future__ import annotations

import pytest

from awr.config import DomainConfig, load_domain_config, load_model_config
from awr.ontology import AnatomyOntology, load_ontology
from awr.scene import AWRScene
from editing.scene_editor import SceneEditor
from generation.generator import GenerationRequest, HeartSceneGenerator
from reasoning.command_engine import CommandEngine
from reasoning.entity_resolver import EntityResolver

# The Step 6 prototype tests need PyTorch, which is an optional extra (ADR 0010).
# Without it the tier-one suite still runs in full; the prototype modules are skipped
# rather than failing at collection, so the AWR core stays testable with one dependency.
try:  # pragma: no cover - depends on the installed extras
    import torch  # noqa: F401

    PROTOTYPE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the minimal environment
    PROTOTYPE_AVAILABLE = False

if PROTOTYPE_AVAILABLE:
    pytest_plugins = ("tests.conftest_prototype",)
else:
    collect_ignore_glob = [
        "test_prototype_*.py",
        "test_synthetic_corpus.py",
        "test_experiment_runner.py",
        "test_visualization.py",
    ]


@pytest.fixture(scope="session")
def config() -> DomainConfig:
    """Typed heart domain configuration."""
    return load_domain_config()


@pytest.fixture(scope="session")
def model_config():
    """Typed model configuration (all placeholders)."""
    return load_model_config()


@pytest.fixture(scope="session")
def ontology(config: DomainConfig) -> AnatomyOntology:
    """Heart Ontology v0.1, loaded and validated."""
    return load_ontology(
        config.domain.ontology_dir, expected_version=config.domain.ontology_version
    )


@pytest.fixture
def generator(config: DomainConfig, ontology: AnatomyOntology) -> HeartSceneGenerator:
    """Deterministic heart generator."""
    return HeartSceneGenerator(config=config, ontology=ontology)


@pytest.fixture
def scene(generator: HeartSceneGenerator) -> AWRScene:
    """A freshly generated heart scene."""
    return generator.generate(GenerationRequest(text="generate a human heart")).scene


@pytest.fixture
def editor(scene: AWRScene) -> SceneEditor:
    """Editor bound to the generated scene."""
    return SceneEditor(scene)


@pytest.fixture
def resolver(ontology: AnatomyOntology, config: DomainConfig) -> EntityResolver:
    """Entity resolver over the heart ontology."""
    return EntityResolver(ontology, config.resolver)


@pytest.fixture
def engine(config: DomainConfig, ontology: AnatomyOntology) -> CommandEngine:
    """Command engine with no scene yet."""
    return CommandEngine(config=config, ontology=ontology)


@pytest.fixture
def live_engine(engine: CommandEngine) -> CommandEngine:
    """Command engine with a generated heart scene already in place."""
    response = engine.execute("generate a human heart")
    assert response.ok, response.message
    return engine
