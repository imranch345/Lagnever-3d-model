"""Shared fixtures for the Step 6 prototype tests.

Imported by ``tests/conftest.py`` so the heavy PyTorch fixtures live apart from the
Step 4 fixtures and can be skipped as a group if the prototype extra is absent.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import SceneSpec, build_scene
from datasets.synthetic.sampling import SamplingConfig
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch


@pytest.fixture(scope="session")
def scene_specs(ontology: AnatomyOntology) -> list[SceneSpec]:
    """A small set of generated scenes at one level of detail."""
    return [
        build_scene(ontology, scene_index=index, family_id=index % 4, lod=2)
        for index in range(8)
    ]


@pytest.fixture(scope="session")
def mixed_lod_specs(ontology: AnatomyOntology) -> list[SceneSpec]:
    """Scenes spanning several levels of detail."""
    return [
        build_scene(ontology, scene_index=100 + index, family_id=index % 4, lod=(index % 3) + 1)
        for index in range(12)
    ]


@pytest.fixture
def builder(ontology: AnatomyOntology, config: DomainConfig) -> BatchBuilder:
    """Batch builder with small point counts, for fast tests."""
    return BatchBuilder(
        ontology, config, sampling=SamplingConfig(scene_points=96, entity_points=12)
    )


@pytest.fixture
def batch(builder: BatchBuilder, scene_specs: Sequence[SceneSpec]) -> PrototypeBatch:
    """One batch of four scenes."""
    return builder.build(list(scene_specs[:4]))
