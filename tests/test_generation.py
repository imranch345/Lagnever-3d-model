"""Generation pipeline: encoding, planning, scene construction, neural placeholders."""

from __future__ import annotations

import pytest

from awr.errors import NotYetImplementedError, OntologyError
from awr.ontology import AnatomyOntology
from awr.schema import EducationalLevel
from generation.anatomical_reasoner import NeuralAnatomicalReasoner, OntologyDrivenReasoner
from generation.generator import GenerationRequest, HeartSceneGenerator
from generation.neural_interfaces import (
    STEP5_OPEN_DECISIONS,
    AnimationDecoder,
    MaterialDecoder,
    ThreeDLatent,
    ThreeDLatentModel,
)
from generation.text_encoder import LexicalLanguageEncoder, NeuralLanguageEncoder


def test_pipeline_stages_are_recorded(generator: HeartSceneGenerator) -> None:
    """A generation result exposes every stage it went through."""
    result = generator.generate(GenerationRequest(text="Generate a human heart"))
    assert result.encoding.encoder_id == "lexical-v0.1"
    assert result.plan.reasoner_id == "ontology-driven-v0.1"
    assert result.scene.metadata["generator"] == "heart-deterministic-v0.1"
    assert len(result.correspondence) == len(result.scene.renderable_ids())


def test_lexical_encoder_produces_no_embedding(ontology: AnatomyOntology) -> None:
    """The symbolic encoder is explicit about not being a model."""
    encoding = LexicalLanguageEncoder(ontology).encode("show the left ventricle and the aorta")
    assert encoding.embedding is None
    assert encoding.is_neural is False
    assert encoding.entity_ids == ("heart.left_ventricle", "heart.aorta")


def test_encoder_prefers_the_longest_matching_term(ontology: AnatomyOntology) -> None:
    """Overlapping terms resolve to the most specific match."""
    encoding = LexicalLanguageEncoder(ontology).encode("the left ventricular chamber")
    assert encoding.entity_ids == ("heart.left_ventricle",)
    assert encoding.term_matches[0].matched_text == "left ventricular chamber"


def test_plan_instantiates_the_whole_ontology(generator: HeartSceneGenerator) -> None:
    """The AWR always holds more than the scene shows."""
    result = generator.generate(GenerationRequest(text="generate a human heart"))
    assert len(result.plan.entity_ids) == len(generator.ontology)
    assert result.plan.focus_entity_ids == ("heart",)
    assert len(result.scene.visible_ids()) < len(result.plan.entity_ids)


def test_requests_outside_the_domain_are_refused(generator: HeartSceneGenerator) -> None:
    """The prototype covers the heart only, and says so."""
    with pytest.raises(OntologyError, match="human heart only"):
        generator.generate(GenerationRequest(text="generate a liver"))


def test_explicit_level_override(generator: HeartSceneGenerator) -> None:
    """A caller can pin the audience level and level of detail."""
    result = generator.generate(
        GenerationRequest(
            text="generate a human heart",
            lod=4,
            educational_level=EducationalLevel.MEDICAL,
        )
    )
    assert result.scene.active_lod == 4
    assert result.scene.educational_level is EducationalLevel.MEDICAL
    assert result.scene.get("heart.bundle_of_his").visibility is True


def test_generation_without_geometry_allocation(generator: HeartSceneGenerator) -> None:
    """Geometry reservation can be switched off; the anatomy is unaffected."""
    result = generator.generate(
        GenerationRequest(text="generate a human heart", allocate_geometry=False)
    )
    assert len(result.correspondence) == 0
    assert result.scene.get("heart.aorta").geometry_reference is None
    assert len(result.scene) == 42


def test_reasoner_reports_its_rationale(ontology: AnatomyOntology) -> None:
    """The deterministic reasoner explains what it did."""
    encoder = LexicalLanguageEncoder(ontology)
    reasoner = OntologyDrivenReasoner(default_lod=1, default_level=EducationalLevel.SCHOOL)
    plan = reasoner.plan(encoder.encode("generate a human heart"), ontology)
    assert any("Matched ontology terms" in note for note in plan.rationale)
    assert plan.summary()["entities"] == len(ontology)


@pytest.mark.parametrize("cls", [NeuralLanguageEncoder, NeuralAnatomicalReasoner])
def test_neural_stand_ins_are_declared_only(cls: type) -> None:
    """No neural component can be constructed before Step 5 defines it."""
    with pytest.raises(NotYetImplementedError, match="Step 5"):
        cls()


@pytest.mark.parametrize(
    ("cls", "method"),
    [(ThreeDLatentModel, "encode"), (MaterialDecoder, "decode"), (AnimationDecoder, "decode")],
)
def test_neural_interfaces_cannot_be_brought_to_life(cls: type, method: str) -> None:
    """Even a subclass that implements the interface refuses to construct.

    The abstract methods make these real interfaces; the constructor guard means
    a subclass cannot quietly become a working component before Step 5 decides
    what the component is.
    """
    subclass = type("Attempt", (cls,), {method: lambda self, *args, **kwargs: None})
    with pytest.raises(NotYetImplementedError, match="Step 5"):
        subclass()


def test_step5_open_decisions_are_recorded() -> None:
    """The decisions Step 5 owns are listed in code, not only in prose."""
    assert set(STEP5_OPEN_DECISIONS) == {
        "model architecture",
        "tensor schema",
        "latent dimensionality",
        "training objectives",
        "attention mechanisms",
        "multimodal fusion",
        "3D representation",
        "geometry decoding strategy",
    }


def test_latent_payload_is_unconstrained() -> None:
    """The latent type is left open on purpose."""
    latent = ThreeDLatent(scene_id="s", entity_ids=("heart",))
    assert latent.is_populated is False
    assert latent.payload is None
