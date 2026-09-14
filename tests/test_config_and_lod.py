"""Configuration loading and the level-of-detail policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from awr.config import (
    CardiacCycleConfig,
    CardiacCyclePhaseSpec,
    DomainConfig,
    ModelConfig,
    load_domain_config,
)
from awr.errors import ConfigError, InvalidStateError, SchemaError
from awr.lod import (
    EducationalLevelMap,
    EducationalLevelSpec,
    LodLadder,
    LodLevelSpec,
    default_visibility_at,
)
from awr.paths import repo_root
from awr.schema import EducationalLevel


def test_domain_config_is_typed(config: DomainConfig) -> None:
    """Configuration arrives as typed objects, not nested dictionaries."""
    assert config.domain.id == "heart"
    assert config.domain.ontology_dir.is_dir()
    assert config.scene_defaults.educational_level is EducationalLevel.SCHOOL
    assert config.opacity_presets.get("transparent") == 0.3
    assert config.opacity_presets.get("opaque") == 1.0
    assert config.resolver.allow_ambiguous_autopick is False


def test_configuration_is_not_hard_coded(config: DomainConfig) -> None:
    """Every audience level maps to a level that exists on the ladder."""
    for level in config.educational_levels.levels():
        assert config.educational_levels.lod_for(level) in config.lod_ladder


def test_unknown_opacity_preset_lists_the_alternatives(config: DomainConfig) -> None:
    """An unknown preset error names what is available."""
    with pytest.raises(ConfigError, match="Available presets"):
        config.opacity_presets.get("sparkly")


def test_paths_are_relative_to_the_repository(config: DomainConfig) -> None:
    """No absolute path is baked into configuration."""
    assert config.domain.ontology_dir.is_relative_to(repo_root())
    text = (repo_root() / "configs" / "heart.yaml").read_text(encoding="utf-8")
    assert "/Users/" not in text and "C:\\" not in text


def test_model_config_is_entirely_unspecified(model_config: ModelConfig) -> None:
    """No neural architecture decision has been made."""
    assert model_config.status == "placeholder"
    assert model_config.is_specified is False
    assert model_config.section("latent_3d")["latent_dim"] is None
    assert len(model_config.open_questions()) >= 4


def test_requiring_an_undecided_model_value_points_at_step_5(model_config: ModelConfig) -> None:
    """Depending on an undecided value fails with an explanation."""
    with pytest.raises(ConfigError, match="Step 5"):
        model_config.require("geometry_decoder", "strategy")


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    """A missing configuration file is an explicit error."""
    with pytest.raises(ConfigError, match="not found"):
        load_domain_config(tmp_path / "absent.yaml")


def test_invalid_config_section_is_reported(tmp_path: Path) -> None:
    """A configuration missing a required section names the section."""
    path = tmp_path / "partial.yaml"
    path.write_text("config_version: '0.1.0'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required configuration section"):
        load_domain_config(path)


def test_cardiac_cycle_fractions_must_sum_to_one() -> None:
    """Phase fractions are validated at load time."""
    with pytest.raises(ConfigError, match="sum to 1.0"):
        CardiacCycleConfig(
            phases=(
                CardiacCyclePhaseSpec("a", 0.5),
                CardiacCyclePhaseSpec("b", 0.2),
            )
        )


def test_lod_ladder_rejects_unknown_levels(config: DomainConfig) -> None:
    """An LOD outside the ladder is refused with the valid levels."""
    with pytest.raises(InvalidStateError, match="Valid levels"):
        config.lod_ladder.validate(7)


def test_lod_ladder_steps_are_clamped(config: DomainConfig) -> None:
    """Relative steps never leave the ladder."""
    ladder = config.lod_ladder
    assert ladder.step(ladder.max_level, 1) == ladder.max_level
    assert ladder.step(ladder.min_level, -1) == ladder.min_level
    assert ladder.step(1, 1) == 2


def test_ladder_must_be_ordered_and_unique() -> None:
    """A malformed ladder is rejected."""
    with pytest.raises(ConfigError, match="ascending order"):
        LodLadder((LodLevelSpec(2, "b", ""), LodLevelSpec(1, "a", "")))
    with pytest.raises(ConfigError, match="unique"):
        LodLadder((LodLevelSpec(1, "a", ""), LodLevelSpec(1, "b", "")))


def test_default_visibility_policy() -> None:
    """Visibility depends on renderability and the LOD window, nothing else."""
    assert default_visibility_at(renderable=True, min_lod=1, max_lod=None, lod=2) is True
    assert default_visibility_at(renderable=True, min_lod=3, max_lod=None, lod=2) is False
    assert default_visibility_at(renderable=True, min_lod=0, max_lod=0, lod=1) is False
    assert default_visibility_at(renderable=False, min_lod=0, max_lod=None, lod=4) is False


def test_educational_level_map_round_trip(config: DomainConfig) -> None:
    """Audience level and LOD map to each other consistently."""
    levels = config.educational_levels
    assert levels.lod_for(EducationalLevel.MEDICAL) == 4
    assert levels.level_for_lod(4) is EducationalLevel.MEDICAL
    assert levels.level_for_lod(99) is None


def test_unconfigured_level_is_reported() -> None:
    """Asking for a level that is not configured is an error."""
    mapping = EducationalLevelMap((EducationalLevelSpec(EducationalLevel.SCHOOL, 1, ""),))
    with pytest.raises(ConfigError, match="not configured"):
        mapping.lod_for(EducationalLevel.MEDICAL)


def test_unknown_enum_value_is_rejected() -> None:
    """Controlled vocabularies do not accept new values by accident."""
    from awr.schema import coerce_enum

    with pytest.raises(SchemaError, match="Allowed values"):
        coerce_enum(EducationalLevel, "postgraduate", field_name="educational_level")
