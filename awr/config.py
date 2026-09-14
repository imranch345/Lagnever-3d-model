"""Configuration loading.

Configuration is data, not code: numbers such as "transparent means 0.3" or
"medical level means LOD 4" live in ``configs/*.yaml`` so that they can be tuned
and versioned without touching the engine.

PyYAML is imported lazily and is the project's only runtime dependency; the AWR
core itself runs on the standard library alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awr.errors import ConfigError
from awr.lod import EducationalLevelMap, LodLadder
from awr.paths import config_dir, resolve_repo_path
from awr.schema import EducationalLevel, LodLevel, coerce_enum

__all__ = [
    "DomainSpec",
    "SceneDefaults",
    "OpacityPresets",
    "ResolverConfig",
    "GeometryConfig",
    "CardiacCyclePhaseSpec",
    "CardiacCycleConfig",
    "DomainConfig",
    "ModelConfig",
    "load_yaml",
    "load_domain_config",
    "load_model_config",
]


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` with actionable errors."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError(
            "PyYAML is required to read Lagnav configuration files. "
            "Install the project with 'pip install -e .'."
        ) from exc
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Which anatomical domain a configuration describes."""

    id: str
    display_name: str
    ontology_dir: Path
    ontology_version: str


@dataclass(frozen=True, slots=True)
class SceneDefaults:
    """Initial scene settings applied by a generate command."""

    scene_name: str
    active_lod: LodLevel
    educational_level: EducationalLevel
    default_opacity: float


@dataclass(frozen=True, slots=True)
class OpacityPresets:
    """Named opacity values used by the command layer."""

    values: Mapping[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        """Return a preset value or raise with the available names."""
        try:
            return float(self.values[name])
        except KeyError:
            raise ConfigError(
                f"Opacity preset {name!r} is not configured. "
                f"Available presets: {', '.join(sorted(self.values))}."
            ) from None

    def names(self) -> tuple[str, ...]:
        """All configured preset names, sorted."""
        return tuple(sorted(self.values))


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    """Entity-resolver tuning."""

    fuzzy_threshold: float = 0.82
    max_suggestions: int = 5
    allow_ambiguous_autopick: bool = False


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """How symbolic geometry component ids are formed."""

    component_id_prefix: str = "geometry_part_"
    component_id_digits: int = 4


@dataclass(frozen=True, slots=True)
class CardiacCyclePhaseSpec:
    """One phase of the provisional cardiac cycle."""

    name: str
    fraction: float
    description: str = ""


@dataclass(frozen=True, slots=True)
class CardiacCycleConfig:
    """Phase structure of one cardiac cycle, used by the animation layer."""

    clip_id: str = "blood_flow"
    cycle_duration_seconds: float = 0.85
    phases: tuple[CardiacCyclePhaseSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.phases:
            return
        total = sum(phase.fraction for phase in self.phases)
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(
                f"Cardiac cycle phase fractions must sum to 1.0, got {total:.6f}."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CardiacCycleConfig:
        """Build from the ``cardiac_cycle`` config section."""
        return cls(
            clip_id=str(payload.get("clip_id", "blood_flow")),
            cycle_duration_seconds=float(payload.get("cycle_duration_seconds", 0.85)),
            phases=tuple(
                CardiacCyclePhaseSpec(
                    name=str(phase["name"]),
                    fraction=float(phase["fraction"]),
                    description=str(phase.get("description", "")),
                )
                for phase in payload.get("phases", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Fully typed view of ``configs/<domain>.yaml``."""

    config_version: str
    domain: DomainSpec
    scene_defaults: SceneDefaults
    lod_ladder: LodLadder
    educational_levels: EducationalLevelMap
    opacity_presets: OpacityPresets
    resolver: ResolverConfig
    geometry: GeometryConfig
    cardiac_cycle: CardiacCycleConfig = field(default_factory=CardiacCycleConfig)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        self.lod_ladder.validate(self.scene_defaults.active_lod)
        for level in self.educational_levels.levels():
            self.lod_ladder.validate(self.educational_levels.lod_for(level))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], source_path: Path | None = None
    ) -> DomainConfig:
        """Build a typed configuration from a parsed YAML mapping."""
        try:
            domain_raw = payload["domain"]
            defaults_raw = payload["scene_defaults"]
            lod_raw = payload["lod_levels"]
            edu_raw = payload["educational_levels"]
        except KeyError as exc:
            raise ConfigError(f"Missing required configuration section: {exc.args[0]!r}.") from exc

        domain = DomainSpec(
            id=str(domain_raw["id"]),
            display_name=str(domain_raw.get("display_name", domain_raw["id"])),
            ontology_dir=resolve_repo_path(str(domain_raw["ontology_dir"])),
            ontology_version=str(domain_raw["ontology_version"]),
        )
        defaults = SceneDefaults(
            scene_name=str(defaults_raw.get("scene_name", domain.display_name)),
            active_lod=int(defaults_raw["active_lod"]),
            educational_level=coerce_enum(
                EducationalLevel,
                str(defaults_raw["educational_level"]),
                field_name="scene_defaults.educational_level",
            ),
            default_opacity=float(defaults_raw.get("default_opacity", 1.0)),
        )
        resolver_raw = payload.get("resolver", {})
        geometry_raw = payload.get("geometry", {})
        return cls(
            config_version=str(payload.get("config_version", "0")),
            domain=domain,
            scene_defaults=defaults,
            lod_ladder=LodLadder.from_mapping(lod_raw),
            educational_levels=EducationalLevelMap.from_mapping(edu_raw),
            opacity_presets=OpacityPresets(
                {str(k): float(v) for k, v in payload.get("opacity_presets", {}).items()}
            ),
            resolver=ResolverConfig(
                fuzzy_threshold=float(resolver_raw.get("fuzzy_threshold", 0.82)),
                max_suggestions=int(resolver_raw.get("max_suggestions", 5)),
                allow_ambiguous_autopick=bool(resolver_raw.get("allow_ambiguous_autopick", False)),
            ),
            geometry=GeometryConfig(
                component_id_prefix=str(geometry_raw.get("component_id_prefix", "geometry_part_")),
                component_id_digits=int(geometry_raw.get("component_id_digits", 4)),
            ),
            cardiac_cycle=CardiacCycleConfig.from_mapping(payload.get("cardiac_cycle", {})),
            source_path=source_path,
        )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Typed view of ``configs/model.yaml``.

    Every architectural field is deliberately unspecified. :meth:`require` exists
    so that code which needs a decision fails with a message pointing at Step 5
    instead of silently inventing a default.
    """

    config_version: str
    status: str
    raw: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def is_specified(self) -> bool:
        """Whether a neural architecture has been decided yet."""
        return self.status != "placeholder"

    def section(self, name: str) -> Mapping[str, Any]:
        """Return one configuration section (all values may be ``None``)."""
        value = self.raw.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"Model configuration has no section {name!r}.")
        return value

    def require(self, section: str, key: str) -> Any:
        """Return a decided value, or raise explaining that Step 5 owns it."""
        value = self.section(section).get(key)
        if value is None:
            raise ConfigError(
                f"model.yaml {section}.{key} is intentionally unspecified. "
                "The Lagnav neural architecture is defined in Step 5; nothing may "
                "depend on this value before then."
            )
        return value

    def open_questions(self) -> tuple[str, ...]:
        """The recorded open research questions."""
        return tuple(self.raw.get("open_questions", ()))

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], source_path: Path | None = None
    ) -> ModelConfig:
        """Build from a parsed YAML mapping."""
        return cls(
            config_version=str(payload.get("config_version", "0")),
            status=str(payload.get("status", "placeholder")),
            raw=dict(payload),
            source_path=source_path,
        )


def load_domain_config(path: str | Path | None = None, *, domain: str = "heart") -> DomainConfig:
    """Load ``configs/<domain>.yaml`` (or an explicit path) into a typed object."""
    config_path = Path(path) if path is not None else config_dir() / f"{domain}.yaml"
    return DomainConfig.from_mapping(load_yaml(config_path), source_path=config_path)


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    """Load ``configs/model.yaml`` (or an explicit path) into a typed object."""
    config_path = Path(path) if path is not None else config_dir() / "model.yaml"
    return ModelConfig.from_mapping(load_yaml(config_path), source_path=config_path)
