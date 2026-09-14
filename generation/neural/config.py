"""Typed loading and validation of neural experiment configuration.

STATUS: implemented. The configuration files describe experiments that have not
been run; this module makes sure they at least describe something coherent.

Three checks earn their keep:

* every loss weight names an objective declared in :mod:`generation.neural.losses`,
* no objective held out for evaluation is given a training weight,
* every curriculum stage exists and appears in dependency order.

Architecture widths are not duplicated in YAML. A configuration names a scale, and
the widths come from :mod:`generation.neural.scales`, so the two cannot disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from awr.config import load_yaml
from awr.errors import ConfigError, ContractError
from awr.paths import config_dir
from generation.neural.curriculum import CURRICULUM, stage
from generation.neural.losses import LOSSES, validate_strategy
from generation.neural.scales import ScaleSpec, scale

__all__ = [
    "OptimizationConfig",
    "DataConfig",
    "NeuralExperimentConfig",
    "load_neural_config",
    "available_neural_configs",
]


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimiser settings for one run."""

    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 20_000
    batch_scenes: int = 8
    query_points_per_scene: int = 4096
    precision: str = "bf16"
    gradient_clip: float = 1.0
    seed: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> OptimizationConfig:
        """Build from the ``optimization`` section."""
        return cls(
            optimizer=str(payload.get("optimizer", "adamw")),
            learning_rate=float(payload.get("learning_rate", 3.0e-4)),
            weight_decay=float(payload.get("weight_decay", 0.01)),
            warmup_steps=int(payload.get("warmup_steps", 500)),
            max_steps=int(payload.get("max_steps", 20_000)),
            batch_scenes=int(payload.get("batch_scenes", 8)),
            query_points_per_scene=int(payload.get("query_points_per_scene", 4096)),
            precision=str(payload.get("precision", "bf16")),
            gradient_clip=float(payload.get("gradient_clip", 1.0)),
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Dataset settings for one run."""

    tier: str = "synthetic"
    scenes: int = 0
    split: Mapping[str, float] = field(default_factory=dict)
    hold_out: str = ""
    licensed_assets: bool = False

    def __post_init__(self) -> None:
        if self.split:
            total = sum(self.split.values())
            if abs(total - 1.0) > 1e-6:
                raise ConfigError(f"Data split fractions must sum to 1.0, got {total:.6f}.")
        if self.licensed_assets and self.tier == "synthetic":
            raise ConfigError(
                "A synthetic tier must not claim licensed assets; the two are different data "
                "provenance stories and conflating them hides a licensing decision."
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DataConfig:
        """Build from the ``data`` section."""
        return cls(
            tier=str(payload.get("tier", "synthetic")),
            scenes=int(payload.get("scenes", 0)),
            split={str(k): float(v) for k, v in (payload.get("split") or {}).items()},
            hold_out=str(payload.get("hold_out", "")),
            licensed_assets=bool(payload.get("licensed_assets", False)),
        )


@dataclass(frozen=True, slots=True)
class NeuralExperimentConfig:
    """A validated neural experiment configuration."""

    config_version: str
    status: str
    scale_name: str
    experiment_id: str
    arm: str
    stages: tuple[str, ...]
    skipped_stages: Mapping[str, str]
    loss_weights: Mapping[str, float]
    optimization: OptimizationConfig
    data: DataConfig
    metrics: tuple[str, ...] = ()
    architecture_overrides: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def is_proposed(self) -> bool:
        """Whether this configuration describes an unrun proposal."""
        return self.status == "proposed"

    @property
    def scale(self) -> ScaleSpec:
        """The architecture scale this configuration references."""
        return scale(self.scale_name)

    def validate(self) -> None:
        """Check losses, stages and the scale reference."""
        scale(self.scale_name)
        validate_strategy(self.loss_weights)
        known = {s.key for s in CURRICULUM}
        unknown = sorted(set(self.stages) - known)
        if unknown:
            raise ConfigError(f"Configuration names unknown curriculum stages {unknown}.")
        order = {s.key: index for index, s in enumerate(CURRICULUM)}
        positions = [order[key] for key in self.stages]
        if positions != sorted(positions):
            raise ConfigError(
                f"Curriculum stages {list(self.stages)} are out of dependency order."
            )
        for key in self.stages:
            for dependency in stage(key).blocked_by:
                if dependency in self.stages:
                    continue
                if dependency in self.skipped_stages:
                    continue
                raise ConfigError(
                    f"Stage {key} depends on {dependency}, which this run neither includes nor "
                    "declares as deliberately skipped. Skipping a stage is allowed, but it has "
                    "to be stated with a reason under curriculum.skips."
                )
        unknown_skips = sorted(set(self.skipped_stages) - known)
        if unknown_skips:
            raise ConfigError(f"Configuration skips unknown curriculum stages {unknown_skips}.")

    def summary(self) -> dict[str, Any]:
        """Compact description for reports and tests."""
        return {
            "experiment": self.experiment_id,
            "arm": self.arm,
            "scale": self.scale_name,
            "status": self.status,
            "stages": list(self.stages),
            "skipped": sorted(self.skipped_stages),
            "losses": dict(self.loss_weights),
            "scenes": self.data.scenes,
            "tier": self.data.tier,
        }

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], source_path: Path | None = None
    ) -> NeuralExperimentConfig:
        """Build a configuration from a parsed YAML mapping."""
        experiment = payload.get("experiment", {})
        curriculum = payload.get("curriculum", {})
        config = cls(
            config_version=str(payload.get("config_version", "0")),
            status=str(payload.get("status", "proposed")),
            scale_name=str(payload.get("scale", "prototype")),
            experiment_id=str(experiment.get("id", "unnamed")),
            arm=str(experiment.get("arm", "lagnav_structured")),
            stages=tuple(str(key) for key in curriculum.get("stages", ())),
            skipped_stages={
                str(key): str(reason) for key, reason in (curriculum.get("skips") or {}).items()
            },
            loss_weights={str(k): float(v) for k, v in (payload.get("losses") or {}).items()},
            optimization=OptimizationConfig.from_mapping(payload.get("optimization", {})),
            data=DataConfig.from_mapping(payload.get("data", {})),
            metrics=tuple(str(m) for m in (payload.get("evaluation", {}) or {}).get("metrics", ())),
            architecture_overrides=dict(payload.get("architecture", {})),
            source_path=source_path,
        )
        config.validate()
        return config


def load_neural_config(name: str = "prototype") -> NeuralExperimentConfig:
    """Load and validate ``configs/neural/<name>.yaml``."""
    path = config_dir() / "neural" / f"{name}.yaml"
    try:
        return NeuralExperimentConfig.from_mapping(load_yaml(path), source_path=path)
    except ContractError as exc:  # surfaced as a configuration problem for the caller
        raise ConfigError(f"{path.name}: {exc}") from exc


def available_neural_configs() -> tuple[str, ...]:
    """Names of the neural configurations present in the repository."""
    folder = config_dir() / "neural"
    return tuple(sorted(path.stem for path in folder.glob("*.yaml")))


def declared_loss_names() -> tuple[str, ...]:
    """Every declared objective name, for configuration authoring."""
    return tuple(spec.name for spec in LOSSES)


def sequence_of_stages(config: NeuralExperimentConfig) -> Sequence[str]:
    """Stages this configuration will run, in order."""
    return config.stages
