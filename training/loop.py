"""The training loop for the Step 6 prototype.

Small on purpose: one file, readable top to bottom, no framework of its own. It does
what the brief asks and nothing more: configuration, batching, forward, loss,
backward, optimiser, schedule, checkpointing, validation, metrics, seeds and logging.

Batches are bucketed by level of detail. Every scene in a batch therefore shares one
visible entity set, which keeps the padded entity axis dense enough to be worth
computing and makes the level-of-detail prefix well defined for the batch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import torch
from torch import nn

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.synthetic.heart_corpus import REQUIRED_SPATIAL_RELATIONS, SceneSpec
from datasets.synthetic.sampling import SamplingConfig
from generation.neural.losses import initial_loss_strategy
from generation.neural.multimodal import OntologyNegativeSampler
from generation.neural.nn import metrics as metric_module
from generation.neural.nn.baseline import (
    AppearanceBaseline,
    BaselineConfig,
    matched_baseline_config,
)
from generation.neural.nn.device import DeviceChoice, resolve_device, seed_everything
from generation.neural.nn.geometry import GeometryConfig
from generation.neural.nn.losses import LossInputs, PrototypeLoss
from generation.neural.nn.model import LagnavPrototype, ModelOutput, PrototypeConfig
from generation.neural.nn.tensors import BatchBuilder, PrototypeBatch
from training.manifest import RunManifest


class _ReportsParameters(Protocol):
    """A model that can describe where its parameters went."""

    def parameter_groups(self) -> dict[str, int]:
        """Parameter count per component."""

__all__ = ["TrainingConfig", "LodBucketLoader", "Trainer", "build_model"]

ArmName = Literal["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything a run needs. Identical across arms except for ``arm``."""

    arm: ArmName = "A3"
    seed: int = 0
    steps: int = 600
    batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_clip: float = 1.0
    scene_points: int = 256
    entity_points: int = 32
    eval_every: int = 200
    log_every: int = 50
    eval_batches: int = 6
    device: str = "cpu"
    loss_weights: Mapping[str, float] = field(default_factory=initial_loss_strategy)
    train_limit: int | None = None
    eval_limit: int | None = None
    coarse_prefix: int = 8
    lod_prefixes: tuple[int, ...] = (4, 8, 16, 24, 32)

    def sampling(self) -> SamplingConfig:
        """Point-sampling configuration derived from the training settings."""
        return SamplingConfig(scene_points=self.scene_points, entity_points=self.entity_points)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the manifest."""
        return {
            "arm": self.arm,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "scene_points": self.scene_points,
            "entity_points": self.entity_points,
            "device": self.device,
            "loss_weights": dict(self.loss_weights),
            "train_limit": self.train_limit,
            "eval_limit": self.eval_limit,
            "coarse_prefix": self.coarse_prefix,
            "lod_prefixes": list(self.lod_prefixes),
            "optimizer": "adamw",
            "schedule": "linear warmup then cosine decay",
        }


class LodBucketLoader:
    """Yields batches whose scenes share a level of detail."""

    def __init__(
        self,
        specs: Sequence[SceneSpec],
        builder: BatchBuilder,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if not specs:
            raise ValueError("The loader needs at least one scene.")
        self.builder = builder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator().manual_seed(seed)
        self.buckets: dict[int, list[SceneSpec]] = {}
        for spec in specs:
            self.buckets.setdefault(spec.active_lod, []).append(spec)

    def __len__(self) -> int:
        return sum(
            max(1, len(items) // self.batch_size) for items in self.buckets.values()
        )

    def epoch(self, epoch_index: int = 0) -> Iterator[PrototypeBatch]:
        """One pass over every bucket."""
        order = sorted(self.buckets)
        batches: list[list[SceneSpec]] = []
        for lod in order:
            items = list(self.buckets[lod])
            if self.shuffle:
                permutation = torch.randperm(len(items), generator=self.generator).tolist()
                items = [items[index] for index in permutation]
            step = self.batch_size
            stop = len(items) - step + 1 if self.drop_last else len(items)
            for start in range(0, max(stop, 0), step):
                group = items[start : start + step]
                if group:
                    batches.append(group)
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[index] for index in permutation]
        for group in batches:
            yield self.builder.build(group, epoch=epoch_index)

    def infinite(self) -> Iterator[PrototypeBatch]:
        """Cycle over epochs indefinitely."""
        epoch_index = 0
        while True:
            yield from self.epoch(epoch_index)
            epoch_index += 1


def build_model(
    arm: ArmName,
    builder: BatchBuilder,
    *,
    text_features: int,
    geometry: GeometryConfig | None = None,
) -> tuple[nn.Module, dict[str, int]]:
    """Build one arm and return it with its parameter breakdown.

    ``A0`` is the appearance-driven baseline, sized to match the structured arm's
    parameter count before training starts.
    """
    sizes = builder.vocabulary_sizes()
    structured = PrototypeConfig(
        entity_vocabulary=sizes["entity"],
        anatomy_type_vocabulary=sizes["anatomy_type"],
        semantic_role_vocabulary=sizes["semantic_role"],
        laterality_vocabulary=sizes["laterality"],
        relation_vocabulary=sizes["relation"],
        text_features=text_features,
        geometry=geometry or GeometryConfig(),
    )
    if arm == "A0":
        reference = LagnavPrototype(
            structured.with_ablation("A3"), builder.inverse_relation_table()
        )
        target = reference.parameter_groups()["total"]
        config = matched_baseline_config(
            target,
            base=BaselineConfig(
                text_features=text_features,
                entity_vocabulary=sizes["entity"],
                geometry=geometry or GeometryConfig(),
            ),
        )
        model: nn.Module = AppearanceBaseline(config)
        groups = cast(_ReportsParameters, model).parameter_groups()
        groups["matched_target"] = target
        return model, groups
    model = LagnavPrototype(structured.with_ablation(arm), builder.inverse_relation_table())
    return model, model.parameter_groups()


def _hard_negative_table(ontology: AnatomyOntology, width: int = 6) -> torch.Tensor:
    sampler = OntologyNegativeSampler(ontology)
    ids = list(ontology.ids())
    index = {entity_id: position for position, entity_id in enumerate(ids)}
    table = torch.full((len(ids), width), -1, dtype=torch.int64)
    for entity_id in ids:
        for column, negative in enumerate(sampler.hard_negatives(entity_id, width)):
            table[index[entity_id], column] = index[negative]
    return table


class Trainer:
    """Trains one arm on the tier-0 corpus and evaluates it."""

    def __init__(
        self,
        config: TrainingConfig,
        ontology: AnatomyOntology,
        domain: DomainConfig,
        train_specs: Sequence[SceneSpec],
        eval_specs: Sequence[SceneSpec],
        *,
        dataset_info: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.ontology = ontology
        self.device_choice: DeviceChoice = resolve_device(config.device)
        seed_everything(config.seed)
        self.builder = BatchBuilder(ontology, domain, sampling=config.sampling())
        self.train_loader = LodBucketLoader(
            train_specs, self.builder, batch_size=config.batch_size, seed=config.seed
        )
        self.eval_loader = LodBucketLoader(
            eval_specs,
            self.builder,
            batch_size=config.batch_size,
            seed=config.seed + 1,
            shuffle=False,
            drop_last=False,
        )
        first_lod = train_specs[0].active_lod
        probe_specs = [spec for spec in train_specs if spec.active_lod == first_lod][
            : config.batch_size
        ]
        probe = self.builder.build(probe_specs)
        self.text_features = int(probe.text_features.shape[1])
        self.model, self.parameter_groups = build_model(
            config.arm, self.builder, text_features=self.text_features
        )
        self.model.to(self.device_choice.device)
        self.loss = PrototypeLoss(
            config.loss_weights,
            hard_negatives=_hard_negative_table(ontology).to(self.device_choice.device),
            alignment_mode="pairwise" if config.arm == "A6" else "prototype",
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.ids = list(ontology.ids())
        self.slot_of = {entity_id: position for position, entity_id in enumerate(self.ids)}
        self.manifest = RunManifest(
            run_id=f"{config.arm}-seed{config.seed}",
            arm=config.arm,
            seed=config.seed,
            config=config.to_dict(),
            dataset=dict(dataset_info or {}),
            parameters=self.parameter_groups,
        )
        self.manifest.environment = {**self.manifest.environment, "device": str(self.device_choice)}

    # ------------------------------------------------------------------
    @property
    def geometry_state_invariant(self) -> bool:
        """Whether this arm's geometry can be moved by a presentation edit."""
        return bool(getattr(self.model, "geometry_is_state_invariant", False))

    def _learning_rate(self, step: int) -> float:
        warmup = max(1, self.config.warmup_steps)
        if step < warmup:
            return self.config.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, self.config.steps - warmup)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _prototypes(self) -> torch.Tensor | None:
        table = getattr(self.model, "alignment_prototypes", None)
        return table.weight if table is not None else None

    def _edited_batch(self, batch: PrototypeBatch) -> tuple[PrototypeBatch, torch.Tensor] | None:
        """Build a presentation-edit pair for the edit-consistency objective."""
        present = batch.entity_present[0].nonzero().flatten()
        if present.numel() < 2:
            return None
        hidden = int(present[-1].item())
        import copy

        edited = copy.copy(batch)
        edited.text_features = batch.text_features.clone()
        edited.text_features[:, hidden] = 0.0
        untouched = (batch.part_owner >= 0) & (batch.part_owner != hidden)
        return edited, untouched

    def train_step(self, batch: PrototypeBatch, step: int) -> dict[str, float]:
        """One optimisation step."""
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self._learning_rate(step)
        batch = batch.to(self.device_choice.device)
        output = self.model(batch)
        prefix_output = (
            self.model(batch, token_prefix=self.config.coarse_prefix)
            if self.config.loss_weights.get("lod_consistency", 0.0) > 0
            else None
        )
        edited_output: ModelOutput | None = None
        untouched: torch.Tensor | None = None
        if (
            self.config.loss_weights.get("edit_consistency", 0.0) > 0
            and not self.geometry_state_invariant
        ):
            pair = self._edited_batch(batch)
            if pair is not None:
                edited_batch, untouched = pair
                edited_output = self.model(edited_batch)
        breakdown = self.loss(
            LossInputs(
                output=output,
                batch=batch,
                prefix_output=prefix_output,
                edited_output=edited_output,
                edited_untouched_mask=untouched,
                geometry_state_invariant=self.geometry_state_invariant,
            ),
            self._prototypes(),
        )
        self.optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        record: dict[str, float] = breakdown.as_floats()
        record["grad_norm"] = float(grad_norm)
        record["learning_rate"] = self._learning_rate(step)
        record["step"] = float(step)
        return record

    @torch.no_grad()
    def evaluate(self, *, batches: int | None = None, full: bool = False) -> dict[str, float]:
        """Evaluate on the held-out split and aggregate the metrics."""
        self.model.eval()
        limit = batches if batches is not None else self.config.eval_batches
        totals: dict[str, list[float]] = {}
        per_entity: dict[str, list[float]] = {}
        count = 0
        for batch in self.eval_loader.epoch(0):
            if count >= limit:
                break
            count += 1
            batch = batch.to(self.device_choice.device)
            visible = [int(slot) for slot in batch.structure.visible_slots]
            slot_names = {slot: self.ids[slot] for slot in visible}
            output = self.model(batch)
            results = [
                metric_module.part_control(output, batch, slot_names),
                metric_module.scene_geometry(output, batch),
                metric_module.relationship_accuracy(
                    output, batch, REQUIRED_SPATIAL_RELATIONS, self.slot_of
                ),
            ]
            if full:
                results.append(
                    metric_module.lod_consistency(
                        self.model,
                        batch,
                        prefixes=self.config.lod_prefixes,
                        slot_names=slot_names,
                    )
                )
                hidden = visible[-1] if visible else 0
                results.append(metric_module.edit_drift(self.model, batch, hidden_slot=hidden))
            for result in results:
                for key, value in result.values.items():
                    totals.setdefault(key, []).append(value)
                for key, value in result.per_entity.items():
                    per_entity.setdefault(key, []).append(value)
        summary: dict[str, float] = {
            key: sum(values) / len(values) for key, values in totals.items()
        }
        if full:
            summary.update(
                {f"detail/{key}": sum(values) / len(values) for key, values in per_entity.items()}
            )
        summary["eval_batches"] = float(count)
        return dict(summary)

    def fit(self, *, checkpoint_dir: str | Path | None = None) -> RunManifest:
        """Train for the configured number of steps, then evaluate fully."""
        started = time.time()
        stream = self.train_loader.infinite()
        for step in range(self.config.steps):
            batch = next(stream)
            record = self.train_step(batch, step)
            if step % self.config.log_every == 0 or step == self.config.steps - 1:
                self.manifest.train_history.append(record)
            if self.config.eval_every and step > 0 and step % self.config.eval_every == 0:
                validation = self.evaluate()
                validation["step"] = float(step)
                self.manifest.validation_history.append(validation)
        self.manifest.steps_completed = self.config.steps
        final = self.evaluate(batches=self.config.eval_batches * 2, full=True)
        final["step"] = float(self.config.steps)
        self.manifest.validation_history.append(final)
        self.manifest.results = dict(final)
        self.manifest.runtime_seconds = round(time.time() - started, 2)
        self.manifest.finished_at = None
        if checkpoint_dir is not None:
            self.save_checkpoint(Path(checkpoint_dir) / f"{self.manifest.run_id}.pt")
            self.manifest.save(Path(checkpoint_dir) / f"{self.manifest.run_id}.manifest.json")
        return self.manifest

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> Path:
        """Write model and optimiser state."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "parameters": self.parameter_groups,
                "steps_completed": self.manifest.steps_completed,
            },
            target,
        )
        return target

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model and optimiser state."""
        payload = torch.load(Path(path), map_location=self.device_choice.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.manifest.steps_completed = int(payload.get("steps_completed", 0))


def with_arm(config: TrainingConfig, arm: ArmName, seed: int) -> TrainingConfig:
    """Copy a configuration for another arm and seed, changing nothing else."""
    return replace(config, arm=arm, seed=seed)
