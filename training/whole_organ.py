"""Training on the whole-organ corpus, with level-specific supervision.

Two differences from the Step 6 loop, both deliberate:

* Batches carry **one relationship graph per scene**, measured from that organ.
* The level-of-detail objective supervises each token prefix against **that level's own
  target**, rather than against the full-detail decode. Step 6 trained every prefix
  against the same target, which is the most likely reason its later tokens went unused.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as functional_ops

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.sampling import WholeOrganSampling
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.losses import LossInputs, PrototypeLoss
from generation.neural.nn.model import LagnavPrototype
from generation.neural.nn.whole_organ import WholeOrganBatch, WholeOrganBatchBuilder
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE
from training.loop import _hard_negative_table, build_model
from training.manifest import RunManifest

__all__ = ["WholeOrganConfig", "WholeOrganLoader", "WholeOrganTrainer"]


@dataclass(frozen=True, slots=True)
class WholeOrganConfig:
    """Training settings. Identical across arms except for ``arm``."""

    arm: str = "A3"
    seed: int = 0
    steps: int = 900
    batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    gradient_clip: float = 1.0
    scene_points: int = 256
    entity_points: int = 24
    eval_every: int = 300
    log_every: int = 50
    eval_batches: int = 6
    device: str = "cpu"
    level_supervision_weight: float = 0.5
    loss_weights: Mapping[str, float] = field(default_factory=initial_loss_strategy)

    def sampling(self) -> WholeOrganSampling:
        """Point sampling derived from the training settings."""
        return WholeOrganSampling(
            scene_points=self.scene_points, entity_points=self.entity_points
        )

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
            "scene_points": self.scene_points,
            "entity_points": self.entity_points,
            "device": self.device,
            "level_supervision_weight": self.level_supervision_weight,
            "loss_weights": dict(self.loss_weights),
            "optimizer": "adamw",
            "schedule": "linear warmup then cosine decay",
            "corpus": "whole-organ",
        }


class WholeOrganLoader:
    """Yields batches whose scenes share a level of detail."""

    def __init__(
        self,
        scenes: Sequence[WholeOrganScene],
        builder: WholeOrganBatchBuilder,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if not scenes:
            raise ValueError("The loader needs at least one scene.")
        self.builder = builder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator().manual_seed(seed)
        self.buckets: dict[int, list[WholeOrganScene]] = {}
        for scene in scenes:
            self.buckets.setdefault(scene.active_lod, []).append(scene)

    def epoch(self, epoch_index: int = 0) -> Iterator[WholeOrganBatch]:
        """One pass over every bucket."""
        batches: list[list[WholeOrganScene]] = []
        for lod in sorted(self.buckets):
            items = list(self.buckets[lod])
            if self.shuffle:
                order = torch.randperm(len(items), generator=self.generator).tolist()
                items = [items[index] for index in order]
            step = self.batch_size
            stop = len(items) - step + 1 if self.drop_last else len(items)
            for start in range(0, max(stop, 0), step):
                group = items[start : start + step]
                if group:
                    batches.append(group)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[index] for index in order]
        for group in batches:
            yield self.builder.build(group, epoch=epoch_index)

    def infinite(self) -> Iterator[WholeOrganBatch]:
        """Cycle over epochs."""
        epoch_index = 0
        while True:
            yield from self.epoch(epoch_index)
            epoch_index += 1


class WholeOrganTrainer:
    """Trains one arm on the whole-organ corpus."""

    def __init__(
        self,
        config: WholeOrganConfig,
        ontology: AnatomyOntology,
        domain: DomainConfig,
        train_scenes: Sequence[WholeOrganScene],
        eval_scenes: Sequence[WholeOrganScene],
        *,
        dataset_info: Mapping[str, Any] | None = None,
    ) -> None:
        from generation.neural.nn.device import resolve_device, seed_everything

        self.config = config
        self.ontology = ontology
        self.device_choice = resolve_device(config.device)
        seed_everything(config.seed)
        self.builder = WholeOrganBatchBuilder(ontology, domain, sampling=config.sampling())
        self.train_loader = WholeOrganLoader(
            train_scenes, self.builder, batch_size=config.batch_size, seed=config.seed
        )
        self.eval_loader = WholeOrganLoader(
            eval_scenes,
            self.builder,
            batch_size=config.batch_size,
            seed=config.seed + 1,
            shuffle=False,
            drop_last=False,
        )
        probe_level = train_scenes[0].active_lod
        probe = self.builder.build(
            [s for s in train_scenes if s.active_lod == probe_level][: config.batch_size]
        )
        self.text_features = int(probe.batch.text_features.shape[1])
        self.model, self.parameter_groups = build_model(
            cast(Any, config.arm), self.builder, text_features=self.text_features
        )
        self.model.to(self.device_choice.device)
        self.loss = PrototypeLoss(
            config.loss_weights,
            hard_negatives=_hard_negative_table(ontology).to(self.device_choice.device),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.ids = list(ontology.ids())
        self.slot_of = {entity_id: index for index, entity_id in enumerate(self.ids)}
        self.manifest = RunManifest(
            run_id=f"wo-{config.arm}-seed{config.seed}",
            arm=config.arm,
            seed=config.seed,
            config=config.to_dict(),
            dataset=dict(dataset_info or {}),
            parameters=self.parameter_groups,
        )
        self.manifest.environment = {
            **self.manifest.environment,
            "device": str(self.device_choice),
        }

    # ------------------------------------------------------------------
    def _learning_rate(self, step: int) -> float:
        warmup = max(1, self.config.warmup_steps)
        if step < warmup:
            return self.config.learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, self.config.steps - warmup)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def _prototypes(self) -> torch.Tensor | None:
        table = getattr(self.model, "alignment_prototypes", None)
        return table.weight if table is not None else None

    def _level_supervision(self, whole: WholeOrganBatch, step: int) -> torch.Tensor:
        """Supervise a token prefix against its own level's target.

        The Step 7 fix for level of detail: each prefix is scored on what that level is
        supposed to show, so a longer prefix has something to add.
        """
        level = (step % 3) + 1
        prefix = LOD_TOKEN_SCHEDULE[min(level, len(LOD_TOKEN_SCHEDULE) - 1)]
        target = whole.lod_targets[level].to(self.device_choice.device)
        output = self.model(whole.batch, token_prefix=prefix)
        return functional_ops.binary_cross_entropy_with_logits(output.scene_logits, target)

    def train_step(self, whole: WholeOrganBatch, step: int) -> dict[str, float]:
        """One optimisation step."""
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self._learning_rate(step)
        batch = whole.batch.to(self.device_choice.device)
        output = self.model(batch)
        breakdown = self.loss(
            LossInputs(
                output=output,
                batch=batch,
                prefix_output=None,
                geometry_state_invariant=bool(
                    getattr(self.model, "geometry_is_state_invariant", False)
                ),
            ),
            self._prototypes(),
        )
        total = breakdown.total
        level_term = torch.zeros((), device=total.device)
        if self.config.level_supervision_weight > 0:
            level_term = self._level_supervision(whole, step)
            total = total + self.config.level_supervision_weight * level_term

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        record: dict[str, float] = breakdown.as_floats()
        record["total"] = float(total.detach())
        record["level_supervision"] = float(level_term.detach())
        record["grad_norm"] = float(grad_norm)
        record["learning_rate"] = self._learning_rate(step)
        record["step"] = float(step)
        return record

    @torch.no_grad()
    def evaluate(self, *, batches: int | None = None, full: bool = False) -> dict[str, float]:
        """Evaluate on held-out scenes."""
        from experiments.step7.metrics import evaluate_whole_organ

        self.model.eval()
        limit = batches if batches is not None else self.config.eval_batches
        totals: dict[str, list[float]] = {}
        count = 0
        for whole in self.eval_loader.epoch(0):
            if count >= limit:
                break
            count += 1
            values = evaluate_whole_organ(
                self.model,
                whole,
                self.slot_of,
                self.ids,
                device=self.device_choice.device,
                full=full,
                builder=self.builder,
            )
            for key, value in values.items():
                totals.setdefault(key, []).append(value)
        summary = {key: sum(values) / len(values) for key, values in totals.items()}
        summary["eval_batches"] = float(count)
        return dict(summary)

    def fit(self, *, checkpoint_dir: str | Path | None = None) -> RunManifest:
        """Train, then evaluate fully."""
        started = time.time()
        stream = self.train_loader.infinite()
        for step in range(self.config.steps):
            record = self.train_step(next(stream), step)
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
        if checkpoint_dir is not None:
            target = Path(checkpoint_dir)
            target.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "config": self.config.to_dict(),
                    "parameters": self.parameter_groups,
                },
                target / f"{self.manifest.run_id}.pt",
            )
            self.manifest.save(target / f"{self.manifest.run_id}.manifest.json")
        return self.manifest


def with_arm(config: WholeOrganConfig, arm: str, seed: int) -> WholeOrganConfig:
    """Copy a configuration for another arm and seed."""
    return replace(config, arm=arm, seed=seed)
