"""Step 8 training: predicted placement and genuinely nested level of detail.

Two differences from the Step 7 loop, both aimed at the defects Step 7 recorded.

**Placement is learned, not supplied.** Step 7 trained and evaluated with each entity's
true canonical frame handed to the decoder, so the frame head was never exercised and
asking the model to place structures itself halved its ownership accuracy. Here the frame
head is supervised directly and the decoder is weaned off the true frames on a schedule,
so that by the end of training the model is decoding from its own placement. The final
evaluation never sees a true frame.

**The level-of-detail prefixes are asked to nest.** Step 7 supervised each prefix against
its own level's target, which was necessary and not sufficient: the finer prefixes were
used and made things worse. Two further terms ask the fine prefix not to withdraw
occupancy the coarse prefix asserted, and not to break what the coarse prefix already had
right. See :mod:`generation.neural.nn.nested_lod`.

The teacher-forcing ratio is reported at every logged step and in the run manifest, so a
reader can see what the model was actually decoding from at any point in training.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch.nn import functional as functional_ops

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.sampling import WholeOrganSampling
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.losses import LossInputs, PrototypeLoss
from generation.neural.nn.nested_lod import (
    NestedLodWeights,
    containment_loss,
    preservation_loss,
)
from generation.neural.nn.whole_organ import WholeOrganBatch, WholeOrganBatchBuilder
from generation.neural.three_d_latent import LOD_TOKEN_SCHEDULE
from training.loop import _hard_negative_table, build_model
from training.manifest import RunManifest
from training.whole_organ import WholeOrganLoader

__all__ = ["Step8Config", "Step8Trainer", "teacher_forcing_at"]

#: Levels supervised, and the token prefix each reads.
LEVELS: tuple[int, ...] = (1, 2, 3)


def teacher_forcing_at(step: int, total: int, config: Step8Config) -> float:
    """Teacher-forcing ratio at a training step.

    Three stages, declared before the runs:

    ``P0``  frames supplied while the frame head learns to predict them at all.
    ``P1``  the ratio falls linearly, so the geometry decoder sees its own placement
            errors gradually rather than all at once.
    ``P2``  frames are never supplied; the model decodes entirely from its own placement.

    The schedule is a fraction of the total budget rather than a step count, so the same
    configuration means the same thing at any budget.
    """
    if total <= 0:
        return 0.0
    progress = step / total
    if progress < config.stage_p0:
        return 1.0
    if progress >= config.stage_p2:
        return 0.0
    span = max(config.stage_p2 - config.stage_p0, 1e-9)
    return float(1.0 - (progress - config.stage_p0) / span)


@dataclass(frozen=True, slots=True)
class Step8Config:
    """Training settings. Identical across arms except for ``arm``."""

    arm: str = "A3Lite"
    seed: int = 0
    steps: int = 1_200
    batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 60
    gradient_clip: float = 1.0
    scene_points: int = 256
    entity_points: int = 24
    eval_every: int = 400
    log_every: int = 50
    device: str = "cpu"

    #: Curriculum boundaries, as fractions of the total budget.
    stage_p0: float = 0.15
    stage_p2: float = 0.60

    frame_weight: float = 2.0
    """Weight on direct frame supervision.

    Higher than the other terms because placement is the thing Step 8 is testing and it
    starts from nothing. Reported so the choice is visible rather than buried.
    """

    lod_weight: float = 0.5
    nesting: NestedLodWeights = field(default_factory=NestedLodWeights)
    loss_weights: Mapping[str, float] = field(default_factory=initial_loss_strategy)

    def sampling(self) -> WholeOrganSampling:
        """Point-sampling settings implied by this configuration."""
        return WholeOrganSampling(
            scene_points=self.scene_points, entity_points=self.entity_points
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the run manifest."""
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
            "curriculum": {
                "stage_p0_end": self.stage_p0,
                "stage_p2_start": self.stage_p2,
                "final_teacher_forcing": teacher_forcing_at(self.steps - 1, self.steps, self),
                "inference_uses_predicted_frames": True,
            },
            "frame_weight": self.frame_weight,
            "lod_weight": self.lod_weight,
            "nesting": {
                "reconstruction": self.nesting.reconstruction,
                "containment": self.nesting.containment,
                "preservation": self.nesting.preservation,
            },
            "loss_weights": dict(self.loss_weights),
            "optimizer": "adamw",
            "schedule": "linear warmup then cosine decay",
            "corpus": "step8-continuous",
        }


class Step8Trainer:
    """Trains one arm with predicted placement and nested level-of-detail objectives."""

    def __init__(
        self,
        config: Step8Config,
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
        self.device = self.device_choice.device
        seed_everything(config.seed)
        self.builder = WholeOrganBatchBuilder(ontology, domain, sampling=config.sampling())
        self.train_loader = WholeOrganLoader(
            train_scenes, self.builder, batch_size=config.batch_size, seed=config.seed
        )
        self.eval_scenes = list(eval_scenes)
        probe_level = train_scenes[0].active_lod
        probe = self.builder.build(
            [s for s in train_scenes if s.active_lod == probe_level][: config.batch_size]
        )
        self.text_features = int(probe.batch.text_features.shape[1])
        self.model, self.parameter_groups = build_model(
            cast(Any, config.arm), cast(Any, self.builder), text_features=self.text_features
        )
        self.model.to(self.device)
        self.loss = PrototypeLoss(
            config.loss_weights,
            hard_negatives=_hard_negative_table(ontology).to(self.device),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.ids = list(ontology.ids())
        self.slot_of = {entity_id: index for index, entity_id in enumerate(self.ids)}
        self.manifest = RunManifest(
            run_id=f"s8-{config.arm}-seed{config.seed}",
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
        warmup = self.config.warmup_steps
        if step < warmup:
            return self.config.learning_rate * (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(self.config.steps - warmup, 1)
        return self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _prototypes(self) -> torch.Tensor | None:
        weights = getattr(self.model, "alignment_prototypes", None)
        return None if weights is None else weights.weight

    def _frame_loss(self, output: Any, batch: Any) -> torch.Tensor:
        """Direct supervision on the predicted frame.

        Supervising placement is not the same as supplying it. The model is told what the
        right answer was during training and must produce it unaided at inference, which
        is the ordinary arrangement for any predictive head.
        """
        predicted = output.frames
        if predicted is None:
            return torch.zeros((), device=self.device)
        truth = batch.entity_frames
        present = batch.entity_present.unsqueeze(-1).to(predicted.dtype)
        translation = (predicted[..., 0:3] - truth[..., 0:3]).abs()
        log_scale = (predicted[..., 3:6] - truth[..., 3:6]).abs()
        rotation = (predicted[..., 6:12] - truth[..., 6:12]).abs()
        weighted = (
            translation.sum(dim=-1, keepdim=True)
            + 0.5 * log_scale.sum(dim=-1, keepdim=True)
            + 0.25 * rotation.sum(dim=-1, keepdim=True)
        )
        loss: torch.Tensor = (weighted * present).sum() / present.sum().clamp_min(1)
        return loss

    def _nested_lod_loss(
        self, whole: WholeOrganBatch, batch: Any, teacher_forcing: float
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Level reconstruction plus the two nesting terms.

        All three levels are decoded every step. Step 7 rotated through one level per
        step, which meant the containment relation between two levels was never present
        in a single graph and could not be asked for.
        """
        decodes: dict[int, torch.Tensor] = {}
        targets: dict[int, torch.Tensor] = {}
        for level in LEVELS:
            prefix = LOD_TOKEN_SCHEDULE[min(level, len(LOD_TOKEN_SCHEDULE) - 1)]
            output = self.model(batch, token_prefix=prefix, teacher_forcing=teacher_forcing)
            decodes[level] = output.scene_logits
            targets[level] = whole.lod_targets[level].to(self.device)

        weights = self.config.nesting
        reconstruction = sum(
            functional_ops.binary_cross_entropy_with_logits(decodes[level], targets[level])
            for level in LEVELS
        ) / len(LEVELS)
        containment = torch.zeros((), device=self.device)
        preservation = torch.zeros((), device=self.device)
        pairs = list(zip(LEVELS, LEVELS[1:], strict=False))
        for coarse, fine in pairs:
            containment = containment + containment_loss(decodes[coarse], decodes[fine])
            preservation = preservation + preservation_loss(
                decodes[coarse], decodes[fine], targets[coarse], targets[fine]
            )
        containment = containment / max(len(pairs), 1)
        preservation = preservation / max(len(pairs), 1)

        total = (
            weights.reconstruction * cast(torch.Tensor, reconstruction)
            + weights.containment * containment
            + weights.preservation * preservation
        )
        record = {
            "lod_reconstruction": float(cast(torch.Tensor, reconstruction).detach()),
            "lod_containment": float(containment.detach()),
            "lod_preservation": float(preservation.detach()),
        }
        return total, record

    # ------------------------------------------------------------------
    def train_step(self, whole: WholeOrganBatch, step: int) -> dict[str, float]:
        """One optimisation step."""
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self._learning_rate(step)
        ratio = teacher_forcing_at(step, self.config.steps, self.config)
        batch = whole.batch.to(self.device)

        output = self.model(batch, teacher_forcing=ratio)
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
        frame_term = self._frame_loss(output, batch)
        lod_term, lod_record = self._nested_lod_loss(whole, batch, ratio)
        total = (
            breakdown.total
            + self.config.frame_weight * frame_term
            + self.config.lod_weight * lod_term
        )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()

        record: dict[str, float] = breakdown.as_floats()
        record.update(lod_record)
        record["total"] = float(total.detach())
        record["frame"] = float(frame_term.detach())
        record["teacher_forcing"] = ratio
        record["grad_norm"] = float(grad_norm)
        record["learning_rate"] = self._learning_rate(step)
        record["step"] = float(step)
        return record

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, scenes: Sequence[WholeOrganScene] | None = None) -> dict[str, float]:
        """Evaluate with **predicted** placement, over a whole split.

        There is no teacher-forcing argument here on purpose. The headline condition is
        predicted placement and a schedule left set by mistake must not be able to turn it
        into an oracle result.
        """
        from experiments.step8.evaluation import evaluate_step8

        self.model.eval()
        return evaluate_step8(
            self.model,
            scenes if scenes is not None else self.eval_scenes,
            self.builder,
            self.slot_of,
            device=self.device,
            batch_size=self.config.batch_size,
            use_predicted_frames=True,
        )

    def fit(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        on_step: Callable[[Step8Trainer, int], None] | None = None,
    ) -> RunManifest:
        """Train, then evaluate with predicted placement.

        ``on_step`` is Step 16's diagnostic hook. It is called with the trainer and the step
        *about to run*, so a call at step 0 observes the initialisation, and once more after the
        loop with ``steps`` as the step. It defaults to ``None``, in which case this method is
        byte-for-byte the loop every run from Step 8 onward used: a callback that is never
        installed cannot change a trajectory. A callback that touches RNG, the optimiser or
        ``model.training`` *would*, so the one in ``experiments/step16`` restores all three and
        a test asserts the frozen result is unchanged.
        """
        started = time.time()
        stream: Iterator[WholeOrganBatch] = self.train_loader.infinite()
        for step in range(self.config.steps):
            if on_step is not None:
                on_step(self, step)
            record = self.train_step(next(stream), step)
            if step % self.config.log_every == 0 or step == self.config.steps - 1:
                self.manifest.train_history.append(record)
            if self.config.eval_every and step > 0 and step % self.config.eval_every == 0:
                validation = self.evaluate()
                validation["step"] = float(step)
                self.manifest.validation_history.append(validation)

        if on_step is not None:
            on_step(self, self.config.steps)
        self.manifest.steps_completed = self.config.steps
        final = self.evaluate()
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


def with_arm(config: Step8Config, arm: str, seed: int) -> Step8Config:
    """Copy a configuration for another arm and seed."""
    return replace(config, arm=arm, seed=seed)
