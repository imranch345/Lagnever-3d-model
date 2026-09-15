"""Step 9 training: the Step 8 loop with two hypotheses made switchable.

Everything not named here is the Step 8 loop unchanged, so a Step 9 arm differs from its
Step 8 counterpart in exactly the thing under test. The two switches:

**P4, scene context.** ``frame_scene_context`` gives the frame head a masked mean of the
present entities' latents. The frame target is a scene-global centroid while the head reads
a latent whose graph attention reaches about four neighbours with no global pooling
anywhere, so the information needed to place an entity has never reached the head that
places it.

**Objective alignment.** ``frame_objective`` chooses how translation is penalised. Step 8
used per-component L1, whose optimum is the per-axis median, while the metric is Euclidean
distance, whose optimum is the geometric median. ``euclidean`` penalises the distance the
metric actually measures. Scale stays L1 in both, because scale is reported per axis.

The rotation term is left exactly as Step 8 had it. The audit found the frame target's
rotation is a constant identity for every entity in every scene, so the term contributes
nothing; removing it would change the Step 8 composite and break comparability for no gain.
It is reported as structurally zero instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import torch

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.whole_organ.corpus import WholeOrganScene
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.losses import PrototypeLoss
from generation.neural.nn.nested_lod import NestedLodWeights
from training.loop import _hard_negative_table, build_model
from training.manifest import RunManifest
from training.step8 import Step8Config, Step8Trainer

__all__ = ["FrameObjective", "Step9Config", "Step9Trainer"]

FrameObjective = Literal["l1", "euclidean"]


@dataclass(frozen=True, slots=True)
class Step9Config(Step8Config):
    """Step 8's settings plus the two Step 9 switches."""

    frame_scene_context: bool = False
    """P4: give the frame head a masked mean of the present entities' latents."""

    frame_objective: FrameObjective = "l1"
    """How translation is penalised. ``l1`` reproduces Step 8 exactly."""

    def overrides(self) -> dict[str, Any]:
        """Configuration fields this run changes on the model itself."""
        return {"frame_scene_context": self.frame_scene_context}

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the run manifest."""
        payload = super().to_dict()
        payload["step9"] = {
            "frame_scene_context": self.frame_scene_context,
            "frame_objective": self.frame_objective,
            "rotation_term": (
                "unchanged from Step 8, and structurally zero: the frame target's rotation "
                "is a constant identity"
            ),
        }
        payload["corpus"] = "step9-continuous"
        return payload


class Step9Trainer(Step8Trainer):
    """The Step 8 trainer with the frame head's inputs and objective switchable."""

    config: Step9Config

    def __init__(
        self,
        config: Step9Config,
        ontology: AnatomyOntology,
        domain: DomainConfig,
        train_scenes: Sequence[WholeOrganScene],
        eval_scenes: Sequence[WholeOrganScene],
        *,
        dataset_info: Mapping[str, Any] | None = None,
    ) -> None:
        from generation.neural.nn.device import resolve_device, seed_everything
        from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
        from training.whole_organ import WholeOrganLoader

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
            cast(Any, config.arm),
            cast(Any, self.builder),
            text_features=self.text_features,
            overrides=config.overrides(),
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
            run_id=f"s9-{config.arm}-{self._tag()}-seed{config.seed}",
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

    def _tag(self) -> str:
        """Short name for what this run varies, so run ids say what they are."""
        parts = []
        parts.append("ctx" if self.config.frame_scene_context else "noctx")
        parts.append(self.config.frame_objective)
        return "-".join(parts)

    def _frame_loss(self, output: Any, batch: Any) -> torch.Tensor:
        """Direct supervision on the predicted frame.

        ``l1`` is the Step 8 term, bit for bit. ``euclidean`` replaces the translation part
        with the distance the metric measures, leaving scale and rotation as they were, so
        the comparison isolates the alignment and nothing else.
        """
        predicted = output.frames
        if predicted is None:
            return torch.zeros((), device=self.device)
        truth = batch.entity_frames
        present = batch.entity_present.unsqueeze(-1).to(predicted.dtype)
        log_scale = (predicted[..., 3:6] - truth[..., 3:6]).abs()
        rotation = (predicted[..., 6:12] - truth[..., 6:12]).abs()

        if self.config.frame_objective == "euclidean":
            translation = torch.linalg.norm(
                predicted[..., 0:3] - truth[..., 0:3], dim=-1, keepdim=True
            )
        else:
            translation = (predicted[..., 0:3] - truth[..., 0:3]).abs().sum(dim=-1, keepdim=True)

        weighted = (
            translation
            + 0.5 * log_scale.sum(dim=-1, keepdim=True)
            + 0.25 * rotation.sum(dim=-1, keepdim=True)
        )
        loss: torch.Tensor = (weighted * present).sum() / present.sum().clamp_min(1)
        return loss

    def train_step(self, whole: Any, step: int) -> dict[str, float]:
        """One optimisation step, recording which Step 9 switches are active."""
        record = super().train_step(whole, step)
        record["frame_scene_context"] = float(self.config.frame_scene_context)
        return record


def with_variant(
    config: Step9Config, *, scene_context: bool, objective: FrameObjective, seed: int
) -> Step9Config:
    """Copy a configuration for one Step 9 variant."""
    return replace(
        config, frame_scene_context=scene_context, frame_objective=objective, seed=seed
    )


def default_config(arm: str = "A3Lite") -> Step9Config:
    """The Step 8 configuration, unchanged, as the Step 9 control."""
    return Step9Config(
        arm=arm,
        loss_weights=dict(initial_loss_strategy()),
        nesting=NestedLodWeights(),
    )
