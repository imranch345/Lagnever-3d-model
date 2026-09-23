"""Step 10 training: the Step 9 loop with the placement target switchable.

Change 1 asks for placement predicted relative to a parent. What that changes here is one
setting, ``placement_target``, and what it deliberately leaves alone is everything used to
*judge* the result. The head predicts local frames; the model composes them into scene
coordinates inside :meth:`predict_frames`; the loss, the metrics, the decoder and the
evaluation then see a global frame exactly as they did in Step 9. Step 10's claim is that a
different target teaches the head something the old one could not, and that claim is only
testable if the ruler does not move with it.

Three arms, not two
-------------------

``global``
    Step 9's target, bit for bit, as the control.

``parent_relative`` over the **spatial** hierarchy
    The construction-derived tree from :mod:`datasets.whole_organ.hierarchy`.

``parent_relative`` over the **taxonomic** hierarchy
    AWR's own hierarchy, in which every whole-organ entity is a root because the only
    parents on offer are geometry-free category nodes. It is therefore numerically the same
    target as ``global``, and that is the point: it measures what "use AWR's hierarchy"
    would have delivered, and separates the hierarchy from the rest of Change 1.

What the floor already says about the outcome
---------------------------------------------

``experiments/step10/placement_floor.json``, computed before any of this was trained, says
the parent-relative target is worth 24% of position error to a predictor that places parents
perfectly, and *costs* 1.9% to one that does not. So the expected result is that the change
helps the arms that already clear the Step 9 placement-blind floor and does nothing for the
arms that do not. That prediction is on record first, so confirming it is a result and
contradicting it is also a result.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
import torch

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.hierarchy import HIERARCHIES, parent_slots
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.nested_lod import NestedLodWeights
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.placement_integrity import check_source_table, validate_placement
from generation.neural.nn.transforms import (
    DEFAULT_PARENT_CONVENTION,
    PARENT_CONVENTIONS,
    rotation_chordal,
)
from training.step9 import Step9Config, Step9Trainer

__all__ = [
    "PlacementTargetMode",
    "RotationObjective",
    "Step10Config",
    "Step10Trainer",
    "default_config",
    "with_variant",
]

PlacementTargetMode = Literal["global", "parent_relative"]
RotationObjective = Literal["step8_6d_l1", "chordal"]


@dataclass(frozen=True, slots=True)
class Step10Config(Step9Config):
    """Step 9's settings plus the Change 1 switch."""

    placement_target: PlacementTargetMode = "global"
    """``global`` reproduces Step 9 exactly."""

    placement_hierarchy: str = "spatial"
    """Which tree to compose over. ``taxonomic`` is AWR's, and has no parents at all."""

    parent_convention: str = DEFAULT_PARENT_CONVENTION
    """How much of a parent propagates; only ``isotropic`` and ``rigid`` stay closed."""

    placement_parents: tuple[int, ...] = ()
    """Filled by :meth:`resolved` from the ontology; never set this by hand."""

    relation_values: bool = False
    """Whether a relation contributes a message as well as an attention weight.

    Step 11 measured relation type as worth 0.0003 degrees, because its only channel was a
    scalar attention bias that trained to 0.55% of the logit scale. This is the channel Step
    12 adds. Off reproduces every earlier run exactly.
    """

    rotation_objective: RotationObjective = "step8_6d_l1"
    """How rotation is penalised.

    ``step8_6d_l1`` is Step 8's term — the absolute difference of the six stored numbers —
    and reproduces Change 1 exactly. It is a defensible term only while the target's
    rotation is constant, which on the Change 1 corpus it was: the term was identically
    zero. ``chordal`` is Change 2's: the squared Frobenius distance between the two
    *rotation matrices*, normalised to [0, 1], which is a proper distance on rotations and
    is smooth at zero where the geodesic angle's gradient diverges.
    """

    rotation_loss_weight: float = 0.25
    """The rotation term's coefficient inside the frame loss.

    0.25 is Step 8's declared emphasis, and is correct for ``step8_6d_l1``. For ``chordal``
    the two terms are on different scales, so the value must be calibrated on training data
    before any confirmatory run and recorded in the manifest. See
    ``experiments/step10/calibrate_rotation_weight.py``.
    """

    def __post_init__(self) -> None:
        """Reject a configuration that would train a different arm than it names."""
        if self.placement_hierarchy not in HIERARCHIES:
            raise ValueError(
                f"Unknown hierarchy {self.placement_hierarchy!r}; "
                f"expected one of {tuple(HIERARCHIES)}."
            )
        if self.parent_convention not in PARENT_CONVENTIONS:
            raise ValueError(
                f"Unknown parent convention {self.parent_convention!r}; "
                f"expected one of {PARENT_CONVENTIONS}."
            )

    def resolved(self, ontology: AnatomyOntology) -> Step10Config:
        """Fill in the parent table from the ontology's slot order.

        The batch is indexed by ``ontology.ids()``, not by the twenty whole-organ entities,
        and the two orderings differ. Resolving here, against the same list the batch
        builder uses, is what stops the table from addressing the wrong entities.
        """
        if self.placement_target == "global":
            return self
        # Before the model exists: a corrupt table can contain a cycle, and the model would
        # otherwise fail on it with an error that does not say the table is the problem.
        check_source_table(self.placement_hierarchy)
        slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        return replace(self, placement_parents=parent_slots(self.placement_hierarchy, slot_of))

    def overrides(self) -> dict[str, Any]:
        """Configuration fields this run changes on the model itself."""
        payload = super().overrides()
        payload.update(
            {
                "relation_values": self.relation_values,
                "placement_target": self.placement_target,
                "placement_parents": self.placement_parents,
                "placement_hierarchy": self.placement_hierarchy,
                "parent_convention": self.parent_convention,
            }
        )
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the run manifest."""
        payload = super().to_dict()
        payload["step10"] = {
            "relation_values": self.relation_values,
            "rotation_objective": self.rotation_objective,
            "rotation_loss_weight": self.rotation_loss_weight,
            "placement_target": self.placement_target,
            "placement_hierarchy": self.placement_hierarchy,
            "parent_convention": self.parent_convention,
            "parented_entities": sum(1 for slot in self.placement_parents if slot >= 0),
            "placement_parents": list(self.placement_parents),
            "metric": (
                "unchanged from Step 9: frames are composed into scene coordinates inside "
                "the model, so the loss and every metric see a global frame"
            ),
        }
        return payload


class Step10Trainer(Step9Trainer):
    """The Step 9 trainer with the placement target resolved against the ontology."""

    config: Step10Config

    def __init__(
        self,
        config: Step10Config,
        ontology: AnatomyOntology,
        domain: DomainConfig,
        train_scenes: Sequence[WholeOrganScene],
        eval_scenes: Sequence[WholeOrganScene],
        *,
        dataset_info: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config.resolved(ontology),
            ontology,
            domain,
            train_scenes,
            eval_scenes,
            dataset_info=dataset_info,
        )
        self.manifest.run_id = f"s10-{config.arm}-{self._tag()}-seed{config.seed}"
        self.integrity = self.validate(train_scenes)
        self.manifest.config["step10"]["integrity"] = self.integrity

    def validate(self, scenes: Sequence[WholeOrganScene]) -> dict[str, Any]:
        """Run the §27 gate on one batch of ``scenes``; raise rather than train on bad metadata.

        Every random source is restored afterwards. Building a batch can sample points, and a
        gate that advanced the generator would make this run diverge from Step 9 for a reason
        that has nothing to do with placement.
        """
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            level = scenes[0].active_lod
            whole = self.builder.build(
                [s for s in scenes if s.active_lod == level][: self.config.batch_size]
            )
            report = validate_placement(
                cast(HierarchicalPlacement | None, getattr(self.model, "placement", None)),
                slot_of=self.builder.slot_of,
                hierarchy=self.config.placement_hierarchy,
                frames=whole.batch.entity_frames,
                present=whole.batch.entity_present,
            )
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        return report

    def _tag(self) -> str:
        """Short name for what this run varies, so run ids say what they are."""
        parts = [super()._tag()]
        if self.config.placement_target == "parent_relative":
            parts.append(self.config.placement_hierarchy)
            if self.config.parent_convention != DEFAULT_PARENT_CONVENTION:
                parts.append(self.config.parent_convention)
        else:
            parts.append("globalframe")
        return "-".join(parts)

    def _frame_loss(self, output: Any, batch: Any) -> torch.Tensor:
        """Direct supervision on the predicted frame, with Change 2's rotation term.

        ``step8_6d_l1`` defers to Step 9 and is bit-for-bit Change 1. ``chordal`` keeps the
        translation and scale terms exactly as they were and replaces only the rotation term,
        so a difference between the two is the rotation objective and nothing else.

        The chordal term is a distance between rotation *matrices*, taken after the model's
        own Gram-Schmidt, so it cannot be reduced by inflating the stored basis vectors —
        which the six-number absolute difference could be.
        """
        if self.config.rotation_objective == "step8_6d_l1":
            return super()._frame_loss(output, batch)
        predicted = output.frames
        if predicted is None:
            return torch.zeros((), device=self.device)
        truth = batch.entity_frames
        present = batch.entity_present.unsqueeze(-1).to(predicted.dtype)
        parts = self._frame_loss_parts(predicted, truth)
        weighted = (
            parts["translation"]
            + 0.5 * parts["scale"]
            + self.config.rotation_loss_weight * parts["rotation"]
        )
        loss: torch.Tensor = (weighted * present).sum() / present.sum().clamp_min(1)
        return loss

    def _frame_loss_parts(
        self, predicted: torch.Tensor, truth: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """The three frame terms, unweighted and unmasked, each ``[..., 1]``.

        Exposed separately because the rotation weight has to be calibrated against the
        other two terms' magnitudes, and a calibration that recomputed them slightly
        differently from the loss would calibrate the wrong thing.
        """
        if self.config.frame_objective == "euclidean":
            translation = torch.linalg.norm(
                predicted[..., 0:3] - truth[..., 0:3], dim=-1, keepdim=True
            )
        else:
            translation = (
                (predicted[..., 0:3] - truth[..., 0:3]).abs().sum(dim=-1, keepdim=True)
            )
        return {
            "translation": translation,
            "scale": (predicted[..., 3:6] - truth[..., 3:6]).abs().sum(dim=-1, keepdim=True),
            "rotation": rotation_chordal(predicted, truth).unsqueeze(-1),
        }

    def train_step(self, whole: Any, step: int) -> dict[str, float]:
        """One optimisation step, recording which Step 10 switch is active."""
        record = super().train_step(whole, step)
        record["parent_relative"] = float(self.config.placement_target == "parent_relative")
        record["rotation_loss_weight"] = float(self.config.rotation_loss_weight)
        return record


def with_variant(
    config: Step10Config,
    *,
    placement_target: PlacementTargetMode,
    hierarchy: str = "spatial",
    seed: int,
) -> Step10Config:
    """Copy a configuration for one Step 10 variant."""
    return replace(
        config, placement_target=placement_target, placement_hierarchy=hierarchy, seed=seed
    )


def default_config(arm: str = "A3Lite") -> Step10Config:
    """The Step 9 configuration, unchanged, as the Step 10 control."""
    return Step10Config(
        arm=arm,
        loss_weights=dict(initial_loss_strategy()),
        nesting=NestedLodWeights(),
    )
