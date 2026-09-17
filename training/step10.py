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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from awr.config import DomainConfig
from awr.ontology import AnatomyOntology
from datasets.whole_organ.corpus import WholeOrganScene
from datasets.whole_organ.hierarchy import HIERARCHIES, parent_slots
from generation.neural.losses import initial_loss_strategy
from generation.neural.nn.nested_lod import NestedLodWeights
from generation.neural.nn.transforms import DEFAULT_PARENT_CONVENTION, PARENT_CONVENTIONS
from training.step9 import Step9Config, Step9Trainer

__all__ = ["PlacementTargetMode", "Step10Config", "Step10Trainer", "default_config", "with_variant"]

PlacementTargetMode = Literal["global", "parent_relative"]


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
        slot_of = {entity_id: index for index, entity_id in enumerate(ontology.ids())}
        return replace(self, placement_parents=parent_slots(self.placement_hierarchy, slot_of))

    def overrides(self) -> dict[str, Any]:
        """Configuration fields this run changes on the model itself."""
        payload = super().overrides()
        payload.update(
            {
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

    def train_step(self, whole: Any, step: int) -> dict[str, float]:
        """One optimisation step, recording which Step 10 switch is active."""
        record = super().train_step(whole, step)
        record["parent_relative"] = float(self.config.placement_target == "parent_relative")
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
