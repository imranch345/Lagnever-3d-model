"""Step 10 Change 1: placement expressed relative to a parent.

What changes and what deliberately does not
-------------------------------------------

The head predicts each entity's frame **in its parent's coordinates** instead of in scene
coordinates, and the global frame is recovered by composing down the hierarchy. What does
not change is how the result is *scored*: the composed global frame is compared to the
corpus's global frame with the Step 9 metric, unaltered. That is deliberate. Step 10's
whole claim is that a different target teaches the head something the old one could not, and
that claim is only testable if the ruler stays the same. Scoring in local coordinates would
make the numbers incomparable with Step 9 and would flatter the change, because a local
target has smaller residuals by construction.

The hierarchy this composes over is not AWR's. AWR's is taxonomic — every whole-organ
entity is a leaf under a geometry-free category node — so
:mod:`datasets.whole_organ.hierarchy` writes down the generator's construction order
instead, and keeps the taxonomic version as the control arm.

Why the composition is safe
---------------------------

:mod:`generation.neural.nn.transforms` reduces a parent to rotation and isotropic scale
before composing, which is what keeps twelve-number frames closed under composition once
Change 2 makes rotations real. Without that, the round trip from global to local and back
costs up to 9% of the placement-blind floor and several degrees of rotation, under every
arm equally and indistinguishably from model error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from datasets.whole_organ.hierarchy import parent_slots, topological_order
from generation.neural.nn.transforms import (
    DEFAULT_PARENT_CONVENTION,
    compose,
    compose_hierarchy,
    relative,
    resolve_parents,
)

__all__ = ["HierarchicalPlacement", "PlacementTarget"]


@dataclass(frozen=True)
class PlacementTarget:
    """What the head is asked to predict, and where each entity's parent ended up.

    ``local`` is the supervision target. ``parents`` is per scene because levels of detail
    hide parents, so it is not a property of the hierarchy alone.
    """

    local: torch.Tensor
    parents: torch.Tensor

    @property
    def is_root(self) -> torch.Tensor:
        """``[B, N]``, true where the local frame is already a global one."""
        return self.parents < 0


class HierarchicalPlacement:
    """Converts between global frames and parent-relative ones for one slot layout.

    Built once per run from the batch builder's slot map, because the batch is indexed by
    the full ontology and not by the twenty whole-organ entities. Indexing it with a
    position in ``WHOLE_ORGAN_ENTITIES`` addresses a different entity and raises nothing.
    """

    def __init__(
        self,
        slot_of: dict[str, int],
        *,
        slots: int,
        hierarchy: str = "spatial",
        convention: str = DEFAULT_PARENT_CONVENTION,
    ) -> None:
        self.hierarchy = hierarchy
        self.convention = convention
        self._build(parent_slots(hierarchy, slot_of, slots=slots))

    @classmethod
    def from_parents(
        cls,
        parents: Sequence[int],
        *,
        hierarchy: str = "spatial",
        convention: str = DEFAULT_PARENT_CONVENTION,
    ) -> HierarchicalPlacement:
        """Build from an explicit parent table rather than from a slot map.

        The model is constructed without the batch builder, so the table reaches it through
        the run configuration. Recording it there rather than recomputing it also means the
        manifest says which tree a checkpoint was actually trained against, instead of
        naming a table that may since have changed.
        """
        instance = cls.__new__(cls)
        instance.hierarchy = hierarchy
        instance.convention = convention
        instance._build(tuple(parents))
        return instance

    def _build(self, parents: tuple[int, ...]) -> None:
        self._static = torch.tensor(parents, dtype=torch.long)
        self._order = torch.tensor(topological_order(parents), dtype=torch.long)

    def to(self, device: torch.device) -> HierarchicalPlacement:
        """Move the index tensors; they are small and constant."""
        self._static = self._static.to(device)
        self._order = self._order.to(device)
        return self

    @property
    def static_parents(self) -> torch.Tensor:
        """``[N]`` parent slots before presence is taken into account."""
        return self._static

    def parents_for(self, present: torch.Tensor) -> torch.Tensor:
        """``[B, N]`` parent slots after reparenting around entities this level hides."""
        if self._static.device != present.device:
            self.to(present.device)
        self._pad_to(int(present.shape[1]))
        return resolve_parents(self._static, present, self._order)

    def _pad_to(self, width: int) -> None:
        """Extend the table with roots to the batch's padded entity width.

        The ontology has 42 entities and the batch pads to 64. A table sized to the
        ontology would broadcast against the wider presence mask and silently misalign, so
        the padding slots are made explicit roots instead.
        """
        have = int(self._static.shape[0])
        if width <= have:
            return
        extra = torch.full((width - have,), -1, dtype=torch.long, device=self._static.device)
        self._static = torch.cat([self._static, extra])
        self._order = torch.cat(
            [self._order, torch.arange(have, width, device=self._order.device)]
        )

    @property
    def table(self) -> tuple[int, ...]:
        """The static parent table, for the run manifest."""
        return tuple(int(slot) for slot in self._static.tolist())

    def to_local(self, frames: torch.Tensor, present: torch.Tensor) -> PlacementTarget:
        """Turn the corpus's global frames into the parent-relative target.

        Roots keep their global frame, which is the convention: a root is already expressed
        in scene coordinates, so the head predicts exactly what it predicted in Step 9 for
        those entities. Half of the twenty entities are roots, so half of the target is
        unchanged by Change 1 and the report says so.
        """
        parents = self.parents_for(present)
        local = frames.clone()
        rows = torch.arange(frames.shape[0], device=frames.device)
        for slot in self._order.tolist():
            has_parent = parents[:, slot] >= 0
            if not bool(has_parent.any()):
                continue
            parent_frames = frames[rows, parents[:, slot].clamp_min(0)]
            candidate = relative(parent_frames, frames[:, slot], self.convention)
            local[:, slot] = torch.where(has_parent.unsqueeze(-1), candidate, frames[:, slot])
        return PlacementTarget(local=local, parents=parents)

    def to_global(self, local: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        """Compose predicted local frames into scene coordinates.

        ``parents`` must be the same tensor :meth:`to_local` produced, not recomputed from a
        different presence mask, or prediction and target are composed down different trees.
        """
        return compose_hierarchy(local, parents, self._order, self.convention)

    def compose_one(self, parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        """A single step, for tests and for the depth analysis."""
        return compose(parent, child, self.convention)
