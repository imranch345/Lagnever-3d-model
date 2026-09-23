"""Make two architectures share every weight they have in common at initialisation.

Step 12 established why this matters. A3's three-seed spread is about 1.0 degree, which is also
the smallest effect these studies are built to detect, so a treatment whose initialisation
differs from its control differs by *seed* as well as by architecture — and the two are then
indistinguishable in the result.

Step 12's confound was a new tensor consuming RNG, and `torch.random.fork_rng` solved it. Step
13's H6 cannot be solved that way: enabling the frame head's scene context **reshapes** an
existing weight (`trunk.0.weight` widens from 256 to 512 inputs), so no amount of stream
management makes the two constructions line up, and every module built after the frame head
shifts too. Measured on A3: 37 shared parameters differ, including the geometry tokeniser.

So instead of trying to make one stream produce both, this builds both and copies. Every tensor
the two models share by name *and* shape is taken from the reference, leaving only the tensors
the treatment genuinely introduces or changes shape. The reference is never modified, so a
control's initialisation stays exactly what it was before this module existed and the frozen
references remain reproducible.

What survives as a difference is then, by construction, the treatment itself — and
:meth:`AlignmentReport.isolated_to` is how an experiment asserts it expected precisely those.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["AlignmentReport", "align_shared_initialisation"]


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """What was shared, and what is left differing on purpose."""

    copied: tuple[str, ...]
    treatment_only: tuple[str, ...]
    reshaped: tuple[str, ...]
    reference_only: tuple[str, ...]

    @property
    def intended_differences(self) -> tuple[str, ...]:
        """Tensors the treatment owns: new ones, plus ones whose shape it changed."""
        return tuple(sorted(self.treatment_only + self.reshaped))

    def isolated_to(self, expected: Collection[str]) -> bool:
        """Whether the differences are exactly ``expected`` and nothing else."""
        return set(self.intended_differences) == set(expected)

    def to_dict(self) -> dict[str, object]:
        """Plain data, for a run manifest."""
        return {
            "tensors_copied_from_control": len(self.copied),
            "treatment_only": list(self.treatment_only),
            "reshaped_by_treatment": list(self.reshaped),
            "reference_only": list(self.reference_only),
            "intended_differences": list(self.intended_differences),
        }


def align_shared_initialisation(treatment: nn.Module, reference: nn.Module) -> AlignmentReport:
    """Copy every same-name, same-shape tensor from ``reference`` into ``treatment``.

    Both models must have been constructed at the same seed for this to mean anything: the
    point is to remove the *incidental* differences that a shape change causes downstream, not
    to transplant one seed's weights into another's.

    ``reference`` is read only. Returns what moved and what did not.
    """
    target = treatment.state_dict()
    source = reference.state_dict()

    shared = {
        name: tensor
        for name, tensor in source.items()
        if name in target and target[name].shape == tensor.shape
    }
    reshaped = tuple(
        sorted(n for n in source if n in target and target[n].shape != source[n].shape)
    )
    treatment.load_state_dict(shared, strict=False)

    with torch.no_grad():
        for name, tensor in shared.items():
            if not torch.equal(treatment.state_dict()[name], tensor):  # pragma: no cover
                raise RuntimeError(f"alignment did not take effect for {name}")

    return AlignmentReport(
        copied=tuple(sorted(shared)),
        treatment_only=tuple(sorted(set(target) - set(source))),
        reshaped=reshaped,
        reference_only=tuple(sorted(set(source) - set(target))),
    )
