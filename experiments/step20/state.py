"""Step 20: full training state, so a run can be branched exactly from mid-training.

Steps 16 to 19 searched a single trajectory for a precursor and found none. The question left is
causal -- is the escape already determined by the weights, or does the subsequent data ordering
decide it -- and causal questions need an intervention. That needs something the project has never
had: a checkpoint you can *continue from*.

The existing checkpoints hold ``{config, model, parameters}``. They carry no optimiser moments, no
step counter, and nothing about where the data stream had reached, so a continuation from one is a
different experiment wearing the same name. This module adds the missing pieces.

What exact continuation actually requires here, established by audit rather than assumption:

* **model parameters** -- obviously;
* **AdamW state** -- ``exp_avg``, ``exp_avg_sq`` and ``step`` per parameter;
* **the step counter** -- the learning-rate schedule is a pure function of the step
  (``_learning_rate``), so there is no separate scheduler state to save;
* **the data stream's position** -- see :class:`StreamState`;
* **global RNG** -- *required inside the teacher-forcing ramp*. Step 20 measured ``train_step``
  consuming none at step 805, which holds there: 805 is in ``P2``, where the ratio is zero. But
  during ``P1`` (steps 180-719 at the frozen budget) the model draws a per-entity mask with
  ``torch.rand`` (``generation/neural/nn/model.py``). Step 21 found this by branching inside the
  ramp, where a reference that did not restore RNG failed to reproduce frozen seed 9. The state
  always saved RNG; :func:`load_state` now restores it.

The data stream is the subtle one. ``WholeOrganLoader.epoch`` draws twice from its own generator to
build a whole epoch's batch list up front, then yields from it, and ``infinite`` counts epochs in a
local variable that cannot be pickled. Saving the generator mid-epoch is therefore not enough to
resume: the permutation it produced is already spent. :class:`StreamState` instead records the
generator state **as it was at the start of the current epoch**, plus the epoch index and how many
batches have been consumed from it, which is sufficient to rebuild the epoch and skip forward.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = ["StreamState", "ResumableStream", "save_state", "load_state", "fingerprint"]


@dataclass
class StreamState:
    """Where the training data stream had reached.

    ``generator_at_epoch_start`` is the loader generator's state *before* the current epoch drew
    its permutations, because those draws are what determine the epoch's batch list. Restoring
    mid-epoch state instead would rebuild a different epoch.
    """

    generator_at_epoch_start: torch.Tensor
    epoch_index: int
    batches_consumed_in_epoch: int

    def to_payload(self) -> dict[str, Any]:
        """Plain tensors and ints, for ``torch.save``."""
        return {
            "generator_at_epoch_start": self.generator_at_epoch_start.clone(),
            "epoch_index": self.epoch_index,
            "batches_consumed_in_epoch": self.batches_consumed_in_epoch,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> StreamState:
        """Rebuild from a saved payload."""
        return cls(
            generator_at_epoch_start=payload["generator_at_epoch_start"],
            epoch_index=int(payload["epoch_index"]),
            batches_consumed_in_epoch=int(payload["batches_consumed_in_epoch"]),
        )


class ResumableStream:
    """``loader.infinite()``, but able to say where it is and to be put back there.

    Wraps rather than modifies the loader, so nothing about the frozen training path changes when
    this is not used.
    """

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self._epoch_index = 0
        self._consumed = 0
        self._generator_at_epoch_start = loader.generator.get_state().clone()

    def state(self) -> StreamState:
        """The current position, in a form :meth:`restore` accepts."""
        return StreamState(
            generator_at_epoch_start=self._generator_at_epoch_start.clone(),
            epoch_index=self._epoch_index,
            batches_consumed_in_epoch=self._consumed,
        )

    def iterate(self) -> Iterator[Any]:
        """Yield batches forever, tracking the epoch boundary and the count within it."""
        while True:
            self._generator_at_epoch_start = self.loader.generator.get_state().clone()
            self._consumed = 0
            for whole in self.loader.epoch(self._epoch_index):
                self._consumed += 1
                yield whole
            self._epoch_index += 1

    def restore(self, state: StreamState) -> Iterator[Any]:
        """Return a stream positioned exactly where ``state`` was captured.

        The epoch is rebuilt from the generator state that produced it and then fast-forwarded,
        which is exact rather than approximate: the same permutations are drawn in the same order.
        """
        self.loader.generator.set_state(state.generator_at_epoch_start.clone())
        self._epoch_index = state.epoch_index
        skip = state.batches_consumed_in_epoch

        def stream() -> Iterator[Any]:
            self._generator_at_epoch_start = self.loader.generator.get_state().clone()
            self._consumed = 0
            for index, whole in enumerate(self.loader.epoch(self._epoch_index)):
                self._consumed = index + 1
                if index < skip:
                    continue
                yield whole
            self._epoch_index += 1
            yield from self.iterate()

        return stream()


def save_state(
    *, trainer: Any, stream: ResumableStream, step: int, path: Path, label: str
) -> dict[str, Any]:
    """Everything needed to continue this run exactly, plus a fingerprint of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "step": step,
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "stream": stream.state().to_payload(),
        "config": trainer.config.to_dict(),
        # Required inside the teacher-forcing ramp, where the model draws its mask with
        # torch.rand; not consumed in P0 or P2. Restored by load_state.
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    torch.save(payload, path)
    return fingerprint(payload)


def load_state(*, trainer: Any, path: Path) -> dict[str, Any]:
    """Restore model, optimiser and global RNG in place; return the step and stream position.

    A branch reseeds only the loader's own generator afterwards, so restoring global RNG here
    keeps the teacher-forcing draws identical across branches and the reference -- data order
    remains the only thing a branch changes. Outside the ramp nothing consumes global RNG, so
    this changes no continuation from a ``P0`` or ``P2`` state, Step 20's 805 included.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    return {
        "step": int(payload["step"]),
        "stream": StreamState.from_payload(payload["stream"]),
        "fingerprint": fingerprint(payload),
    }


def fingerprint(payload: dict[str, Any]) -> dict[str, Any]:
    """A compact, comparable summary of a training state.

    Used to demonstrate that every branch started from bit-identical weights and optimiser state.
    Sums and norms rather than a hash of the bytes, because they say *how* two states differ when
    they do, where a hash only says that they do.
    """

    def digest(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
        flat = torch.cat([t.detach().flatten().double() for t in tensors.values()])
        return {
            "count": int(flat.numel()),
            "sum": float(flat.sum()),
            "sum_of_squares": float(flat.pow(2).sum()),
            "min": float(flat.min()),
            "max": float(flat.max()),
        }

    model = digest(payload["model"])
    optimizer_state = payload["optimizer"]["state"]
    moments = {
        f"{key}:{name}": value
        for key, entry in optimizer_state.items()
        for name, value in entry.items()
        if isinstance(value, torch.Tensor)
    }
    return {
        "step": int(payload["step"]),
        "model": model,
        "optimizer_moments": digest(moments) if moments else None,
        "optimizer_entries": len(optimizer_state),
        "stream_epoch": int(payload["stream"]["epoch_index"]),
        "stream_batches_consumed": int(payload["stream"]["batches_consumed_in_epoch"]),
    }
