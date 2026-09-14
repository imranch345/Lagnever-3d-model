"""Device selection and seeding for the prototype.

CUDA if present, then Apple MPS, then CPU. MPS is not chosen by default: at this
model size the kernel launch overhead outweighs the throughput, which was measured
rather than assumed, so the default is CPU and MPS is opt-in.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["DeviceChoice", "resolve_device", "seed_everything", "device_report"]


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    """The device the prototype will run on, and why."""

    device: torch.device
    name: str
    reason: str

    def __str__(self) -> str:
        return f"{self.name} ({self.reason})"


def resolve_device(preference: str = "auto") -> DeviceChoice:
    """Pick a device.

    Args:
        preference: ``auto``, ``cpu``, ``cuda`` or ``mps``. ``auto`` prefers CUDA,
            then CPU. MPS must be asked for explicitly.

    """
    wanted = preference.lower()
    if wanted == "cuda" or (wanted == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return DeviceChoice(torch.device("cuda"), "cuda", "requested or auto-detected")
    if wanted == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return DeviceChoice(torch.device("mps"), "mps", "requested explicitly")
    return DeviceChoice(torch.device("cpu"), "cpu", "default for this model size")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every random source the prototype uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA in this environment
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def device_report(choice: DeviceChoice) -> dict[str, object]:
    """Environment facts recorded in every experiment manifest."""
    return {
        "device": choice.name,
        "reason": choice.reason,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
