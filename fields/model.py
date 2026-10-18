"""The Level 2b network: a label for every point of a thorax, from one code per thorax.

``LabelField`` maps a point in millimetres and a subject code to K + 1 logits, one per structure
plus "nothing". Because each point gets exactly one label, two structures can never occupy the
same place: non-overlap is a property of the representation, not a check passed afterwards.

* Points are scaled by ``scale_mm`` and encoded with sinusoidal features at ``bands``
  frequencies, so a small network can draw sharp boundaries.
* The code is concatenated to every layer's input (DeepSDF-style), with one skip connection
  halfway.
* ``AutoDecoder`` holds one learned code per training subject. A new subject is fitted by
  optimising a fresh code with the network frozen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

__all__ = ["DEFAULT_FIELD", "AutoDecoder", "FieldConfig", "LabelField", "pick_device"]


@dataclass(frozen=True)
class FieldConfig:
    """Everything that defines the network's shape; saved with every checkpoint."""

    n_classes: int = 8
    latent_dim: int = 64
    hidden: int = 256
    layers: int = 8
    bands: int = 8
    scale_mm: float = 250.0

    def to_dict(self) -> dict[str, float | int]:
        """Plain data for checkpoints."""
        return asdict(self)


#: The default shape; frozen, so sharing one instance is safe.
DEFAULT_FIELD = FieldConfig()


class LabelField(nn.Module):
    """f(x, z): logits over structures for points ``x`` (mm) given a thorax code ``z``."""

    def __init__(self, config: FieldConfig = DEFAULT_FIELD) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "frequencies", (2.0 ** torch.arange(config.bands)) * math.pi, persistent=False
        )
        encoded = 3 + 6 * config.bands
        width_in = encoded + config.latent_dim
        self.skip_at = config.layers // 2
        layers: list[nn.Module] = []
        for index in range(config.layers):
            fan_in = width_in if index == 0 else config.hidden
            if index == self.skip_at:
                fan_in += width_in
            layers.append(nn.Linear(fan_in, config.hidden))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(config.hidden, config.n_classes)
        self.activation = nn.SiLU()

    def encode(self, points_mm: torch.Tensor) -> torch.Tensor:
        """Scaled coordinates and their sines and cosines."""
        x = points_mm / self.config.scale_mm
        frequencies = cast(torch.Tensor, self.frequencies)  # a registered buffer
        angles = x.unsqueeze(-1) * frequencies
        return torch.cat([x, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)

    def forward(self, points_mm: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """``points_mm`` (B, N, 3) and ``codes`` (B, latent) to logits (B, N, classes)."""
        features = torch.cat(
            [self.encode(points_mm), codes.unsqueeze(1).expand(-1, points_mm.shape[1], -1)],
            dim=-1,
        )
        hidden = features
        for index, layer in enumerate(self.layers):
            if index == self.skip_at:
                hidden = torch.cat([hidden, features], dim=-1)
            hidden = self.activation(layer(hidden))
        logits: torch.Tensor = self.head(hidden)
        return logits


class AutoDecoder(nn.Module):
    """The field plus one learned code per training subject."""

    def __init__(self, n_subjects: int, config: FieldConfig = DEFAULT_FIELD) -> None:
        super().__init__()
        self.field = LabelField(config)
        self.codes = nn.Embedding(n_subjects, config.latent_dim)
        nn.init.normal_(self.codes.weight, mean=0.0, std=0.01)

    def forward(self, points_mm: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        """Logits for the subjects' own codes."""
        logits: torch.Tensor = self.field(points_mm, self.codes(subjects))
        return logits


def pick_device(preferred: str | None = None) -> torch.device:
    """CUDA on a rented GPU, Apple's MPS on this Mac, otherwise the CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
