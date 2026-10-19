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
* ``PartField`` (when ``part_of`` is set) gives each part of the thorax its own code and
  trunk: one for the heart, one for the lungs. Each trunk scores "background" and its own
  structures; the scores combine as a product of experts over one softmax:
  - nothing = Σ backgrounds;
  - structure k of part p = its score + the other parts' backgrounds.

  One label per point still holds. A part's code cannot change how the *other* parts decide
  among their own labels and "nothing", so a new person's heart and lungs can be fitted
  without trading one against the other, and "make the heart bigger" leaves the lungs alone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import torch
from torch import nn

__all__ = [
    "DEFAULT_FIELD",
    "AutoDecoder",
    "Field",
    "FieldConfig",
    "LabelField",
    "PartField",
    "make_field",
    "pick_device",
]


@dataclass(frozen=True)
class FieldConfig:
    """Everything that defines the network's shape; saved with every checkpoint."""

    n_classes: int = 8
    latent_dim: int = 64
    hidden: int = 256
    layers: int = 8
    bands: int = 8
    scale_mm: float = 250.0
    #: For classes 1..K, the part each belongs to; empty for one field over every class.
    part_of: tuple[int, ...] = ()

    @property
    def n_parts(self) -> int:
        """How many parts, each with its own code (1 without ``part_of``)."""
        return max(self.part_of) + 1 if self.part_of else 1

    @property
    def code_dim(self) -> int:
        """Length of a subject's whole code: ``latent_dim`` per part."""
        return self.latent_dim * self.n_parts

    def to_dict(self) -> dict[str, Any]:
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


class PartField(nn.Module):
    """One ``LabelField`` trunk and code per part, combined over a single softmax."""

    def __init__(self, config: FieldConfig) -> None:
        super().__init__()
        if not config.part_of or len(config.part_of) != config.n_classes - 1:
            raise ValueError("part_of needs one part index for each class 1..K")
        self.config = config
        self.members = [
            [k for k, part in enumerate(config.part_of, start=1) if part == p]
            for p in range(config.n_parts)
        ]
        if any(not members for members in self.members):
            raise ValueError("every part needs at least one class")
        self.trunks = nn.ModuleList(
            LabelField(replace(config, n_classes=1 + len(members), part_of=()))
            for members in self.members
        )
        order = torch.empty(config.n_classes, dtype=torch.long)
        order[0] = 0
        position = 1
        for members in self.members:
            for k in members:
                order[k] = position
                position += 1
        self.register_buffer("order", order, persistent=False)

    def forward(self, points_mm: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """``points_mm`` (B, N, 3) and ``codes`` (B, parts × latent) to logits (B, N, classes)."""
        size = self.config.latent_dim
        outputs = [
            trunk(points_mm, codes[:, p * size : (p + 1) * size])
            for p, trunk in enumerate(self.trunks)
        ]
        backgrounds = torch.stack([out[..., 0] for out in outputs], dim=-1)
        total = backgrounds.sum(-1)
        parts = [
            out[..., 1:] + (total - backgrounds[..., p]).unsqueeze(-1)
            for p, out in enumerate(outputs)
        ]
        grouped = torch.cat([total.unsqueeze(-1), *parts], dim=-1)
        order = cast(torch.Tensor, self.order)  # a registered buffer
        return grouped.index_select(-1, order)


#: Either kind of field; both take (points, codes) and carry their ``config``.
Field = LabelField | PartField


def make_field(config: FieldConfig = DEFAULT_FIELD) -> Field:
    """A ``PartField`` when ``config.part_of`` is set, otherwise a ``LabelField``."""
    return PartField(config) if config.part_of else LabelField(config)


class AutoDecoder(nn.Module):
    """The field plus one learned code per training subject."""

    def __init__(self, n_subjects: int, config: FieldConfig = DEFAULT_FIELD) -> None:
        super().__init__()
        self.field = make_field(config)
        self.codes = nn.Embedding(n_subjects, config.code_dim)
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
