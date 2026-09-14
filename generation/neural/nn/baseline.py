"""The appearance-driven baseline: arm A0 of the first experiment.

This is the controlled comparison the research hypothesis rests on, so it is built
to be strong rather than easy to beat:

* the same conditioning input as the structured arm,
* the same transformer depth and width, so the parameter budget matches,
* full self-attention over its latent tokens, unconstrained by any graph,
* the same field decoder architecture.

What it does **not** have is the representation under test: no entity axis, no typed
relations, no per-entity geometry, no canonical frames. Part identity is recovered
afterwards by a segmentation head, which is how appearance-driven systems handle it.

If this arm wins, the hypothesis is wrong, and that is the point of running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from generation.neural.nn.geometry import GeometryConfig, OccupancyFieldDecoder
from generation.neural.nn.model import ModelOutput
from generation.neural.nn.tensors import PrototypeBatch

__all__ = ["BaselineConfig", "AppearanceBaseline", "matched_baseline_config"]


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Configuration of the appearance-driven arm."""

    text_features: int = 65
    entity_vocabulary: int = 42
    latent_tokens: int = 40
    width: int = 256
    layers: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    arm: str = "A0_appearance_baseline"


class AppearanceBaseline(nn.Module):
    """Flat latent, shared geometry field, part identity recovered afterwards."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.scene_encoder = nn.Sequential(
            nn.Linear(config.text_features, config.width),
            nn.GELU(),
            nn.Linear(config.width, config.width),
        )
        self.presence_head = nn.Linear(config.width, config.entity_vocabulary)
        self.latent_queries = nn.Parameter(torch.randn(config.latent_tokens, config.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=int(config.width * config.mlp_ratio),
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.token_projection = nn.Linear(config.width, config.geometry.token_width)
        self.field = OccupancyFieldDecoder(config.geometry)
        self.part_head = nn.Sequential(
            nn.Linear(config.geometry.field_width, config.geometry.field_width),
            nn.GELU(),
            nn.Linear(config.geometry.field_width, config.entity_vocabulary + 1),
        )

    def parameter_groups(self) -> dict[str, int]:
        """Parameter count per component, for the matched-budget check."""
        groups = {
            "scene_encoder": sum(p.numel() for p in self.scene_encoder.parameters())
            + sum(p.numel() for p in self.presence_head.parameters()),
            "latent_queries": int(self.latent_queries.numel()),
            "trunk": sum(p.numel() for p in self.trunk.parameters()),
            "token_projection": sum(p.numel() for p in self.token_projection.parameters()),
            "field_decoder": sum(p.numel() for p in self.field.parameters()),
            "part_head": sum(p.numel() for p in self.part_head.parameters()),
        }
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups

    def forward(
        self,
        batch: PrototypeBatch,
        *,
        lod: int | None = None,
        use_predicted_frames: bool = False,
        token_prefix: int | None = None,
    ) -> ModelOutput:
        """Run the baseline. Frame arguments are accepted and ignored: it has none."""
        del use_predicted_frames
        level = batch.structure.lod if lod is None else lod
        active = token_prefix or self.config.geometry.tokens_at(level)

        scene_latent = self.scene_encoder(batch.text_features)
        presence_logits = self.presence_head(scene_latent)
        queries = self.latent_queries.unsqueeze(0).expand(batch.batch_size, -1, -1)
        tokens = self.trunk(queries + scene_latent.unsqueeze(1))
        geometry_tokens = self.token_projection(tokens).unsqueeze(1)

        points = batch.scene_points.unsqueeze(1)
        features = self.field.features(points, geometry_tokens, active_tokens=active)
        scene_logits = self.field.head(features).squeeze(-1).squeeze(1)
        part_raw = self.part_head(features.squeeze(1))
        entity_count = batch.structure.entity_count
        vocabulary = self.config.entity_vocabulary
        part_logits = part_raw.new_full(
            (batch.batch_size, points.shape[2], entity_count + 1), -1.0e9
        )
        part_logits[..., :vocabulary] = part_raw[..., :vocabulary]
        part_logits[..., entity_count] = part_raw[..., vocabulary]
        # Both arms know which entities were requested, so absent classes are masked out
        # here too. Leaving them live would spend the baseline's probability mass on
        # structures the scene does not contain, which would understate it.
        absent = ~batch.entity_present
        part_logits[..., :entity_count] = part_logits[..., :entity_count].masked_fill(
            absent.unsqueeze(1), -1.0e9
        )
        return ModelOutput(
            scene_latent=scene_latent,
            geometry_tokens=geometry_tokens,
            scene_logits=scene_logits,
            part_logits=part_logits,
            presence_logits=presence_logits,
            active_tokens=active,
        )


def matched_baseline_config(
    target_parameters: int,
    *,
    base: BaselineConfig | None = None,
    tolerance: float = 0.05,
) -> BaselineConfig:
    """Size the baseline to match the structured arm's parameter count.

    Searches the feed-forward ratio on a small grid, then trims with the latent token
    count, whose cost per token is exactly the model width. The token count is capped
    at 128 so the match cannot be bought with an absurd attention cost: the arms have
    to be comparable in compute as well as in parameters. A comparison at unmatched
    budget would measure the budget, so this runs before training and the result is
    recorded in the experiment manifest.

    Raises:
        ValueError: if no configuration lands inside ``tolerance``. Failing is
            correct here: quietly running an unmatched comparison is worse.

    """
    template = base or BaselineConfig()
    best: tuple[float, BaselineConfig] | None = None
    for ratio_tenths in range(25, 45):
        ratio = ratio_tenths / 10.0
        candidate = replace(template, mlp_ratio=ratio, latent_tokens=template.latent_tokens)
        count = sum(p.numel() for p in AppearanceBaseline(candidate).parameters())
        shortfall = target_parameters - count
        extra_tokens = int(round(shortfall / candidate.width))
        tokens = min(128, max(16, template.latent_tokens + extra_tokens))
        candidate = replace(candidate, latent_tokens=tokens)
        count = sum(p.numel() for p in AppearanceBaseline(candidate).parameters())
        gap = abs(count - target_parameters) / target_parameters
        if best is None or gap < best[0]:
            best = (gap, candidate)
    assert best is not None
    gap, chosen = best
    if gap > tolerance:
        raise ValueError(
            f"Could not match the baseline to {target_parameters} parameters within "
            f"{tolerance:.0%}; closest was {gap:.2%}. Adjust the search space rather than "
            "running an unmatched comparison."
        )
    return chosen
