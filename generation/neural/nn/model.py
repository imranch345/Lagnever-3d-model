"""The Lagnav structured prototype, with ablation switches.

Implements the Step 5 pipeline as an executable model:

    AWR features -> entity latents -> partitioned graph encoder -> anatomical latent
                 -> per-entity geometry tokens -> per-entity occupancy fields
                 -> scene composition + part attribution

Ablations are configuration, not separate code paths copied by hand, so the arms of
the experiment cannot drift apart:

======  ========================================  ==============================
key     switch                                    isolates
======  ========================================  ==============================
A1      ``use_graph_encoder=False``               factorisation without relations
A2      ``graph_scope="structure_only"``          hierarchy without space or flow
A3      defaults                                  the full proposal
A4      ``per_entity_geometry=False``             per-entity fields
A5      ``per_lod_blocks=True``                   nested level-of-detail prefixes
A6      ``alignment="pairwise"``                  ontology-anchored alignment
A3Lite  ``use_untyped_graph=True``                relations without relation types
======  ========================================  ==============================

**Placement (Step 8).** Frames were teacher-forced throughout Step 7: the decoder always
consumed the ground-truth canonical frame, so the frame head was never exercised and
asking the model to place structures itself halved its ownership accuracy. Step 8 adds a
``teacher_forcing`` ratio that falls to zero during training, and a frame head with enough
capacity to be worth training. ``use_predicted_frames`` overrides the ratio entirely, so
an evaluation can never see a true frame however the schedule is left set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from torch import nn
from torch import nn as _nn

from awr.relationships import GraphKind
from generation.neural.graph_encoder import HeadAllocation, HeadRole
from generation.neural.nn.entity import EntityLatentComposer, EntityLatentConfig
from generation.neural.nn.geometry import (
    FrameHead,
    FramePredictor,
    GeometryConfig,
    GeometryTokenGenerator,
    OccupancyFieldDecoder,
    compose_scene,
    points_to_local,
)
from generation.neural.nn.graph import GraphEncoderConfig, PartitionedGraphEncoder
from generation.neural.nn.graph_lite import GraphLiteConfig, UntypedGraphEncoder
from generation.neural.nn.placement import HierarchicalPlacement
from generation.neural.nn.tensors import AWRStructure, PrototypeBatch

__all__ = ["PrototypeConfig", "ModelOutput", "LagnavPrototype"]

GraphScope = Literal["all", "structure_only"]
AlignmentMode = Literal["prototype", "pairwise"]


@dataclass(frozen=True, slots=True)
class PrototypeConfig:
    """Configuration of the structured prototype, including every ablation switch."""

    entity_vocabulary: int = 42
    anatomy_type_vocabulary: int = 14
    semantic_role_vocabulary: int = 23
    laterality_vocabulary: int = 4
    relation_vocabulary: int = 18
    entity_width: int = 256
    scene_width: int = 256
    identity_width: int = 96
    graph_layers: int = 4
    heads: HeadAllocation = field(default_factory=HeadAllocation.default)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    align_width: int = 256
    text_features: int = 65
    scene_conditioning: Literal["lod_only", "text"] = "lod_only"
    """What the geometry path conditions on.

    ``lod_only`` is the design's position: geometry depends on anatomy and level of
    detail, never on presentation state. It is what makes a visibility or opacity edit
    provably free, and it is checked by test rather than asserted.
    """

    lod_levels: int = 5
    use_graph_encoder: bool = True
    entity_mixer_layers: int = 0
    """Per-entity depth used when the graph encoder is absent.

    Ablation A1-matched: the same capacity as the graph encoder, applied to each entity
    independently, so no information can move between entities. It is what separates
    "relations matter" from "capacity matters".
    """

    graph_scope: GraphScope = "all"
    relation_values: bool = False
    """Whether a relation contributes a message as well as an attention weight.

    Off everywhere up to Step 11, whose measurement was that relation type changed the
    answer by 0.0003 degrees. Step 12 tests whether giving a relation its own vector closes
    part of the 9-degree gap to a graph-aware lookup.
    """
    use_untyped_graph: bool = False
    """Step 8: replace the partitioned typed encoder with untyped graph attention.

    Mutually exclusive with ``use_graph_encoder``. Step 7 found the model read the
    graph's connectivity and almost never its relation types, so the typed machinery is
    dropped and the connectivity kept.
    """

    untyped_layers: int = 2
    untyped_heads: int = 4
    untyped_mlp_ratio: float = 1.0
    deep_frame_head: bool = True
    """Step 8: use the residual-MLP frame predictor instead of the linear probe.

    Step 7's head was 3,084 parameters and was never exercised, because evaluation
    supplied the true frames. Predicting placement is a real task and needs a real head.
    """

    frame_hidden: int = 256
    frame_layers: int = 2
    frame_scene_context: bool = False
    """Step 9, hypothesis P4: give the frame head a summary of the whole scene.

    The frame target is a scene-global centroid, while the head reads a latent whose graph
    attention reaches about four neighbours with no global pooling. Off by default, so the
    Step 8 configuration is preserved exactly and the comparison is between two settings of
    one module.
    """
    placement_target: str = "global"
    """Step 10, Change 1: whether the frame head predicts global or parent-relative frames.

    ``parent_relative`` makes the head predict each entity in its parent's coordinates and
    composes the result back into scene coordinates before anything downstream sees it. The
    loss, the metrics, the decoder and the evaluation are therefore unchanged: Step 10's
    claim is that a different *target* teaches the head something the old one could not, and
    that is only testable if the ruler stays the same.
    """

    placement_parents: tuple[int, ...] = ()
    """Parent slot per entity slot, -1 for a root. Empty unless ``placement_target`` is set.

    Carried here rather than derived, because the model is built without the batch builder
    and because a checkpoint should record the tree it was trained against.
    """

    placement_hierarchy: str = "spatial"
    parent_convention: str = "isotropic"
    """See :mod:`generation.neural.nn.transforms`; ``isotropic`` is the one that stays closed."""

    per_entity_geometry: bool = True
    alignment: AlignmentMode = "prototype"
    arm: str = "lagnav_structured"

    def with_ablation(self, key: str) -> PrototypeConfig:
        """Return the configuration for an ablation key."""
        from dataclasses import replace

        if key == "A3":
            return replace(self, arm="A3_full")
        if key == "A1":
            return replace(self, use_graph_encoder=False, arm="A1_entity_axis_only")
        if key == "A1M":
            return replace(
                self,
                use_graph_encoder=False,
                entity_mixer_layers=6,
                arm="A1matched_no_graph_matched_capacity",
            )
        if key == "A3L":
            return replace(
                self,
                graph_layers=1,
                heads=HeadAllocation(
                    {
                        HeadRole.STRUCTURE: 1,
                        HeadRole.SPATIAL: 1,
                        HeadRole.FUNCTIONAL: 1,
                        HeadRole.GLOBAL: 1,
                    }
                ),
                arm="A3lite_small_graph_encoder",
            )
        if key == "A3Lite":
            return replace(
                self,
                use_graph_encoder=False,
                use_untyped_graph=True,
                arm="A3lite_untyped_graph_attention",
            )
        if key == "A2":
            return replace(self, graph_scope="structure_only", arm="A2_structure_graph_only")
        if key == "A4":
            return replace(self, per_entity_geometry=False, arm="A4_shared_field")
        if key == "A5":
            return replace(
                self,
                geometry=replace(self.geometry, per_lod_blocks=True),
                arm="A5_per_lod_latents",
            )
        if key == "A6":
            return replace(self, alignment="pairwise", arm="A6_pairwise_alignment")
        raise ValueError(
            f"Unknown ablation key {key!r}. A0 is the separate appearance baseline; "
            "structured ablations are A1 to A6."
        )


@dataclass(slots=True)
class ModelOutput:
    """Everything a forward pass produces, for losses, metrics and diagnostics."""

    scene_latent: torch.Tensor
    geometry_tokens: torch.Tensor
    scene_logits: torch.Tensor
    part_logits: torch.Tensor
    presence_logits: torch.Tensor
    active_tokens: int
    entity_latent_in: torch.Tensor | None = None
    entity_latent: torch.Tensor | None = None
    frames: torch.Tensor | None = None
    entity_field_logits: torch.Tensor | None = None
    scene_entity_logits: torch.Tensor | None = None
    alignment: torch.Tensor | None = None

    @property
    def has_per_entity_geometry(self) -> bool:
        """Whether this arm decodes a separate field per entity."""
        return self.entity_field_logits is not None


class _PerEntityMixer(_nn.Module):
    """Depth applied to each entity on its own, with no path between entities.

    The control for "is it the relations or is it the capacity?". Same width, same
    residual structure and the same write-protected identity slice as the graph encoder;
    the only thing missing is attention across the entity axis.
    """

    def __init__(
        self, *, width: int, identity_width: int, layers: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.identity_width = identity_width
        context = width - identity_width
        hidden = int(width * mlp_ratio)
        self.blocks = _nn.ModuleList(
            [
                _nn.Sequential(
                    _nn.LayerNorm(width),
                    _nn.Linear(width, hidden),
                    _nn.GELU(),
                    _nn.Linear(hidden, context),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Update the context slice only."""
        identity = latent[..., : self.identity_width]
        context = latent[..., self.identity_width :]
        for block in self.blocks:
            context = context + block(torch.cat([identity, context], dim=-1))
        return torch.cat([identity, context], dim=-1)


def _masked_graph_structure(structure: AWRStructure, scope: GraphScope) -> AWRStructure:
    """Restrict which graphs the encoder may use, for ablation A2."""
    if scope == "all":
        return structure
    keep = list(GraphKind).index(GraphKind.STRUCTURE)
    adjacency = torch.zeros_like(structure.graph_adjacency)
    adjacency[keep] = structure.graph_adjacency[keep]
    edge_mask = structure.edge_mask & (structure.edge_graph == keep)
    return AWRStructure(
        lod=structure.lod,
        entity_ids=structure.entity_ids,
        entity_type=structure.entity_type,
        semantic_role=structure.semantic_role,
        laterality=structure.laterality,
        parent_index=structure.parent_index,
        depth=structure.depth,
        lod_min=structure.lod_min,
        lod_max=structure.lod_max,
        entity_state=structure.entity_state,
        entity_mask=structure.entity_mask,
        edge_source=structure.edge_source,
        edge_target=structure.edge_target,
        edge_relation=structure.edge_relation,
        edge_graph=structure.edge_graph,
        edge_mask=edge_mask,
        graph_adjacency=adjacency,
        visible_slots=structure.visible_slots,
    )


@dataclass(frozen=True, slots=True)
class _Staged:
    """Everything the pipeline produces before the geometry decode."""

    level: int
    active_tokens: int
    entity_latent_in: torch.Tensor
    entity_latent: torch.Tensor
    scene_latent: torch.Tensor
    tokens: torch.Tensor | None


class LagnavPrototype(nn.Module):
    """The structured arm: entity axis, typed graphs, per-entity geometry."""

    def __init__(self, config: PrototypeConfig, inverse_relations: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        heads = (
            config.heads
            if config.graph_scope == "all"
            else HeadAllocation({HeadRole.STRUCTURE: 6, HeadRole.GLOBAL: 2})
        )
        self.composer = EntityLatentComposer(
            EntityLatentConfig(
                entity_vocabulary=config.entity_vocabulary,
                anatomy_type_vocabulary=config.anatomy_type_vocabulary,
                semantic_role_vocabulary=config.semantic_role_vocabulary,
                laterality_vocabulary=config.laterality_vocabulary,
                width=config.entity_width,
            )
        )
        self.entity_mixer: _PerEntityMixer | None = None
        if not config.use_graph_encoder and config.entity_mixer_layers > 0:
            self.entity_mixer = _PerEntityMixer(
                width=config.entity_width,
                identity_width=config.identity_width,
                layers=config.entity_mixer_layers,
            )
        self.untyped_graph: UntypedGraphEncoder | None = None
        if config.use_untyped_graph:
            if config.use_graph_encoder:
                raise ValueError(
                    "use_untyped_graph and use_graph_encoder are alternatives; enabling "
                    "both would make the arm neither."
                )
            self.untyped_graph = UntypedGraphEncoder(
                GraphLiteConfig(
                    width=config.entity_width,
                    identity_width=config.identity_width,
                    layers=config.untyped_layers,
                    heads=config.untyped_heads,
                    mlp_ratio=config.untyped_mlp_ratio,
                )
            )
        self.graph_encoder: PartitionedGraphEncoder | None = None
        if config.use_graph_encoder:
            self.graph_encoder = PartitionedGraphEncoder(
                GraphEncoderConfig(
                    width=config.entity_width,
                    identity_width=config.identity_width,
                    layers=config.graph_layers,
                    heads=heads,
                    relation_vocabulary=config.relation_vocabulary,
                    relation_values=config.relation_values,
                ),
                inverse_relations,
            )
        conditioning_width = (
            config.lod_levels if config.scene_conditioning == "lod_only" else config.text_features
        )
        self.scene_encoder = nn.Sequential(
            nn.Linear(conditioning_width, config.scene_width),
            nn.GELU(),
            nn.Linear(config.scene_width, config.scene_width),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_features, config.scene_width),
            nn.GELU(),
            nn.Linear(config.scene_width, config.scene_width),
        )
        self.presence_head = nn.Linear(config.scene_width, config.entity_vocabulary)
        self.frame_head: nn.Module = (
            FramePredictor(
                config.entity_width,
                hidden=config.frame_hidden,
                layers=config.frame_layers,
                scene_context=config.frame_scene_context,
            )
            if config.deep_frame_head
            else FrameHead(config.entity_width)
        )
        self.placement: HierarchicalPlacement | None = None
        if config.placement_target == "parent_relative":
            if not config.placement_parents:
                raise ValueError(
                    "placement_target='parent_relative' needs placement_parents; an empty "
                    "table would make every entity a root and the arm would silently be "
                    "the global control."
                )
            self.placement = HierarchicalPlacement.from_parents(
                config.placement_parents,
                hierarchy=config.placement_hierarchy,
                convention=config.parent_convention,
            )
        elif config.placement_target != "global":
            raise ValueError(
                f"Unknown placement_target {config.placement_target!r}; "
                "expected 'global' or 'parent_relative'."
            )
        self.tokens = GeometryTokenGenerator(config.geometry)
        self.field = OccupancyFieldDecoder(config.geometry)
        self.alignment_entity = nn.Linear(config.entity_width, config.align_width)
        self.alignment_prototypes = nn.Embedding(config.entity_vocabulary, config.align_width)
        self.shared_part_head: nn.Sequential | None = None
        if not config.per_entity_geometry:
            self.shared_part_head = nn.Sequential(
                nn.Linear(config.geometry.field_width, config.geometry.field_width),
                nn.GELU(),
                nn.Linear(config.geometry.field_width, config.entity_vocabulary + 1),
            )
            self.shared_projection = nn.Linear(config.entity_width, config.entity_width)

    # ------------------------------------------------------------------
    def parameter_groups(self) -> dict[str, int]:
        """Parameter count per component, for the matched-budget check."""
        groups = {
            "entity_composer": sum(p.numel() for p in self.composer.parameters()),
            "graph_encoder": (
                sum(p.numel() for p in self.graph_encoder.parameters())
                if self.graph_encoder is not None
                else 0
            ),
            "untyped_graph": (
                sum(p.numel() for p in self.untyped_graph.parameters())
                if self.untyped_graph is not None
                else 0
            ),
            "entity_mixer": (
                sum(p.numel() for p in self.entity_mixer.parameters())
                if self.entity_mixer is not None
                else 0
            ),
            "scene_encoder": sum(p.numel() for p in self.scene_encoder.parameters())
            + sum(p.numel() for p in self.text_encoder.parameters())
            + sum(p.numel() for p in self.presence_head.parameters()),
            "frame_head": sum(p.numel() for p in self.frame_head.parameters()),
            "geometry_tokens": sum(p.numel() for p in self.tokens.parameters()),
            "field_decoder": sum(p.numel() for p in self.field.parameters()),
            "alignment": sum(p.numel() for p in self.alignment_entity.parameters())
            + sum(p.numel() for p in self.alignment_prototypes.parameters()),
            "shared_part_head": (
                sum(p.numel() for p in self.shared_part_head.parameters())
                + sum(p.numel() for p in self.shared_projection.parameters())
                if self.shared_part_head is not None
                else 0
            ),
        }
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups

    def scene_summary(self, entity_latent: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Mean of the present entities' latents: a permutation-invariant scene summary.

        Masked by presence, so padded slots cannot dilute it and a scene's summary does not
        depend on how many slots happen to be padded. It is a function of latents the model
        already produced, so it introduces no new information and cannot leak placement.
        """
        weights = present.unsqueeze(-1).to(entity_latent.dtype)
        return (entity_latent * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def predict_frames(self, entity_latent: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Run the frame head, and return frames in **scene** coordinates either way.

        Under ``parent_relative`` the head's output is read as each entity's frame in its
        parent's coordinates and composed down the hierarchy here, so every caller keeps
        receiving a global frame. Composing at this one point is what guarantees the Step 9
        metric is applied unchanged; doing it in the loss would leave evaluation scoring
        local frames against global truth and the numbers would be quietly incomparable.
        """
        if self.config.frame_scene_context:
            predicted = cast(
                torch.Tensor,
                self.frame_head(entity_latent, self.scene_summary(entity_latent, present)),
            )
        else:
            predicted = cast(torch.Tensor, self.frame_head(entity_latent))
        if self.placement is None:
            return predicted
        return self.placement.to_global(predicted, self.placement.parents_for(present))

    def _select_frames(
        self,
        predicted: torch.Tensor,
        truth: torch.Tensor,
        *,
        use_predicted_frames: bool,
        teacher_forcing: float,
    ) -> torch.Tensor:
        """Choose between predicted and supplied frames, per entity.

        Evaluation takes the first branch and can never see a true frame, whatever the
        schedule says. That ordering is deliberate: a teacher-forcing ratio left set by
        mistake must not be able to turn a headline result into an oracle result.
        """
        if use_predicted_frames:
            return predicted
        if teacher_forcing >= 1.0:
            return truth
        if teacher_forcing <= 0.0:
            return predicted
        keep = torch.rand(truth.shape[:-1], device=truth.device) < teacher_forcing
        return torch.where(keep.unsqueeze(-1), truth, predicted)

    def stage(
        self,
        batch: PrototypeBatch,
        *,
        lod: int | None = None,
        token_prefix: int | None = None,
    ) -> _Staged:
        """Run the pipeline up to the geometry tokens and stop.

        :meth:`forward` uses this, and so does the edit head, so an edit is applied to
        exactly the latents and tokens the ordinary forward pass would have produced.
        Reimplementing the stage outside the model is how the two quietly diverge.
        """
        structure = batch.structure
        level = structure.lod if lod is None else lod
        active_tokens = token_prefix or self.config.geometry.tokens_at(level)

        entity_in = self.composer(structure, batch.batch_size)
        if self.graph_encoder is not None:
            scoped = _masked_graph_structure(structure, self.config.graph_scope)
            entity_latent = self.graph_encoder(entity_in, scoped)
        elif self.untyped_graph is not None:
            entity_latent = self.untyped_graph(entity_in, structure)
        elif self.entity_mixer is not None:
            entity_latent = self.entity_mixer(entity_in)
        else:
            entity_latent = entity_in

        conditioning = (
            self._lod_one_hot(level, batch.batch_size, batch.text_features)
            if self.config.scene_conditioning == "lod_only"
            else batch.text_features
        )
        scene_latent = self.scene_encoder(conditioning)
        tokens = (
            self.tokens(entity_latent, scene_latent, lod=level)
            if self.config.per_entity_geometry
            else None
        )
        return _Staged(
            level=level,
            active_tokens=active_tokens,
            entity_latent_in=entity_in,
            entity_latent=entity_latent,
            scene_latent=scene_latent,
            tokens=tokens,
        )

    def decode_tokens(
        self,
        batch: PrototypeBatch,
        tokens: torch.Tensor,
        *,
        frames: torch.Tensor,
        present: torch.Tensor,
        active_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render a scene from geometry token blocks supplied from outside.

        Separated from :meth:`forward` so that an edit head can rewrite the token blocks
        and have the result rendered by exactly the same decoder, with no second code
        path that could drift from the first.
        """
        structure = batch.structure
        # Decode only the slots that are visible at this level of detail. Batches are
        # bucketed by level, so the visible set is constant across the batch; decoding
        # all 64 padded slots would cost four times as much for identical results.
        slots = structure.visible_slots
        if not bool(present[:, slots].all()):
            raise ValueError(
                "Presence disagrees with the level-of-detail visible set; batches must be "
                "bucketed by level for the gathered decode to be valid."
            )
        gathered_tokens = tokens.index_select(1, slots)
        gathered_frames = frames.index_select(1, slots)
        local_entity_points = points_to_local(
            batch.entity_points.index_select(1, slots), gathered_frames
        )
        gathered_entity_logits = self.field(
            local_entity_points, gathered_tokens, active_tokens=active_tokens
        )
        scene_points = batch.scene_points.unsqueeze(1).expand(-1, slots.numel(), -1, -1)
        local_scene_points = points_to_local(scene_points, gathered_frames)
        gathered_scene_logits = self.field(
            local_scene_points, gathered_tokens, active_tokens=active_tokens
        )
        entity_field_logits = self._scatter_slots(
            gathered_entity_logits, slots, structure.entity_count, fill=0.0
        )
        scene_entity_logits = self._scatter_slots(
            gathered_scene_logits, slots, structure.entity_count, fill=-1.0e9
        )
        scene_logits, part_logits = compose_scene(scene_entity_logits, present)
        return scene_logits, part_logits, entity_field_logits, scene_entity_logits

    def forward(
        self,
        batch: PrototypeBatch,
        *,
        lod: int | None = None,
        use_predicted_frames: bool = False,
        token_prefix: int | None = None,
        teacher_forcing: float = 1.0,
    ) -> ModelOutput:
        """Run the pipeline for one batch.

        Args:
            batch: entities, relations, sampled points and their labels.
            lod: level of detail to decode at; the batch's own level when omitted.
            use_predicted_frames: decode from the head's own frames. This is the Step 8
                headline condition and the only honest one, since supplying frames hands
                the model the arrangement a relationship graph would otherwise supply.
            token_prefix: how many geometry tokens to read; the level's own count when
                omitted.
            teacher_forcing: probability, per entity, of substituting the true frame
                during training. 1.0 reproduces the Step 7 behaviour, 0.0 is full
                predicted placement. Ignored when ``use_predicted_frames`` is set, so an
                evaluation cannot accidentally leak frames through the schedule.

        """
        structure = batch.structure
        staged = self.stage(batch, lod=lod, token_prefix=token_prefix)
        level = staged.level
        active_tokens = staged.active_tokens
        entity_in = staged.entity_latent_in
        entity_latent = staged.entity_latent
        scene_latent = staged.scene_latent
        presence_logits = self.presence_head(self.text_encoder(batch.text_features))
        present = batch.entity_present
        frames_pred = self.predict_frames(entity_latent, present)
        frames = self._select_frames(
            frames_pred,
            batch.entity_frames,
            use_predicted_frames=use_predicted_frames,
            teacher_forcing=teacher_forcing,
        )

        alignment = self.alignment_entity(entity_latent)

        if self.config.per_entity_geometry:
            assert staged.tokens is not None
            tokens = staged.tokens
            (
                scene_logits,
                part_logits,
                entity_field_logits,
                scene_entity_logits,
            ) = self.decode_tokens(
                batch, tokens, frames=frames, present=present, active_tokens=active_tokens
            )
            return ModelOutput(
                scene_latent=scene_latent,
                geometry_tokens=tokens,
                scene_logits=scene_logits,
                part_logits=part_logits,
                presence_logits=presence_logits,
                active_tokens=active_tokens,
                entity_latent_in=entity_in,
                entity_latent=entity_latent,
                frames=frames_pred,
                entity_field_logits=entity_field_logits,
                scene_entity_logits=scene_entity_logits,
                alignment=alignment,
            )

        # Ablation A4: one shared token block and a part-label head.
        pooled = (entity_latent * present.unsqueeze(-1)).sum(dim=1) / present.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        shared = self.shared_projection(pooled).unsqueeze(1)
        tokens = self.tokens(shared, scene_latent, lod=level)
        points = batch.scene_points.unsqueeze(1)
        features = self.field.features(points, tokens, active_tokens=active_tokens)
        scene_logits = self.field.head(features).squeeze(-1).squeeze(1)
        assert self.shared_part_head is not None
        part_raw = self.shared_part_head(features.squeeze(1))
        part_logits = self._expand_part_logits(part_raw, structure.entity_count)
        part_logits[..., : structure.entity_count] = part_logits[
            ..., : structure.entity_count
        ].masked_fill(~present.unsqueeze(1), -1.0e9)
        return ModelOutput(
            scene_latent=scene_latent,
            geometry_tokens=tokens,
            scene_logits=scene_logits,
            part_logits=part_logits,
            presence_logits=presence_logits,
            active_tokens=active_tokens,
            entity_latent_in=entity_in,
            entity_latent=entity_latent,
            frames=frames_pred,
            alignment=alignment,
        )

    @staticmethod
    def _scatter_slots(
        gathered: torch.Tensor, slots: torch.Tensor, entity_count: int, *, fill: float
    ) -> torch.Tensor:
        """Place gathered per-entity results back on the padded entity axis."""
        batch, _, points = gathered.shape
        full = gathered.new_full((batch, entity_count, points), fill)
        return full.index_copy(1, slots, gathered)

    def _lod_one_hot(self, lod: int, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        """One-hot level of detail, the only conditioning the geometry path receives."""
        encoding = reference.new_zeros((batch_size, self.config.lod_levels))
        encoding[:, min(max(lod, 0), self.config.lod_levels - 1)] = 1.0
        return encoding

    @property
    def geometry_is_state_invariant(self) -> bool:
        """Whether presentation state can influence geometry at all.

        ``True`` for the default configuration: the geometry path never sees
        visibility, opacity or the requested entity set, so a state edit cannot move a
        single token. Tested directly in ``tests/test_prototype_editing.py``.
        """
        return self.config.scene_conditioning == "lod_only"

    def _expand_part_logits(self, raw: torch.Tensor, entity_count: int) -> torch.Tensor:
        """Pad a codebook-sized part prediction out to the padded entity axis."""
        batch, points, classes = raw.shape
        vocabulary = classes - 1
        padded = raw.new_full((batch, points, entity_count + 1), -1.0e9)
        padded[..., :vocabulary] = raw[..., :vocabulary]
        padded[..., entity_count] = raw[..., vocabulary]
        return padded
