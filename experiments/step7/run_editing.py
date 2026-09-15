"""Step 7 Experiments 8 and 9: persistent local geometric editing.

Three arms, declared in `docs/step7/02_preregistration_amendments.md` before running:

``E-LOCAL``
    The edit head may rewrite only the edited entity's geometry token block.
``E-FREE``
    The same head, same capacity, may rewrite every entity's block.
``E-REGEN``
    No edit head. The whole scene is re-decoded from the edited parameters. This is the
    control that decides the architecture question: if regenerating everything is as
    local as editing one entity, per-entity token blocks are not buying local editing.

The backbone is frozen in all three arms, so the comparison is between edit mechanisms
and not between differently trained generators. Every arm is scored at the same points,
including ``E-REGEN``, whose regenerated scene is evaluated at the before-scene's points
so that a difference cannot be an artefact of resampling.

    python -m experiments.step7.run_editing \
        --corpus datasets/processed/whole_organ_1600_v2 \
        --checkpoint experiments/runs/step7-whole-organ/checkpoints/wo-A3-seed0.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.nn import functional as functional_ops

from awr.config import load_domain_config
from awr.ontology import load_ontology
from awr.paths import repo_root
from datasets.whole_organ.corpus import load_manifest, load_split
from datasets.whole_organ.sampling import EDIT_SPECIFICATIONS, EditOperation
from experiments.step7.editing import (
    OPERATIONS,
    EditBatch,
    EditCase,
    build_edit_batch,
    build_edit_case,
)
from generation.neural.nn.device import resolve_device, seed_everything
from generation.neural.nn.editing_head import EditHeadConfig, LocalEditHead
from generation.neural.nn.model import LagnavPrototype
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.manifest import environment_report, git_commit

__all__ = ["EditingConfig", "EditingRunner", "run_editing", "main"]

#: Reported metrics. Fixed before the runs.
REPORTED: tuple[str, ...] = (
    "edit_target_accuracy",
    "local_edit_locality",
    "unrelated_entity_drift",
    "target_field_change",
    "identity_preservation",
    "edit_persistence",
    "sequence_consistency",
    "commutativity_gap",
    "cases",
)


@dataclass(frozen=True, slots=True)
class EditingConfig:
    """Settings for the editing experiment. Identical across arms except ``scope``."""

    arm: str
    seed: int = 0
    steps: int = 400
    batch_size: int = 6
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    device: str = "cpu"
    change_threshold: float = 0.08
    hidden: int = 128

    def to_dict(self) -> dict[str, Any]:
        """Plain data, for the report."""
        return {
            "arm": self.arm,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "change_threshold": self.change_threshold,
            "hidden": self.hidden,
        }


@dataclass(frozen=True, slots=True)
class _EditPrediction:
    """What one arm produces for one edit batch, in a form every arm shares."""

    #: ``[B, P, N + 1]`` composed part logits before the edit.
    before_parts: torch.Tensor
    #: ``[B, P, N + 1]`` composed part logits after the edit.
    after_parts: torch.Tensor
    before_latent: torch.Tensor
    after_latent: torch.Tensor
    present: torch.Tensor


def _other_operation(case: EditCase) -> EditOperation:
    """An operation that edits a different entity, for the two-edit sequence.

    The target is declared in the specification, so this reads it there rather than
    building an edit case to find out. Building one evaluates two whole organs.
    """
    for operation in OPERATIONS:
        if operation is case.operation:
            continue
        if EDIT_SPECIFICATIONS[operation][2] != case.target:
            return operation
    raise ValueError(f"No operation edits an entity other than {case.target}.")


def _background_target(ownership: torch.Tensor, entities: int) -> torch.Tensor:
    """Map unowned points (-1) onto the background class index."""
    return torch.where(ownership < 0, torch.full_like(ownership, entities), ownership)


class EditingRunner:
    """Trains and evaluates one edit mechanism against a frozen backbone."""

    def __init__(
        self,
        config: EditingConfig,
        model: LagnavPrototype,
        builder: WholeOrganBatchBuilder,
        train_cases: Sequence[EditCase],
        eval_cases: Sequence[EditCase],
    ) -> None:
        self.config = config
        self.device_choice = resolve_device(config.device)
        self.device = self.device_choice.device
        seed_everything(config.seed)
        self.builder = builder
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.train_cases = list(train_cases)
        self.eval_cases = list(eval_cases)
        self.head: LocalEditHead | None = None
        if config.arm != "E-REGEN":
            geometry = self.model.config.geometry
            self.head = LocalEditHead(
                EditHeadConfig(
                    entity_width=self.model.config.entity_width,
                    token_width=geometry.token_width,
                    tokens=geometry.tokens,
                    operations=len(OPERATIONS),
                    identity_width=self.model.config.identity_width,
                    hidden=config.hidden,
                    scope="target" if config.arm == "E-LOCAL" else "free",
                )
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.head.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    def _stage(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Run the frozen backbone up to the geometry tokens.

        Delegates to the model's own staging so that an edit is applied to exactly the
        latents and tokens an ordinary forward pass would have produced.
        """
        staged = self.model.stage(batch)
        if staged.tokens is None:
            raise ValueError(
                "The editing experiment needs per-entity geometry token blocks; the "
                "shared-token ablation has nothing local to edit."
            )
        return staged.entity_latent, staged.scene_latent, staged.tokens, staged.active_tokens

    def _render(
        self, batch: Any, tokens: torch.Tensor, active: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode tokens into scene logits, part logits and per-entity fields."""
        scene_logits, part_logits, _, scene_entity_logits = self.model.decode_tokens(
            batch,
            tokens,
            frames=batch.entity_frames,
            present=batch.entity_present,
            active_tokens=active,
        )
        return scene_logits, part_logits, scene_entity_logits

    def predict_after(self, edits: EditBatch) -> _EditPrediction:
        """Render the scene before and after the edit, at one shared point set.

        Every arm returns the same quantities from the same decoder, so the arms differ
        only in how the edit is applied.
        """
        before_batch = edits.before.batch.to(self.device)
        staged = self.model.stage(before_batch)
        if staged.tokens is None:
            raise ValueError(
                "The editing experiment needs per-entity geometry token blocks; the "
                "shared-token ablation has nothing local to edit."
            )
        _, before_parts, _ = self._render(before_batch, staged.tokens, staged.active_tokens)

        if self.head is None:
            # E-REGEN: re-decode the edited scene, but at the before-scene's points so
            # that the comparison is between two fields over one point set.
            after_batch = edits.after.batch.to(self.device)
            regenerated = dataclass_replace(
                after_batch,
                scene_points=before_batch.scene_points,
                entity_points=before_batch.entity_points,
            )
            after_staged = self.model.stage(regenerated)
            assert after_staged.tokens is not None
            _, after_parts, _ = self._render(
                regenerated, after_staged.tokens, after_staged.active_tokens
            )
            return _EditPrediction(
                before_parts=before_parts,
                after_parts=after_parts,
                before_latent=staged.entity_latent,
                after_latent=after_staged.entity_latent,
                present=before_batch.entity_present,
            )

        edited = self.head(
            staged.tokens,
            staged.entity_latent,
            edits.edit.to(self.device),
            edits.target_mask.to(self.device),
        )
        _, after_parts, _ = self._render(before_batch, edited, staged.active_tokens)
        return _EditPrediction(
            before_parts=before_parts,
            after_parts=after_parts,
            before_latent=staged.entity_latent,
            # The edit head writes geometry token blocks only. The entity latent, and
            # with it the write-protected identity subspace, is carried through
            # unchanged. Returned rather than assumed so the metric can check it.
            after_latent=staged.entity_latent,
            present=before_batch.entity_present,
        )

    # ------------------------------------------------------------------
    def train(self, batches: Sequence[EditBatch]) -> None:
        """Fit the edit head. ``E-REGEN`` has nothing to fit."""
        if self.head is None:
            return
        self.head.train()
        stream = list(batches)
        if not stream:
            raise ValueError("No edit batches to train on.")
        generator = torch.Generator().manual_seed(self.config.seed)
        for step in range(self.config.steps):
            pick = int(torch.randint(len(stream), (1,), generator=generator))
            edits = stream[pick]
            before_batch = edits.before.batch.to(self.device)
            entity_latent, _, tokens, active = self._stage(before_batch)
            edited = self.head(
                tokens,
                entity_latent,
                edits.edit.to(self.device),
                edits.target_mask.to(self.device),
            )
            scene_logits, part_logits, _ = self._render(before_batch, edited, active)

            entities = int(before_batch.entity_present.shape[1])
            ownership = _background_target(edits.after_ownership.to(self.device), entities)
            occupancy = functional_ops.binary_cross_entropy_with_logits(
                scene_logits, edits.after_occupancy.to(self.device)
            )
            correspondence = functional_ops.cross_entropy(
                part_logits.reshape(-1, part_logits.shape[-1]), ownership.reshape(-1)
            )
            total = occupancy + correspondence
            self.optimizer.zero_grad(set_to_none=True)
            cast(Any, total).backward()
            norm = torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.optimizer.step()
            if step % 50 == 0 or step == self.config.steps - 1:
                self.history.append(
                    {
                        "step": float(step),
                        "total": float(total.detach()),
                        "occupancy": float(occupancy.detach()),
                        "correspondence": float(correspondence.detach()),
                        "grad_norm": float(norm),
                    }
                )
        self.head.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, batches: Sequence[EditBatch]) -> dict[str, float]:
        """Measure the pre-registered editing metrics.

        Occupancy change is measured on the **composed** scene, that is on each entity's
        probability of owning a point, not on its independent field. The distinction
        matters: in the target-scoped arm the untouched entities' independent fields are
        identically equal before and after by construction, which would make the
        locality ratio infinite for structural reasons and comparable across no arms at
        all. Composition is also what the plan's wording, "occupancy change", refers to,
        and it is the quantity a user would see.
        """
        target_ious: list[float] = []
        inside: list[float] = []
        outside: list[float] = []
        identity: list[float] = []
        count = 0
        for edits in batches:
            prediction = self.predict_after(edits)
            before = torch.softmax(prediction.before_parts, dim=-1)
            after = torch.softmax(prediction.after_parts, dim=-1)
            change = (after - before).abs()

            target_mask = edits.target_mask.to(self.device).bool()
            unchanged = edits.unchanged_mask.to(self.device).bool()
            entities = int(edits.before.batch.entity_present.shape[1])
            truth = _background_target(edits.after_ownership.to(self.device), entities)
            predicted = prediction.after_parts.argmax(dim=-1)

            for index in range(edits.batch_size):
                slot = int(target_mask[index].nonzero()[0])
                truth_owned = truth[index] == slot
                guess_owned = predicted[index] == slot
                union = int((truth_owned | guess_owned).sum())
                if union:
                    target_ious.append(float((truth_owned & guess_owned).sum()) / float(union))
                inside.append(float(change[index, :, slot].mean()))
                others = unchanged[index].nonzero().flatten()
                if others.numel():
                    outside.append(float(change[index][:, others].mean()))
                count += 1

            # Identity preservation is a real before/after comparison, restricted to the
            # entities actually in the scene. The edit head leaves the entity latent
            # alone, so it must read exactly 1.0; regeneration re-derives the latent from
            # a changed structure, so it need not. That contrast is the point.
            width = self.model.config.identity_width
            first = prediction.before_latent[..., :width]
            second = prediction.after_latent[..., :width]
            similarity = functional_ops.cosine_similarity(first, second, dim=-1)
            present = prediction.present
            if bool(present.any()):
                identity.append(float(similarity[present].mean()))

        inside_mean = float(np.mean(inside)) if inside else 0.0
        outside_mean = float(np.mean(outside)) if outside else 0.0
        return {
            "edit_target_accuracy": float(np.mean(target_ious)) if target_ious else 0.0,
            "local_edit_locality": (
                inside_mean / outside_mean if outside_mean > 1.0e-12 else float("nan")
            ),
            "unrelated_entity_drift": outside_mean,
            "target_field_change": inside_mean,
            "identity_preservation": float(np.mean(identity)) if identity else 0.0,
            "cases": float(count),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_sequence(self, cases: Sequence[EditCase]) -> dict[str, float]:
        """Experiment 9: edit, read a relation, edit again, check the first edit held."""
        from experiments.step7.relation_baseline import AXIS_OF

        if self.head is None:
            return self._sequence_regen(cases)
        persistence: list[float] = []
        consistency: list[float] = []
        commutativity: list[float] = []
        for case in cases:
            first = build_edit_batch([case], self.builder)
            before_batch = first.before.batch.to(self.device)
            entity_latent, _, tokens, active = self._stage(before_batch)
            slot_of = self.builder.slot_of

            once = self.head(
                tokens,
                entity_latent,
                first.edit.to(self.device),
                first.target_mask.to(self.device),
            )
            _, part_once, _ = self._render(before_batch, once, active)

            # A second edit on a different entity, applied on top of the first.
            second_operation = _other_operation(case)
            second_case = build_edit_case(case.scene, second_operation)
            second = build_edit_batch([second_case], self.builder)
            twice = self.head(
                once,
                entity_latent,
                second.edit.to(self.device),
                second.target_mask.to(self.device),
            )
            _, part_twice, _ = self._render(before_batch, twice, active)

            # Persistence is judged on what the scene says the first target owns, so a
            # second edit that quietly takes the region back counts as forgetting.
            slot = slot_of[case.target]
            owned_once = part_once.argmax(dim=-1)[0] == slot
            owned_twice = part_twice.argmax(dim=-1)[0] == slot
            union = int((owned_once | owned_twice).sum())
            persistence.append(
                float((owned_once & owned_twice).sum()) / float(union) if union else 1.0
            )

            # Read a relation off the geometry produced after the first edit and compare
            # it with the same relation measured from the generator's true edited organ.
            predicted = part_once.argmax(dim=-1)[0]
            points = before_batch.scene_points[0]
            centroid: dict[str, torch.Tensor] = {}
            for name, index in slot_of.items():
                owned = predicted == index
                if int(owned.sum()) >= 3:
                    centroid[name] = points[owned].mean(dim=0)
            truth_edges = case.after_scene.edges
            agreed = 0
            total = 0
            for edge in truth_edges:
                if edge.relation not in AXIS_OF:
                    continue
                left = centroid.get(edge.subject)
                right = centroid.get(edge.object)
                if left is None or right is None:
                    continue
                axis, sign = AXIS_OF[edge.relation]
                total += 1
                agreed += int(sign * float(left[axis] - right[axis]) > 0.0)
            if total:
                consistency.append(agreed / total)

            # Order should not matter when the two edits touch disjoint entities.
            other_first = self.head(
                tokens,
                entity_latent,
                second.edit.to(self.device),
                second.target_mask.to(self.device),
            )
            reversed_tokens = self.head(
                other_first,
                entity_latent,
                first.edit.to(self.device),
                first.target_mask.to(self.device),
            )
            _, part_reversed, _ = self._render(before_batch, reversed_tokens, active)
            commutativity.append(
                float(
                    (torch.softmax(part_twice, dim=-1) - torch.softmax(part_reversed, dim=-1))
                    .abs()
                    .mean()
                )
            )
        return {
            "edit_persistence": float(np.mean(persistence)) if persistence else 0.0,
            "sequence_consistency": float(np.mean(consistency)) if consistency else 0.0,
            "commutativity_gap": float(np.mean(commutativity)) if commutativity else 0.0,
        }

    @torch.no_grad()
    def _sequence_regen(self, cases: Sequence[EditCase]) -> dict[str, float]:
        """The regeneration control's version of the loop: both edits, then re-decode.

        Regeneration has no state to forget, so persistence is measured against the
        generator's own composition of the two edits rather than against a stored edit.
        """
        from experiments.step7.editing import edited_parameters, rebuild_scene
        from experiments.step7.relation_baseline import AXIS_OF

        persistence: list[float] = []
        consistency: list[float] = []
        commutativity: list[float] = []
        for case in cases:
            second_operation = _other_operation(case)
            first_parameters = edited_parameters(case.scene.parameters, case.operation)
            both = edited_parameters(first_parameters, second_operation)
            other = edited_parameters(
                edited_parameters(case.scene.parameters, second_operation), case.operation
            )
            once_scene = rebuild_scene(case.scene, first_parameters)
            twice_scene = rebuild_scene(case.scene, both)
            reversed_scene = rebuild_scene(case.scene, other)

            base = build_edit_batch([case], self.builder).before.batch.to(self.device)
            parts: list[torch.Tensor] = []
            for scene in (once_scene, twice_scene, reversed_scene):
                batch = self.builder.build([scene]).batch.to(self.device)
                shared = dataclass_replace(
                    batch,
                    scene_points=base.scene_points,
                    entity_points=base.entity_points,
                )
                _, _, tokens, active = self._stage(shared)
                _, part_logits, _ = self._render(shared, tokens, active)
                parts.append(part_logits)

            slot = self.builder.slot_of[case.target]
            owned_once = parts[0].argmax(dim=-1)[0] == slot
            owned_twice = parts[1].argmax(dim=-1)[0] == slot
            union = int((owned_once | owned_twice).sum())
            persistence.append(
                float((owned_once & owned_twice).sum()) / float(union) if union else 1.0
            )

            predicted = parts[0].argmax(dim=-1)[0]
            points = base.scene_points[0]
            centroid: dict[str, torch.Tensor] = {}
            for name, index in self.builder.slot_of.items():
                owned = predicted == index
                if int(owned.sum()) >= 3:
                    centroid[name] = points[owned].mean(dim=0)
            agreed = 0
            total = 0
            for edge in once_scene.edges:
                if edge.relation not in AXIS_OF:
                    continue
                left = centroid.get(edge.subject)
                right = centroid.get(edge.object)
                if left is None or right is None:
                    continue
                axis, sign = AXIS_OF[edge.relation]
                total += 1
                agreed += int(sign * float(left[axis] - right[axis]) > 0.0)
            if total:
                consistency.append(agreed / total)
            commutativity.append(
                float(
                    (torch.softmax(parts[1], dim=-1) - torch.softmax(parts[2], dim=-1)).abs().mean()
                )
            )
        return {
            "edit_persistence": float(np.mean(persistence)) if persistence else 0.0,
            "sequence_consistency": float(np.mean(consistency)) if consistency else 0.0,
            "commutativity_gap": float(np.mean(commutativity)) if commutativity else 0.0,
        }


def _cases_for(
    scenes: Sequence[Any], operations: Sequence[EditOperation], *, threshold: float
) -> dict[EditOperation, list[EditCase]]:
    """Build edit cases grouped by operation, since a batch shares one operation."""
    grouped: dict[EditOperation, list[EditCase]] = {}
    dropped: dict[str, int] = {}
    for operation in operations:
        for scene in scenes:
            case = build_edit_case(scene, operation, threshold=threshold)
            if not case.is_measurable:
                # The edit removed a structure outright. Locality compares a before and
                # an after over one entity set, so there is nothing to compare here.
                dropped[str(operation)] = dropped.get(str(operation), 0) + 1
                continue
            grouped.setdefault(operation, []).append(case)
    if dropped:
        print(f"[step7-edit] cases dropped for removing an entity: {dropped}", flush=True)
    return grouped


def _batches_for(
    grouped: Mapping[EditOperation, Sequence[EditCase]],
    builder: WholeOrganBatchBuilder,
    *,
    batch_size: int,
) -> list[EditBatch]:
    """Batch cases so that every batch shares one operation and one level of detail."""
    batches: list[EditBatch] = []
    for operation, cases in grouped.items():
        del operation
        by_level: dict[int, list[EditCase]] = {}
        for case in cases:
            by_level.setdefault(case.scene.active_lod, []).append(case)
        for level_cases in by_level.values():
            for start in range(0, len(level_cases), batch_size):
                group = level_cases[start : start + batch_size]
                if group:
                    batches.append(build_edit_batch(group, builder))
    return batches


def run_editing(
    *,
    corpus_dir: Path,
    checkpoint: Path,
    arms: Sequence[str],
    steps: int,
    seed: int,
    batch_size: int,
    train_scenes: int,
    eval_scenes: int,
    sequence_cases: int,
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate every edit mechanism against one frozen backbone."""
    started = time.time()
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    manifest = load_manifest(corpus_dir)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone_arm = str(state["config"]["arm"])

    train = load_split(corpus_dir, "train", limit=train_scenes)
    evaluation = load_split(corpus_dir, "test", limit=eval_scenes)
    train_groups = _cases_for(train, OPERATIONS, threshold=0.08)
    eval_groups = _cases_for(evaluation, OPERATIONS, threshold=0.08)
    train_batches = _batches_for(train_groups, builder, batch_size=batch_size)
    eval_batches = _batches_for(eval_groups, builder, batch_size=batch_size)
    loop_cases = [case for cases in eval_groups.values() for case in cases][:sequence_cases]

    results: dict[str, Any] = {}
    for arm in arms:
        config = EditingConfig(
            arm=arm, seed=seed, steps=steps, batch_size=batch_size, device=device
        )
        model, _ = build_model(
            cast(Any, backbone_arm),
            cast(Any, builder),
            text_features=int(train_batches[0].before.batch.text_features.shape[1]),
        )
        model.load_state_dict(state["model"])
        runner = EditingRunner(config, cast(LagnavPrototype, model), builder, [], [])
        print(f"[step7-edit] {arm}: training", flush=True)
        runner.train(train_batches)
        print(f"[step7-edit] {arm}: evaluating", flush=True)
        values = runner.evaluate(eval_batches)
        values.update(runner.evaluate_sequence(loop_cases))
        head_parameters = (
            dict(runner.head.parameter_groups()) if runner.head is not None else {"total": 0}
        )
        results[arm] = {
            "config": config.to_dict(),
            "head_parameters": head_parameters,
            "metrics": values,
            "train_history": list(runner.history),
        }
        summary = {key: round(float(values[key]), 4) for key in REPORTED if key in values}
        print(f"[step7-edit]   {arm}: {summary}", flush=True)

    report = {
        "experiment_id": "step7-editing",
        "commit": git_commit(repo_root()),
        "environment": environment_report(),
        "backbone": {
            "checkpoint": str(checkpoint),
            "arm": backbone_arm,
            "frozen": True,
        },
        "dataset": {
            "corpus_id": manifest.corpus_id,
            "corpus_dir": str(corpus_dir),
            "train_scenes": len(train),
            "eval_scenes": len(evaluation),
            "train_batches": len(train_batches),
            "eval_batches": len(eval_batches),
            "sequence_cases": len(loop_cases),
            "operations": [str(operation) for operation in OPERATIONS],
            "data_label": manifest.data_label,
        },
        "arms": results,
        "runtime_seconds": round(time.time() - started, 1),
        "notes": [
            "The backbone is frozen; only the edit mechanism differs between arms.",
            "Before and after are evaluated at the same points, E-REGEN included.",
            "Entities the generator's own measurement says the edit did not alter are "
            "the ones leakage is scored against.",
            "Anatomical structures are coupled, so a targeted edit has genuine "
            "non-local consequences. Those entities are excluded from the leakage "
            "denominator rather than counted as errors.",
            "Synthetic research data. Not anatomy, not validated, not clinical.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "editing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Step 7 editing experiments.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["E-LOCAL", "E-FREE", "E-REGEN"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--train-scenes", type=int, default=48)
    parser.add_argument("--eval-scenes", type=int, default=24)
    parser.add_argument("--sequence-cases", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("experiments/runs/step7-editing"))
    args = parser.parse_args(argv)

    report = run_editing(
        corpus_dir=args.corpus,
        checkpoint=args.checkpoint,
        arms=args.arms,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        train_scenes=args.train_scenes,
        eval_scenes=args.eval_scenes,
        sequence_cases=args.sequence_cases,
        device=args.device,
        output_dir=args.out,
    )
    print(
        json.dumps({arm: values["metrics"] for arm, values in report["arms"].items()}, indent=2)[
            :2500
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
