"""Step 13: the two treatments do what they claim, and the control is untouched.

Step 13 asks whether the relational gap comes from too few propagation hops (H2) or from no
global graph context (H6). Both treatments are small, and both could fail silently in ways that
would produce a clean-looking negative result for entirely the wrong reason:

* recurrence could be wired so the second pass never runs, making H2 a re-run of the control;
* the alignment that isolates H6's initialisation could quietly copy nothing, restoring the
  Step 12 confound where the arms differ by seed as well as by architecture;
* either change could alter the control's own initialisation, which would break the frozen
  references the whole comparison is read against.

Each of those is asserted here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from awr.config import load_domain_config
from awr.ontology import load_ontology
from datasets.whole_organ.arrangement import sample_arrangement
from datasets.whole_organ.corpus import build_scene
from generation.neural.nn.graph import GraphEncoderConfig
from generation.neural.nn.init_alignment import align_shared_initialisation
from generation.neural.nn.whole_organ import WholeOrganBatchBuilder
from training.loop import build_model
from training.step10 import default_config, with_variant

#: The frame-head tensors H6 owns. Anything else differing means the isolation failed.
H6_INTENDED = {
    "frame_head.context_norm.bias",
    "frame_head.context_norm.weight",
    "frame_head.trunk.0.weight",
}


@pytest.fixture(scope="module")
def setup():
    """A real batch and the builder that made it."""
    domain = load_domain_config()
    ontology = load_ontology(
        domain.domain.ontology_dir, expected_version=domain.domain.ontology_version
    )
    builder = WholeOrganBatchBuilder(ontology, domain)
    rng = np.random.default_rng(0)
    scenes = [
        build_scene(scene_index=i, family_id=i % 3, lod=3, arrangement=sample_arrangement(rng))
        for i in range(4)
    ]
    return builder.build(scenes).batch, builder


def _model(builder, *, seed: int = 0, **overrides):
    """One A3 model, built the way the runner builds it."""
    torch.manual_seed(seed)
    model, groups = build_model(
        "A3",
        builder,
        text_features=64,
        overrides={"frame_scene_context": False, **overrides},
    )
    return model, groups


class TestTheControlIsUntouched:
    """Step 13's code must be inert when switched off, or the frozen references move."""

    def test_the_step_10_control_config_selects_neither_treatment(self) -> None:
        """A3 as Steps 10-12 ran it: four hops, no global pooling."""
        config = with_variant(
            default_config("A3"), placement_target="global", hierarchy="spatial", seed=0
        )
        assert config.graph_recurrence == 1
        assert config.frame_scene_context is False

    def test_recurrence_defaults_to_one(self) -> None:
        """Every result up to Step 12 was measured at a single pass."""
        assert GraphEncoderConfig().recurrence == 1

    def test_a_single_pass_is_bit_identical_to_the_old_path(self, setup) -> None:
        """Explicit recurrence=1 must equal the default, not merely resemble it."""
        batch, builder = setup
        plain, _ = _model(builder)
        explicit, _ = _model(builder, graph_recurrence=1)
        plain.eval()
        explicit.eval()
        with torch.no_grad():
            assert torch.equal(
                plain.stage(batch).entity_latent, explicit.stage(batch).entity_latent
            )


class TestTheDepthTreatment:
    """H2 must actually propagate further, at no capacity cost."""

    def test_propagation_steps_are_layers_times_recurrence(self, setup) -> None:
        """The quantity H2 varies, reported rather than inferred."""
        _, builder = setup
        control, _ = _model(builder)
        deeper, _ = _model(builder, graph_recurrence=2)
        assert control.graph_encoder.propagation_steps == 4
        assert deeper.graph_encoder.propagation_steps == 8

    def test_depth_adds_no_parameters(self, setup) -> None:
        """Weight tying is the whole reason this arm is interpretable."""
        _, builder = setup
        control, _ = _model(builder)
        deeper, _ = _model(builder, graph_recurrence=2)
        assert sum(p.numel() for p in deeper.parameters()) == sum(
            p.numel() for p in control.parameters()
        )

    def test_depth_changes_the_latent(self, setup) -> None:
        """A second pass that changed nothing would make H2 a re-run of the control."""
        batch, builder = setup
        control, _ = _model(builder)
        deeper, _ = _model(builder, graph_recurrence=2)
        control.eval()
        deeper.eval()
        with torch.no_grad():
            assert not torch.allclose(
                control.stage(batch).entity_latent, deeper.stage(batch).entity_latent
            )

    def test_depth_starts_from_identical_weights(self, setup) -> None:
        """No new parameters means no RNG consumed, so isolation should be free."""
        _, builder = setup
        control, _ = _model(builder)
        deeper, _ = _model(builder, graph_recurrence=2)
        base, other = control.state_dict(), deeper.state_dict()
        assert set(base) == set(other)
        assert [k for k in base if not torch.equal(base[k], other[k])] == []

    def test_the_identity_slice_survives_the_second_pass(self, setup) -> None:
        """Protected identity is protected however many times the stack runs."""
        batch, builder = setup
        deeper, _ = _model(builder, graph_recurrence=2)
        encoder = deeper.graph_encoder
        latent = torch.randn(batch.batch_size, batch.structure.entity_count, encoder.config.width)
        with torch.no_grad():
            out = encoder(latent, batch.structure)
        width = encoder.config.identity_width
        assert torch.equal(out[..., :width], latent[..., :width])


class TestTheGlobalContextTreatment:
    """H6 must reach the frame head, and cost only what it claims."""

    def test_enabling_scene_context_changes_only_the_frame_head(self, setup) -> None:
        """The parameter delta must be where the plan says it is."""
        _, builder = setup
        _, control = _model(builder)
        _, treated = _model(builder, frame_scene_context=True)
        assert treated["frame_head"] > control["frame_head"]
        for group in ("graph_encoder", "entity_composer", "geometry_tokens", "field_decoder"):
            assert treated[group] == control[group], f"{group} changed and should not have"

    def test_the_delta_is_the_declared_size(self, setup) -> None:
        """+66,048: a widened first trunk layer plus one LayerNorm."""
        _, builder = setup
        _, control = _model(builder)
        _, treated = _model(builder, frame_scene_context=True)
        assert treated["total"] - control["total"] == 66048

    def test_the_context_is_a_presence_masked_mean(self, setup) -> None:
        """Padded slots must not dilute the summary, or it depends on batch packing."""
        batch, builder = setup
        model, _ = _model(builder, frame_scene_context=True)
        model.eval()
        with torch.no_grad():
            latent = model.stage(batch).entity_latent
            present = batch.entity_present.bool()
            summary = model.scene_summary(latent, present)
            manual = torch.stack(
                [latent[i][present[i]].mean(dim=0) for i in range(latent.shape[0])]
            )
        assert torch.allclose(summary, manual, atol=1e-5)

    def test_the_context_actually_reaches_the_prediction(self, setup) -> None:
        """Perturbing the summary must move the frames, or H6 is untested by this arm.

        The head's output projection is zero-initialised on purpose, so an untrained model
        emits the canonical frame whatever it reads. Testing the pathway at initialisation
        would therefore pass trivially for a disconnected context and fail for a connected
        one, so the projection is given weights first, as it has after any training step.
        """
        batch, builder = setup
        model, _ = _model(builder, frame_scene_context=True)
        projection = model.frame_head.projection
        with torch.no_grad():
            projection.weight.copy_(
                torch.randn(projection.weight.shape, generator=torch.Generator().manual_seed(13))
                * 0.02
            )
        model.eval()
        present = batch.entity_present.bool()
        with torch.no_grad():
            latent = model.stage(batch).entity_latent
            baseline = model.predict_frames(latent, present)
            original = model.scene_summary
            model.scene_summary = lambda entity_latent, p: original(entity_latent, p) + 1.0  # type: ignore[method-assign]
            shifted = model.predict_frames(latent, present)
            model.scene_summary = original  # type: ignore[method-assign]
        assert not torch.allclose(baseline, shifted)

    def test_a_missing_context_fails_loudly(self, setup) -> None:
        """Silently dropping the summary would make H6 look tested when it was not."""
        _, builder = setup
        model, _ = _model(builder, frame_scene_context=True)
        latent = torch.randn(2, 5, model.config.entity_width)
        with pytest.raises(ValueError, match="needs a"):
            model.frame_head(latent, None)


class TestTheInitialisationAlignment:
    """The isolation H6 depends on, since fork_rng cannot reshape a weight."""

    def test_alignment_is_needed_at_all(self, setup) -> None:
        """Documents the confound: without alignment, unrelated modules differ."""
        _, builder = setup
        control, _ = _model(builder)
        treated, _ = _model(builder, frame_scene_context=True)
        base, other = control.state_dict(), treated.state_dict()
        shared = [k for k in base if k in other and base[k].shape == other[k].shape]
        differing = [k for k in shared if not torch.equal(base[k], other[k])]
        assert len(differing) > 20, "expected the documented RNG shift; the confound is gone?"
        assert any(k.startswith("tokens.") for k in differing)

    def test_alignment_removes_every_incidental_difference(self, setup) -> None:
        """After alignment only the treatment's own tensors may differ."""
        _, builder = setup
        control, _ = _model(builder)
        treated, _ = _model(builder, frame_scene_context=True)
        report = align_shared_initialisation(treated, control)
        base, other = control.state_dict(), treated.state_dict()
        shared = [k for k in base if k in other and base[k].shape == other[k].shape]
        assert [k for k in shared if not torch.equal(base[k], other[k])] == []
        assert report.isolated_to(H6_INTENDED)

    def test_depth_needs_no_alignment_but_passes_it(self, setup) -> None:
        """H2 has nothing to isolate, and the check must say so rather than invent work."""
        _, builder = setup
        control, _ = _model(builder)
        deeper, _ = _model(builder, graph_recurrence=2)
        report = align_shared_initialisation(deeper, control)
        assert report.intended_differences == ()
        assert report.isolated_to(set())

    def test_the_reference_model_is_not_modified(self, setup) -> None:
        """The control's initialisation must survive, or the frozen references move."""
        _, builder = setup
        control, _ = _model(builder)
        before = {k: v.clone() for k, v in control.state_dict().items()}
        treated, _ = _model(builder, frame_scene_context=True)
        align_shared_initialisation(treated, control)
        after = control.state_dict()
        assert [k for k in before if not torch.equal(before[k], after[k])] == []

    def test_alignment_reports_what_it_did(self, setup) -> None:
        """The run manifest carries this, so it must be populated and plain data."""
        _, builder = setup
        control, _ = _model(builder)
        treated, _ = _model(builder, frame_scene_context=True)
        payload = align_shared_initialisation(treated, control).to_dict()
        assert payload["tensors_copied_from_control"] == 140
        assert set(payload["intended_differences"]) == H6_INTENDED


class TestTheStep13Harnesses:
    """The scoring code decides the verdict, so its arithmetic is pinned here.

    Three arms are compared against a 1.0 degree rule, and the arms' checkpoints live in
    different directories under different names. A wrong stem would load the wrong weights and
    still produce a plausible table, so the table itself is asserted.
    """

    def test_the_arm_table_matches_the_plan(self) -> None:
        """Recurrence and scene context per arm, exactly as preregistered."""
        from experiments.step13.evaluate import ARMS

        assert ARMS["A3"][2:] == (1, False)
        assert ARMS["A3-depth"][2:] == (2, False)
        assert ARMS["A3-global"][2:] == (1, True)

    def test_the_global_arm_has_its_own_checkpoint_stem(self) -> None:
        """Step 9's run_id switches noctx to ctx; loading the wrong stem would load the control."""
        from experiments.step13.evaluate import ARMS

        assert "noctx" in ARMS["A3"][1]
        assert "noctx" in ARMS["A3-depth"][1]
        assert "-ctx-" in ARMS["A3-global"][1]

    def test_the_threshold_is_the_seed_spread(self) -> None:
        """1.0 deg, because A3's own spread is 1.03 and anything smaller is indistinguishable."""
        from experiments.step13.evaluate import MINIMUM_MEANINGFUL_DEG

        assert MINIMUM_MEANINGFUL_DEG == 1.0

    def test_the_ablation_conditions_are_the_briefs_five(self) -> None:
        """No extra ablation may be invented before the primary result is read."""
        from experiments.step13.ablations import CONDITIONS

        assert CONDITIONS == (
            "intact",
            "entities_only",
            "drop_spatial",
            "drop_structure",
            "drop_functional",
        )

    def test_relative_error_is_one_for_a_mean_predictor(self) -> None:
        """The probe scale's zero point: learning nothing scores 1.0, not 0.0."""
        from experiments.step13.probes import _relative_error

        target = torch.randn(256, 4)
        mean = target.mean(dim=0, keepdim=True)

        class _Mean(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return mean.expand(inputs.shape[0], -1)

        assert _relative_error(_Mean(), torch.randn(256, 8), target) == pytest.approx(1.0, abs=0.02)

    def test_relative_error_is_zero_for_a_perfect_probe(self) -> None:
        """The other end of the same scale."""
        from experiments.step13.probes import _relative_error

        target = torch.randn(128, 5)
        assert _relative_error(torch.nn.Identity(), target, target) < 1e-6

    def test_every_global_target_is_named(self) -> None:
        """Section 15 lists five things the context is probed for; none may be dropped."""
        from experiments.step13.probes import GLOBAL_TARGETS

        assert set(GLOBAL_TARGETS) == {
            "relationship_configuration",
            "graph_arrangement",
            "scene_transformation",
            "entity_placement",
            "entity_rotation",
        }


class TestNeitherTreatmentLeaks:
    """A treatment that read the answer would improve for a reason that is not its hypothesis.

    H6's global context is a mean of entity latents, and the docstring asserts it "cannot leak
    placement". That is the kind of claim worth testing rather than reading: if the summary
    depended on the true frames, an H6 gain would be leakage wearing a mechanism's name.
    """

    @staticmethod
    def _corrupt_frames(batch):
        """The same batch with the supervision target replaced by noise."""
        from dataclasses import replace as dc_replace

        poisoned = torch.randn_like(batch.entity_frames) * 5.0
        return dc_replace(batch, entity_frames=poisoned)

    @pytest.mark.parametrize(
        "overrides",
        ({"graph_recurrence": 2}, {"frame_scene_context": True}),
        ids=("depth", "global"),
    )
    def test_the_prediction_ignores_the_true_frames(self, setup, overrides) -> None:
        """Both treatments must predict identically when the target is replaced by noise."""
        batch, builder = setup
        model, _ = _model(builder, **overrides)
        projection = model.frame_head.projection
        with torch.no_grad():
            projection.weight.copy_(
                torch.randn(projection.weight.shape, generator=torch.Generator().manual_seed(13))
                * 0.02
            )
        model.eval()
        present = batch.entity_present.bool()
        with torch.no_grad():
            clean = model.predict_frames(model.stage(batch).entity_latent, present)
            other = self._corrupt_frames(batch)
            poisoned = model.predict_frames(model.stage(other).entity_latent, present)
        assert torch.equal(clean, poisoned)

    def test_the_scene_summary_ignores_the_true_frames(self, setup) -> None:
        """The H6 pathway specifically, at the point where a leak would enter."""
        batch, builder = setup
        model, _ = _model(builder, frame_scene_context=True)
        model.eval()
        present = batch.entity_present.bool()
        with torch.no_grad():
            clean = model.scene_summary(model.stage(batch).entity_latent, present)
            other = self._corrupt_frames(batch)
            poisoned = model.scene_summary(model.stage(other).entity_latent, present)
        assert torch.equal(clean, poisoned)

    def test_the_scene_summary_ignores_padded_slots(self, setup) -> None:
        """A summary that counted padding would vary with batch packing, not with the scene."""
        batch, builder = setup
        model, _ = _model(builder, frame_scene_context=True)
        model.eval()
        with torch.no_grad():
            latent = model.stage(batch).entity_latent
            present = batch.entity_present.bool()
            baseline = model.scene_summary(latent, present)
            noisy = latent.clone()
            noisy[~present] = 99.0
            assert torch.allclose(baseline, model.scene_summary(noisy, present), atol=1e-5)
