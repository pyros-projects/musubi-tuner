"""Tests for CREPA cross-frame representation alignment on MiniMax H3."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from musubi_tuner.minimax_h3.crepa import CREPAConfig, CREPAModule, parse_crepa_args  # noqa: E402
from musubi_tuner.minimax_h3.model import MiniMaxH3Model  # noqa: E402


def _tiny_model() -> MiniMaxH3Model:
    model = MiniMaxH3Model(
        hidden_size=12,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=12,
        time_embed_dim=6,
        rope_inv_freq_len=1,
    )
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
    return model


def _crepa(config: CREPAConfig | None = None, dim: int = 12) -> CREPAModule:
    module = CREPAModule(config or CREPAConfig(student_block_idx=0, teacher_block_idx=1, num_neighbors=1), dim)
    return module


class TestConfig:
    def test_parse_and_defaults(self):
        config = parse_crepa_args(["student_block_idx=3", "lambda_crepa=0.2", "normalize=false"])
        assert config.student_block_idx == 3
        assert config.lambda_crepa == 0.2
        assert config.normalize is False
        assert config.teacher_block_idx == 32

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown"):
            parse_crepa_args(["bogus=1"])

    def test_validate_ordering(self):
        config = CREPAConfig(student_block_idx=5, teacher_block_idx=3)
        with pytest.raises(ValueError, match="teacher_block_idx"):
            config.validate(50)

    def test_schedule_warmup_and_decay(self):
        module = _crepa(CREPAConfig(student_block_idx=0, teacher_block_idx=1, lambda_crepa=0.4, warmup_steps=10, schedule="linear", max_steps=110))
        module.on_step(0)
        assert module._current_lambda == 0.0
        module.on_step(5)
        assert module._current_lambda == pytest.approx(0.2)
        module.on_step(10)
        assert module._current_lambda == pytest.approx(0.4)
        module.on_step(110)
        assert module._current_lambda == pytest.approx(0.0)


class TestLossMath:
    def test_matches_manual_computation(self):
        config = CREPAConfig(student_block_idx=0, teacher_block_idx=1, lambda_crepa=1.0, tau=1.0, num_neighbors=1, normalize=True)
        module = _crepa(config, dim=4)
        # Bypass projector: identity-like check via direct _similarity_loss.
        torch.manual_seed(0)
        proj = torch.randn(1, 3, 4)
        teach = torch.randn(1, 3, 4)
        loss = module._similarity_loss(proj.clone(), teach.clone(), 3)

        p = torch.nn.functional.normalize(proj, dim=-1)[0]
        q = torch.nn.functional.normalize(teach, dim=-1)[0]
        sim = p @ q.T
        w = math.exp(-1.0)
        manual = -(
            sim[0, 0] + sim[1, 1] + sim[2, 2]
            + w * (sim[0, 1] + sim[1, 0] + sim[1, 2] + sim[2, 1])
        )
        num_terms = 3 + 4
        manual = manual / (num_terms / 3)
        torch.testing.assert_close(loss, manual, rtol=1e-5, atol=1e-6)

    def test_perfect_alignment_is_minimal(self):
        config = CREPAConfig(student_block_idx=0, teacher_block_idx=1, lambda_crepa=1.0, num_neighbors=1)
        module = _crepa(config, dim=4)
        frames = torch.randn(1, 4, 4)
        aligned = module._similarity_loss(frames.clone(), frames.clone(), 4)
        shuffled = module._similarity_loss(frames.clone(), torch.randn(1, 4, 4), 4)
        assert aligned < shuffled


class TestCaptureIntegration:
    def _forward_serial(self, model, t_frames: int):
        video = torch.randn(1, 2, t_frames, 4, 4)
        audio = torch.randn(1, 3, 2, 3)
        context = torch.randn(1, 4, 8)
        return model(video, audio, torch.tensor([0.7]), context)

    def test_capture_grad_connected_under_checkpointing(self):
        model = _tiny_model()
        model.enable_gradient_checkpointing()
        model.train()
        model.set_crepa_capture(0, 1)
        self._forward_serial(model, t_frames=2)

        cap = model._crepa_capture
        assert cap["student"] is not None and cap["teacher"] is not None
        # The whole point of loop-level capture: student stays graph-connected
        # even with non-reentrant checkpointing; teacher is detached.
        assert cap["student"].grad_fn is not None
        assert cap["teacher"].grad_fn is None
        assert cap["latent_t"] == 2
        assert cap["video_slice"] is not None

    def test_crepa_loss_backprops_into_model(self):
        model = _tiny_model()
        model.enable_gradient_checkpointing()
        model.train()
        model.set_crepa_capture(0, 1)
        module = _crepa()
        module.on_step(100)
        self._forward_serial(model, t_frames=2)

        loss = module.compute_loss(model._crepa_capture)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        block0_grads = [p.grad for p in model.blocks[0].parameters() if p.grad is not None]
        assert block0_grads, "CREPA gradient must reach block 0 parameters"
        # Teacher path is detached: block 1 must receive no gradient.
        assert all(p.grad is None for p in model.blocks[1].parameters())

    def test_image_batch_skips(self):
        model = _tiny_model()
        model.set_crepa_capture(0, 1)
        module = _crepa()
        module.on_step(100)
        self._forward_serial(model, t_frames=1)
        assert module.compute_loss(model._crepa_capture) is None

    def test_batched_path_capture(self):
        model = _tiny_model()
        model.set_crepa_capture(0, 1)
        video = torch.randn(2, 2, 2, 4, 4)
        audio = torch.randn(2, 3, 2, 3)
        contexts = [torch.randn(4, 8), torch.randn(3, 8)]
        tags = [torch.ones(4, dtype=torch.long), torch.ones(3, dtype=torch.long)]
        model(video, audio, torch.tensor([0.7, 0.5]), contexts, tags)
        cap = model._crepa_capture
        assert cap["student"] is not None and cap["student"].ndim == 3
        assert cap["student"].shape[0] == 2

        module = _crepa()
        module.on_step(100)
        loss = module.compute_loss(cap)
        assert loss is not None and torch.isfinite(loss)

    def test_fused_walk_captures_cond_stream(self):
        model = _tiny_model()
        model.set_crepa_capture(0, 1)
        video = torch.randn(1, 2, 2, 4, 4)
        audio = torch.randn(1, 3, 2, 3)
        context = [torch.randn(4, 8)]
        tags = [torch.ones(4, dtype=torch.long)]
        uncond = torch.randn(2, 8)
        uncond_tags = torch.ones(2, dtype=torch.long)
        model.train()
        model.forward_with_uncond(video, audio, torch.tensor([0.7]), context, tags, uncond, uncond_tags)
        cap = model._crepa_capture
        assert cap["student"] is not None
        # Cond stream requires grad in train mode; the uncond stream never does.
        assert cap["student"].requires_grad

    def test_disarm(self):
        model = _tiny_model()
        model.set_crepa_capture(0, 1)
        model.set_crepa_capture(None)
        assert model._crepa_capture is None
        self._forward_serial(model, t_frames=2)  # must not record or crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
