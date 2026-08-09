import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.video_vae import decode_single_frame_latent, decode_video_latents, select_preview_frame
from musubi_tuner.minimax_h3_train_network import (
    MiniMaxH3NetworkTrainer,
    build_h3_sigma_schedule,
    minimax_h3_setup_parser,
    sample_h3_image_latents,
)


def test_h3_sigma_schedule_is_shifted_and_descending():
    sigmas = build_h3_sigma_schedule(4, 12.0, torch.device("cpu"))

    assert sigmas.shape == (5,)
    assert sigmas[0] == 1
    assert sigmas[-1] == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_h3_sampling_keeps_true_single_frame_latents():
    class FakeTransformer:
        def __init__(self):
            self.sigmas = []

        def __call__(self, video, audio, sigma, context, tags):
            assert video.shape == (1, 24, 1, 4, 6)
            assert audio.shape == (1, 32, 2, 0)
            assert len(context) == len(tags) == 1
            assert context[0].shape == (3, 8)
            assert tags[0].shape == (3,)
            self.sigmas.append(float(sigma))
            return torch.ones_like(video), torch.empty_like(audio)

    transformer = FakeTransformer()
    sampled = sample_h3_image_latents(
        transformer,
        torch.zeros(3, 8),
        torch.ones(3, dtype=torch.long),
        height=64,
        width=96,
        sample_steps=4,
        video_flow_shift=12.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(1),
        latent_frames=1,
        latents=torch.zeros(1, 24, 1, 4, 6),
    )

    assert sampled.shape == (1, 24, 1, 4, 6)
    torch.testing.assert_close(sampled, torch.ones_like(sampled))
    assert len(transformer.sigmas) == 4
    assert transformer.sigmas == sorted(transformer.sigmas, reverse=True)


def test_h3_sample_prompt_loads_automatic_prompt_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "minimax_h3_sample_prompts_cache.pt"
    torch.save(
        {
            "version": 1,
            "architecture": "minimax_h3",
            "prompt_cache": [
                {
                    "prompt": "lucy the cat",
                    "h3_text_embed": torch.arange(24, dtype=torch.float32).reshape(3, 8),
                    "h3_token_tags": torch.tensor([1, 1, 0]),
                }
            ],
        },
        cache_path,
    )
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        f'''[[datasets]]\nimage_directory = "{tmp_path}"\ncache_directory = "{cache_dir}"\n''',
        encoding="utf-8",
    )
    prompt_path = tmp_path / "samples.toml"
    prompt_path.write_text(
        """
[prompt]
width = 320
height = 256
frame_count = 1
sample_steps = 8

[[prompt.subset]]
prompt = "lucy the cat"
seed = 42
""".strip(),
        encoding="utf-8",
    )

    args = SimpleNamespace(dataset_config=str(dataset_path))
    samples = MiniMaxH3NetworkTrainer().process_sample_prompts(args, None, str(prompt_path))

    assert len(samples) == 1
    assert samples[0]["prompt"] == "lucy the cat"
    assert samples[0]["frame_count"] == 1
    assert samples[0]["h3_text_embed"].shape == (3, 8)
    assert samples[0]["h3_token_tags"].tolist() == [1, 1, 0]


def test_h3_sample_prompt_aligns_video_frame_counts(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(
        {
            "version": 1,
            "architecture": "minimax_h3",
            "prompt_cache": [
                {
                    "prompt": "a woman contorting her body",
                    "h3_text_embed": torch.zeros(2, 8),
                    "h3_token_tags": torch.ones(2, dtype=torch.long),
                }
            ],
        },
        cache_dir / "minimax_h3_sample_prompts_cache.pt",
    )
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        f'[[datasets]]\nimage_directory = "{tmp_path}"\ncache_directory = "{cache_dir}"\n',
        encoding="utf-8",
    )
    prompt_path = tmp_path / "samples.json"
    prompt_path.write_text(
        json.dumps([{"prompt": "a woman contorting her body", "frame_count": 50, "width": 64, "height": 64}]),
        encoding="utf-8",
    )

    samples = MiniMaxH3NetworkTrainer().process_sample_prompts(
        SimpleNamespace(dataset_config=str(dataset_path)), None, str(prompt_path)
    )

    assert samples[0]["frame_count"] == 39  # 50 aligned down to 17*2+5


def test_h3_single_frame_decode_duplicates_t1_and_returns_last_decoder_frame():
    latent = torch.randn(1, 24, 1, 2, 2)
    seen = {}

    def fake_decode_video(value):
        seen["value"] = value.clone()
        decoded = torch.zeros(1, 3, 4, 4, 4)
        decoded[:, :, 0] = 1
        decoded[:, :, -1] = 2
        return decoded

    decoded = decode_single_frame_latent(latent, fake_decode_video)

    assert seen["value"].shape[2] == 2
    torch.testing.assert_close(seen["value"][:, :, :1], latent)
    torch.testing.assert_close(seen["value"][:, :, 1:], latent)
    assert decoded.shape == (1, 3, 1, 4, 4)
    torch.testing.assert_close(decoded, torch.full_like(decoded, 2))


def test_h3_lora_target_default_is_pruned_checkpoint_portable():
    args = minimax_h3_setup_parser(argparse.ArgumentParser()).parse_args([])

    assert args.lora_target_preset == "attn_mlp"
    assert args.image_audio_mode == "none"
    assert args.sample_latent_frames == 2
    assert args.sample_audio_mode == "auto"
    assert args.sample_solver == "ab2"
    assert args.sample_frame_select == "dup_last"


def test_h3_inference_returns_pixels_in_saver_range():
    events = []
    sigmas = []

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            sigmas.append(float(sigma.item()))
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            events.append("park")

        def restore_main_block_weights_after_decode(self):
            events.append("restore")

    class FakeVae:
        def to(self, device):
            events.append(f"vae:{torch.device(device).type}")
            return self

        def decode(self, latents, frame_select="first"):
            return torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)

    pixels = MiniMaxH3NetworkTrainer().do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(video_flow_shift=12.0, audio_flow_shift=3.0),
        {"h3_text_embed": torch.zeros(3, 8), "h3_token_tags": torch.ones(3, dtype=torch.long)},
        FakeVae(),
        torch.float32,
        FakeTransformer(),
        3.0,
        2,
        64,
        64,
        1,
        torch.Generator().manual_seed(1),
        False,
        1.0,
        None,
    )

    torch.testing.assert_close(pixels.flatten(), torch.tensor([0.0, 0.5, 1.0]))
    assert sigmas == pytest.approx([1.0, 0.75])
    assert events == ["park", "vae:cpu", "vae:cpu", "restore"]


def test_h3_inference_restores_base_blocks_when_decode_fails():
    events = []

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            events.append("park")

        def restore_main_block_weights_after_decode(self):
            events.append("restore")

    class FakeVae:
        def to(self, device):
            events.append(f"vae:{torch.device(device).type}")
            return self

        def decode(self, latents, frame_select="first"):
            events.append("decode")
            raise RuntimeError("decode failed")

    sample = {
        "h3_text_embed": torch.zeros(3, 8),
        "h3_token_tags": torch.ones(3, dtype=torch.long),
    }
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(RuntimeError, match="decode failed"):
        trainer.do_inference(
            SimpleNamespace(device=torch.device("cpu")),
            SimpleNamespace(video_flow_shift=12.0, audio_flow_shift=3.0),
            sample,
            FakeVae(),
            torch.float32,
            FakeTransformer(),
            12.0,
            2,
            64,
            64,
            1,
            torch.Generator().manual_seed(1),
            False,
            1.0,
            None,
        )

    assert events == ["park", "vae:cpu", "decode", "vae:cpu", "restore"]


def test_h3_sampling_two_latents_with_silent_audio():
    shapes = []

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            shapes.append((tuple(video.shape), tuple(audio.shape)))
            return torch.ones_like(video), torch.zeros_like(audio)

    sampled = sample_h3_image_latents(
        FakeTransformer(),
        torch.zeros(3, 8),
        torch.ones(3, dtype=torch.long),
        height=64,
        width=96,
        sample_steps=3,
        video_flow_shift=12.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(1),
        latent_frames=2,
        audio_latent_frames=8,
        latents=torch.zeros(1, 24, 2, 4, 6),
    )

    assert sampled.shape == (1, 24, 2, 4, 6)
    assert all(shape == ((1, 24, 2, 4, 6), (1, 32, 2, 8)) for shape in shapes)
    torch.testing.assert_close(sampled, torch.ones_like(sampled))


def test_h3_ab2_beats_euler_on_sigma_dependent_field():
    class SigmaField:
        def __call__(self, video, audio, sigma, context, tags):
            return torch.full_like(video, float(sigma.item())), torch.empty_like(audio)

    common = dict(
        context=torch.zeros(2, 8),
        token_tags=torch.ones(2, dtype=torch.long),
        height=32,
        width=32,
        sample_steps=4,
        video_flow_shift=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(1),
        latent_frames=1,
    )
    zero = torch.zeros(1, 24, 1, 2, 2)
    euler = sample_h3_image_latents(SigmaField(), common.pop("context"), common.pop("token_tags"), solver="euler", latents=zero.clone(), **common)
    ab2 = sample_h3_image_latents(SigmaField(), torch.zeros(2, 8), torch.ones(2, dtype=torch.long), solver="ab2", latents=zero.clone(), **common)

    # Exact integral of v(sigma) = sigma from 1 to 0 is 0.5.
    euler_error = float((euler.mean() - 0.5).abs())
    ab2_error = float((ab2.mean() - 0.5).abs())
    assert ab2_error < euler_error
    assert euler_error == pytest.approx(0.125)
    assert ab2_error == pytest.approx(0.03125)


def _fake_decode_clip(chunk):
    batch, _, tokens, height, width = chunk.shape
    frames = torch.zeros(batch, 3, tokens * 4, height, width)
    for index in range(tokens * 4):
        frames[:, :, index] = index / 100.0
    return frames


@pytest.mark.parametrize(("latent_t", "expected_frames"), [(2, 5), (7, 22), (12, 39)])
def test_h3_chunked_video_decode_frame_counts(latent_t, expected_frames):
    chunks_seen = []

    def decode_clip(chunk):
        chunks_seen.append(int(chunk.shape[2]))
        return _fake_decode_clip(chunk)

    frames = decode_video_latents(torch.randn(1, 24, latent_t, 2, 2), decode_clip)

    assert frames.shape == (1, 3, expected_frames, 2, 2)
    # every chunk the decoder sees is a full 5-token chunk plus 2 overlap tokens
    assert all(size == 7 for size in chunks_seen)


def test_h3_select_preview_frame():
    frames = torch.zeros(1, 3, 5, 8, 8)
    for index in range(5):
        frames[:, :, index] = index / 10.0
    frames[:, :, 2, ::2, ::2] = 1.0  # frame 2 carries the only detail

    torch.testing.assert_close(select_preview_frame(frames, "first"), frames[:, :, :1])
    torch.testing.assert_close(select_preview_frame(frames, "last"), frames[:, :, 4:5])
    torch.testing.assert_close(select_preview_frame(frames, "sharpest"), frames[:, :, 2:3])
    with pytest.raises(ValueError, match="frame_select"):
        select_preview_frame(frames, "median")


def test_h3_video_preview_inference_uses_aligned_frame_count():
    seen = {}

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            seen["video_shape"] = tuple(video.shape)
            seen["audio_shape"] = tuple(audio.shape)
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            pass

        def restore_main_block_weights_after_decode(self):
            pass

    class FakeVae:
        def to(self, device):
            return self

        def decode_video(self, latents):
            seen["decoded_latent_t"] = int(latents.shape[2])
            return torch.zeros(1, 3, 22, 6, 4)

    args = SimpleNamespace(
        video_flow_shift=12.0,
        audio_flow_shift=3.0,
        sample_latent_frames=2,
        sample_audio_mode="none",
        sample_solver="euler",
        sample_frame_select="dup_last",
        image_audio_mode="none",
    )
    sample = {
        "h3_text_embed": torch.zeros(3, 8),
        "h3_token_tags": torch.ones(3, dtype=torch.long),
        "frame_count": 22,
    }

    pixels = MiniMaxH3NetworkTrainer().do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        args,
        sample,
        FakeVae(),
        torch.float32,
        FakeTransformer(),
        12.0,
        1,
        64,
        96,
        18,  # deliberately wrong base-trainer rounding; must be ignored
        torch.Generator().manual_seed(1),
        False,
        1.0,
        None,
    )

    assert seen["video_shape"] == (1, 24, 7, 6, 4)
    assert seen["audio_shape"] == (1, 32, 2, 0)
    assert seen["decoded_latent_t"] == 7
    assert pixels.shape == (1, 3, 22, 6, 4)


def test_h3_prompt_level_sampling_overrides_beat_cli_flags():
    seen = {}

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            seen["video_shape"] = tuple(video.shape)
            seen["audio_shape"] = tuple(audio.shape)
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            pass

        def restore_main_block_weights_after_decode(self):
            pass

    class FakeVae:
        def to(self, device):
            return self

        def decode(self, latents, frame_select="first"):
            seen["frame_select"] = frame_select
            seen["latent_t"] = int(latents.shape[2])
            return torch.zeros(1, 3, 1, 1, 1)

    args = SimpleNamespace(
        video_flow_shift=12.0,
        audio_flow_shift=3.0,
        sample_latent_frames=2,
        sample_audio_mode="silent",
        sample_solver="ab2",
        sample_frame_select="sharpest",
        image_audio_mode="none",
    )
    sample = {
        "h3_text_embed": torch.zeros(3, 8),
        "h3_token_tags": torch.ones(3, dtype=torch.long),
        "sample_latent_frames": 1,
        "sample_audio_mode": "none",
        "sample_frame_select": "last",
    }

    MiniMaxH3NetworkTrainer().do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        args,
        sample,
        FakeVae(),
        torch.float32,
        FakeTransformer(),
        12.0,
        1,
        64,
        64,
        1,
        torch.Generator().manual_seed(1),
        False,
        1.0,
        None,
    )

    assert seen["video_shape"] == (1, 24, 1, 4, 4)
    assert seen["audio_shape"] == (1, 32, 2, 0)
    assert seen["latent_t"] == 1
    assert seen["frame_select"] == "last"


def test_h3_prompt_level_sampling_overrides_are_validated(tmp_path):
    prompt_path = tmp_path / "samples.json"
    prompt_path.write_text(json.dumps([{"prompt": "lucy", "sample_solver": "dpm"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_solver"):
        MiniMaxH3NetworkTrainer().process_sample_prompts(SimpleNamespace(), None, str(prompt_path))


def _bare_overlay_sd(scale_up=1.0):
    return {
        "blocks.0.mlp.fc1.lora_A.weight": torch.ones(1, 4),
        "blocks.0.mlp.fc1.lora_B.weight": torch.full((4, 1), float(scale_up)),
    }


def test_h3_overlay_normalize_formats_equivalent():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    down, up = torch.randn(2, 4), torch.randn(4, 2)
    bare = {"blocks.3.attn.qkv_proj.lora_A.weight": down, "blocks.3.attn.qkv_proj.lora_B.weight": up}
    comfy = {
        "diffusion_model.blocks.3.attn.qkv_proj.lora_A.weight": down,
        "diffusion_model.blocks.3.attn.qkv_proj.lora_B.weight": up,
    }
    kohya = {
        "lora_unet_blocks_3_attn_qkv_proj.lora_down.weight": down,
        "lora_unet_blocks_3_attn_qkv_proj.lora_up.weight": up,
        "lora_unet_blocks_3_attn_qkv_proj.alpha": torch.tensor(1.0),
    }
    n_bare = slo.normalize_overlay_state_dict(bare)
    n_comfy = slo.normalize_overlay_state_dict(comfy)
    n_kohya = slo.normalize_overlay_state_dict(kohya)
    assert list(n_bare) == list(n_comfy) == list(n_kohya) == ["blocks.3.attn.qkv_proj"]
    assert n_bare["blocks.3.attn.qkv_proj"][0]["alpha"] is None
    assert n_kohya["blocks.3.attn.qkv_proj"][0]["alpha"] == 1.0
    with pytest.raises(ValueError):
        slo.normalize_overlay_state_dict({"totally.unknown.key": down})


def test_h3_overlay_linear_math_and_clear():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.fc1 = torch.nn.Linear(4, 4, bias=False)
            self.blocks = torch.nn.ModuleList([block])

    model = Tiny()
    torch.nn.init.eye_(model.blocks[0].mlp.fc1.weight)
    x = torch.ones(1, 4)

    # kohya alpha=rank/2 must halve the delta on top of the 0.5 strength
    sd = {
        "lora_unet_blocks_0_mlp_fc1.lora_down.weight": torch.ones(2, 4),
        "lora_unet_blocks_0_mlp_fc1.lora_up.weight": torch.ones(4, 2),
        "lora_unet_blocks_0_mlp_fc1.alpha": torch.tensor(1.0),  # rank 2 -> scale 0.5
    }
    modules = slo.normalize_overlay_state_dict(sd)
    stats = slo.attach_sampling_lora_overlays(model, [(modules, 0.5)])
    assert stats == {"backbone": 1, "adaln_grid": 0, "skipped": []}
    # delta = up @ down @ x = 8 per element; scaled by 0.5 (strength) * 0.5 (alpha/rank)
    torch.testing.assert_close(model.blocks[0].mlp.fc1(x), torch.ones(1, 4) + 2.0)

    assert slo.clear_sampling_lora_overlays(model) == 1
    torch.testing.assert_close(model.blocks[0].mlp.fc1(x), torch.ones(1, 4))
    assert not hasattr(model.blocks[0].mlp.fc1, "_h3_sampling_overlay_orig_forward")


def _tiny_curve_model():
    from musubi_tuner.minimax_h3.model import AdalnProj

    class TinyCurve(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.adaln_proj = AdalnProj(time_dim=8, hidden=2, expand=6, modalities=3, apply_silu=False)
            self.blocks = torch.nn.ModuleList([block])
            self.use_adaln_curves = True
            self.sigma_shift_video = 12.0
            self.sigma_shift_audio = 3.0

    return TinyCurve()


def test_h3_overlay_adaln_grid_injection():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    model = _tiny_curve_model()
    adaln = model.blocks[0].adaln_proj
    full_dim, out_dim = 16, 6 * 2 * 3
    grid = torch.linspace(0.0, 1.0, 5).unsqueeze(1).repeat(1, full_dim)  # row value == t
    sd = {
        "blocks.0.adaln_proj.linear.lora_A.weight": torch.ones(1, full_dim),
        "blocks.0.adaln_proj.linear.lora_B.weight": torch.ones(out_dim, 1),
    }
    coeffs = torch.zeros(2, 8)
    base = torch.cat(adaln(coeffs), dim=-1)

    stats = slo.attach_sampling_lora_overlays(model, [(slo.normalize_overlay_state_dict(sd), 1.0)], temb_grid=grid)
    assert stats == {"backbone": 0, "adaln_grid": 1, "skipped": []}

    # Before update_time_state the overlay must be inert.
    torch.testing.assert_close(torch.cat(adaln(coeffs), dim=-1), base)

    slo.update_time_state(model, torch.tensor(0.5))
    # t_video=0.5, sigma_audio=time_shift(0.5,12,3)=0.2 -> t_audio=0.8; rows sorted [0.5, 0.8]
    out = torch.cat(adaln(coeffs), dim=-1).view(2, 3, 2 * 6)  # [times, modalities, expand*hidden]
    base_v = base.view(2, 3, 2 * 6)
    torch.testing.assert_close(out[0] - base_v[0], torch.full((3, 12), 16 * 0.5))
    torch.testing.assert_close(out[1] - base_v[1], torch.full((3, 12), 16 * 0.8))

    assert slo.clear_sampling_lora_overlays(model) >= 1
    torch.testing.assert_close(torch.cat(adaln(coeffs), dim=-1), base)


def test_h3_overlay_adaln_skipped_without_grid():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    model = _tiny_curve_model()
    sd = {
        "blocks.0.adaln_proj.linear.lora_A.weight": torch.ones(1, 16),
        "blocks.0.adaln_proj.linear.lora_B.weight": torch.ones(36, 1),
    }
    stats = slo.attach_sampling_lora_overlays(model, [(slo.normalize_overlay_state_dict(sd), 1.0)], temb_grid=None)
    assert stats["adaln_grid"] == 0 and stats["skipped"] == ["blocks.0.adaln_proj.linear"]
    assert not hasattr(model.blocks[0].adaln_proj, "_h3_sampling_overlay_orig_forward")
    slo.update_time_state(model, torch.tensor(0.5))  # no shared state -> no-op


def test_h3_do_inference_overlay_toggle():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    probes = []

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.fc1 = torch.nn.Linear(4, 4, bias=False)
            self.blocks = torch.nn.ModuleList([block])
            torch.nn.init.eye_(self.blocks[0].mlp.fc1.weight)

        def forward(self, video, audio, sigma, context, tags):
            probes.append(self.blocks[0].mlp.fc1(torch.ones(1, 4)).detach().clone())
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            pass

        def restore_main_block_weights_after_decode(self):
            pass

    class FakeVae:
        def to(self, device):
            return self

        def decode_video(self, latents):
            return torch.zeros(1, 3, 22, 6, 4)

    args = SimpleNamespace(
        video_flow_shift=12.0,
        audio_flow_shift=3.0,
        sample_latent_frames=2,
        sample_audio_mode="none",
        sample_solver="euler",
        sample_frame_select="dup_last",
        image_audio_mode="none",
    )
    sample = {
        "h3_text_embed": torch.zeros(3, 8),
        "h3_token_tags": torch.ones(3, dtype=torch.long),
        "frame_count": 22,
    }

    trainer = MiniMaxH3NetworkTrainer()
    trainer._sampling_overlays = [(slo.normalize_overlay_state_dict(_bare_overlay_sd()), 1.0)]
    trainer._sampling_overlay_grid = None
    transformer = TinyTransformer()

    def run(sample_parameter):
        return trainer.do_inference(
            SimpleNamespace(device=torch.device("cpu"), print=lambda *a, **k: None),
            args, sample_parameter, FakeVae(), torch.float32, transformer,
            12.0, 1, 64, 96, 18, torch.Generator().manual_seed(1), False, 1.0, None,
        )

    run(sample)
    # overlay active during sampling: delta = 4 on top of identity ones
    torch.testing.assert_close(probes[-1], torch.full((1, 4), 5.0))
    # and detached afterwards
    assert not hasattr(transformer.blocks[0].mlp.fc1, "_h3_sampling_overlay_orig_forward")
    torch.testing.assert_close(transformer.blocks[0].mlp.fc1(torch.ones(1, 4)), torch.ones(1, 4))

    probes.clear()
    run({**sample, "sample_lora_overlay": 0})
    torch.testing.assert_close(probes[-1], torch.ones(1, 4))


def test_h3_prompt_subset_overrides_prompt_defaults(tmp_path):
    from musubi_tuner.training.sampling_prompts import load_prompts

    prompt_file = tmp_path / "p.toml"
    prompt_file.write_text(
        """
[prompt]
width = 768
sample_steps = 4

[[prompt.subset]]
prompt = "a"

[[prompt.subset]]
prompt = "b"
sample_steps = 20
sample_lora_overlay = 0
""",
        encoding="utf-8",
    )
    prompts = load_prompts(str(prompt_file))
    assert [p["sample_steps"] for p in prompts] == [4, 20]
    assert prompts[1]["sample_lora_overlay"] == 0
    assert all("subset" not in p for p in prompts)


def test_h3_preview_solver_guard():
    from musubi_tuner.minimax_h3_train_network import resolve_preview_solver

    assert resolve_preview_solver("ab2", 4) == "euler"
    assert resolve_preview_solver("ab2", 7) == "euler"
    assert resolve_preview_solver("ab2", 8) == "ab2"
    assert resolve_preview_solver("ab2", 20) == "ab2"
    assert resolve_preview_solver("euler", 4) == "euler"


def test_h3_preview_summary_logged(caplog):
    import logging as _logging

    class FakeTransformer:
        def __call__(self, video, audio, sigma, context, tags):
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            pass

        def restore_main_block_weights_after_decode(self):
            pass

    class FakeVae:
        def to(self, device):
            return self

        def decode_video(self, latents):
            return torch.zeros(1, 3, 22, 6, 4)

    args = SimpleNamespace(
        video_flow_shift=12.0, audio_flow_shift=3.0, sample_latent_frames=2,
        sample_audio_mode="none", sample_solver="ab2", sample_frame_select="dup_last",
        image_audio_mode="none",
    )
    sample = {"h3_text_embed": torch.zeros(3, 8), "h3_token_tags": torch.ones(3, dtype=torch.long), "frame_count": 22}
    with caplog.at_level(_logging.INFO, logger="musubi_tuner.minimax_h3_train_network"):
        MiniMaxH3NetworkTrainer().do_inference(
            SimpleNamespace(device=torch.device("cpu"), print=lambda *a, **k: None),
            args, sample, FakeVae(), torch.float32, FakeTransformer(),
            12.0, 4, 64, 96, 18, torch.Generator().manual_seed(1), False, 1.0, None,
        )
    summary = [r.message for r in caplog.records if "H3 preview done" in r.message]
    assert len(summary) == 1
    # 4-step ab2 must have been guarded down to euler, and timings must be present
    assert "solver=euler" in summary[0] and "steps=4" in summary[0] and "frames=22" in summary[0]
    assert "sample " in summary[0] and "decode " in summary[0]


def test_h3_do_inference_overlay_prompt_strength():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    probes = []

    class TinyTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.fc1 = torch.nn.Linear(4, 4, bias=False)
            self.blocks = torch.nn.ModuleList([block])
            torch.nn.init.eye_(self.blocks[0].mlp.fc1.weight)

        def forward(self, video, audio, sigma, context, tags):
            probes.append(self.blocks[0].mlp.fc1(torch.ones(1, 4)).detach().clone())
            return torch.zeros_like(video), torch.empty_like(audio)

        def park_main_block_weights_for_decode(self):
            pass

        def restore_main_block_weights_after_decode(self):
            pass

    class FakeVae:
        def to(self, device):
            return self

        def decode_video(self, latents):
            return torch.zeros(1, 3, 22, 6, 4)

    args = SimpleNamespace(
        video_flow_shift=12.0, audio_flow_shift=3.0, sample_latent_frames=2,
        sample_audio_mode="none", sample_solver="euler", sample_frame_select="dup_last",
        image_audio_mode="none",
    )
    base = {"h3_text_embed": torch.zeros(3, 8), "h3_token_tags": torch.ones(3, dtype=torch.long), "frame_count": 22}
    trainer = MiniMaxH3NetworkTrainer()
    trainer._sampling_overlays = [(slo.normalize_overlay_state_dict(_bare_overlay_sd()), 1.0)]
    trainer._sampling_overlay_grid = None
    transformer = TinyTransformer()

    def run(extra):
        probes.clear()
        trainer.do_inference(
            SimpleNamespace(device=torch.device("cpu"), print=lambda *a, **k: None),
            args, {**base, **extra}, FakeVae(), torch.float32, transformer,
            12.0, 1, 64, 96, 18, torch.Generator().manual_seed(1), False, 1.0, None,
        )
        return probes[-1]

    # delta at full strength is +4; per-prompt 0.5 halves it, 0 disables
    torch.testing.assert_close(run({"sample_lora_overlay": 0.5}), torch.full((1, 4), 3.0))
    torch.testing.assert_close(run({"sample_lora_overlay": 0}), torch.ones(1, 4))
    torch.testing.assert_close(run({"sample_lora_overlay": "true"}), torch.full((1, 4), 5.0))


def test_h3_overlay_diffusers_peft_mapping():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    r, hidden, inner, ffn = 2, 6, 4, 3
    sd = {}
    for name in ("to_q", "to_k", "to_v"):
        sd[f"transformer_blocks.7.attn.{name}.lora_A.default.weight"] = torch.randn(r, hidden)
        sd[f"transformer_blocks.7.attn.{name}.lora_B.default.weight"] = torch.randn(inner, r)
    sd["transformer_blocks.7.attn.to_out.0.lora_A.default.weight"] = torch.randn(r, inner)
    sd["transformer_blocks.7.attn.to_out.0.lora_B.default.weight"] = torch.randn(hidden, r)
    ff_up = torch.randn(2 * ffn, r)
    sd["transformer_blocks.7.ff.net.0.proj.lora_A.default.weight"] = torch.randn(r, hidden)
    sd["transformer_blocks.7.ff.net.0.proj.lora_B.default.weight"] = ff_up
    sd["token_refiner.refiner_blocks.1.ff.net.2.lora_A.default.weight"] = torch.randn(r, ffn)
    sd["token_refiner.refiner_blocks.1.ff.net.2.lora_B.default.weight"] = torch.randn(hidden, r)

    modules = slo.normalize_overlay_state_dict(sd)
    assert sorted(modules) == [
        "blocks.7.attn.out_proj",
        "blocks.7.attn.qkv_proj",
        "blocks.7.mlp.fc1",
        "token_refiner.blocks.1.mlp.fc2",
    ]
    qkv = modules["blocks.7.attn.qkv_proj"]
    assert sorted(e["offset"] for e in qkv) == [0, inner, 2 * inner]
    assert all(e["alpha"] == slo.DIFFUSERS_PEFT_ALPHA for e in qkv)
    # SwiGLU halves swapped: diffusers [value; gate] -> ours [gate; value]
    fc1 = modules["blocks.7.mlp.fc1"][0]
    torch.testing.assert_close(fc1["up"], torch.cat([ff_up[ffn:], ff_up[:ffn]], dim=0))
    assert modules["blocks.7.attn.out_proj"][0]["offset"] is None


def test_h3_overlay_split_qkv_matches_fused_delta():
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    hidden, inner, r = 6, 4, 2

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.qkv_proj = torch.nn.Linear(hidden, 3 * inner, bias=False)
            self.blocks = torch.nn.ModuleList([block])

    torch.manual_seed(3)
    model = Tiny()
    sd = {}
    parts = {}
    for name in ("to_q", "to_k", "to_v"):
        a, b = torch.randn(r, hidden), torch.randn(inner, r)
        sd[f"transformer_blocks.0.attn.{name}.lora_A.default.weight"] = a
        sd[f"transformer_blocks.0.attn.{name}.lora_B.default.weight"] = b
        parts[name] = (a, b)

    x = torch.randn(1, 5, hidden)
    base = model.blocks[0].attn.qkv_proj(x)
    slo.attach_sampling_lora_overlays(model, [(slo.normalize_overlay_state_dict(sd), 1.0)])
    got = model.blocks[0].attn.qkv_proj(x)

    scale = slo.DIFFUSERS_PEFT_ALPHA / r
    expected = base.clone()
    for i, name in enumerate(("to_q", "to_k", "to_v")):
        a, b = parts[name]
        expected[..., i * inner : (i + 1) * inner] += scale * (x @ a.T @ b.T)
    torch.testing.assert_close(got, expected)
    slo.clear_sampling_lora_overlays(model)
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj(x), base)


def test_h3_overlay_ai_toolkit_adapter_keys():
    # The ostris minimax_h3_training_adapter format: diffusion_model. prefix,
    # bare lora_A/lora_B suffixes (no .default.), fused qkv, no alpha keys.
    from musubi_tuner.minimax_h3 import sampling_lora_overlay as slo

    sd = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.randn(2, 4),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.randn(12, 2),
        "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_A.weight": torch.randn(2, 4),
        "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_B.weight": torch.randn(16, 2),
    }
    modules = slo.normalize_overlay_state_dict(sd)

    assert set(modules) == {"blocks.0.attn.qkv_proj", "token_refiner.blocks.1.mlp.fc1"}
    entry = modules["blocks.0.attn.qkv_proj"][0]
    assert entry["alpha"] is None and entry["offset"] is None
    # alpha-less PEFT-style export => scale is the strength verbatim
    assert slo._module_scale(entry, 0.7) == pytest.approx(0.7)


def test_h3_train_overlay_lifecycle(tmp_path):
    import safetensors.torch

    from musubi_tuner.minimax_h3_train_network import _parse_overlay_spec

    assert _parse_overlay_spec("/a/b.safetensors") == ("/a/b.safetensors", 1.0)
    assert _parse_overlay_spec("/a/b.safetensors:0.5") == ("/a/b.safetensors", 0.5)

    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.fc1 = torch.nn.Linear(4, 8, bias=False)
            self.blocks = torch.nn.ModuleList([block])

        def forward(self, x):
            return self.blocks[0].mlp.fc1(x)

    model = Tiny()
    # Simulate the trainable network's forward wrapper (kohya LoRA apply_to).
    fc1 = model.blocks[0].mlp.fc1
    base_forward = fc1.forward
    fc1.forward = lambda x: base_forward(x) + 1.0

    down, up = torch.randn(2, 4), torch.randn(8, 2)
    adapter_path = tmp_path / "adapter.safetensors"
    safetensors.torch.save_file(
        {
            "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": down.to(torch.bfloat16),
            "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": up.to(torch.bfloat16),
        },
        str(adapter_path),
    )

    trainer = MiniMaxH3NetworkTrainer()
    accelerator = SimpleNamespace(print=lambda *a, **k: None, unwrap_model=lambda m: m)
    args = SimpleNamespace(train_lora_overlay=f"{adapter_path}:0.5")

    x = torch.randn(3, 4, requires_grad=True)
    wrapped_out = model(x)

    trainer._ensure_train_overlay(args, accelerator, model)
    overlaid = model(x)
    bf = torch.bfloat16
    expected_delta = 0.5 * (x.to(bf) @ down.to(bf).T @ up.to(bf).T).to(overlaid.dtype)
    torch.testing.assert_close(overlaid, wrapped_out + expected_delta, rtol=2e-2, atol=2e-2)
    assert overlaid.grad_fn is not None  # frozen assistant must not block autograd

    # Previews detach the assistant but must restore the TRAINED wrapper, not the bare module.
    trainer._detach_train_overlay(accelerator, model)
    torch.testing.assert_close(model(x), wrapped_out)
    assert trainer._train_overlay_attached is False

    # The next training step lazily re-attaches from the already-normalized tensors.
    trainer._ensure_train_overlay(args, accelerator, model)
    torch.testing.assert_close(model(x), overlaid)


def test_h3_train_overlay_off_by_default():
    trainer = MiniMaxH3NetworkTrainer()
    accelerator = SimpleNamespace(print=lambda *a, **k: None, unwrap_model=lambda m: m)
    model = torch.nn.Linear(2, 2)

    trainer._ensure_train_overlay(SimpleNamespace(train_lora_overlay=None), accelerator, model)

    assert not getattr(trainer, "_train_overlay_attached", False)
    assert not hasattr(model, "_h3_sampling_overlay_orig_forward")


def test_h3_adapter_cosine_flags_anti_adapter(tmp_path):
    import safetensors.torch

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.fc1 = torch.nn.Linear(6, 8, bias=False)
            self.blocks = torch.nn.ModuleList([block])

    torch.manual_seed(0)
    da, ua = torch.randn(3, 6), torch.randn(8, 3)
    adapter_path = tmp_path / "adapter.safetensors"
    safetensors.torch.save_file(
        {
            "diffusion_model.blocks.0.mlp.fc1.lora_A.weight": da,
            "diffusion_model.blocks.0.mlp.fc1.lora_B.weight": ua,
        },
        str(adapter_path),
    )

    class FakeLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_name = "lora_unet_blocks_0_mlp_fc1"
            self.lora_down = torch.nn.Linear(6, 3, bias=False)
            self.lora_up = torch.nn.Linear(3, 8, bias=False)
            with torch.no_grad():
                self.lora_down.weight.copy_(da)
                self.lora_up.weight.copy_(-ua)  # exact anti-adapter delta

    class FakeNetwork:
        unet_loras = [FakeLora()]

    model = Tiny()
    trainer = MiniMaxH3NetworkTrainer()
    accelerator = SimpleNamespace(print=lambda *a, **k: None, unwrap_model=lambda m: m)
    args = SimpleNamespace(train_lora_overlay=str(adapter_path))

    assert trainer._adapter_cosine(FakeNetwork(), model) is None  # overlay not attached
    assert trainer.extra_postfix_logs(args, accelerator, FakeNetwork(), model) == {}
    assert trainer.extra_step_logs(args, {}) == {}

    trainer._ensure_train_overlay(args, accelerator, model)
    assert trainer._adapter_cosine(FakeNetwork(), model) == pytest.approx(-1.0, abs=1e-4)
    assert trainer.extra_postfix_logs(args, accelerator, FakeNetwork(), model)["acos"] == pytest.approx(-1.0, abs=1e-3)
    assert trainer.extra_step_logs(args, {})["network/adapter_cos"] == pytest.approx(-1.0, abs=1e-4)


def test_h3_vae_int8_marker_scan(tmp_path):
    import safetensors.torch

    from musubi_tuner.minimax_h3.video_vae import inspect_vae_int8_layers

    def marker(payload):
        import json as _json

        return torch.frombuffer(bytearray(_json.dumps(payload).encode()), dtype=torch.uint8).clone()

    path = tmp_path / "vae_int8.safetensors"
    safetensors.torch.save_file(
        {
            "decoder.transformer_blocks.0.attn.to_out.comfy_quant": marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
            ),
            "decoder.transformer_blocks.0.attn.to_out.weight": torch.zeros(8, 256, dtype=torch.int8),
            "decoder.transformer_blocks.0.attn.to_out.weight_scale": torch.ones(8, 1),
            # encoder-side quantization must be ignored by the decoder loader
            "encoder.blocks.0.proj.comfy_quant": marker(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
            ),
            "encoder.blocks.0.proj.weight": torch.zeros(4, 256, dtype=torch.int8),
            "encoder.blocks.0.proj.weight_scale": torch.ones(4, 1),
            "decoder.conv.weight": torch.zeros(2, 2, 1, 1, 1),
        },
        str(path),
    )

    layers = inspect_vae_int8_layers([path])

    assert set(layers) == {"decoder.transformer_blocks.0.attn.to_out"}
    config = layers["decoder.transformer_blocks.0.attn.to_out"]
    assert config.group_size == 256
    assert config.scale_shape == (8, 1)

    # unsupported formats fail loudly instead of silently loading garbage
    bad = tmp_path / "vae_bad.safetensors"
    safetensors.torch.save_file(
        {
            "decoder.x.comfy_quant": marker({"format": "fp8_scaled"}),
            "decoder.x.weight": torch.zeros(4, 256, dtype=torch.int8),
            "decoder.x.weight_scale": torch.ones(4, 1),
        },
        str(bad),
    )
    with pytest.raises(ValueError, match="INT8 ConvRot"):
        inspect_vae_int8_layers([bad])


@pytest.mark.skipif(
    not __import__("os").path.exists("/home/pyro/models/comfy/vae/minimax_h3_video_vae_int8_convrot.safetensors"),
    reason="local kijai INT8 VAE not present",
)
def test_h3_vae_int8_decoder_loads_real_checkpoint():
    from musubi_tuner.minimax_h3.video_vae import load_video_vae_decoder

    model = load_video_vae_decoder(
        "/home/pyro/models/comfy/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        device="cpu",
        dtype=torch.float16,
    )
    quantized = [m for m in model.modules() if getattr(m, "int8_convrot_group_size", None)]
    assert len(quantized) == 144
    assert all(m.weight.dtype == torch.int8 for m in quantized)
    assert all(m.weight_scale.dtype == torch.float32 for m in quantized)


def test_h3_cfg_augmentation_math_and_gradients():
    trainer = MiniMaxH3NetworkTrainer()
    pred = torch.tensor([2.0, 4.0], requires_grad=True)
    uncond = torch.tensor([1.0, 1.0])

    out = trainer._apply_cfg_augmentation(pred, uncond, 4.0)

    torch.testing.assert_close(out, torch.tensor([5.0, 13.0]))  # uncond + 4*(pred-uncond)
    out.sum().backward()
    torch.testing.assert_close(pred.grad, torch.full((2,), 4.0))  # cond branch scaled by s
    assert not uncond.requires_grad


def test_h3_cfg_augmented_scale_validation():
    trainer = MiniMaxH3NetworkTrainer()

    def make_args(**overrides):
        base = dict(
            fp8_base=False,
            fp8_scaled=False,
            mixed_precision="bf16",
            video_flow_shift=12.0,
            audio_flow_shift=3.0,
            audio_loss_weight=0.0,
            sample_blocks_to_swap=None,
            sample_prompts=None,
            block_swap_h2d_only=False,
            train_lora_overlay=None,
            cfg_augmented_scale=1.0,
            lora_target_preset="attn_mlp",
            network_args=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    trainer.handle_model_specific_args(make_args(cfg_augmented_scale=4.0))  # ok

    with pytest.raises(ValueError, match="cfg_augmented_scale"):
        trainer.handle_model_specific_args(make_args(cfg_augmented_scale=0.5))
    with pytest.raises(ValueError, match="use one"):
        trainer.handle_model_specific_args(make_args(cfg_augmented_scale=4.0, train_lora_overlay="x.safetensors"))


def test_h3_uncond_cache_roundtrip(tmp_path):
    from musubi_tuner.minimax_h3_cache_text_encoder_outputs import _precache_uncond

    cache_dir = tmp_path / "cache"
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text(
        f'[[datasets]]\nimage_directory = "{tmp_path}"\ncache_directory = "{cache_dir}"\n'
    )
    te = tmp_path / "te.safetensors"
    te.write_bytes(b"stub")

    def encode_prompt_list(prompts):
        assert prompts == [""]
        return [torch.ones(5, 8)], [torch.ones(5, dtype=torch.long)]

    args = SimpleNamespace(dataset_config=str(dataset_toml), text_encoder=str(te), max_token_length=1024)
    _precache_uncond(args, None, encode_prompt_list)

    trainer = MiniMaxH3NetworkTrainer()
    accelerator = SimpleNamespace(device="cpu", print=lambda *a, **k: None)
    embed, tags = trainer._get_uncond_context(
        SimpleNamespace(dataset_config=str(dataset_toml)), accelerator, torch.float32
    )
    assert embed.shape == (5, 8) and tags.shape == (5,)
    assert trainer._get_uncond_context(None, None, None) == (embed, tags)  # cached, args unused

    # missing cache produces an actionable error
    trainer2 = MiniMaxH3NetworkTrainer()
    empty_toml = tmp_path / "ds2.toml"
    empty_toml.write_text(f'[[datasets]]\nimage_directory = "{tmp_path}"\ncache_directory = "{tmp_path / "nope"}"\n')
    with pytest.raises(ValueError, match="precache_uncond"):
        trainer2._get_uncond_context(SimpleNamespace(dataset_config=str(empty_toml)), accelerator, torch.float32)
