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
