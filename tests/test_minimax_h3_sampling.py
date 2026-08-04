import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.minimax_h3.video_vae import decode_single_frame_latent
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


def test_h3_sample_prompt_requires_single_frame(tmp_path):
    prompt_path = tmp_path / "samples.json"
    prompt_path.write_text(json.dumps([{"frame_count": 5}]), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_count=1"):
        MiniMaxH3NetworkTrainer().process_sample_prompts(SimpleNamespace(), None, str(prompt_path))


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

        def decode(self, latents):
            return torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)

    pixels = MiniMaxH3NetworkTrainer().do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(video_flow_shift=12.0),
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

        def decode(self, latents):
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
            SimpleNamespace(video_flow_shift=12.0),
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
