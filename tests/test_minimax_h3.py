from contextlib import contextmanager, nullcontext
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

from musubi_tuner import minimax_h3_cache_latents
from musubi_tuner import minimax_h3_cache_text_encoder_outputs
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.minimax_h3_utils import (
    align_frame_count,
    audio_latent_length,
    inspect_transformer_checkpoint,
    is_valid_frame_count,
    load_selected_weights,
    video_latent_length,
)
from musubi_tuner.minimax_h3.model import (
    MiniMaxH3Model,
    pack_audio,
    patchify_video,
    time_shift_sigma,
    time_shift_slope,
    unpack_audio,
    unpatchify_video,
)
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer
from musubi_tuner.minimax_h3_train_network import _apply_h3_lora_target_preset
from musubi_tuner.minimax_h3_train_network import minimax_h3_setup_parser
from musubi_tuner.networks import lora
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.modules.int8_optimization_utils import (
    Int8ConvRotConfig,
    _convrot_hadamard,
    apply_int8_convrot_monkey_patch,
)


def test_h3_frame_and_audio_alignment():
    assert align_frame_count(5) == 5
    assert align_frame_count(21) == 5
    assert align_frame_count(22) == 22
    assert align_frame_count(23, "up") == 39
    assert is_valid_frame_count(39)
    assert not is_valid_frame_count(38)
    assert video_latent_length(5) == 2
    assert video_latent_length(22) == 7
    assert video_latent_length(39) == 12
    assert audio_latent_length(24) == 40
    assert BucketSelector((512, 512), architecture=ARCHITECTURE_MINIMAX_H3).reso_steps == 32


def test_h3_time_shift_slope_matches_derivative():
    sigma = torch.linspace(0.05, 0.95, 19, dtype=torch.float64)
    epsilon = 1e-6
    numerical = (time_shift_sigma(sigma + epsilon, 12.0, 3.0) - time_shift_sigma(sigma - epsilon, 12.0, 3.0)) / (2 * epsilon)
    torch.testing.assert_close(time_shift_slope(sigma, 12.0, 3.0), numerical, rtol=1e-6, atol=1e-7)


def test_h3_pack_round_trips():
    video = torch.randn(2, 3, 2, 6, 8)
    rows = patchify_video(video)
    restored = unpatchify_video(rows, 2, 3, 4, 3)
    torch.testing.assert_close(restored, video)

    audio = torch.randn(2, 4, 2, 7)
    torch.testing.assert_close(unpack_audio(pack_audio(audio)), audio)


def test_tiny_h3_forward_and_backward():
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
    video = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    audio = torch.randn(1, 3, 2, 3, requires_grad=True)
    context = torch.randn(1, 4, 8, requires_grad=True)
    video_out, audio_out = model(video, audio, torch.tensor([0.7]), context)
    assert video_out.shape == video.shape
    assert audio_out.shape == audio.shape
    (video_out.square().mean() + audio_out.square().mean()).backward()
    assert video.grad is not None
    assert audio.grad is not None
    assert context.grad is not None

    image = torch.randn(1, 2, 1, 4, 4, requires_grad=True)
    no_audio = torch.empty(1, 3, 2, 0, requires_grad=True)
    image_context = torch.randn(1, 4, 8, requires_grad=True)
    image_out, audio_out = model(image, no_audio, torch.tensor([0.7]), image_context)
    assert image_out.shape == image.shape
    assert audio_out.shape == no_audio.shape
    image_out.square().mean().backward()
    assert image.grad is not None
    assert image_context.grad is not None


def test_h3_parser_preserves_explicit_zero_sample_blocks_to_swap():
    parser = setup_parser_common()

    assert parser.parse_args([]).sample_blocks_to_swap is None
    assert parser.parse_args(["--sample_blocks_to_swap", "0"]).sample_blocks_to_swap == 0


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(["--sample_blocks_to_swap", "-1", "--blocks_to_swap", "2"], id="negative"),
        pytest.param(["--sample_blocks_to_swap", "49", "--blocks_to_swap", "2"], id="over-48"),
        pytest.param(["--sample_blocks_to_swap", "1"], id="training-unset"),
        pytest.param(["--sample_blocks_to_swap", "1", "--blocks_to_swap", "0"], id="training-zero"),
    ],
)
def test_h3_rejects_invalid_sampling_block_swap_at_startup(options):
    args = minimax_h3_setup_parser(setup_parser_common()).parse_args(options)

    with pytest.raises(ValueError, match="sample_blocks_to_swap"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_sampling_block_swap_override_restores_after_error_and_normal_exit():
    class FakeOffloader:
        def __init__(self):
            self.blocks_to_swap = 2
            self.forward_only = False
            self.mode_calls = []
            self.prepare_calls = []

        def set_forward_only(self, value):
            self.mode_calls.append(value)
            self.forward_only = value

        def prepare_block_devices_before_forward(self, blocks):
            self.prepare_calls.append((self.blocks_to_swap, len(blocks)))

    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList(torch.nn.Linear(1, 1) for _ in range(4))
    model.blocks_to_swap = 2
    model.offloader = FakeOffloader()
    placement_counts = []
    move_to_device = model.move_to_device_except_swap_blocks

    def track_model_placement(device):
        placement_counts.append(model.blocks_to_swap)
        move_to_device(device)

    model.move_to_device_except_swap_blocks = track_model_placement

    with pytest.raises(RuntimeError, match="boom"):
        with model.override_block_swap_for_sampling(0, torch.device("cpu")) as effective_count:
            assert effective_count == 0
            assert model.blocks_to_swap == 0
            assert model.offloader.blocks_to_swap == 0
            raise RuntimeError("boom")

    assert model.blocks_to_swap == 2
    assert model.offloader.blocks_to_swap == 2
    assert model.offloader.mode_calls == [True, False]
    assert model.offloader.prepare_calls[-1] == (2, 4)
    assert placement_counts == [0]

    with model.override_block_swap_for_sampling(1, torch.device("cpu")) as effective_count:
        assert effective_count == 1
        assert model.blocks_to_swap == 1
        assert model.offloader.blocks_to_swap == 1
        assert model.offloader.forward_only
        assert model.offloader.prepare_calls[-1] == (1, 4)

    assert model.blocks_to_swap == 2
    assert model.offloader.blocks_to_swap == 2
    assert not model.offloader.forward_only
    assert model.offloader.mode_calls == [True, False, True, False]
    assert model.offloader.prepare_calls == [(2, 4), (1, 4), (2, 4)]


@pytest.mark.parametrize(
    ("sample_blocks_to_swap", "expected_events"),
    [
        (None, ["inference_mode", "sample", "training_mode"]),
        (0, [("enter", 0, "cpu"), "sample", ("exit", 0)]),
    ],
)
def test_h3_sample_images_restores_sampling_block_swap_after_failure(
    monkeypatch, tmp_path, sample_blocks_to_swap, expected_events
):
    events = []

    class FakeState:
        num_processes = 1

    class FakeAccelerator:
        device = torch.device("cpu")

        @staticmethod
        def unwrap_model(transformer):
            return transformer

        @staticmethod
        def autocast():
            return nullcontext()

    class FakeTransformer:
        blocks_to_swap = 2

        @staticmethod
        def switch_block_swap_for_inference():
            events.append("inference_mode")

        @staticmethod
        def switch_block_swap_for_training():
            events.append("training_mode")

        @contextmanager
        def override_block_swap_for_sampling(self, count, device):
            events.append(("enter", count, device.type))
            try:
                yield count
            finally:
                events.append(("exit", count))

    trainer = MiniMaxH3NetworkTrainer()

    def fail_sample(*_args, **_kwargs):
        events.append("sample")
        raise RuntimeError("sample failed")

    trainer.sample_image_inference = fail_sample
    monkeypatch.setattr("musubi_tuner.training.trainer_base.PartialState", lambda: FakeState())
    args = SimpleNamespace(
        sample_at_first=False,
        sample_every_n_steps=1,
        sample_every_n_epochs=None,
        sample_prompts="samples.toml",
        sample_blocks_to_swap=sample_blocks_to_swap,
        output_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="sample failed"):
        trainer.sample_images(FakeAccelerator(), args, None, 1, None, FakeTransformer(), [{}], torch.float32)

    assert events == expected_events


@pytest.mark.parametrize(
    ("preset", "expected_modules"),
    [("attn", 6), ("attn_mlp", 12), ("no_packed_attn", 8), ("full", 14)],
)
def test_h3_lora_target_presets_select_expected_modules(preset, expected_modules):
    args = SimpleNamespace(lora_target_preset=preset, network_args=None)
    _apply_h3_lora_target_preset(args)
    network_kwargs = dict(arg.split("=", 1) for arg in (args.network_args or []))
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

    network = lora.create_arch_network(1.0, 4, 4, None, None, model, **network_kwargs)

    assert len(network.unet_loras) == expected_modules
    if preset == "no_packed_attn":
        assert {module.lora_name for module in network.unet_loras} == {
            "lora_unet_token_refiner_blocks_0_attn_qkv_proj",
            "lora_unet_token_refiner_blocks_0_attn_out_proj",
            "lora_unet_token_refiner_blocks_0_mlp_fc1",
            "lora_unet_token_refiner_blocks_0_mlp_fc2",
            "lora_unet_blocks_0_mlp_fc1",
            "lora_unet_blocks_0_mlp_fc2",
            "lora_unet_blocks_1_mlp_fc1",
            "lora_unet_blocks_1_mlp_fc2",
        }


def test_h3_image_only_latent_cache(monkeypatch):
    item = ItemInfo(
        "lucy.jpg",
        "lucy the cat",
        (32, 32),
        content=np.zeros((32, 32, 3), dtype=np.uint8),
    )

    class FakeVideoVae:
        device = torch.device("cpu")
        dtype = torch.float32

        @staticmethod
        def encode(content):
            assert content.shape == (1, 3, 1, 32, 32)
            return torch.zeros(1, 24, 1, 2, 2)

    saved = {}

    def save_cache(_item, video_latent, audio_latent):
        saved["video"] = video_latent
        saved["audio"] = audio_latent

    monkeypatch.setattr(minimax_h3_cache_latents, "save_latent_cache_minimax_h3", save_cache)
    minimax_h3_cache_latents.encode_and_save_batch(FakeVideoVae(), None, [item])

    assert saved["video"].shape == (24, 1, 2, 2)
    assert saved["audio"].shape == (32, 2, 0)


def test_h3_image_latent_cache_can_repeat_five_frames(monkeypatch):
    item = ItemInfo(
        "lucy.jpg",
        "lucy the cat",
        (32, 32),
        content=np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3),
    )

    class FakeVideoVae:
        device = torch.device("cpu")
        dtype = torch.float32

        @staticmethod
        def encode(content):
            assert content.shape == (1, 3, 5, 32, 32)
            torch.testing.assert_close(content, content[:, :, :1].expand_as(content))
            return torch.zeros(1, 24, 2, 2, 2)

    saved = {}

    def save_cache(_item, video_latent, audio_latent):
        saved["frame_count"] = _item.frame_count
        saved["video"] = video_latent
        saved["audio"] = audio_latent

    monkeypatch.setattr(minimax_h3_cache_latents, "save_latent_cache_minimax_h3", save_cache)
    minimax_h3_cache_latents.encode_and_save_batch(FakeVideoVae(), None, [item], image_frame_count=5)

    assert saved["frame_count"] == 5
    assert saved["video"].shape == (24, 2, 2, 2)
    assert saved["audio"].shape == (32, 2, 0)


@pytest.mark.parametrize(
    ("latent_frames", "image_audio_mode", "audio_frames"),
    [(1, "none", 0), (2, "none", 0), (1, "silent", 2), (2, "silent", 8)],
)
def test_h3_image_only_loss_is_finite(latent_frames, image_audio_mode, audio_frames):
    trainer = MiniMaxH3NetworkTrainer()
    trainer.get_noisy_model_input_and_timesteps = lambda *args, **kwargs: (torch.tensor([0.5]), None)
    seen = {}

    class FakeAccelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

    class FakeTransformer:
        @staticmethod
        def __call__(video, audio, sigma_video, contexts, tags):
            seen["audio_shape"] = audio.shape
            return torch.zeros_like(video), torch.empty_like(audio)

    latents = torch.randn(1, 2, latent_frames, 4, 4)
    noise = torch.randn_like(latents)
    batch = {
        "latents_audio": torch.empty(1, 32, 2, 0),
        "h3_text_embed": [torch.zeros(2, 8)],
        "timesteps": None,
    }
    args = SimpleNamespace(
        audio_flow_shift=3.0,
        audio_loss_weight=0.0,
        gradient_checkpointing=False,
        image_audio_mode=image_audio_mode,
        video_flow_shift=12.0,
        weighting_scheme="none",
    )

    loss, metrics = trainer.process_batch(
        args,
        FakeAccelerator(),
        FakeTransformer(),
        None,
        batch,
        latents,
        noise,
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, (latents - noise).square().mean())
    assert seen["audio_shape"] == (1, 32, 2, audio_frames)
    assert metrics["loss/audio"] == 0.0

    args.audio_loss_weight = 1.0
    with pytest.raises(ValueError, match="audio_loss_weight 0"):
        trainer.process_batch(
            args,
            FakeAccelerator(),
            FakeTransformer(),
            None,
            batch,
            latents,
            noise,
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )


def test_h3_comfy_text_cache_reuses_identical_captions(monkeypatch):
    items = [ItemInfo(f"lucy-{index}.jpg", "lucy the cat", (32, 32)) for index in range(2)]

    class FakeClip:
        calls = 0

        @staticmethod
        def tokenize(prompt):
            assert prompt == "lucy the cat"
            return {"qwen3vl_32b": [[(index, 1.0) for index in range(4)]]}

        def encode_from_tokens(self, tokens, return_dict):
            assert return_dict
            assert len(tokens["qwen3vl_32b"][0]) == 2
            self.calls += 1
            return {
                "cond": torch.zeros(1, 2, 8),
                "minimax_token_tags": torch.ones(2, dtype=torch.long),
            }

    saved = []
    monkeypatch.setattr(
        minimax_h3_cache_text_encoder_outputs,
        "save_text_encoder_output_cache_minimax_h3",
        lambda item, embed, tags: saved.append((item, embed, tags)),
    )
    clip = FakeClip()
    minimax_h3_cache_text_encoder_outputs.encode_and_save_batch_comfy(clip, items, {}, 2)

    assert clip.calls == 1
    assert len(saved) == 2
    assert all(embed.shape == (2, 8) and tags.shape == (2,) for _, embed, tags in saved)


def test_h3_comfy_loader_forwards_device(monkeypatch, tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / "comfy").mkdir(parents=True)
    (comfy_root / "comfy" / "sd.py").touch()

    calls = []
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy_sd = ModuleType("comfy.sd")
    comfy_sd.CLIPType = SimpleNamespace(MINIMAX="minimax")
    comfy_sd.load_clip = lambda **kwargs: calls.append(kwargs) or "clip"
    comfy.sd = comfy_sd
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)

    device = torch.device("cuda:1")
    assert minimax_h3_cache_text_encoder_outputs.load_comfy_clip(str(comfy_root), "qwen.safetensors", device) == "clip"
    assert calls == [
        {
            "ckpt_paths": ["qwen.safetensors"],
            "clip_type": "minimax",
            "model_options": {"load_device": device},
        }
    ]

    with pytest.raises(ValueError, match="ComfyUI source not found"):
        minimax_h3_cache_text_encoder_outputs.load_comfy_clip(str(tmp_path / "missing"), "qwen.safetensors", device)


def test_h3_prompt_cache_hit_skips_text_encoder_load(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        f'''
[general]
resolution = [512, 512]

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
'''.strip(),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompts.toml"
    prompt_path.write_text("""[[prompt.subset]]\nprompt = "lucy the cat"\n""", encoding="utf-8")
    text_encoder_path = tmp_path / "qwen.safetensors"
    text_encoder_path.write_bytes(b"fake")
    text_encoder_stat = text_encoder_path.stat()
    torch.save(
        {
            "version": 1,
            "architecture": "minimax_h3",
            "text_encoder": str(text_encoder_path.resolve()),
            "text_encoder_size": text_encoder_stat.st_size,
            "text_encoder_mtime_ns": text_encoder_stat.st_mtime_ns,
            "max_token_length": 1024,
            "prompt_cache": [
                {
                    "prompt": "lucy the cat",
                    "h3_text_embed": torch.zeros(2, 8),
                    "h3_token_tags": torch.ones(2, dtype=torch.long),
                }
            ],
        },
        cache_dir / "minimax_h3_sample_prompts_cache.pt",
    )
    loads = []
    monkeypatch.setattr(
        minimax_h3_cache_text_encoder_outputs,
        "load_comfy_clip",
        lambda *args, **kwargs: loads.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimax_h3_cache_text_encoder_outputs.py",
            "--dataset_config",
            str(dataset_path),
            "--text_encoder",
            str(text_encoder_path),
            "--comfyui_path",
            str(tmp_path / "ComfyUI"),
            "--device",
            "cpu",
            "--precache_sample_prompts",
            "--sample_prompts",
            str(prompt_path),
            "--cache_sample_prompts_only",
        ],
    )

    minimax_h3_cache_text_encoder_outputs.main()

    assert loads == []


def test_h3_existing_dataset_text_cache_skips_text_encoder_load(monkeypatch, tmp_path):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    Image.new("RGB", (32, 32)).save(image_dir / "lucy.png")
    (image_dir / "lucy.txt").write_text("lucy the cat", encoding="utf-8")
    save_file({"cached": torch.zeros(1)}, cache_dir / "lucy_h3_te.safetensors", metadata={"caption1": "lucy the cat"})
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        f'''
[general]
resolution = [32, 32]
caption_extension = ".txt"
batch_size = 1
enable_bucket = false

[[datasets]]
image_directory = "{image_dir}"
cache_directory = "{cache_dir}"
'''.strip(),
        encoding="utf-8",
    )
    text_encoder_path = tmp_path / "qwen.safetensors"
    text_encoder_path.write_bytes(b"fake")
    loads = []
    monkeypatch.setattr(
        minimax_h3_cache_text_encoder_outputs,
        "load_comfy_clip",
        lambda *args, **kwargs: loads.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimax_h3_cache_text_encoder_outputs.py",
            "--dataset_config",
            str(dataset_path),
            "--text_encoder",
            str(text_encoder_path),
            "--comfyui_path",
            str(tmp_path / "ComfyUI"),
            "--device",
            "cpu",
            "--skip_existing",
        ],
    )

    minimax_h3_cache_text_encoder_outputs.main()

    assert loads == []


def test_h3_prompt_cache_miss_loads_encoder_once(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        f'''[[datasets]]\nimage_directory = "{tmp_path}"\ncache_directory = "{cache_dir}"\n''',
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompts.toml"
    prompt_path.write_text(
        """
[[prompt.subset]]
prompt = "lucy the cat"

[[prompt.subset]]
prompt = "a portrait of lucy"
""".strip(),
        encoding="utf-8",
    )
    text_encoder_path = tmp_path / "qwen.safetensors"
    text_encoder_path.write_bytes(b"fake")

    class FakeClip:
        prompts = []

        def tokenize(self, prompt):
            self.prompts.append(prompt)
            return {"qwen3vl_32b": [[(0, 1.0), (1, 1.0)]]}

        @staticmethod
        def encode_from_tokens(tokens, return_dict):
            assert return_dict
            return {
                "cond": torch.zeros(1, 2, 8),
                "minimax_token_tags": torch.ones(2, dtype=torch.long),
            }

    clip = FakeClip()
    loads = []
    monkeypatch.setattr(
        minimax_h3_cache_text_encoder_outputs,
        "load_comfy_clip",
        lambda *args, **kwargs: loads.append((args, kwargs)) or clip,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "minimax_h3_cache_text_encoder_outputs.py",
            "--dataset_config",
            str(dataset_path),
            "--text_encoder",
            str(text_encoder_path),
            "--comfyui_path",
            str(tmp_path / "ComfyUI"),
            "--device",
            "cpu",
            "--precache_sample_prompts",
            "--sample_prompts",
            str(prompt_path),
            "--cache_sample_prompts_only",
        ],
    )

    minimax_h3_cache_text_encoder_outputs.main()

    assert len(loads) == 1
    assert clip.prompts == ["lucy the cat", "a portrait of lucy"]
    cache = torch.load(cache_dir / "minimax_h3_sample_prompts_cache.pt", weights_only=True)
    assert [entry["prompt"] for entry in cache["prompt_cache"]] == ["lucy the cat", "a portrait of lucy"]


def test_int8_convrot_linear_forward_and_backward():
    torch.manual_seed(123)
    layer = torch.nn.Linear(16, 12)
    original_weight = layer.weight.detach().clone()
    rotated_weight = _convrot_hadamard(original_weight, 16)
    weight_scale = (rotated_weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    quantized_weight = (rotated_weight / weight_scale).round().clamp(-128, 127).to(torch.int8)

    apply_int8_convrot_monkey_patch(layer, {"": Int8ConvRotConfig(16, tuple(weight_scale.shape))})
    layer.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
    layer.weight_scale.copy_(weight_scale)
    assert layer.__class__.__name__ == "Linear"  # LoRA discovery relies on the exact class name.

    dequantized_weight = _convrot_hadamard(quantized_weight.float() * weight_scale, 16)
    value = torch.randn(5, 16, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    probe = torch.randn(5, 12)

    output = layer(value)
    reference = F.linear(reference_value, dequantized_weight, layer.bias)
    torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)

    (output * probe).sum().backward()
    (reference * probe).sum().backward()
    torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-5, atol=1e-5)

    bf16_value = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
    bf16_output = layer(bf16_value)
    assert bf16_output.dtype == torch.bfloat16
    bf16_output.float().sum().backward()
    assert bf16_value.grad is not None


def test_h3_int8_checkpoint_metadata_and_pruned_curve(tmp_path):
    marker = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}).encode()),
        dtype=torch.uint8,
    )
    checkpoint = tmp_path / "h3_int8.safetensors"
    save_file(
        {
            "fc.weight": torch.zeros(8, 64, dtype=torch.int8),
            "fc.weight_scale": torch.ones(8, 1),
            "fc.comfy_quant": marker,
            "adaln_t_table": torch.zeros(17, 4),
        },
        checkpoint,
    )

    layout = inspect_transformer_checkpoint([checkpoint])
    assert layout.adaln_curve_grid == 17
    assert layout.adaln_curve_dim == 4
    assert layout.int8_convrot_layers["fc"] == Int8ConvRotConfig(64, (8, 1))

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(64, 8, bias=False)

    with torch.device("meta"):
        model = TinyModel()
    apply_int8_convrot_monkey_patch(model, layout.int8_convrot_layers)
    load_selected_weights(
        model,
        [checkpoint],
        device="cpu",
        dtype=lambda key: torch.int8 if key == "fc.weight" else torch.float32,
    )
    assert model.fc.weight.dtype == torch.int8
    value = torch.randn(2, 64, requires_grad=True)
    model.fc(value).sum().backward()
    assert value.grad is not None


def test_tiny_pruned_h3_forward_and_backward():
    model = MiniMaxH3Model(
        hidden_size=12,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=6,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=3,
        text_dim=8,
        time_embed_dim=4,
        adaln_curve_grid=9,
        rope_inv_freq_len=1,
    )
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
        model.adaln_t_table.normal_()
    video = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    audio = torch.randn(1, 3, 2, 3, requires_grad=True)
    context = torch.randn(1, 4, 8, requires_grad=True)
    video_out, audio_out = model(video, audio, torch.tensor([0.7]), context)
    (video_out.square().mean() + audio_out.square().mean()).backward()
    assert video.grad is not None
    assert audio.grad is not None
    assert context.grad is not None


def test_h3_lora_comfy_module_name_conversion():
    from musubi_tuner.minimax_h3.convert_lora_to_comfy import convert_module_name_to_comfy

    cases = {
        "lora_unet_blocks_0_mlp_fc1": "blocks.0.mlp.fc1",
        "lora_unet_blocks_49_mlp_fc2": "blocks.49.mlp.fc2",
        "lora_unet_blocks_12_attn_qkv_proj": "blocks.12.attn.qkv_proj",
        "lora_unet_blocks_7_attn_out_proj": "blocks.7.attn.out_proj",
        "lora_unet_blocks_3_adaln_proj_linear": "blocks.3.adaln_proj.linear",
        "lora_unet_token_refiner_blocks_1_attn_qkv_proj": "token_refiner.blocks.1.attn.qkv_proj",
        "lora_unet_token_refiner_blocks_0_mlp_fc2": "token_refiner.blocks.0.mlp.fc2",
    }
    for kohya, comfy in cases.items():
        assert convert_module_name_to_comfy(kohya) == comfy


def test_h3_lora_comfy_conversion_folds_alpha(tmp_path):
    from musubi_tuner.minimax_h3.convert_lora_to_comfy import convert_lora_to_comfy
    from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

    rank, h = 4, 8
    torch.manual_seed(0)
    sd = {}
    # alpha == rank: weights must pass through unchanged
    sd["lora_unet_blocks_0_mlp_fc1.lora_down.weight"] = torch.randn(rank, h, dtype=torch.bfloat16)
    sd["lora_unet_blocks_0_mlp_fc1.lora_up.weight"] = torch.randn(h, rank, dtype=torch.bfloat16)
    sd["lora_unet_blocks_0_mlp_fc1.alpha"] = torch.tensor(float(rank))
    # alpha == rank/2: lora_B must come out scaled by 0.5
    sd["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_down.weight"] = torch.randn(rank, h, dtype=torch.bfloat16)
    sd["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_up.weight"] = torch.randn(h, rank, dtype=torch.bfloat16)
    sd["lora_unet_token_refiner_blocks_1_attn_qkv_proj.alpha"] = torch.tensor(rank / 2.0)

    src = tmp_path / "lora.safetensors"
    dst = tmp_path / "lora.comfy.safetensors"
    save_file(sd, str(src), metadata={"ss_network_dim": "4"})

    assert convert_lora_to_comfy(str(src), str(dst)) == 2

    with MemoryEfficientSafeOpen(str(dst)) as f:
        out = {k: f.get_tensor(k) for k in f.keys()}
        metadata = f.metadata()

    assert sorted(out) == [
        "diffusion_model.blocks.0.mlp.fc1.lora_A.weight",
        "diffusion_model.blocks.0.mlp.fc1.lora_B.weight",
        "diffusion_model.token_refiner.blocks.1.attn.qkv_proj.lora_A.weight",
        "diffusion_model.token_refiner.blocks.1.attn.qkv_proj.lora_B.weight",
    ]
    assert not any(".alpha" in k for k in out)
    torch.testing.assert_close(out["diffusion_model.blocks.0.mlp.fc1.lora_A.weight"], sd["lora_unet_blocks_0_mlp_fc1.lora_down.weight"])
    torch.testing.assert_close(out["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"], sd["lora_unet_blocks_0_mlp_fc1.lora_up.weight"])
    expected_scaled = (sd["lora_unet_token_refiner_blocks_1_attn_qkv_proj.lora_up.weight"].float() * 0.5).to(torch.bfloat16)
    torch.testing.assert_close(out["diffusion_model.token_refiner.blocks.1.attn.qkv_proj.lora_B.weight"], expected_scaled)
    assert metadata["ss_network_dim"] == "4"
    assert metadata["ss_comfy_converted"] == "minimax_h3"


def test_h3_on_post_save_writes_comfy_twin(tmp_path):
    trainer = MiniMaxH3NetworkTrainer()

    rank, h = 2, 4
    sd = {
        "lora_unet_blocks_5_mlp_fc2.lora_down.weight": torch.randn(rank, h, dtype=torch.bfloat16),
        "lora_unet_blocks_5_mlp_fc2.lora_up.weight": torch.randn(h, rank, dtype=torch.bfloat16),
        "lora_unet_blocks_5_mlp_fc2.alpha": torch.tensor(float(rank)),
    }
    ckpt_name = "demo-step00000010.safetensors"
    save_file(sd, str(tmp_path / ckpt_name))

    args = SimpleNamespace(output_dir=str(tmp_path), no_convert_to_comfy=False, huggingface_repo_id=None)
    accelerator = SimpleNamespace(print=lambda *a, **k: None)
    trainer.on_post_save(args, accelerator, None, None, ckpt_name, torch.bfloat16, {}, False)
    assert (tmp_path / "demo-step00000010.comfy.safetensors").exists()

    args.no_convert_to_comfy = True
    trainer.on_post_save(args, accelerator, None, None, "other-step00000020.safetensors", torch.bfloat16, {}, False)
    assert not (tmp_path / "other-step00000020.comfy.safetensors").exists()
