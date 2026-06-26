import argparse
import contextlib
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakePrintAccelerator:
    def __init__(self):
        self.messages = []

    def print(self, message):
        self.messages.append(str(message))


class _FakeCPUAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


def _sample_boogu_lora_state():
    return {
        "lora_unet_double_stream_blocks_0_img_instruct_attn_to_q.lora_down.weight": torch.ones(2, 4),
        "lora_unet_double_stream_blocks_0_img_instruct_attn_to_q.lora_up.weight": torch.ones(4, 2),
        "lora_unet_double_stream_blocks_0_img_instruct_attn_to_q.alpha": torch.tensor(1.0),
        "lora_unet_single_stream_blocks_3_feed_forward_linear_1.lora_down.weight": torch.ones(2, 4) * 2,
        "lora_unet_single_stream_blocks_3_feed_forward_linear_1.lora_up.weight": torch.ones(4, 2) * 3,
        "lora_unet_single_stream_blocks_3_feed_forward_linear_1.alpha": torch.tensor(2.0),
        "lora_unet_context_refiner_1_attn_to_out_0.lora_down.weight": torch.ones(2, 4) * 4,
        "lora_unet_context_refiner_1_attn_to_out_0.lora_up.weight": torch.ones(4, 2) * 5,
        "lora_unet_context_refiner_1_attn_to_out_0.alpha": torch.tensor(2.0),
    }


def test_boogu_parser_import_registration_and_architecture_constants():
    from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer, boogu_image_setup_parser
    from musubi_tuner.dataset.image_video_dataset import (
        ARCHITECTURE_BOOGU_IMAGE,
        ARCHITECTURE_BOOGU_IMAGE_FULL,
        BucketSelector,
    )

    assert ARCHITECTURE_BOOGU_IMAGE == "bog"
    assert ARCHITECTURE_BOOGU_IMAGE_FULL == "boogu_image"

    bucket = BucketSelector((1024, 1024), enable_bucket=False, architecture=ARCHITECTURE_BOOGU_IMAGE)
    assert bucket.reso_steps == 16

    parser = boogu_image_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--fp8_scaled", "--text_encoder", "/tmp/qwen3-vl"])
    assert args.fp8_scaled is True
    assert args.text_encoder == "/tmp/qwen3-vl"

    trainer = BooguImageNetworkTrainer()
    assert trainer.architecture == ARCHITECTURE_BOOGU_IMAGE
    assert trainer.architecture_full_name == ARCHITECTURE_BOOGU_IMAGE_FULL

    with pytest.raises(ValueError, match="scaled fp8"):
        trainer.handle_model_specific_args(
            types.SimpleNamespace(
                fp8_base=True,
                fp8_scaled=False,
                dit="/tmp/boogu_image_base_bf16.safetensors",
                mixed_precision="bf16",
            )
        )

    with pytest.raises(ValueError, match="torchao"):
        trainer.handle_model_specific_args(
            types.SimpleNamespace(
                fp8_base=False,
                fp8_scaled=False,
                dit="/models/Boogu-Image-0.1-Base-fp8/model.bin",
                mixed_precision="bf16",
            )
        )


def test_boogu_latent_and_text_cache_metadata_and_dtype_keys(tmp_path):
    from musubi_tuner.dataset.image_video_dataset import (
        ARCHITECTURE_BOOGU_IMAGE_FULL,
        ItemInfo,
        save_latent_cache_boogu_image,
        save_text_encoder_output_cache_boogu_image,
    )

    item = ItemInfo("sample", "caption", (512, 384), (512, 384))
    item.latent_cache_path = str(tmp_path / "sample_latents.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "sample_te.safetensors")

    save_latent_cache_boogu_image(item, torch.ones(16, 48, 64, dtype=torch.float32))
    latent_sd = load_file(item.latent_cache_path)
    assert list(latent_sd) == ["latents_48x64_float32"]
    with safe_open(item.latent_cache_path, framework="pt") as f:
        assert f.metadata()["architecture"] == ARCHITECTURE_BOOGU_IMAGE_FULL
        assert f.metadata()["format_version"] == "1.0.1"

    save_text_encoder_output_cache_boogu_image(item, torch.ones(7, 4096, dtype=torch.bfloat16))
    text_sd = load_file(item.text_encoder_output_cache_path)
    assert list(text_sd) == ["varlen_boogu_instruction_embed_bfloat16"]
    assert text_sd["varlen_boogu_instruction_embed_bfloat16"].shape == (7, 4096)
    with safe_open(item.text_encoder_output_cache_path, framework="pt") as f:
        assert f.metadata()["architecture"] == ARCHITECTURE_BOOGU_IMAGE_FULL
        assert f.metadata()["caption1"] == "caption"


def test_boogu_latent_cache_script_preprocesses_and_normalizes_vae_latents():
    import numpy as np

    from musubi_tuner.boogu_image_cache_latents import normalize_boogu_vae_latents, preprocess_contents_boogu_image
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    item = ItemInfo(
        "sample",
        "caption",
        (2, 2),
        (2, 2),
        content=np.array([[[0, 127, 255], [255, 127, 0]], [[64, 128, 192], [192, 128, 64]]], dtype=np.uint8),
    )

    contents = preprocess_contents_boogu_image([item])

    assert contents.shape == (1, 3, 2, 2)
    assert contents.dtype == torch.float32
    assert contents[0, 0, 0, 0].item() == pytest.approx(-1.0)
    assert contents[0, 2, 0, 0].item() == pytest.approx(1.0)

    normalized = normalize_boogu_vae_latents(torch.tensor([[[[1.0]]]]), scaling_factor=0.3611, shift_factor=0.1159)
    assert normalized.item() == pytest.approx((1.0 - 0.1159) * 0.3611)


def test_boogu_single_file_vae_loader_uses_flux_autoencoder_config():
    from musubi_tuner.boogu_image.boogu_utils import BOOGU_AUTOENCODER_KL_SINGLE_FILE_CONFIG, load_boogu_autoencoder_kl

    class _FakeVAE:
        pass

    class _FakeAutoencoderKL:
        calls = []

        @classmethod
        def from_single_file(cls, path, **kwargs):
            cls.calls.append(("single_file", path, kwargs))
            return _FakeVAE()

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append(("pretrained", path, kwargs))
            return _FakeVAE()

    vae = load_boogu_autoencoder_kl("/tmp/ae.safetensors", torch.bfloat16, autoencoder_kl_cls=_FakeAutoencoderKL)

    assert isinstance(vae, _FakeVAE)
    assert len(_FakeAutoencoderKL.calls) == 1
    mode, path, kwargs = _FakeAutoencoderKL.calls[0]
    assert mode == "single_file"
    assert path == "/tmp/ae.safetensors"
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["latent_channels"] == 16
    assert kwargs["sample_size"] == 1024
    assert kwargs["scaling_factor"] == pytest.approx(0.3611)
    assert kwargs["shift_factor"] == pytest.approx(0.1159)
    assert kwargs["use_quant_conv"] is False
    assert kwargs["use_post_quant_conv"] is False
    assert BOOGU_AUTOENCODER_KL_SINGLE_FILE_CONFIG.items() <= kwargs.items()


def test_boogu_text_encoder_load_plan_supports_comfy_qwen3vl_single_file():
    from musubi_tuner.boogu_image_cache_text_encoder_outputs import resolve_boogu_text_encoder_load_plan

    plan = resolve_boogu_text_encoder_load_plan(
        "/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        subfolder="auto",
        processor_path="/models/qwen3-vl-processor",
    )

    assert plan.kind == "comfy_qwen3vl_8b"
    assert plan.model_path == "/models/comfy/text_encoders/qwen3vl_8b_fp8_scaled.safetensors"
    assert plan.processor_path == "/models/qwen3-vl-processor"
    assert plan.model_subfolder is None
    assert plan.processor_subfolder is None


def test_boogu_qwen3vl_8b_config_matches_boogu_instruction_width_and_local_shapes():
    from musubi_tuner.boogu_image_cache_text_encoder_outputs import build_boogu_qwen3vl_8b_config

    config = build_boogu_qwen3vl_8b_config()

    assert config.text_config.hidden_size == 4096
    assert config.text_config.intermediate_size == 12288
    assert config.text_config.num_hidden_layers == 36
    assert config.text_config.num_attention_heads == 32
    assert config.text_config.num_key_value_heads == 8
    assert config.vision_config.hidden_size == 1152
    assert config.vision_config.depth == 27
    assert config.vision_config.intermediate_size == 4304
    assert config.vision_config.out_hidden_size == 4096


def test_boogu_comfy_qwen3vl_fp8_state_dict_conversion_maps_keys_and_preserves_scales():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 support is required for this regression")

    from musubi_tuner.boogu_image_cache_text_encoder_outputs import convert_comfy_qwen3vl_8b_state_dict

    fp8_weight = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": fp8_weight,
        "model.layers.0.self_attn.q_proj.weight_scale": torch.tensor(0.5, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.comfy_quant": torch.tensor([123, 34, 102, 111], dtype=torch.uint8),
        "model.layers.0.input_layernorm.weight": torch.ones(2, dtype=torch.float32),
        "model.visual.patch_embed.proj.weight": torch.ones(1, 1, 1, 1, 1, dtype=torch.float32),
        "lm_head.weight": torch.ones(4, 2, dtype=torch.float32),
    }

    converted = convert_comfy_qwen3vl_8b_state_dict(state_dict, dtype=torch.bfloat16)

    q_proj = "model.language_model.layers.0.self_attn.q_proj"
    assert converted[f"{q_proj}.weight"].dtype is torch.float8_e4m3fn
    assert converted[f"{q_proj}.scale_weight"].dtype is torch.bfloat16
    assert converted[f"{q_proj}.scale_weight"].item() == pytest.approx(0.5)
    assert "model.language_model.layers.0.self_attn.q_proj.weight_scale" not in converted
    assert "model.language_model.layers.0.self_attn.q_proj.comfy_quant" not in converted
    assert converted["model.language_model.layers.0.input_layernorm.weight"].dtype is torch.bfloat16
    assert converted["model.visual.patch_embed.proj.weight"].dtype is torch.bfloat16
    assert converted["lm_head.weight"].dtype is torch.bfloat16


def test_boogu_text_cache_script_encodes_natural_length_instruction_features(tmp_path):
    from musubi_tuner.boogu_image_cache_text_encoder_outputs import encode_and_save_batch
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    class _FakeProcessor:
        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, **kwargs):
            self.messages.append(messages)
            return {
                "input_ids": torch.tensor([[11, 12, 13]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }

    class _FakeEncoder:
        def __call__(self, input_ids, attention_mask):
            hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            return types.SimpleNamespace(last_hidden_state=hidden)

    item = ItemInfo("sample", "draw a sign", (0, 0), (0, 0))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_te.safetensors")
    processor = _FakeProcessor()

    encode_and_save_batch(processor, _FakeEncoder(), [item], device=torch.device("cpu"), dtype=torch.float32, max_length=16)

    saved = load_file(item.text_encoder_output_cache_path)
    assert list(saved) == ["varlen_boogu_instruction_embed_float32"]
    assert saved["varlen_boogu_instruction_embed_float32"].shape == (2, 4)
    assert "high-quality images" in processor.messages[0][0][0]["content"][0]["text"]


def test_boogu_instruction_feature_padding_preserves_natural_lengths():
    from musubi_tuner.boogu_image.boogu_utils import pad_instruction_features

    first = torch.ones(2, 4)
    second = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    padded, mask = pad_instruction_features([first, second], device=torch.device("cpu"), dtype=torch.float32)

    assert padded.shape == (2, 4, 4)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[True, True, False, False], [True, True, True, True]]))
    assert torch.equal(padded[0, :2], first)
    assert torch.equal(padded[0, 2:], torch.zeros(2, 4))
    assert torch.equal(padded[1], second)


def test_boogu_reference_model_modules_are_importable():
    from musubi_tuner.boogu_image import attention_processor, block_lumina2, embeddings, pipeline, rope
    from musubi_tuner.boogu_image.transformer import BooguImageTransformer2DModel

    assert attention_processor.BooguImageAttnProcessor is not None
    assert block_lumina2.LuminaFeedForward is not None
    assert embeddings.TimestepEmbedding is not None
    assert embeddings.apply_rotary_emb is not None
    assert block_lumina2.Lumina2CombinedTimestepCaptionEmbedding is not None
    assert pipeline.run_boogu_transformer is not None
    assert rope.BooguImageDoubleStreamRotaryPosEmbed is not None
    assert rope.get_freqs_cis is not None
    assert BooguImageTransformer2DModel.__name__ == "BooguImageTransformer2DModel"


def test_boogu_time_schedule_and_velocity_sign_conversion():
    from musubi_tuner.boogu_image.boogu_utils import (
        boogu_loss_target,
        boogu_raw_velocity_to_musubi_velocity,
        boogu_time_schedule,
        musubi_timestep_to_boogu_time,
    )

    schedule = boogu_time_schedule(num_steps=4, num_patch_tokens=4096)
    assert schedule.shape == (5,)
    assert schedule[0].item() == pytest.approx(0.0, abs=1e-6)
    assert schedule[-1].item() == pytest.approx(1.0)
    assert torch.all(schedule[1:] > schedule[:-1])

    musubi_timesteps = torch.tensor([1000.0, 250.0, 0.0])
    assert torch.allclose(musubi_timestep_to_boogu_time(musubi_timesteps), torch.tensor([0.0, 0.75, 1.0]))

    raw = torch.tensor([1.0, -2.0])
    assert torch.equal(boogu_raw_velocity_to_musubi_velocity(raw), torch.tensor([-1.0, 2.0]))
    assert torch.equal(boogu_loss_target(torch.tensor([3.0]), torch.tensor([1.0])), torch.tensor([2.0]))


def test_boogu_lora_targets_existing_transformer_linears_without_deleted_projection_assumptions():
    from musubi_tuner.networks.lora_boogu_image import BOOGU_IMAGE_TARGET_REPLACE_MODULES, create_arch_network

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False)])
            # Boogu double-stream attention deletes/omits some projections in places.
            self.to_v = None

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_instruct_attn = _Attention()
            self.feed_forward = nn.Module()
            self.feed_forward.linear_1 = nn.Linear(4, 4, bias=False)

    class BooguImageTransformer2DModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.double_stream_blocks = nn.ModuleList([_Block()])

    assert BOOGU_IMAGE_TARGET_REPLACE_MODULES == ["BooguImageTransformer2DModel"]

    network = create_arch_network(1.0, 2, 2.0, None, [], BooguImageTransformer2DModel())
    names = {lora.lora_name for lora in network.unet_loras}

    assert "lora_unet_double_stream_blocks_0_img_instruct_attn_to_q" in names
    assert "lora_unet_double_stream_blocks_0_img_instruct_attn_to_out_0" in names
    assert "lora_unet_double_stream_blocks_0_feed_forward_linear_1" in names
    assert not any("to_v" in name for name in names)


def test_boogu_lora_key_conversion_and_alpha_scaling():
    from musubi_tuner.boogu_image.convert_lora_to_comfy import convert_lora_name_to_comfy_module, convert_state_dict_to_comfy

    assert (
        convert_lora_name_to_comfy_module("lora_unet_double_stream_blocks_0_img_instruct_attn_to_q")
        == "double_stream_blocks.0.img_instruct_attn.to_q"
    )
    assert (
        convert_lora_name_to_comfy_module("lora_unet_single_stream_blocks_3_feed_forward_linear_1")
        == "single_stream_blocks.3.feed_forward.linear_1"
    )
    assert convert_lora_name_to_comfy_module("lora_unet_context_refiner_1_attn_to_out_0") == "context_refiner.1.attn.to_out.0"

    converted = convert_state_dict_to_comfy(_sample_boogu_lora_state())

    assert "diffusion_model.double_stream_blocks.0.img_instruct_attn.to_q.lora_down.weight" in converted
    assert "diffusion_model.single_stream_blocks.3.feed_forward.linear_1.lora_up.weight" in converted
    assert "diffusion_model.context_refiner.1.attn.to_out.0.lora_up.weight" in converted
    assert not any(key.endswith(".alpha") for key in converted)
    assert torch.allclose(
        converted["diffusion_model.double_stream_blocks.0.img_instruct_attn.to_q.lora_up.weight"],
        torch.ones(4, 2) * 0.5,
    )


def test_boogu_sai_metadata_can_be_built_for_checkpoint_save():
    from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_BOOGU_IMAGE
    from musubi_tuner.utils import sai_model_spec

    metadata = sai_model_spec.build_metadata(
        None,
        ARCHITECTURE_BOOGU_IMAGE,
        0,
        "boogu-test",
        None,
        None,
        None,
        None,
        None,
    )

    assert metadata["modelspec.architecture"] == "Boogu-Image/lora"
    assert metadata["modelspec.implementation"] == "https://github.com/ostris/ai-toolkit"
    assert metadata["modelspec.title"] == "boogu-test"


def test_boogu_post_save_hook_writes_converted_checkpoint_and_respects_save_original_flag(tmp_path):
    from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer

    trainer = BooguImageNetworkTrainer()
    accelerator = _FakePrintAccelerator()
    ckpt_file = tmp_path / "boogu.safetensors"
    save_file(_sample_boogu_lora_state(), str(ckpt_file), metadata={"ss_output_name": "boogu-test"})
    args = types.SimpleNamespace(
        convert_to_comfy=True,
        save_original_lora=False,
        output_dir=str(tmp_path),
        huggingface_repo_id=None,
    )

    trainer.post_save_checkpoint_hook(args, str(ckpt_file), ckpt_file.name, accelerator)

    comfy_file = ckpt_file.with_name("boogu.comfy.safetensors")
    assert comfy_file.exists()
    assert not ckpt_file.exists()
    converted = load_file(str(comfy_file))
    assert "diffusion_model.double_stream_blocks.0.img_instruct_attn.to_q.lora_down.weight" in converted
    with safe_open(str(comfy_file), framework="pt") as f:
        assert f.metadata()["ss_output_name"] == "boogu-test"


def test_boogu_call_dit_uses_cached_instruction_features_and_boogu_time_sign():
    from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer

    class _FakeBooguTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                patch_size=2,
                axes_dim_rope=(2, 2, 2),
                axes_lens=(16, 16, 16),
                in_channels=16,
            )
            self.calls = []

        def forward(
            self,
            hidden_states,
            timestep,
            instruction_hidden_states,
            freqs_cis,
            instruction_attention_mask,
            ref_image_hidden_states=None,
            return_dict=False,
        ):
            self.calls.append(
                {
                    "hidden_states_shape": tuple(hidden_states.shape),
                    "timestep": timestep.detach().clone(),
                    "instruction_shape": tuple(instruction_hidden_states.shape),
                    "mask": instruction_attention_mask.detach().clone(),
                    "hidden_requires_grad": hidden_states.requires_grad,
                    "instruction_requires_grad": instruction_hidden_states.requires_grad,
                    "freqs_count": len(freqs_cis),
                    "ref_image_hidden_states": ref_image_hidden_states,
                    "return_dict": return_dict,
                }
            )
            return torch.full_like(hidden_states, 3.0)

    trainer = BooguImageNetworkTrainer()
    accelerator = _FakeCPUAccelerator()
    transformer = _FakeBooguTransformer()
    latents = torch.ones(2, 16, 4, 4)
    noise = torch.ones_like(latents) * 5
    noisy_model_input = torch.zeros_like(latents)
    batch = {
        "boogu_instruction_embed": [
            torch.ones(2, 4),
            torch.ones(3, 4) * 2,
        ]
    }
    timesteps = torch.tensor([1000.0, 0.0])
    args = types.SimpleNamespace(gradient_checkpointing=True)

    model_pred, target = trainer.call_dit(
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        torch.float32,
    )

    assert torch.equal(model_pred, torch.full_like(latents, -3.0))
    assert torch.equal(target, torch.full_like(latents, 4.0))
    assert len(transformer.calls) == 1
    call = transformer.calls[0]
    assert call["hidden_states_shape"] == (2, 16, 4, 4)
    assert torch.allclose(call["timestep"], torch.tensor([0.0, 1.0]))
    assert call["instruction_shape"] == (2, 3, 4)
    assert torch.equal(call["mask"], torch.tensor([[True, True, False], [True, True, True]]))
    assert call["hidden_requires_grad"] is True
    assert call["instruction_requires_grad"] is True
    assert call["freqs_count"] == 3
    assert call["ref_image_hidden_states"] is None
    assert call["return_dict"] is False


def test_boogu_transformer_gradient_checkpointing_toggle_sets_model_flag():
    from musubi_tuner.boogu_image.transformer import BooguImageTransformer2DModel

    model = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=120,
        num_layers=1,
        num_double_stream_layers=0,
        num_refiner_layers=0,
        num_attention_heads=1,
        num_kv_heads=1,
        multiple_of=8,
        axes_dim_rope=(40, 40, 40),
        axes_lens=(16, 16, 16),
    )

    assert model.gradient_checkpointing is False
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing is True
    assert callable(model._gradient_checkpointing_func)

    calls = []

    def fake_checkpoint(layer, *args):
        calls.append((layer, args))
        return layer(*args) + 1

    model._gradient_checkpointing_func = fake_checkpoint
    x = torch.tensor(2.0, requires_grad=True)
    with torch.enable_grad():
        assert model._ckpt(lambda value: value * 3, x).item() == 7.0
    assert len(calls) == 1

    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    calls.clear()
    with torch.enable_grad():
        assert model._ckpt(lambda value: value * 3, x).item() == 6.0
    assert calls == []


def test_boogu_transformer_exposes_noop_block_swap_sampler_contract():
    from musubi_tuner.boogu_image.transformer import BooguImageTransformer2DModel

    model = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=120,
        num_layers=1,
        num_double_stream_layers=0,
        num_refiner_layers=0,
        num_attention_heads=1,
        num_kv_heads=1,
        multiple_of=8,
        axes_dim_rope=(40, 40, 40),
        axes_lens=(16, 16, 16),
    )

    assert model.blocks_to_swap == 0
    model.switch_block_swap_for_inference()
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_training()
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    assert model.blocks_to_swap == 0


def test_boogu_shared_sampler_wrapper_accepts_transformer_contract(tmp_path):
    from musubi_tuner.boogu_image.transformer import BooguImageTransformer2DModel
    from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer

    class _ContractTrainer(BooguImageNetworkTrainer):
        def __init__(self):
            super().__init__()
            self.sampled = []

        def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
            self.sampled.append(
                {
                    "device": accelerator.device,
                    "save_dir": save_dir,
                    "sample": sample_parameter,
                    "steps": steps,
                }
            )

    trainer = _ContractTrainer()
    accelerator = _FakeCPUAccelerator()
    transformer = BooguImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        hidden_size=120,
        num_layers=1,
        num_double_stream_layers=0,
        num_refiner_layers=0,
        num_attention_heads=1,
        num_kv_heads=1,
        multiple_of=8,
        axes_dim_rope=(40, 40, 40),
        axes_lens=(16, 16, 16),
    )
    transformer.train()
    args = types.SimpleNamespace(
        output_dir=str(tmp_path),
        sample_prompts="p_smoke.toml",
        sample_with_offloading=False,
        sample_blocks_to_swap=None,
        sampling_lora_weight=None,
        compile=False,
    )

    trainer.sample_images(
        accelerator=accelerator,
        args=args,
        epoch=None,
        steps=1,
        vae=object(),
        transformer=transformer,
        sample_parameters=[{"prompt": "lucy", "boogu_instruction_embed": torch.ones(1, 4)}],
        dit_dtype=torch.float32,
        force_sample=True,
    )

    assert len(trainer.sampled) == 1
    assert trainer.sampled[0]["steps"] == 1
    assert transformer.training is True
    assert (tmp_path / "sample").is_dir()


def test_boogu_preview_sampler_uses_cached_embeddings_cfg_and_decodes_pixels():
    from musubi_tuner.boogu_image_train_network import BooguImageNetworkTrainer

    class _FakeBooguTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                patch_size=2,
                axes_dim_rope=(2, 2, 2),
                axes_lens=(16, 16, 16),
                in_channels=16,
            )
            self.calls = []

        def forward(
            self,
            hidden_states,
            timestep,
            instruction_hidden_states,
            freqs_cis,
            instruction_attention_mask,
            ref_image_hidden_states=None,
            return_dict=False,
        ):
            self.calls.append(
                {
                    "hidden_states_shape": tuple(hidden_states.shape),
                    "timestep": float(timestep.item()),
                    "instruction_shape": tuple(instruction_hidden_states.shape),
                    "mask_shape": tuple(instruction_attention_mask.shape),
                }
            )
            scale = 2.0 if instruction_hidden_states.sum() > 0 else 0.5
            return torch.ones_like(hidden_states) * scale

    class _FakeVAE(nn.Module):
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.config = {"scaling_factor": 1.0, "shift_factor": 0.0}
            self.decode_shapes = []

        def decode(self, latents):
            self.decode_shapes.append(tuple(latents.shape))
            bsz, _channels, height, width = latents.shape
            return types.SimpleNamespace(sample=torch.zeros(bsz, 3, height * 8, width * 8))

    trainer = BooguImageNetworkTrainer()
    accelerator = _FakeCPUAccelerator()
    transformer = _FakeBooguTransformer()
    vae = _FakeVAE()
    sample_parameter = {
        "boogu_instruction_embed": torch.ones(2, 4),
        "negative_boogu_instruction_embed": torch.zeros(1, 4),
    }
    generator = torch.Generator(device="cpu").manual_seed(1)

    pixels = trainer.do_inference(
        accelerator=accelerator,
        args=types.SimpleNamespace(),
        sample_parameter=sample_parameter,
        vae=vae,
        dit_dtype=torch.float32,
        transformer=transformer,
        discrete_flow_shift=3.0,
        sample_steps=2,
        width=16,
        height=16,
        frame_count=1,
        generator=generator,
        do_classifier_free_guidance=True,
        guidance_scale=4.0,
        cfg_scale=3.0,
    )

    assert pixels.shape == (1, 3, 1, 16, 16)
    assert torch.allclose(pixels, torch.full_like(pixels, 0.5))
    assert vae.decode_shapes == [(1, 16, 2, 2)]
    assert len(transformer.calls) == 4
    assert {call["instruction_shape"] for call in transformer.calls} == {(1, 2, 4), (1, 1, 4)}
    assert all(call["hidden_states_shape"] == (1, 16, 2, 2) for call in transformer.calls)
    assert transformer.calls[0]["timestep"] == pytest.approx(0.0, abs=1e-6)
    assert transformer.calls[-1]["timestep"] < 1.0
