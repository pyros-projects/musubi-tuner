import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from musubi_tuner.ltx2_generate_video import _build_prompt_list
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer


class _FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()


class _FakePartialState:
    def __init__(self):
        self.num_processes = 1


class _FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.blocks_to_swap = 0
        self.inference_switches = 0
        self.training_switches = 0

    def switch_block_swap_for_inference(self):
        self.inference_switches += 1

    def switch_block_swap_for_training(self):
        self.training_switches += 1


class _RecordingTrainer(LTX2NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.applied = []
        self.restored = []
        self.sampled = []

    def _apply_sampling_lora(self, args, transformer, device):
        self.applied.append((tuple(args.sampling_lora_weight), device.type))
        return [("dummy.safetensors", {})]

    def _restore_sampling_lora(self, args, transformer, device, applied):
        self.restored.append((tuple(args.sampling_lora_weight), len(applied), device.type))

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        vae,
        save_dir,
        sample_parameter,
        epoch,
        steps,
        audio_decoder=None,
        vocoder=None,
    ):
        self.sampled.append(sample_parameter["prompt"])


class _FakeSamplingLora:
    def __init__(self, org_module: torch.nn.Module):
        self.lora_name = "lora_unet_model_linear"
        self.org_module_ref = [org_module]
        self.multiplier = 1.0
        self.scale = 1.0
        self.split_dims = None


class _Float8OverlayTrainer(LTX2NetworkTrainer):
    def _sampling_lora_float8_dtypes(self):
        return (torch.float8_e4m3fn,)


class _FakeCaptionProjection:
    def __init__(self, in_features: int):
        self.linear_1 = type("Linear1", (), {"in_features": in_features})()


class LTX2SamplingLoraTests(unittest.TestCase):
    def test_parse_sample_sigmas_accepts_comma_string(self):
        trainer = LTX2NetworkTrainer()
        parsed = trainer._parse_sample_sigmas_value("1.0, 0.99375, 0.5, 0.0")
        self.assertEqual(parsed, [1.0, 0.99375, 0.5, 0.0])

    def test_apply_sample_defaults_uses_custom_sigma_list_as_default(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            height=512,
            width=768,
            sample_num_frames=45,
            guidance_scale=1.0,
            discrete_flow_shift=5.0,
            sample_sigmas="1.0,0.9,0.5,0.0",
        )

        params = trainer._apply_sample_defaults(args, [{"prompt": "hello"}])

        self.assertEqual(params[0]["sample_sigmas"], [1.0, 0.9, 0.5, 0.0])
        self.assertEqual(params[0]["sample_steps"], 3)

    def test_resolve_sample_sigmas_uses_explicit_values(self):
        trainer = LTX2NetworkTrainer()

        sigmas, effective_steps = trainer._resolve_sample_sigmas(
            {"sample_sigmas": [1.0, 0.9, 0.5, 0.0]},
            sample_steps=20,
            device=torch.device("cpu"),
        )

        self.assertEqual(effective_steps, 3)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.9, 0.5, 0.0], dtype=torch.float32)))

    def test_generator_build_prompt_list_parses_cli_sample_sigmas(self):
        trainer = LTX2NetworkTrainer()
        args = Namespace(
            sample_prompts=None,
            prompt="hello",
            negative_prompt="",
            height=512,
            width=768,
            frame_count=45,
            frame_rate=25.0,
            sample_steps=20,
            sample_sigmas="1.0,0.9,0.5,0.0",
            guidance_scale=1.0,
            discrete_flow_shift=5.0,
            seed=123,
            cfg_scale=None,
        )

        prompts = _build_prompt_list(trainer, args, _FakeAccelerator())

        self.assertEqual(prompts[0]["sample_sigmas"], [1.0, 0.9, 0.5, 0.0])
        self.assertEqual(prompts[0]["sample_steps"], 3)

    def test_sampling_prompt_embeds_trim_to_video_dims_when_audio_preview_disabled(self):
        trainer = LTX2NetworkTrainer()
        transformer = type(
            "FakeTransformer",
            (),
            {
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
                "caption_projection": None,
            },
        )()
        prompt_embeds = torch.randn(2, 8, 6144)

        resolved = trainer._resolve_sampling_prompt_embeds(
            transformer=transformer,
            prompt_embeds=prompt_embeds,
            enable_audio_preview=False,
            audio_ref_only_ic_sampling=False,
        )

        self.assertEqual(resolved.shape, (2, 8, 4096))
        self.assertTrue(torch.equal(resolved, prompt_embeds[..., :4096]))

    def test_sampling_prompt_embeds_keep_full_dims_when_audio_preview_enabled(self):
        trainer = LTX2NetworkTrainer()
        transformer = type(
            "FakeTransformer",
            (),
            {
                "cross_attention_dim": 4096,
                "audio_cross_attention_dim": 2048,
                "caption_projection": _FakeCaptionProjection(4096),
            },
        )()
        prompt_embeds = torch.randn(2, 8, 6144)

        resolved = trainer._resolve_sampling_prompt_embeds(
            transformer=transformer,
            prompt_embeds=prompt_embeds,
            enable_audio_preview=True,
            audio_ref_only_ic_sampling=False,
        )

        self.assertIs(resolved, prompt_embeds)

    def test_nf4_runtime_overlay_avoids_merge_against_packed_weight_shape(self):
        trainer = LTX2NetworkTrainer()
        module = torch.nn.Linear(4, 4, bias=False)
        module._nf4_quantized = True
        module.register_buffer("scale_weight", torch.ones((4, 1, 1), dtype=torch.bfloat16))
        del module.weight
        module.register_buffer("weight", torch.zeros((4, 2), dtype=torch.uint8))
        network = type("Net", (), {"text_encoder_loras": [], "unet_loras": [_FakeSamplingLora(module)]})()
        weights_sd = {
            "lora_unet_model_linear.lora_down.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "lora_unet_model_linear.lora_up.weight": torch.randn(4, 2, dtype=torch.bfloat16),
        }

        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(
            network,
            weights_sd,
            device=torch.device("cpu"),
        )

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(backups, {})

    @unittest.skipIf(not hasattr(torch, "float8_e4m3fn"), "float8 dtype unavailable")
    def test_float8_runtime_overlay_preserves_lora_weight_dtype(self):
        trainer = _Float8OverlayTrainer()
        module = torch.nn.Linear(4, 4, bias=False)
        module.weight = torch.nn.Parameter(module.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
        network = type("Net", (), {"text_encoder_loras": [], "unet_loras": [_FakeSamplingLora(module)]})()
        weights_sd = {
            "lora_unet_model_linear.lora_down.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "lora_unet_model_linear.lora_up.weight": torch.randn(4, 2, dtype=torch.bfloat16),
        }

        merged, runtime_attached, backups = trainer._apply_sampling_lora_network(
            network,
            weights_sd,
            device=torch.device("cpu"),
        )

        self.assertEqual(merged, 0)
        self.assertEqual(runtime_attached, 1)
        self.assertEqual(backups, {})
        adapter = module._sampling_lora_runtime_adapters[0]
        self.assertEqual(getattr(module, adapter["A"]).dtype, torch.bfloat16)
        self.assertEqual(getattr(module, adapter["B"]).dtype, torch.bfloat16)

    def test_runtime_overlay_weights_follow_module_device_moves(self):
        trainer = LTX2NetworkTrainer()
        module = torch.nn.Linear(4, 4, bias=False)

        trainer._attach_sampling_lora_runtime_overlay(
            module,
            torch.randn(2, 4),
            torch.randn(4, 2),
            0.6,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
        )

        adapter = module._sampling_lora_runtime_adapters[0]
        self.assertIsInstance(adapter["A"], str)
        self.assertIsInstance(adapter["B"], str)
        self.assertEqual(getattr(module, adapter["A"]).device.type, "cpu")
        self.assertEqual(getattr(module, adapter["B"]).device.type, "cpu")

        module.to("meta")

        self.assertEqual(getattr(module, adapter["A"]).device.type, "meta")
        self.assertEqual(getattr(module, adapter["B"]).device.type, "meta")

    def test_normalize_sampling_lora_weights_converts_comfy_ltx_keys(self):
        trainer = LTX2NetworkTrainer()
        weights_sd = {
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.transformer_blocks.0.attn1.to_k.lora_B.weight": torch.randn(4, 2),
            "diffusion_model.transformer_blocks.0.attn1.to_k.alpha": torch.tensor(2.0),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.lora_A.weight": torch.randn(2, 4),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.lora_B.weight": torch.randn(4, 2),
            "diffusion_model.transformer_blocks.0.audio_attn2.to_out.0.alpha": torch.tensor(2.0),
        }

        normalized = trainer.normalize_sampling_lora_weights(
            Namespace(network_module="musubi_tuner.networks.lora_ltx2"),
            weights_sd,
            "dummy.safetensors",
        )

        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.lora_up.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_attn1_to_k.alpha",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_audio_attn2_to_out_0.lora_down.weight",
            normalized,
        )
        self.assertIn(
            "lora_unet_model_transformer_blocks_0_audio_attn2_to_out_0.lora_up.weight",
            normalized,
        )

    def test_sample_images_temporarily_applies_sampling_lora(self):
        trainer = _RecordingTrainer()
        accelerator = _FakeAccelerator()
        transformer = _FakeTransformer()
        args = Namespace(
            sample_at_first=True,
            sample_every_n_steps=None,
            sample_every_n_epochs=None,
            sample_prompts="dummy.txt",
            output_dir=tempfile.mkdtemp(prefix="ltx2-sample-lora-"),
            sample_with_offloading=False,
            use_precached_sample_prompts=False,
            precache_sample_prompts=False,
            sample_audio_subprocess=True,
            sample_disable_audio=False,
            sample_audio_only=False,
            ltx_mode="video",
            vae_dtype=None,
            sampling_lora_weight=["/tmp/sample-lora.safetensors"],
            sampling_lora_multiplier=[0.8],
            compile=False,
        )
        sample_parameters = [{"prompt": "hello world", "enum": 0}]

        with mock.patch("musubi_tuner.ltx2_train_network.PartialState", _FakePartialState), mock.patch(
            "musubi_tuner.ltx2_train_network.clean_memory_on_device", lambda device: None
        ):
            trainer.sample_images(
                accelerator=accelerator,
                args=args,
                epoch=0,
                steps=0,
                vae=object(),
                transformer=transformer,
                sample_parameters=sample_parameters,
                dit_dtype=torch.bfloat16,
            )

        self.assertEqual(trainer.applied, [(("/tmp/sample-lora.safetensors",), "cpu")])
        self.assertEqual(trainer.restored, [(("/tmp/sample-lora.safetensors",), 1, "cpu")])
        self.assertEqual(trainer.sampled, ["hello world"])
        self.assertEqual(transformer.inference_switches, 1)
        self.assertEqual(transformer.training_switches, 1)


if __name__ == "__main__":
    unittest.main()
