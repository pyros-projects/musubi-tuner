from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from musubi_tuner import zimage_generate_image
from musubi_tuner.modules.fp8_optimization_utils import fp8_linear_forward_patch


class _DummyZImageModel:
    def __init__(self):
        self.noise_refiner = object()
        self.context_refiner = object()
        self.layers = object()

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def requires_grad_(self, _flag):
        return self


class ZImageGenerateImageTest(unittest.TestCase):
    def _make_args(self, **overrides):
        args = types.SimpleNamespace(
            blocks_to_swap=0,
            lycoris=False,
            lora_weight=[],
            lora_multiplier=[],
            include_patterns=None,
            exclude_patterns=None,
            fp8_scaled=False,
            save_merged_model=None,
            disable_numpy_memmap=False,
            dit="/tmp/dit.safetensors",
            attn_mode="torch",
            use_32bit_attention=False,
            compile=False,
            use_pinned_memory_for_block_swap=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_parse_args_normalizes_lora_multipliers(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "zimage_generate_image.py",
                "--text_encoder",
                "/tmp/text_encoder",
                "--save_path",
                "/tmp/output",
                "--prompt",
                "hello",
                "--lora_weight",
                "a.safetensors",
                "b.safetensors",
                "--lora_multiplier",
                "0.5",
            ],
        ):
            args = zimage_generate_image.parse_args()

        self.assertEqual(args.lora_weight, ["a.safetensors", "b.safetensors"])
        self.assertEqual(args.lora_multiplier, [0.5, 1.0])

    def test_parse_args_accepts_sample_prompts_alias(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "zimage_generate_image.py",
                "--text_encoder",
                "/tmp/text_encoder",
                "--save_path",
                "/tmp/output",
                "--sample_prompts",
                "/tmp/prompts.txt",
            ],
        ):
            args = zimage_generate_image.parse_args()

        self.assertEqual(args.from_file, "/tmp/prompts.txt")

    def test_parse_prompt_line_supports_lora_overrides(self):
        prompt_data = zimage_generate_image.parse_prompt_line(
            "hello world --lora_weight a.safetensors b.safetensors --lora_multiplier 0.25 0.5"
        )

        self.assertEqual(prompt_data["prompt"], "hello world")
        self.assertEqual(prompt_data["lora_weight"], ["a.safetensors", "b.safetensors"])
        self.assertEqual(prompt_data["lora_multiplier"], [0.25, 0.5])

    def test_parse_prompt_line_supports_override_only_lines(self):
        prompt_data = zimage_generate_image.parse_prompt_line("--d 123 --lora_weight a.safetensors")

        self.assertNotIn("prompt", prompt_data)
        self.assertEqual(prompt_data["seed"], 123)
        self.assertEqual(prompt_data["lora_weight"], ["a.safetensors"])

    def test_apply_overrides_normalizes_lora_override_args(self):
        args = self._make_args(
            prompt="base prompt",
            image_size=[256, 256],
            lora_weight=["base.safetensors"],
            lora_multiplier=[1.0],
        )

        overridden = zimage_generate_image.apply_overrides(
            args,
            {
                "prompt": "override",
                "lora_weight": ["a.safetensors", "b.safetensors"],
                "lora_multiplier": [0.25],
            },
        )

        self.assertEqual(overridden.prompt, "override")
        self.assertEqual(overridden.lora_weight, ["a.safetensors", "b.safetensors"])
        self.assertEqual(overridden.lora_multiplier, [0.25, 1.0])

    def test_prompt_embedding_cache_round_trip(self):
        args = self._make_args(text_encoder="/tmp/text_encoder", fp8_llm=False)
        cache_dir = tempfile.mkdtemp()
        embed = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.bool)
        cache_key = zimage_generate_image.build_prompt_embedding_cache_key("hello", args)

        zimage_generate_image.save_prompt_embedding_cache(cache_dir, cache_key, "hello", embed, mask, args)
        loaded_embed, loaded_mask = zimage_generate_image.load_prompt_embedding_cache(cache_dir, cache_key)

        self.assertIsNotNone(loaded_embed)
        self.assertIsNotNone(loaded_mask)
        self.assertTrue(torch.equal(loaded_embed, embed))
        self.assertTrue(torch.equal(loaded_mask, mask))

    def test_main_dispatches_immediate_save_for_prompt_files(self):
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "zimage_generate_image.py",
                    "--text_encoder",
                    "/tmp/text_encoder",
                    "--save_path",
                    "/tmp/output",
                    "--sample_prompts",
                    "/tmp/prompts.txt",
                    "--save_strategy",
                    "immediate",
                ],
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("builtins.open", mock.mock_open(read_data="hello")),
            mock.patch.object(zimage_generate_image, "preprocess_prompts_for_batch", return_value=[{"prompt": "hello"}]),
            mock.patch.object(zimage_generate_image, "process_prompts_with_immediate_save") as process_immediate,
            mock.patch.object(zimage_generate_image, "process_batch_prompts") as process_batch,
        ):
            zimage_generate_image.main()

        process_immediate.assert_called_once()
        process_batch.assert_not_called()

    def test_main_dispatches_deferred_save_for_prompt_files(self):
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "zimage_generate_image.py",
                    "--text_encoder",
                    "/tmp/text_encoder",
                    "--save_path",
                    "/tmp/output",
                    "--sample_prompts",
                    "/tmp/prompts.txt",
                    "--save_strategy",
                    "deferred",
                ],
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("builtins.open", mock.mock_open(read_data="hello")),
            mock.patch.object(zimage_generate_image, "preprocess_prompts_for_batch", return_value=[{"prompt": "hello"}]),
            mock.patch.object(zimage_generate_image, "process_prompts_with_immediate_save") as process_immediate,
            mock.patch.object(zimage_generate_image, "process_batch_prompts") as process_batch,
        ):
            zimage_generate_image.main()

        process_batch.assert_called_once()
        process_immediate.assert_not_called()

    def test_main_dispatches_chunked_save_for_prompt_files(self):
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "zimage_generate_image.py",
                    "--text_encoder",
                    "/tmp/text_encoder",
                    "--save_path",
                    "/tmp/output",
                    "--sample_prompts",
                    "/tmp/prompts.txt",
                    "--save_every_n_images",
                    "5",
                ],
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("builtins.open", mock.mock_open(read_data="hello")),
            mock.patch.object(zimage_generate_image, "preprocess_prompts_for_batch", return_value=[{"prompt": "hello"}]),
            mock.patch.object(zimage_generate_image, "process_prompts_with_immediate_save") as process_immediate,
            mock.patch.object(zimage_generate_image, "process_batch_prompts") as process_batch,
            mock.patch.object(zimage_generate_image, "process_prompts_with_chunked_save") as process_chunked,
        ):
            zimage_generate_image.main()

        process_chunked.assert_called_once()
        process_batch.assert_not_called()
        process_immediate.assert_not_called()

    def test_process_prompts_with_chunked_save_loads_dit_once_per_chunk(self):
        args = self._make_args(
            text_encoder="/tmp/text_encoder",
            vae="/tmp/vae",
            prompt="base prompt",
            image_size=[256, 256],
            save_every_n_images=2,
            output_type="images",
        )
        prompts_data = [{"prompt": "first"}, {"prompt": "second"}, {"prompt": "third"}]

        with (
            mock.patch.object(
                zimage_generate_image,
                "get_generation_settings",
                return_value=zimage_generate_image.GenerationSettings(torch.device("cpu"), torch.float32),
            ),
            mock.patch.object(zimage_generate_image, "load_shared_models", return_value={}),
            mock.patch.object(
                zimage_generate_image,
                "apply_overrides",
                side_effect=[
                    self._make_args(prompt="first", image_size=[256, 256], save_every_n_images=2, output_type="images"),
                    self._make_args(prompt="second", image_size=[256, 256], save_every_n_images=2, output_type="images"),
                    self._make_args(prompt="third", image_size=[256, 256], save_every_n_images=2, output_type="images"),
                ],
            ),
            mock.patch.object(
                zimage_generate_image,
                "prepare_text_inputs",
                side_effect=[
                    ({"embed": torch.zeros(1), "mask": torch.ones(1, dtype=torch.bool), "prompt": "first"}, {"embed": None, "mask": None, "prompt": None}),
                    ({"embed": torch.zeros(1), "mask": torch.ones(1, dtype=torch.bool), "prompt": "second"}, {"embed": None, "mask": None, "prompt": None}),
                    ({"embed": torch.zeros(1), "mask": torch.ones(1, dtype=torch.bool), "prompt": "third"}, {"embed": None, "mask": None, "prompt": None}),
                ],
            ),
            mock.patch.object(zimage_generate_image.zimage_autoencoder, "load_autoencoder_kl", return_value=_DummyZImageModel()),
            mock.patch.object(zimage_generate_image, "load_dit_model", return_value=_DummyZImageModel()) as load_dit_model,
            mock.patch.object(zimage_generate_image, "generate", return_value=torch.zeros(1, 1, 1, 1, 1)),
            mock.patch.object(zimage_generate_image, "save_output"),
            mock.patch.object(zimage_generate_image, "clean_memory_on_device"),
            mock.patch.object(zimage_generate_image, "synchronize_device"),
            mock.patch.object(zimage_generate_image, "check_inputs", return_value=(256, 256)),
        ):
            zimage_generate_image.process_prompts_with_chunked_save(prompts_data, args)

        self.assertEqual(load_dit_model.call_count, 2)

    def test_load_dit_model_normalizes_zimage_inference_lora_weights(self):
        args = self._make_args(lora_weight=["dummy.safetensors"], lora_multiplier=[1.0])
        raw_weights = {
            "lora_unet_transformer_blocks_0_attention_qkv.lora_down.weight": torch.ones(2, 4),
            "lora_unet_transformer_blocks_0_attention_qkv.lora_up.weight": torch.ones(12, 2),
        }
        captured = {}

        def fake_filter(weights_sd, *_args, **_kwargs):
            captured["keys"] = set(weights_sd.keys())
            return weights_sd

        with (
            mock.patch.object(zimage_generate_image, "load_file", return_value=raw_weights),
            mock.patch.object(zimage_generate_image, "filter_lora_state_dict", side_effect=fake_filter),
            mock.patch.object(zimage_generate_image.zimage_model, "load_zimage_model", return_value=_DummyZImageModel()),
            mock.patch.object(zimage_generate_image, "clean_memory_on_device"),
        ):
            zimage_generate_image.load_dit_model(args, torch.device("cpu"), dit_weight_dtype=None)

        self.assertIn("lora_unet_transformer_blocks_0_attention_to_q.lora_down.weight", captured["keys"])
        self.assertIn("lora_unet_transformer_blocks_0_attention_to_k.lora_up.weight", captured["keys"])
        self.assertIn("lora_unet_transformer_blocks_0_attention_to_v.alpha", captured["keys"])

    def test_load_dit_model_passes_converter_for_lycoris(self):
        args = self._make_args(lycoris=True, lora_weight=["dummy.safetensors"], lora_multiplier=[1.0])

        with (
            mock.patch.object(zimage_generate_image.zimage_model, "load_zimage_model", return_value=_DummyZImageModel()),
            mock.patch.object(zimage_generate_image, "merge_lora_weights") as merge_lora_weights,
            mock.patch.object(zimage_generate_image, "clean_memory_on_device"),
        ):
            zimage_generate_image.load_dit_model(args, torch.device("cpu"), dit_weight_dtype=None)

        self.assertEqual(merge_lora_weights.call_count, 1)
        self.assertIs(
            merge_lora_weights.call_args.kwargs["converter"],
            zimage_generate_image.normalize_zimage_inference_lora_weights,
        )

    def test_fp8_linear_forward_patch_accepts_bfloat16_input_with_float32_scale_weight(self):
        module = torch.nn.Linear(4, 3, bias=False)
        module.weight = torch.nn.Parameter(module.weight.detach().to(torch.float8_e4m3fn), requires_grad=False)
        module.register_buffer("scale_weight", torch.ones((3, 1), dtype=torch.float32))

        x = torch.randn(2, 4, dtype=torch.bfloat16)

        output = fp8_linear_forward_patch(module, x, use_scaled_mm=False)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
