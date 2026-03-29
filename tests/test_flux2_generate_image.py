from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from musubi_tuner import flux_2_generate_image as flux2_generate_image


class _FakeMovable:
    def to(self, _device):
        return self


class Flux2GenerateImageTest(unittest.TestCase):
    def _make_args(self, **overrides):
        model_version = next(iter(flux2_generate_image.flux2_utils.FLUX2_MODEL_INFO.keys()))
        args = types.SimpleNamespace(
            model_version=model_version,
            text_encoder="/tmp/text_encoder",
            vae="/tmp/vae",
            fp8_text_encoder=False,
            lora_weight=[],
            lora_multiplier=[],
            include_patterns=None,
            exclude_patterns=None,
            save_merged_model=None,
            lycoris=False,
            output_type="images",
            blocks_to_swap=0,
            prompt="base prompt",
            negative_prompt=None,
            image_size=[1024, 1024],
            infer_steps=20,
            save_path=tempfile.mkdtemp(),
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_parse_args_accepts_sample_prompts_alias_and_normalizes_lora_multipliers(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "flux_2_generate_image.py",
                "--text_encoder",
                "/tmp/text_encoder",
                "--save_path",
                "/tmp/output",
                "--sample_prompts",
                "/tmp/prompts.txt",
                "--lora_weight",
                "a.safetensors",
                "b.safetensors",
                "--lora_multiplier",
                "0.75",
            ],
        ):
            args = flux2_generate_image.parse_args()

        self.assertEqual(args.from_file, "/tmp/prompts.txt")
        self.assertEqual(args.lora_weight, ["a.safetensors", "b.safetensors"])
        self.assertEqual(args.lora_multiplier, [0.75, 1.0])

    def test_parse_prompt_line_supports_lora_overrides(self):
        prompt_data = flux2_generate_image.parse_prompt_line(
            "hello world --lora_weight a.safetensors b.safetensors --lora_multiplier 0.25 0.5"
        )

        self.assertEqual(prompt_data["prompt"], "hello world")
        self.assertEqual(prompt_data["lora_weight"], ["a.safetensors", "b.safetensors"])
        self.assertEqual(prompt_data["lora_multiplier"], [0.25, 0.5])

    def test_apply_overrides_normalizes_lora_override_args(self):
        args = self._make_args(lora_weight=["base.safetensors"], lora_multiplier=[1.0])

        overridden = flux2_generate_image.apply_overrides(
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
        args = self._make_args()
        cache_dir = tempfile.mkdtemp()
        ctx_vec = torch.randn(1, 4, 8)
        cache_key = flux2_generate_image.build_prompt_embedding_cache_key("hello", args)

        flux2_generate_image.save_prompt_embedding_cache(cache_dir, cache_key, "hello", ctx_vec, args)
        loaded = flux2_generate_image.load_prompt_embedding_cache(cache_dir, cache_key)

        self.assertIsNotNone(loaded)
        self.assertTrue(torch.equal(loaded, ctx_vec))

    def test_process_batch_prompts_loads_dit_once(self):
        args = self._make_args()
        prompts_data = [{"prompt": "first"}, {"prompt": "second"}]

        with (
            mock.patch.object(
                flux2_generate_image,
                "get_generation_settings",
                return_value=flux2_generate_image.GenerationSettings(torch.device("cpu"), torch.float32),
            ),
            mock.patch.object(flux2_generate_image.flux2_utils, "load_ae", return_value=_FakeMovable()),
            mock.patch.object(flux2_generate_image.flux2_utils, "load_text_embedder", return_value=_FakeMovable()),
            mock.patch.object(flux2_generate_image, "prepare_image_inputs", return_value=(1024, 1024, None)),
            mock.patch.object(
                flux2_generate_image,
                "prepare_text_inputs",
                side_effect=[
                    ({"ctx_vec": torch.zeros(1), "prompt": "first"}, None),
                    ({"ctx_vec": torch.zeros(1), "prompt": "second"}, None),
                ],
            ),
            mock.patch.object(flux2_generate_image, "load_dit_model", return_value=object()) as load_dit_model,
            mock.patch.object(flux2_generate_image, "generate", return_value=(None, torch.zeros(1, 1, 1, 1, 1))),
            mock.patch.object(flux2_generate_image, "save_output"),
            mock.patch.object(flux2_generate_image, "clean_memory_on_device"),
            mock.patch.object(flux2_generate_image, "synchronize_device"),
        ):
            flux2_generate_image.process_batch_prompts(prompts_data, args)

        self.assertEqual(load_dit_model.call_count, 1)

    def test_process_batch_prompts_rejects_mixed_lora_specs(self):
        args = self._make_args()
        prompts_data = [
            {"prompt": "first", "lora_weight": ["a.safetensors"], "lora_multiplier": [1.0]},
            {"prompt": "second", "lora_weight": ["b.safetensors"], "lora_multiplier": [1.0]},
        ]

        with (
            mock.patch.object(
                flux2_generate_image,
                "get_generation_settings",
                return_value=flux2_generate_image.GenerationSettings(torch.device("cpu"), torch.float32),
            ),
            mock.patch.object(flux2_generate_image.flux2_utils, "load_ae", return_value=_FakeMovable()),
            mock.patch.object(flux2_generate_image.flux2_utils, "load_text_embedder", return_value=_FakeMovable()),
            mock.patch.object(flux2_generate_image, "prepare_image_inputs", return_value=(1024, 1024, None)),
            mock.patch.object(
                flux2_generate_image,
                "prepare_text_inputs",
                side_effect=[
                    ({"ctx_vec": torch.zeros(1), "prompt": "first"}, None),
                    ({"ctx_vec": torch.zeros(1), "prompt": "second"}, None),
                ],
            ),
            mock.patch.object(flux2_generate_image, "load_dit_model") as load_dit_model,
            mock.patch.object(flux2_generate_image, "clean_memory_on_device"),
        ):
            with self.assertRaisesRegex(ValueError, "same LoRA configuration"):
                flux2_generate_image.process_batch_prompts(prompts_data, args)

        load_dit_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
