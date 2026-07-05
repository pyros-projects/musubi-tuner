from __future__ import annotations

import contextlib
import sys
import types
import unittest
from unittest import mock

import torch

from musubi_tuner import boogu_image_generate_image


class _FakeAccelerator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


class BooguImageGenerateImageTest(unittest.TestCase):
    def test_parse_args_accepts_sample_prompts_alias_and_fp8_scaled(self):
        args = boogu_image_generate_image.parse_args(
            [
                "--dit",
                "/tmp/boogu_edit.safetensors",
                "--vae",
                "/tmp/ae.safetensors",
                "--text_encoder",
                "/tmp/qwen.safetensors",
                "--from_file",
                ".pyro/boogu/cfg/p_test.toml",
                "--fp8_base",
                "--fp8_scaled",
                "--sampling_lora_weight",
                "/tmp/turbo.safetensors",
                "--sampling_lora_multiplier",
                "0.5",
            ]
        )

        self.assertEqual(args.sample_prompts, ".pyro/boogu/cfg/p_test.toml")
        self.assertEqual(args.dit, "/tmp/boogu_edit.safetensors")
        self.assertTrue(args.fp8_base)
        self.assertTrue(args.fp8_scaled)
        self.assertEqual(args.sampling_lora_weight, ["/tmp/turbo.safetensors"])
        self.assertEqual(args.sampling_lora_multiplier, [0.5])

    def test_main_uses_trainer_sample_images_for_prompt_file(self):
        class _FakeTrainer:
            def __init__(self):
                self.blocks_to_swap = None
                self.handled_args = None
                self.processed_prompt_path = None
                self.loaded_transformer_args = None
                self.loaded_vae_args = None
                self.sample_call = None
                self.dit_dtype = torch.bfloat16

            def handle_model_specific_args(self, args):
                self.handled_args = args

            def process_sample_prompts(self, args, accelerator, sample_prompts):
                self.processed_prompt_path = sample_prompts
                return [{"prompt": "make Yuna's dress blue", "enum": 0}]

            def load_transformer(self, **kwargs):
                self.loaded_transformer_args = kwargs
                return object()

            def load_vae(self, args, vae_dtype, vae_path):
                self.loaded_vae_args = (args, vae_dtype, vae_path)
                return object()

            def sample_images(self, **kwargs):
                self.sample_call = kwargs

        trainer = _FakeTrainer()

        with (
            mock.patch.object(boogu_image_generate_image, "Accelerator", _FakeAccelerator),
            mock.patch.object(boogu_image_generate_image, "BooguImageNetworkTrainer", return_value=trainer),
            mock.patch.object(
                sys,
                "argv",
                [
                    "boogu_image_generate_image.py",
                    "--dit",
                    "/tmp/boogu_edit.safetensors",
                    "--vae",
                    "/tmp/ae.safetensors",
                    "--text_encoder",
                    "/tmp/qwen.safetensors",
                    "--processor",
                    "/tmp/qwen_processor",
                    "--sample_prompts",
                    ".pyro/boogu/cfg/p_test.toml",
                    "--output_dir",
                    "/tmp/out",
                    "--output_name",
                    "yuna_test",
                ],
            ),
        ):
            boogu_image_generate_image.main()

        self.assertEqual(trainer.blocks_to_swap, 0)
        self.assertEqual(trainer.processed_prompt_path, ".pyro/boogu/cfg/p_test.toml")
        self.assertEqual(trainer.loaded_transformer_args["dit_path"], "/tmp/boogu_edit.safetensors")
        self.assertEqual(trainer.loaded_vae_args[2], "/tmp/ae.safetensors")
        self.assertEqual(trainer.sample_call["sample_parameters"], [{"prompt": "make Yuna's dress blue", "enum": 0}])
        self.assertEqual(trainer.sample_call["args"].output_dir, "/tmp/out")
        self.assertEqual(trainer.sample_call["args"].output_name, "yuna_test")
        self.assertTrue(trainer.sample_call["force_sample"])


if __name__ == "__main__":
    unittest.main()
