from __future__ import annotations

import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from musubi_tuner import flux_2_generate_image as flux2_generate_image


class _FakeMovable:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, _device):
        self.device = torch.device(_device)
        return self


class _FakeTextEmbedder(_FakeMovable):
    def __init__(self):
        self.calls = []
        self.device = torch.device("cpu")

    def __call__(self, prompts):
        self.calls.append(list(prompts))
        return torch.arange(len(prompts), dtype=torch.float32).view(len(prompts), 1, 1)


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
            save_embeddings=None,
            save_strategy="immediate",
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
            mock.patch.object(flux2_generate_image.flux2_utils, "load_text_embedder", return_value=_FakeTextEmbedder()),
            mock.patch.object(flux2_generate_image, "prepare_image_inputs", return_value=(1024, 1024, None)),
            mock.patch.object(
                flux2_generate_image,
                "precompute_text_inputs_for_prompts",
                side_effect=[
                    [
                        ({"ctx_vec": torch.zeros(1), "prompt": "first"}, None),
                        ({"ctx_vec": torch.zeros(1), "prompt": "second"}, None),
                    ]
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

    def test_main_dispatches_chunked_save_for_prompt_files(self):
        with (
            mock.patch.object(
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
                    "--save_every_n_images",
                    "4",
                ],
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("builtins.open", mock.mock_open(read_data="hello")),
            mock.patch.object(flux2_generate_image, "preprocess_prompts_for_batch", return_value=[{"prompt": "hello"}]),
            mock.patch.object(flux2_generate_image, "process_prompts_with_immediate_save") as process_immediate,
            mock.patch.object(flux2_generate_image, "process_batch_prompts") as process_batch,
            mock.patch.object(flux2_generate_image, "process_prompts_with_chunked_save") as process_chunked,
        ):
            flux2_generate_image.main()

        process_chunked.assert_called_once()
        process_batch.assert_not_called()
        process_immediate.assert_not_called()

    def test_process_prompts_with_chunked_save_loads_dit_once_per_chunk(self):
        args = self._make_args(save_every_n_images=2, output_type="images")
        prompts_data = [{"prompt": "first"}, {"prompt": "second"}, {"prompt": "third"}]

        with (
            mock.patch.object(
                flux2_generate_image,
                "get_generation_settings",
                return_value=flux2_generate_image.GenerationSettings(torch.device("cpu"), torch.float32),
            ),
            mock.patch.object(flux2_generate_image, "load_shared_models", return_value={}),
            mock.patch.object(
                flux2_generate_image,
                "apply_overrides",
                side_effect=[
                    self._make_args(prompt="first", save_every_n_images=2, output_type="images"),
                    self._make_args(prompt="second", save_every_n_images=2, output_type="images"),
                    self._make_args(prompt="third", save_every_n_images=2, output_type="images"),
                ],
            ),
            mock.patch.object(
                flux2_generate_image,
                "precompute_text_inputs_for_prompts",
                return_value=[
                    ({"ctx_vec": torch.zeros(1), "prompt": "first"}, None),
                    ({"ctx_vec": torch.zeros(1), "prompt": "second"}, None),
                    ({"ctx_vec": torch.zeros(1), "prompt": "third"}, None),
                ],
            ),
            mock.patch.object(flux2_generate_image.flux2_utils, "load_ae", return_value=_FakeMovable()),
            mock.patch.object(flux2_generate_image, "prepare_image_inputs", return_value=(1024, 1024, None)),
            mock.patch.object(flux2_generate_image, "load_dit_model", return_value=object()) as load_dit_model,
            mock.patch.object(flux2_generate_image, "generate", return_value=(None, torch.zeros(1, 1, 1, 1, 1))),
            mock.patch.object(flux2_generate_image, "save_output"),
            mock.patch.object(flux2_generate_image, "clean_memory_on_device"),
            mock.patch.object(flux2_generate_image, "synchronize_device"),
        ):
            flux2_generate_image.process_prompts_with_chunked_save(prompts_data, args)

        self.assertEqual(load_dit_model.call_count, 2)

    def test_precompute_text_inputs_batches_unique_prompts(self):
        args = self._make_args(negative_prompt=None, save_embeddings=None)
        shared_models = {"text_embedder": _FakeTextEmbedder(), "conds_cache": {}}
        all_prompt_args_list = [
            self._make_args(prompt="first", negative_prompt=None, save_embeddings=None),
            self._make_args(prompt="second", negative_prompt=None, save_embeddings=None),
            self._make_args(prompt="first", negative_prompt=None, save_embeddings=None),
        ]

        with mock.patch.dict(
            flux2_generate_image.flux2_utils.FLUX2_MODEL_INFO,
            {
                args.model_version: types.SimpleNamespace(guidance_distilled=False),
            },
            clear=False,
        ):
            all_precomputed_text_data = flux2_generate_image.precompute_text_inputs_for_prompts(
                all_prompt_args_list,
                args,
                torch.device("cpu"),
                shared_models,
            )

        self.assertEqual(shared_models["text_embedder"].calls, [["first", " ", "second"]])
        self.assertEqual(len(all_precomputed_text_data), 3)
        self.assertTrue(torch.equal(all_precomputed_text_data[0][0]["ctx_vec"], torch.tensor([[[0.0]]])))
        self.assertTrue(torch.equal(all_precomputed_text_data[0][1]["ctx_vec"], torch.tensor([[[1.0]]])))
        self.assertTrue(torch.equal(all_precomputed_text_data[1][0]["ctx_vec"], torch.tensor([[[2.0]]])))


if __name__ == "__main__":
    unittest.main()
